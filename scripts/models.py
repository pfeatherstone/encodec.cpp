from   copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn.utils.parametrize import remove_parametrizations
from   einops import rearrange
from   einops.layers.torch import Rearrange
from   transformers.models.encodec.modeling_encodec import EncodecModel


##########################################################################################################
##########################################################################################################
### Encodec
##########################################################################################################
##########################################################################################################


def apply(x: torch.Tensor, layers):
    y = x
    for l in layers: 
        y = l(y)
    return y


def count_parameters(net: nn.Module, trainableOnly: bool = False):
    return sum(p.numel() for p in net.parameters() if p.requires_grad or not trainableOnly)


def normalize(x:torch.Tensor, eps=1e-8):
    mono  = x.mean(1, keepdim=True)
    scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps
    x     = x / scale
    scale = scale.view(-1, 1)
    return x, scale


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, is24=True):
        super().__init__()
        pad_total   = k - s
        pad_right   = pad_total//2
        pad_left    = pad_total - pad_right
        self.pads   = [pad_total, 0] if is24 else [pad_left, pad_right]
        self.conv   = nn.Conv1d(c1, c2, k, s)
        self.norm   = nn.GroupNorm(1, c2) if not is24 else None
    def forward(self, x):
        x = F.pad(x, self.pads, mode='reflect')
        x = self.conv(x)
        if self.norm is not None: x = self.norm(x)
        return x
    

class ConvTranspose(nn.Module):
    def __init__(self, c1, c2, k, s, is24=True):
        super().__init__()
        pad_total   = k - s
        pad_right   = pad_total//2
        pad_left    = pad_total - pad_right
        self.pads   = [0, pad_total] if is24 else [pad_left, pad_right]
        self.conv   = nn.ConvTranspose1d(c1, c2, k, s, dilation=1)
        self.norm   = nn.GroupNorm(1, c2) if not is24 else None
    def forward(self, x):
        x = self.conv(x)
        if self.norm  is not None: x = self.norm(x)
        x = x[...,self.pads[0]:-self.pads[1]]
        return x


class EncodecLSTM(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_layers)
    def forward(self, x: torch.Tensor):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0] + x
        x = x.permute(1, 2, 0)
        return x


class EncodecResnetBlock(nn.Module):
    def __init__(self, c, is24=True):
        super().__init__()
        self.b0 = Conv(c, c//2, k=3, is24=is24)
        self.b1 = Conv(c//2, c, k=1, is24=is24)
        self.b2 = Conv(c, c,    k=1, is24=is24)
    def forward(self, x: torch.Tensor):
        y = apply(x, [F.elu, self.b0, F.elu, self.b1])
        y = self.b2(x) + y
        return y
    
    
def EncodecEncoderBlock(c1, c2, s, is24=True):
    return nn.Sequential(EncodecResnetBlock(c1, is24=is24), nn.ELU(), Conv(c1, c2, k=s*2, s=s, is24=is24))


def EncodecDecoderBlock(c1, c2, s, is24=True):
    return nn.Sequential(nn.ELU(), ConvTranspose(c1, c2, k=s*2, s=s, is24=is24), EncodecResnetBlock(c2, is24=is24) )
    

def EncodecEncoder(is24=True):
    inc = 1 if is24 else 2
    return nn.Sequential(Conv(inc, 32, 7, is24=is24),
                         EncodecEncoderBlock( 32,  64, s=2, is24=is24),
                         EncodecEncoderBlock( 64, 128, s=4, is24=is24),
                         EncodecEncoderBlock(128, 256, s=5, is24=is24),
                         EncodecEncoderBlock(256, 512, s=8, is24=is24),
                         EncodecLSTM(512, 2),
                         nn.ELU(), Conv(512, 128, 7, is24=is24),
                         Rearrange('b f n -> b n f'))


def EncodecDecoder(is24=True):
    outc = 1 if is24 else 2
    return nn.Sequential(Rearrange('b n f -> b f n'),
                         Conv(128, 512, 7, is24=is24),
                         EncodecLSTM(512, 2),
                         EncodecDecoderBlock(512, 256, s=8, is24=is24),
                         EncodecDecoderBlock(256, 128, s=5, is24=is24),
                         EncodecDecoderBlock(128,  64, s=4, is24=is24),
                         EncodecDecoderBlock( 64,  32, s=2, is24=is24),
                         nn.ELU(), Conv(32, outc, 7, is24=is24))


def cdist(x, y):
    x2 = x.pow(2).sum(2).unsqueeze(-1)
    y2 = y.pow(2).sum(1).unsqueeze(0)
    xy = x @ y.T                   
    return x2 + y2 - 2*xy


class RVQ(nn.Module):
    def __init__(self, dim, codebook_size, num):
        super().__init__()
        self.codebooks = nn.Parameter(torch.empty(num, codebook_size, dim), requires_grad=False)

    def encode(self, x: torch.Tensor):
        indices = []
        for codebook in self.codebooks:
            dist = cdist(x, codebook)
            idx  = dist.argmin(-1)
            q    = F.embedding(idx, codebook)
            x    = x - q
            indices.append(idx)
        return torch.stack(indices, -1)
    
    def decode(self, codes: torch.Tensor):
        codes = rearrange(codes, 'b n q -> q b n')
        q = torch.tensor(0.0, device=codes.device)
        for idx, codebook in zip(codes, self.codebooks):
            q = q + F.embedding(idx, codebook)
        return q


##########################################################################################################
##########################################################################################################
### Official
##########################################################################################################
##########################################################################################################


def remove_all_parametrizations(module: torch.nn.Module):
    def remove_all_parametrizations_(m: torch.nn.Module):
        for child in m.children():
            if hasattr(child, "parametrizations"):
                for param_name in list(child.parametrizations.keys()):
                    remove_parametrizations(child, param_name)
            remove_all_parametrizations_(child)
    module_new = deepcopy(module)
    remove_all_parametrizations_(module_new)
    return module_new


def encodec_layers(net):
    types = (nn.Conv1d, nn.ConvTranspose1d, nn.GroupNorm, nn.LSTM)
    return (m for m in net.modules() if isinstance(m, types))


@torch.no_grad()
def load_layers(dst, src):
    for d, s in zip(encodec_layers(dst), encodec_layers(src), strict=True):
        assert type(d) is type(s)
        d.load_state_dict(s.state_dict())


@torch.no_grad()
def load_pretrained(enc, rvq, dec, model):
    load_layers(enc, model.encoder)
    load_layers(dec, model.decoder)
    codebooks = torch.stack([l.codebook.embed for l in model.quantizer.layers], 0)
    rvq.codebooks.copy_(codebooks)


##########################################################################################################
##########################################################################################################
### Export to cpp
##########################################################################################################
##########################################################################################################


def get_parameters(net):
    params = []
    for l in encodec_layers(net):
        if isinstance(l, (nn.Conv1d,nn.ConvTranspose1d)):
            params.extend([l.weight.transpose(1,2).contiguous(), l.bias])
        elif isinstance(l, nn.GroupNorm):
            params.extend([l.weight, l.bias])
        elif isinstance(l, nn.LSTM):
            params.extend(l.state_dict().values())
    return params


def write_to_cpp_file(data, file:str, name:str, values_per_line:int = 8):
    weights = f"{name.upper()}_WEIGHTS"
    size    = f"{name.upper()}_SIZE"
    with open(file, 'wt') as f:
        f.write("#include <cstddef>\n")
        f.write("#include <span>\n\n")
        f.write(f"namespace encodec\n")
        f.write(f"{{\n")
        f.write(f"    alignas(32) const float {weights}[] = {{\n")
        for i in range(0, len(data), values_per_line):
            row = data[i:i+values_per_line]
            literals = ", ".join(f"{np.float32(v).item():.9g}f" for v in row)
            f.write(f"        {literals},\n")

        f.write(f"    }};\n\n")
        f.write(f"    const std::size_t {size} = sizeof({weights}) / sizeof(float);\n\n")
        f.write(f"    std::span<const float> get_{name}_weights() {{return std::span{{{weights}, {size}}};}}\n")
        f.write("}")


@torch.no_grad()
def save_cpp(net, file:str, name:str, values_per_line:int = 8):
    data = np.concatenate([p.numpy().ravel() for p in get_parameters(net)])
    write_to_cpp_file(data, file, name, values_per_line)

    
##########################################################################################################
##########################################################################################################
### Tests
##########################################################################################################
##########################################################################################################


@torch.inference_mode()
def test_against_official(enc, dec, rvq:RVQ, net1:EncodecModel):
    # Test encoder only
    inc  = net1.config.audio_channels
    x    = torch.randn(1, inc, 24000)
    out0 = enc(x)
    out1 = net1.encoder(x).permute(0,2,1)
    torch.testing.assert_close(out0, out1)
    
    # Test decoder only
    x    = torch.randn(1,75,128)
    out0 = dec(x)
    out1 = net1.decoder(x.permute(0,2,1))
    torch.testing.assert_close(out0, out1)

    # RVQ encode
    x    = torch.randn(1, inc, 24000)
    out0 = rvq.encode(enc(x))
    out1 = net1.quantizer.encode(net1.encoder(x)).permute(1, 2, 0)
    torch.testing.assert_close(out0, out1)

    # RVQ decode
    codes = out0
    out0  = rvq.decode(codes)
    out1  = net1.quantizer.decode(codes.permute(2, 0, 1)).permute(0, 2, 1)
    torch.testing.assert_close(out0, out1)


if __name__ == '__main__':
    print("Starting")
    is24 = False
    enc  = EncodecEncoder(is24=is24).eval()
    dec  = EncodecDecoder(is24=is24).eval()
    rvq  = RVQ(128, 1024, 32 if is24 else 16)
    net1 = remove_all_parametrizations(EncodecModel.from_pretrained(f"facebook/encodec_{24 if is24 else 48}khz").eval())
    print(f"enc0 size {count_parameters(enc)}")
    print(f"enc1 size {count_parameters(net1.encoder)}")
    print(f"dec0 size {count_parameters(dec)}")
    print(f"dec1 size {count_parameters(net1.decoder)}")
    load_pretrained(enc, rvq, dec, net1)
    test_against_official(enc, dec, rvq, net1)
    # save_cpp(enc, "encoder48.cpp", "encoder48")
    # save_cpp(dec, "decoder48.cpp", "decoder48")
    # write_to_cpp_file(rvq.codebooks.numpy().ravel(), "rvq48.cpp", "rvq48")   