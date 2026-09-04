from   copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn.utils.parametrize import remove_parametrizations
from   einops import rearrange
from   einops.layers.torch import Rearrange
from   transformers.models.encodec.modeling_encodec import EncodecModel, EncodecConv1d, EncodecConvTranspose1d

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


class CausalConv1d(nn.Module):
    def __init__(self, c1, c2, k, s=1, d=1):
        super().__init__()
        self.pad  = d * (k - 1) + 1 - s
        self.conv = nn.Conv1d(c1, c2, k, s)
    def forward(self, x):
        x = F.pad(x, (self.pad, 0), mode='reflect')
        x = self.conv(x)
        return x
    

class CausalConv1dTranspose(nn.Module):
    def __init__(self, c1, c2, k, s, d=1):
        super().__init__()
        self.pad  = d * (k - 1) + 1 - s
        self.conv = nn.ConvTranspose1d(c1, c2, k, s, dilation=1)
    def forward(self, x):
        x = self.conv(x)
        x = x[...,:-self.pad]
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
    def __init__(self, c):
        super().__init__()
        self.b0 = CausalConv1d(c, c//2, k=3)
        self.b1 = CausalConv1d(c//2, c, k=1)
        self.b2 = CausalConv1d(c, c,    k=1)
    def forward(self, x: torch.Tensor):
        y = apply(x, [F.elu, self.b0, F.elu, self.b1])
        y = self.b2(x) + y
        return y
    
    
def EncodecEncoderBlock(c1, c2, s):
    return nn.Sequential(EncodecResnetBlock(c1), nn.ELU(), CausalConv1d(c1, c2, k=s*2, s=s))


def EncodecDecoderBlock(c1, c2, s):
    return nn.Sequential(nn.ELU(), CausalConv1dTranspose(c1, c2, k=s*2, s=s), EncodecResnetBlock(c2) )
    

def EncodecEncoder():
    return nn.Sequential(CausalConv1d(1, 32, 7),
                         EncodecEncoderBlock( 32,  64, s=2),
                         EncodecEncoderBlock( 64, 128, s=4),
                         EncodecEncoderBlock(128, 256, s=5),
                         EncodecEncoderBlock(256, 512, s=8),
                         EncodecLSTM(512, 2),
                         nn.ELU(), CausalConv1d(512, 128, 7),
                         Rearrange('b f n -> b n f'))


def EncodecDecoder():
    return nn.Sequential(Rearrange('b n f -> b f n'),
                         CausalConv1d(128, 512, 7),
                         EncodecLSTM(512, 2),
                         EncodecDecoderBlock(512, 256, s=8),
                         EncodecDecoderBlock(256, 128, s=5),
                         EncodecDecoderBlock(128,  64, s=4),
                         EncodecDecoderBlock( 64,  32, s=2),
                         nn.ELU(), CausalConv1d(32, 1, 7))


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


def encodec_official():
    net = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
    net = remove_all_parametrizations(net)
    return net


def iterate_children(net, types):
    if isinstance(net, types): 
        yield net
    else:
        for m in net.children():
            yield from iterate_children(m, types)


def encodec_offical_layers(net):
    yield from iterate_children(net, (EncodecConv1d, EncodecConvTranspose1d, nn.LSTM))


def encodec_layers(net):
    yield from iterate_children(net, (CausalConv1d, CausalConv1dTranspose, nn.LSTM))


def load_layers(layers1, layers2):
    for l0, l1 in zip(layers1, layers2, strict=True):
        if   isinstance(l0, CausalConv1d) and isinstance(l1, EncodecConv1d):
            l0.conv.weight.data.copy_(l1.conv.weight.data)
            l0.conv.bias.data.copy_(l1.conv.bias.data)
        elif isinstance(l0, CausalConv1dTranspose) and isinstance(l1, EncodecConvTranspose1d):
            l0.conv.weight.data.copy_(l1.conv.weight.data)
            l0.conv.bias.data.copy_(l1.conv.bias.data)
        elif isinstance(l0, nn.LSTM) and isinstance(l1, nn.LSTM):
            l0.load_state_dict(l1.state_dict())
        else:
            assert False, "bug in loading layers"


@torch.no_grad()
def load_pretrained(enc, rvq, dec, model):
    load_layers(encodec_layers(enc), encodec_offical_layers(model.encoder))
    load_layers(encodec_layers(dec), encodec_offical_layers(model.decoder))
    codebooks = torch.stack([l.codebook.embed for l in model.quantizer.layers], 0)
    rvq.codebooks.data.copy_(codebooks.data)


##########################################################################################################
##########################################################################################################
### Export to cpp
##########################################################################################################
##########################################################################################################


def get_parameters(net):
    params = []
    for l in encodec_layers(net):
        if isinstance(l, (CausalConv1d,CausalConv1dTranspose)):
            params.append(l.conv.weight.transpose(1,2).contiguous())
            params.append(l.conv.bias)
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
    x    = torch.randn(1, 1, 24000)
    out0 = enc(x)
    out1 = net1.encoder(x).permute(0,2,1)
    torch.testing.assert_close(out0, out1)
    
    # Test decoder only
    x    = torch.randn(1,75,128)
    out0 = dec(x)
    out1 = net1.decoder(x.permute(0,2,1))
    torch.testing.assert_close(out0, out1)

    # Test encoder + RVQ
    x    = torch.randn(1, 1, 24000)
    out0 = rvq.encode(enc(x))
    out1 = net1.encode(x, bandwidth=24.0)
    torch.testing.assert_close(out0, out1.audio_codes[0].permute(0,2,1))

    # Test decoder + RVQ
    x    = out0
    out0 = dec(rvq.decode(x))
    out1 = net1.decode(out1.audio_codes, out1.audio_scales).audio_values
    torch.testing.assert_close(out0, out1, rtol=1e-2, atol=1e-4)


if __name__ == '__main__':
    print("Starting")
    rvq  = RVQ(128, 1024, 32)
    enc  = EncodecEncoder().eval()
    dec  = EncodecDecoder().eval()
    net1 = encodec_official().eval()
    load_pretrained(enc, rvq, dec, net1)
    print(f"enc0 size {count_parameters(enc)}")
    print(f"enc1 size {count_parameters(net1.encoder)}")
    print(f"dec0 size {count_parameters(dec)}")
    print(f"dec1 size {count_parameters(net1.decoder)}")
    test_against_official(enc, dec, rvq, net1)
    save_cpp(enc, "encoder24.cpp", "encoder24")
    save_cpp(dec, "decoder24.cpp", "decoder24")
    write_to_cpp_file(rvq.codebooks.numpy().ravel(), "rvq24.cpp", "rvq24")   