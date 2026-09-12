import pytest
import torch
from   transformers.models.encodec.modeling_encodec import EncodecModel
from models import EncodecEncoder, EncodecDecoder, RVQ, remove_all_parametrizations, load_pretrained


@pytest.fixture(scope="session", params=[24, 48], ids=["24khz", "48khz"])
def encodec_models(request):
    is24 = request.param == 24
    enc  = EncodecEncoder(is24).eval()
    dec  = EncodecDecoder(is24).eval()
    rvq  = RVQ(128, 1024, 32 if is24 else 16).eval()
    net  = remove_all_parametrizations(EncodecModel.from_pretrained(f"facebook/encodec_{24 if is24 else 48}khz").eval())
    load_pretrained(enc, rvq, dec, net)
    return enc, dec, rvq, net


@torch.inference_mode()
def test_components(encodec_models):
    enc, dec, rvq, net = encodec_models
    inc = net.config.audio_channels

    # Test encoder only
    x    = torch.randn(1, inc, 24000)
    out0 = enc(x)
    out1 = net.encoder(x).permute(0,2,1)
    torch.testing.assert_close(out0, out1)
    
    # Test decoder only
    x    = torch.randn(1,75,128)
    out0 = dec(x)
    out1 = net.decoder(x.permute(0,2,1))
    torch.testing.assert_close(out0, out1)

    # RVQ encode
    x = torch.randn(1, 75, 128)
    out0 = rvq.encode(x)
    out1 = net.quantizer.encode(x.permute(0, 2, 1)).permute(1, 2, 0)
    torch.testing.assert_close(out0, out1)

    # RVQ decode
    codes = out0
    out0  = rvq.decode(codes)
    out1  = net.quantizer.decode(codes.permute(2, 0, 1)).permute(0, 2, 1)
    torch.testing.assert_close(out0, out1)