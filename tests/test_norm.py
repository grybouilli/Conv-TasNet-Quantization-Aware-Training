import torch

from model.norm import GlobalLayerNorm


def test_global_layer_norm_vs_torch_layer_norm():
    batch, channel, length = 1, 5, 10
    x = torch.randn(batch, channel, length)

    gln = GlobalLayerNorm(channel_size=channel)
    gln.gamma.data.copy_(torch.randn_like(gln.gamma))
    gln.beta.data.copy_(torch.randn_like(gln.beta))

    res_gln = gln(x)
    res_torch_ln = gln.forward_torch_layer_norm(x)
    torch.testing.assert_close(res_gln, res_torch_ln)
