from enum import Enum

import torch
from torch import nn


class NormType(Enum):
    global_layer_norm = "global_layer_norm"
    channel_layer_norm = "channel_layer_norm"


def global_layer_norm(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Global Layer Normalization (gLN)

    Attributes:
        input (torch.Tensor): the input tensor of shape [batch, channel_size, length]
        gamma (torch.Tensor): the scale parameter of shape [1, channel_size, 1]
        beta (torch.Tensor): the shift parameter of shape [1, channel_size, 1]
        eps (float): a small value to avoid division by zero
    """
    mean = input.mean(dim=[1, 2], keepdim=True)
    # mean is of shape [batch, 1, 1]
    var = input.var(dim=[1, 2], keepdim=True, unbiased=False)
    # var is of shape [batch, 1, 1]

    value = (input - mean) / torch.sqrt(var + eps)
    return value * gamma + beta


def global_layer_norm_using_torch_layer_norm(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    channel_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Global Layer Normalization (gLN)

    This method is equivalent to the forward method but use `nn.LayerNorm` instead of the custom implementation.
    This requires to reshape and `repeat` the weights to match the expected shape of `nn.LayerNorm`, introducing a computational overhead.

    The method can be useful to replace the custom implementation by an optimized LayerNorm during ONNX export or inference.

    Note: this method requires to compute the length of the input tensor to repeat the weights, which may not be compatible
    with dynamic input shapes.

    Attributes:
        input (torch.Tensor): the input tensor of shape [batch, channel_size, length]
        gamma (torch.Tensor): the scale parameter of shape [1, channel_size, 1]
        beta (torch.Tensor): the shift parameter of shape [1, channel_size, 1]
        eps (float): a small value to avoid division by zero
    """
    length = input.size(-1)
    weight = gamma.squeeze(0).repeat(1, length)
    bias = beta.squeeze(0).repeat(1, length)
    return nn.functional.layer_norm(input, [channel_size, length], weight, bias, eps)


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization (gLN)

    GlobalLayerNorm compute

    $ "gLN"(F) = (F - E[F]) / sqrt("Var"[F] + epsilon) gamma + beta $

    where $gamma, beta in RR^(N times 1)$. It intends to normalize across both channel and time dimensions sharing the same weights for the time dimension.

    It differ slightly from `nn.LayerNorm` which weights have the same shape as `normalized_shape` and compute the esperance and variance over the last $D$ dimensions, where D is the dimension of `normalized_shape`.

    Attributes:
        channel_size (int): the number of channels in the input tensor

    The input tensor $x$ should have shape [batch, channel_size, length] where length correspond to the time dimension.
    The output tensor will have the same shape as the input tensor.
    """

    def __init__(self, channel_size: int, eps: float = 1e-8):
        super(GlobalLayerNorm, self).__init__()
        self.channel_size = channel_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return global_layer_norm(x, self.gamma, self.beta, self.eps)

    def forward_torch_layer_norm(self, x: torch.Tensor):
        """
        This method is equivalent to the forward method but use `nn.LayerNorm` instead of the custom implementation.
        This requires to reshape and `repeat` the weights to match the expected shape of `nn.LayerNorm`, introducing a computational overhead.

        The method can be useful to replace the custom implementation by an optimized LayerNorm during ONNX export or inference.
        """
        return global_layer_norm_using_torch_layer_norm(
            x, self.gamma, self.beta, self.channel_size, self.eps
        )
