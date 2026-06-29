####################################################################################################
# Inspired from pytorch's fused batch norm on github :
# https://github.com/pytorch/pytorch/blob/v2.9.1/torch/ao/nn/intrinsic/qat/modules/conv_fused.py
####################################################################################################

# mypy: allow-untyped-defs
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from hatorch.quantizers.base import BaseQuantizer
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.layers.quantized_relu import QuantReLU

__all__ = [
    "QuantConvBn2d",
    "QuantConvBnReLU2d",
    "QuantConvReLU2d",
]


class QuantConvBn2d(nn.Module):
    fold_batch_norm: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        reversed_padding_repeated_twice: list[int] = None,
        # BatchNorm args
        bn_eps: torch.Tensor = 1e-05,
        bn_momentum: torch.Tensor = 0.1,
        # Quantization args
        weight_quantizer: Optional[BaseQuantizer] = None,
        activation_quantizer: Optional[BaseQuantizer] = None,
        bias_quantizer: Optional[BaseQuantizer] = None,
        freeze_bn: bool = False,
        fold_batch_norm: bool = True,
    ):
        super().__init__()
        self.fold_batch_norm = fold_batch_norm

        # Conv parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.reversed_padding_repeated_twice = reversed_padding_repeated_twice

        # Initialize conv weight and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_conv_parameters()

        # BatchNorm
        self.bn = nn.BatchNorm2d(out_channels, bn_eps, bn_momentum, True, True)
        self.freeze_bn = freeze_bn
        self.reset_bn_parameters()

        # Quantizers
        from hatorch.quantizers.base import IdentityQuantizer
        self.weight_quantizer = weight_quantizer if weight_quantizer is not None else IdentityQuantizer()
        self.activation_quantizer = activation_quantizer if activation_quantizer is not None else IdentityQuantizer()
        self.bias_quantizer = bias_quantizer if bias_quantizer is not None else IdentityQuantizer()

    def reset_conv_parameters(self) -> None:
        """Initialize conv parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actually for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)  # Pytorch loves accessing private members
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super().reset_parameters()

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.eval()
        return self

    def _conv2d(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Perform 2D convolution with proper padding handling."""
        if self.padding_mode != 'zeros':
            x = F.pad(x, self.reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(x, weight, bias, self.stride, (0, 0), self.dilation, self.groups)
        else:
            return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_conv_bn(
        cls,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        weight_quantizer: Optional[BaseQuantizer] = None,
        activation_quantizer: Optional[BaseQuantizer] = None,
        bias_quantizer: Optional[BaseQuantizer] = None,
        freeze_bn: bool = False,
        fold_batch_norm: bool = True,
    ) -> 'QuantConvBn2d':
        """
        Create a QuantConvBn2d from existing Conv2d and BatchNorm2d layers.

        Args:
            conv: Original Conv2d layer
            bn: Original BatchNorm2d layer
            weight_quantizer: Quantizer for weights
            activation_quantizer: Quantizer for activations
            bias_quantizer: Quantizer for bias
            freeze_bn: Whether to freeze BN statistics

        Returns:
            QuantConvBn2d layer with copied parameters
        """
        quant_conv_bn = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            reversed_padding_repeated_twice=conv._reversed_padding_repeated_twice,  # No idea why this is private
            bn_eps=bn.eps,
            bn_momentum=bn.momentum,
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            bias_quantizer=bias_quantizer,
            freeze_bn=freeze_bn,
            fold_batch_norm=fold_batch_norm,
        )

        # Copy conv weights
        with torch.no_grad():
            quant_conv_bn.weight.copy_(conv.weight)
            if conv.bias is not None:
                quant_conv_bn.bias.copy_(conv.bias)

        # Copy BN parameters
        with torch.no_grad():
            quant_conv_bn.bn.weight.copy_(bn.weight)
            quant_conv_bn.bn.bias.copy_(bn.bias)
            if bn.running_mean is not None:
                quant_conv_bn.bn.running_mean.copy_(bn.running_mean)
            if bn.running_var is not None:
                quant_conv_bn.bn.running_var.copy_(bn.running_var)
            if bn.num_batches_tracked is not None:
                quant_conv_bn.bn.num_batches_tracked.copy_(bn.num_batches_tracked)

        return quant_conv_bn

    def forward(self, x: Union[QuantTensor, Tensor]) -> QuantTensor:
        """
        Conv+BN with optional folding.

        If fold_batch_norm is True, use a fused approximation. Otherwise, apply
        standard conv then BatchNorm.

        Conv: Y = WX + B_c
        Conv without bias: Y0 = WX = Y - B_c, Y = Y0 + B_c

        Batch statistics:
          mean_Y = Y.mean()
                 = Y0.mean() + B_c
          var_Y = (Y - mean_Y)^2.mean()
                = (Y0 - Y0.mean())^2.mean()
        BN (r: bn.weight, beta: bn.bias):
          Z = r * (Y - mean_Y) / sqrt(var_Y + eps) + beta
            = r * (Y0 - Y0.mean()) / sqrt(var_Y + eps) + beta

        Fused Conv BN training (std_Y = sqrt(var_Y + eps)):
          Z = (r * W / std_Y) * X + r * (B_c - mean_Y) / std_Y + beta
            = (r * W / std_Y) * X - r * Y0.mean() / std_Y + beta

        Fused Conv BN inference (running_std = sqrt(running_var + eps)):
          Z = (r * W / running_std) * X - r * (running_mean - B_c) / running_std + beta

        QAT with fused conv bn:
          Z_train = fake_quant(r * W / running_std) * X * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
                  = conv(X, fake_quant(r * W / running_std)) * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
          Z_inference = conv(X, fake_quant(r * W / running_std)) - r * (running_mean - B_c) / running_std + beta
        """

        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1

        # zero-point fusing into bias ----------------------------
        # For affine quantization: (sw*W) * (sa*A+z) + B = sw*sa*(W*A) + B + sw*W*z
        # The term sw*W*z can be precomputed and folded into the bias
        # This is computed by convolving W with a tensor of ones, then multiplying by the zero point
        conv_bias = self.bias
        if not self.activation_quantizer.symmetric:
            # Create a tensor of ones with the same shape as the input
            # We need to compute the contribution of the zero point through the convolution
            # For each output channel: contribution = sw * z * Σ(W[c,i,j,k])
            # We compute this by convolving W with ones
            ones_like_input = torch.ones(1, self.in_channels, 1, 1,
                                         device=self.weight.device, dtype=self.weight.dtype)

            # Get activation zero point and clone to avoid in-place issues
            z = self.activation_quantizer.zero_point.clone()

            # Get weight scale (could be per-channel or scalar)
            sw = self.weight_quantizer.current_scale()
            # Handle per-channel weight scales
            if sw.numel() > 1:
                # sw has shape [out_channels], broadcast to [out_channels, 1, 1, 1]
                sw_broadcast = sw.view(self.out_channels, 1, 1, 1)
                scaled_weight_for_zp = sw_broadcast * self.weight
            else:
                scaled_weight_for_zp = sw * self.weight

            # Compute zero point contribution: convolve scaled weights with ones, multiply by z
            # This computes sw * Σ(W[c,i,j,k]) * z for each output channel
            weight_sum = F.conv2d(
                ones_like_input,
                scaled_weight_for_zp,
                bias=None,
                stride=self.stride,
                padding=(0, 0) if self.padding_mode != 'zeros' else self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            zero_point_contribution = weight_sum * z

            # Keep a 1D bias even when out_channels == 1.
            fused_zero_point_bias = zero_point_contribution.reshape(-1)
            if conv_bias is None:
                conv_bias = fused_zero_point_bias
            else:
                conv_bias = conv_bias + fused_zero_point_bias
        # end zero-point fusing into bias ------------------------

        # Quantize input activation
        if isinstance(x, QuantTensor):
            input_quantized = self.activation_quantizer(x.value, x.scale)
        else:
            input_quantized = self.activation_quantizer(x)

        if not self.fold_batch_norm:
            quant_weight = self.weight_quantizer(self.weight)
            bias_scale = input_quantized.scale.squeeze() * quant_weight.scale.squeeze()
            if conv_bias is not None:
                quant_bias = self.bias_quantizer(conv_bias, bias_scale)
            else:
                quant_bias = None

            conv_out = self._conv2d(input_quantized.value, quant_weight.value, quant_bias)

            output_scale = bias_scale
            if output_scale.numel() > 1:
                output_scale_broadcast = output_scale.view(bias_shape)
            else:
                output_scale_broadcast = output_scale

            conv_out = conv_out * output_scale_broadcast
            bn_out = self.bn(conv_out)
            bn_out = bn_out / output_scale_broadcast

            return QuantTensor(bn_out, output_scale, None)

        if self.training and not self.freeze_bn:
            # get batch statistics
            surrogate_conv = self._conv2d(input_quantized.value * input_quantized.scale, self.weight, conv_bias)
            # detach the bn from the graph, only used to compute the stats, don't want gamma and beta to update here
            with torch.no_grad():
                self.bn(surrogate_conv)
            # compute the stats
            avg_dims = [0] + list(range(2, surrogate_conv.ndim))
            batch_mean = torch.mean(surrogate_conv, dim=avg_dims)
            batch_std = torch.sqrt(torch.var(surrogate_conv, dim=avg_dims, correction=0) + self.bn.eps)

        # always needed
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        running_mean = self.bn.running_mean
        conv_bias_for_fold = torch.zeros_like(running_mean) if conv_bias is None else conv_bias.reshape(-1)

        fold_scale = self.bn.weight / running_std
        quantized_weights_normalized = self.weight_quantizer(self.weight * fold_scale.reshape(weight_shape))
        normalized_conv = self._conv2d(input_quantized.value, quantized_weights_normalized.value, None)
        bias_scale = input_quantized.scale.squeeze() * quantized_weights_normalized.scale.squeeze()
        quantized_bias_normalized = self.bias_quantizer(fold_scale * (conv_bias_for_fold - running_mean) + self.bn.bias, bias_scale)

        # undo running stats by batch stats to recover separate bn behavior
        if self.training and not self.freeze_bn:
            normalized_conv = normalized_conv * (running_std / batch_std).reshape(bias_shape)
            bias_correction = self.bn.weight * (
                (conv_bias_for_fold - batch_mean) / batch_std
                - (conv_bias_for_fold - running_mean) / running_std
            ) / bias_scale.reshape(-1)
            quantized_bias_normalized = quantized_bias_normalized + bias_correction

        normalized_conv = normalized_conv + quantized_bias_normalized.reshape(bias_shape)

        # HACK to let conv bias participate in loss to avoid DDP error (parameters
        #   were not used in producing loss)
        if conv_bias is not None:
            normalized_conv += (conv_bias - conv_bias).reshape(bias_shape)
        return QuantTensor(normalized_conv, bias_scale, None)

    def extra_repr(self):
        return super().extra_repr()

    def train(self, mode=True):
        """
        Keep quantizers and child modules in the requested mode, while preserving
        explicitly frozen BN statistics when requested.
        """
        super().train(mode)
        if self.freeze_bn:
            self.bn.eval()
        return self


class QuantConvBnReLU2d(QuantConvBn2d):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d, and ReLU,
    attached with quantizers for weight, activation, and bias,
    used in quantization-aware training.

    We combined the interface of: class:`torch.nn.Conv2d`,
    :class:`torch.nn.BatchNorm2d` and: class:`torch.nn.ReLU`.

    Attributes:
        weight_quantizer: quantizer for weight
        activation_quantizer: quantizer for activations
        bias_quantizer: quantizer for bias
    """

    def __init__(
        self,
        # Conv2d args
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        reversed_padding_repeated_twice: list[int] = None,
        # BatchNorm2d args
        bn_eps: float = 1e-05,
        bn_momentum: float = 0.1,
        # Quantization args
        weight_quantizer: Optional[BaseQuantizer] = None,
        activation_quantizer: Optional[BaseQuantizer] = None,
        bias_quantizer: Optional[BaseQuantizer] = None,
        freeze_bn: bool = False,
        fold_batch_norm: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            reversed_padding_repeated_twice=reversed_padding_repeated_twice,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            bias_quantizer=bias_quantizer,
            freeze_bn=freeze_bn,
            fold_batch_norm=fold_batch_norm,
        )

        self.relu = QuantReLU(True)

    def forward(self, x: QuantTensor) -> QuantTensor:
        return self.relu(super().forward(x))

    @classmethod
    def from_conv_bn(cls, conv: nn.Conv2d, bn: nn.BatchNorm2d, **kwargs) -> 'QuantConvBnReLU2d':
        """Create QuantConvBnReLU2d from Conv2d and BatchNorm2d layers.

        Args:
            conv: Original Conv2d layer
            bn: Original BatchNorm2d layer
            **kwargs: Additional arguments (weight_quantizer, activation_quantizer,
                     bias_quantizer, freeze_bn)

        Returns:
            QuantConvBnReLU2d layer with copied parameters
        """
        # Delegate to parent class factory method, but instantiate child class
        return super(QuantConvBnReLU2d, cls).from_conv_bn(conv, bn, **kwargs)


class QuantConvReLU2d(nn.Module):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU,
    attached with quantizers for weight, activation, and bias,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and :class:`torch.nn.ReLU`.

    Attributes:
        weight_quantizer: quantizer for weight
        activation_quantizer: quantizer for activations
        bias_quantizer: quantizer for bias
    """

    def __init__(
        self,
        # Conv2d args
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        reversed_padding_repeated_twice: list[int] = None,
        # Quantization args
        weight_quantizer: Optional[BaseQuantizer] = None,
        activation_quantizer: Optional[BaseQuantizer] = None,
        bias_quantizer: Optional[BaseQuantizer] = None,
    ):
        super().__init__()
        self.relu = QuantReLU(True)

        # Import and create QuantConv2d
        from hatorch.layers.quantized_conv2d import QuantConv2d

        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            reversed_padding_repeated_twice=reversed_padding_repeated_twice,
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            bias_quantizer=bias_quantizer,
        )

    def forward(self, x: QuantTensor) -> QuantTensor:
        return self.relu(self.conv(x))

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, **kwargs) -> 'QuantConvReLU2d':
        """Create QuantConvReLU2d from Conv2d layer.

        Args:
            conv: Original Conv2d layer
            **kwargs: Additional arguments (weight_quantizer, activation_quantizer,
                     bias_quantizer)

        Returns:
            QuantConvReLU2d layer with copied parameters
        """
        from hatorch.layers.quantized_conv2d import QuantConv2d

        # Use QuantConv2d's factory method to create the inner conv
        quant_conv = QuantConv2d.from_conv2d(conv, **kwargs)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Assign the created conv
        instance.conv = quant_conv

        return instance
