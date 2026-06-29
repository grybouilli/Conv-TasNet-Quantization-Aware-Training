import math
from typing import Literal, Optional, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from hatorch.modules.hatorch_quantized_layer import HatorchQuantizedLayer
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.base import BaseQuantizer, IdentityQuantizer
from hatorch.transforms.config import LayerQuantConfig
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single


class QuantConv1d(HatorchQuantizedLayer[nn.Conv1d]):
    """
    Quantized 1D convolution layer.

    Applies quantization to weights, activations, and optionally bias.
    Supports different quantizers for each component.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution (default: 1)
        padding: Padding added to input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections (default: 1)
        bias: Whether to include bias term (default: True)
        weight_quantizer: Quantizer for weights (default: IdentityQuantizer)
        activation_quantizer: Quantizer for activations (default: IdentityQuantizer)
        bias_quantizer: Quantizer for bias (default: IdentityQuantizer)
    """

    layer_non_quantized_class = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t | str = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        # TODO: this should not be here
        reversed_padding_repeated_twice: list[int] | None = None,
        weight_quantizer: Optional[BaseQuantizer] = None,
        activation_quantizer: Optional[BaseQuantizer] = None,
        output_quantizer: Optional[BaseQuantizer] = None,
        bias_quantizer: Optional[BaseQuantizer] = None,
    ):
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int] = _single(kernel_size)
        self.stride: tuple[int] = _single(stride)
        self.padding: tuple[int] | str = (
            padding if isinstance(padding, str) else _single(padding)
        )
        self.dilation: tuple[int] = _single(dilation)

        self.padding_mode = padding_mode
        self.reversed_padding_repeated_twice = reversed_padding_repeated_twice
        self.groups = groups

        # Initialize weight and bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Set up quantizers
        self.weight_quantizer = (
            weight_quantizer if weight_quantizer is not None else IdentityQuantizer()
        )
        self.activation_quantizer = (
            activation_quantizer
            if activation_quantizer is not None
            else IdentityQuantizer()
        )
        self.bias_quantizer: BaseQuantizer = (
            bias_quantizer if bias_quantizer is not None else IdentityQuantizer()
        )
        self.output_quantizer = (
            output_quantizer if output_quantizer is not None else IdentityQuantizer()
        )

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward_hatorch(self, x: QuantTensor) -> QuantTensor:
        """
        Forward pass with quantization.

        Args:
            x: QuantTensor

        Returns:
            QuantTensor
        """

        # Quantize input activation (quantizers now return QuantTensor)
        quant_input_qt: QuantTensor = self.activation_quantizer(x.dequantized_value)

        # Quantize weight (returns QuantTensor)
        quant_weight_qt: QuantTensor = self.weight_quantizer(self.weight)

        # Quantize bias if present (returns QuantTensor)
        if self.bias is not None:
            quant_bias = self.bias_quantizer(self.bias)
        else:
            quant_bias = None

        if self.padding_mode != "zeros":
            x_value = F.pad(
                quant_input_qt.dequantized_value,
                self.reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )

        else:
            x_value = quant_input_qt.dequantized_value

        output = F.conv1d(
            x_value,
            quant_weight_qt.dequantized_value,
            quant_bias.dequantized_value if quant_bias is not None else None,
            stride=self.stride,
            # TODO: the padding should probably be 0 here if padding mode is not zeros
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return self.output_quantizer(output)

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        quant_input_qt = self.activation_quantizer.onnx_export_forward(x)

        quant_weight_qt = self.weight_quantizer.onnx_export_forward(self.weight)

        quant_bias = None
        if self.bias is not None:
            quant_bias = self.bias_quantizer.onnx_export_forward(self.bias)

        if self.padding_mode != "zeros":
            x_value = F.pad(
                quant_input_qt,
                self.reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )

        else:
            x_value = quant_input_qt

        output = F.conv1d(
            x_value,
            quant_weight_qt,
            quant_bias if quant_bias is not None else None,
            stride=self.stride,
            # TODO: the padding should probably be 0 here if padding mode is not zeros
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return self.output_quantizer.onnx_export_forward(output)

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}"
        )
        if self.padding != (0, 0):
            s += f", padding={self.padding}"
        if self.dilation != (1, 1):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        return s

    @classmethod
    def from_non_quantized_layer(
        cls,
        layer: nn.Conv1d,
        layer_config: LayerQuantConfig,
    ) -> Self:

        # Create quantizers from config
        weight_quantizer = None
        if layer_config.weight_quantizer is not None:
            weight_quantizer = layer_config.weight_quantizer.create_quantizer(
                num_channels=layer.out_channels, channels_axis=0, input_dim=3
            )

        activation_quantizer = None
        if layer_config.activation_quantizer is not None:
            activation_quantizer = layer_config.activation_quantizer.create_quantizer()

        output_quantizer = None
        if layer_config.output_quantizer is not None:
            output_quantizer = layer_config.output_quantizer.create_quantizer()

        bias_quantizer = None
        if layer_config.bias_quantizer is not None:
            bias_quantizer = layer_config.bias_quantizer

        quant_conv = cls(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
            reversed_padding_repeated_twice=layer._reversed_padding_repeated_twice,  # No idea why this is private #TODO: because its automatically computed and shoult nto be set by hand
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            output_quantizer=output_quantizer,
            bias_quantizer=bias_quantizer,
        )

        # Copy weights
        with torch.no_grad():
            quant_conv.weight.copy_(layer.weight)
            if layer.bias is not None:
                quant_conv.bias.copy_(layer.bias)

        return quant_conv
