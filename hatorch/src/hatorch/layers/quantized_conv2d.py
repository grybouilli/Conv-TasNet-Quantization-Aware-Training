"""
Quantized 2D convolution layer.

This module provides a quantized version of PyTorch's Conv2d layer,
using QuantTensor for proper quantization flow and configurable quantizers.
"""

import math
from typing import Optional, Union, Literal, Self
import torch
import torch.nn as nn
import torch.nn.functional as F

from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.modules.hatorch_quantized_layer import HatorchQuantizedLayer
from hatorch.quantizers.base import BaseQuantizer, IdentityQuantizer
from hatorch.transforms.config import LayerQuantConfig


class QuantConv2d(HatorchQuantizedLayer[nn.Conv2d]):
    """
    Quantized 2D convolution layer.
    
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

    layer_non_quantized_class = nn.Conv2d
    
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
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        reversed_padding_repeated_twice: list[int] = None,
        weight_quantizer: Optional[BaseQuantizer] = None,
        activation_quantizer: Optional[BaseQuantizer] = None,
        bias_quantizer: Optional[BaseQuantizer] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding_mode = padding_mode
        self.reversed_padding_repeated_twice = reversed_padding_repeated_twice
        self.groups = groups

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        # Set up quantizers
        self.weight_quantizer = weight_quantizer if weight_quantizer is not None else IdentityQuantizer()
        self.activation_quantizer = activation_quantizer if activation_quantizer is not None else IdentityQuantizer()
        self.bias_quantizer = bias_quantizer if bias_quantizer is not None else IdentityQuantizer()
    
    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: QuantTensor) -> QuantTensor:
        """
        Forward pass with quantization.
        
        Args:
            x: QuantTensor
            
        Returns:
            QuantTensor
        """
        # Quantize input activation (quantizers now return QuantTensor)
        quant_input_qt = self.activation_quantizer(x.dequantized_value)
        
        # Quantize weight (returns QuantTensor)
        quant_weight_qt = self.weight_quantizer(self.weight)

        # Quantize bias if present (returns QuantTensor)
        if self.bias is not None:
            # bias scale = input_scale * weight_scale
            # Handle per-channel weight scales: both should be [out_channels] or scalar
            bias_scale = quant_input_qt.scale.squeeze() * quant_weight_qt.scale.squeeze()
            quant_bias = self.bias_quantizer(self.bias, bias_scale)
        else:
            quant_bias = None
        
        # Perform convolution with quantized values (extract .value from QuantTensors)
        if self.padding_mode != "zeros":
            x = F.pad(
                quant_input_qt.dequantized_value,
                self.reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
        else:
            x = quant_input_qt.dequantized_value

        output = F.conv2d(
            x,
            quant_weight_qt.dequantized_value,
            quant_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # The result is a fully dequantized QuantTensor
        return QuantTensor(
            value=output,
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=True,
        )
    
    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        quant_input_qt = self.activation_quantizer.onnx_export_forward(x)

        quant_weight_qt = self.weight_quantizer.onnx_export_forward(self.weight)
        quant_bias = None
        if self.bias is not None:
            quant_bias = self.bias_quantizer.onnx_export_forward(self.bias)
        
        # TODO: padding may be missing
        
        output = F.conv2d(
            quant_input_qt,
            quant_weight_qt,
            quant_bias if quant_bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return output
    
    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        s = (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
             f"stride={self.stride}")
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
        layer: nn.Conv2d,
        layer_config: LayerQuantConfig,
    ) -> Self:
        """
        Create a QuantConv2d from an existing Conv2d layer.
        
        Copies weights and configuration from the original layer.

        Returns:
            QuantConv2d layer with copied parameters
        """
        # Create quantizers from config
        weight_quantizer = None
        if layer_config.weight_quantizer is not None:
            weight_quantizer = layer_config.weight_quantizer.create_quantizer(num_channels=layer.out_channels, channels_axis=0, input_dim=3)
        
        activation_quantizer = None
        if layer_config.activation_quantizer is not None:
            activation_quantizer = layer_config.activation_quantizer.create_quantizer()
        
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
            reversed_padding_repeated_twice=layer._reversed_padding_repeated_twice, # No idea why this is private
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            bias_quantizer=bias_quantizer,
        )
        
        # Copy weights
        with torch.no_grad():
            quant_conv.weight.copy_(layer.weight)
            if layer.bias is not None:
                quant_conv.bias.copy_(layer.bias)
        
        return quant_conv
