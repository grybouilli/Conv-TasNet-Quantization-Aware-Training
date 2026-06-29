"""
Quantized linear (fully connected) layer.

This module provides a quantized version of PyTorch's Linear layer,
using QuantTensor for proper quantization flow and configurable quantizers.
"""

import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.base import BaseQuantizer, IdentityQuantizer


class QuantLinear(nn.Module):
    """
    Quantized linear (fully connected) layer.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term (default: True)
        weight_quantizer: Quantizer for weights
        activation_quantizer: Quantizer for activations
        bias_quantizer: Quantizer for bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_quantizer: Optional[BaseQuantizer] = None,
        activation_quantizer: Optional[BaseQuantizer] = None,
        bias_quantizer: Optional[BaseQuantizer] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
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
    
    def forward(self, x: Union[QuantTensor, Tensor]) -> QuantTensor:
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor or QuantTensor
            
        Returns:
            Output tensor or QuantTensor
        """
        # Accept plain Tensor by wrapping with identity quantization
        if isinstance(x, QuantTensor):
            quant_input_qt = self.activation_quantizer(x.value, x.scale)
        else:
            quant_input_qt = self.activation_quantizer(x)
        
        # Quantize weight (returns QuantTensor)
        quant_weight_qt = self.weight_quantizer(self.weight)
        
        # zero-point fusing into bias ----------------------------
        # For affine quantization: (sw*W) * (sa*A+z) + B = sw*sa*(W*A) + B + sw*W*z
        # The term sw*W*z can be precomputed and folded into the bias
        # For linear layers: sw*W*z sums to sw * sum(W, dim=1) * z per output neuron
        linear_bias = self.bias
        if not self.activation_quantizer.symmetric:
            # Get activation zero point
            z = self.activation_quantizer.zero_point
            
            # Get weight scale (could be per-channel or scalar)
            sw = self.weight_quantizer.current_scale()
            
            # For linear layers: sw*W*z = sw * sum(W, dim=1) * z per output neuron
            # sum over input features dimension (dim=1)
            if sw.numel() > 1:
                # Per-channel weight scales: each output neuron has its own sw[i]
                zero_point_contribution = sw * torch.sum(self.weight, dim=1) * z
            else:
                # Scalar weight scale
                zero_point_contribution = sw * torch.sum(self.weight, dim=1) * z
            
            if linear_bias is None:
                linear_bias = zero_point_contribution
            else:
                linear_bias = linear_bias + zero_point_contribution
        # end zero-point fusing into bias ------------------------
        
        # Quantize bias if present (returns QuantTensor)
        if linear_bias is not None:
            # bias scale = input_scale * weight_scale
            # Handle per-channel weight scales: both should be [out_channels] or scalar
            bias_scale = quant_input_qt.scale.squeeze() * quant_weight_qt.scale.squeeze()
            quant_bias = self.bias_quantizer(linear_bias, bias_scale)
        else:
            quant_bias = None

        # Reshape scales/zero_points for per-channel quantization (channel-first)
        if quant_weight_qt.value.numel() > 1:
            # Weight: [out_channels, in_channels/groups, kH, kW]
            # Scale: [out_channels] -> [out_channels, 1, 1, 1]
            new_shape = [quant_weight_qt.scale.numel()] + [1] * (quant_input_qt.value.ndim - 1)
            quant_weight_qt.scale = quant_weight_qt.scale.view(new_shape)
            if quant_weight_qt.zero_point is not None:
                quant_weight_qt.zero_point = quant_weight_qt.zero_point.view(new_shape)
        
        # Perform linear operation with quantized values (extract .value from QuantTensors)
        # Note: quantizers return dequantized values in .value
        output = F.linear(
            quant_input_qt.value,
            quant_weight_qt.value,
            quant_bias
        )

        # Compute output scale: weight_scale * input_scale
        output_scale = quant_weight_qt.scale.squeeze() * quant_input_qt.scale.squeeze()

        # Handle per-channel scales (weight quantization might be per-channel)
        if output_scale.numel() > 1:
            # Per-channel output scale - reshape to broadcast correctly
            output_scale = output_scale.view(-1, 1)

        return QuantTensor(
            value=output,
            scale=output_scale,
            zero_point=None,
            bit_width=quant_input_qt.bit_width,
            signed=True,
        )
    
    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_quantizer: Optional[BaseQuantizer] = None,
        activation_quantizer: Optional[BaseQuantizer] = None,
        bias_quantizer: Optional[BaseQuantizer] = None,
    ) -> 'QuantLinear':
        """
        Create a QuantLinear from an existing Linear layer.
        
        Args:
            linear: Original Linear layer
            weight_quantizer: Quantizer for weights
            activation_quantizer: Quantizer for activations
            bias_quantizer: Quantizer for bias

        Returns:
            QuantLinear layer with copied parameters
        """
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            bias_quantizer=bias_quantizer,
        )
        
        # Copy weights
        with torch.no_grad():
            quant_linear.weight.copy_(linear.weight)
            if linear.bias is not None:
                quant_linear.bias.copy_(linear.bias)
        
        return quant_linear
