"""
Bias quantizer that constrains scale to be weight_scale * activation_scale.

This quantizer enforces the constraint that bias quantization scale must equal
the product of weight and activation scales, which is necessary for proper
numerical behavior in quantized inference.

The scale is provided by the layer (Conv/Linear) at forward time.
"""

import torch
import torch.nn as nn
from torch import Tensor

from hatorch.quantizers.autograd.lsq_uniform_quantizer import LsqUniformQuantizer as LsqUniformQuantizerFunction


class BiasQuantizer(nn.Module):
    """
    Simplified bias quantizer with constrained scale.
    
    This quantizer takes a scale as input (computed by the layer as weight_scale * activation_scale)
    and applies quantization/dequantization. It returns a plain Tensor, not a QuantTensor.
    
    The layer (Conv/Linear) is responsible for computing the scale and passing it to this quantizer.
    
    Args:
        bit_width: Number of bits for bias quantization (default: 32)
        signed: Whether to use signed quantization (default: True)
        enabled: Whether quantization is enabled (default: True)
    """
    
    def __init__(
        self,
        bit_width: int = 32,
        signed: bool = True,
        enabled: bool = True,
    ):
        super().__init__()
        
        self.bit_width = bit_width
        self.signed = signed
        self.enabled = enabled
        
        # Calculate quantization range [qn, qp]
        if signed:
            self.qn = -(2 ** (bit_width - 1))
            self.qp = 2 ** (bit_width - 1) - 1
        else:
            self.qn = 0
            self.qp = 2 ** bit_width - 1
    
    def forward(self, bias: Tensor, scale: Tensor) -> Tensor:
        """
        Apply bias quantization with provided scale.
        
        Args:
            bias: Bias tensor to quantize
            scale: Scale to use (should be weight_scale * activation_scale)
            
        Returns:
            Quantized and dequantized bias tensor
        """
        if not self.enabled:
            return bias
        
        # Ensure scale is on the same device as bias
        if scale.device != bias.device:
            scale = scale.to(bias.device)

        # Create zero-point tensor (always zero for bias quantization)
        zero_point = torch.zeros_like(scale)
        quantized_bias = LsqUniformQuantizerFunction.apply(
            bias, scale, zero_point, self.qn, self.qp
        )
        
        return quantized_bias
    
    def extra_repr(self) -> str:
        """Return string representation of quantizer parameters."""
        s = (f"bit_width={self.bit_width}, signed={self.signed}, "
             f"symmetric=True, constrained_scale=True")
        s += f", qn={self.qn}, qp={self.qp}, enabled={self.enabled}"
        return s
