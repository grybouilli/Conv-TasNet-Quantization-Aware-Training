
from typing import Optional, Self

import torch
from hatorch.modules.hatorch_quantized_layer import HatorchQuantizedOperator
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.base import BaseQuantizer, IdentityQuantizer
from hatorch.transforms.config import LayerQuantConfig
class QuantAdd(HatorchQuantizedOperator):
    """
    Quantized addition operator.

    Applies quantization to input tensors and performs addition.
    Supports different quantizers for each input tensor.

    Args:
        activation_quantizer: Quantizer for activations (default: IdentityQuantizer)
        output_quantizer: Quantizer for output (default: IdentityQuantizer)
    """

    operator_name = "add"

    def __init__(
        self,
        quantize_left_activation: bool,
        quantize_right_activation: bool,
        left_activation_quantizer: Optional[BaseQuantizer] = None,
        right_activation_quantizer: Optional[BaseQuantizer] = None,
        output_quantizer: Optional[BaseQuantizer] = None,
    ):
        super().__init__()

        self.left_activation_quantizer = (
            left_activation_quantizer if left_activation_quantizer is not None else IdentityQuantizer()
        )
        self.right_activation_quantizer = (
            right_activation_quantizer if right_activation_quantizer is not None else IdentityQuantizer()
        )
        self.output_quantizer = (
            output_quantizer if output_quantizer is not None else IdentityQuantizer()
        )

        self.quantize_left_activation = quantize_left_activation
        self.quantize_right_activation = quantize_right_activation

    def forward_hatorch(self, input: QuantTensor, other: QuantTensor, *, alpha=1, out=None) -> QuantTensor:
        # Quantize input activation (quantizers now return QuantTensor)
        
        left = input
        if isinstance(input, QuantTensor):
            left = input.dequantized_value
        right = other
        if isinstance(other, QuantTensor):
            right = other.dequantized_value

        if self.quantize_left_activation:
            left = self.left_activation_quantizer(left).dequantized_value
        if self.quantize_right_activation:
            right = self.right_activation_quantizer(right).dequantized_value

        output = torch.add(
            left,
            right,
            alpha=alpha,
            out=out
        )

        if self.quantize_left_activation or self.quantize_right_activation:
            return self.output_quantizer(output)
        return output

    def forward_onnx(self, input: torch.Tensor, other: torch.Tensor, *, alpha=1, out=None) -> torch.Tensor:
        if self.quantize_left_activation:
            quant_input_qt = self.left_activation_quantizer.onnx_export_forward(input)
        else:
            quant_input_qt = input

        if self.quantize_right_activation:
            quant_other_qt = self.right_activation_quantizer.onnx_export_forward(other)
        else:
            quant_other_qt = other

        output = torch.add(
            quant_input_qt,
            quant_other_qt,
            alpha=alpha,
            out=out
        )

        if self.output_quantizer is not None:
            return self.output_quantizer.onnx_export_forward(output)
        else:
            return output


    @classmethod
    def from_config(
        cls,
        quantize_left_activation: bool,
        quantize_right_activation: bool,
        layer_config: LayerQuantConfig,
    ) -> Self:
        
        left_activation_quantizer = None
        right_activation_quantizer = None
        output_quantizer = None

        if quantize_left_activation and layer_config.activation_quantizer is not None:
            left_activation_quantizer = layer_config.activation_quantizer.create_quantizer()
        if quantize_right_activation and layer_config.activation_quantizer is not None:
            right_activation_quantizer = layer_config.activation_quantizer.create_quantizer()

        
        if (quantize_left_activation or quantize_right_activation) and layer_config.output_quantizer is not None:
            output_quantizer = layer_config.output_quantizer.create_quantizer()

        layer = cls(
            quantize_left_activation=quantize_left_activation,
            quantize_right_activation=quantize_right_activation,
            left_activation_quantizer=left_activation_quantizer,
            right_activation_quantizer=right_activation_quantizer,
            output_quantizer=output_quantizer,
        )

        return layer
