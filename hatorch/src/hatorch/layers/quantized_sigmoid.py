from typing import Self

import torch
import torch.nn as nn
from hatorch.modules.hatorch_quantized_layer import HatorchQuantizedLayer
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.base import BaseQuantizer, IdentityQuantizer
from hatorch.transforms.config import LayerQuantConfig


class QuantSigmoid(HatorchQuantizedLayer[nn.Sigmoid]):
    """
    Quantized version of nn.Sigmoid.
    Handles QuantTensor inputs and outputs while performing Sigmoid activation.
    """

    layer_non_quantized_class = nn.Sigmoid

    def __init__(
        self,
        activation_quantizer: BaseQuantizer | None = None,
    ) -> None:
        super().__init__()

        self.activation_quantizer = (
            activation_quantizer
            if activation_quantizer is not None
            else IdentityQuantizer()
        )

    def forward_hatorch(self, x: QuantTensor) -> QuantTensor:
        quant_input_qt = self.activation_quantizer(x.dequantized_value)

        output = torch.sigmoid(quant_input_qt.dequantized_value)

        return QuantTensor(
            value=output,
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=True,
        )

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    @classmethod
    def from_non_quantized_layer(
        cls,
        layer: nn.Sigmoid,
        layer_config: LayerQuantConfig,
    ) -> Self:

        activation_quantizer = None
        if layer_config.activation_quantizer is not None:
            activation_quantizer = layer_config.activation_quantizer.create_quantizer()

        return cls(
            activation_quantizer=activation_quantizer,
        )
