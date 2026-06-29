import math
from typing import Literal, Optional, Self, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hatorch.modules.hatorch_quantized_layer import HatorchQuantizedLayer
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.base import BaseQuantizer, IdentityQuantizer
from hatorch.transforms.config import LayerQuantConfig
from torch import Tensor, nn
from torch.nn import Parameter


class QuantPReLU(HatorchQuantizedLayer[nn.PReLU]):
    """
    Quantized version of nn.PReLU.
    Handles QuantTensor inputs and outputs while performing ReLU activation.
    """

    layer_non_quantized_class = nn.PReLU

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        device=None,
        dtype=None,
        activation_quantizer: Optional[BaseQuantizer] = None,
    ) -> None:
        super().__init__()

        # todo: maybe quantize weight
        self.weight = Parameter(torch.empty(num_parameters, device=device, dtype=dtype))
        self.init = init
        self.reset_parameters()

        self.activation_quantizer = (
            activation_quantizer
            if activation_quantizer is not None
            else IdentityQuantizer()
        )

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.init)

    def forward_hatorch(self, x: QuantTensor) -> QuantTensor:

        # Quantize input activation (quantizers now return QuantTensor)
        quant_input_qt = self.activation_quantizer(x.dequantized_value)

        # Quantize weight (returns QuantTensor)

        output = F.prelu(
            quant_input_qt.dequantized_value,
            self.weight,
        )

        return QuantTensor(
            value=output,
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=True,
        )

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        quant_input_qt = self.activation_quantizer.onnx_export_forward(x)

        output = F.prelu(
            quant_input_qt,
            self.weight,
        )

        return output

    @classmethod
    def from_non_quantized_layer(
        cls,
        layer: nn.PReLU,
        layer_config: LayerQuantConfig,
    ) -> Self:
        """
        Create a QuantPReLU from an existing PReLU layer.

        Copies weights and configuration from the original layer.

        Args:
            prelu: Original PReLU layer
            weight_quantizer: Quantizer for weights
            activation_quantizer: Quantizer for activations
            bias_quantizer: Quantizer for bias

        Returns:
            QuantPReLU layer with copied parameters
        """
        activation_quantizer = None
        if layer_config.activation_quantizer is not None:
            activation_quantizer = layer_config.activation_quantizer.create_quantizer()

        quant_prelu = cls(
            num_parameters=layer.num_parameters,
            init=layer.init,
            activation_quantizer=activation_quantizer,
        )

        # Copy weights
        with torch.no_grad():
            quant_prelu.weight.copy_(layer.weight)

        return quant_prelu
