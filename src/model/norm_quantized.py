from typing import Self

import torch
import torch.nn as nn
from hatorch.modules.hatorch_quantized_layer import HatorchQuantizedLayer
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.base import BaseQuantizer, IdentityQuantizer
from hatorch.transforms.config import LayerQuantConfig
from torch import nn

from model.norm import (
    GlobalLayerNorm,
    global_layer_norm,
    global_layer_norm_using_torch_layer_norm,
)


class QuantGlobalLayerNorm(HatorchQuantizedLayer[GlobalLayerNorm]):
    """
    Quantized version of GlobalLayerNorm.
    Handles QuantTensor inputs and outputs while performing ReLU activation.
    """

    layer_non_quantized_class = GlobalLayerNorm

    def __init__(
        self,
        channel_size: int,
        eps: float = 1e-8,
        activation_quantizer: BaseQuantizer | None = None,
        gamma_weight_quantizer: BaseQuantizer | None = None,
        beta_weight_quantizer: BaseQuantizer | None = None,
    ) -> None:
        super().__init__()

        self.activation_quantizer = (
            activation_quantizer
            if activation_quantizer is not None
            else IdentityQuantizer()
        )

        self.gamma_weight_quantizer = (
            gamma_weight_quantizer
            if gamma_weight_quantizer is not None
            else IdentityQuantizer()
        )

        self.beta_weight_quantizer = (
            beta_weight_quantizer
            if beta_weight_quantizer is not None
            else IdentityQuantizer()
        )

        self.channel_size = channel_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1), requires_grad=True)

    def forward_hatorch(self, x: QuantTensor) -> QuantTensor:

        quant_input_qt = self.activation_quantizer(x.dequantized_value)

        # Quantize weight (returns QuantTensor)
        gamma_qt: QuantTensor = self.gamma_weight_quantizer(self.gamma)
        beta_qt: QuantTensor = self.beta_weight_quantizer(self.beta)

        # Compute output of layer norm using quantized input
        value = quant_input_qt.dequantized_value

        output = global_layer_norm(
            value, gamma_qt.dequantized_value, beta_qt.dequantized_value, self.eps
        )

        return QuantTensor(
            value=output,
            scale=torch.ones(1, device=x.device, dtype=x.dtype),
            zero_point=None,
            bit_width=quant_input_qt.bit_width,  # Use input bit width for output
            signed=True,
        )

    def forward_onnx(self, x: torch.Tensor):
        quant_input_qt = self.activation_quantizer.onnx_export_forward(x)

        gamma = self.gamma_weight_quantizer.onnx_export_forward(self.gamma)
        beta = self.beta_weight_quantizer.onnx_export_forward(self.beta)

        # We use global_layer_norm_using_torch_layer_norm to export a LayerNorm node in the ONNX graph, which can be optimized by ONNX Runtime
        # output = global_layer_norm(quant_input_qt, self.gamma, self.beta, self.eps)
        output = global_layer_norm_using_torch_layer_norm(
            quant_input_qt, gamma, beta, self.channel_size, self.eps
        )

        return output

    @classmethod
    def from_non_quantized_layer(
        cls,
        layer: GlobalLayerNorm,
        layer_config: LayerQuantConfig,
    ) -> Self:
        """
        Create a QuantGlobalLayerNorm from an existing GlobalLayerNorm layer.

        Copies weights and configuration from the original layer.

        Args:
            global_layernorm: Original GlobalLayerNorm layer
            activation_quantizer: Quantizer for activations

        Returns:
            QuantGlobalLayerNorm layer with copied parameters
        """
        activation_quantizer = None
        if layer_config.activation_quantizer is not None:
            activation_quantizer = layer_config.activation_quantizer.create_quantizer()

        gamma_weight_quantizer = None
        if layer_config.weight_quantizer is not None:
            gamma_weight_quantizer = layer_config.weight_quantizer.create_quantizer(
                layer.channel_size, channels_axis=1, input_dim=3
            )
        beta_weight_quantizer = None
        if layer_config.weight_quantizer is not None:
            beta_weight_quantizer = layer_config.weight_quantizer.create_quantizer(
                layer.channel_size, channels_axis=1, input_dim=3
            )

        quant_globallayernorm = cls(
            channel_size=layer.channel_size,
            activation_quantizer=activation_quantizer,
            gamma_weight_quantizer=gamma_weight_quantizer,
            beta_weight_quantizer=beta_weight_quantizer,
        )

        # Copy weights
        with torch.no_grad():
            quant_globallayernorm.gamma.copy_(layer.gamma)
            quant_globallayernorm.beta.copy_(layer.beta)

        return quant_globallayernorm
