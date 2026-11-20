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
        output = global_layer_norm(quant_input_qt, self.gamma, self.beta, self.eps)
        # output = global_layer_norm_using_torch_layer_norm(
        #     quant_input_qt, gamma, beta, self.channel_size, self.eps
        # )

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

from typing import Self
import torch.nn.functional as F


class QuantBatchNorm1d(HatorchQuantizedLayer[nn.BatchNorm1d]):
    """
    Quantized version of nn.BatchNorm1d.

    Accepts QuantTensor inputs of shape (N, C) or (N, C, L) and returns a
    QuantTensor. The input activation and the affine parameters (gamma/beta)
    are fake-quantized; running statistics are kept in full precision.

    State-dict compatible with nn.BatchNorm1d (weight, bias, running_mean,
    running_var, num_batches_tracked).
    """

    layer_non_quantized_class = nn.BatchNorm1d

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
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

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_input_dim(x: torch.Tensor) -> None:
        if x.dim() not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {x.dim()}D input)")

    def _momentum_factor(self) -> float:
        """Same momentum logic as torch's _BatchNorm (supports momentum=None)."""
        factor = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # cumulative moving average
                factor = 1.0 / float(self.num_batches_tracked.item())
        return factor

    def _use_batch_stats(self) -> bool:
        if self.training:
            return True
        return self.running_mean is None and self.running_var is None

    # ------------------------------------------------------------------ #
    # Forward paths
    # ------------------------------------------------------------------ #
    def forward_hatorch(self, x: QuantTensor) -> QuantTensor:
        self._check_input_dim(x.dequantized_value)

        # Quantize input activation
        quant_input_qt = self.activation_quantizer(x.dequantized_value)
        value = quant_input_qt.dequantized_value

        # Quantize affine parameters (viewed as (1, C, 1) so per-channel
        # quantizers created with channels_axis=1, input_dim=3 apply)
        gamma = beta = None
        if self.affine:
            gamma_qt: QuantTensor = self.gamma_weight_quantizer(
                self.weight.view(1, -1, 1)
            )
            beta_qt: QuantTensor = self.beta_weight_quantizer(
                self.bias.view(1, -1, 1)
            )
            gamma = gamma_qt.dequantized_value.reshape(-1)
            beta = beta_qt.dequantized_value.reshape(-1)

        use_batch_stats = self._use_batch_stats()
        output = F.batch_norm(
            value,
            # Only pass running stats when they should be used / updated
            self.running_mean if (not self.training or self.track_running_stats) else None,
            self.running_var if (not self.training or self.track_running_stats) else None,
            gamma,
            beta,
            use_batch_stats,
            self._momentum_factor(),
            self.eps,
        )

        return QuantTensor(
            value=output,
            scale=torch.ones(1, device=output.device, dtype=output.dtype),
            zero_point=None,
            bit_width=quant_input_qt.bit_width,  # Use input bit width for output
            signed=True,
        )

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        quant_input = self.activation_quantizer.onnx_export_forward(x)

        gamma = beta = None
        if self.affine:
            gamma = self.gamma_weight_quantizer.onnx_export_forward(
                self.weight.view(1, -1, 1)
            ).reshape(-1)
            beta = self.beta_weight_quantizer.onnx_export_forward(
                self.bias.view(1, -1, 1)
            ).reshape(-1)

        # Export always uses running stats (inference), producing a
        # BatchNormalization node that ONNX Runtime can fuse/optimize.
        return F.batch_norm(
            quant_input,
            self.running_mean,
            self.running_var,
            gamma,
            beta,
            False,
            0.0,
            self.eps,
        )

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #
    @classmethod
    def from_non_quantized_layer(
        cls,
        layer: nn.BatchNorm1d,
        layer_config: LayerQuantConfig,
    ) -> Self:
        """
        Create a QuantBatchNorm1d from an existing nn.BatchNorm1d layer.

        Copies parameters, running statistics and configuration.

        Args:
            layer: Original BatchNorm1d layer
            layer_config: Quantization config (activation / weight quantizers)

        Returns:
            QuantBatchNorm1d layer with copied parameters and buffers
        """
        activation_quantizer = None
        if layer_config.activation_quantizer is not None:
            activation_quantizer = layer_config.activation_quantizer.create_quantizer()

        gamma_weight_quantizer = None
        beta_weight_quantizer = None
        if layer.affine and layer_config.weight_quantizer is not None:
            gamma_weight_quantizer = layer_config.weight_quantizer.create_quantizer(
                layer.num_features, channels_axis=1, input_dim=3
            )
            beta_weight_quantizer = layer_config.weight_quantizer.create_quantizer(
                layer.num_features, channels_axis=1, input_dim=3
            )

        quant_bn = cls(
            num_features=layer.num_features,
            eps=layer.eps,
            momentum=layer.momentum,
            affine=layer.affine,
            track_running_stats=layer.track_running_stats,
            activation_quantizer=activation_quantizer,
            gamma_weight_quantizer=gamma_weight_quantizer,
            beta_weight_quantizer=beta_weight_quantizer,
        )

        # Copy parameters and buffers
        with torch.no_grad():
            if layer.affine:
                quant_bn.weight.copy_(layer.weight)
                quant_bn.bias.copy_(layer.bias)
            if layer.track_running_stats:
                quant_bn.running_mean.copy_(layer.running_mean)
                quant_bn.running_var.copy_(layer.running_var)
                quant_bn.num_batches_tracked.copy_(layer.num_batches_tracked)

        quant_bn.train(layer.training)
        return quant_bn