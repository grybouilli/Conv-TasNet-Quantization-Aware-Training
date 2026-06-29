"""
Configuration classes for quantization.

This module provides configuration classes to specify which quantizers
should be used for different components (weights, activations, bias) in quantized layers.
"""

import inspect
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import copy

from torch import Tensor

from hatorch.utils.logger import logger
from hatorch.quantizers.base import BaseQuantizer
from hatorch.quantizers.common.scale_approximation import ScaleApproximation
from hatorch.quantizers.init.lsq_uniform import LsqUniformQuantizer
from hatorch.quantizers.init.step_driven import StepDrivenQuantizer
from hatorch.quantizers.bias.bias_quantizer import BiasQuantizer


@dataclass
class QuantizerSpec:
    """
    Specification for a quantizer instance.

    Attributes:
        quantizer_class: The quantizer class to instantiate
        is_activation: Whether this quantizer is for activations (True) or weights (False)
        symmetric: Whether to use symmetric quantization around zero
        per_channel: Whether to use per-channel quantization
        kwargs: Additional keyword arguments to pass to the quantizer constructor
    """
    quantizer_class: type[BaseQuantizer]
    is_activation: bool
    symmetric: bool
    per_channel: bool
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the quantizer specification."""
        if self.is_activation:
            if not issubclass(self.quantizer_class, LsqUniformQuantizer):
                raise ValueError("Activations can only use LsqUniformQuantizer.")
            if self.per_channel:
                logger.warning(
                    "Per-channel quantization is unsupported for activations. This will be ignored.")
                self.per_channel = False
        else:
            if not self.symmetric:
                logger.warning(
                    "Asymmetric quantization for weights is unsupported by the lowering process. This will be ignored.")
                self.symmetric = True

    def create_quantizer(self, num_channels: Optional[int] = None, channels_axis: Optional[int] = None, input_dim: Optional[int] = None) -> 'BaseQuantizer':
        """
        Create an instance of the quantizer.

        Returns:
            Instantiated quantizer
        """

        # Deep copy kwargs to avoid sharing mutable objects (like observers) between quantizers
        kwargs_copy = copy.deepcopy(self.kwargs)
        signature = inspect.signature(self.quantizer_class.__init__)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            kwargs_copy = {
                key: value
                for key, value in kwargs_copy.items()
                if key in signature.parameters
            }
        return self.quantizer_class(is_activation=self.is_activation, symmetric=self.symmetric, per_channel=self.per_channel, num_channels=num_channels, channels_axis=channels_axis, input_dim=input_dim, **kwargs_copy)


@dataclass
class LayerQuantConfig:
    """
    Quantization configuration for a single layer.
    
    Specifies which quantizers to use for weights, activations, and bias.
    
    Attributes:
        weight_quantizer: Specification for weight quantizer (None = no quantization)
        activation_quantizer: Specification for activation quantizer (None = no quantization)
        output_quantizer: Specification for output quantizer (None = no quantization)
        bias_quantizer: Specification for bias quantizer (None = no quantization)
    """
    weight_quantizer: Optional[QuantizerSpec] = None
    activation_quantizer: Optional[QuantizerSpec] = None
    output_quantizer: Optional[QuantizerSpec] = None
    bias_quantizer: Optional[BiasQuantizer] = None
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure specs are QuantizerSpec instances or None
        if self.weight_quantizer is not None and not isinstance(self.weight_quantizer, QuantizerSpec):
            raise TypeError("weight_quantizer must be a QuantizerSpec or None")
        if self.activation_quantizer is not None and not isinstance(self.activation_quantizer, QuantizerSpec):
            raise TypeError("activation_quantizer must be a QuantizerSpec or None")
        if self.output_quantizer is not None and not isinstance(self.output_quantizer, QuantizerSpec):
            raise TypeError("output_quantizer must be a QuantizerSpec or None")
        if self.bias_quantizer is not None and not isinstance(self.bias_quantizer, BiasQuantizer):
            raise TypeError("bias_quantizer must be a BiasQuantizer or None")


@dataclass
class ModelQuantConfig:
    """
    Quantization configuration for an entire model.
    
    Provides default quantization settings and per-layer overrides.
    
    Attributes:
        default_config: Default configuration for all layers
        layer_configs: Dictionary mapping layer names to specific configurations
        first_layer_config: Optional configuration specifically for the first quantized layer
        last_layer_config: Optional configuration specifically for the last quantized layer
        first_layer_symmetric_activation: Whether first layer activation should be symmetric (default: True)
        scale_approximation: Scale approximation method (NONE, POWER_OF_TWO, FIXED_POINT)
        fixed_point_bits: Number of fractional bits for fixed-point approximation
        fold_batch_norm: Whether to fold BN into conv during QAT (default: True)
        freeze_approximated_scales: Whether LSQ scales should be frozen when
            scale approximation is enabled for them.
    """
    default_config: LayerQuantConfig
    layer_configs: Dict[str, LayerQuantConfig] = field(default_factory=dict)
    first_layer_config: Optional[LayerQuantConfig] = None
    last_layer_config: Optional[LayerQuantConfig] = None
    first_layer_symmetric_activation: bool = True
    scale_approximation: ScaleApproximation = ScaleApproximation.NONE
    fixed_point_bits: int = 32
    fold_batch_norm: bool = True
    target_scale_approximation: Optional[ScaleApproximation] = None
    scale_approximation_start_epoch: int = 1
    freeze_approximated_scales: bool = False
    
    def get_config_for_layer(self, layer_name: str, is_first_layer: bool = False, is_last_layer: bool = False) -> LayerQuantConfig:
        """
        Get quantization configuration for a specific layer.
        
        Args:
            layer_name: Name/identifier of the layer
            is_first_layer: Whether this is the first quantized layer
            is_last_layer: Whether this is the last quantized layer
            
        Returns:
            LayerQuantConfig for the specified layer (falls back to default if not specified)
        """
        import copy
        
        # Priority: layer_configs > first/last_layer_config > default_config
        if layer_name in self.layer_configs:
            base_config = self.layer_configs[layer_name]
        elif is_last_layer and self.last_layer_config is not None:
            base_config = self.last_layer_config
        elif is_first_layer and self.first_layer_config is not None:
            base_config = self.first_layer_config
        else:
            base_config = self.default_config
        
        # If first layer and we need symmetric activation, modify the config
        if is_first_layer and self.first_layer_symmetric_activation and base_config.activation_quantizer is not None:
            # Deep copy to avoid modifying the original
            config = copy.deepcopy(base_config)
            # Modify activation quantizer to be signed
            if 'signed' in config.activation_quantizer.kwargs:
                config.activation_quantizer.kwargs['signed'] = True
            return config
        
        return base_config


# Convenience functions for common configurations

def uniform_8bit_config(
    quantize_weights: bool = True,
    quantize_activations: bool = True,
    quantize_output: bool = False,
    quantize_bias: bool = False,
    affine_activations: bool = True,
    affine_output: bool = True,
    per_channel_weights: bool = False,
    signed_weights: bool = True,
    signed_activations: bool = False,
    signed_output: bool = False,
) -> LayerQuantConfig:
    """
    Create a configuration for uniform 8-bit quantization.
    
    Args:
        quantize_weights: Whether to quantize weights
        quantize_activations: Whether to quantize activations
        quantize_output: Whether to quantize output
        quantize_bias: Whether to quantize bias
        affine_activations: Whether activations are affine (affects symmetry)
        affine_output: Whether output is affine (affects symmetry)
        per_channel_weights: Whether to use per-channel quantization for weights
        signed_weights: Whether weights should be signed (default: True)
        signed_activations: Whether activations should be signed (default: False)
        signed_output: Whether output should be signed (default: False)
        
    Returns:
        LayerQuantConfig with 8-bit LSQ quantizers
    """

    weight_spec = None
    if quantize_weights:
        weight_spec = QuantizerSpec(
            quantizer_class=LsqUniformQuantizer,
            is_activation=False,
            symmetric=True,
            per_channel=per_channel_weights,
            kwargs={
                'bit_width': 8,
                'signed': signed_weights,
            }
        )

    activation_spec = None
    if quantize_activations:
        activation_spec = QuantizerSpec(
            quantizer_class=LsqUniformQuantizer,
            is_activation=True,
            symmetric=not affine_activations,
            per_channel=False,
            kwargs={
                'bit_width': 8,
                'signed': signed_activations,
            }
        )
    
    output_spec = None
    if quantize_output:
        output_spec = QuantizerSpec(
            quantizer_class=LsqUniformQuantizer,
            is_activation=True,
            symmetric=not affine_output,
            per_channel=False,
            kwargs={
                'bit_width': 8,
                'signed': signed_output,
            }
        )

    bias_spec = None
    if quantize_bias:
        bias_spec = BiasQuantizer(bit_width=32, signed=True, enabled=True)

    return LayerQuantConfig(
        weight_quantizer=weight_spec,
        activation_quantizer=activation_spec,
        output_quantizer=output_spec,
        bias_quantizer=bias_spec,
    )

def mixed_precision_config(
    quantize_activations: bool = True,
    quantize_weights: bool = True,
    quantize_output: bool = False,
    weight_bits: int | Tensor = 8,
    activation_bits: int = 8,
    output_bits: int = 8,
    quantize_bias: bool = True,
    bias_bits: int = 32,
    affine_activations: bool = True,
    affine_output: bool = True,
    per_channel_weights: bool = False,
    scale_approximation: ScaleApproximation = ScaleApproximation.NONE,
    fixed_point_bits: int = 8,
    signed_weights: bool = True,
    signed_activations: bool = False,
    signed_output: bool = False,
) -> LayerQuantConfig:
    """
    Create a mixed-precision quantization configuration.
    
    Args:
        weight_bits: Integer bit width for uniform quantization, or tensor codebook
            targets for step-driven non-uniform quantization.
        activation_bits: Bit width for activations
        bias_bits: Bit width for bias
        affine_activations: Whether activations are affine (affects symmetry)
        per_channel_weights: Whether to use per-channel quantization for weights
        scale_approximation: Scale approximation method (NONE, POWER_OF_TWO, FIXED_POINT)
        fixed_point_bits: Number of fractional bits for fixed-point approximation

    Returns:
        LayerQuantConfig with specified bit widths
    """
    weight_spec = None
    if quantize_weights:
        if isinstance(weight_bits, Tensor):
            weight_quantizer_class = StepDrivenQuantizer
            weight_kwargs = {
                'bit_width': weight_bits,
                'signed': signed_weights,
            }
        else:
            weight_quantizer_class = LsqUniformQuantizer
            weight_kwargs = {
                'bit_width': int(weight_bits),
                'signed': signed_weights,
                'scale_approximation': scale_approximation,
                'fixed_point_bits': fixed_point_bits,
            }

        weight_spec = QuantizerSpec(
            quantizer_class=weight_quantizer_class,
            is_activation=False,
            symmetric=True,
            per_channel=per_channel_weights,
            kwargs=weight_kwargs
        )
    
    activation_spec = None
    if quantize_activations:
        activation_spec = QuantizerSpec(
            quantizer_class=LsqUniformQuantizer,
            is_activation=True,
            symmetric=not affine_activations,
            per_channel=False,
            kwargs={
                'bit_width': activation_bits,
                'signed': signed_activations,
                'scale_approximation': scale_approximation,
                'fixed_point_bits': fixed_point_bits,
            }
        )
    
    output_spec = None
    if quantize_output:
        output_spec = QuantizerSpec(
            quantizer_class=LsqUniformQuantizer,
            is_activation=True,
            symmetric=not affine_output,
            per_channel=False,
            kwargs={
                'bit_width': output_bits,
                'signed': signed_output,
                'scale_approximation': scale_approximation,
                'fixed_point_bits': fixed_point_bits,
            }
        )
    
    bias_quantizer = None
    if quantize_bias:
        bias_quantizer = BiasQuantizer(bit_width=bias_bits, signed=True, enabled=True)
    
    return LayerQuantConfig(
        weight_quantizer=weight_spec,
        activation_quantizer=activation_spec,
        output_quantizer=output_spec,
        bias_quantizer=bias_quantizer,
    )


def create_model_config_with_edge_layers(
    default_weight_bits: int | Tensor = 8,
    default_activation_bits: int | Tensor = 8,
    first_layer_weight_bits: int | Tensor | None = None,
    first_layer_activation_bits: int = None,
    last_layer_weight_bits: int | Tensor | None = None,
    last_layer_activation_bits: int = None,
    bias_bits: int = 32,
    affine_activations: bool = True,
    per_channel_weights: bool = False,
    scale_approximation: ScaleApproximation = ScaleApproximation.NONE,
    scale_approximation_start_epoch: int = 1,
    freeze_approximated_scales: bool = False,
    fixed_point_bits: int = 8,
    fold_batch_norm: bool = True,
) -> ModelQuantConfig:
    """
    Create a ModelQuantConfig with separate bit widths for first and last layers.
    
    This is a convenience function to easily configure different quantization
    bit widths for the first layer (typically higher precision for input features)
    and last layer (typically higher precision for classification logits).
    
    Args:
        default_weight_bits: Integer bit width for uniform quantization, or tensor
            codebook targets for step-driven non-uniform quantization.
        default_activation_bits: Bit width for activations in middle layers
        first_layer_weight_bits: Weight settings for first layer (None = use default)
        first_layer_activation_bits: Bit width for first layer activations (None = use default)
        last_layer_weight_bits: Weight settings for last layer (None = use default)
        last_layer_activation_bits: Bit width for last layer activations (None = use default)
        bias_bits: Bit width for bias terms (typically high precision)
        affine_activations: Whether activations are affine (affects symmetry)
        per_channel_weights: Whether to use per-channel quantization for weights
        scale_approximation: Scale approximation method (NONE, POWER_OF_TWO, FIXED_POINT)
        scale_approximation_start_epoch: 1-based epoch where scale approximation is enabled.
            Use 1 to enable it from the start. Values greater than 1 instantiate the
            model with exact scales and rely on training epoch callbacks to enable it.
        freeze_approximated_scales: Freeze only LSQ scales that use scale approximation,
            at the same time the approximation is enabled.
        fixed_point_bits: Number of fractional bits for fixed-point approximation
        fold_batch_norm: Whether to fold BN into conv during QAT

    Returns:
        ModelQuantConfig with separate configurations for first, middle, and last layers
        
    Example:
        >>> # 8-bit for most layers, but 6-bit for first layer activations and 4-bit weights elsewhere
        >>> config = create_model_config_with_edge_layers(
        ...     default_weight_bits=4,
        ...     default_activation_bits=8,
        ...     first_layer_activation_bits=6,
        ...     last_layer_weight_bits=8,
        ... )
    """
    if scale_approximation_start_epoch < 1:
        raise ValueError("scale_approximation_start_epoch must be >= 1.")

    initial_scale_approximation = (
        scale_approximation
        if scale_approximation == ScaleApproximation.NONE or scale_approximation_start_epoch == 1
        else ScaleApproximation.NONE
    )

    # Create default config for middle layers
    default_config = mixed_precision_config(
        weight_bits=default_weight_bits,
        activation_bits=default_activation_bits,
        bias_bits=bias_bits,
        affine_activations=affine_activations,
        per_channel_weights=per_channel_weights,
        scale_approximation=initial_scale_approximation,
        fixed_point_bits=fixed_point_bits,
    )
    
    # Create first layer config if any edge parameter is specified, or when
    # scale approximation is enabled so the dataset/input scale can stay exact.
    first_layer_config = None
    if (
        first_layer_weight_bits is not None
        or first_layer_activation_bits is not None
        or scale_approximation != ScaleApproximation.NONE
    ):
        first_weight_bits = (
            first_layer_weight_bits if first_layer_weight_bits is not None else default_weight_bits
        )
        first_layer_config = mixed_precision_config(
            weight_bits=first_weight_bits,
            activation_bits=first_layer_activation_bits if first_layer_activation_bits is not None else default_activation_bits,
            bias_bits=bias_bits,
            affine_activations=affine_activations,
            per_channel_weights=per_channel_weights,
            scale_approximation=initial_scale_approximation,
            fixed_point_bits=fixed_point_bits,
        )
        _set_quantizer_spec_scale_approximation(
            first_layer_config.activation_quantizer,
            ScaleApproximation.NONE,
        )
    
    # Create last layer config if any edge parameter is specified, or when
    # scale approximation is enabled so the final output scale can stay exact.
    last_layer_config = None
    if (
        last_layer_weight_bits is not None
        or last_layer_activation_bits is not None
        or scale_approximation != ScaleApproximation.NONE
    ):
        last_weight_bits = (
            last_layer_weight_bits if last_layer_weight_bits is not None else default_weight_bits
        )
        last_layer_config = mixed_precision_config(
            weight_bits=last_weight_bits,
            activation_bits=last_layer_activation_bits if last_layer_activation_bits is not None else default_activation_bits,
            bias_bits=bias_bits,
            affine_activations=affine_activations,
            per_channel_weights=per_channel_weights,
            scale_approximation=initial_scale_approximation,
            fixed_point_bits=fixed_point_bits,
        )
        _set_quantizer_spec_scale_approximation(
            last_layer_config.weight_quantizer,
            ScaleApproximation.NONE,
        )
    
    return ModelQuantConfig(
        default_config=default_config,
        first_layer_config=first_layer_config,
        last_layer_config=last_layer_config,
        scale_approximation=initial_scale_approximation,
        fixed_point_bits=fixed_point_bits,
        fold_batch_norm=fold_batch_norm,
        target_scale_approximation=scale_approximation,
        scale_approximation_start_epoch=scale_approximation_start_epoch,
        freeze_approximated_scales=freeze_approximated_scales,
    )


def _set_quantizer_spec_scale_approximation(
    spec: Optional[QuantizerSpec],
    scale_approximation: ScaleApproximation,
) -> None:
    if spec is not None and "scale_approximation" in spec.kwargs:
        spec.kwargs["scale_approximation"] = scale_approximation
