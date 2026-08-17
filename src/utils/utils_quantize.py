from typing import TypedDict

import torch
import torch.nn as nn
from hatorch.quantizers import ScaleApproximation
from hatorch.transforms import utils as transforms_utils
from hatorch.transforms.config import (
    LayerQuantConfig,
    ModelQuantConfig,
    mixed_precision_config,
)

from model.norm_quantized import QuantGlobalLayerNorm


def generate_layers_recursively(
    module: nn.Module, prefix: str, layers_name: list[str]
) -> None:
    """
    model: the non quantized model
    prefix: the prefix of the current module (empty for the root module)
    layers_name: the list to fill with the names of the layers to quantize
    """

    children_list = list(module.named_children())

    for i, (name, child) in enumerate(children_list):
        full_name = f"{prefix}.{name}" if prefix else name
        layers_name.append(full_name)

        # Always recursively process child modules (for nested structures)
        if hasattr(child, "children") and len(list(child.children())) > 0:
            generate_layers_recursively(child, full_name, layers_name)


def generate_quantization_config(
    model: nn.Module,
    encoder_config: LayerQuantConfig | None = None,
    tcn_config: LayerQuantConfig | None = None,
    decoder_config: LayerQuantConfig | None = None,
) -> dict[str, LayerQuantConfig]:

    layers = []

    # The first prefix is "module" because the model will be wrapped in a `QuantizeWrapper`
    generate_layers_recursively(model, prefix="module", layers_name=layers)

    config: dict[str, LayerQuantConfig] = {}
    for layer in layers:
        if "encoder." in layer:
            config[layer] = encoder_config

        elif "tcn." in layer:
            config[layer] = tcn_config
            # For 4 bits config, we encode the tcn layer_norm with the encoder config, and the rest of the tcn with the tcn config
            if tcn_config is not None and (
                tcn_config.weight_quantizer.kwargs.get("bit_width") == 4
                or tcn_config.activation_quantizer.kwargs.get("bit_width") == 4
            ):
                if any(
                    substring in layer
                    for substring in [
                        "module.tcn.layer_norm",
                        "module.tcn.bottleneck_layer",
                    ]
                ):  # , "module.tcn.bottleneck_layer", "module.tcn.non_linearity_out", "module.tcn.conv1d_out", "module.tcn.mask_activation"]):
                    config[layer] = encoder_config

            # if any(substring in layer for substring in ["module.tcn.bottleneck_layer", "module.tcn.non_linearity_out", "module.tcn.conv1d_out",  "module.tcn.mask_activation"]):
            #     config[layer] = tcn_layers_config
            # if "module.tcn.layers." in layer:
            #     # index = layer.replace("module.tcn.layers.", "").split(".")[0]
            #     # index = int(index)
            #     #if index > 10:
            #     config[layer] =

        elif "decoder." in layer:
            config[layer] = decoder_config

    for i in range(148):
        config[f"quant_add_{i}"] = encoder_config
    for i in range(51):
        config[f"quant_mul_{i}"] = encoder_config

        # print(f"{layer}: {config.get(layer, None)}")

    config = {k: v for k, v in config.items() if v is not None}

    return config


class QuantizationConfig(TypedDict):
    encoder_config: LayerQuantConfig | None
    tcn_config: LayerQuantConfig | None
    decoder_config: LayerQuantConfig | None


def quantization_configs() -> dict[str, QuantizationConfig]:

    config_S16S16 = mixed_precision_config(
        quantize_activations=True,
        quantize_weights=True,
        quantize_output=True,
        weight_bits=16,
        activation_bits=16,
        output_bits=16,
        quantize_bias=False,
        bias_bits=32,
        affine_activations=True,
        affine_output=False,
        per_channel_weights=True,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=True,
        signed_activations=True,
        signed_output=True,
    )

    config_U16S16 = mixed_precision_config(
        quantize_activations=True,
        quantize_weights=True,
        quantize_output=True,
        weight_bits=16,
        activation_bits=16,
        output_bits=16,
        quantize_bias=False,
        bias_bits=32,
        affine_activations=True,
        affine_output=True,
        per_channel_weights=True,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=True,
        signed_activations=False,
        signed_output=False,
    )

    config_U16U16 = mixed_precision_config(
        quantize_activations=True,
        quantize_weights=True,
        quantize_output=True,
        weight_bits=16,
        activation_bits=16,
        output_bits=16,
        quantize_bias=False,
        bias_bits=32,
        affine_activations=True,
        affine_output=True,
        per_channel_weights=True,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=False,
        signed_activations=False,
        signed_output=False,
    )

    config_S8S8 = mixed_precision_config(
        quantize_activations=True,
        quantize_weights=True,
        quantize_output=True,
        weight_bits=8,
        activation_bits=8,
        output_bits=8,
        quantize_bias=False,
        bias_bits=32,
        affine_activations=True,
        affine_output=True,
        per_channel_weights=True,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=True,
        signed_activations=True,
        signed_output=True,
    )

    config_S4S4 = mixed_precision_config(
        quantize_activations=True,
        quantize_weights=True,
        quantize_output=True,
        weight_bits=4,
        activation_bits=4,
        output_bits=4,
        quantize_bias=False,
        bias_bits=32,
        affine_activations=True,
        affine_output=True,
        per_channel_weights=True,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=True,
        signed_activations=True,
        signed_output=True,
    )

    config_S16S8 = mixed_precision_config(
        quantize_activations=True,
        quantize_weights=True,
        quantize_output=True,
        weight_bits=8,
        activation_bits=16,
        output_bits=16,
        quantize_bias=False,
        bias_bits=32,
        affine_activations=True,
        affine_output=True,
        per_channel_weights=True,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=True,
        signed_activations=True,
        signed_output=True,
    )

    configs: dict[str, QuantizationConfig] = {
        "all_S16S16": {
            "encoder_config": config_S16S16,
            "tcn_config": config_S16S16,
            "decoder_config": config_S16S16,
        },
        "all_U16S16": {
            "encoder_config": config_U16S16,
            "tcn_config": config_U16S16,
            "decoder_config": config_U16S16,
        },
        "all_U16U16": {
            "encoder_config": config_U16U16,
            "tcn_config": config_U16U16,
            "decoder_config": config_U16U16,
        },
        "all_S8S8": {
            "encoder_config": config_S8S8,
            "tcn_config": config_S8S8,
            "decoder_config": config_S8S8,
        },
        "all_S4S4": {
            "encoder_config": config_S4S4,
            "tcn_config": config_S4S4,
            "decoder_config": config_S4S4,
        },
        "encdec_S16S16_tcn_S4S4": {
            "encoder_config": config_S16S16,
            "tcn_config": config_S4S4,
            "decoder_config": config_S16S16,
        },
        "encdec_U16S16_tcn_S4S4": {
            "encoder_config": config_U16S16,
            "tcn_config": config_S4S4,
            "decoder_config": config_U16S16,
        },
        "encdec_S16S16_tcn_S8S8": {
            "encoder_config": config_S16S16,
            "tcn_config": config_S8S8,
            "decoder_config": config_S16S16,
        },
        "all_S16S8": {
            "encoder_config": config_S16S8,
            "tcn_config": config_S16S8,
            "decoder_config": config_S16S8,
        },
    }

    return configs


def validate_quantization_config_names(
    config_names: list[str], accept_full_precision: bool = False
) -> None:
    configs = quantization_configs()
    valid_names = list(configs.keys())
    if accept_full_precision:
        valid_names.append("full_precision")
    for name in config_names:
        if name not in valid_names:
            raise ValueError(
                f"Invalid quantization config name: {name}. Valid names are: {valid_names}"
            )


def quantize_model(
    model: nn.Module,
    config_name: str,
    example_args: tuple[torch.Tensor],
) -> nn.Module:

    identity_layer_config = LayerQuantConfig()

    configs = quantization_configs()

    config = configs[config_name]

    layer_configs = generate_quantization_config(model, **config)

    model_config = ModelQuantConfig(
        default_config=identity_layer_config,
        layer_configs=layer_configs,
        scale_approximation=ScaleApproximation.FIXED_POINT,
    )

    quantized_model = transforms_utils.quantize_model(
        model,
        model_config,
        example_args=example_args,
        use_wrapper=True,
        replace_quantizable_operators_using_tracing=False,
        additional_hatorch_quantized_layers_class=[QuantGlobalLayerNorm],
    )
    return quantized_model
