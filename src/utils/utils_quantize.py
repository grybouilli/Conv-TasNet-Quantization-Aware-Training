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

        elif "decoder." in layer:
            config[layer] = decoder_config

        print(f"{layer}: {config.get(layer, None)}")

    config = {k: v for k, v in config.items() if v is not None}

    return config


def quantize_model(
    model: nn.Module,
    config_name: str,
) -> nn.Module:

    identity_layer_config = LayerQuantConfig()

    config_S16S16 = mixed_precision_config(
        quantize_activations=True,
        quantize_weights=True,
        quantize_output=True,
        weight_bits=16,
        activation_bits=16,
        output_bits=16,
        quantize_bias=False,
        bias_bits=32,
        affine_activations=False,
        affine_output=False,
        per_channel_weights=True,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=True,
        signed_activations=True,
        signed_output=True,
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
        affine_activations=False,
        affine_output=False,
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
        affine_activations=False,
        affine_output=False,
        per_channel_weights=True,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=True,
        signed_activations=True,
        signed_output=True,
    )

    configs = {
        "all_S16S16": {
            "encoder_config": config_S16S16,
            "tcn_config": config_S16S16,
            "decoder_config": config_S16S16,
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
        "encdec_S16S16_tcn_S8S8": {
            "encoder_config": config_S16S16,
            "tcn_config": config_S8S8,
            "decoder_config": config_S16S16,
        },
    }

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
        use_wrapper=True,
        additional_hatorch_quantized_layers_class=[QuantGlobalLayerNorm],
    )
    return quantized_model
