import pytest
import torch

from hatorch.models import (
    AlexNet,
    DenseNetBC100,
    MobileNet,
    Lenet,
    GoogLeNet,
    Squeezenet,
    ResNet18,
    ResNet20,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet56,
)
from hatorch.utils import ModelQuantizer, QuantizeConfig
from hatorch.quantizers import LsqUniformWeights
from hatorch.utils import QuantizedLinear, QuantizedConv2d

PRETRAINED = False
DEVICE = torch.device("cpu")

MODEL_CLASSES = [
    AlexNet,
    DenseNetBC100,
    MobileNet,
    Lenet,
    GoogLeNet,
    Squeezenet,
    ResNet18,
    ResNet20,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet56,
]


@pytest.mark.parametrize("ModelClass", MODEL_CLASSES)
def test_layer_replacement(ModelClass):
    # Instantiate model
    model_instance = ModelClass()
    # Collect original layers
    original_layers = [
        layer
        for layer in model_instance.model.modules()
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d))
    ]

    # Setup quantization config
    configs = [QuantizeConfig(weight=LsqUniformWeights(bits)) for bits in (2, 4, 6, 8)]
    quantizer = ModelQuantizer(model_instance, *configs)
    quantizer.quantize_model()

    # Collect quantized layers
    quantized_layers = [
        layer
        for layer in model_instance.model.modules()
        if isinstance(layer, (QuantizedLinear, QuantizedConv2d))
    ]

    # Ensure counts match
    assert len(quantized_layers) == len(original_layers)

    # Check quantization parameters at boundaries
    depth = 0
    for layer in quantized_layers:
        if depth == 0:
            assert layer._quant_weights_params["qn"] == -32
        elif depth == len(quantized_layers) - 1:
            assert layer._quant_weights_params["qn"] == -128
        else:
            if isinstance(layer, QuantizedLinear):
                assert layer._quant_weights_params["qn"] == -2
            else:
                assert layer._quant_weights_params["qn"] == -8
        depth += 1
