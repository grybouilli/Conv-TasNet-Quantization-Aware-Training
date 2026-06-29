# test_models_init.py
import pytest

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
def test_model_initialization(ModelClass):
    # Ensure each model can be constructed without errors
    model = ModelClass()
    assert model is not None
