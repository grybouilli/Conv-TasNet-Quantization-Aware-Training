import os
import torch
import pytest

from hatorch.models import Squeezenet

DEVICE = torch.device("cpu")

SQUEEZE_NET_CUSTOM_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "squeezenet_trained_cifar10.pth"
)


def test_custom_weights_loading():
    # Default (scratch) vs custom-loaded weights
    squeezenet_scratch = Squeezenet(pretrained=False)
    squeezenet_trained = Squeezenet(pretrained=False)
    squeezenet_trained.load_custom_weights(SQUEEZE_NET_CUSTOM_WEIGHTS_PATH, DEVICE)
    # Compare parameters
    for p_s, p_c in zip(
        squeezenet_scratch.model.parameters(), squeezenet_trained.model.parameters()
    ):
        assert not torch.allclose(p_s.data, p_c.data)
