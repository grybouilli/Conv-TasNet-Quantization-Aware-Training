from pytorchcv.model_provider import get_model as ptcv_get_model

from hatorch.datasets import DatasetName
from .model import Model
import torch

class ResNet18(Model):
    name = "ResNet18"
    compatible_datasets = [DatasetName.IMAGENET1K]
    default_dataset = DatasetName.IMAGENET1K

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        return ptcv_get_model("resnet18", pretrained=pretrained)
