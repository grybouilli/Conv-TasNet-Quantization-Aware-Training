from pytorchcv.model_provider import get_model as ptcv_get_model

from hatorch.datasets import DatasetName
from .model import Model
import torch

class ResNet20(Model):
    name = "ResNet20"
    compatible_datasets = [DatasetName.CIFAR10, DatasetName.CIFAR100]
    default_dataset = DatasetName.CIFAR10

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        model_name = "resnet20_cifar100" if self._dataset_name == DatasetName.CIFAR100 else "resnet20_cifar10"
        return ptcv_get_model(model_name, pretrained=pretrained)
