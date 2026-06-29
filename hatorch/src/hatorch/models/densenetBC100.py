from pytorchcv.model_provider import get_model as ptcv_get_model

from hatorch.datasets import DatasetName
from .model import Model
import torch


class DenseNetBC100(Model):
    name = "DenseNet-BC-100"
    compatible_datasets = [DatasetName.CIFAR10, DatasetName.CIFAR100]
    default_dataset = DatasetName.CIFAR100

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        model_name = "densenet40_k12_bc_cifar100" if self._dataset_name == DatasetName.CIFAR100 else "densenet40_k12_bc_cifar10"
        return ptcv_get_model(model_name, pretrained=pretrained)
