import torchvision.models

from hatorch.datasets import DatasetName
from .model import Model
import torch

class AlexNet(Model):
    name = "AlexNet"
    compatible_datasets = [DatasetName.IMAGENET1K]
    default_dataset = DatasetName.IMAGENET1K

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        if pretrained:
            return torch.hub.load('pytorch/vision', 'alexnet', weights=torchvision.models.AlexNet_Weights.DEFAULT)
        return torchvision.models.alexnet()
