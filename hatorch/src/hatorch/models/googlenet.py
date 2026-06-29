import torchvision.models

from hatorch.datasets import DatasetName
from .model import Model
import torch

class GoogLeNet(Model):
    name = "GoogLeNet"
    compatible_datasets = [DatasetName.IMAGENET1K]
    default_dataset = DatasetName.IMAGENET1K

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        if pretrained:
            return torch.hub.load('pytorch/vision', 'googlenet', weights=torchvision.models.GoogLeNet_Weights.DEFAULT)
        return torchvision.models.googlenet()
