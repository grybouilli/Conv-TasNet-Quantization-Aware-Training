import torchvision.models

from hatorch.datasets import DatasetName
from .model import Model
import torch

class MobileNet(Model):
    name = "MobileNet"
    compatible_datasets = [DatasetName.IMAGENET1K]
    default_dataset = DatasetName.IMAGENET1K

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        if pretrained:
            return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        return torchvision.models.mobilenet_v2()
