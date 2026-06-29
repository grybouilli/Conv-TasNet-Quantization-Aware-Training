import torchvision.models

from hatorch.datasets import DatasetName
from .model import Model
import torch

class Squeezenet(Model):
    name = "Squeezenet"
    compatible_datasets = [DatasetName.IMAGENET1K]
    default_dataset = DatasetName.IMAGENET1K

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        if pretrained:
            return torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT)
        return torchvision.models.squeezenet1_1()
