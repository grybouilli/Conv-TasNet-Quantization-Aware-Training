"""
Adapted from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
Author: Cheng-Yang Fu
"""

import math
import os
from pathlib import Path
from typing import Optional, Literal

import torch
import torch.nn as nn
import torchvision

from hatorch.datasets import DatasetName
from .model import Model
from hatorch.utils import logger

__all__ = [
    'VGG'
]

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

class _VGG(nn.Module):
    """
    VGG model
    """
    def __init__(self, features, num_classes=10):
        super(_VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        # Flatten to (batch, features) before the classifier
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(selected_cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in selected_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(Model):
    name = "VGG"
    compatible_datasets = [DatasetName.CIFAR10, DatasetName.CIFAR100]
    default_dataset = DatasetName.CIFAR10

    def __init__(
        self,
        batch_norm: bool = False,
        nb_layers: Literal[11, 13, 16, 19] = 11,
        dataset_name: Optional[DatasetName] = None,
        batch_size: Optional[int] = 32,
        padding: Optional[int] = 0,
        pretrained: Optional[bool] = False,
        train_transform: Optional[torchvision.transforms.Compose] = None,
        valid_transform: Optional[torchvision.transforms.Compose] = None,
    ):
        self._batch_norm = batch_norm
        self._nb_layers = nb_layers
        super().__init__(
            dataset_name, batch_size, padding, pretrained, train_transform, valid_transform
        )

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        # Determine number of classes based on dataset
        if self._dataset_name == DatasetName.CIFAR100:
            num_classes = 100
        elif self._dataset_name == DatasetName.CIFAR10:
            num_classes = 10
        else:
            # Default to 10 classes for backward compatibility
            num_classes = 10
            
        if self._nb_layers == 11:
            if self._batch_norm:
                model = _VGG(make_layers(cfg['A'], batch_norm=True), num_classes)
            else:
                model = _VGG(make_layers(cfg['A'], batch_norm=False), num_classes)
        elif self._nb_layers == 13:
            if self._batch_norm:
                model = _VGG(make_layers(cfg['B'], batch_norm=True), num_classes)
            else:
                model = _VGG(make_layers(cfg['B'], batch_norm=False), num_classes)
        elif self._nb_layers == 16:
            if self._batch_norm:
                model = _VGG(make_layers(cfg['D'], batch_norm=True), num_classes)
            else:
                model = _VGG(make_layers(cfg['D'], batch_norm=False), num_classes)
        elif self._nb_layers == 19:
            if self._batch_norm:
                model = _VGG(make_layers(cfg['E'], batch_norm=True), num_classes)
            else:
                model = _VGG(make_layers(cfg['E'], batch_norm=False), num_classes)
        else:
            raise ValueError("nb_layers must be one of [11, 13, 16, 19]")

        if pretrained:
            if not self._nb_layers == 11 or not self._batch_norm:
                logger.warning("Pretrained weights are only available for VGG11 with batch normalization for now.")
            weights_dir = Path(__file__).parent.parent / "weights" / "vgg"
            checkpoint_path = weights_dir / (
                "vggbn11_cifar100.wgt" if self._dataset_name == DatasetName.CIFAR100 else "vggbn11_cifar10.wgt")
            print(checkpoint_path)
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint)  # Direct load since you save state_dict directly
                logger.info(f"Loaded pretrained weights from {checkpoint_path}")
            else:
                logger.warning(f"No pretrained weights found at {checkpoint_path}")

        return model
