from typing import Optional

import torch.nn as nn

from hatorch.datasets import DatasetName
from .model import Model
import torch


class _Lenet(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # Convolution
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=True
        )
        self.relu_conv1 = nn.ReLU()
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu_conv2 = nn.ReLU()
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)

        # Fully connected layer
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = torch.nn.Linear(120, 84)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.max_pool1(self.relu_conv1(self.conv1(x)))
        x = self.max_pool2(self.relu_conv2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        logits = self.fc3(x)
        return logits


class Lenet(Model):
    name = "Lenet"
    compatible_datasets = [DatasetName.MNIST]
    default_dataset = DatasetName.MNIST

    def get_model(self, pretrained: bool) -> torch.nn.Module:
        return _Lenet()
