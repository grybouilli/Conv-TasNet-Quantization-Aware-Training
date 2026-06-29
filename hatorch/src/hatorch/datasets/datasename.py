# dataset_registry.py
from enum import Enum
from typing import Optional

import torchvision

from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .dataloader import DataLoader
from .imagenet import ImageNet
from .mnist import MNIST


class DatasetName(str, Enum):
    CIFAR100 = "cifar100"
    CIFAR10 = "cifar10"
    MNIST = "mnist"
    IMAGENET1K = "imagenet1k"

def get_dataset_from_name(
    dataset_name: DatasetName,
    batch_size: int,
    padding: int = 0,
    train_transform: Optional[torchvision.transforms.Compose] = None,
    valid_transform: Optional[torchvision.transforms.Compose] = None,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    # Switch case to return the appropriate DataLoader instance
    if dataset_name == DatasetName.CIFAR100:
        return CIFAR100(
            batch_size,
            padding,
            train_transform,
            valid_transform,
            num_workers,
            persistent_workers,
            pin_memory,
            prefetch_factor,
        )
    elif dataset_name == DatasetName.CIFAR10:
        return CIFAR10(
            batch_size,
            padding,
            train_transform,
            valid_transform,
            num_workers,
            persistent_workers,
            pin_memory,
            prefetch_factor,
        )
    elif dataset_name == DatasetName.MNIST:
        return MNIST(
            batch_size,
            padding,
            train_transform,
            valid_transform,
            num_workers,
            persistent_workers,
            pin_memory,
            prefetch_factor,
        )
    elif dataset_name == DatasetName.IMAGENET1K:
        return ImageNet(
            batch_size=batch_size,
            crop_size=None,
            padding=padding,
            train_transform=train_transform,
            valid_transform=valid_transform,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
