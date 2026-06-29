from typing import Tuple, Optional

import torchvision
from torchvision import transforms, datasets
from .dataloader import DataLoader


class ImageNet(DataLoader):
    """
    ImageNet DataLoader

    A DataLoader subclass for the ImageNet (ILSVRC2012) dataset, which consists of
    approximately 1.2 million training images and 50,000 validation images
    across 1,000 object classes.

    References
    ----------
    For more information, see:
    https://www.image-net.org/challenges/LSVRC/2012/
    """
    image_size = (224, 224)

    def __init__(
        self,
        batch_size: int,
        crop_size: int | None = None,
        padding: int = 0,
        train_transform: Optional[torchvision.transforms.Compose] = None,
        valid_transform: Optional[torchvision.transforms.Compose] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
    ):
        super().__init__(
            batch_size,
            padding,
            train_transform,
            valid_transform,
            num_workers,
            persistent_workers,
            pin_memory,
            prefetch_factor,
        )
        if crop_size is None:
            self._crop_size = (224, 224)
        else:
            # Accept int or tuple; ensure both dimensions > 0
            if isinstance(crop_size, int):
                self._crop_size = (crop_size, crop_size)
            else:
                self._crop_size = tuple(crop_size)
            if self._crop_size[0] <= 0 or self._crop_size[1] <= 0:
                raise ValueError(f"crop_size must be > 0, got {self._crop_size}")

    def get_transforms(self):
        # Standard ImageNet-1k normalization used by torchvision reference training scripts.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        target_size = self._crop_size
        # Keep the common 256 -> 224 validation scale ratio when crop_size changes.
        val_resize_size = int(round((256 / 224) * min(target_size)))

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=target_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.Resize(val_resize_size, antialias=True),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_transform, valid_transform

    def load_datasets(self, train_transform, valid_transform, download: bool = True):
        train_dataset = datasets.ImageNet(
            root="/srv/storage/emeraude@storage1.lyon.grid5000.fr/bbarbe/imagenet",
            split="train",
            transform=train_transform,
        )
        valid_dataset = datasets.ImageNet(
            root="/srv/storage/emeraude@storage1.lyon.grid5000.fr/bbarbe/imagenet",
            split="val",
            transform=valid_transform,
        )
        return train_dataset, valid_dataset
