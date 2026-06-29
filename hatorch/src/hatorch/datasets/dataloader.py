from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import numpy as np
import torchvision
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler


class DataLoader(ABC):
    def __init__(
        self,
        batch_size: int,
        padding: int = 0,
        train_transform: Optional[torchvision.transforms.Compose] = None,
        valid_transform: Optional[torchvision.transforms.Compose] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.padding = padding
        self._custom_train_transform = train_transform
        self._custom_valid_transform = valid_transform
        self._num_workers = 0
        self._persistent_workers = False
        self._pin_memory = False
        self._prefetch_factor: Optional[int] = None
        self.set_dataloader_workers(
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def persistent_workers(self) -> bool:
        return self._persistent_workers

    @property
    def pin_memory(self) -> bool:
        return self._pin_memory

    @property
    def prefetch_factor(self) -> Optional[int]:
        return self._prefetch_factor

    def set_dataloader_workers(
        self,
        num_workers: int,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
    ) -> None:
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        if persistent_workers and num_workers == 0:
            raise ValueError(
                "persistent_workers=True requires num_workers > 0."
            )
        if prefetch_factor is not None:
            if prefetch_factor < 1:
                raise ValueError(
                    f"prefetch_factor must be >= 1, got {prefetch_factor}"
                )
            if num_workers == 0:
                raise ValueError("prefetch_factor requires num_workers > 0.")
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory
        self._prefetch_factor = prefetch_factor

    def _build_torch_loader(self, dataset, shuffle: bool = False, sampler=None):
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)

    @property
    @abstractmethod
    def image_size(self) -> Tuple[int, int] | None:
        pass

    @abstractmethod
    def get_transforms(
        self,
    ) -> [torchvision.transforms.Compose, torchvision.transforms.Compose]:
        pass

    @abstractmethod
    def load_datasets(
        self, train_transform, valid_transform, download: bool = True
    ) -> [torch.utils.data.Dataset, torch.utils.data.Dataset]:
        pass

    def get_train_valid_loader(
        self,
        train_transform: Optional[torchvision.transforms.Compose] = None,
        valid_transform: Optional[torchvision.transforms.Compose] = None,
    ):

        if train_transform is None:
            train_transform = self._custom_train_transform
        if valid_transform is None:
            valid_transform = self._custom_valid_transform

        if train_transform is None or valid_transform is None:
            default_train, default_valid = self.get_transforms()
            if train_transform is None:
                train_transform = default_train
            if valid_transform is None:
                valid_transform = default_valid

        is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if is_distributed:
            is_main = torch.distributed.get_rank() == 0
            if is_main:
                train_dataset, valid_dataset = self.load_datasets(
                    train_transform, valid_transform, download=True
                )
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                train_dataset, valid_dataset = self.load_datasets(
                    train_transform, valid_transform, download=False
                )
        else:
            train_dataset, valid_dataset = self.load_datasets(
                train_transform, valid_transform, download=True
            )

        if valid_dataset is None:
            num_train = len(train_dataset)
            indices = torch.randperm(num_train).tolist()
            split = int(np.floor(0.1 * num_train))
            train_subset = Subset(train_dataset, indices[split:])
            valid_subset = Subset(train_dataset, indices[:split])

            if is_distributed:
                train_sampler = DistributedSampler(train_subset, shuffle=True)
                valid_sampler = DistributedSampler(valid_subset, shuffle=False)
                train_loader = self._build_torch_loader(
                    train_subset, sampler=train_sampler
                )
                valid_loader = self._build_torch_loader(
                    valid_subset, sampler=valid_sampler
                )
            else:
                train_loader = self._build_torch_loader(train_subset, shuffle=True)
                valid_loader = self._build_torch_loader(valid_subset, shuffle=False)
        else:
            if is_distributed:
                train_loader = self._build_torch_loader(
                    train_dataset,
                    sampler=DistributedSampler(train_dataset, shuffle=True),
                )
                valid_loader = self._build_torch_loader(
                    valid_dataset,
                    sampler=DistributedSampler(valid_dataset, shuffle=False),
                )
            else:
                train_loader = self._build_torch_loader(train_dataset, shuffle=True)
                valid_loader = self._build_torch_loader(valid_dataset, shuffle=False)
        return train_loader, valid_loader
