from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import torch
import torchvision

from hatorch.datasets import DatasetName, get_dataset_from_name
from hatorch.utils import logger


class Model(ABC):
    name = "Model"

    @property
    @abstractmethod
    def compatible_datasets(self) -> List[DatasetName]:
        pass

    @property
    @abstractmethod
    def default_dataset(self) -> List[DatasetName]:
        pass

    def __init__(
        self,
        gpu_id: int,
        dataset_name: Optional[DatasetName] = None,
        batch_size: Optional[int] = 32,
        padding: Optional[int] = 0,
        pretrained: Optional[bool] = False,
        train_transform: Optional[torchvision.transforms.Compose] = None,
        valid_transform: Optional[torchvision.transforms.Compose] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
    ):
        self._pretrained = pretrained
        self._dataset_name = dataset_name
        self.is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.model = self._load_model_ddp_safe(pretrained)

        if dataset_name is None:
            dataset_name = self.default_dataset

        if dataset_name not in self.compatible_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not compatible with {self.name} model. "
                f"Compatible datasets: {self.compatible_datasets}"
            )

        self.dataloader = get_dataset_from_name(dataset_name,
                                                batch_size=batch_size,
                                                padding=padding,
                                                num_workers=num_workers,
                                                persistent_workers=persistent_workers,
                                                pin_memory=pin_memory,
                                                prefetch_factor=prefetch_factor,
                                                train_transform=train_transform,
                                                valid_transform=valid_transform)

        self.log_model_info()

    def _is_main_process(self) -> bool:
        if not self.is_distributed:
            return True
        return torch.distributed.get_rank() == 0

    def _load_model_ddp_safe(self, pretrained: bool) -> torch.nn.Module:
        # Avoid concurrent pretrained weight downloads in DDP.
        if not (self.is_distributed and pretrained):
            return self.get_model(pretrained)

        if self._is_main_process():
            model = self.get_model(pretrained)
            torch.distributed.barrier()
            return model

        torch.distributed.barrier()
        return self.get_model(pretrained)

    @abstractmethod
    def get_model(self, pretrained: bool) -> torch.nn.Module:
        pass

    def to_device(self, device: torch.device):
        if self._is_main_process():
            logger.info(f"Model moved to {device}")
        self.model.to(device)

    def log_model_info(self) -> None:
        if not self._is_main_process():
            return
        pretrained_part = "pretrained" if self._pretrained else "uninitialized"
        logger.info(f"Loaded {self.name} model with {pretrained_part} weights")

    def load_custom_weights(
        self, custom_weights_uri: str, device: torch.device
    ) -> None:
        logger.info(f"Loading custom weights from: {custom_weights_uri}")
        state_dict: Dict[str, torch.Tensor] = torch.load(
            custom_weights_uri, map_location=device
        )
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        if (
            isinstance(state_dict, dict)
            and state_dict
            and all(isinstance(key, str) and key.startswith("module.") for key in state_dict.keys())
        ):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        model_to_load.load_state_dict(state_dict)
