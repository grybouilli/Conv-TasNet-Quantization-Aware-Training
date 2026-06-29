"""Base trainer callback interface.

Inspired by Hugging Face Transformers’ TrainerCallback
(https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py),
licensed under Apache License 2.0.
"""

from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler

from hatorch.models import Model
from hatorch.train.trainer_state import TrainerState
from hatorch.train.trainer_control import TrainerControl
from hatorch.train.training_args import TrainingArguments


class TrainerCallback:
    """
    Callback base class for Trainer events.
    """

    def on_init_end(
        self,
        model: Model,
        state: TrainerState,
        control: TrainerControl,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
    ):
        """
        Event called at the end of the Trainer initialization.
        """

    def on_train_begin(
        self,
        model: Model,
        state: TrainerState,
        control: TrainerControl,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
    ):
        """
        Event called at the beginning of training.
        """

    def on_train_end(
        self,
        model: Model,
        state: TrainerState,
        control: TrainerControl,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
    ):
        """
        Event called at the end of training.
        """

    def on_epoch_begin(
        self,
        model: Model,
        state: TrainerState,
        control: TrainerControl,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
    ):
        """
        Event called at the beginning of an epoch.
        """

    def on_epoch_end(
        self,
        model: Model,
        state: TrainerState,
        control: TrainerControl,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
    ):
        """
        Event called at the end of an epoch.
        """

    def on_step_end(
        self,
        model: Model,
        state: TrainerState,
        control: TrainerControl,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
        step_loss: float,
    ):
        """
        Event called after a training optimizer step.
        """

    def on_log(
        self,
        model: Model,
        state: TrainerState,
        control: TrainerControl,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
        # Specific to `on_log`:
        last_metrics: Dict[str, float],
        summary_writer: SummaryWriter,
    ):
        """
        Event called after logging metrics.
        """

    def on_evaluate(
        self,
        model: Model,
        state: TrainerState,
        control: TrainerControl,
        args: TrainingArguments,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[LRScheduler],
    ):
        """
        Event called after an evaluation phase.
        """
