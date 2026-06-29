"""Callback that logs model metrics to TensorBoard during training."""

from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler

from hatorch.models import Model
from hatorch.train.trainer_state import TrainerState
from hatorch.train.trainer_control import TrainerControl
from hatorch.train.training_args import TrainingArguments
from hatorch.train.utils import LoggingStrategy

from .trainer_callback import TrainerCallback


class LogMetricsCallback(TrainerCallback):
    """
    Logs scalar metrics to TensorBoard during training.

    Metrics are logged either per step or per epoch depending on the `logging_strategy`
    specified in the training arguments. The appropriate step or epoch value is used as
    the x-axis when writing scalar values with the SummaryWriter.
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
        step = 0
        if args.logging_strategy == LoggingStrategy.STEPS:
            step = state.global_step
        else:
            step = state.epoch

        for key, value in last_metrics.items():
            summary_writer.add_scalar(key, value, step)
