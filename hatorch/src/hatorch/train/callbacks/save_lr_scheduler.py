"""Callback for saving the learning-rate scheduler’s parameters to JSON at the start of training."""

from typing import Optional
import json
import os

import torch
from torch.optim.lr_scheduler import LRScheduler

from hatorch.models import Model
from hatorch.train.trainer_state import TrainerState
from hatorch.train.trainer_control import TrainerControl
from hatorch.train.training_args import TrainingArguments

from .trainer_callback import TrainerCallback


LR_SCHEDULER_PARAM_FILENAME = "lr_scheduler_param.json"


class SaveLRSchedulerCallback(TrainerCallback):
    """
    Saves the LR scheduler’s configuration and state_dict to
    `<output_dir>/lr_scheduler_param.json` when training begins.

    Use JSON with UTF-8 encoding.
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
        if lr_scheduler is not None:
            lr_scheduler_save_path = os.path.join(
                args.output_dir, LR_SCHEDULER_PARAM_FILENAME
            )
            lr_scheduler_state_dict = lr_scheduler.state_dict()
            lr_scheduler_dict = {
                "name": lr_scheduler.__class__.__name__,
                "state_dict": lr_scheduler_state_dict,
            }
            with open(lr_scheduler_save_path, "w", encoding="utf-8") as f:
                json.dump(lr_scheduler_dict, f, indent=4)
