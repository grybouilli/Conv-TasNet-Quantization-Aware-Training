"""Callback for saving optimizer parameters to JSON at the start of training."""

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


OPTIMIZER_PARAM_FILENAME = "optimizer_param.json"


class SaveOptimizerCallback(TrainerCallback):
    """
    Saves the optimizer’s class name and parameter groups to
    `<output_dir>/optimizer_param.json` when training begins.
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
        optimizer_save_path = os.path.join(args.output_dir, OPTIMIZER_PARAM_FILENAME)
        optimizer_state_dict = optimizer.state_dict()
        optimizer_param = {
            "optimizer_name": optimizer.__class__.__name__,
            "param_groups": optimizer_state_dict["param_groups"],
        }
        with open(optimizer_save_path, "w", encoding="utf-8") as f:
            json.dump(optimizer_param, f, indent=4)
