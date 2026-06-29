"""Callback for saving the trainer’s arguments to JSON on initialization."""

from typing import Optional
import os

import torch
from torch.optim.lr_scheduler import LRScheduler

from hatorch.models import Model
from hatorch.train.trainer_state import TrainerState
from hatorch.train.trainer_control import TrainerControl
from hatorch.train.training_args import TrainingArguments

from .trainer_callback import TrainerCallback

TRAIN_ARGS_FILENAME = "training_args.json"


class SaveTrainingArgsCallback(TrainerCallback):
    """
    Writes the TrainingArguments as JSON to
    `<output_dir>/training_args.json` once the trainer is initialized.

    Use UTF-8 encoding.
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
        training_args_save_path = os.path.join(args.output_dir, TRAIN_ARGS_FILENAME)
        with open(training_args_save_path, "w", encoding="utf-8") as f:
            f.write(args.to_json_string())
