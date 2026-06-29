"""Callback that logs gradient histograms to TensorBoard."""

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


class LogGradCallback(TrainerCallback):
    """
    Logs gradient histograms of model parameters to TensorBoard during training.

    Histograms are tagged by module and parameter name using a slash-separated format
    for hierarchical visualization in TensorBoard.

    Example tag: "conv1/bias/grad"
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

        for module_name, module in model.model.named_modules():
            module_name = module_name.replace(".", "/")
            for param_name, param in module.named_parameters(
                recurse=False
            ):  # local params only
                if param.grad is not None:
                    tag = f"{module_name}/{param_name}/grad"
                    summary_writer.add_histogram(tag, param.grad, step)
