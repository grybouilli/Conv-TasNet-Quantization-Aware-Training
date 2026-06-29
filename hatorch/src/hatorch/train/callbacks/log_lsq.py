"""Callback for logging learned LSQ quantizer scales to TensorBoard."""

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


class LogLSQScaleCallback(TrainerCallback):
    """
    Logs learned LSQ scale parameters (weight, bias, activation, output) to TensorBoard.
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

        for layer_name, layer in model.model.named_modules():
            layer_name = layer_name.replace(".", "/")
            for quant_name in [
                "quant_weight",
                "quant_bias",
                "quant_activation",
                "quant_output",
            ]:
                param_name = f"{quant_name}_scale"
                summary_write_key = layer_name + "/" + param_name
                if param_name in layer._parameters:
                    scale_param = layer._parameters[param_name]
                    summary_writer.add_scalar(summary_write_key, scale_param, step)
