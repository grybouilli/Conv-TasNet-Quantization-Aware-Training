"""
Callback that logs model parameter histograms to TensorBoard.
Include support for quantized parameters.
"""

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler

from hatorch.models import Model
from hatorch.train.trainer_state import TrainerState
from hatorch.train.trainer_control import TrainerControl
from hatorch.train.training_args import TrainingArguments
from hatorch.train.utils import LoggingStrategy

from .trainer_callback import TrainerCallback


class LogParametersHistogramCallback(TrainerCallback):
    """
    Logs weight and bias histograms (both float and quantized) to TensorBoard.

    For each module in the model:
    - If the quantization function attribute (e.g. `Qw` or `Qb`) is set, it calls
      the corresponding method (`qweight()` or `qbias()`) to get the discrete values
      and plots them as a bar chart.
    - Otherwise, it logs the raw tensor histogram via TensorBoard's `add_histogram`.
    """

    # map each float param to its quantized param getter and the quantization function
    _QUANT_PARAM_MAP = {
        "weight": ("_quantized_weight", "_weight_quantizer"),
        "bias": ("_quantized_bias", "_bias_quantizer"),
    }

    def _log_param(
        self,
        module: torch.nn.Module,
        writer: SummaryWriter,
        step: int,
        module_name: str,
        param_name: str,
    ):
        """
        Log either a quantized or float histogram for a single parameter.

        Args:
            layer_name: slash-separated module name for TensorBoard tagging.
            param_name: either "weight" or "bias".
            _QUANT_PARAM_MAP[param_name] gives:
              - quant_method: method to call for quantized tensor (qweight/qbias)
              - quant_fn_attr: attribute holding the quantization function (Qw/Qb)
        """
        quant_method, quant_fn_attr = self._QUANT_PARAM_MAP[param_name]
        tag = f"{module_name}/{param_name}"

        # Quantized path: only if the quantization function is active
        if (
            hasattr(module, quant_method)
            and getattr(module, quant_fn_attr, None) is not None
        ):
            qvals: Tensor = getattr(module, quant_method)()
            self._add_quantized_histogram(writer, tag, qvals, step)

        # Float path: fallback to the raw tensor
        elif hasattr(module, param_name):
            tensor = getattr(module, param_name)
            if tensor is not None:
                writer.add_histogram(
                    tag, tensor.detach().cpu().numpy(), global_step=step
                )

    def _add_quantized_histogram(
        self,
        writer: SummaryWriter,
        tag: str,
        qparam: Tensor,
        step: int,
    ):
        """
        Plot a bar chart of unique quantized levels vs. their counts,
        then add it as a TensorBoard figure.
        """
        arr = qparam.detach().cpu().numpy().ravel()
        levels, counts = np.unique(arr, return_counts=True)
        gaps = np.diff(levels)
        min_gap = float(np.min(gaps)) if gaps.size else 1.0

        fig, ax = plt.subplots()
        ax.bar(levels, counts, width=min_gap * 0.5)
        ax.set_xlabel("Quantized Value")
        ax.set_ylabel("Count")
        fig.tight_layout()

        writer.add_figure(tag, fig, global_step=step)
        plt.close(fig)

    def _save_histogram(self, model: Model, writer: SummaryWriter, step: int):
        for module_name, module in model.model.named_modules():
            module_name = module_name.replace(".", "/") or "root"
            for param in ("weight", "bias"):
                self._log_param(module, writer, step, module_name, param)

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
        step = (
            state.global_step
            if args.logging_strategy == LoggingStrategy.STEPS
            else state.epoch
        )

        self._save_histogram(model, summary_writer, step)
