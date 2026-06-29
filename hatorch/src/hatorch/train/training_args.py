# Credits to https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/training_args.py

import json
from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional, Any, Callable

from .utils import SaveStrategy, EvaluationStrategy, LoggingStrategy

from hatorch.utils.logger import logger


@dataclass
class TrainingArguments:
    """
    Configuration arguments for training and evaluation.

    Attributes:
        output_dir (str):
            Directory for model outputs and checkpoints.
            Defaults to "trainer_output" if not specified.
        overwrite_output_dir (bool):
            Whether to overwrite the contents of `output_dir` if it already exists.
            Default: True.
        num_epochs (int):
            Total number of training epochs.
            Default: 50.
        save_strategy (SaveStrategy):
            Checkpoint saving strategy. Options:
                - SaveStrategy.NO:      No checkpoints are saved.
                - SaveStrategy.EPOCH:   Save at the end of each epoch and at the end of training.
                - SaveStrategy.STEPS:   Save every `save_steps` steps and at the end of training.
                - SaveStrategy.BEST:    Save only when a new best metric is achieved.
            Default: SaveStrategy.BEST.
        save_steps (int):
            Number of update steps between checkpoint saves when using SaveStrategy.STEPS.
            Default: 500.
        amp (bool):
            Whether to use automatic mixed precision (AMP).
            Default: False.
        eval_strategy (EvaluationStrategy):
            Evaluation frequency. Options:
                - EvaluationStrategy.NO:     No evaluation is performed.
                - EvaluationStrategy.STEPS:  Evaluate every `eval_steps` steps.
                - EvaluationStrategy.EPOCH:  Evaluate at the end of each epoch.
            Default: EvaluationStrategy.EPOCH.
        eval_on_start (bool):
            Run evaluation before the first training step.
            Default: False.
        eval_steps (int):
            Number of update steps between evaluations when using EvaluationStrategy.STEPS.
            Default: 100.
        logging_strategy (LoggingStrategy):
            Logging frequency and callback invocation. Options:
                - LoggingStrategy.NO:     No training logs or callbacks.
                - LoggingStrategy.STEPS:  Log training metrics and call `on_log` callbacks every `logging_steps` steps.
                - LoggingStrategy.EPOCH:  Log training metrics and call `on_log` callbacks at the end of each epoch.
            Default: LoggingStrategy.EPOCH.

            Note:
                Evaluation metrics (from `eval_strategy`) are always logged automatically,
                regardless of the chosen logging strategy.
        logging_steps (int):
            Number of update steps between logs when using LoggingStrategy.STEPS.
            Default: 100.
        num_workers (int):
            Number of subprocesses used by dataset DataLoader workers.
            Default: 0 (load data in the main process).
        persistent_workers (bool):
            Keep DataLoader worker processes alive between epochs.
            Requires `num_workers > 0`.
            Default: False.
        pin_memory (bool):
            Use pinned CPU memory in DataLoader for faster host-to-device copies.
            Default: False.
        prefetch_factor (int, optional):
            Number of batches loaded in advance by each DataLoader worker.
            Requires `num_workers > 0` when set.
            Default: None (PyTorch default behavior).
        tqdm_steps (int):
            Number of steps between tqdm progress bar updates.
            Default: 300.
        epoch_calls (dict[int, callable | list[callable]], optional):
            One-time callbacks triggered at the start of the given epoch (1-based).
            This field is runtime-only and is intentionally excluded from JSON
            serialization.
    """

    output_dir: Optional[str] = None
    overwrite_output_dir: bool = True
    num_epochs: int = 50
    save_strategy: SaveStrategy = SaveStrategy.BEST
    save_steps: int = 500
    amp: bool = False
    eval_strategy: EvaluationStrategy = EvaluationStrategy.EPOCH
    eval_on_start: bool = False
    eval_steps: int = 100
    # TODO: per_device_train_batch_size: int = 8
    # TODO: per_device_eval_batch_size: int = 8
    logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH
    logging_steps: int = 100
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    tqdm_steps: int = 300
    epoch_calls: Optional[dict[int, Callable[[], None] | list[Callable[[], None]]]] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = "trainer_output"
            logger.info(
                "No output directory specified, defaulting to 'trainer_output'. "
                "To change this behavior, specify --output_dir when creating TrainingArguments."
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers=True requires num_workers > 0."
            )
        if self.prefetch_factor is not None:
            if self.prefetch_factor < 1:
                raise ValueError(
                    f"prefetch_factor must be >= 1, got {self.prefetch_factor}"
                )
            if self.num_workers == 0:
                raise ValueError(
                    "prefetch_factor requires num_workers > 0."
                )
        if self.epoch_calls is not None:
            normalized: dict[int, list[Callable[[], None]]] = {}
            for epoch, callbacks in self.epoch_calls.items():
                if not isinstance(epoch, int) or epoch < 1:
                    raise ValueError("epoch_calls keys must be positive integers (1-based epochs).")
                if callable(callbacks):
                    normalized[epoch] = [callbacks]
                    continue
                if isinstance(callbacks, list) and all(callable(cb) for cb in callbacks):
                    normalized[epoch] = callbacks
                    continue
                raise TypeError(
                    "Each epoch_calls value must be a callable or a list of callables."
                )
            self.epoch_calls = normalized

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, list):
            return [self._serialize(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        if hasattr(value, "__dataclass_fields__"):
            return {
                field.name: self._serialize(getattr(value, field.name))
                for field in fields(value)
            }
        return value

    def to_dict(self):
        """
        Serializes this instance, converting any Enum fields to their values for JSON support.
        """
        # filter out fields that are defined as field(init=False)
        d = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.init
        }
        # Runtime-only callbacks are intentionally not serialized.
        d.pop("epoch_calls", None)

        return {k: self._serialize(v) for k, v in d.items()}

    def to_json_string(self):
        """
        Returns this instance serialized as a pretty-printed JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)
