# The class Trainer is simplified from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

import os
import random
import logging
from dataclasses import asdict
from functools import partial
from typing import Optional, Callable, List
import shutil

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from hatorch.models import Model
from hatorch.utils.logger import logger

from .trainer_state import TrainerState
from .trainer_control import TrainerControl
from .training_args import TrainingArguments
from .callbacks import (
    TrainerCallback,
    LogMetricsCallback,
    SaveTrainingArgsCallback,
    SaveOptimizerCallback,
    SaveLRSchedulerCallback,
)

from hatorch.transforms import utils as transforms_utils

from .utils import (
    EvaluationStrategy,
    SaveStrategy,
    LoggingStrategy,
)

DEFAULT_CALLBACKS = [
    SaveTrainingArgsCallback(),
    SaveOptimizerCallback(),
    SaveLRSchedulerCallback(),
    LogMetricsCallback(),
]


class Trainer:

    def __init__(
        self,
        model: Model,
        device: torch.device,
        train_loader,
        valid_loader,
        optimizer: torch.optim.Optimizer,
        compute_loss_func: Optional[Callable],
        rank = 0,
        args: TrainingArguments = None,
        lr_scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        if args is None:
            logger.info("No `TrainingArguments`. Using default")
            args = TrainingArguments()

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.compute_loss_func = compute_loss_func
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.world_size = (
            torch.distributed.get_world_size() if self.is_distributed else 1
        )
        if self.is_distributed and torch.distributed.get_rank() != 0:
            logger.setLevel(logging.ERROR)
        self._ddp_find_unused_parameters: Optional[bool] = None
        if isinstance(self.model.model, DDP):
            self._ddp_find_unused_parameters = bool(
                getattr(self.model.model, "find_unused_parameters", False)
            )
        self._non_blocking_transfers = bool(self.args.pin_memory)

        if callbacks is None:
            self.callbacks = DEFAULT_CALLBACKS
        else:
            self.callbacks = DEFAULT_CALLBACKS + callbacks

        self.state = TrainerState()
        self.control = TrainerControl()
        self._executed_epoch_calls: set[int] = set()

        # Create output directory once in multi-process mode.
        if self._is_main_process():
            self._create_output_dir()
        if self.is_distributed:
            torch.distributed.barrier()

        # Setup loggers
        self.summary_writer = (
            SummaryWriter(self.args.output_dir) if self._is_main_process() else None
        )

        # Callback "on_init_end"
        self._call_callbacks("on_init_end")

    @staticmethod
    def _shutdown_dataloader_workers(dataloader) -> None:
        iterator = getattr(dataloader, "_iterator", None)
        if iterator is None:
            return
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        dataloader._iterator = None

    def close(self) -> None:
        if self.summary_writer is not None:
            self.summary_writer.flush()
            self.summary_writer.close()
            self.summary_writer = None
        self._shutdown_dataloader_workers(self.train_loader)
        self._shutdown_dataloader_workers(self.valid_loader)

    def _is_main_process(self) -> bool:
        if self.is_distributed:
            return torch.distributed.get_rank() == 0
        return self.rank == 0

    def _tqdm_unit_scale(self):
        # In DDP, report aggregate throughput (all ranks), not per-rank throughput.
        return self.world_size if self.world_size > 1 else False

    def _unwrap_model(self) -> torch.nn.Module:
        wrapped_model = self.model.model
        return wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model

    @staticmethod
    def _strip_module_prefix_if_present(state_dict: dict) -> dict:
        if (
            isinstance(state_dict, dict)
            and state_dict
            and all(isinstance(key, str) and key.startswith("module.") for key in state_dict.keys())
        ):
            return {key[len("module."):]: value for key, value in state_dict.items()}
        return state_dict

    def _move_batch_to_device(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.to(self.device, non_blocking=self._non_blocking_transfers)
        targets = targets.to(self.device, non_blocking=self._non_blocking_transfers)
        return inputs, targets

    def _set_ddp_mode(self, find_unused_parameters: bool) -> None:
        if not self.is_distributed:
            return

        wrapped_model = self.model.model
        is_wrapped = isinstance(wrapped_model, DDP)
        if is_wrapped and self._ddp_find_unused_parameters == find_unused_parameters:
            return

        base_model = wrapped_model.module if is_wrapped else wrapped_model
        if torch.cuda.is_available():
            device_id = self.rank if isinstance(self.rank, int) else torch.cuda.current_device()
            base_model = base_model.to(torch.device(f"cuda:{device_id}"))
            self.model.model = DDP(
                base_model,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            self.model.model = DDP(
                base_model,
                find_unused_parameters=find_unused_parameters,
            )
        self._ddp_find_unused_parameters = find_unused_parameters

    def calibrate(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: Optional[int] = None,
        num_batches: Optional[int] = None,
        tqdm_steps: int = 50,
    ):
        """
        Calibrate quantizers by running forward and backward passes.
        
        Args:
            optimizer: Optimizer for calibration (typically SGD with small LR)
            num_epochs: Number of full epochs for calibration (mutually exclusive with num_batches)
            num_batches: Number of batches for calibration (mutually exclusive with num_epochs)
            tqdm_steps: Frequency of progress bar updates
        """
        if num_epochs is None and num_batches is None:
            raise ValueError("Must specify either num_epochs or num_batches for calibration")
        if num_epochs is not None and num_batches is not None:
            raise ValueError("Cannot specify both num_epochs and num_batches")
        
        logger.info("Starting calibration phase")

        if self.is_distributed:
            self._set_ddp_mode(find_unused_parameters=True)
        
        # Freeze model parameters and set quantizers to calibration mode
        frozen_params = transforms_utils.freeze_model_parameters(self.model.model)

        try:
            transforms_utils.set_quantizers_to_mode(self.model.model, "calibration")
            self.model.model.train()

            if num_epochs is not None:
                # Calibrate for specified number of epochs
                for epoch in range(num_epochs):
                    self._calibrate_one_epoch(optimizer, epoch + 1, num_epochs, tqdm_steps)
            else:
                # Calibrate for specified number of batches
                self._calibrate_batches(optimizer, num_batches, tqdm_steps)
        finally:
            # Restore quantizers to training mode and unfreeze parameters even on failure.
            transforms_utils.set_quantizers_to_mode(self.model.model, "training")
            transforms_utils.unfreeze_model_parameters(self.model.model, frozen_params)
        
        logger.info("Calibration phase completed")
    
    def _calibrate_one_epoch(self, optimizer, current_epoch, total_epochs, tqdm_steps):
        """Run calibration for one full epoch."""
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Calibration Epoch {current_epoch}/{total_epochs}",
            unit="batch",
            unit_scale=self._tqdm_unit_scale(),
            dynamic_ncols=True,
            colour="yellow",
            disable=not self._is_main_process(),
        )
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = self._move_batch_to_device(inputs, targets)
            optimizer.zero_grad()
            outputs = self.model.model(inputs)
            loss = self.compute_loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % tqdm_steps == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        logger.info(f"Calibration Epoch {current_epoch}/{total_epochs} completed")
        return epoch_loss / len(self.train_loader)
    
    def _calibrate_batches(self, optimizer, num_batches, tqdm_steps):
        """Run calibration for a specified number of batches."""
        progress_bar = tqdm(
            total=num_batches,
            desc=f"Calibration",
            unit="batch",
            unit_scale=self._tqdm_unit_scale(),
            dynamic_ncols=True,
            colour="yellow",
            disable=not self._is_main_process(),
        )
        
        batch_count = 0
        for inputs, targets in self.train_loader:
            if batch_count >= num_batches:
                break
                
            inputs, targets = self._move_batch_to_device(inputs, targets)
            optimizer.zero_grad()
            outputs = self.model.model(inputs)
            loss = self.compute_loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_count % tqdm_steps == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            progress_bar.update(1)
            batch_count += 1
        
        progress_bar.close()
        logger.info(f"Calibration completed after {batch_count} batches")

    def _create_output_dir(self):
        args = self.args
        if os.path.isdir(args.output_dir):
            logger.info("Output directory '%s' already exists.", args.output_dir)

            # Overwriting logic
            if args.overwrite_output_dir:
                logger.warning("Overwriting output directory '%s'.", args.output_dir)
                shutil.rmtree(args.output_dir)
                os.makedirs(args.output_dir, exist_ok=False)
            else:
                logger.info(
                    "Keeping existing output directory '%s' (overwrite disabled).",
                    args.output_dir,
                )
        else:
            logger.info("Creating output directory '%s'.", args.output_dir)
            os.makedirs(args.output_dir, exist_ok=False)

    def _call_callbacks(self, event_name: str, **kwargs):
        if not self._is_main_process():
            return
        for cb in self.callbacks:
            getattr(cb, event_name)(
                self.model,
                self.state,
                self.control,
                self.args,
                self.optimizer,
                self.lr_scheduler,
                **kwargs,
            )

    def _log(self, last_metrics: dict[str, float]):
        # Historic
        self.state.log_history.append(last_metrics.copy())
        # Callback
        self._call_callbacks(
            "on_log", last_metrics=last_metrics, summary_writer=self.summary_writer
        )

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        args = self.args

        if self.is_distributed:
            self._set_ddp_mode(find_unused_parameters=False)

        # Reset TrainerControl
        self.control.new_training()

        # AMP Initialization
        if args.amp:
            logger.info("Using Automatic Mixed Precision (AMP)")
        self.scaler = torch.GradScaler(device=self.device, enabled=args.amp)

        # Reset TrainerState unless resuming
        if resume_from_checkpoint:
            checkpoint_path = self._resolve_checkpoint_path(resume_from_checkpoint)
            self._load_checkpoint(checkpoint_path)
            start_epoch = self.state.epoch + 1
            logger.info(
                "Resuming training from '%s' (epoch %d, global_step %d).",
                checkpoint_path,
                self.state.epoch,
                self.state.global_step,
            )
        else:
            self.state.epoch = 0
            self.state.global_step = 0
            self.state.best_metric = None
            self.state.best_global_step = None
            self.state.log_history = []
            start_epoch = 1

        # Reset metrics
        epoch_metrics = {}

        self._call_callbacks("on_train_begin")
        self._executed_epoch_calls = set()

        merged_epoch_calls: dict[int, list[Callable[[], None]]] = {}
        if self.args.epoch_calls:
            for epoch, callbacks in self.args.epoch_calls.items():
                merged_epoch_calls.setdefault(epoch, []).extend(callbacks)

        self._epoch_calls = merged_epoch_calls

        if self._epoch_calls:
            for epoch in self._epoch_calls:
                if epoch < start_epoch:
                    self._executed_epoch_calls.add(epoch)

        if args.eval_on_start:
            avg_vloss, top1_acc, top5_acc = self.evaluate(log_prefix="SoftEvaluation")
            self._update_metrics(epoch_metrics, avg_vloss, top1_acc, top5_acc)
            self._log_evaluation_metrics(0, epoch_metrics)
            self._update_best_metric(avg_vloss)
            self._call_callbacks("on_evaluate")

        # TODO: no early-stopping mechanism
        if start_epoch > args.num_epochs:
            logger.info(
                "Checkpoint already reached target num_epochs=%d. Nothing to train.",
                args.num_epochs,
            )
            return

        for epoch in range(start_epoch, args.num_epochs + 1):
            # Reset metrics
            epoch_metrics = {}

            # Epoch reset of TrainerControl
            self.control.new_epoch()

            self.state.epoch = epoch
            if self.optimizer.param_groups:
                logger.info(
                    "Epoch %d learning rate: %.6f",
                    self.state.epoch,
                    self.optimizer.param_groups[0]["lr"],
                )
            self._run_epoch_calls(self.state.epoch)

            avg_epoch_loss = self._train_one_epoch(self.scaler)

            # TrainerControl flags : Epoch logic
            ## evaluation
            self.control.should_evaluate = (
                args.eval_strategy == EvaluationStrategy.EPOCH
            )
            ## save
            self.control.should_save = args.save_strategy == SaveStrategy.EPOCH
            ## log
            self.control.should_log = args.logging_strategy == LoggingStrategy.EPOCH

            if self.control.should_evaluate:
                avg_vloss, top1_acc, top5_acc = self.evaluate(log_prefix="SoftEvaluation")
                self._update_metrics(epoch_metrics, avg_vloss, top1_acc, top5_acc)
                self._log_evaluation_metrics(self.state.epoch, epoch_metrics)
                self._update_best_metric(avg_vloss)
                self._call_callbacks("on_evaluate")

            if self.control.should_log:
                epoch_metrics["Training/Loss"] = avg_epoch_loss
                self._log_training_metrics(self.state.epoch, epoch_metrics)
                self._log(epoch_metrics)

            if self.control.should_save:
                self._save_checkpoint(epoch)

        self._call_callbacks("on_train_end")

    def _run_epoch_calls(self, epoch: int) -> None:
        epoch_calls = self._epoch_calls
        if not epoch_calls or epoch in self._executed_epoch_calls:
            return

        callbacks = epoch_calls.get(epoch, [])
        for callback in callbacks:
            callback_name = self._callback_name(callback)
            logger.info(
                "Applying %s at epoch %d.",
                callback_name,
                epoch,
            )
            callback()
        self._executed_epoch_calls.add(epoch)

    @staticmethod
    def _callback_name(callback: Callable) -> str:
        if isinstance(callback, partial):
            return Trainer._callback_name(callback.func)
        return getattr(callback, "__name__", callback.__class__.__name__)

    def disable_weight_decay_for_lsq_parameters(
        self,
        include_scale: bool = True,
        include_zero_point: bool = False,
        weights_only: bool = True,
        activations_only: bool = False,
    ) -> int:
        """
        Move selected LSQ learnable parameters into optimizer groups with zero weight decay.

        Args:
            include_scale: Disable decay for LSQ scale parameters.
            include_zero_point: Disable decay for learnable zero-points.
            weights_only: Apply only to weight quantizers.
            activations_only: Apply only to activation quantizers.

        Returns:
            Number of parameters moved to zero-decay groups.
        """
        lsq_params = transforms_utils.get_lsq_learnable_parameters(
            self.model.model,
            include_scale=include_scale,
            include_zero_point=include_zero_point,
            weights_only=weights_only,
            activations_only=activations_only,
        )
        moved = transforms_utils.set_optimizer_weight_decay_for_parameters(
            self.optimizer,
            lsq_params,
            weight_decay=0.0,
        )
        if moved > 0 and self.lr_scheduler is not None and hasattr(self.lr_scheduler, "base_lrs"):
            current_lrs = [group["lr"] for group in self.optimizer.param_groups]
            self.lr_scheduler.base_lrs = current_lrs.copy()
            if hasattr(self.lr_scheduler, "_last_lr"):
                self.lr_scheduler._last_lr = current_lrs.copy()
        logger.info(
            "Disabled weight decay for %d LSQ parameters (scale=%s, zero_point=%s).",
            moved,
            include_scale,
            include_zero_point,
        )
        return moved

    def _train_one_epoch(self, scaler: torch.GradScaler):
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(self.state.epoch)
        args = self.args
        epoch_loss = 0.0
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.state.epoch}/{args.num_epochs}",
            unit="batch",
            unit_scale=self._tqdm_unit_scale(),
            dynamic_ncols=True,
            colour="cyan",
            disable=not self._is_main_process(),
        )
        progress_bar.set_postfix(step=self.state.global_step, loss=f"{0:.4f}")

        self.model.model.train(True)

        for input, target in progress_bar:
            # metrics for current step
            step_metrics = {}

            # Reset step of TrainerControl
            self.control.new_step()

            self.optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with torch.autocast(device_type=self.device, enabled=args.amp):
                input, target = self._move_batch_to_device(input, target)
                output = self.model.model(input)
                loss = self.compute_loss_func(output, target)

            # Exits the context manager before backward()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            step_loss = loss.item()
            epoch_loss += step_loss
            self._call_callbacks("on_step_end", step_loss=step_loss)

            # TrainerControl flags
            ## evaluation
            self.control.should_evaluate = (
                args.eval_strategy == EvaluationStrategy.STEPS
                and self.state.global_step % args.eval_steps == 0
            )
            ## save
            self.control.should_save = (
                args.save_strategy == SaveStrategy.STEPS
                and self.state.global_step % args.save_steps == 0
            )
            ## log
            self.control.should_log = (
                args.logging_strategy == LoggingStrategy.STEPS
                and self.state.global_step % args.logging_steps == 0
            )

            # TQDM Description
            if self.state.global_step % args.tqdm_steps == 0:
                progress_bar.set_description(
                    f"Epoch {self.state.epoch}/{args.num_epochs}"
                )
                progress_bar.set_postfix(
                    step=self.state.global_step, loss=f"{step_loss:.4f}"
                )

            if self.control.should_evaluate:
                avg_vloss, top1_acc, top5_acc = self.evaluate(log_prefix="SoftEvaluation")
                self._update_metrics(step_metrics, avg_vloss, top1_acc, top5_acc)
                self._log_evaluation_metrics(self.state.global_step, step_metrics)
                self._update_best_metric(avg_vloss)
                self._call_callbacks("on_evaluate")
                self.model.model.train(True)

            if self.control.should_save:
                self._save_checkpoint(self.state.global_step)

            if self.control.should_log:
                step_metrics["Training/Loss"] = step_loss
                self._log_training_metrics(self.state.global_step, step_metrics)
                self._log(step_metrics)

            self.state.global_step += 1

        if self.lr_scheduler:
            self.lr_scheduler.step()

        avg_epoch_loss = epoch_loss / len(self.train_loader)

        self.model.model.eval()

        return avg_epoch_loss

    def evaluate(self, log_prefix: str = "Evaluation"):
        self.model.model.eval()
        total_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total = 0
        num_batches = 0
        with torch.inference_mode():
            for input, target in self.valid_loader:
                # Enables autocasting for the forward pass (model + loss)
                with torch.autocast(device_type=self.device, enabled=self.args.amp):
                    input, target = self._move_batch_to_device(input, target)
                    output = self.model.model(input)

                batch_size, num_classes = output.shape
                # Top-1
                _, top1_prediction = torch.max(output, dim=1)
                top1_correct += (top1_prediction == target).sum().item()

                # Top-5
                # TODO: user should give metrics
                # Top-k : min entre 5 et num_classes
                k = min(5, num_classes)
                _, topk_prediction = output.topk(k=k, dim=1, largest=True, sorted=True)
                top5_correct += (
                    topk_prediction.eq(target.view(-1, 1))
                    .sum(dim=1)
                    .clamp(max=1)
                    .sum()
                    .item()
                )

                # Loss
                loss = self.compute_loss_func(output, target)
                total_loss += loss.item()
                num_batches += 1

                # Total samples
                total += batch_size

        if self.is_distributed:
            metrics = torch.tensor(
                [total_loss, float(top1_correct), float(top5_correct), float(total), float(num_batches)],
                device=self.device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            total_loss, top1_correct, top5_correct, total, num_batches = metrics.tolist()

        if total == 0 or num_batches == 0:
            return 0.0, 0.0, 0.0

        top1_accuracy = 100 * top1_correct / total
        top5_accuracy = 100 * top5_correct / total
        avg_vloss = total_loss / num_batches

        if self._is_main_process():
            epoch_display = self.state.epoch if self.state.epoch is not None else "-"
            step_display = self.state.global_step if self.state.global_step is not None else "-"
            logger.info(
                "%s (E%s|S%s) - Loss: %.4f | Top-1: %.2f%% | Top-5: %.2f%%",
                log_prefix,
                epoch_display,
                step_display,
                avg_vloss,
                top1_accuracy,
                top5_accuracy,
            )
        return avg_vloss, top1_accuracy, top5_accuracy

    def _save_checkpoint(self, step):
        if not self._is_main_process():
            return
        ckpt_name = f"checkpoint-{step}.pt"
        ckpt_path = os.path.join(self.args.output_dir, ckpt_name)
        logger.info("Saving checkpoint to '%s'.", ckpt_path)
        checkpoint = {
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_state": asdict(self.state),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "training_args": self.args.to_dict(),
        }
        if torch.cuda.is_available():
            checkpoint["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        scaler = getattr(self, "scaler", None)
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        torch.save(checkpoint, ckpt_path)
        self.save_network(step=step)

    def save_network(self, path: Optional[str] = None, step: Optional[int] = None) -> str:
        """Save model weights for reuse in other trainings."""
        if path is None:
            if step is None:
                step = self.state.global_step
            path = os.path.join(self.args.output_dir, f"model-{step}.pt")
        if not self._is_main_process():
            return path
        logger.info("Saving model weights to '%s'.", path)
        torch.save(self._unwrap_model().state_dict(), path)
        return path

    def load_network(self, path: str, strict: bool = True):
        """Load model weights without restoring optimizer/scheduler states."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model weights file '{path}' does not exist.")
        logger.info("Loading model weights from '%s'.", path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        state_dict = self._strip_module_prefix_if_present(state_dict)
        incompatible = self._unwrap_model().load_state_dict(state_dict, strict=strict)
        self.model.model.to(self.device)
        if not strict:
            logger.info(
                "Loaded weights with missing keys: %s; unexpected keys: %s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )

    def _update_metrics(self, metrics, avg_vloss, top1_acc, top5_acc):
        # Save evaluation metrics for current step
        metrics["Evaluation/Loss"] = avg_vloss
        metrics["Evaluation/Top1"] = top1_acc
        metrics["Evaluation/Top5"] = top5_acc

    def _update_best_metric(self, avg_vloss):
        if self.state.best_metric is None or avg_vloss < self.state.best_metric:
            self.state.best_metric = avg_vloss
            self.state.best_global_step = self.state.global_step
            # TODO: bit ugly
            self.control.should_save |= self.args.save_strategy == SaveStrategy.BEST

    def _log_evaluation_metrics(self, step, metrics):
        if not self._is_main_process() or self.summary_writer is None:
            return
        for key in ["Evaluation/Loss", "Evaluation/Top1", "Evaluation/Top5"]:
            value = metrics[key]
            self.summary_writer.add_scalar(key, value, step)

    def _log_training_metrics(self, step, metrics):
        if not self._is_main_process() or self.summary_writer is None:
            return
        key = "Training/Loss"
        self.summary_writer.add_scalar(key, metrics[key], step)

    def get_last_checkpoint(self, directory: Optional[str] = None) -> Optional[str]:
        """Return the latest checkpoint path in `directory` (or output_dir if None)."""
        ckpt_dir = directory or self.args.output_dir
        if not os.path.isdir(ckpt_dir):
            return None

        checkpoints = [
            os.path.join(ckpt_dir, name)
            for name in os.listdir(ckpt_dir)
            if name.startswith("checkpoint-") and name.endswith(".pt")
        ]
        if not checkpoints:
            return None

        checkpoints.sort(key=self._checkpoint_sort_key)
        return checkpoints[-1]

    def _checkpoint_sort_key(self, path: str):
        filename = os.path.basename(path)
        step_str = filename.replace("checkpoint-", "").replace(".pt", "")
        try:
            return int(step_str)
        except ValueError:
            return filename

    def _resolve_checkpoint_path(self, resume_from_checkpoint: str) -> str:
        if os.path.isdir(resume_from_checkpoint):
            last_checkpoint = self.get_last_checkpoint(resume_from_checkpoint)
            if last_checkpoint is None:
                raise FileNotFoundError(
                    f"No checkpoint found in directory '{resume_from_checkpoint}'."
                )
            return last_checkpoint

        if not os.path.isfile(resume_from_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint file '{resume_from_checkpoint}' does not exist."
            )
        return resume_from_checkpoint

    def _load_checkpoint(self, checkpoint_path: str):
        logger.info("Loading checkpoint from '%s'.", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model_state_dict = self._strip_module_prefix_if_present(checkpoint["model_state_dict"])
        self._unwrap_model().load_state_dict(model_state_dict)
        self.model.model.to(self.device)

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)

        lr_scheduler_state = checkpoint.get("lr_scheduler_state_dict")
        if self.lr_scheduler is not None and lr_scheduler_state is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state)

        scaler_state = checkpoint.get("scaler_state_dict")
        if getattr(self, "scaler", None) is not None and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        trainer_state = checkpoint.get("trainer_state")
        if trainer_state is not None:
            self.state = TrainerState(**trainer_state)
        else:
            # Backward compatibility: minimum state information
            self.state.epoch = checkpoint.get("epoch", 0)
            self.state.global_step = checkpoint.get("global_step", 0)

        random_state = checkpoint.get("random_state")
        if random_state is not None:
            random.setstate(random_state)

        numpy_state = checkpoint.get("numpy_random_state")
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        torch_state = checkpoint.get("torch_random_state")
        if torch_state is not None:
            torch.set_rng_state(torch_state.cpu())

        torch_cuda_state = checkpoint.get("torch_cuda_random_state_all")
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch_cuda_state = [state.cpu() for state in torch_cuda_state]
            torch.cuda.set_rng_state_all(torch_cuda_state)
