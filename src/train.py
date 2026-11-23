"""
Train a ConvTasNet model with the specified architecture and quantization configuration.

Usage:
    python src/train.py <model_architecture>

Arguments:
    model_architecture:
        Model architecture identifier (e.g. ``0.5``, ``1.5``).

Examples:
    python src/train.py 0.5
"""

import os
import sys

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    pit_permutate,
)
import torch.nn.functional as F
from model.convtasnet import ConvTasNet
from utils.conf import Conf
from utils.lightning_module import ConvTasNetLightningModule
from utils.utils import construct_model, get_data_loaders, neg_sisdr

accelerator = "gpu"
devices = "auto"
strategy = "auto"


class PIT_SISDR_L1_Loss(torch.nn.Module):
    def __init__(self, sisdr_ratio=0.9, l1_ratio=0.2):
        super().__init__()
        self.sisdr_ratio = sisdr_ratio if sisdr_ratio is not None else 1.0
        self.l1_ratio = l1_ratio if l1_ratio is not None else 1.0

    def forward(self, est: torch.Tensor, target: torch.Tensor):
        sisdr, l1 = self._terms(est, target)
        return self.sisdr_ratio * sisdr + self.l1_ratio * l1

    def _terms(self, est, target):
        # choose ONE permutation (SI-SDR) and apply it to both terms
        _, perm = permutation_invariant_training(
            est, target, neg_sisdr, eval_func="min"
        )
        est_p = pit_permutate(est, perm)
        l_sisdr = neg_sisdr(est_p.flatten(0, 1), target.flatten(0, 1)).mean()
        l_l1 = (est_p - target).abs().mean()
        return l_sisdr, l_l1


def train(conf: Conf, resume_from: None | str = None):

    if conf.training.seed:
        print(f"Setting seed to {conf.training.seed}")
        torch.manual_seed(conf.training.seed)

    train_loader, val_loader = get_data_loaders(conf=conf)

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    model: ConvTasNet = construct_model(conf=conf)
    model.compile()  # in-place: state_dict keys keep their names (no `_orig_mod.` prefix)

    optimizer = Adam(
        model.parameters(),
        lr=conf.optim.lr,
        weight_decay=conf.optim.weight_decay,
    )

    # Define scheduler
    scheduler = None
    if conf.training.half_lr:
        # TODO: patience=5 in asteroid
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3)

    pit_scale_invariant_signal_distortion_ratio = PermutationInvariantTraining(
        neg_sisdr, eval_func="min"
    )
    sisdr_ratio, l1_ratio = conf.training.sisdr_weight, conf.training.l1_weight
    print(f"sisdr_ratio, l1_ratio = {sisdr_ratio}, {l1_ratio}")
    pit_sisdr_with_l1 = PIT_SISDR_L1_Loss(sisdr_ratio, l1_ratio)
    module = ConvTasNetLightningModule(
        model=model,
        loss=(pit_sisdr_with_l1
            if conf.training.with_l1 is not None and conf.training.with_l1
            else pit_scale_invariant_signal_distortion_ratio
        ),
        optimizer=optimizer,
        scheduler=scheduler,
    )

    exp_dir = conf.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf_data, outfile)

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf.training.early_stop:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True)
        )

    trainer = pl.Trainer(
        max_epochs=conf.training.epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        # TODO: change to train the full model
        # limit_train_batches = 100,
        # limit_val_batches=100,
        # limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        # It may slow down training to log on every single batch. By default, Lightning logs every 50 rows, or 50 training steps
        # log_every_n_steps=2,
    )
    trainer_exp_path = trainer.log_dir
    conf_path = os.path.join(trainer_exp_path, "conf.yml")
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from,
    )
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf_data, outfile)
    export_path = os.path.join(trainer_exp_path, "compiled_model.pth")
    torch.save(module.model.state_dict(), export_path)
    print(f"Compiled model saved to {export_path}")
    export_path = os.path.join(exp_dir, "compiled_model.pth")
    torch.save(module.model.state_dict(), export_path)
    print(f"Compiled model saved to {export_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--config_dir", type=str, default=".")
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    model_archi = args.config
    resume_from = args.resume_from

    print(f"Using model architecture: {model_archi}")

    CONF_PATH = os.path.join(args.config_dir, f"conf_{model_archi}.yml")
    with open(CONF_PATH) as f:
        conf_data = yaml.safe_load(f)

    conf = Conf(conf_data)

    def train_loop():
        while True:
            try:
                train(conf=conf, resume_from=resume_from)
                break
            except torch.cuda.OutOfMemoryError as e:
                print(f"Caught a RuntimeError: {e}")
                conf.training.batch_size = conf.training.batch_size // 2
                torch.cuda.empty_cache()

                if conf.training.batch_size == 0:
                    print(f"Reached batch size = 0")
                    break
                print(f"Retrying with batch size: {conf.training.batch_size}")

    train_loop()
