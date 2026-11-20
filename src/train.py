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
import torch.nn.functional as F
from model.convtasnet import ConvTasNet
from utils.conf import Conf
from utils.lightning_module import ConvTasNetLightningModule
from utils.utils import construct_model, get_data_loaders, neg_sisdr

accelerator = "gpu"
devices = "auto"
strategy = "auto"


class PIT_SISDR_L1_Loss(torch.nn.Module):
    @staticmethod
    def _create_loss(sisdr_ratio=0.9, l1_ratio=0.2):
        return lambda est, target: sisdr_ratio * neg_sisdr(est, target) + l1_ratio * (
            est - target
        ).abs().mean(dim=-1)

    def __init__(self, sisdr_ratio=0.9, l1_ratio=0.2):
        super().__init__()

        self.pit_sisdr_l1 = PermutationInvariantTraining(
            self._create_loss(sisdr_ratio=sisdr_ratio, l1_ratio=l1_ratio),
            eval_func="min",
        )
        
        self._pit_sisdr = PermutationInvariantTraining(
            neg_sisdr,
            eval_func="min",
        )

        self._pit_l1 = PermutationInvariantTraining(
            lambda est, target : (est - target).abs().mean(dim=-1),
            eval_func="min",
        )
        self.sisdr_ratio = sisdr_ratio
        self.l1_ratio = l1_ratio

    def forward(self, est: torch.Tensor, target: torch.Tensor):
        return self.pit_sisdr_l1(est, target)
    
    def pit_sisdr(self, est, target):
        return self._pit_sisdr(est, target)
    
    def pit_l1(self, est, target):
        return self._pit_l1(est, target)


def train(conf: Conf, resume_from: None | str = None, with_l1: bool = False):

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
    pit_sisdr_with_l1 = PIT_SISDR_L1_Loss(0.446, 1.0)
    module = ConvTasNetLightningModule(
        model=model,
        loss=(
            pit_scale_invariant_signal_distortion_ratio
            if not with_l1
            else pit_sisdr_with_l1
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
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from,
    )

    export_path = os.path.join(exp_dir, "compiled_model.pth")
    torch.save(module.model.state_dict(), export_path)
    print(f"Compiled model saved to {export_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--config_dir", type=str, default=".")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--with_l1", action=BooleanOptionalAction, default=False)

    args = parser.parse_args()

    model_archi = args.config
    resume_from = args.resume_from

    print(f"Using model architecture: {model_archi}")

    CONF_PATH = os.path.join(args.config_dir, f"conf_{model_archi}.yml")
    with open(CONF_PATH) as f:
        conf_data = yaml.safe_load(f)

    conf = Conf(conf_data)

    train(conf=conf, resume_from=resume_from, with_l1=args.with_l1)
