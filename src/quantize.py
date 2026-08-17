"""
Quantization Aware Training (QAT) of a trained ConvTasNet model using HATorch.

Usage:
    python src/quantize.py <model_architecture> <quantization_config>

Arguments:
    model_architecture:
        Model architecture identifier (e.g. ``0.5``, ``1.5``).

    quantization_config:
        Name of the quantization configuration to apply (e.g. ``all_S8S8``, ``all_S16S16``...).

The number of QAT epochs is defined in the configuration file (conf_<model_architecture>.yml) under the key ``quantize.epochs``. The default value is 30.

Examples:
    python src/quantize.py 0.5 all_S8S8
"""

import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
import yaml
from hatorch.transforms import utils as transforms_utils
from matplotlib import use
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.audio import PermutationInvariantTraining

from utils.conf import Conf
from utils.lightning_module import ConvTasNetLightningModule
from utils.utils import get_data_loaders, load_trained_model, neg_sisdr
from utils.utils_quantize import quantize_model, validate_quantization_config_names

accelerator = "gpu"
devices = "auto"
strategy = "auto"

strategy = "ddp_find_unused_parameters_true"


CHECKPOINT_FILENAME: str | None = None
# CHECKPOINT_FILENAME = "epoch=2-step=3174.ckpt"


def train(conf: Conf, model_archi: str, quantization_config_name: str):

    if conf.training.seed:
        print(f"Setting seed to {conf.training.seed}")
        torch.manual_seed(conf.training.seed)

    if False:
        pass
    # elif quantization_config_name == "all_S8S8":
    #     CHECKPOINT_FILENAME = "epoch=49-step=52900.ckpt"
    # elif quantization_config_name == "all_S16S16":
    #     CHECKPOINT_FILENAME = "epoch=24-step=26450.ckpt"
    # elif quantization_config_name == "encdec_S16S16_tcn_S8S8":
    #     CHECKPOINT_FILENAME = "epoch=26-step=28566.ckpt"
    # elif quantization_config_name == "encdec_S16S16_tcn_S4S4":
    #     CHECKPOINT_FILENAME = "epoch=56-step=60306.ckpt"
    else:
        CHECKPOINT_FILENAME = None

    if CHECKPOINT_FILENAME is not None:
        print(f"Using checkpoint: {CHECKPOINT_FILENAME}")

    # Note: we should not compile the model prior to quantization
    # pytorch lightning will compile the model
    model = load_trained_model(conf=conf, eval=False, compile=False)

    # The quantized model will be saved in a new directory: ConvTasNet_<quantization_config_name>/
    exp_dir = conf.exp_dir
    if exp_dir[-1] == "/":
        exp_dir = exp_dir[:-1]
    exp_dir = exp_dir + f"_{quantization_config_name}/"
    conf.exp_dir = exp_dir
    conf_data["exp_dir"] = exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    print("Experiment directory:", exp_dir)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf_data, outfile)

    # if an exported model exists, we rename it
    export_path = os.path.join(exp_dir, "compiled_model.pth")
    if os.path.isfile(export_path):
        export_path_old = os.path.join(exp_dir, "compiled_model_old.pth")
        os.rename(export_path, export_path_old)

    quantized_model = quantize_model(
        model,
        config_name=quantization_config_name,
        example_args=(
            torch.rand(
                conf.training.batch_size, conf.data.sample_rate * conf.data.segment
            ),
        ),
    )

    transforms_utils.print_quantization_summary(quantized_model)

    print(quantized_model)

    train_loader, val_loader = get_data_loaders(
        conf=conf,
        # load_mini=True,
    )

    optimizer = Adam(
        quantized_model.parameters(),
        # TODO: parametrize
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
    module = ConvTasNetLightningModule(
        model=quantized_model,
        loss=pit_scale_invariant_signal_distortion_ratio,
        optimizer=optimizer,
        scheduler=scheduler,
    )

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

    # callbacks.append(MyCallback())

    trainer = pl.Trainer(
        # TODO: parametrize
        max_epochs=conf.quantize.epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        # TODO: change to train the full model
        # limit_train_batches=30,
        # limit_val_batches=30,
        # limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        # It may slow down training to log on every single batch. By default, Lightning logs every 50 rows, or 50 training steps
        # log_every_n_steps=2,
    )

    start = datetime.now()
    if CHECKPOINT_FILENAME:
        trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=f"{conf.exp_dir}checkpoints/{CHECKPOINT_FILENAME}",
        )
    else:
        trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    end = datetime.now()

    export_path = os.path.join(exp_dir, "compiled_model.pth")
    torch.save(module.model.state_dict(), export_path)

    stats_path = os.path.join(exp_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Training time: {end - start}\n")

    print(f"Training time: {end - start}")
    print(f"Compiled model saved to {export_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    model_archi = sys.argv[1]
    print(f"Using model architecture: {model_archi}")

    quantize_config_name = sys.argv[2]
    validate_quantization_config_names([quantize_config_name])
    print(f"Using quantization config: {quantize_config_name}")

    CONF_PATH = f"conf_{model_archi}.yml"
    with open(CONF_PATH) as f:
        conf_data = yaml.safe_load(f)

    conf = Conf(conf_data)

    train(
        conf=conf,
        model_archi=model_archi,
        quantization_config_name=quantize_config_name,
    )
