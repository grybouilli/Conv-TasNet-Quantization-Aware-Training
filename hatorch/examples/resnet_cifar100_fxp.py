"""untrained VGG11-CIFAR100 QAT with int weights & activations"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from hatorch.datasets import DatasetName
from hatorch.models import VGG
from hatorch.train import (
    Trainer,
    TrainingArguments,
    SaveStrategy,
)
from hatorch.train.utils import enable_full_determinism
from hatorch.transforms import utils as transforms_utils
from hatorch.transforms.config import create_model_config_with_edge_layers
from hatorch.quantizers.lsq import ScaleApproximation
from hatorch.models import ResNet20, ResNet56

EPOCHS = 40
EXTRA_EPOCHS = 10
DEVICE = "cuda"
LR = 0.0005
ETA_MIN = 0.00001
BATCH_SIZE = 128
LSQ_BITS = 4
NU_STEPS = torch.tensor([-10, -7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 11], dtype=torch.float).to(DEVICE)
OUTPUT_DIR = "RESNET20/INT4FX8"

# First and last layer configuration
FIRST_LAYER_WEIGHT_BITS = 8  # Higher precision for first layer weights
FIRST_LAYER_ACTIVATION_BITS = 8  # Higher precision for first layer activations
LAST_LAYER_WEIGHT_BITS = 8  # Higher precision for last layer weights
LAST_LAYER_ACTIVATION_BITS = 8  # Higher precision for last layer activations

def main():
    # Set seed BEFORE any model operations
    # enable_full_determinism(42)

    model = ResNet20(DatasetName.CIFAR100, BATCH_SIZE, 0, pretrained=True)
    train_loader, valid_loader = model.dataloader.get_train_valid_loader()

    # qat config - Option 1: Using the convenience function
    model_config = create_model_config_with_edge_layers(
        default_weight_bits=LSQ_BITS,
        default_activation_bits=LSQ_BITS,
        first_layer_weight_bits=FIRST_LAYER_WEIGHT_BITS,
        first_layer_activation_bits=FIRST_LAYER_ACTIVATION_BITS,
        last_layer_weight_bits=LAST_LAYER_WEIGHT_BITS,
        last_layer_activation_bits=LAST_LAYER_ACTIVATION_BITS,
        bias_bits=32,
        affine_activations=True,
        per_channel_weights=True,
        fold_batch_norm=True,
        # Scale approximation options:
        # - ScaleApproximation.NONE: Exact scales (default, best accuracy)
        # - ScaleApproximation.POWER_OF_TWO: Round to power of 2 (hardware-friendly)
        # - ScaleApproximation.FIXED_POINT: Fixed-point with configurable precision
        scale_approximation=ScaleApproximation.FIXED_POINT,
        fixed_point_bits=6,  # Only used if scale_approximation=FIXED_POINT
    )

    transforms_utils.quantize_model(model.model, model_config, inplace=True)
    # transforms_utils.print_quantization_summary(model.model)
    print(model.model)

    # Move model to device before calibration
    model.to_device(DEVICE)

    # print(model.model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    cosine_scheduler = CosineAnnealingLR(optimizer, EPOCHS, eta_min=ETA_MIN)
    constant_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.00001 / LR if LR != 0 else 0.0,
        end_factor=0.000001 / LR if LR != 0 else 0.0,
        total_iters=EXTRA_EPOCHS,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[cosine_scheduler, constant_scheduler],
        milestones=[EPOCHS],
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_epochs=EPOCHS + EXTRA_EPOCHS,
        save_strategy=SaveStrategy.BEST,
        overwrite_output_dir=True,
        amp=False,
        tqdm_steps=50,
        freeze_bn_stats_epoch=5,
    )

    trainer = Trainer(
        model,
        DEVICE,
        train_loader,
        valid_loader,
        optimizer,
        loss_fn,
        lr_scheduler=scheduler,
        args=args,
    )

    # LSQ+ requires a few calibration epochs to initialize scale and zero-point parameters
    # The calibration is ultra sensitive to learning rate at very low bit-widths
    # Prefer a very low LR for very low bit-width calibration if training diverges
    optimizer_calib = torch.optim.SGD(model.model.parameters(), lr=0.00001, momentum=0.9)
    trainer.calibrate(optimizer_calib, num_epochs=2, tqdm_steps=50)

    # Execute training
    trainer.train()
    trainer.evaluate()

    transforms_utils.set_scale_approximation(model.model, ScaleApproximation.FIXED_POINT, fixed_point_bits=32)
    trainer.evaluate()
    transforms_utils.set_scale_approximation(model.model, ScaleApproximation.FIXED_POINT, fixed_point_bits=16)
    trainer.evaluate()
    transforms_utils.set_scale_approximation(model.model, ScaleApproximation.FIXED_POINT, fixed_point_bits=8)
    trainer.evaluate()
    transforms_utils.set_scale_approximation(model.model, ScaleApproximation.FIXED_POINT, fixed_point_bits=7)
    trainer.evaluate()
    transforms_utils.set_scale_approximation(model.model, ScaleApproximation.FIXED_POINT, fixed_point_bits=6)
    trainer.evaluate()
    transforms_utils.set_scale_approximation(model.model, ScaleApproximation.FIXED_POINT, fixed_point_bits=5)
    trainer.evaluate()
    transforms_utils.set_scale_approximation(model.model, ScaleApproximation.FIXED_POINT, fixed_point_bits=4)
    trainer.evaluate()


if __name__ == "__main__":
    main()
