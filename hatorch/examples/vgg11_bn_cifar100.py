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

EPOCHS = 1
DEVICE = "cuda"
LR = 0.005
BATCH_SIZE = 128
LSQ_BITS = 2
NU_STEPS = torch.tensor([-20, -13, -8, -6, -5, -3, -2, -1, 0, 1, 2, 4, 5, 7, 12, 19], dtype=torch.float).to(DEVICE)

# First and last layer configuration
FIRST_LAYER_WEIGHT_BITS = 4  # Higher precision for first layer weights
FIRST_LAYER_ACTIVATION_BITS = 4  # Higher precision for first layer activations
LAST_LAYER_WEIGHT_BITS = 4  # Higher precision for last layer weights
LAST_LAYER_ACTIVATION_BITS = 4  # Higher precision for last layer activations
OUTPUT_DIR = "examples_log/VGG/11BNQATTEST"
# Set to a checkpoint path or directory to resume training; keep as None to start fresh.
RESUME_FROM = None

def main():
    # Set seed BEFORE any model operations
    enable_full_determinism(42)

    model = VGG(True, 11, DatasetName.CIFAR100, BATCH_SIZE, 0, pretrained=True)
    print(model.model)
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
        # Scale approximation options:
        # - ScaleApproximation.NONE: Exact scales (default, best accuracy)
        # - ScaleApproximation.POWER_OF_TWO: Round to power of 2 (hardware-friendly)
        # - ScaleApproximation.FIXED_POINT: Fixed-point with configurable precision
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=42,  # Only used if scale_approximation=FIXED_POINT
    )

    print(sum(p.numel() for p in model.model.parameters()))


    transforms_utils.quantize_model(model.model, model_config, inplace=True)
    # transforms_utils.print_quantization_summary(model.model)

    # Move model to device before calibration
    # model.to_device(DEVICE)

    # print(model.model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, EPOCHS)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_epochs=EPOCHS,
        save_strategy=SaveStrategy.EPOCH,
        overwrite_output_dir=False,
        amp=False,
        tqdm_steps=50,
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

    trainer.load_network("/home/smith/PycharmProjects/hatorch/examples/examples_log/VGG/11BNQATTEST/model-2.pt")

    resume_from_checkpoint = RESUME_FROM
    # if resume_from_checkpoint:
    #     print(f"Resuming training from {resume_from_checkpoint}")
    # else:
    #     # LSQ+ requires a few calibration epochs to initialize scale and zero-point parameters
    #     # The calibration is ultra sensitive to learning rate at very low bit-widths
    #     # Prefer a very low LR for very low bit-width calibration if training diverges
    #     #optimizer_calib = torch.optim.SGD(model.model.parameters(), lr=0.0001, momentum=0.9)
    #     #trainer.calibrate(optimizer_calib, num_epochs=1, tqdm_steps=50)

    # Execute training (creates epoch checkpoints in OUTPUT_DIR)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.evaluate()


if __name__ == "__main__":
    main()
