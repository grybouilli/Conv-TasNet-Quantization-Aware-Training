"""untrained VGG11-CIFAR100 QAT with int weights & activations"""
import copy

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from hatorch.datasets import DatasetName
from hatorch.models import VGG
from hatorch.train import (
    Trainer,
    TrainingArguments,
    SaveStrategy,
)

from models import ResNet20, ResNet18

EPOCHS = 100
DEVICE = "cuda"
LR = 0.005
BATCH_SIZE = 128
LSQ_BITS = 3

# First and last layer configuration
FIRST_LAYER_WEIGHT_BITS = 4  # Higher precision for first layer weights
FIRST_LAYER_ACTIVATION_BITS = 4  # Higher precision for first layer activations
LAST_LAYER_WEIGHT_BITS = 4  # Higher precision for last layer weights
LAST_LAYER_ACTIVATION_BITS = 4  # Higher precision for last layer activations

def main():
    model = ResNet18(DatasetName.IMAGENET1K, batch_size=BATCH_SIZE, pretrained=True)
    train_loader, valid_loader = model.dataloader.get_train_valid_loader()

    # Move model to device before calibration
    model.to_device(DEVICE)

    # print(model.model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, EPOCHS)

    args = TrainingArguments(
        output_dir="examples_log/VGG/11BNQAT",
        num_epochs=EPOCHS,
        save_strategy=SaveStrategy.NO,
        overwrite_output_dir=True,
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
        keep_fixed_point=False
    )

    # Execute training
    trainer.evaluate()

if __name__ == "__main__":
    main()
