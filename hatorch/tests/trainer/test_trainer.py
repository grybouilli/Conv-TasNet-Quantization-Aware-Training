import pytest
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hatorch.datasets import DatasetName
from hatorch.models import Model
from hatorch.train import (
    Trainer,
    TrainingArguments,
    LoggingStrategy,
    EvaluationStrategy,
    SaveStrategy,
)

TEST_DEVICE = "cpu"
INPUT_SIZE = (2, 2)
NUM_CLASSES = 3
NUM_EPOCHS = 1
BATCH_SIZE = 2
DATASET_SIZE = 4
SEED = 0xC0FFEE


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1)
        self.lin = torch.nn.Linear(4 * 2 * 2, NUM_CLASSES)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.lin(x)


class DummyModel(Model):
    name = "DummyModel"
    recommended_input_size = INPUT_SIZE
    compatible_datasets = [DatasetName.MNIST]
    default_dataset = DatasetName.MNIST

    def get_model(self, _pretrained: bool) -> torch.nn.Module:
        return _DummyModel()


def make_trainer(
    logging_strategy=LoggingStrategy.NO,
    eval_strategy=EvaluationStrategy.NO,
    save_strategy=SaveStrategy.NO,
    logging_steps=1,
    eval_steps=1,
    save_steps=1,
):
    inps = torch.ones((DATASET_SIZE, 1, *INPUT_SIZE), dtype=torch.float32)
    class_one = torch.tensor(1)
    tgts = class_one.repeat(DATASET_SIZE)
    dataset = TensorDataset(inps, tgts)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=0.1)

    args = TrainingArguments(
        output_dir="test_trainer",
        num_epochs=NUM_EPOCHS,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        overwrite_output_dir=True,
        full_determinism=True,
        seed=SEED,
    )

    return Trainer(
        model=model,
        device=TEST_DEVICE,
        train_loader=loader,
        valid_loader=loader,
        optimizer=optimizer,
        compute_loss_func=nn.CrossEntropyLoss(),
        args=args,
    )


def test_trainer_state():
    trainer = make_trainer()
    trainer.train()
    assert trainer.state.global_step == 2  # DATASET_SIZE / BATCH_SIZE
    assert trainer.state.epoch == NUM_EPOCHS
    assert trainer.state.best_metric is None
    assert trainer.state.best_global_step is None
    assert trainer.state.log_history == []


def test_eval_every_step():
    trainer = make_trainer(
        eval_strategy=EvaluationStrategy.STEPS,
        eval_steps=1,
    )
    trainer.train()
    assert trainer.state.log_history == []
    assert math.isclose(
        trainer.state.best_metric,
        0.17028550803661346,
        rel_tol=1e-6,  # relative tolerance
    )
    assert trainer.state.best_global_step == 1


def test_log_every_step():
    trainer = make_trainer(
        logging_strategy=LoggingStrategy.STEPS,
        eval_strategy=EvaluationStrategy.STEPS,
        eval_steps=1,
        logging_steps=1,
    )
    trainer.train()

    # Logging
    assert len(trainer.state.log_history) == 2

    first_step = trainer.state.log_history[0]
    assert math.isclose(
        first_step["Evaluation/Top1"],
        100.0,
        rel_tol=1e-6,  # relative tolerance
    )
    assert math.isclose(
        first_step["Evaluation/Top5"],
        100.0,
        rel_tol=1e-6,  # relative tolerance
    )
    assert math.isclose(
        first_step["Training/Loss"],
        1.59091055393219,
        rel_tol=1e-6,  # relative tolerance
    )
    assert math.isclose(
        first_step["Evaluation/Loss"],
        0.23174569010734558,
        rel_tol=1e-6,  # relative tolerance
    )

    second_step = trainer.state.log_history[1]
    assert math.isclose(
        second_step["Evaluation/Top1"],
        100.0,
        rel_tol=1e-6,  # relative tolerance
    )
    assert math.isclose(
        second_step["Evaluation/Top5"],
        100.0,
        rel_tol=1e-6,  # relative tolerance
    )
    assert math.isclose(
        second_step["Training/Loss"],
        0.23174569010734558,
        rel_tol=1e-6,  # relative tolerance
    )
    assert math.isclose(
        second_step["Evaluation/Loss"],
        0.12311551719903946,
        rel_tol=1e-6,  # relative tolerance
    )

    # Evaluation
    assert math.isclose(
        trainer.state.best_metric,
        0.12311551719903946,
        rel_tol=1e-6,  # relative tolerance
    )
    assert trainer.state.best_global_step == 1
