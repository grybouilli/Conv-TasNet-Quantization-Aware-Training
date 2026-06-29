import pytest

from hatorch.train import TrainingArguments


def test_outpur_dir():
    args = TrainingArguments()

    assert args.output_dir == "trainer_output"
