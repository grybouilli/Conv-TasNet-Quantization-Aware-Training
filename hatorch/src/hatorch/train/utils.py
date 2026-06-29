from enum import Enum
import random
import os

import numpy as np
import torch


class EvaluationStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class LoggingStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SaveStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"
    BEST = "best"


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

