"""
Print the quantized model architecture

Usage:
    python src/print_model.py <model_architecture> <quantization_config>

Arguments:
    model_architecture:
        Model architecture identifier (e.g. ``0.5``, ``1.5``).

    quantization_config:
        Name of the quantization configuration to apply (e.g. ``all_S8S8``, ``all_S16S16``...).

The number of QAT epochs is defined in the configuration file (conf_<model_architecture>.yml) under the key ``quantize.epochs``. The default value is 30.

Examples:
    python src/print_model.py 0.5 all_S8S8
"""

import os
import sys

import torch
import torch.nn as nn
import yaml

from utils.conf import Conf
from utils.utils import (
    construct_model,
    load_trained_model,
)
from utils.utils_quantize import quantize_model

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

model_archi = sys.argv[1]
config_name = sys.argv[2]

CONF_PATH = f"conf_{model_archi}.yml"

if __name__ == "__main__":
    with open(CONF_PATH) as f:
        conf_data = yaml.safe_load(f)

    conf = Conf(conf_data)

    model = construct_model(conf=conf)

    example_args = (
        torch.rand(1, conf.data.sample_rate * 3),
    )  # Example input for tracing
    if config_name is not None:
        model = quantize_model(
            model, config_name=config_name, example_args=example_args
        )

    print(model)

    res = model(*example_args)
    print(res)
