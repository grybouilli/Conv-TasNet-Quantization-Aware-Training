"""
Use Pytorch operation counters to compute the number of FLOPs, parameters and buffers of a model.

Usage:
    python src/compute_model_size.py

Examples:
    python src/compute_model_size.py
"""

import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.flop_counter import FlopCounterMode

from model import norm
from utils.conf import Conf
from utils.utils import construct_model
from utils.utils_quantize import validate_quantization_config_names


def get_flops(model, inp: torch.Tensor | tuple, with_backward=False):
    """
    Use PyTorch flop counter to compute the number of FLOPs for a given model and input.

    https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/

    Parameters
    ----------
    model : nn.Module
        The PyTorch model for which to compute FLOPs.
    inp : torch.Tensor or tuple
        The input tensor or a tuple representing the shape of the input tensor.
    with_backward : bool, optional
        Whether to include backward pass FLOPs in the computation. Default is False.
    """

    training = model.training
    model.eval()

    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops = flop_counter.get_total_flops()
    if training:
        model.train()
    return total_flops


for CONF in ("conf_0.5.yml", "conf_1.5.yml"):
    print()
    print()
    print("----------------------------------------------------------------")
    print(f"Configuration model: {CONF}")
    print("----------------------------------------------------------------")
    with open(CONF) as f:
        conf_data = yaml.safe_load(f)

    conf = Conf(conf_data)

    model = construct_model(conf=conf)

    print(
        f"Input for 1 sample of {conf.data.segment} seconds at {conf.data.sample_rate} Hz: {1, 1, conf.data.segment * conf.data.sample_rate}"
    )

    input_shape = (1, conf.data.segment * conf.data.sample_rate)

    total_flops_direct = get_flops(model, input_shape, with_backward=False)
    print(
        f"Total FLOPs (Direct): {total_flops_direct:,} ({total_flops_direct / 1e9:.3f} GFLOPs)"
    )

    input_shape = (1, 1, conf.data.segment * conf.data.sample_rate)
    x = torch.randn(input_shape)
    enc_flops = get_flops(model.encoder, x, with_backward=False)
    print(f"Estimated Encoder FLOPs: {enc_flops:,} ({enc_flops / 1e9:.3f} GFLOPs)")

    x_tcn = model.encoder(x)

    tcn_flops = get_flops(model.tcn, x_tcn, with_backward=False)
    print(f"Estimated TCN FLOPs: {tcn_flops:,} ({tcn_flops / 1e9:.3f} GFLOPs)")

    encdec_flops = total_flops_direct - tcn_flops

    all8_bits_flops = total_flops_direct * 8 * 8
    print(f"all_8bits: {all8_bits_flops:,} BitOps")

    all16_bits = total_flops_direct * 16 * 16
    print(f"all_16bits: {all16_bits:,} BitOps")

    tcn_8bits_encdec_16bits = tcn_flops * 8 * 8 + encdec_flops * 16 * 16
    print(f"tcn_8bits_encdec_16bits: {tcn_8bits_encdec_16bits:,} BitOps")

    tcn_4bits_encdec_16bits = tcn_flops * 4 * 4 + encdec_flops * 16 * 16
    print(f"tcn_4bits_encdec_16bits: {tcn_4bits_encdec_16bits:,} BitOps")

    print()

    total_nb_parameters = 0
    for base in [model.encoder, model.tcn, model.decoder]:
        nb = sum(p.nelement() for p in base.parameters())

        print(f"Total number of parameters for {base.__class__.__name__}: {nb}")
        total_nb_parameters += nb

    print(f"Total number of parameters for the entire model: {total_nb_parameters}")

    param_size = None
    for p in model.parameters():
        if param_size is None:
            param_size = p.element_size()
            print(
                f"Parameter: {p.shape}, NElement: {p.nelement()}, Size: {p.element_size()} bytes"
            )
        elif param_size != p.element_size():
            raise ValueError("All parameters should have the same element size.")

    print(
        f"Model total parameter size: {total_nb_parameters * param_size} bytes ({(total_nb_parameters * param_size) / 1e6:.3f} MB)"
    )

    print()

    print(f"Receptive field of the TCN: {model.receptive_field} samples")
    print(
        f"Receptive field of the TCN: {model.receptive_field / conf.data.sample_rate:.3f} seconds"
    )
