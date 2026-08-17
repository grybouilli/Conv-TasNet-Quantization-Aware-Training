"""
Export a set of ONNX models. Figures can be exported to svg using Netron (https://netron.app/) on a Chromium based browser.

Netron settings:
- Show Horizontal for the GlobalLayerNorm
- Hide weights
- Export as svg

Export done using Firefox result in padding issues in exported nodes

Usage:
    python src/export_onnx_figures.py
"""

import os

import torch
import torch.nn as nn
from hatorch.quantizers import ScaleApproximation
from hatorch.transforms import utils as transforms_utils
from hatorch.transforms.config import (
    LayerQuantConfig,
    ModelQuantConfig,
    mixed_precision_config,
)

from model import norm
from model.norm_quantized import QuantGlobalLayerNorm

FOLDER = "onnx_figures/"
os.makedirs(FOLDER, exist_ok=True)


def export_onnx_model(model, example_inputs_shape, filename):

    x = torch.rand(example_inputs_shape)
    example_inputs = (x,)

    onnx_program = torch.onnx.export(
        model,
        example_inputs,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=23,
    )
    if onnx_program is None:
        raise RuntimeError("ONNX export failed.")

    path = os.path.join(FOLDER, filename)
    onnx_program.save(path)
    print(f"ONNX export successful. Model saved: {path}")


class ModelLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(100)

    def forward(self, x):
        return self.norm(x)


export_onnx_model(ModelLN(), (100,), "layer_norm.onnx")


class ModelgLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gln = norm.GlobalLayerNorm(channel_size=100)

    def forward(self, x):
        return self.gln(x)


export_onnx_model(ModelgLN(), (6, 100, 10), "global_layer_norm.onnx")


class ModelConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=100,
            out_channels=100,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        return self.conv1d(x)


config_S8S8 = mixed_precision_config(
    quantize_activations=True,
    quantize_weights=True,
    quantize_output=True,
    weight_bits=8,
    activation_bits=8,
    output_bits=8,
    quantize_bias=False,
    bias_bits=32,
    affine_activations=False,
    affine_output=False,
    per_channel_weights=True,
    scale_approximation=ScaleApproximation.NONE,
    fixed_point_bits=8,
    signed_weights=True,
    signed_activations=True,
    signed_output=True,
)

model_config = ModelQuantConfig(
    default_config=config_S8S8,
    scale_approximation=ScaleApproximation.FIXED_POINT,
)

model_conv = ModelConv()

quantized_model_conv = transforms_utils.quantize_model(
    model_conv,
    model_config,
    use_wrapper=True,
)

x = torch.rand((6, 100, 10))
example_inputs = (x,)

quantized_model_conv.onnx_export(
    example_inputs=example_inputs,
    export_path=FOLDER + "quantized_conv.onnx",
    input_names=["input"],
    output_names=["output"],
)


print("Exporting figures in folder:", FOLDER)
