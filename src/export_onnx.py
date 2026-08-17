"""
Export an ONNX model for a given model architecture and quantization
configuration.

Usage:
    python src/export_onnx.py <model_architecture> <quantization_config> [input_type]

Arguments:
    model_architecture:
        Model architecture identifier (e.g. ``0.5``, ``1.5``).

    quantization_config:
        Name of the quantization configuration to apply (e.g. ``all_S8S8``, ``all_S16S16``...).

    input_type:
        Input used during export. Supported values are:
        - ``random`` (default)
        - ``librimix``
        - ``minilibrimix``

If ``input_type`` is omitted, a random input is used.

When ``librimix`` or ``minilibrimix`` is specified, the exported ONNX model is
evaluated against the corresponding PyTorch model, and the SI-SDR between their
outputs is computed to verify the export.

Examples:
    python src/export_onnx.py 0.5 full_precision
    python src/export_onnx.py 0.5 all_S8S8
    python src/export_onnx.py 0.5 all_S8S8 librimix
"""

import os
import sys

import soundfile as sf
import torch
import torch.nn as nn
import yaml
from hatorch.modules.hatorch_quantized_layer import ForwardMode
from hatorch.onnx.quantize_dequantize_op import onnx_quantize_dequantize
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_distortion_ratio,
)

from utils.conf import Conf
from utils.utils import (
    construct_model,
    get_data_loaders,
    get_test_data_loaders,
    load_trained_model,
    rms_normalize,
)
from utils.utils_quantize import validate_quantization_config_names

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

model_archi = sys.argv[1]
print(f"Using model architecture: {model_archi}")

quantize_config_name = sys.argv[2]
validate_quantization_config_names([quantize_config_name], accept_full_precision=True)
print(f"Using quantization config: {quantize_config_name}")

if len(sys.argv) >= 4:
    if sys.argv[3] not in ["librimix", "minilibrimix", "random"]:
        raise ValueError(
            "Invalid input type. Must be one of: librimix, minilibrimix, random"
        )
    input_type = sys.argv[3]
else:
    input_type = "random"

if quantize_config_name == "full_precision":
    PATH = f"exp/ConvTasNet_{model_archi}/conf.yml"
else:
    PATH = f"exp/ConvTasNet_{model_archi}_{quantize_config_name}/conf.yml"

with open(PATH) as f:
    conf_data = yaml.safe_load(f)

conf = Conf(conf_data)

model = load_trained_model(
    conf=conf,
    eval=False,
    compile=False,
    config_name=quantize_config_name
    if quantize_config_name != "full_precision"
    else None,
)

if input_type == "random":
    inputs = torch.rand(1, conf.data.sample_rate * conf.data.segment)
elif input_type == "librimix":
    train_loader, val_loader = get_data_loaders(
        conf=conf, load_mini=False, force_batch_size=1
    )
    inputs, targets = next(iter(val_loader))
elif input_type == "minilibrimix":
    train_loader, val_loader = get_data_loaders(
        conf=conf, load_mini=True, force_batch_size=1
    )
    inputs, targets = next(iter(val_loader))

example_inputs = (inputs,)
export_path = f"model_{model_archi}_{quantize_config_name}.onnx"
export_path = os.path.join(conf.exp_dir, export_path)
export_audios_path = os.path.join(conf.exp_dir, "onnx_export_audios")


model.eval()

if quantize_config_name != "full_precision":
    model.onnx_export(
        example_inputs=example_inputs,
        export_path=export_path,
        input_names=["input"],
        output_names=["output"],
    )
else:
    torch.onnx.export(
        model,
        example_inputs,
        export_path,
        dynamo=True,
        opset_version=23,
        input_names=["input"],
        output_names=["output"],
        external_data=False,
    )


print("Model exported to:", export_path)


if input_type in ["librimix", "minilibrimix"]:
    print("Evaluating exported ONNX model against HATorch model...")

    hatorch_output = model(inputs)
    print("HATorch output:")
    print(hatorch_output)

    sisdr, _ = permutation_invariant_training(
        hatorch_output,
        targets,
        scale_invariant_signal_distortion_ratio,
        eval_func="max",
    )
    print(f"SI-SDR between hatorch output and targets: {sisdr.mean()}")

    print()

    import onnxruntime as ort

    sess_options = ort.SessionOptions()

    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )  # ORT_ENABLE_ALL

    sess_options.enable_profiling = True

    ort_sess = ort.InferenceSession(
        export_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    print("Providers:", ort_sess.get_providers())

    inputs_numpy = inputs.numpy()

    onnx_outputs = ort_sess.run(
        None,
        {
            "input": inputs_numpy,
        },
    )

    profile_file = ort_sess.end_profiling()

    print("profile_file:", profile_file)

    onnx_output = onnx_outputs[0]

    print("Output ONNX:", onnx_output.shape)

    print(onnx_output)

    onnx_output_torch = torch.from_numpy(onnx_output)

    sisdr, _ = permutation_invariant_training(
        onnx_output_torch,
        targets,
        scale_invariant_signal_distortion_ratio,
        eval_func="max",
    )

    print(f"SI-SDR between onnx output and targets: {sisdr.mean()}")

    sisdr, _ = permutation_invariant_training(
        onnx_output_torch,
        hatorch_output,
        scale_invariant_signal_distortion_ratio,
        eval_func="max",
    )
    print(f"SI-SDR between hatorch and onnx inference: {sisdr.mean()}")

    # Relative difference between the two outputs
    relative_diff = torch.abs(hatorch_output - onnx_output_torch) / (
        torch.abs(hatorch_output) + 1e-8
    )
    print(
        f"Relative difference between hatorch and onnx inference: {relative_diff.mean()}"
    )

    os.makedirs(export_audios_path, exist_ok=True)

    with torch.no_grad():
        for name, est in [("hatorch", hatorch_output), ("ONNX", onnx_output_torch)]:
            est_target_1 = est[:, 0, :]
            est_target_2 = est[:, 1, :]

            # linear rms level and scaling factor
            est_target_1 = rms_normalize(est_target_1, rms_level_db=-20.0)
            est_target_2 = rms_normalize(est_target_2, rms_level_db=-20.0)

            path = os.path.join(
                export_audios_path,
                f"{model_archi}_{quantize_config_name}_{name}_est_target_1.flac",
            )
            sf.write(path, est_target_1.view(-1).cpu(), conf.data.sample_rate)

            path = os.path.join(
                export_audios_path,
                f"{model_archi}_{quantize_config_name}_{name}_est_target_2.flac",
            )
            sf.write(path, est_target_2.view(-1).cpu(), conf.data.sample_rate)

    try:
        torch.testing.assert_close(
            hatorch_output, onnx_output_torch, atol=1e-4, rtol=1e-6
        )
    except AssertionError as e:
        print(e)
