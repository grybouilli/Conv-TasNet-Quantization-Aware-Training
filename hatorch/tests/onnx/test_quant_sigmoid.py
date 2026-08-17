import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from hatorch.modules.dequantize import Dequantize
from hatorch.modules.hatorch_quantized_layer import ForwardMode, HatorchQuantizedLayer
from hatorch.modules.quantize import Quantize
from hatorch.modules.quantize_wrapper import QuantizeWrapper
from hatorch.quantizers import ScaleApproximation
from hatorch.transforms.config import (
    LayerQuantConfig,
    ModelQuantConfig,
    create_model_config_with_edge_layers,
    mixed_precision_config,
    uniform_8bit_config,
)
from hatorch.transforms.utils import quantize_model


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Sigmoid()

    def forward(self, x):
        return self.conv1d(x)


@pytest.mark.parametrize(
    "bit_width,signed,per_channel,atol,rtol",
    [
        (4, True, True, 1e-4, 1e-6),
        (8, True, True, None, None),
        (16, True, True, None, None),
        (8, False, True, None, None),
        (8, False, False, None, None),
    ],
)
def test_quant_conv1d(
    bit_width: int,
    signed: bool,
    per_channel: bool,
    atol: float | None,
    rtol: float | None,
):
    config: LayerQuantConfig = mixed_precision_config(
        quantize_activations=True,
        quantize_weights=True,
        quantize_output=True,
        weight_bits=bit_width,
        activation_bits=bit_width,
        output_bits=bit_width,
        quantize_bias=False,
        bias_bits=32,
        affine_activations=False,
        affine_output=False,
        per_channel_weights=per_channel,
        scale_approximation=ScaleApproximation.NONE,
        fixed_point_bits=8,
        signed_weights=signed,
        signed_activations=signed,
        signed_output=signed,
    )
    model_config = ModelQuantConfig(
        default_config=config,
    )

    model = Model()
    quant_model = quantize_model(
        model,
        model_config,
        use_wrapper=True,
    )

    quant_model.eval()
    model.eval()
    x = torch.rand(1, 10, 10)

    output_full_precision = model(x)
    print("Output full precision:", output_full_precision.shape)
    print(output_full_precision)

    outputs_hatorch = quant_model(x)
    print("Output Hatorch:", outputs_hatorch.shape)
    print(outputs_hatorch)

    with pytest.raises(AssertionError, match=r"Tensor-likes are not close!"):
        torch.testing.assert_close(outputs_hatorch, output_full_precision)

    example_inputs = (x,)
    export_path = f"test_model.onnx"

    quant_model.onnx_export(
        example_inputs=example_inputs,
        export_path=export_path,
    )

    sess_options = ort.SessionOptions()
    # It fail with ORT_ENABLE_EXTENDED
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    sess_options.optimized_model_filepath = "optimized.onnx"

    ort_sess = ort.InferenceSession(
        export_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    x_np = x.numpy()

    outputs = ort_sess.run(
        None,
        {
            "input": x_np,
        },
    )

    print("Output ONNX:", outputs[0].shape)
    print(outputs[0])

    # Relative difference between the two outputs
    relative_diff = torch.abs(outputs_hatorch - torch.from_numpy(outputs[0])) / (torch.abs(outputs_hatorch) + 1e-8)
    print(f"Relative difference between hatorch and onnx inference: {relative_diff.mean()}")

    torch.testing.assert_close(
        outputs_hatorch, torch.from_numpy(outputs[0]), atol=atol, rtol=rtol
    )
