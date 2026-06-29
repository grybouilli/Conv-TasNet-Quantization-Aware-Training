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
    mixed_precision_config,
)


class QuantModel(HatorchQuantizedLayer):
    def __init__(self, config: LayerQuantConfig):
        super().__init__()
        self.activation_quantizer = config.activation_quantizer.create_quantizer()

    def forward_hatorch(self, x):
        x = self.activation_quantizer(x.dequantized_value)
        return x

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_quantizer.onnx_export_forward(x)


@pytest.mark.parametrize(
    "bit_width,signed,per_channel",
    [
        (4, True, True),
        (8, True, True),
        (16, True, True),
        (8, False, True),
        (8, False, False),
    ],
)
def test_qdq(bit_width: int, signed: bool, per_channel: bool):
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

    quant_model = QuantModel(config=config)
    quant_model = QuantizeWrapper(quant_model)

    quant_model.eval()
    x = torch.rand(1, 10, dtype=torch.float32)

    print("Input:", x.shape)
    print(x)

    outputs_hatorch = quant_model(x)
    print("Output Hatorch:", outputs_hatorch.shape)
    print(outputs_hatorch)

    example_inputs = (x,)
    export_path = f"test_model.onnx"

    quant_model.onnx_export(
        example_inputs=example_inputs,
        export_path=export_path,
    )

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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

    torch.testing.assert_close(outputs_hatorch, torch.from_numpy(outputs[0]))
