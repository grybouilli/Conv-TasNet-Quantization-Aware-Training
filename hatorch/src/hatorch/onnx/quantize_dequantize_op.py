import torch
from onnx import TensorProto
from onnxscript import FLOAT
from onnxscript import opset23 as op


@torch.library.custom_op("hatorch::quantize_dequantize", mutates_args=())
def quantize_dequantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    axis: int = 1,
    output_dtype: int = int(TensorProto.INT8),
) -> torch.Tensor:
    """
    Quantize Dequantize (QDQ) operator for ONNX export.

    See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#onnx-quantization-representation-format

    Note: conventions are not the same as Hatorch:
     - https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-23
     - https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-23

    Dequantization: $ s_"hatorch" q + z_"hatorch" = s_"onnx" (q - z_"onnx") $

    Attributes:
        input: the quantized input tensor to be quantized and dequantized
        scale: the scale tensor for quantization
        zero_point: the zero point tensor for quantization (optional, a zero zero point will be used if not provided)
        axis: the axis along which to apply quantization (default: 1)
        output_dtype: the desired output data type after dequantization (default: int8)
            Must correspond to a type from `onnx.TensorProto`
    """

    quantized = (
        torch.round(input / scale) + zero_point
        if zero_point is not None
        else torch.round(input / scale)
    )
    return (
        (quantized - zero_point) * scale
        if zero_point is not None
        else quantized * scale
    )


@quantize_dequantize.register_fake
def _quantize_dequantize_fake(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int = 1,
    output_dtype: int = int(TensorProto.INT8),
):
    return torch.empty_like(input)


def onnx_quantize_dequantize(
    x: FLOAT,
    scale: FLOAT,
    zero_point: FLOAT | None,
    axis: int = 1,
    output_dtype: int = int(TensorProto.INT8),
) -> FLOAT:
    """
    ONNX QDQ operator. Should be passed with `custom_translation_table` argument to `torch.onnx.export`

    For example:
    ```python
    torch.onnx.export(
        quantized_model,
        example_inputs,
        export_path,
        dynamo=True,
        opset_version=23,
        input_names=["input"],
        output_names=["output"],
        custom_translation_table={
            torch.ops.hatorch.quantize_dequantize.default: onnx_quantize_dequantize,
        },
    )
    ```
    """

    if zero_point is not None:
        # Should we use a op.Constant(value_ints=
        casted_zero_point = op.Cast(
            zero_point,
            to=output_dtype,
        )
    else:
        casted_zero_point = None

    q = op.QuantizeLinear(
        x,
        scale,
        casted_zero_point,
        axis=axis,
        output_dtype=output_dtype,
    )

    dq = op.DequantizeLinear(
        q,
        scale,
        casted_zero_point,
        axis=axis,
        output_dtype=int(TensorProto.FLOAT),
    )

    return dq
