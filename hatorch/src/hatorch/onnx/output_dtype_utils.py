from onnx import TensorProto

signed_types: dict[int, TensorProto.DataType] = {
    4: TensorProto.INT4,
    8: TensorProto.INT8,
    16: TensorProto.INT16,
}

non_signed_types: dict[int, TensorProto.DataType] = {
    4: TensorProto.UINT4,
    8: TensorProto.UINT8,
    16: TensorProto.UINT16,
}


def get_output_dtype(signed: bool, bit_width: int) -> int:
    """
    Compute `onnx.TensorProto` type for a given Hatorch QuantTensor configuration.
    """
    # TODO: complete this method
    if signed:
        if bit_width in signed_types:
            return int(signed_types[bit_width])
        raise ValueError(f"Unsupported bit width {bit_width} for signed type")

    if bit_width in non_signed_types:
        return int(non_signed_types[bit_width])
    raise ValueError(f"Unsupported bit width {bit_width} for non-signed type")
