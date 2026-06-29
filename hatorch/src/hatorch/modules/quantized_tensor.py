"""
Quantized Tensor implementation inspired by Brevitas.

This module provides a QuantTensor class that encapsulates a quantized tensor
along with its quantization parameters (scale and zero-point).
"""

import functools
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

QUANT_TENSOR_FN_HANDLER = {}


def implements(torch_function):

    @functools.wraps(torch_function)
    def decorator(func):
        QUANT_TENSOR_FN_HANDLER[torch_function] = func
        return func

    return decorator


def quant_invariant_handler(fn, inp: "QuantTensor", *args, **kwargs) -> "QuantTensor":
    out_value = fn(inp.dequantized_value, *args, **kwargs)
    return QuantTensor(
        value=out_value,
        scale=None,
        zero_point=None,
        bit_width=None,
        signed=None,
    )


@implements(F.pad)
def pad_handler(*args, **kwargs):
    # TODO check padding value is legal
    return quant_invariant_handler(F.pad, *args, **kwargs)


@implements(torch.sqrt)
def sqrt_handler(*args, **kwargs):
    return quant_invariant_handler(torch.sqrt, *args, **kwargs)


@implements(torch.Tensor.mul)
def mul_handler(first, second):
    if isinstance(first, torch.Tensor):
        return second.__rmul__(first)
    return first.__mul__(second)


@implements(torch.Tensor.add)
def add_handler(first, second):
    if isinstance(first, torch.Tensor):
        return second.__add__(first)
    return first.__add__(second)


class QuantTensor:
    """
    A Brevitas-style quantized tensor that holds the quantized values along with
    scale and zero-point quantization parameters.

    Attributes:
        value (Tensor): The quantized tensor values (typically int representation).
        scale (Optional[Tensor]): The scaling factor(s) for dequantization.
        zero_point (Optional[Tensor]): The zero-point offset(s) for dequantization.
        bit_width (Optional[Tensor]): The bit width used for quantization.
        signed (Optional[bool]): Whether the quantized values are signed.

    A QuantTensor can store a _non quantized_ value. In this case, the scale and zero_point should be None.
    `bit_width` and `signed` should only be used if `value` respect defined bit width and signedness

    Scale and zero_point should have a shape compatible with the value tensor.
    For example, for per-tensor quantization of a (10, 10, 10) tensor, scale and zero_point should have shape (1, 1, 1)
    For per-channel quantization of a (10, 10, 10) tensor, on axis 1, scale and zero_point should have shape (1, 10, 1)

    To get the dequantized value, use the `dequantized_value` property, which will compute the dequantized value using the formula:
    dequantized_value = value * scale + zero_point
    """

    def __init__(
        self,
        value: Tensor,
        scale: Optional[Tensor],
        zero_point: Optional[Tensor] = None,
        bit_width: Optional[int | Tensor] = None,
        signed: Optional[bool] = True,
    ):
        """
        Initialize a QuantTensor.

        Args:
            value: The quantized tensor values.
            scale: The scaling factor for dequantization.
            zero_point: The zero-point offset.
            bit_width: The bit width used for quantization.
            signed: Whether the quantization is signed.
        """
        self.value = value
        self.scale = scale
        self.zero_point = zero_point
        self.bit_width = bit_width
        self.signed = signed

    @property
    def dequantized_value(self) -> Tensor:
        """
        Compute the dequantized value from the quantized representation.

        Returns:
            A tensor representing the dequantized values.
        """
        output = self.value
        if self.scale is not None:
            output = output * self.scale
        # if self.zero_point is not None:
        #     output = output + self.zero_point
        return output

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # Brevitas inspired
        # See https://github.com/docarray/notes/blob/main/blog/02-this-weeks-in-docarray-01.md#__torch_function__--or-how-to-give-pytorch-a-little-bit-more-confidence
        if kwargs is None:
            kwargs = {}
        if func in QUANT_TENSOR_FN_HANDLER:
            return QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)

        raise NotImplementedError(
            f"Function {func} not implemented for QuantTensor.", QUANT_TENSOR_FN_HANDLER
        )

    def detach(self) -> "QuantTensor":
        """
        Detach all tensors from the computation graph.

        Returns:
            A new QuantTensor with detached tensors.
        """
        return QuantTensor(
            value=self.value.detach(),
            scale=self.scale.detach() if self.scale is not None else None,
            zero_point=self.zero_point.detach()
            if self.zero_point is not None
            else None,
            bit_width=self.bit_width.detach()
            if isinstance(self.bit_width, torch.Tensor)
            else self.bit_width,
            signed=self.signed,
        )

    def clone(self) -> "QuantTensor":
        """
        Create a deep copy of this QuantTensor.

        Returns:
            A new QuantTensor with cloned tensors.
        """
        return QuantTensor(
            value=self.value.clone(),
            scale=self.scale.clone() if self.scale is not None else None,
            zero_point=self.zero_point.clone() if self.zero_point is not None else None,
            bit_width=self.bit_width.clone()
            if isinstance(self.bit_width, torch.Tensor)
            else self.bit_width,
            signed=self.signed,
        )

    def to(self, *args, **kwargs) -> "QuantTensor":
        """
        Move and/or cast the tensors to a specified device or dtype.

        Returns:
            A new QuantTensor with converted tensors.
        """
        return QuantTensor(
            value=self.value.to(*args, **kwargs),
            scale=self.scale.to(*args, **kwargs) if self.scale is not None else None,
            zero_point=self.zero_point.to(*args, **kwargs)
            if self.zero_point is not None
            else None,
            bit_width=self.bit_width.to(*args, **kwargs)
            if isinstance(self.bit_width, torch.Tensor)
            else self.bit_width,
            signed=self.signed,
        )

    def cpu(self) -> "QuantTensor":
        """Move all tensors to CPU."""
        return self.to("cpu")

    def cuda(self, device: Optional[Union[int, str]] = None) -> "QuantTensor":
        """Move all tensors to CUDA."""
        return self.to("cuda" if device is None else f"cuda:{device}")

    @property
    def device(self) -> torch.device:
        """Return the device of the value tensor."""
        return self.value.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the value tensor."""
        return self.value.dtype

    def type(
        self, dtype: None | str | torch.dtype = None, non_blocking: bool = False
    ) -> "str | QuantTensor":
        """
        Returns the type if dtype is not provided, else casts this object to the specified type.
        If this is already of the correct type, no copy is performed and the original object is returned.
        """
        if dtype is None:
            return self.value.type()

        casted_value = self.value.type(dtype, non_blocking)
        if casted_value is self.value:
            return self

        return QuantTensor(
            value=casted_value,
            scale=self.scale,
            zero_point=self.zero_point,
            bit_width=self.bit_width,
            signed=self.signed,
        )

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the value tensor."""
        return self.value.shape

    def dim(self) -> int:
        """Return the number of dimensions of the value tensor."""
        return self.value.dim()

    def unsqueeze(self, dim: int) -> "QuantTensor":
        """
        Returns a new tensor with a dimension of size one inserted at the specified position.
        """
        return QuantTensor(
            value=self.value.unsqueeze(dim),
            scale=self.scale.unsqueeze(dim) if self.scale is not None else None,
            zero_point=self.zero_point.unsqueeze(dim)
            if self.zero_point is not None
            else None,
            bit_width=self.bit_width,
            signed=self.signed,
        )

    def mean(
        self,
        *args,
        **kwargs,
    ) -> "QuantTensor":
        """
        Returns the mean value of each element in the input tensor across the specified dimensions.

        This method will integrate the scale and zero_point in the value of the output QuantTensor
        """

        return QuantTensor(
            value=self.dequantized_value.mean(*args, **kwargs),
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=None,
        )

    def var(
        self,
        *args,
        **kwargs,
    ) -> "QuantTensor":
        """
        Returns the variance of each element in the input tensor across the specified dimensions.

        This method will integrate the scale and zero_point in the value of the output QuantTensor
        """

        return QuantTensor(
            value=self.dequantized_value.var(*args, **kwargs),
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=None,
        )

    def size(self, dim: Optional[int] = None) -> torch.Size | int:
        """Mirror Tensor.size() on the wrapped value tensor."""
        return self.value.size(dim)

    def __neg__(self):
        return QuantTensor(
            value=-self.value,
            scale=self.scale,
            zero_point=-self.zero_point if self.zero_point is not None else None,
            bit_width=self.bit_width,
            signed=True,
        )

    def __add__(self, other):
        if isinstance(other, QuantTensor):
            return QuantTensor(
                value=self.dequantized_value + other.dequantized_value,
                scale=None,
                zero_point=None,
                bit_width=None,
                signed=None,
            )

        elif (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, Tensor)
        ):
            return QuantTensor(
                value=self.dequantized_value + other,
                scale=None,
                zero_point=None,
                bit_width=None,
                signed=None,
            )
        else:
            raise NotImplementedError(
                f"Addition not implemented between QuantTensor and {type(other)}"
            )

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, QuantTensor):
            # The scale and the zero_point will be integrated in the value of the output.
            # This should not be an issue as the output QuantTensor is expected to be dequantized before being passed to another Quantizer
            return QuantTensor(
                value=self.dequantized_value * other.dequantized_value,
                scale=None,
                zero_point=None,
                bit_width=None,
                signed=None,
            )
        elif (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, Tensor)
        ):
            return QuantTensor(
                value=self.dequantized_value * other,
                scale=None,
                zero_point=None,
                bit_width=None,
                signed=None,
            )

        raise NotImplementedError(
            f"Multiplication is not implemented between QuantTensor and {type(other)}"
        )

    def __rmul__(self, other):
        # Elementwise multiplication is commutative, so we can just call __mul__ with the arguments reversed.
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, QuantTensor):
            return QuantTensor(
                value=self.dequantized_value / other.dequantized_value,
                scale=None,
                zero_point=None,
                bit_width=None,
                signed=None,
            )
        elif (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, Tensor)
        ):
            return QuantTensor(
                value=self.dequantized_value / other,
                scale=None,
                zero_point=None,
                bit_width=None,
                signed=None,
            )
        else:
            raise NotImplementedError(
                f"Division not implemented between QuantTensor and {type(other)}"
            )

    def transpose(self, dim0: int, dim1: int) -> "QuantTensor":
        return QuantTensor(
            value=self.dequantized_value.transpose(dim0, dim1),
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=None,
        )

    def view(self, *size) -> "QuantTensor":
        """Return a view of the wrapped value tensor while preserving quantization metadata."""
        # TODO: It should be possible to view a QuantTensor without loosing the scale, zero_point, bit_width and signedness
        # we could store in the quant tensor the quantization axis,
        # then view the scale accordingly
        return QuantTensor(
            value=self.dequantized_value.view(*size),
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=None,
        )

    def reshape(self, *shape) -> "QuantTensor":
        """Return a reshaped wrapped value tensor while preserving quantization metadata."""
        # TODO: It should be possible to reshape a QuantTensor without loosing the scale, zero_point, bit_width and signedness
        # we could store in the quant tensor the quantization axis,
        # then reshape the scale accordingly
        return QuantTensor(
            value=self.dequantized_value.reshape(*shape),
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=None,
        )

    def __getitem__(self, index):
        return QuantTensor(
            value=self.dequantized_value[index],
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=None,
        )

    @property
    def requires_grad(self) -> bool:
        """Check if any tensor requires gradients."""
        return (
            self.value.requires_grad
            or (self.scale is not None and self.scale.requires_grad)
            or (self.zero_point is not None and self.zero_point.requires_grad)
        )

    def contiguous(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> "QuantTensor":
        return QuantTensor(
            value=self.value.contiguous(memory_format=memory_format),
            scale=self.scale.contiguous(memory_format=memory_format)
            if self.scale is not None
            else None,
            zero_point=self.zero_point.contiguous(memory_format=memory_format)
            if self.zero_point is not None
            else None,
            bit_width=self.bit_width,
            signed=self.signed,
        )

    def __repr__(self) -> str:
        """Return a string representation of the QuantTensor."""
        return (
            f"QuantTensor(value={self.value.shape}, "
            f"scale={self.scale.shape if self.scale is not None else None}, "
            f"zero_point={self.zero_point.shape if self.zero_point is not None else None}, "
            f"bit_width={self.bit_width.item() if isinstance(self.bit_width, Tensor) and self.bit_width.numel() == 1 else self.bit_width}, "
            f"signed={self.signed}, "
            f"device={self.device})"
        )

    def __str__(self) -> str:
        """Return a string representation of the QuantTensor."""
        return self.__repr__()
