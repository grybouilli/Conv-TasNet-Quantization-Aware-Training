from hatorch.modules.quantized_tensor import QuantTensor
from torch import Tensor
from torch.nn import Module


class Quantize(Module):
    """
    Quantization layer that converts a regular PyTorch tensor to a QuantTensor.

    This layer is typically placed at the beginning of a quantized network to convert
    the input tensor into a QuantTensor with identity quantization (scale=1, zero_point=0).
    This allows the network to properly propagate quantization information through
    subsequent layers.
    """

    def forward(self, x: Tensor) -> QuantTensor:
        """
        Convert a regular tensor to a QuantTensor with identity quantization.

        Args:
            x: Regular PyTorch tensor input

        Returns:
            QuantTensor with scale=1 and zero_point=0 (identity quantization)
        """
        return QuantTensor(
            value=x,
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=None,
        )
