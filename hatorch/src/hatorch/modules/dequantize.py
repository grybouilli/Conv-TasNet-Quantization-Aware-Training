from hatorch.modules.quantized_tensor import QuantTensor
from torch import Tensor
from torch.nn import Module


class Dequantize(Module):
    """
    Dequantization layer that extracts the tensor value from a QuantTensor.

    This layer is typically placed at the end of a quantized network to convert
    the output QuantTensor back to a regular PyTorch tensor.
    """

    def forward(self, x: QuantTensor) -> Tensor:
        """
        Extract the dequantized value from a QuantTensor.

        Args:
            x: QuantTensor input

        Returns:
            Regular PyTorch tensor (the dequantized value)
        """

        return x.dequantized_value
