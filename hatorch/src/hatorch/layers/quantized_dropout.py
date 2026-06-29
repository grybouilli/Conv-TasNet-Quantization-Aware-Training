from typing import Union
import torch
from torch.nn import Dropout

from hatorch.modules.quantized_tensor import QuantTensor


class QuantDropout(Dropout):
    """
    Quantized version of nn.Dropout.
    Handles QuantTensor inputs and outputs while performing dropout.
    """
    def forward(self, x: Union[QuantTensor, torch.Tensor]) -> QuantTensor:
        # Allow both QuantTensor and plain Tensor for compatibility with legacy models
        if isinstance(x, QuantTensor):
            quant_x = x
        else:
            quant_x = QuantTensor(
                value=x,
                scale=torch.ones(1, device=x.device, dtype=x.dtype),
                zero_point=torch.zeros(1, device=x.device, dtype=x.dtype),
            )

        quant_x.value = super().forward(quant_x.value)
        return quant_x
