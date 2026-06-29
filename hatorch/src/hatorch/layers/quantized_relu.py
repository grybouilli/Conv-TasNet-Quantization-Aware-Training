from torch.nn import ReLU

from hatorch.modules.quantized_tensor import QuantTensor


class QuantReLU(ReLU):
    """
    Quantized version of nn.ReLU.
    Handles QuantTensor inputs and outputs while performing ReLU activation.
    """
    def forward(self, x: 'QuantTensor') -> 'QuantTensor':
        x.value = super().forward(x.value)
        return x
