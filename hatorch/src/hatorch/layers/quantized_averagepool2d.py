from torch.nn import AvgPool2d

from hatorch.modules.quantized_tensor import QuantTensor


class QuantAveragePool2d(AvgPool2d):
    """
    Quantized version of nn.AvgPool2d.
    Handles QuantTensor inputs and outputs while performing average pooling.
    """
    def forward(self, x: QuantTensor) -> QuantTensor:
        x.value = super().forward(x.value)
        return x
