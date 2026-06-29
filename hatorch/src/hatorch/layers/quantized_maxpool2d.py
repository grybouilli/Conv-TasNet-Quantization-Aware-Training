from torch.nn import MaxPool2d

from hatorch.modules.quantized_tensor import QuantTensor


class QuantMaxPool2d(MaxPool2d):
    """
    Quantized version of nn.MaxPool2d.
    Handles QuantTensor inputs and outputs while performing max pooling.
    """
    def forward(self, x: QuantTensor) -> QuantTensor:
        x.value = super().forward(x.value)
        return x
