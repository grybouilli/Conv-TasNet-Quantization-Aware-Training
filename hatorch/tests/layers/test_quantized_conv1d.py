import torch
import torch.nn as nn
from hatorch.layers.quantized_conv1d import QuantConv1d
from hatorch.layers.quantized_convtranspose1d import QuantConvTranspose1d
from hatorch.layers.quantized_prelu import QuantPReLU
from hatorch.quantizers.base import IdentityQuantizer
from hatorch.quantizers.lsq import LsqQuantizer


def test_conv1d():
    x = torch.rand(6, 100, 500)
    quantizer = IdentityQuantizer()
    quant_x = quantizer(x)
    print("Quantized input:", quant_x)

    quant_conv1d = QuantConv1d(100, 2, kernel_size=3, bias=False)
    quant_output = quant_conv1d(quant_x)

    print("quant_output:", quant_output)

    conv1d = nn.Conv1d(100, 2, kernel_size=3, bias=False)
    output = conv1d(x)
    assert output.shape == quant_output.shape, (
        f"Expected output shape {output.shape}, got {quant_output.shape}"
    )


def test_conv1d_2():
    x = torch.rand(6, 1, 500)
    quantizer = IdentityQuantizer()
    quant_x = quantizer(x)
    print("Quantized input:", quant_x)

    quant_conv1d = QuantConv1d(1, 512, kernel_size=(16,), groups=1, bias=False)
    quant_output = quant_conv1d(quant_x)

    print("quant_output:", quant_output)

    conv1d = nn.Conv1d(1, 512, kernel_size=(16,), groups=1, bias=False)
    output = conv1d(x)
    assert output.shape == quant_output.shape, (
        f"Expected output shape {output.shape}, got {quant_output.shape}"
    )


def test_conv1d_other():
    x = torch.rand(6, 100, 500)
    quantizer = IdentityQuantizer()
    quant_x = quantizer(x)
    print("quant_x.scale.shape:", quant_x.scale.shape)
    quant_x.scale = torch.rand(100, 1, 1)  # Simulate per-channel scales for weights
    print("quant_x.scale.shape:", quant_x.scale.shape)
    print("Quantized input:", quant_x)

    quant_conv1d = QuantConv1d(100, 2, kernel_size=3, bias=False)
    quant_output = quant_conv1d(quant_x)

    print("quant_output:", quant_output)

    conv1d = nn.Conv1d(100, 2, kernel_size=3, bias=False)
    output = conv1d(x)
    assert output.shape == quant_output.shape, (
        f"Expected output shape {output.shape}, got {quant_output.shape}"
    )


if __name__ == "__main__":
    test_conv1d()
    test_conv1d_2()
    test_conv1d_other()
