import torch
import torch.nn as nn
from hatorch.layers.quantized_conv1d import QuantConv1d
from hatorch.layers.quantized_convtranspose1d import QuantConvTranspose1d
from hatorch.layers.quantized_prelu import QuantPReLU
from hatorch.quantizers.base import IdentityQuantizer


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


def test_prelu():
    x = torch.rand(6, 1, 500)
    quantizer = IdentityQuantizer()
    quant_x = quantizer(x)
    print("Quantized input:", quant_x)

    quant_prelu = QuantPReLU()
    quant_output = quant_prelu(quant_x)

    print("quant_Output:", quant_output)

    prelu = nn.PReLU()
    output = prelu(x)
    assert output.shape == quant_output.shape, (
        f"Expected output shape {output.shape}, got {quant_output.shape}"
    )


# def test_quantized_convtranspose1d():
#     x = torch.rand(6, 1, 500)
#     quantizer = IdentityQuantizer()
#     quant_x = quantizer(x)
#     print("Quantized input:", quant_x)

#     quant_convtranspose1d = QuantConvTranspose1d(1, 512, kernel_size=(16,), groups=1, bias=False)
#     quant_output = quant_convtranspose1d(quant_x)


#     print("quant_output:", quant_output)

#     convtranspose1d = nn.ConvTranspose1d(1, 512, kernel_size=(16,), groups=1, bias=False)
#     output = convtranspose1d(x)
#     assert output.shape == quant_output.shape, f"Expected output shape {output.shape}, got {quant_output.shape}"


def test_quantized_convtranspose1d():
    x = torch.rand(12, 512, 14)
    quantizer = IdentityQuantizer()
    quant_x = quantizer(x)

    print("Quantized input:", quant_x)

    quant_convtranspose1d = QuantConvTranspose1d(
        512, 1, kernel_size=(16,), groups=1, bias=False, padding=0, dilation=1
    )
    print("quant_convtranspose1d.weight.shape:", quant_convtranspose1d.weight.shape)
    quant_output = quant_convtranspose1d(quant_x)

    print("quant_output:", quant_output)

    convtranspose1d = nn.ConvTranspose1d(
        512, 1, kernel_size=(16,), groups=1, bias=False, padding=0, dilation=1
    )
    output = convtranspose1d(x)
    assert output.shape == quant_output.shape, (
        f"Expected output shape {output.shape}, got {quant_output.shape}"
    )


if __name__ == "__main__":
    test_conv1d()
    test_conv1d_2()
    test_prelu()
    test_quantized_convtranspose1d()
