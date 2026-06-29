from typing import Generic, TypeVar

import torch
from hatorch.modules.dequantize import Dequantize
from hatorch.modules.hatorch_quantized_layer import (
    ForwardMode,
    HatorchExportableModule,
    HatorchQuantizedLayer,
)
from hatorch.modules.quantize import Quantize
from hatorch.modules.quantized_tensor import QuantTensor

LayerType = TypeVar("LayerType", bound=HatorchQuantizedLayer)


class QuantizeWrapper(Generic[LayerType], HatorchExportableModule):
    def __init__(self, module: LayerType):
        super().__init__()

        self.quantize = Quantize()
        self.module = module
        self.dequantize = Dequantize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: maybe replace assert with type casting
        if self.forward_mode == ForwardMode.ONNX:
            assert isinstance(x, torch.Tensor), (
                "ONNX forward mode requires Tensor input"
            )
            return self.forward_onnx(x)
        elif self.forward_mode == ForwardMode.HATORCH:
            assert isinstance(x, torch.Tensor), (
                "QuantTensor forward mode requires Tensor input"
            )
            return self.forward_hatorch(x)
        else:
            raise ValueError(f"Unsupported forward mode: {self.forward_mode}")

    def forward_hatorch(self, x: torch.Tensor) -> torch.Tensor:

        x: QuantTensor = self.quantize(x)
        x: QuantTensor = self.module(x)
        output: torch.Tensor = self.dequantize(x)

        return output

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        # The Quantize operation consist in transforming the input tensor into a QuantTensor. There is no quantization involved.
        # The input will be quantized by the first activation of module layers
        return self.module(x)
