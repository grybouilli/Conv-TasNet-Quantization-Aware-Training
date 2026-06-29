from enum import Enum
from typing import Generic, Self, TypeVar, overload

import torch
import torch.nn as nn
from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.onnx.quantize_dequantize_op import onnx_quantize_dequantize
from hatorch.transforms.config import LayerQuantConfig


class ForwardMode(Enum):
    HATORCH = "hatorch"
    ONNX = "onnx"


class HatorchExportableModule(nn.Module):
    def __init__(self):
        """
        An exportable and trainable hatorch module.
        """
        super().__init__()

        self.forward_mode: ForwardMode = ForwardMode.HATORCH
        """
        Forward mode for the layer. The default is ForwardMode.HATORCH, which uses the standard Hatorch forward pass.
        When set to ForwardMode.ONNX, the layer will use a custom forward pass designed for ONNX export.
        """

    def set_recursive_forward_mode(self, mode: ForwardMode):
        """
        Recursively set the forward mode for this layer and all submodules that are instances of HatorchQuantizedLayer.

        Args:
            mode: The ForwardMode to set for this layer and its submodules.
        """

        def _set_mode(module: nn.Module) -> None:
            if isinstance(module, HatorchExportableModule):
                module.forward_mode = mode

        self.apply(_set_mode)

    def onnx_export(
        self,
        example_inputs: tuple,
        export_path: str,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ):
        """
        Export the layer to ONNX format using the provided example inputs.

        Args:
            example_inputs: A tuple of example inputs to use for tracing the model during export.
            export_path: The file path where the ONNX model will be saved.
        """
        previous_mode = self.forward_mode
        self.set_recursive_forward_mode(ForwardMode.ONNX)
        torch.onnx.export(
            self,
            example_inputs,
            export_path,
            dynamo=True,
            opset_version=23,
            custom_translation_table={
                torch.ops.hatorch.quantize_dequantize.default: onnx_quantize_dequantize,
            },
            input_names=input_names if input_names is not None else ["input"],
            output_names=output_names if output_names is not None else ["output"],
        )
        self.set_recursive_forward_mode(previous_mode)


NonQuantizedModuleType = TypeVar("NonQuantizedModuleType", bound=nn.Module)


class HatorchQuantizedLayer(Generic[NonQuantizedModuleType], HatorchExportableModule):
    """
    Base class for quantized layers in Hatorch.
    """

    layer_non_quantized_class: type[NonQuantizedModuleType]

    @overload
    def forward(self, x: QuantTensor) -> QuantTensor: ...

    @overload
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @torch.compile(dynamic=True)
    def forward(self, x: QuantTensor | torch.Tensor) -> QuantTensor | torch.Tensor:
        # TODO: maybe replace assert with type casting
        if self.forward_mode == ForwardMode.ONNX:
            return self.forward_onnx(x)
        elif self.forward_mode == ForwardMode.HATORCH:
            return self.forward_hatorch(x)
        else:
            raise ValueError(f"Unsupported forward mode: {self.forward_mode}")

    def forward_hatorch(self, x: QuantTensor) -> QuantTensor:
        raise NotImplementedError("Hatorch forward not implemented for this layer")

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ONNX export not implemented for this layer")

    @classmethod
    def from_non_quantized_layer(
        cls,
        layer: NonQuantizedModuleType,
        layer_config: LayerQuantConfig,
    ) -> Self:
        """
        Create a quantized layer from a non-quantized layer using the provided configuration.

        The method except parameters to be copied from the non-quantized layer.

        Args:
            layer: The non-quantized layer to convert.
            weight_quantizer: The quantizer to use for the weights.
            activation_quantizer: The quantizer to use for the activations.
            bias_quantizer: The quantizer to use for the bias.
        """
        raise NotImplementedError(
            "from_non_quantized_layer not implemented for this layer"
        )
