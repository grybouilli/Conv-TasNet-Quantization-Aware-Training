from hatorch.onnx.output_dtype_utils import get_output_dtype
from hatorch.onnx.quantize_dequantize_op import quantize_dequantize
from typing import Optional

import torch
from torch import Tensor

from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.common._common_quantizer import BaseLearnedQuantizer
from hatorch.quantizers.autograd.lsq_uniform_quantizer import LsqUniformQuantizer as LsqUniformAutograd
from hatorch.quantizers.common.scale_approximation import ScaleApproximation


class LsqUniformQuantizer(BaseLearnedQuantizer):
    def __init__(
        self,
        is_activation: bool = False,
        symmetric: bool = True,
        per_channel: bool = False,
        num_channels: Optional[int] = None,
        channels_axis: Optional[int] = None,
        input_dim: Optional[int] = None,
        bit_width: int = 8,
        signed: bool = True,
        enabled: bool = True,
        g: Optional[float] = None,
        auto_compute_g: bool = True,
        scale_approximation: ScaleApproximation = ScaleApproximation.NONE,
        fixed_point_bits: int = 8,
    ):
        if not isinstance(bit_width, int):
            raise ValueError("LsqUniformQuantizer requires an integer bit_width.")
        if signed:
            qn = -(2 ** (bit_width - 1))
            qp = 2 ** (bit_width - 1) - 1
        else:
            qn = 0
            qp = 2 ** bit_width - 1
        super().__init__(
            is_activation=is_activation,
            symmetric=symmetric,
            per_channel=per_channel,
            num_channels=num_channels,
            channels_axis=channels_axis,
            input_dim=input_dim,
            signed=signed,
            enabled=enabled,
            qn=qn,
            qp=qp,
            g=g,
            auto_compute_g=auto_compute_g,
            scale_trainable=True,
        )
        self.bit_width = bit_width
        self.scale_approximation = scale_approximation
        self.fixed_point_bits = fixed_point_bits
        self.is_non_uniform_bitwidth = False

    def scale_zp_initializer(
        self,
        mean_v: Tensor,
        std_v: Tensor,
        max_v: Tensor,
        min_v: Tensor,
    ) -> None:
        with torch.no_grad():
            if not self.is_activation:
                effective_level_count = 2 ** (self.bit_width - 1)
                init_scale = torch.max(abs(mean_v - 3 * std_v), abs(mean_v + 3 * std_v)) / effective_level_count
                init_scale = init_scale.clamp_min(1e-8)
            else:
                init_scale = (max_v - min_v) / (self.qp - self.qn)
                if not self.symmetric:
                    init_zp = min_v - self.qn * init_scale
                    self.zero_point.copy_(init_zp.to(self.zero_point.device))
            init_scale = init_scale.reshape(self.scale_and_zero_point_shape)
            self._set_positive_scale(init_scale.to(self.scale.device))

    def forward(self, x: Tensor, in_scale: Optional[Tensor] = None) -> QuantTensor:
        if not self.enabled:
            return self._disabled_quant_tensor(x)

        effective_scale = self.current_scale()
        effective_zero_point = self.zero_point
        fused_scales = False
        if in_scale is not None:
            detached_in_scale = in_scale.detach()
            is_identity_input_scale = detached_in_scale.numel() == 1 and float(detached_in_scale.item()) == 1.0
            if self._scale_initialized and not is_identity_input_scale:
                fused_scales = True
            else:
                x = x * in_scale

        if not bool(self._scale_initialized):
            if self.per_channel:
                reduce_dims = [i for i in range(x.ndim) if i != self.channels_axis]
                mean_v = x.mean(dim=reduce_dims, keepdim=False)
                # For an input of shape (1, C, 1), when reducing over dimensions [0, 2],
                # the resulting degrees of freedom is <= 0, which causes std to be NaN
                # To avoid this, we set correction=0
                std_v = x.std(dim=reduce_dims, keepdim=False, correction=0)
                max_v = x.amax(dim=reduce_dims, keepdim=False)
                min_v = x.amin(dim=reduce_dims, keepdim=False)
            else:
                mean_v = x.mean()
                std_v = x.std()
                max_v = x.max()
                min_v = x.min()
            self.scale_zp_initializer(mean_v, std_v, max_v, min_v)
            effective_scale = self.current_scale()
            effective_zero_point = self.zero_point

        self._maybe_initialize_g(x)
        g_scale = self.g.to(device=effective_scale.device, dtype=effective_scale.dtype)
        if (not self.symmetric) and self.is_activation and isinstance(self.zero_point, torch.nn.Parameter):
            g_zp = self.g.to(device=effective_scale.device, dtype=effective_scale.dtype)
        else:
            g_zp = effective_scale.new_tensor(1.0)

        quant_value, export_scale, export_zero_point = LsqUniformAutograd.apply(
            x,
            effective_scale,
            effective_zero_point,
            self.qn,
            self.qp,
            g_scale,
            g_zp,
            fused_scales,
            self.scale_approximation,
            self.fixed_point_bits if self.scale_approximation == ScaleApproximation.FIXED_POINT else None,
            in_scale,
            self.symmetric,
            True,
        )
        return QuantTensor(
            value=quant_value,
            scale=export_scale,
            zero_point=export_zero_point,
            bit_width=self.bit_width,
            signed=self.signed,
        )

    def extra_repr(self) -> str:
        return (
            f"bit_width={self.bit_width}, signed={self.signed}, symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}"
            + (f", num_channels={self.num_channels}" if self.per_channel else "")
            + f", qn={self.qn}, qp={self.qp}, {self._scale_repr()}, {self._zero_point_repr()}, "
            f"g={self.g}, learnable_zp={not self.symmetric}, enabled={self.enabled}"
        )
    
    def onnx_export_forward(self, x: torch.Tensor):
        if not self.enabled:
            return x
        
        if self.is_non_uniform_bitwidth:
            raise NotImplementedError("ONNX export for non-uniform bit width is not implemented.")

        # For per-tensor quantization, y_scale must be a scalar, otherwise it must be a 1-D tensor, with the length of the quantization axis
        # See https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-23
        if self.per_channel:
            scale = self.scale.reshape(self.num_channels)
            channels_axis = self.channels_axis
        else:
            assert self.scale.numel() == 1, "Expected single scale value for per-tensor quantization."
            # We reshape it to ensure it will be traited as a scalar in ONNX export
            scale = self.scale.reshape(())
            channels_axis = 1
        # If scale is a scalar, the `axis` attribute will be ignored

        # We need to compute the scale and zero point for ONNX conventions: s_onnx = s_hatorch, z_onnx = - z_hatorch / s_hatorch
        zero_point = None
        if not self.symmetric:
            zero_point = - self.zero_point / scale
        
        output_dtype = get_output_dtype(self.signed, self.bit_width)
        
        return quantize_dequantize(x, scale, zero_point, axis=channels_axis, output_dtype=output_dtype)

