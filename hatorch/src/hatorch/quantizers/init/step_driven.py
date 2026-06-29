from typing import Optional

import torch
from torch import Tensor

from hatorch.modules.quantized_tensor import QuantTensor
from hatorch.quantizers.common._common_quantizer import BaseLearnedQuantizer
from hatorch.quantizers.autograd.lsq_step_driven_quantizer import LsqStepDrivenQuantizer


class StepDrivenQuantizer(BaseLearnedQuantizer):
    def __init__(
        self,
        is_activation: bool = False,
        symmetric: bool = True,
        per_channel: bool = False,
        num_channels: Optional[int] = None,
        bit_width: int | Tensor = 8,
        signed: bool = True,
        enabled: bool = True,
        g: Optional[float] = None,
        auto_compute_g: bool = True,
    ):
        if is_activation:
            raise ValueError("StepDrivenQuantizer is only supported for weight quantizers.")
        self.is_non_uniform_bitwidth = isinstance(bit_width, Tensor)
        if isinstance(bit_width, Tensor):
            qn = int(bit_width.min().item())
            qp = int(bit_width.max().item())
        else:
            raise ValueError("StepDrivenQuantizer requires Tensor codebook targets.")
        super().__init__(
            is_activation=is_activation,
            symmetric=symmetric,
            per_channel=per_channel,
            num_channels=num_channels,
            signed=signed,
            enabled=enabled,
            qn=qn,
            qp=qp,
            g=g,
            auto_compute_g=auto_compute_g,
            scale_trainable=True,
            with_zero_point=False,
        )

        self.register_buffer("bit_width", bit_width.detach().clone())

    def scale_initializer(self, x: Tensor) -> None:
        with torch.no_grad():
            codebook = self.bit_width.to(device=x.device, dtype=x.dtype)
            max_abs_code = codebook.abs().amax(dim=-1).clamp_min(1.0)
            if self.per_channel:
                dims = list(range(1, x.ndim))
                mean_v = x.mean(dim=dims, keepdim=False)
                std_v = x.std(dim=dims, keepdim=False)
            else:
                mean_v = x.mean()
                std_v = x.std()
            span = torch.maximum(
                (mean_v - 3 * std_v).abs(),
                (mean_v + 3 * std_v).abs(),
            )
            init_scale = (span / max_abs_code).clamp_min(1e-8)
            self._set_positive_scale(init_scale.to(self.scale.device, dtype=self.scale.dtype))

    def _current_codebook_targets(self) -> Tensor:
        return self.bit_width

    def forward(self, x: Tensor, in_scale: Optional[Tensor] = None) -> QuantTensor:
        if not self.enabled:
            return self._disabled_quant_tensor(x)
        if in_scale is not None:
            raise ValueError("StepDrivenQuantizer is only supported for weights and does not accept input scales.")

        if not bool(self._scale_initialized):
            self.scale_initializer(x)

        effective_scale = self.current_scale()
        self._maybe_initialize_g(x)
        g_scale = self.g.to(device=effective_scale.device, dtype=effective_scale.dtype)
        quant_value, export_scale = LsqStepDrivenQuantizer.apply(
            x,
            effective_scale,
            self.bit_width,
            int(self.qn),
            int(self.qp),
            g_scale,
            True,
        )
        return QuantTensor(
            value=quant_value,
            scale=export_scale,
            zero_point=None,
            bit_width=self.bit_width,
            signed=self.signed,
        )

    def extra_repr(self) -> str:
        return (
            f"bit_width=step_driven, signed={self.signed}, symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}"
            + (f", num_channels={self.num_channels}" if self.per_channel else "")
            + f", qn={self.qn}, qp={self.qp}, {self._scale_repr()}, enabled={self.enabled}"
        )
