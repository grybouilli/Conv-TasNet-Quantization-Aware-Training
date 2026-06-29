import torch
from torch import Tensor

from hatorch.quantizers.common.scale_approximation import ScaleApproximation


class LsqUniformQuantizer:
    @staticmethod
    def gradscale(x: Tensor, scale: float | Tensor) -> Tensor:
        y_out = x
        y_grad = x * scale
        return (y_out - y_grad).detach() + y_grad

    @staticmethod
    def roundpass(x: Tensor) -> Tensor:
        y_out = x.round()
        y_grad = x
        return (y_out - y_grad).detach() + y_grad

    @staticmethod
    def _forward_with_real_backward(forward_tensor: Tensor, backward_tensor: Tensor) -> Tensor:
        return (forward_tensor - backward_tensor).detach() + backward_tensor

    @staticmethod
    def apply(
        x: Tensor,
        scale: Tensor,
        zp: Tensor,
        qn: int,
        qp: int,
        grad_scale_factor: float | Tensor = 1.0,
        grad_zp_factor: float | Tensor = 1.0,
        fused_scales: bool = False,
        scale_approx_type: ScaleApproximation = ScaleApproximation.NONE,
        approx_bits: int | None = None,
        in_scale: Tensor | None = None,
        symmetric: bool = True,
        return_metadata: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        r"""Uniform LSQ quantization with STE-style detach tricks."""

        # Apply LSQ grad scaling directly here (instead of external hooks).
        meta_scale = LsqUniformQuantizer.gradscale(scale, grad_scale_factor)
        meta_zp = LsqUniformQuantizer.gradscale(zp, grad_zp_factor)

        if fused_scales and in_scale is not None:
            scale = meta_scale / in_scale.view(-1)
            if not symmetric:
                zp = (meta_zp / meta_scale.view(-1)).detach()
            else:
                zp = meta_zp
        else:
            scale = meta_scale
            zp = meta_zp

        if fused_scales:
            # then this is an activation quantizer where scale is m=s1*s2/s3
            # pre-scaled zero-point in lsq class, just round to integer
            zp = zp.round()
            real_scaled = (x / scale) - zp
            if scale_approx_type == ScaleApproximation.NONE:
                scaled = real_scaled
            elif scale_approx_type == ScaleApproximation.FIXED_POINT:
                if approx_bits is None:
                    raise ValueError("approx_bits must be provided for FIXED_POINT scale approximation.")
                int_scale = torch.round((1 / scale) * float(1 << approx_bits)).to(torch.int64)  # int64 multipliers
                prod = (int_scale * x).to(torch.int64)  # convert to int64 for fixed-point arithmetic
                round_to_nearest = 1 << (approx_bits - 1)
                prod_rounded = prod + torch.sign(prod.float()).to(torch.int64) * round_to_nearest
                approx_scaled = ((prod_rounded >> approx_bits) - zp).float()
                scaled = LsqUniformQuantizer._forward_with_real_backward(approx_scaled, real_scaled)
            elif scale_approx_type == ScaleApproximation.POWER_OF_TWO:
                # Approximate each scale value to the nearest positive power of two.
                tiny = torch.finfo(scale.dtype).tiny
                scale_abs = scale.abs().clamp_min(tiny)
                nearest_exp = torch.round(torch.log2(scale_abs))
                approx_scaled = (x * torch.pow(2, -nearest_exp)) - zp
                scaled = LsqUniformQuantizer._forward_with_real_backward(approx_scaled, real_scaled)
            else:
                raise ValueError(f"Unsupported scale approximation type: {scale_approx_type}")
        else:
            scaled = (x - zp) / scale

        clipped = scaled.clamp(qn, qp)
        rounded = LsqUniformQuantizer.roundpass(clipped)
        if return_metadata:
            return rounded, meta_scale, meta_zp
        return rounded
