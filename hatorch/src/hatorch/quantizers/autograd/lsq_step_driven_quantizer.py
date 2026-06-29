import torch
from torch import Tensor


class LsqStepDrivenQuantizer:
    @staticmethod
    def gradscale(x: Tensor, scale: float | Tensor) -> Tensor:
        y_out = x
        y_grad = x * scale
        return (y_out - y_grad).detach() + y_grad

    @staticmethod
    def _forward_with_real_backward(forward_tensor: Tensor, backward_tensor: Tensor) -> Tensor:
        return (forward_tensor - backward_tensor).detach() + backward_tensor

    @staticmethod
    def _reshape_per_channel_scale(scale: Tensor, x: Tensor) -> tuple[Tensor, int]:
        channel_dim = 0
        if scale.numel() > 1:
            new_shape = [scale.numel()] + [1] * (x.ndim - 1)
            scale = scale.view(new_shape)
        return scale, channel_dim

    @staticmethod
    def _round_to_nearest_targets(
        scaled: Tensor,
        targets: Tensor,
        channel_dim: int,
    ) -> Tensor:
        scaled_c_first = torch.movedim(scaled, channel_dim, 0)
        channels = scaled_c_first.shape[0]
        if targets.ndim == 1:
            targets_per_channel = targets.unsqueeze(0).expand(channels, -1)
        else:
            targets_per_channel = targets
            if targets_per_channel.shape[0] == 1 and channels > 1:
                targets_per_channel = targets_per_channel.expand(channels, -1)

        if targets_per_channel.shape[-1] == 1:
            rounded_c_first = targets_per_channel[:, :1].view(
                channels,
                *([1] * (scaled_c_first.ndim - 1)),
            ).expand_as(scaled_c_first)
            return torch.movedim(rounded_c_first, 0, channel_dim)

        sorted_targets = torch.sort(targets_per_channel, dim=-1).values
        thresholds = 0.5 * (sorted_targets[:, :-1] + sorted_targets[:, 1:])
        flat_values = scaled_c_first.reshape(channels, -1)
        flat_indices = torch.searchsorted(thresholds, flat_values, right=False)
        flat_rounded = torch.gather(sorted_targets, dim=1, index=flat_indices)
        rounded_c_first = flat_rounded.view_as(scaled_c_first)
        return torch.movedim(rounded_c_first, 0, channel_dim)

    @staticmethod
    def apply(
        x: Tensor,
        scale: Tensor,
        targets: Tensor,
        qn: int,
        qp: int,
        grad_scale_factor: float | Tensor = 1.0,
        return_metadata: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        The formula used is:

            q(x) = argmin_{c_k in targets} |(x+b)/s - c_k|
        """
        export_scale = LsqStepDrivenQuantizer.gradscale(scale, grad_scale_factor)
        scale_ste, channel_dim = LsqStepDrivenQuantizer._reshape_per_channel_scale(export_scale, x)
        scaled = x / scale_ste

        clipped = scaled.clamp(qn, qp)
        rounded = LsqStepDrivenQuantizer._round_to_nearest_targets(
            clipped,
            targets,
            channel_dim,
        )
        backward = clipped
        quant = LsqStepDrivenQuantizer._forward_with_real_backward(rounded, backward)
        if return_metadata:
            return quant, export_scale
        return quant
