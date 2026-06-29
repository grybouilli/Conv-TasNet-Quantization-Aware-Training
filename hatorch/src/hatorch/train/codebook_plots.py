"""Reusable plotting utilities for LSQ codebook analysis."""

from __future__ import annotations

import os
import re

import matplotlib
import torch
import torch.nn as nn

from hatorch.quantizers.common._common_quantizer import BaseLearnedQuantizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sanitize_layer_name(layer_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", layer_name)
    return safe.strip("._") or "layer"


def _sort_codebook(codebook: torch.Tensor) -> torch.Tensor:
    if codebook.ndim <= 1:
        return torch.sort(codebook.reshape(-1)).values
    return torch.sort(codebook, dim=-1).values


def _get_levels_for_quantizer(
    weight_quantizer: BaseLearnedQuantizer,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    codebook_fp = None
    codebook_quantized = None
    if hasattr(weight_quantizer, "_current_codebook_targets"):
        codebook_quantized = weight_quantizer._current_codebook_targets()
        if codebook_quantized is not None:
            codebook_quantized = _sort_codebook(codebook_quantized.detach())
            codebook_fp = codebook_quantized

    if codebook_quantized is not None:
        return codebook_fp, codebook_quantized, "codebook"

    # Uniform INTX quantization path: use full integer range [qn, qp].
    if not hasattr(weight_quantizer, "qn") or not hasattr(weight_quantizer, "qp"):
        return None, None, ""
    if getattr(weight_quantizer, "is_non_uniform_bitwidth", False):
        return None, None, ""
    qn_i = int(round(float(weight_quantizer.qn)))
    qp_i = int(round(float(weight_quantizer.qp)))
    if qn_i > qp_i:
        return None, None, ""
    levels = torch.arange(qn_i, qp_i + 1, dtype=torch.float32, device=weight_quantizer.current_scale().device)
    levels = _sort_codebook(levels)
    return levels, levels, "uniform_intx"


def _compute_usage_percentages(
    quantized_weight: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    codebook = _sort_codebook(codebook).to(device=quantized_weight.device, dtype=quantized_weight.dtype)

    if codebook.ndim <= 1:
        flat_weight = quantized_weight.reshape(-1)
        if codebook.numel() > 1:
            indices = torch.argmin(torch.abs(flat_weight.unsqueeze(-1) - codebook.unsqueeze(0)), dim=-1)
        else:
            indices = torch.zeros_like(flat_weight, dtype=torch.long)
        counts = torch.bincount(indices, minlength=codebook.numel()).float()
        labels = [f"{float(v):.4g}" for v in codebook.cpu()]
    else:
        # For per-channel codebooks [C, M], aggregate usage by code index k across channels.
        channels, num_levels = codebook.shape
        weight_c_first = torch.movedim(quantized_weight, 0, 0).reshape(channels, -1)
        if num_levels > 1:
            indices = torch.argmin(
                torch.abs(weight_c_first.unsqueeze(-1) - codebook.unsqueeze(1)),
                dim=-1,
            ).reshape(-1)
        else:
            indices = torch.zeros(weight_c_first.numel(), device=weight_c_first.device, dtype=torch.long)
        counts = torch.bincount(indices, minlength=num_levels).float()
        labels = [f"k={k}" for k in range(num_levels)]

    total = counts.sum().clamp_min(1.0)
    percentages = (counts / total) * 100.0
    return labels, percentages.cpu()


def _effective_fp_weight_for_analysis(layer: nn.Module) -> torch.Tensor:
    # Match QuantConvBn2d eval-time behavior only when BN is actually folded.
    if (
        getattr(layer, "fold_batch_norm", False)
        and hasattr(layer, "bn")
        and hasattr(layer.bn, "running_var")
        and hasattr(layer.bn, "weight")
    ):
        running_std = torch.sqrt(layer.bn.running_var + layer.bn.eps)
        scale_factor = layer.bn.weight / running_std
        weight_shape = [layer.weight.shape[0]] + [1] * (layer.weight.ndim - 1)
        return layer.weight.detach() * scale_factor.reshape(weight_shape)
    return layer.weight.detach()


def _project_fp_weight_to_code_domain(
    fp_weight: torch.Tensor,
    weight_quantizer: BaseLearnedQuantizer,
) -> torch.Tensor:
    scale = weight_quantizer.current_scale().detach().to(device=fp_weight.device, dtype=fp_weight.dtype)
    if weight_quantizer.zero_point is None:
        zero_point = torch.zeros_like(scale)
    else:
        zero_point = weight_quantizer.zero_point.detach().to(device=fp_weight.device, dtype=fp_weight.dtype)

    if scale.numel() > 1:
        reshape = [scale.numel()] + [1] * (fp_weight.ndim - 1)
        scale = scale.view(reshape)
        zero_point = zero_point.view(reshape)

    return (fp_weight - zero_point) / scale


def _plot_layer_weight_histograms(
    layer_name: str,
    fp_weight: torch.Tensor,
    projected_weight: torch.Tensor,
    fp_codebook: torch.Tensor,
    labels: list[str],
    percentages: torch.Tensor,
    output_dir: str,
    fold_batch_norm: bool,
) -> str:
    fig, (ax_fp, ax_proj, ax_usage) = plt.subplots(1, 3, figsize=(18, 4))

    fp_flat = fp_weight.detach().float().cpu().reshape(-1)
    ax_fp.hist(fp_flat.numpy(), bins=80, color="tab:blue", alpha=0.85)
    if fold_batch_norm:
        ax_fp.set_title(f"Effective FP weights (after BN fold) - {layer_name}")
    else:
        ax_fp.set_title(f"FP conv weights (BN not folded) - {layer_name}")
    ax_fp.set_xlabel("Weight value")
    ax_fp.set_ylabel("Count")

    proj_flat = projected_weight.detach().float().cpu().reshape(-1)
    ax_proj.hist(proj_flat.numpy(), bins=80, color="tab:green", alpha=0.8)
    sorted_codebook = _sort_codebook(fp_codebook).detach().float().cpu()
    if sorted_codebook.ndim == 1:
        for code in sorted_codebook.tolist():
            ax_proj.axvline(code, color="tab:red", alpha=0.25, linewidth=1.0)
    ax_proj.set_title("Projected FP weights / scale (code domain)")
    ax_proj.set_xlabel("Projected value")
    ax_proj.set_ylabel("Count")

    x = list(range(len(labels)))
    ax_usage.bar(x, percentages.tolist(), color="tab:orange")
    ax_usage.set_xticks(x)
    ax_usage.set_xticklabels(labels, rotation=45, ha="right")
    ax_usage.set_ylabel("Usage (%)")
    ax_usage.set_xlabel("Sorted code")
    ax_usage.set_title("Quantized code usage")

    fig.tight_layout()

    file_name = f"{sanitize_layer_name(layer_name)}_code_usage.png"
    file_path = os.path.join(output_dir, file_name)
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return file_path


def summarize_and_plot_codebook_usage(model_module: nn.Module, output_root: str) -> list[str]:
    code_usage_dir = os.path.join(output_root, "codebook_usage")
    os.makedirs(code_usage_dir, exist_ok=True)

    print("\n=== Per-layer code/level usage ===")
    found_any = False
    chart_paths: list[str] = []

    model_module.eval()
    with torch.no_grad():
        for layer_name, layer in model_module.named_modules():
            if not hasattr(layer, "weight") or not hasattr(layer, "weight_quantizer"):
                continue

            weight_quantizer = layer.weight_quantizer
            if not isinstance(weight_quantizer, BaseLearnedQuantizer):
                continue

            fp_levels, quantized_levels, quantizer_kind = _get_levels_for_quantizer(weight_quantizer)
            if fp_levels is None or quantized_levels is None:
                continue
            found_any = True
            fp_weight = _effective_fp_weight_for_analysis(layer)
            projected_weight = _project_fp_weight_to_code_domain(fp_weight, weight_quantizer)
            quantized_weight = weight_quantizer(fp_weight).value.detach()
            labels, percentages = _compute_usage_percentages(quantized_weight, quantized_levels)
            chart_path = _plot_layer_weight_histograms(
                layer_name,
                fp_weight,
                projected_weight,
                fp_levels,
                labels,
                percentages,
                code_usage_dir,
                bool(getattr(layer, "fold_batch_norm", False)),
            )
            chart_paths.append(chart_path)

            fp_levels_cpu = fp_levels.detach().cpu()
            quantized_levels_cpu = quantized_levels.detach().cpu()
            used_mask = percentages > 0.0
            used_indices = torch.nonzero(used_mask, as_tuple=False).squeeze(-1).tolist()
            print(f"\nLayer: {layer_name}")
            print(f"Quantizer mode: {quantizer_kind}")
            if not torch.equal(fp_levels_cpu, quantized_levels_cpu):
                print(f"FP codebook: {fp_levels_cpu.tolist()}")
                print(f"Quantized levels: {quantized_levels_cpu.tolist()}")
            else:
                print(f"Levels: {quantized_levels_cpu.tolist()}")
            for idx in used_indices:
                label = labels[idx]
                pct = float(percentages[idx].item())
                print(f"  [{idx:02d}] code={label}: {pct:.2f}%")
            print(f"Histogram: {chart_path}")

    if not found_any:
        print("No quantized weight layers found.")
    return chart_paths
