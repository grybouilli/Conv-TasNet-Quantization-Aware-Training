#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")
    return checkpoint


def _iter_weight_tensors(
    state_dict: Dict[str, torch.Tensor],
    name_regex: str | None,
) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if not name.endswith(".weight"):
            continue
        if tensor.ndim < 2:
            continue
        if name_regex and not re.search(name_regex, name):
            continue
        yield name, tensor.detach().flatten()


def _sample_values(values: torch.Tensor, max_samples: int, rng: torch.Generator) -> torch.Tensor:
    if max_samples <= 0 or values.numel() <= max_samples:
        return values
    indices = torch.randperm(values.numel(), generator=rng)[:max_samples]
    return values[indices]


def _kmeans_1d(values: torch.Tensor, k: int, iters: int, tol: float) -> torch.Tensor:
    values = values.float().view(-1)
    if values.numel() < k:
        raise ValueError(f"Not enough samples ({values.numel()}) for k={k}")
    quantiles = torch.linspace(0, 1, k, device=values.device)
    centers = torch.quantile(values, quantiles)
    for _ in range(iters):
        distances = (values[:, None] - centers[None, :]).abs()
        assignments = distances.argmin(dim=1)
        sums = torch.zeros_like(centers)
        counts = torch.zeros_like(centers)
        sums.scatter_add_(0, assignments, values)
        counts.scatter_add_(0, assignments, torch.ones_like(values))
        new_centers = torch.where(counts > 0, sums / counts, centers)
        shift = torch.max(torch.abs(new_centers - centers)).item()
        centers = new_centers
        if shift < tol:
            break
    centers, _ = torch.sort(centers)
    return centers


def _format_list(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.cpu().tolist()]


def _integerize_centers(centers: torch.Tensor, k: int) -> torch.Tensor:
    centers = centers.view(-1).float()
    rounded = torch.round(centers).to(torch.int64)
    chosen = []
    used = set()
    for value in rounded.tolist():
        if value not in used:
            chosen.append(value)
            used.add(value)
        if len(chosen) == k:
            break
    if len(chosen) < k:
        min_val = int(torch.floor(centers.min()).item()) - k
        max_val = int(torch.ceil(centers.max()).item()) + k
        candidates = [c for c in range(min_val, max_val + 1) if c not in used]
        center_vals = centers.tolist()
        candidates.sort(key=lambda c: min(abs(c - v) for v in center_vals))
        for value in candidates:
            chosen.append(value)
            used.add(value)
            if len(chosen) == k:
                break
    return torch.tensor(sorted(chosen), dtype=centers.dtype)


def _scale_centers_to_bits(centers: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    centers = centers.view(-1).float()
    if bits <= 0:
        raise ValueError("bits must be >= 1")
    if signed:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        max_abs = centers.abs().max()
        if max_abs.item() == 0:
            return centers
        scaled = centers * (qmax / max_abs)
        return scaled.clamp(qmin, qmax)
    qmin = 0
    qmax = (1 << bits) - 1
    min_val = centers.min()
    max_val = centers.max()
    if max_val.item() == min_val.item():
        return (centers - min_val).clamp(qmin, qmax)
    scaled = (centers - min_val) * (qmax / (max_val - min_val))
    return scaled.clamp(qmin, qmax)


def _resolve_ptcv_model(model_name: str, dataset: str) -> str:
    model_key = model_name.lower()
    dataset_key = dataset.lower()
    mapping = {
        "resnet20": {"cifar10": "resnet20_cifar10", "cifar100": "resnet20_cifar100"},
        "resnet56": {"cifar10": "resnet56_cifar10", "cifar100": "resnet56_cifar100"},
        "resnet18": {"imagenet1k": "resnet18"},
        "resnet34": {"imagenet1k": "resnet34"},
        "resnet50": {"imagenet1k": "resnet50"},
        "resnet101": {"imagenet1k": "resnet101"},
        "resnet152": {"imagenet1k": "resnet152"},
    }
    if model_key not in mapping or dataset_key not in mapping[model_key]:
        raise ValueError(f"Unsupported model/dataset pair: {model_name} with {dataset}")
    return mapping[model_key][dataset_key]


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive non-uniform codebooks with k-means.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=Path)
    source_group.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Scale centers to signed range [-2^(bits-1), 2^(bits-1)-1].",
    )
    parser.add_argument("--unsigned", action="store_true", help="Use unsigned scaling range.")
    parser.add_argument("--integer", dest="integer", action="store_true", default=True)
    parser.add_argument("--no-integer", dest="integer", action="store_false")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument("--name-regex", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.k < 2:
        raise ValueError("k must be >= 2")

    rng = torch.Generator().manual_seed(args.seed)
    random.seed(args.seed)

    if args.checkpoint is not None:
        state_dict = _load_state_dict(args.checkpoint)
    else:
        ptcv_name = _resolve_ptcv_model(args.model, args.dataset)
        model = ptcv_get_model(ptcv_name, pretrained=args.pretrained)
        state_dict = model.state_dict()
    layers = list(_iter_weight_tensors(state_dict, args.name_regex))
    if not layers:
        print("No weight tensors found (ndim>=2 and name endswith .weight).")
        return 1

    results = {}
    if args.per_layer:
        for name, values in layers:
            samples = _sample_values(values, args.max_samples, rng)
            centers = _kmeans_1d(samples, args.k, args.iters, args.tol)
            if args.bits is not None:
                centers = _scale_centers_to_bits(centers, args.bits, signed=not args.unsigned)
            if args.integer:
                centers = _integerize_centers(centers, args.k)
            results[name] = _format_list(centers)
    else:
        values = torch.cat([v for _, v in layers])
        samples = _sample_values(values, args.max_samples, rng)
        centers = _kmeans_1d(samples, args.k, args.iters, args.tol)
        if args.bits is not None:
            centers = _scale_centers_to_bits(centers, args.bits, signed=not args.unsigned)
        if args.integer:
            centers = _integerize_centers(centers, args.k)
        results["global"] = _format_list(centers)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
