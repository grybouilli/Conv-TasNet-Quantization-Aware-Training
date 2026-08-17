#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

from derive_codebook import (
    _integerize_centers,
    _kmeans_1d,
    _load_state_dict,
    _resolve_ptcv_model,
    _scale_centers_to_bits,
    _sample_values,
    _iter_weight_tensors,
)


def _parse_sets(path: Path) -> List[List[int]]:
    sets = []
    line_re = re.compile(r"Nb outputs:\s*\d+\s*\|([^|]+)\|")
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    has_csv_outputs = any("NbOutputs" in line and "Outputs" in line for line in lines[:10])
    for line in lines:
        match = line_re.search(line)
        if match:
            values = [int(v) for v in re.findall(r"-?\d+", match.group(1))]
            if values:
                sets.append(sorted(values))
            continue
        if has_csv_outputs:
            if not line or not line[0].isdigit():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            values = [int(v) for v in re.findall(r"-?\d+", parts[1])]
            if values:
                sets.append(sorted(values))
    return sets


def _is_encodable(values: List[int], bits: int) -> bool:
    if bits <= 0:
        return False
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    return all(qmin <= v <= qmax for v in values)


def _is_symmetric(values: List[int], allow_unpaired: int = 1) -> bool:
    value_set = set(values)
    if 0 not in value_set:
        return False
    positives = sorted(v for v in value_set if v > 0)
    negatives = sorted(-v for v in value_set if v < 0)
    pos_set = set(positives)
    neg_set = set(negatives)
    unmatched_pos = [v for v in positives if v not in neg_set]
    unmatched_neg = [v for v in negatives if v not in pos_set]
    return (len(unmatched_pos) + len(unmatched_neg)) <= allow_unpaired


def _closest_sets(
    reference: torch.Tensor, sets: List[List[int]], top_k: int
) -> List[tuple[List[int], float]]:
    distances = []
    ref = reference.float().view(-1)
    seen = set()
    for values in sets:
        key = tuple(values)
        if key in seen:
            continue
        seen.add(key)
        candidate = torch.tensor(values, dtype=torch.float32)
        if candidate.numel() != ref.numel():
            continue
        dist = torch.mean((candidate - ref) ** 2).item()
        distances.append((values, dist))
    distances.sort(key=lambda item: item[1])
    return distances[:top_k]


def _available_sizes(sets: List[List[int]]) -> List[int]:
    sizes = sorted({len(values) for values in sets})
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description="Find closest codebook set and filter symmetric sets.")
    parser.add_argument("--config", type=Path, default=Path("scripts/4bitsConf.txt"))
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=Path)
    source_group.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--bits", type=int, required=True, help="Signed range: [-2^(bits-1), 2^(bits-1)-1].")
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name-regex", type=str, default=None)
    parser.add_argument("--no-integer", dest="integer", action="store_false", default=True)
    args = parser.parse_args()

    rng = torch.Generator().manual_seed(args.seed)
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

    values = torch.cat([v for _, v in layers])
    samples = _sample_values(values, args.max_samples, rng)
    centers = _kmeans_1d(samples, args.k, args.iters, args.tol)
    centers = _scale_centers_to_bits(centers, args.bits, signed=not args.unsigned)
    if args.integer:
        centers = _integerize_centers(centers, args.k)

    all_sets = _parse_sets(args.config)
    if not all_sets:
        print("No sets found in config file.")
        return 1

    closest_sets = _closest_sets(centers, all_sets, top_k=5)
    if not closest_sets:
        sizes = _available_sizes(all_sets)
        print(
            "No candidate sets with matching size. "
            f"Found sizes in config: {sizes}"
        )
        return 1
    print("Top 5 closest sets:")
    for idx, (values, dist) in enumerate(closest_sets, start=1):
        print(f"{idx}. {values} (MSE: {dist:.6f})")

    encodable_sets = [values for values in all_sets if _is_encodable(values, args.bits)]
    symmetric_sets = [values for values in encodable_sets if _is_symmetric(values)]

    print("\nSymmetric encodable sets:")
    for values in symmetric_sets:
        print(values)

    if not symmetric_sets:
        total = len(all_sets)
        encodable = len(encodable_sets)
        print(
            f"No symmetric encodable sets found (total={total}, encodable={encodable})."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
