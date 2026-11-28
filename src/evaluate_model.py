"""
Evaluate trained models on the test set. We compute the average SDR, SI-SDR, SI-SNR
and STOI metrics over the test set and save the result to a yaml file in each
experiment directory (``conf.exp_dir``).

Usage:
    python src/evaluate_model.py CONF_PATH [CONF_PATH ...] [-q QUANT_CONFIG ...]
                                 [--batch-size N] [--nb-batches N]
                                 [--nb-exported-batches N]

Examples:
    # Single full precision model
    python src/evaluate_model.py exp/ConvTasNet_0.5/conf.yml

    # Several models, each with its own quantization config (matched by position)
    python src/evaluate_model.py \
        exp/ConvTasNet_0.5/conf.yml \
        exp/ConvTasNet_0.5_all_S8S8/conf.yml \
        exp/ConvTasNet_0.5_all_S16S16/conf.yml \
        -q full_precision all_S8S8 all_S16S16
"""

import argparse
import os
import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    short_time_objective_intelligibility,
    signal_distortion_ratio,
)

from utils.conf import Conf
from utils.utils import (
    get_test_data_loaders,
    load_trained_model,
    rms_normalize,
)
from utils.utils_quantize import validate_quantization_config_names

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained models on the test set (SDR, SI-SDR, SI-SNR, STOI) "
            "and save the results in each model's experiment directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "conf_paths",
        nargs="+",
        type=Path,
        metavar="CONF_PATH",
        help=(
            "Full path(s) to the config file(s) of the model(s) to evaluate "
            "(e.g. exp/ConvTasNet_0.5/conf.yml). The experiment directory is read "
            "from conf.exp_dir. The first config is also used to build the test "
            "data loader."
        ),
    )
    parser.add_argument(
        "-q",
        "--quantization-configs",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Quantization config name for each CONF_PATH, matched by position "
            "(e.g. full_precision all_S8S8). Defaults to full_precision for all."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--nb-batches",
        type=int,
        default=None,
        help="Number of batches to evaluate. Default: the entire test set.",
    )
    parser.add_argument(
        "--nb-exported-batches",
        type=int,
        default=4,
        help=(
            "Number of random batches to export as audio, in addition to the "
            "first batch (default: %(default)s)."
        ),
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Number of random batches to export as audio, in addition to the "
            "first batch (default: %(default)s)."
        ),
    )

    parser.add_argument(
        "--mini_librimix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Number of random batches to export as audio, in addition to the "
            "first batch (default: %(default)s)."
        ),
    )

    parser.add_argument(
        "--eval_path",
        type=str,
        default="evaluation.yaml",
        help=("Yaml path to save eval metrics values"),
    )

    args = parser.parse_args()

    if args.quantization_configs is None:
        args.quantization_configs = ["full_precision"] * len(args.conf_paths)
    elif len(args.quantization_configs) != len(args.conf_paths):
        parser.error(
            f"Got {len(args.conf_paths)} config path(s) but "
            f"{len(args.quantization_configs)} quantization config(s); "
            "they must match one-to-one."
        )

    return args


def load_conf(path: Path) -> Conf:
    with open(path) as f:
        return Conf(yaml.safe_load(f))


def evaluate(
    test_set: DataLoader,
    nb_batches_to_evaluate: int | None,
    model: nn.Module,
    sample_rate: int,
    export_batches_index: set[int],
):
    if nb_batches_to_evaluate is None:
        nb_batches_to_evaluate = len(test_set)
    nb_batches_to_evaluate = min(nb_batches_to_evaluate, len(test_set))

    dataset_size = nb_batches_to_evaluate
    print(f"Test set length: {dataset_size} batches")

    sisdrs, sisnrs, sdrs, stois, maxes = (
        torch.zeros(dataset_size),
        torch.zeros(dataset_size),
        torch.zeros(dataset_size),
        torch.zeros(dataset_size),
        torch.zeros(dataset_size),
    )

    timings = torch.zeros(dataset_size)

    audios = {}

    for i, batch in enumerate(test_set):
        input, targets = batch
        input = input.to(device)
        targets = targets.to(device)

        print(f"Batch {i} - Input shape: {input.shape}, Targets shape: {targets.shape}")
        # Compute estimated targets
        with torch.no_grad():
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

            est_targets = model(input)

            if device == "cuda":
                end.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded

        print(f"Batch {i} - Estimated targets shape: {est_targets.shape}")

        if device == "cuda":
            timings[i] = start.elapsed_time(end) / input.size(0)  # Time per sample

        sdr, _ = permutation_invariant_training(
            est_targets, targets, signal_distortion_ratio, eval_func="max"
        )
        sdrs[i] = sdr.mean()
        print(f"Batch {i} - SDR: {sdr.mean()}")

        sisdr, _ = permutation_invariant_training(
            est_targets,
            targets,
            scale_invariant_signal_distortion_ratio,
            eval_func="max",
        )
        sisdrs[i] = sisdr.mean()
        print(f"Batch {i} - SI-SDR: {sisdr.mean()}")

        sisnr, _ = permutation_invariant_training(
            est_targets,
            targets,
            scale_invariant_signal_noise_ratio,
            eval_func="max",
        )
        sisnrs[i] = sisnr.mean()
        print(f"Batch {i} - SI-SNR: {sisnr.mean()}")

        stoi, _ = permutation_invariant_training(
            est_targets,
            targets,
            short_time_objective_intelligibility,
            eval_func="max",
            fs=sample_rate,
        )
        stois[i] = stoi.mean()
        print(f"Batch {i} - STOI: {stoi.mean()}")

        maxes[i] = torch.max(est_targets.abs())

        if i in export_batches_index:
            audios[i] = {
                "input": input.cpu(),
                "est_targets": est_targets.cpu(),
                "targets": targets.cpu(),
            }

        if i >= nb_batches_to_evaluate - 1:
            break

    print(f"Average SDR: {sdrs.mean()}")
    print(f"Average SI-SDR: {sisdrs.mean()}")
    print(f"Average SI-SNR: {sisnrs.mean()}")
    print(f"Average STOI: {stois.mean()}")
    print(f"Average STOI: {maxes.mean()}")
    print(f"Average inference time per sample: {timings.mean()} ms")

    results = {
        "average_sdr": sdrs.mean().item(),
        "average_sisdr": sisdrs.mean().item(),
        "average_sisnr": sisnrs.mean().item(),
        "average_stoi": stois.mean().item(),
        "average_max": maxes.mean().item(),
        "max_max": maxes.max().item(),
        f"average_inference_time_ms_{dataset_size}_batches": timings.mean().item(),
    }

    return results, audios


def export_audios(audios: dict, config_name: str, conf: Conf) -> None:
    eval_dir = os.path.join(conf.exp_dir, "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    sample_rate = conf.data.sample_rate

    for i in audios.keys():
        # print(audios[i]["input"].shape)
        input = audios[i]["input"][0, :]
        target_1 = audios[i]["targets"][0, 0, :]
        target_2 = audios[i]["targets"][0, 1, :]
        est_target_1 = audios[i]["est_targets"][0, 0, :]
        est_target_2 = audios[i]["est_targets"][0, 1, :]

        # linear rms level and scaling factor
        est_target_1 = rms_normalize(est_target_1, rms_level_db=-20.0)
        est_target_2 = rms_normalize(est_target_2, rms_level_db=-20.0)

        ratio_1 = est_target_1.abs().amax() / target_1.abs().amax()
        ratio_2 = est_target_2.abs().amax() / target_2.abs().amax()

        # for name, sig in [
        #     ("input", input),
        #     ("target1", target_1),
        #     ("est1", est_target_1),
        #     ("ratio1", ratio_1),
        #     ("target2", target_2),
        #     ("est2", est_target_2),
        #     ("ratio2", ratio_2),
        # ]:
        #     print(
        #         name,
        #         sig.min().item(),
        #         sig.max().item(),
        #         sig.abs().max().item(),
        #     )

        for suffix, sig in [
            ("input", input),
            ("target_1", target_1),
            ("target_2", target_2),
            ("est_target_1", est_target_1),
            ("est_target_2", est_target_2),
        ]:
            path = os.path.join(eval_dir, f"{config_name}_{i}_{suffix}.flac")
            sf.write(path, sig, sample_rate)


def main() -> None:
    args = parse_args()

    validate_quantization_config_names(
        args.quantization_configs, accept_full_precision=True
    )
    print(f"Using quantization configs: {args.quantization_configs}")

    # The first config is used to build the (shared) test data loader.
    data_conf = load_conf(args.conf_paths[0])
    test_set = get_test_data_loaders(
        conf=data_conf, evaluation_batch_size=args.batch_size, shuffle=False,
    )

    nb_batches = args.nb_batches
    effective_nb_batches = min(nb_batches or len(test_set), len(test_set))

    # Random batches to export, plus the first one.
    export_batches_index = set(
        random.choices(range(effective_nb_batches), k=args.nb_exported_batches)
    )
    export_batches_index.add(0)
    print(f"Exporting batches with indices: {sorted(export_batches_index)}")

    for conf_path, config_name in zip(args.conf_paths, args.quantization_configs):
        print(f"Evaluating configuration: {config_name} ({conf_path})")

        conf = load_conf(conf_path)

        if config_name == "full_precision":
            model = load_trained_model(conf=conf, eval=True, compile=True,trained_model=args.ckpt)
        else:
            model = load_trained_model(
                conf=conf, eval=True, compile=True, config_name=config_name
            )

        os.makedirs(os.path.join(conf.exp_dir, "evaluations"), exist_ok=True)

        model = model.to(device)

        results, audios = evaluate(
            test_set,
            nb_batches,
            model,
            sample_rate=conf.data.sample_rate,
            export_batches_index=export_batches_index,
        )

        export_audios(audios, config_name, conf)

        path = os.path.join(conf.exp_dir, os.path.basename(args.eval_path))
        with open(path, "w") as f:
            yaml.dump(results, f)

        print(f"Evaluation results saved to {path}")


if __name__ == "__main__":
    main()
