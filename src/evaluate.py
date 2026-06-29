import os

import soundfile as sf
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_distortion_ratio,
    short_time_objective_intelligibility,
    signal_distortion_ratio,
)

from utils.conf import Conf
from utils.utils import (
    get_data_loaders,
    get_test_data_loaders,
    load_trained_model,
    rms_normalize,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

evaluation_batch_size: int = 128  # 32
# If nb_batches_to_evaluate is None, the entire set will be used
nb_batches_to_evaluate: int | None = None  # 10


def evaluate(
    test_set: DataLoader,
    nb_batches_to_evaluate: int | None,
    model: nn.Module,
):
    if nb_batches_to_evaluate is None:
        nb_batches_to_evaluate = len(test_set)

    dataset_size = nb_batches_to_evaluate
    print(f"Test set length: {dataset_size} batches")

    sisdrs, sdrs, stois = (
        torch.zeros(dataset_size),
        torch.zeros(dataset_size),
        torch.zeros(dataset_size),
    )

    timings = torch.zeros(dataset_size)

    audios = {}

    # Warmup
    # with torch.no_grad():
    #     input = next(iter(test_set))[0].unsqueeze(1).to(device)  # [batch, channel, time]
    #     print(f"Warmup input shape: {input.shape}")
    #     for _ in range(2):
    #         _ = model(input)

    for i, batch in enumerate(test_set):
        input, targets = batch
        input = input.unsqueeze(1).to(device)

        targets = targets.to(device)  # .unsqueeze(0)  # Add batch dimension to targets

        print(f"Batch {i} - Input shape: {input.shape}, Targets shape: {targets.shape}")
        # Compute estimated targets
        with torch.no_grad():
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

            est_targets = model(input)  # .unsqueeze(0)

            if device == "cuda":
                end.record()
                torch.cuda.synchronize()  # Wait for the events to be recorded

        targets = targets.to(device)  # .unsqueeze(0)  # Add batch dimension to targets
        print(f"Batch {i} - Estimated targets shape: {est_targets.shape}")

        if device == "cuda":
            timings[i] = start.elapsed_time(end) / input.size(0)  # Time per sample

        # best_metric, best_perm =
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

        stoi, _ = permutation_invariant_training(
            est_targets,
            targets,
            short_time_objective_intelligibility,
            eval_func="max",
            fs=conf.data.sample_rate,
        )
        stois[i] = stoi.mean()
        print(f"Batch {i} - STOI: {stoi.mean()}")

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
    print(f"Average STOI: {stois.mean()}")
    print(f"Average inference time per batch: {timings.mean()} ms")

    results = {
        "average_sdr": sdrs.mean().item(),
        "average_sisdr": sisdrs.mean().item(),
        "average_stoi": stois.mean().item(),
        f"average_inference_time_ms_{dataset_size}_batches": timings.mean().item(),
    }

    return results, audios


CONF_PATH = f"exp/ConvTasNet_0.001/conf.yml"

# TODO: max
export_batches_index = torch.randint(0, nb_batches_to_evaluate or 23, (4,)).tolist()
export_batches_index.append(0)  # Ensure the first batch is exported
print(f"Exporting batches with indices: {export_batches_index}")


if __name__ == "__main__":
    with open(CONF_PATH) as f:
        conf_data = yaml.safe_load(f)

    conf = Conf(conf_data)

    test_set = get_test_data_loaders(
        conf=conf, evaluation_batch_size=evaluation_batch_size, shuffle=False
    )

    # train_loader, val_loader = get_data_loaders(
    #     conf=conf, load_mini=True, force_batch_size=1
    # )
    # test_set = val_loader

    configs_names = [
        "encdec_S16S16_tcn_S8S8"
    ]  # ["full_precision", "all_S8S8", "all_S4S4", "all_S16S16"]

    for config_name in configs_names:
        print(f"Evaluating configuration: {config_name}")
        config_conf: Conf
        if config_name == "full_precision":
            conf_path = f"exp/ConvTasNet_0.001/conf.yml"
            with open(conf_path) as f:
                conf_data = yaml.safe_load(f)

            config_conf = Conf(conf_data)
            model = load_trained_model(conf=config_conf, eval=True, compile=True)

        else:
            conf_path = f"exp/ConvTasNet_0.001_{config_name}/conf.yml"
            with open(conf_path) as f:
                conf_data = yaml.safe_load(f)

            config_conf = Conf(conf_data)
            model = load_trained_model(
                conf=config_conf, eval=True, compile=True, config_name=config_name
            )

        try:
            os.mkdir(os.path.join(config_conf.exp_dir, f"evaluations/"))
        except FileExistsError:
            print("Evaluation directory already exists")

        model = model.to(device)

        results, audios = evaluate(test_set, nb_batches_to_evaluate, model)

        for i in audios.keys():
            input = audios[i]["input"][0, 0, :]
            target_1 = audios[i]["targets"][0, 0, :]
            target_2 = audios[i]["targets"][0, 1, :]
            est_target_1 = audios[i]["est_targets"][0, 0, :]
            est_target_2 = audios[i]["est_targets"][0, 1, :]

            # linear rms level and scaling factor
            est_target_1 = rms_normalize(est_target_1, rms_level_db=-20.0)
            est_target_2 = rms_normalize(est_target_2, rms_level_db=-20.0)

            ratio_1 = est_target_1.abs().amax() / target_1.abs().amax()
            ratio_2 = est_target_2.abs().amax() / target_2.abs().amax()

            for name, sig in [
                ("input", input),
                ("target1", target_1),
                ("est1", est_target_1),
                ("ratio1", ratio_1),
                ("target2", target_2),
                ("est2", est_target_2),
                ("ratio2", ratio_2),
            ]:
                print(
                    name,
                    sig.min().item(),
                    sig.max().item(),
                    sig.abs().max().item(),
                )

            path = os.path.join(
                config_conf.exp_dir, f"evaluations/{config_name}_{i}_input.flac"
            )
            sf.write(path, input, config_conf.data.sample_rate)
            path = os.path.join(
                config_conf.exp_dir, f"evaluations/{config_name}_{i}_target_1.flac"
            )
            sf.write(path, target_1, config_conf.data.sample_rate)
            path = os.path.join(
                config_conf.exp_dir, f"evaluations/{config_name}_{i}_target_2.flac"
            )
            sf.write(path, target_2, config_conf.data.sample_rate)
            path = os.path.join(
                config_conf.exp_dir, f"evaluations/{config_name}_{i}_est_target_1.flac"
            )
            sf.write(path, est_target_1, config_conf.data.sample_rate)
            path = os.path.join(
                config_conf.exp_dir, f"evaluations/{config_name}_{i}_est_target_2.flac"
            )
            sf.write(path, est_target_2, config_conf.data.sample_rate)

        path = os.path.join(config_conf.exp_dir, f"evaluation.yaml")
        with open(path, "w") as f:
            yaml.dump(
                results,
                f,
            )
