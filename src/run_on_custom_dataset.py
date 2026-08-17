"""
Run quantized model on a custom folder of speaker audio files. The script will mix all possible combinations of speakers and evaluate the model on these mixtures.

Usage:
    python src/run_on_custom_dataset.py model_archi quantization_config audio_file_folder save_output_files

Arguments:
    model_architecture:
        Model architecture identifier (e.g. ``0.5``, ``1.5``).

    quantization_config:
        Name of the quantization configuration to apply (e.g. ``all_S8S8``, ``all_S16S16``...).

    audio_file_folder:
        Path to a folder containing a set of 1-speaker audio files to be mixed and separated. The audio files should be in .wav format and have the same sample rate as defined in configuration file.

    save_output_files:
        If True, the separated audio files will be saved in a `outputs` subfolder of the `audio_file_folder`.

Examples:
    python src/run_on_custom_dataset.py 0.5 all_S8S8 custom_audios/librimix_extract/ False
"""

import json
import os
import sys

import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from torchmetrics.functional import (
    permutation_invariant_training,
    scale_invariant_signal_distortion_ratio,
)

from utils.conf import Conf
from utils.utils import load_trained_model, rms_normalize

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)

model_archi = sys.argv[1]
config_name = sys.argv[2]
audio_file_folder = sys.argv[3]
save_output_files = sys.argv[4].lower() == "true"

CONF_PATH = f"exp/ConvTasNet_{model_archi}_{config_name}/conf.yml"
OUTPUT_FILES_PATH = os.path.join(audio_file_folder, "outputs/")
OUTPUT_HEATMAP_PATH = os.path.join(
    OUTPUT_FILES_PATH, f"heatmap_{model_archi}_{config_name}.json"
)

os.makedirs(OUTPUT_FILES_PATH, exist_ok=True)

with open(CONF_PATH) as f:
    conf_data = yaml.safe_load(f)

conf = Conf(conf_data)


model = load_trained_model(conf=conf, eval=True, compile=True, config_name=config_name)

model = model.to(device)


def cut_in_segments(
    data: torch.Tensor, samplerate: int, segment_length_seconds: int = 3
) -> torch.Tensor:
    """
    Return a tensor of shape [num_segments, 1, segment_length] containing the segments of the input audio data
    where segment_length = segment_length_seconds * samplerate. If the last segment is shorter than segment_length, it will be padded with zeros.
    """
    segment_length = segment_length_seconds * samplerate  # segment length in samples
    segments = []
    for start in range(0, len(data), segment_length):
        end = start + segment_length
        segment = data[start:end]
        if len(segment) < segment_length:
            # Pad the last segment with zeros if it's shorter than the specified length
            segment = F.pad(segment, (0, segment_length - len(segment)), "constant")
        segment = segment.unsqueeze(0).unsqueeze(0).to(device)
        segments.append(segment)
    return torch.cat(segments, dim=0)


audio_files = [f for f in os.listdir(audio_file_folder) if f.endswith(".wav")]

print(f"Found {len(audio_files)} audio files in {audio_file_folder}:")
print(audio_files)

audio_names = [file_name.split(".")[0] for file_name in audio_files]

inputs = []
for file_name in audio_files:
    filepath = os.path.join(audio_file_folder, file_name)
    data, samplerate = sf.read(filepath, dtype="float32")

    if samplerate != conf.data.sample_rate:
        raise ValueError(
            f"Sample rate of the audio file ({samplerate}) does not match the expected sample rate ({conf.data.sample_rate})."
        )

    inputs.append(torch.from_numpy(data).to(device))

heatmap_sisdrs = []
# We want to do all possible combinations of speakers
for i in range(len(inputs)):
    for j in range(i + 1, len(inputs)):
        input_1 = inputs[i]
        input_2 = inputs[j]

        # # Pad the shorter input with zeros
        # if input_1.shape[0] < input_2.shape[0]:
        #     input_1 = F.pad(input_1, (0, input_2.shape[0] - input_1.shape[0]), 'constant')
        # elif input_2.shape[0] < input_1.shape[0]:
        #     input_2 = F.pad(input_2, (0, input_1.shape[0] - input_2.shape[0]), 'constant')
        # Truncate the longer input to match the shorter one
        min_length = min(input_1.shape[0], input_2.shape[0])
        input_1 = input_1[:min_length]
        input_2 = input_2[:min_length]

        # Mix the two inputs along the channel dimension
        # input_1 = rms_normalize(input_1)
        # input_2 = rms_normalize(input_2)
        combined_input = 0.5 * (input_1 + input_2)

        print(
            f"Combined input shape for speakers {audio_names[i]} and {audio_names[j]}: {combined_input.shape}"
        )

        if save_output_files:
            sf.write(
                os.path.join(
                    OUTPUT_FILES_PATH, f"{audio_names[i]}_{audio_names[j]}_input.wav"
                ),
                rms_normalize(combined_input.cpu()),
                samplerate,
            )

        # Cut in 3s segments
        input_1_segments = cut_in_segments(
            input_1, samplerate, segment_length_seconds=conf.data.segment
        )
        input_2_segments = cut_in_segments(
            input_2, samplerate, segment_length_seconds=conf.data.segment
        )
        combined_input_segments = cut_in_segments(
            combined_input, samplerate, segment_length_seconds=conf.data.segment
        )

        # print(
        #     f"input_1_segments shape: {input_1_segments.shape}, input_2_segments shape: {input_2_segments.shape}, combined_input_segments shape: {combined_input_segments.shape}"
        # )

        with torch.no_grad():
            est_targets = model(
                combined_input_segments.squeeze(1)
            )  # Remove the channel dimension for the model input

        targets = torch.cat(
            [input_1_segments, input_2_segments], dim=1
        )  # [batch, channel, time]

        print(f"est_targets shape: {est_targets.shape}, targets shape: {targets.shape}")

        print(est_targets)

        si_sdr, _ = permutation_invariant_training(
            est_targets,
            targets,
            scale_invariant_signal_distortion_ratio,
            eval_func="max",
        )

        print(si_sdr)

        si_sdr = si_sdr.mean()
        print(f"SI-SDR: {si_sdr}")

        names = (audio_names[i], audio_names[j])
        name_1, name_2 = sorted(names)
        heatmap_sisdrs.append(
            {
                "x": name_1,
                "y": name_2,
                "si_sdr": si_sdr.item(),
            }
        )

        if save_output_files:
            est_target_1 = est_targets[:, 0, :]
            est_target_2 = est_targets[:, 1, :]

            # linear rms level and scaling factor
            est_target_1 = rms_normalize(est_target_1)
            est_target_2 = rms_normalize(est_target_2)

            sf.write(
                os.path.join(
                    OUTPUT_FILES_PATH,
                    f"{audio_names[i]}_{audio_names[j]}_est_target_1.wav",
                ),
                est_target_1.view(-1).cpu(),
                samplerate,
            )
            sf.write(
                os.path.join(
                    OUTPUT_FILES_PATH,
                    f"{audio_names[i]}_{audio_names[j]}_est_target_2.wav",
                ),
                est_target_2.view(-1).cpu(),
                samplerate,
            )

            sf.write(
                os.path.join(
                    OUTPUT_FILES_PATH, f"{audio_names[i]}_{audio_names[j]}_target_1.wav"
                ),
                rms_normalize(input_1_segments).view(-1).cpu(),
                samplerate,
            )
            sf.write(
                os.path.join(
                    OUTPUT_FILES_PATH, f"{audio_names[i]}_{audio_names[j]}_target_2.wav"
                ),
                rms_normalize(input_2_segments).view(-1).cpu(),
                samplerate,
            )


with open(OUTPUT_HEATMAP_PATH, "w") as f:
    heatmap = {
        "rows": sorted(audio_names),
        "cols": sorted(audio_names),
        "values": heatmap_sisdrs,
    }
    json.dump(heatmap, f)

print(f"Heatmap saved to {OUTPUT_HEATMAP_PATH}")
