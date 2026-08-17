"""
Run quantized model on a 2-speaker audio file and save the separated audio files. The audio files should be in .wav format and have the same sample rate as defined in configuration file.

Usage:
    python src/run_on_audio_file.py model_archi quantization_config audio_file

Arguments:
    model_architecture:
        Model architecture identifier (e.g. ``0.5``, ``1.5``).

    quantization_config:
        Name of the quantization configuration to apply (e.g. ``all_S8S8``, ``all_S16S16``...).

    audio_file:
        Path to a 2-speaker audio file to be separated. The audio file should be in .wav format and have the same sample rate as defined in configuration file.

Examples:
    python src/run_on_audio_file.py 0.5 all_S8S8 custom_audios/2speakers.wav
"""

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


if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

model_archi = sys.argv[1]
config_name = sys.argv[2]
audio_file = sys.argv[3]

OUTPUT_FILES_PATH = os.path.join(
    os.path.dirname(audio_file),
    f"{os.path.splitext(os.path.basename(audio_file))[0]}_separated/",
)

CONF_PATH = f"exp/ConvTasNet_{model_archi}_{config_name}/conf.yml"

with open(CONF_PATH) as f:
    conf_data = yaml.safe_load(f)

conf = Conf(conf_data)


model = load_trained_model(conf=conf, eval=True, compile=True, config_name=config_name)

model = model.to(device)


data, samplerate = sf.read(audio_file, dtype="float32")
print("Importing audio file:", audio_file)
print(f"Data shape: {data.shape}, Sample rate: {samplerate}")

if samplerate != conf.data.sample_rate:
    raise ValueError(
        f"Sample rate of the audio file ({samplerate}) does not match the expected sample rate ({conf.data.sample_rate})."
    )

# Cut in 3s segments
segment_length = 3 * samplerate  # 3 seconds in samples
segments = []
for start in range(0, len(data), segment_length):
    end = start + segment_length
    segment = torch.from_numpy(data[start:end])  # .float()
    if len(segment) < segment_length:
        # Pad the last segment with zeros if it's shorter than 3 seconds
        segment = F.pad(segment, (0, segment_length - len(segment)), "constant")
    segment = segment.unsqueeze(0).to(device)
    segments.append(segment)

print(f"Number of segments: {len(segments)}")

input = torch.cat(segments, dim=0)

print(f"Input shape: {input.shape}")


with torch.no_grad():
    est_targets = model(input)


est_target_1 = est_targets[:, 0, :]
est_target_2 = est_targets[:, 1, :]

# linear rms level and scaling factor
est_target_1 = rms_normalize(est_target_1, rms_level_db=-20.0)
est_target_2 = rms_normalize(est_target_2, rms_level_db=-20.0)

print(
    f"est_target_1 shape: {est_target_1.shape}, est_target_2 shape: {est_target_2.shape}"
)

os.makedirs(OUTPUT_FILES_PATH, exist_ok=True)

path = os.path.join(
    OUTPUT_FILES_PATH,
    f"{config_name}_{os.path.splitext(os.path.basename(audio_file))[0]}_input.flac",
)
sf.write(path, data, samplerate)

# for i in range(len(est_target_1)):
#     path = os.path.join(
#         f"custom_audios/{file_name}/", f"{config_name}_{file_name}_est_target_1_segment_{i}.flac"
#     )
#     sf.write(path, est_target_1[i].cpu(), samplerate)

# for i in range(len(est_target_2)):
#     path = os.path.join(
#         f"custom_audios/{file_name}/", f"{config_name}_{file_name}_est_target_2_segment_{i}.flac"
#     )
#     sf.write(path, est_target_2[i].cpu(), samplerate)

path = os.path.join(
    OUTPUT_FILES_PATH,
    f"{config_name}_{os.path.splitext(os.path.basename(audio_file))[0]}_est_target_1.flac",
)
sf.write(path, est_target_1.view(-1).cpu(), samplerate)

path = os.path.join(
    OUTPUT_FILES_PATH,
    f"{config_name}_{os.path.splitext(os.path.basename(audio_file))[0]}_est_target_2.flac",
)
sf.write(path, est_target_2.view(-1).cpu(), samplerate)
