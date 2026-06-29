import os

import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

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

config_name = "all_S8S8"


CONF_PATH = f"exp/ConvTasNet_0.001_{config_name}/conf.yml"

with open(CONF_PATH) as f:
    conf_data = yaml.safe_load(f)

conf = Conf(conf_data)


model = load_trained_model(conf=conf, eval=True, compile=True, config_name=config_name)

model = model.to(device)

file_name = "Test"

data, samplerate = sf.read("custom_audios/" + file_name + ".wav", dtype="float32")
print("Importing audio file:", file_name)
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
    segment = segment.unsqueeze(0).unsqueeze(0).to(device)
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

try:
    os.mkdir(os.path.join("custom_audios/", f"{file_name}/"))
except FileExistsError:
    print("Directory already exists")

path = os.path.join(
    f"custom_audios/{file_name}/", f"{config_name}_{file_name}_input.flac"
)
sf.write(path, data, samplerate)


path = os.path.join(
    f"custom_audios/{file_name}/", f"{config_name}_{file_name}_est_target_1.flac"
)
sf.write(path, est_target_1.view(-1).cpu(), samplerate)

path = os.path.join(
    f"custom_audios/{file_name}/", f"{config_name}_{file_name}_est_target_2.flac"
)
sf.write(path, est_target_2.view(-1).cpu(), samplerate)
