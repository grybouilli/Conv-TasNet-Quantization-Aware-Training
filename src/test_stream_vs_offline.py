import torch
import torch.nn as nn
import torch.nn.functional as F
from model.norm import cLN, NormType, cLN_EMA
from model.convtasnet import ConvTasNet
from model.streamable_convtasnet import (
    StreamableConvTasNet,
    StreamableConvTasNetcLN,
    StreamingEMAcLN,
)
from argparse import ArgumentParser, BooleanOptionalAction
from utils.utils import construct_model, load_trained_model
from utils.conf import Conf
import yaml
from pathlib import Path
import numpy as np
import soundfile as sf
import random
import os
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio


def load_random_mixture(
    path: str | Path, id: str | None, task: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a random speech file from mixed and noise file from noise, then mix them.

    Audios are truncated to the length of the shorter file.

    Args:
        path: Root directory containing the mixed/ and noise/ folders.

    Returns:
        A mono audio array containing the mixture.
    """
    path = Path(path)

    mix_folder = "mix_both" if task == "sep_noisy" else "mix_clean"

    mixed_files = list((path / mix_folder).glob("*.wav"))

    if not mixed_files:
        raise FileNotFoundError(
            f"No WAV files found in {(path / mix_folder)} or {(path / "s1")}."
        )
    if id == None:
        mixed_file = random.choice(mixed_files)
        s1_file = str(mixed_file).replace(mix_folder, "/s1/")
        s2_file = str(mixed_file).replace(mix_folder, "/s2/")
    else:
        mixed_file = os.path.join(path, f"{mix_folder}", id)
        s1_file = str(mixed_file).replace(mix_folder, "/s1/")
        s2_file = str(mixed_file).replace(mix_folder, "/s2/")

    speech, sr_speech = sf.read(mixed_file, dtype=np.float32)
    s1, sr_s1 = sf.read(s1_file, dtype=np.float32)
    s2, sr_s2 = sf.read(s2_file, dtype=np.float32)

    if sr_speech != sr_s1 or sr_speech != sr_s2:
        raise ValueError(
            f"Sample rates differ: speech={sr_speech}, s1={sr_s1} s2={sr_s2}"
        )

    # Convert stereo audio to mono if necessary.
    if speech.ndim == 2:
        speech = speech.mean(axis=1)
    if s1.ndim == 2:
        s1 = s1.mean(axis=1)
    if s2.ndim == 2:
        s2 = s2.mean(axis=1)
    return speech.reshape((1, -1)), s1.reshape((1, -1)), s2.reshape((1, -1))


import torch
import torch.nn.functional as F


@torch.no_grad()
def non_stream_ref(model, mix, N=32000, delay=32):
    """
    Process arbitrary-length audio using a fixed-length ConvTasNet.

    Args:
        model: ConvTasNet expecting [B, N] input.
        mix:   [B, L] input audio.
        N:     Model input length (2 seconds at 16 kHz).
        delay: Initial alignment delay.

    Returns:
        [B, ..., L] processed audio.
    """
    L = mix.shape[-1]

    # Left padding to account for the initial delay.
    x = F.pad(mix, (delay, 0))

    outputs = []

    for start in range(0, x.shape[-1], N):
        chunk = x[..., start : start + N]

        # Pad last chunk to fixed model input length.
        if chunk.shape[-1] < N:
            chunk = F.pad(chunk, (0, N - chunk.shape[-1]))

        y = model(chunk)
        outputs.append(y)

    # Concatenate processed chunks.
    y = torch.cat(outputs, dim=-1)

    # Remove initial delay and truncate to original length.
    return y[..., delay : delay + L]


@torch.no_grad()
def stream_loop(
    model, mix, audio_states, tcn_states, tail_states, cln_states=None, N=128, delay=32
):
    # mix: [B, L]; audio_states already provides the left zero pad
    L = mix.shape[-1]
    n_calls = -(-(L + delay) // N)  # ceil((L + delay) / N)
    mix = F.pad(mix, (0, n_calls * N - L))  # right zeros = flush (+ fill last chunk)

    outs = []
    for i in range(n_calls):
        x = mix[..., i * N : (i + 1) * N]
        if cln_states is None:
            y, audio_states, tcn_states, tail_states = model(
                x, audio_states, tcn_states, tail_states
            )
        else:
            y, audio_states, tcn_states, cln_states, tail_states = model(
                x, audio_states, tcn_states, cln_states, tail_states
            )
        outs.append(y)  # [B, n_src, N]

    out = torch.cat(outs, dim=-1)
    return out[..., delay : delay + L]  # drop 32-sample algorithmic latency


def print_model_info(model: nn.Module, inputs: tuple, label: str) -> None:
    # from fvcore.nn import FlopCountAnalysis

    print(f"--- {label} info ---")
    model_param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters count: {model_param_count}")
    # flops = FlopCountAnalysis(model, inputs)
    # print(f"MACs: {flops.total()/1e6:.2f} M")

    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc:
        model(*inputs)

    macs = fc.get_total_flops() / 2  # this counts true FLOPs, so halve it for MACs
    print(f"MACs: {macs/1e6:.2f} M")

parser = ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("input", type=str)
parser.add_argument("--input_id", type=str, default=None)
parser.add_argument("output_dir", type=str)
parser.add_argument("--ckpt", action=BooleanOptionalAction, default=False)
args = parser.parse_args()

with open(args.config) as f:
    conf_data = yaml.safe_load(f)

conf = Conf(conf_data)
model = load_trained_model(
    conf, eval=False, compile=args.ckpt, config_name=None, trained_model=args.model
)
model.eval()

# N = 15968
mix, s1, s2 = load_random_mixture(args.input, args.input_id, conf.data.task)
mix_t, s1_t, s2_t = (
    torch.from_numpy(mix),
    torch.from_numpy(s1),
    torch.from_numpy(s2),
)
off_out = non_stream_ref(model, mix_t)
# off_out = off_out / off_out.norm()

print_model_info(
    model, (mix_t[..., : conf.data.segment * conf.data.sample_rate],), "Offline model"
)

norm_type = None
if isinstance(model.tcn.layer_norm, nn.BatchNorm1d):
    streamable = StreamableConvTasNet(model)
    norm_type = NormType.batch_norm
elif isinstance(model.tcn.layer_norm, cLN):
    streamable = StreamableConvTasNetcLN(model)
    norm_type = NormType.channel_layer_norm
elif isinstance(model.tcn.layer_norm, cLN_EMA):
    streamable = StreamableConvTasNetcLN(model, StreamingEMAcLN)
    norm_type = NormType.channel_layer_norm
else:
    raise Exception("Non streamable layer norm in TCN")
streamable.eval()

if norm_type == NormType.batch_norm:
    print("Streeaming with BN")
    audio_state, tcn_states, tail_states = streamable.init_states()
    print_model_info(
        streamable,
        (
            mix_t[..., :128],
            audio_state,
            tcn_states,
            tail_states,
        ),
        "Streamable model",
    )
    stream_out = stream_loop(
        streamable,
        mix_t,
        audio_state,
        tcn_states,
        tail_states,
        delay=audio_state.shape[-1],
    )
else:
    print("Streaming with cLN")
    audio_state, tcn_states, cln_states, tail_states = streamable.init_states()
    print_model_info(
        streamable,
        (
            mix_t[..., :128],
            audio_state,
            tcn_states,
            cln_states,
            tail_states,
        ),
        "Streamable model",
    )
    stream_out = stream_loop(
        streamable, mix_t, audio_state, tcn_states, tail_states, cln_states,delay=audio_state.shape[-1]
    )
# stream_out = stream_out / stream_out.norm()
stream_out = stream_out / stream_out.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)

print(f"Output all same: {torch.allclose(off_out, stream_out, atol=10**-5)}")
print(
    f"Output 1 all same: {torch.allclose(off_out[0,0,:], stream_out[0,0,:], atol=10**-5)}"
)
print(
    f"Output 2 all same: {torch.allclose(off_out[0,1,:], stream_out[0,1,:], atol=10**-5)}"
)
diff = (off_out - stream_out).abs()

print("Max:", diff.max().item())
print("Mean:", diff.mean().item())
print("RMSE:", diff.square().mean().sqrt().item())
print("Above 1e-6:", (diff > 1e-6).sum().item())
print("Above 1e-5:", (diff > 1e-5).sum().item())

print("#################")

print(
    f"SI-SDR Offline channel 00: {scale_invariant_signal_distortion_ratio(off_out[:, 0, :], s1_t)}"
)
print(
    f"SI-SDR Offline channel 01: {scale_invariant_signal_distortion_ratio(off_out[:, 0, :], s2_t)}"
)
print(
    f"SI-SDR Offline channel 10: {scale_invariant_signal_distortion_ratio(off_out[:, 1, :], s1_t)}"
)
print(
    f"SI-SDR Offline channel 11: {scale_invariant_signal_distortion_ratio(off_out[:, 1, :], s2_t)}"
)
print(
    f"SI-SDR Streaming channel 00: {scale_invariant_signal_distortion_ratio(stream_out[:, 0, :], s1_t)}"
)
print(
    f"SI-SDR Streaming channel 01: {scale_invariant_signal_distortion_ratio(stream_out[:, 0, :], s2_t)}"
)
print(
    f"SI-SDR Streaming channel 10: {scale_invariant_signal_distortion_ratio(stream_out[:, 1, :], s1_t)}"
)
print(
    f"SI-SDR Streaming channel 11: {scale_invariant_signal_distortion_ratio(stream_out[:, 1, :], s2_t)}"
)
import matplotlib.pyplot as plt

n_off_out, n_stream_out = off_out.detach().numpy(), stream_out.detach().numpy()
fig = plt.figure()
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(s1[0, :])
ax1.set_title("Source 1")

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(s2[0, :])
ax2.set_title("Source 2")

ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(n_off_out[0, 0, :])
ax3.set_title("Offline output 1")

ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(n_stream_out[0, 0, :])
ax4.set_title("Stream output 1")

ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(n_off_out[0, 1, :])
ax5.set_title("Offline output 2")

ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(n_stream_out[0, 1, :])
ax6.set_title("Stream output 2")
fig.tight_layout(pad=0.5)
# Save the full figure...
fig.savefig(os.path.join(args.output_dir, "full_figure.png"))

sf.write(
    os.path.join(args.output_dir, "clean_s1.wav"),
    s1.squeeze(),
    conf.data.sample_rate,
    subtype="FLOAT",
)
sf.write(
    os.path.join(args.output_dir, "clean_s2.wav"),
    s2.squeeze(),
    conf.data.sample_rate,
    subtype="FLOAT",
)
sf.write(
    os.path.join(args.output_dir, "mix.wav"),
    mix.squeeze(),
    conf.data.sample_rate,
    subtype="FLOAT",
)
sf.write(
    os.path.join(args.output_dir, "off_output_chan1.wav"),
    n_off_out[0][0][:],
    conf.data.sample_rate,
    subtype="FLOAT",
)
sf.write(
    os.path.join(args.output_dir, "off_output_chan2.wav"),
    n_off_out[0][1][:],
    conf.data.sample_rate,
    subtype="FLOAT",
)

sf.write(
    os.path.join(args.output_dir, "stream_output_chan1.wav"),
    n_stream_out[0][0][:],
    conf.data.sample_rate,
    subtype="FLOAT",
)
sf.write(
    os.path.join(args.output_dir, "stream_output_chan2.wav"),
    n_stream_out[0][1][:],
    conf.data.sample_rate,
    subtype="FLOAT",
)
print(f"All saved to {args.output_dir}")
