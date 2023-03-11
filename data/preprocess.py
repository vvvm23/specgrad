import os
import random
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import soundfile
import torch
from numpy.typing import NDArray

from utils import str_to_path


def lifter(M, r):
    LI = torch.sin(torch.pi * torch.arange(1, 1 + M.shape[0], dtype=M.dtype) / r)
    LI = LI.reshape(LI.shape[0], 1)

    return M * (1 + (r / 2) * LI)


def get_random_offset(f: soundfile.SoundFile, sample_length: int, sr: int) -> int:
    file_sample_length = int(sample_length * f.samplerate / sr)
    return random.randint(0, max(0, f.frames - file_sample_length - 1))


def load_waveform(
    path: Union[str, Path], sr: int = 24_000, sample_length: int = -1, random_clip: bool = False
) -> NDArray[np.float32]:
    f = soundfile.SoundFile(path, mode="r")

    if random_clip:
        offset = get_random_offset(f, sample_length, sr)
    else:
        offset = 0

    raw, _ = soundfile.read(path, start=offset, frames=max(-1, int(sample_length * f.samplerate / sr)), dtype="float32")
    raw = librosa.to_mono(raw.T)
    raw = librosa.resample(raw, orig_sr=f.samplerate, target_sr=sr)
    raw = librosa.util.normalize(raw) * 0.95
    raw = np.pad(raw, (0, sample_length - raw.shape[0]), constant_values=0)
    assert raw.shape[0] == sample_length, f"{raw.shape[0]}, {sample_length}"

    return raw


# TODO: switch to manual calculating using torch for speed, also use for LS-MAE loss
def calculate_mel(
    raw: NDArray[np.float32],
    sr: int = 24_000,
    window_length: int = 1200,
    hop_length: int = 300,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int = 12_000,
) -> torch.FloatTensor:
    mel = librosa.feature.melspectrogram(
        y=raw,
        sr=sr,
        win_length=window_length,
        hop_length=hop_length + 1,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        norm=2,
        center=True,
    )
    return torch.from_numpy(mel)


def get_pinv_mel_basis(
    sr: int = 24_000, n_fft: int = 2048, n_mels: int = 128, fmin: int = 20, fmax: int = 12_000
) -> torch.FloatTensor:
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel_basis)
    return torch.linalg.pinv(mel_basis)


def calculate_tf_filter(
    mel: torch.FloatTensor,
    pinv_mel_basis: torch.FloatTensor,
    lifter_order: int = 24,
    envelope_min: float = 0.01,
) -> torch.FloatTensor:
    power_spec = pinv_mel_basis @ mel
    liftered_spec = lifter(power_spec, r=lifter_order)
    normalised_spec = torch.clamp(liftered_spec, min=envelope_min)
    return normalised_spec


# TODO: can we torch jit compile
# TODO: check this works after changes visually in a nb
def transform_noise(
    filter_coefficients: torch.FloatTensor,
    noise: torch.FloatTensor,
    n_fft: int = 2048,
    window_length: int = 1200,
    hop_length: int = 300,
    post_norm: bool = True,
) -> torch.FloatTensor:
    noise_spec = torch.stft(
        noise,
        n_fft=n_fft,
        hop_length=hop_length + 1,
        win_length=window_length,
        return_complex=True,
        normalized=True,
        center=True,
    )

    noise_spec *= filter_coefficients
    transformed_noise = torch.istft(
        noise_spec,
        hop_length=hop_length + 1,
        win_length=window_length,
        n_fft=n_fft,
        center=True,
        length=noise.shape[-1],
    )
    if post_norm:
        transformed_noise = (transformed_noise - transformed_noise.min()) / (
            transformed_noise.max() - transformed_noise.min()
        ) * 2.0 - 1.0
    return transformed_noise
