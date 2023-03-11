import logging

from accelerate.logging import get_logger

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

from pathlib import Path
from typing import Optional

import simple_parsing
import soundfile
import torch
from accelerate import Accelerator
from tqdm import tqdm

from config import Config
from data.preprocess import (
    calculate_mel,
    calculate_tf_filter,
    get_pinv_mel_basis,
    load_waveform,
    transform_noise,
)
from model import SpecGrad
from scheduler import SpecGradDDPMScheduler


@torch.no_grad()
def inference(
    model: SpecGrad,
    mel_spectrogram: torch.FloatTensor,
    config: Config,
    pinv_mel_basis: Optional[torch.FloatTensor] = None,
    accelerator: Optional[Accelerator] = None,
    return_numpy: bool = True,
):
    waveform = torch.randn(mel_spectrogram.shape[0], config.data.sample_length)
    if pinv_mel_basis is None:
        pinv_mel_basis = get_pinv_mel_basis(
            config.data.sampling_rate, config.data.n_fft, config.data.n_mels, config.data.fmin, config.data.fmax
        )

    waveform = waveform.to(accelerator.device)
    mel_spectrogram = mel_spectrogram.to(accelerator.device)
    pinv_mel_basis = pinv_mel_basis.to(accelerator.device)

    filter_coefficients = calculate_tf_filter(
        mel_spectrogram, pinv_mel_basis, lifter_order=config.data.lifter_order, envelope_min=config.data.envelope_min
    )
    filter_coefficients = filter_coefficients.to(accelerator.device)

    waveform = transform_noise(
        filter_coefficients,
        waveform,
        n_fft=config.data.n_fft,
        window_length=config.data.window_length,
        hop_length=config.data.hop_length,
    )

    scheduler = SpecGradDDPMScheduler(
        config.model.max_timesteps,
        beta_start=config.training.beta_start,
        beta_end=config.training.beta_end,
        beta_schedule=config.training.beta_schedule,
    )

    for t in tqdm(scheduler.timesteps):
        noise_pred = model(waveform, mel_spectrogram, t)
        waveform = scheduler.step(noise_pred, t, waveform, filter_coefficients, config.data)

    waveform = waveform.clamp(-1, 1)
    if return_numpy:
        waveform = waveform.float().cpu().numpy()
    waveform = waveform[:, : config.data.sample_length]
    return waveform


# TODO: really need to generally refactor to pass around dict of configs
def main(args, config: Config):
    accelerator = Accelerator()
    model = SpecGrad(**vars(config.model))
    model = accelerator.prepare(model)
    if args.resume_dir:
        accelerator.load_state(args.resume_dir)

    waveform = load_waveform(
        args.input_file, sr=config.data.sampling_rate, sample_length=config.data.sample_length, random_clip=False
    )

    mel_spec = calculate_mel(
        waveform,
        sr=config.data.sampling_rate,
        window_length=config.data.window_length,
        hop_length=config.data.hop_length,
        n_mels=config.data.n_mels,
        fmin=config.data.fmin,
        fmax=config.data.fmax,
    ).unsqueeze(0)

    recon = inference(model, mel_spec, config, accelerator=accelerator).T
    soundfile.write("output.wav", recon, config.data.sample_length, subtype="PCM_16")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--resume-dir", type=str, default=None)
    args = parser.parse_args()

    # TODO: hook config into args
    config = Config()
    if args.config_path:
        config = Config.load(args.config_path)
    main(args, config)
