import dataclasses
from typing import Tuple

from simple_parsing import ArgumentParser


@dataclasses.dataclass
class WandbConfig:
    entity: str = "afmck"
    project: str = "specgrad-dev"


@dataclasses.dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 4e-4
    beta_start: float = 1e-4
    beta_end: float = 5e-2
    beta_schedule: str = "linear"
    batch_size: int = 512


# TODO: some params can be derived from data directly
@dataclasses.dataclass
class ModelConfig:
    in_channels: int = 1
    residual_channels: int = 64
    num_residual_layers: int = 30
    dilation_cycle_length: int = 10
    max_timesteps: int = 50  # TODO: can probably be moved to training
    diffusion_embedding_dim: int = 64
    diffusion_projection_dim: int = 512
    leaky_relu_slope: float = 0.4
    spec_upsample_strides: Tuple[int] = (10, 30)  # TODO: assert product equal to hop_length
    n_mels: int = 128


@dataclasses.dataclass
class DataConfig:
    root_dir: str = "dataset/ljspeech"
    sampling_rate: int = 24_000
    n_fft: int = 2048
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 12_000
    sample_length: int = 36_000
    window_length: int = 1200
    hop_length: int = 300
    lifter_order: int = 24
    envelope_min: int = 0.1
    micro_batch_size: int = 8  # TODO: move to training
    num_workers: int = 4


@dataclasses.dataclass
class Config:
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    wandb: WandbConfig = WandbConfig()
