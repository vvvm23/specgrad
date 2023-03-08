import dataclasses

from simple_parsing import ArgumentParser


@dataclasses.dataclass
class TrainingConfig:
    pass


@dataclasses.dataclass
class ModelConfig:
    pass


@dataclasses.dataclass
class DataConfig:
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


@dataclasses.dataclass
class Config:
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
