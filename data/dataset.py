from pathlib import Path
from typing import Literal, Union

import torch
from torch.utils.data import DataLoader, Dataset

from config.config import DataConfig
from utils import str_to_path

from .preprocess import (
    calculate_mel,
    calculate_tf_filter,
    get_pinv_mel_basis,
    load_waveform,
)


class SpecGradDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path], meta_path: str, config: DataConfig):
        super().__init__()
        root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.entries = [l.rstrip() for l in open(root_dir / meta_path, mode="r").readlines()]
        self.root_dir = root_dir

        self.config = config
        self.pinv_mel_basis = get_pinv_mel_basis(
            sr=config.sampling_rate, n_fft=config.n_fft, n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        path = self.root_dir / self.entries[idx]
        waveform = load_waveform(
            path, sr=self.config.sampling_rate, sample_length=self.config.sample_length, random_clip=True
        )
        mel_spectrogram = calculate_mel(
            waveform,
            sr=self.config.sampling_rate,
            window_length=self.config.window_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )
        M = calculate_tf_filter(
            mel_spectrogram,
            self.pinv_mel_basis,
            lifter_order=self.config.lifter_order,
            envelope_min=self.config.envelope_min,
        )

        return torch.from_numpy(waveform), mel_spectrogram, M


def get_dataset(config: DataConfig, split: Literal["train", "valid", "test"] = "train"):
    dataset = SpecGradDataset(str_to_path(config.root_dir), f"{split}.txt", config)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=split == "train", num_workers=config.num_workers, pin_memory=True
    )
    return dataset, dataloader


if __name__ == "__main__":
    from config import DataConfig

    from .preprocess import transform_noise

    dataset, dataloader = get_dataset(DataConfig, split="train")

    waveform, mel_spec, M = dataset.__getitem__(0)
    print(waveform.shape)
    print(mel_spec.shape)
    print(M.shape)

    noise = torch.randn_like(waveform.unsqueeze(0))
    noise = transform_noise(M.unsqueeze(0), noise)
    print(noise.shape)
