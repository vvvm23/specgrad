import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from config.config import Config

from pathlib import Path

from typing import Union

# TODO: compressed version
# TODO: experiment with on-the-fly mel and augmentation for speed
class SpecGradDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path]):
        super().__init__()
        meta_path = root_dir / "metadata.csv"
        self.entries = [l.rstrip() for l in open(meta_path, mode="r").readlines()]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.entries)

    """
        TODO: here, we have to prechunk our waveform and mels, so not to compute filter during training.
        However, we have less augmentation options, index options, and use more disk space to precompute.
        Which approach is better? Computing the filter is quite expensive currently.
        Maybe look at prior grad for inspiration, and just bite the bullet computing filter
        If we use pinv it is possible to do on the fly, but if we use nnls with librosa it is slow :/
    """

    def __getitem__(self, idx: int):
        path = self.entries[idx]
        archive = np.load(path)

        waveform, mel_spectrogram, M = archive["waveform"], archive["mel_spectrogram"], archive["filter"]
        waveform, mel_spectrogram, M = map(torch.from_numpy, (waveform, mel_spectrogram, M))

        archive.close()

        return waveform, mel_spectrogram, M


def get_dataloader(config: Config):
    pass
