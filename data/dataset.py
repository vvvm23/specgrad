from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader, Dataset

from config.config import Config  # TODO: dataconfig specifically

from .preprocess import (
    calculate_mel,
    calculate_tf_filter,
    get_pinv_mel_basis,
    load_waveform,
)


# TODO: parameterise other functions (n_mels, n_ffts, etc.)
class SpecGradDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path], config: Config):
        super().__init__()
        root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        meta_path = root_dir / "metadata.txt"
        self.entries = [l.rstrip() for l in open(meta_path, mode="r").readlines()]
        self.root_dir = root_dir

        self.config = config
        self.pinv_mel_basis = get_pinv_mel_basis(config.sr)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        path = self.root_dir / self.entries[idx]
        waveform = load_waveform(path, sr=self.config.sr, sample_length=self.config.sample_length, random_clip=True)
        mel_spectrogram = calculate_mel(waveform, sr=self.config.sr)
        M = calculate_tf_filter(
            mel_spectrogram,
            self.pinv_mel_basis,
            lifter_order=self.config.lifter_order,
            envelope_min=self.config.envelope_min,
        )

        return waveform, mel_spectrogram, M


def get_dataset(config: Config, root_dir: Union[str, Path], shuffle: bool = True):
    dataset = SpecGradDataset(root_dir, config)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=True
    )
    return dataset, dataloader
