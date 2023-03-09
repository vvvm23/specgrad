from pathlib import Path
from typing import Literal, Union

from torch.utils.data import DataLoader, Dataset

from config.config import DataConfig
from utils import str_to_path

from .preprocess import (
    calculate_mel,
    calculate_tf_filter,
    get_pinv_mel_basis,
    load_waveform,
)


# TODO: parameterise other functions (n_mels, n_ffts, etc.)
class SpecGradDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path], config: DataConfig):
        super().__init__()
        root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        meta_path = root_dir / "metadata.txt"
        self.entries = [l.rstrip() for l in open(meta_path, mode="r").readlines()]
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

        return waveform, mel_spectrogram, M


# TODO: refactor just to use HF dataset with preprocess set
# .. they also use a proprietry dataset so we can't exactly reproduce.
# use LJSpeech to compare with PriorGrad, but need different sampling rate and other preprocess params
def get_dataset(config: DataConfig, split: Literal["train", "test"] = "train"):
    dataset = SpecGradDataset(str_to_path(config.root_dir) / split, config)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=split == "train", num_workers=config.num_workers, pin_memory=True
    )
    return dataset, dataloader
