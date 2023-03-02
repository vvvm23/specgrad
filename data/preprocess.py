from typing import Union
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from utils import str_to_path


def lifter(M, r):
    LI = np.sin(np.pi * np.arange(1, 1 + M.shape[0], dtype=M.dtype) / r)
    LI = LI.reshape(LI.shape[0], 1)

    return M * (1 + (r / 2) * LI)


def load_waveform(path: Union[str, Path]) -> NDArray[np.float32]:
    path = str_to_path(path)


def calculate_mel(
    raw: NDArray[np.float32],
    sr: int = 24_000,
    window_length: int = 1200,
    hop_length: int = 300,
    fmin: int = 20,
    fmax: int = 12_000,
) -> NDArray[np.float32]:
    pass


def calculate_tf_filter(
    mel: NDArray[np.float32],
    sr: int = 24_000,
    fmin: int = 20,
    fmax: int = 12_000,
    lifter_power: int = 24,
    envelope_min: float = 0.01,
) -> NDArray[np.float32]:
    pass
