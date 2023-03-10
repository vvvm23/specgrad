from datetime import datetime
from pathlib import Path
from typing import Union

import wandb
from config import TrainingConfig


def str_to_path(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    return path


def init_wandb(config: TrainingConfig):
    if config.wandb is None:
        return wandb.init(mode="disabled")

    return wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        resume="auto",
    )


def setup_directory(base="exp"):
    root_dir = Path(base)
    root_dir.mkdir(exist_ok=True)

    save_id = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    exp_dir = root_dir / save_id
    exp_dir.mkdir(exist_ok=True)
    return exp_dir
