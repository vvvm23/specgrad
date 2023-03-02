from typing import Union
from pathlib import Path


def str_to_path(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    return path
