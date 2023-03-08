from pathlib import Path
from typing import Union


def str_to_path(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    return path
