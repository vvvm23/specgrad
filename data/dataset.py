from torch.utils.data import Dataset, DataLoader
from config.config import Config


class SpecGradDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        pass


def get_dataloader(config: Config):
    pass
