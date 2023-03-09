import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__file__)
accelerator = Accelerator()

from typing import Tuple

from config import Config
from data import get_dataset, transform_noise
from model import SpecGrad
from utils import init_wandb


def loss_fn(model: SpecGrad, batch, config: Config):
    waveform, mel_spectrogram, filter_coefficients = batch
    noise = torch.randn_like(waveform)
    noise = transform_noise(
        filter_coefficients,
        noise,
        n_fft=config.data.n_fft,
        window_length=config.data.window_length,
        hop_length=config.data.hop_length,
    )


def main(args, config: Config):
    logger.info("config:")
    logger.info(config)
    logger.info(args)

    logger.info("accelerator:", accelerator)

    logger.info("loading dataset")
    train_dataset, train_dataloader = get_dataset(config.data, split="train")
    test_dataset, test_dataloader = get_dataset(config.data, split="test")
    logger.info("train dataset length:", len(train_dataset))
    logger.info("test dataset length:", len(test_dataset))

    logger.info("initialising model")
    model = SpecGrad(**config.model)
    optim = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    logger.info("number of parameters:", sum(torch.numel(p) for p in model.parameters()))

    if accelerator.is_main_process:
        logger.info("initialising weights and biases")
        wandb = init_wandb(config)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(None, None)
