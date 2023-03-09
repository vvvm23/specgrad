import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm

logger = get_logger(__file__)
accelerator = Accelerator()

from diffusers.schedulers import DDPMScheduler

from config import Config
from data import get_dataset, transform_noise
from model import SpecGrad
from utils import init_wandb


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

    # TODO: does model.compile work with accelerate
    logger.info("initialising model")
    model = SpecGrad(**config.model)
    optim = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    logger.info("number of parameters:", sum(torch.numel(p) for p in model.parameters()))

    # TODO: config param this
    noise_scheduler = DDPMScheduler(config.model.max_timesteps, beta_start=1e-4, beta_end=0.05, beta_schedule="linear")

    # TODO: add checkpoint loading

    model, optim, train_dataloader = accelerator.prepare(model, optim, train_dataloader)

    if accelerator.is_main_process:
        logger.info("initialising weights and biases")
        wandb = init_wandb(config)
    accelerator.wait_for_everyone()

    def loss_fn(model: SpecGrad, batch):
        waveform, mel_spectrogram, filter_coefficients = batch
        N = waveform.shape[0]
        noise = torch.randn_like(waveform)
        noise = transform_noise(  # costs 2 STFTs
            filter_coefficients,
            noise,
            n_fft=config.data.n_fft,
            window_length=config.data.window_length,
            hop_length=config.data.hop_length,
        )

        timesteps = torch.randint(0, config.model.max_timesteps, (N,), dtype=torch.int64)
        noisy_waveform = noise_scheduler.add_noise(waveform, noise, timesteps)

        noise_pred = model(noisy_waveform, mel_spectrogram, timesteps)

        noise_diff = noise - noise_pred
        # TODO: optimise square error
        loss = (
            transform_noise(  # costs 2 STFTs
                1 / filter_coefficients,
                noise_diff,
                n_fft=config.data.n_fft,
                window_length=config.data.window_length,
                hop_length=config.data.hop_length,
            )
            .square()
            .mean()
        )

        return loss

    for eid in range(config.training.epochs):
        logger.info("epoch", eid)
        total_loss = 0.0
        model.train()
        pb = tqdm(enumerate(train_dataloader))
        for i, batch in pb:
            optim.zero_grad()
            loss = loss_fn(model, batch)
            accelerator.backward(loss)
            optim.step()

            total_loss += loss.item()
            pb.set_description(f"train loss: {total_loss / (i+1):.4f}")

        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            pb = tqdm(enumerate(test_dataloader))
            for i, batch in pb:
                loss = loss_fn(model, batch)
                total_loss += loss.item()
                pb.set_description(f"test loss: {total_loss / (i+1):.4f}")


if __name__ == "__main__":
    # TODO: parse config, don't just use default
    main(None, Config())
