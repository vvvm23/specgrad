from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

import simple_parsing
import torch
from accelerate import Accelerator
from diffusers.schedulers import DDPMScheduler
from tqdm import tqdm

from config import Config
from data import get_dataset, transform_noise
from model import SpecGrad
from utils import init_wandb, setup_directory


def main(args, config: Config):
    assert config.training.batch_size % config.data.micro_batch_size == 0
    gradient_accumulation_steps = config.training.batch_size // config.data.micro_batch_size
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    print("> config:")
    print(config)
    print(args)

    print("> accelerator:", accelerator)

    print("> loading dataset")
    train_dataset, train_dataloader = get_dataset(config, split="train")
    valid_dataset, valid_dataloader = get_dataset(config, split="valid")
    print("> train dataset length:", len(train_dataset))
    print("> valid dataset length:", len(valid_dataset))

    print("> initialising model")
    model = SpecGrad(**vars(config.model))
    optim = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    print("> number of parameters:", sum(torch.numel(p) for p in model.parameters()))

    noise_scheduler = DDPMScheduler(
        config.model.max_timesteps,
        beta_start=config.training.beta_start,
        beta_end=config.training.beta_end,
        beta_schedule=config.training.beta_schedule,
    )

    model, optim, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optim, train_dataloader, valid_dataloader
    )

    if args and args.resume_dir:
        accelerator.load_state(args.resume_dir)

    if accelerator.is_main_process:
        print("> initialising weights and biases")
        wandb = init_wandb(config)
        exp_dir = setup_directory()
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

    # TODO: calculate log mel spectrogram mean absolute error (LS-MAE) to compare with PriorGrad
    # but requires a full inference loop la
    def ls_mae():
        pass

    for eid in range(config.training.epochs):
        print("> epoch", eid)
        train_loss = 0.0
        model.train()
        pb = tqdm(train_dataloader)
        for i, batch in enumerate(pb):
            with accelerator.accumulate(model):
                optim.zero_grad()
                loss = loss_fn(model, batch)
                accelerator.backward(loss)
                optim.step()

                train_loss += loss.item()
                pb.set_description(f"train loss: {train_loss / (i+1):.4f}")

        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            pb = tqdm(valid_dataloader)
            for i, batch in enumerate(pb):
                loss = loss_fn(model, batch)
                loss = accelerator.gather_for_metrics(loss).mean()
                valid_loss += loss.item()
                pb.set_description(f"valid loss: {valid_loss / (i+1):.4f}")

        wandb.log(
            {
                "train": {"loss": train_loss / len(train_dataloader)},
                "valid": {"loss": valid_loss / len(valid_dataloader)},
            }
        )

        if accelerator.is_local_main_process:
            accelerator.save_state(exp_dir / f"checkpoint-{eid:04}.pt")
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--resume-dir", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    if args.config_path:
        config = Config.load(args.config_path)
    main(args, config)
