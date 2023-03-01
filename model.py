# Adapted from PriorGrad-vocoder as part of the Microsoft NeuralSpeech project:
# https://github.com/microsoft/NeuralSpeech/blob/master/PriorGrad-vocoder/model.py
# Originally licensed under the Apache 2.0 license:
# http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from typing import Optional, Union


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps: int, embedding_dim: int = 64, out_dim: int = 512):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(max_steps, embedding_dim), persistent=False)
        self.projection1 = nn.Linear(embedding_dim * 2, out_dim)
        self.projection2 = nn.Linear(out_dim, out_dim)

    def forward(self, diffusion_step: Union[torch.LongTensor, torch.FloatTensor]):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t: torch.FloatTensor):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps: int, dim: int = 64):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(dim).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / (dim - 1.0))  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=-1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, stride: int = 16, leaky_relu_slope: float = 0.4, num_layers: int = 2):
        super().__init__()
        strides = (10, 30)  # TODO: need to param this properly

        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    # nn.ConvTranspose2d(1, 1, [3, 2 * stride], stride=[1, stride], padding=[1, stride // 2]),
                    nn.ConvTranspose2d(1, 1, [3, 2 * stride], stride=[1, stride], padding=[1, stride // 2]),
                    nn.LeakyReLU(leaky_relu_slope),
                )
                # for _ in range(num_layers)
                for stride in strides
            ]
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.layers(x)
        x = x.squeeze(dim=1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels: int, residual_channels: int, dilation: int, diffusion_projection_dim: int = 512):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(diffusion_projection_dim, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(
        self,
        x: torch.FloatTensor,
        conditioner: torch.FloatTensor,
        diffusion_step: Union[torch.LongTensor, torch.FloatTensor],
    ):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = y.chunk(2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class SpecGrad(nn.Module):
    def __init__(
        self,
        residual_channels: int = 64,
        num_residual_layers: int = 30,
        n_mels: int = 80,
        use_prior: bool = True,
        dilation_cycle_length: int = 10,
        max_timesteps: int = 50,
        diffusion_embedding_dim: int = 64,
        diffusion_projection_dim: int = 512,
        leaky_relu_slope: float = 0.4,
        spec_upsample_stride: int = 16,
        num_spec_upsample_layers: int = 2,
    ):
        super().__init__()
        self.use_prior = use_prior
        self.n_mels = n_mels

        self.input_projection = Conv1d(1, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(
            max_timesteps, embedding_dim=diffusion_embedding_dim, out_dim=diffusion_projection_dim
        )
        self.spectrogram_upsampler = SpectrogramUpsampler(
            stride=spec_upsample_stride, leaky_relu_slope=leaky_relu_slope, num_layers=num_spec_upsample_layers
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    n_mels=n_mels,
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    diffusion_projection_dim=diffusion_projection_dim,
                )
                for i in range(num_residual_layers)
            ]
        )
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        audio: torch.FloatTensor,
        spectrogram: torch.FloatTensor,
        diffusion_step: Union[torch.LongTensor, torch.FloatTensor],
    ):
        x = audio.unsqueeze(dim=1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, spectrogram, diffusion_step)
            skip.append(skip_connection)

        x = torch.stack(skip).sum(dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


if __name__ == "__main__":
    import librosa

    wav, _ = librosa.load("test.wav", sr=24_000, mono=True)
    offset, length = 10_000, 36_000
    wav = wav[offset : offset + length]

    # TODO: why does mel add +1?
    mel = librosa.feature.melspectrogram(
        y=wav, sr=24_000, hop_length=300, win_length=1200, fmin=20, fmax=12_000, power=2
    )[:, :-1]

    wav = torch.from_numpy(wav).unsqueeze(0)
    mel = torch.from_numpy(mel).unsqueeze(0)
    t = torch.FloatTensor([0.5])

    model = SpecGrad(n_mels=128, dilation_cycle_length=10)
    y = model(wav, mel, t)
    print(wav)
    print(y)
    print(y.min(), y.max(), y.mean(), y.shape)
