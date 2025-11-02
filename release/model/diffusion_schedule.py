from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    if schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(min=1e-5, max=0.999)
    raise ValueError(f"Unsupported beta schedule: {schedule}")


def extract(a: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor

    def __post_init__(self) -> None:
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.alphas_cumprod.device), self.alphas_cumprod[:-1]],
            dim=0,
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def timesteps(self) -> int:
        return self.betas.shape[0]

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus * noise) / sqrt_alpha

    def ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        eps_pred, _ = model(x_t, t)
        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_next = extract(self.alphas_cumprod, t_next, x_t.shape)
        sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        dir_term = torch.sqrt(1 - alpha_next - sigma ** 2) * eps_pred
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        return torch.sqrt(alpha_next) * x0_pred + dir_term + sigma * noise
