"""
Unified TG-2S-DARU denoiser used across Stage-1 (coarse SMT) and Stage-2 (fine
block) diffusion models.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=device) / max(half_dim - 1, 1)
        angles = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FixedDiff(nn.Module):
    """Compute first and second order differences for DARU gating."""

    def __init__(self, channels: int = 2):
        super().__init__()
        k1 = torch.tensor([-1.0, 1.0], dtype=torch.float32).view(1, 1, 2)
        k2 = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, 3)
        self.register_buffer("k1", k1)
        self.register_buffer("k2", k2)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad1 = F.pad(x, (1, 0))
        d1 = F.conv1d(pad1, self.k1.repeat(self.channels, 1, 1), groups=self.channels)
        pad2 = F.pad(x, (1, 1))
        d2 = F.conv1d(pad2, self.k2.repeat(self.channels, 1, 1), groups=self.channels)
        return d1, d2


class JitterBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, norm_groups: int = 8, dilation: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, b = self.conv(h).chunk(2, dim=1)
        h = torch.tanh(a) * F.silu(b)
        h = self.dropout(h)
        return self.out(h) + x


class DriftBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, norm_groups: int = 8, dilation: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dw(h)
        h = F.silu(h)
        h = self.dropout(h)
        return self.pw(h) + x


class BranchStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        block_ctor,
        dropout: float,
        norm_groups: int,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        if in_channels != out_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        for _ in range(num_blocks):
            layers.append(block_ctor(out_channels, dropout=dropout, norm_groups=norm_groups))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateNet(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        bottleneck = max(hidden_channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv1d(3, bottleneck, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(bottleneck, 1, kernel_size=1),
        )

    def forward(self, speed: torch.Tensor, curvature: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([speed, curvature, variance], dim=1)))


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, widths: List[int]):
        super().__init__()
        self.cond_dim = cond_dim
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cond_dim, width * 2),
                    nn.SiLU(),
                    nn.Linear(width * 2, width * 2),
                )
                for width in widths
            ]
        )

    def forward(self, cond: torch.Tensor) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        outputs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for mlp in self.layers:
            params = mlp(cond)
            gamma, beta = params.chunk(2, dim=-1)
            outputs.append((gamma, beta))
        return outputs


class UnifiedTG2SDARU(nn.Module):
    """
    Unified denoiser usable for both Stage-1 coarse diffusion and Stage-2 block diffusion.

    Conditioning dictionary keys:
        - extra_channels: (B, extra_channels, T) optional additional input channels.
        - global_cond: (B, global_cond_dim) optional global context vector (FiLM modulation).
        - temporal_features: (B, C, T) optional temporal embeddings added before stem.
    """

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        cycle_length: int,
        base_channels: int = 128,
        scales: int = 4,
        jitter_blocks: int = 2,
        drift_blocks: int = 2,
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        cond_embed_dim: int = 256,
        stay_head: bool = False,
        extra_channels: int = 0,
        global_cond_dim: int = 0,
        norm_groups: int = 8,
    ):
        super().__init__()
        self.base_input_channels = min(input_channels, 2)
        self.extra_channels = extra_channels
        self.total_in_channels = input_channels + extra_channels
        self.sequence_length = sequence_length
        self.cycle_length = cycle_length
        self.stay_head_enabled = stay_head
        self.global_cond_dim = global_cond_dim
        self.cond_embed_dim = cond_embed_dim

        widths = [base_channels * (2 ** i) for i in range(scales)]

        self.time_pos_emb = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        self.down_time = nn.ModuleList([nn.Linear(cond_embed_dim, width) for width in widths])
        self.up_time = nn.ModuleList([nn.Linear(cond_embed_dim, width) for width in reversed(widths[:-1])])

        self.temporal_embed = nn.Embedding(cycle_length, cond_embed_dim) if cycle_length > 0 else None
        if cycle_length > 0:
            max_periods = max(1, math.ceil(sequence_length / cycle_length) + 1)
            self.period_embed = nn.Embedding(max_periods, cond_embed_dim)
        else:
            self.period_embed = None

        self.film = FiLM(global_cond_dim, widths) if global_cond_dim > 0 else None

        self.stem = nn.Sequential(
            nn.Conv1d(self.total_in_channels, widths[0], kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, widths[0]),
        )

        self.temporal_proj = nn.Conv1d(cond_embed_dim, widths[0], kernel_size=1)

        self.fixed_diff = FixedDiff(channels=self.base_input_channels)

        self.down_j = nn.ModuleList()
        self.down_d = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.downsample = nn.ModuleList()

        prev_channels = widths[0]
        for idx, width in enumerate(widths):
            self.down_j.append(
                BranchStage(prev_channels, width, jitter_blocks, JitterBlock, dropout, norm_groups)
            )
            self.down_d.append(
                BranchStage(prev_channels, width, drift_blocks, DriftBlock, dropout, norm_groups)
            )
            self.gates.append(GateNet(width))
            if idx < len(widths) - 1:
                self.downsample.append(
                    nn.Sequential(
                        nn.Conv1d(width, widths[idx + 1], kernel_size=2, stride=2),
                        nn.GroupNorm(norm_groups, widths[idx + 1]),
                    )
                )
            prev_channels = widths[idx + 1] if idx < len(widths) - 1 else width

        bottleneck_layers: List[nn.Module] = [
            DriftBlock(widths[-1], dropout=dropout, norm_groups=norm_groups),
            JitterBlock(widths[-1], dropout=dropout, norm_groups=norm_groups),
        ]
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        self.up_stages = nn.ModuleList()
        for idx, width in enumerate(reversed(widths[:-1])):
            in_ch = widths[-(idx + 1)]
            self.up_stages.append(
                nn.ModuleDict(
                    {
                        "upsample": nn.ConvTranspose1d(in_ch, width, kernel_size=2, stride=2),
                        "time": self.up_time[idx],
                        "block": nn.Sequential(
                            DriftBlock(width * 2, dropout=dropout, norm_groups=norm_groups),
                            JitterBlock(width * 2, dropout=dropout, norm_groups=norm_groups),
                            nn.Conv1d(width * 2, width, kernel_size=1),
                        ),
                    }
                )
            )

        self.out_norm = nn.GroupNorm(norm_groups, widths[0])
        self.out_proj = nn.Conv1d(widths[0], input_channels, kernel_size=1)

        if stay_head:
            self.stay_head = nn.Sequential(
                nn.Conv1d(widths[0], widths[0] // 2, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(widths[0] // 2, 1, kernel_size=1),
            )
        else:
            self.stay_head = None

    def _build_temporal_features(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        if self.temporal_embed is None or self.period_embed is None:
            return torch.zeros(batch_size, 0, length, device=device)
        positions = torch.arange(length, device=device)
        cycle_index = positions % self.cycle_length
        period_index = positions // self.cycle_length
        max_period_idx = self.period_embed.num_embeddings - 1
        period_index = torch.clamp(period_index, max=max_period_idx)
        temporal = self.temporal_embed(cycle_index) + self.period_embed(period_index)
        return temporal.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1)

    def tempo_gate(
        self,
        base_xy: torch.Tensor,
        jitter_feat: torch.Tensor,
        drift_feat: torch.Tensor,
        gate: GateNet,
    ) -> torch.Tensor:
        d1, d2 = self.fixed_diff(base_xy)
        speed = torch.norm(d1, dim=1, keepdim=True)
        curvature = torch.norm(d2, dim=1, keepdim=True)
        if speed.shape[-1] != jitter_feat.shape[-1]:
            speed = F.interpolate(speed, size=jitter_feat.shape[-1], mode="linear", align_corners=False)
        if curvature.shape[-1] != jitter_feat.shape[-1]:
            curvature = F.interpolate(curvature, size=jitter_feat.shape[-1], mode="linear", align_corners=False)
        variance = (jitter_feat - jitter_feat.mean(dim=2, keepdim=True)).pow(2).mean(dim=1, keepdim=True)
        alpha = gate(speed, curvature, variance)
        return alpha * jitter_feat + (1.0 - alpha) * drift_feat

    def forward(
        self,
        noisy_xy: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        conditioning = conditioning or {}
        B, _, T = noisy_xy.shape
        orig_length = T
        total_down = len(self.downsample)
        pad = 0
        if total_down > 0:
            required = 2 ** total_down
            pad = (required - (T % required)) % required
        if pad > 0:
            noisy_xy = F.pad(noisy_xy, (0, pad))
            T = noisy_xy.shape[-1]
        device = noisy_xy.device

        extra = conditioning.get("extra_channels")
        if self.extra_channels > 0:
            if extra is None:
                extra = torch.zeros(B, self.extra_channels, T, device=device, dtype=noisy_xy.dtype)
            if extra.shape[1] != self.extra_channels:
                raise ValueError(f"Expected extra_channels={self.extra_channels}, got {extra.shape[1]}")
            if extra.shape[-1] != T:
                if extra.shape[-1] > T:
                    extra = extra[..., :T]
                else:
                    extra = F.pad(extra, (0, T - extra.shape[-1]))
            x_in = torch.cat([noisy_xy, extra], dim=1)
        else:
            x_in = noisy_xy

        temporal_feat = conditioning.get("temporal_features")
        if temporal_feat is None:
            temporal_feat = self._build_temporal_features(B, T, device)
        else:
            if temporal_feat.shape[-1] != T:
                if temporal_feat.shape[-1] > T:
                    temporal_feat = temporal_feat[..., :T]
                else:
                    temporal_feat = F.pad(temporal_feat, (0, T - temporal_feat.shape[-1]))

        h = self.stem(x_in)
        if temporal_feat.shape[1] > 0:
            if temporal_feat.shape[1] != self.cond_embed_dim:
                raise ValueError(
                    f"temporal_features channel mismatch: expected {self.cond_embed_dim}, got {temporal_feat.shape[1]}"
                )
            h = h + self.temporal_proj(temporal_feat)

        time_emb = self.time_mlp(self.time_pos_emb(timesteps))
        film_params = None
        if self.film is not None:
            cond_vec = conditioning.get("global_cond")
            if cond_vec is None:
                raise ValueError("global_cond is required when global_cond_dim > 0.")
            film_pairs = self.film(cond_vec)
            film_params = [(gamma, beta) for gamma, beta in film_pairs]
        else:
            film_params = [None] * len(self.down_j)

        skips: List[torch.Tensor] = []
        jitter = drift = h
        for idx, (down_j, down_d, gate, time_layer) in enumerate(
            zip(self.down_j, self.down_d, self.gates, self.down_time)
        ):
            jitter = down_j(jitter)
            drift = down_d(drift)
            time_term = time_layer(time_emb).unsqueeze(-1)
            jitter = jitter + time_term
            drift = drift + time_term
            if film_params[idx] is not None:
                gamma, beta = film_params[idx]
                jitter = jitter * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
                drift = drift * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            fused = self.tempo_gate(noisy_xy[:, : self.base_input_channels, :], jitter, drift, gate)
            skips.append(fused)
            if idx < len(self.downsample):
                jitter = self.downsample[idx](fused)
                drift = jitter
            else:
                jitter = fused

        x = self.bottleneck(jitter)

        for idx, stage in enumerate(self.up_stages):
            x = stage["upsample"](x)
            time_term = stage["time"](time_emb).unsqueeze(-1)
            x = x + time_term
            film_idx = len(self.up_stages) - idx - 1
            if film_idx < len(film_params) and film_params[film_idx] is not None:
                gamma, beta = film_params[film_idx]
                x = x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            skip = skips[-(idx + 2)]
            if skip.shape[-1] != x.shape[-1]:
                skip = F.interpolate(skip, size=x.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = stage["block"](x)

        skip0 = skips[0]
        if skip0.shape[-1] != x.shape[-1]:
            skip0 = F.interpolate(skip0, size=x.shape[-1], mode="linear", align_corners=False)
        out = self.out_norm(x + skip0)
        out = F.silu(out)
        eps = self.out_proj(out)
        if pad > 0:
            eps = eps[..., :orig_length]
        stay_logits = self.stay_head(out) if self.stay_head is not None else None
        if stay_logits is not None and pad > 0:
            stay_logits = stay_logits[..., :orig_length]
        return eps, stay_logits


# Backwards compatibility alias
TG2SDARULite = UnifiedTG2SDARU
