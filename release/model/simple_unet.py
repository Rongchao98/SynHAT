"""
Simple U-Net denoiser for debugging. This is a minimal, standard U-Net architecture
to test if the complex TG-2S-DARU design is causing training issues.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time embedding (standard for diffusion models)."""
    
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


class ResBlock(nn.Module):
    """Simple residual block with time embedding."""
    
    def __init__(self, channels: int, time_embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_embed_dim, channels)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_term = self.time_mlp(time_emb).unsqueeze(-1)
        h = h + time_term
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h


class DownBlock(nn.Module):
    """Downsampling block with residual connections."""
    
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.res1 = ResBlock(in_channels, time_embed_dim, dropout)
        self.res2 = ResBlock(in_channels, time_embed_dim, dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res1(x, time_emb)
        h = self.res2(h, time_emb)
        skip = h
        h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res1 = ResBlock(out_channels * 2, time_embed_dim, dropout)  # *2 for concat with skip
        self.res2 = ResBlock(out_channels * 2, time_embed_dim, dropout)
        self.conv_out = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)
        
        # Handle size mismatch
        if h.shape[-1] != skip.shape[-1]:
            h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
        
        h = torch.cat([h, skip], dim=1)
        h = self.res1(h, time_emb)
        h = self.res2(h, time_emb)
        h = self.conv_out(h)
        return h


class SimpleUNet(nn.Module):
    """
    Simple U-Net denoiser for trajectory diffusion.
    
    This is a minimal, standard architecture to test if the complex TG-2S-DARU
    design is causing issues. Uses standard U-Net components:
    - Time embedding via sinusoidal positional encoding
    - Residual blocks with time conditioning
    - Symmetric encoder-decoder with skip connections
    - Optional stay probability head
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        sequence_length: int = 168,
        cycle_length: int = 168,
        base_channels: int = 128,
        scales: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        stay_head: bool = False,
        extra_channels: int = 0,
        **kwargs  # Ignore other TG2SDARU-specific arguments
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.extra_channels = extra_channels
        self.total_in_channels = input_channels + extra_channels
        self.stay_head_enabled = stay_head
        self.scales = scales
        
        # Time embedding
        self.time_pos_emb = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        # Channel widths for each scale
        widths = [base_channels * (2 ** i) for i in range(scales)]
        
        # Initial convolution
        self.init_conv = nn.Conv1d(self.total_in_channels, widths[0], kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        for i in range(scales - 1):
            self.down_blocks.append(
                DownBlock(widths[i], widths[i + 1], time_embed_dim, dropout)
            )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(widths[-1], time_embed_dim, dropout),
            ResBlock(widths[-1], time_embed_dim, dropout),
        )
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(scales - 1):
            in_ch = widths[-(i + 1)]
            out_ch = widths[-(i + 2)]
            self.up_blocks.append(
                UpBlock(in_ch, out_ch, time_embed_dim, dropout)
            )
        
        # Output projection
        self.out_norm = nn.GroupNorm(8, widths[0])
        self.out_conv = nn.Conv1d(widths[0], input_channels, kernel_size=1)
        
        # Optional stay head
        if stay_head:
            self.stay_head = nn.Sequential(
                nn.Conv1d(widths[0], widths[0] // 2, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(widths[0] // 2, 1, kernel_size=1),
            )
        else:
            self.stay_head = None
    
    def forward(
        self,
        noisy_xy: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            noisy_xy: (B, C, T) noisy trajectory
            timesteps: (B,) diffusion timesteps
            conditioning: Optional dict with 'extra_channels' key
            
        Returns:
            eps: (B, C, T) predicted noise
            stay_logits: (B, 1, T) stay probabilities if stay_head enabled
        """
        conditioning = conditioning or {}
        B, _, T = noisy_xy.shape
        orig_length = T
        device = noisy_xy.device
        
        # Pad sequence length to be divisible by 2^scales for downsampling
        pad = 0
        if self.scales > 1:
            required = 2 ** (self.scales - 1)
            pad = (required - (T % required)) % required
        if pad > 0:
            noisy_xy = F.pad(noisy_xy, (0, pad))
            T = noisy_xy.shape[-1]
        
        # Handle extra channels (if any)
        extra = conditioning.get("extra_channels")
        if self.extra_channels > 0:
            if extra is None:
                extra = torch.zeros(B, self.extra_channels, T, device=device, dtype=noisy_xy.dtype)
            if extra.shape[-1] != T:
                if extra.shape[-1] > T:
                    extra = extra[..., :T]
                else:
                    extra = F.pad(extra, (0, T - extra.shape[-1]))
            x = torch.cat([noisy_xy, extra], dim=1)
        else:
            x = noisy_xy
        
        # Time embedding
        time_emb = self.time_mlp(self.time_pos_emb(timesteps))
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Encoder with skip connections
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skips.append(skip)
        
        # Bottleneck
        h = self.bottleneck[0](h, time_emb)
        h = self.bottleneck[1](h, time_emb)
        
        # Decoder with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            h = up_block(h, skip, time_emb)
        
        # Output
        out = self.out_norm(h)
        out = F.silu(out)
        eps = self.out_conv(out)
        
        # Remove padding
        if pad > 0:
            eps = eps[..., :orig_length]
        
        # Optional stay head
        stay_logits = None
        if self.stay_head is not None:
            stay_logits = self.stay_head(out)
            if pad > 0:
                stay_logits = stay_logits[..., :orig_length]
        
        return eps, stay_logits


# Create an alias for compatibility
UnifiedSimpleUNet = SimpleUNet
