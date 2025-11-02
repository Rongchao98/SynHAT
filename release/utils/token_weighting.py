"""
Token weighting utilities for diffusion training with event-focused emphasis.

Implements:
1. Cosine warm-up of event emphasis
2. Soft neighbor weights via 1D Gaussian blur
3. Per-sequence normalization (mean weight = 1)
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Dict


def build_token_weights(
    m: torch.Tensor,
    epoch: int,
    total_epochs: int,
    stage: int,
    alpha0: float = 0.15,
    alpha1_max: float = 6.0,
    alpha_near: float = 0.6,
    r_max_stage: Optional[Dict[int, int]] = None,
    ratio_cap: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
) -> torch.Tensor:
    """
    Build per-token weights for event-focused diffusion training.
    
    Args:
        m: Event mask tensor (B, L) with 1 at events, 0 elsewhere
        epoch: Current training epoch (0-indexed)
        total_epochs: Total number of training epochs
        stage: Training stage (1 or 2) for radius selection
        alpha0: Base weight for all tokens (default: 0.15)
        alpha1_max: Maximum additional weight for event tokens (default: 6.0)
        alpha_near: Weight for neighbor tokens (default: 0.6)
        r_max_stage: Maximum radius by stage {1: 1, 2: 2}
        ratio_cap: Optional clamp range [1/cap, cap] for stability
        warmup_epochs: Number of epochs for warm-up (default: 15% of total_epochs)
    
    Returns:
        w: Token weights (B, L) with mean=1 per sequence
    
    Properties:
        - Cosine warm-up over first warmup_epochs (default 15% of training)
        - Soft neighbor weighting via Gaussian blur
        - Per-sequence normalized (mean=1)
        - Stable at initialization (epoch=0)
    """
    if r_max_stage is None:
        r_max_stage = {1: 1, 2: 2}
    
    B, L = m.shape
    device = m.device
    dtype = m.dtype
    
    # 1. Compute warm-up factor (cosine schedule over first warmup_epochs)
    if warmup_epochs is None:
        warmup_epochs = max(1, int(0.15 * total_epochs))  # Default: 15% of training
    
    if epoch < warmup_epochs:
        # Cosine warm-up: smooth ramp from 0 to 1
        warmup_factor = 0.5 * (1.0 - math.cos(math.pi * epoch / warmup_epochs))
    else:
        warmup_factor = 1.0
    
    # Apply warm-up to event weight and radius
    alpha1_now = alpha1_max * warmup_factor
    r_max = r_max_stage.get(stage, 1)
    r_now = int(round(r_max * warmup_factor))
    
    # 2. Compute soft neighbor weights via Gaussian blur
    if r_now > 0:
        # Create 1D Gaussian kernel
        kernel_size = 2 * r_now + 1
        sigma = r_now / 2.0
        
        # Generate Gaussian kernel
        x = torch.arange(kernel_size, dtype=dtype, device=device) - r_now
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()  # Normalize
        
        # Reshape for conv1d: (out_channels=1, in_channels=1, kernel_size)
        kernel = kernel.view(1, 1, kernel_size)
        
        # Apply conv1d with padding to maintain length
        # Input: (B, 1, L), Output: (B, 1, L)
        m_expanded = m.unsqueeze(1).float()  # (B, 1, L)
        near = F.conv1d(m_expanded, kernel, padding=r_now).squeeze(1)  # (B, L)
        near = near.clamp(0.0, 1.0)  # Ensure [0, 1] range
    else:
        # No blurring at step 0
        near = torch.zeros_like(m, dtype=dtype)
    
    # 3. Compute raw weights
    # w_raw = alpha0 + alpha1_now * m + alpha_near * near
    w_raw = alpha0 + alpha1_now * m.float() + alpha_near * near
    
    # 4. Per-sequence normalization (mean = 1)
    w_mean = w_raw.mean(dim=1, keepdim=True).clamp_min(1e-8)
    w = w_raw / w_mean
    
    # 5. Optional ratio capping for stability
    if ratio_cap is not None:
        w = w.clamp(1.0 / ratio_cap, ratio_cap)
    
    return w.to(dtype)


def get_default_weighting_config(stage: int) -> Dict:
    """
    Get default token weighting configuration for a stage.
    
    Args:
        stage: Training stage (1 or 2)
    
    Returns:
        Dictionary with default parameters
    """
    return {
        "alpha0": 0.15,          # Base weight for all tokens
        "alpha1_max": 6.0,       # Max event weight (after warm-up)
        "alpha_near": 0.6,       # Neighbor weight coefficient
        "r_max_stage": {1: 1, 2: 2},  # Max radius by stage
        "ratio_cap": None,       # Optional stability cap (set to 8.0 if needed)
    }


def validate_token_weights(w: torch.Tensor, m: torch.Tensor, tol: float = 0.1) -> bool:
    """
    Validate token weights satisfy expected properties.
    
    Args:
        w: Token weights (B, L)
        m: Event mask (B, L)
        tol: Tolerance for mean=1 check
    
    Returns:
        True if valid, raises AssertionError otherwise
    """
    # Check shapes match
    assert w.shape == m.shape, f"Shape mismatch: w={w.shape}, m={m.shape}"
    
    # Check mean ≈ 1 per sequence
    w_mean = w.mean(dim=1)
    assert torch.all(torch.abs(w_mean - 1.0) < tol), \
        f"Weight mean not close to 1.0: {w_mean.tolist()}"
    
    # Check no NaN or Inf
    assert not torch.isnan(w).any(), "Weights contain NaN"
    assert not torch.isinf(w).any(), "Weights contain Inf"
    
    # Check positive
    assert torch.all(w > 0), f"Weights contain non-positive values: min={w.min()}"
    
    return True
