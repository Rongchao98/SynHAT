"""
Loss functions for Stage-2 training using token weighting system.

Configure via config.stage2.training:
  use_token_weighting: true
  token_weighting:
    alpha0: 0.15         # Base weight for all tokens
    alpha1_max: 6.0      # Maximum event weight (after warm-up)
    alpha_near: 0.6      # Neighbor weight coefficient
    r_max_stage: {1: 1, 2: 2}  # Gaussian blur radius by stage
    ratio_cap: null      # Optional weight capping
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from types import SimpleNamespace

# Import token weighting system
from utils.token_weighting import build_token_weights, get_default_weighting_config


def compute_weighted_mse_loss(
    eps_hat: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    total_epochs: int,
    token_weighting_cfg: Optional[dict] = None,
    use_token_weighting: bool = True,
) -> torch.Tensor:
    """
    Weighted MSE loss using token weighting system.
    
    Args:
        eps_hat: Predicted noise (B, C, L)
        noise: True noise (B, C, L)
        mask: Loss mask (B, L)
        target: Original target (B, C, L) to extract event positions from indicator channel
        epoch: Current training epoch (for warm-up)
        total_epochs: Total training epochs (for warm-up)
        token_weighting_cfg: Token weighting configuration dict
        use_token_weighting: Whether to apply token weighting (default: True)
    
    Returns:
        Scalar loss
    """
    # Get configuration
    if token_weighting_cfg is None:
        token_weighting_cfg = get_default_weighting_config()
    
    # Extract event mask from target indicator channel (B, L)
    event_mask = (target[:, 2, :] > 0.5).float()  # (B, L)
    
    # Build token weights with warm-up, soft neighbors, normalization
    if use_token_weighting:
        token_weights = build_token_weights(
            m=event_mask,
            epoch=epoch,
            total_epochs=total_epochs,
            stage=2,  # Stage-2
            alpha0=token_weighting_cfg.get('alpha0', 0.15),
            alpha1_max=token_weighting_cfg.get('alpha1_max', 6.0),
            alpha_near=token_weighting_cfg.get('alpha_near', 0.6),
            r_max_stage=token_weighting_cfg.get('r_max_stage', {1: 1, 2: 2}),
            ratio_cap=token_weighting_cfg.get('ratio_cap', None),
            warmup_epochs=token_weighting_cfg.get('warmup_epochs', None),
        )  # (B, L)
    else:
        # Uniform weighting (all tokens have equal weight)
        token_weights = torch.ones_like(event_mask)  # (B, L)
    
    # Compute per-token MSE for each channel
    mse = F.mse_loss(eps_hat, noise, reduction="none")  # (B, C, L)
    
    # MSE per token (sum across channels)
    mse_per_token = mse.sum(dim=1)  # (B, L)
    
    # Apply token weights and sequence mask
    weighted_mse = token_weights * mse_per_token * mask
    
    # Normalize by sum of weights (per-sequence normalization ensures mean=1)
    loss = weighted_mse.sum() / ((token_weights * mask).sum() + 1e-8)
    
    return loss


def compute_stage2_loss(
    eps_hat: torch.Tensor,
    noise: torch.Tensor,
    target: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    mask: torch.Tensor,
    schedule,
    config,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 2000,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute weighted MSE loss for Stage-2 training using token weighting.
    
    Args:
        eps_hat: Predicted noise (B, C, L)
        noise: True noise (B, C, L)
        target: Original target (B, C, L) - used to extract event positions
        x_t: Noisy input (B, C, L) - not used but kept for compatibility
        timesteps: Diffusion timesteps (B,) - not used but kept for compatibility
        mask: Loss mask (B, L)
        schedule: DiffusionSchedule - not used but kept for compatibility
        config: Training config with loss parameters
        device: torch device
        epoch: Current training epoch (for warm-up)
        total_epochs: Total training epochs (for warm-up)
    
    Returns:
        total_loss: Scalar loss for backprop
        loss_dict: Dictionary with loss value for logging
    """
    # Load token weighting configuration
    use_token_weighting = bool(getattr(config.stage2.training, "use_token_weighting", True))
    tw_cfg = getattr(config.stage2.training, "token_weighting", None)
    token_weighting_cfg = None
    
    if tw_cfg is not None:
        if isinstance(tw_cfg, SimpleNamespace):
            token_weighting_cfg = vars(tw_cfg)
        elif isinstance(tw_cfg, dict):
            token_weighting_cfg = tw_cfg
        else:
            token_weighting_cfg = {}
        
        # Handle r_max_stage if it's a SimpleNamespace
        if 'r_max_stage' in token_weighting_cfg:
            r_max = token_weighting_cfg['r_max_stage']
            if isinstance(r_max, SimpleNamespace):
                # Convert SimpleNamespace to dict with integer keys
                token_weighting_cfg['r_max_stage'] = {int(k): int(v) for k, v in vars(r_max).items()}
    
    # Compute loss with token weighting
    total_loss = compute_weighted_mse_loss(
        eps_hat,
        noise,
        mask,  # (B, L)
        target,
        epoch=epoch,
        total_epochs=total_epochs,
        token_weighting_cfg=token_weighting_cfg,
        use_token_weighting=use_token_weighting,
    )
    loss_dict = {"total": total_loss.item()}
    
    return total_loss, loss_dict
