#!/usr/bin/env python3
"""Stage-2 training for fine-grained block diffusion."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn.functional as F
    from torch.cuda import amp
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for train_s2.py. Install torch in the active environment."
    ) from exc

from model.EMA import EMAHelper
from model.diffusion_schedule import DiffusionSchedule, make_beta_schedule
from model.tg2s_daru import UnifiedTG2SDARU
from utils.general_utils import load_config_from_yaml
from utils.logger import Logger
from utils.stage2_dataset import Stage2BlockDataset
from utils.stage2_losses import compute_stage2_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 block diffuser")
    parser.add_argument("--config", default="config/sample_synhat.yaml", type=str, help="YAML config")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--output-dir", type=str, help="Override Stage-2 output dir")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers")
    parser.add_argument("--resume", type=str, help="Checkpoint path to resume from")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_device(config, override: Optional[str]) -> torch.device:
    device_str = override or getattr(getattr(config, "project", None), "device", "cuda")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def load_metadata(config, data_dir: Path) -> Dict:
    meta_path = data_dir / config.data.stage_files.metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_dataloader(npz_path: Path, batch_size: int, num_workers: int, device: torch.device, shuffle: bool) -> DataLoader:
    dataset = Stage2BlockDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )


def build_model(config, metadata: Dict, sample: Dict[str, torch.Tensor], device: torch.device) -> UnifiedTG2SDARU:
    model_cfg = config.stage2.model
    sequence_length = sample["target"].shape[-1]
    input_channels = sample["target"].shape[1]
    extra_channels = sample["base_path"].shape[1]
    global_cond_dim = sample["global_cond"].shape[-1]

    model = UnifiedTG2SDARU(
        input_channels=input_channels,
        sequence_length=sequence_length,
        cycle_length=sequence_length,
        base_channels=int(model_cfg.base_channels),
        scales=int(model_cfg.scales),
        jitter_blocks=int(model_cfg.jitter_blocks),
        drift_blocks=int(model_cfg.drift_blocks),
        dropout=float(model_cfg.dropout),
        time_embed_dim=int(model_cfg.time_embed_dim),
        cond_embed_dim=int(model_cfg.embedding_dim),
        stay_head=False,
        extra_channels=extra_channels,
        global_cond_dim=global_cond_dim,
        norm_groups=int(model_cfg.norm_groups),
    )
    return model.to(device)


def setup_diffusion(config, device: torch.device) -> DiffusionSchedule:
    diff_cfg = config.stage2.diffusion
    betas = make_beta_schedule(
        schedule=diff_cfg.beta_schedule,
        timesteps=int(diff_cfg.timesteps),
        beta_start=float(diff_cfg.beta_start),
        beta_end=float(diff_cfg.beta_end),
    )
    return DiffusionSchedule(betas).to(device)


def train_one_epoch(
    model: UnifiedTG2SDARU,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[amp.GradScaler],
    ema: Optional[EMAHelper],
    device: torch.device,
    epoch: int,
    config,
    logger: Logger,
    global_step: int = 0,
    total_steps: int = 100000,
) -> Tuple[float, int]:
    """
    Train for one epoch.
    
    Args:
        ...existing args...
        global_step: Current global training step (for token weighting warm-up)
        total_steps: Total training steps across all epochs (for token weighting warm-up)
    
    Returns:
        avg_loss: Average loss for the epoch
        updated_global_step: Updated global step counter
    """
    model.train()
    total_loss = 0.0
    batches = 0

    grad_clip = float(getattr(config.stage2.training, "grad_clip", 0.0))
    amp_enabled = bool(config.stage2.training.amp) and device.type == "cuda"

    for step, batch in enumerate(loader, start=1):
        model.train()
        target = batch["target"].to(device)  # Absolute coordinates (not residuals)
        base_path = batch["base_path"].to(device)
        global_cond = batch["global_cond"].to(device)
        mask = batch["mask"].to(device).unsqueeze(1)  # (B, 1, L)
        
        # Apply coordinate augmentation to base_path during training
        # This helps the model generalize to unseen coarse paths at inference
        coord_aug_std = float(getattr(config.stage2.training, "coord_augmentation_std", 0.0))
        if coord_aug_std > 0:
            # Add Gaussian noise to coordinate channels (not indicator channel)
            noise = torch.randn_like(base_path[..., :2]) * coord_aug_std
            base_path = base_path.clone()
            base_path[..., :2] += noise
        
        timesteps = schedule.sample_timesteps(target.shape[0], device)
        x_t, noise = schedule.q_sample(target, timesteps)

        optimizer.zero_grad(set_to_none=True)
        conditioning = {"extra_channels": base_path, "global_cond": global_cond}

        with amp.autocast(enabled=amp_enabled):
            eps_hat, _ = model(x_t, timesteps, conditioning=conditioning)
            
            # Flexible loss computation with token weighting or legacy method
            # See utils/stage2_losses.py for details
            loss, loss_dict = compute_stage2_loss(
                eps_hat, noise, target, x_t, timesteps, mask, schedule, config, device,
                epoch=epoch, total_epochs=int(config.stage2.training.num_epochs)
            )

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += float(loss.detach().cpu())
        batches += 1
        global_step += 1  # Increment global step counter

        if step % int(config.stage2.training.log_interval) == 0:
            avg_loss = total_loss / batches
            logger.info(f"Epoch {epoch} Step {step}/{len(loader)} | Loss: {avg_loss:.4f} | Global Step: {global_step}")

    return total_loss / max(batches, 1), global_step


@torch.no_grad()
def evaluate(
    model: UnifiedTG2SDARU,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    config,
    device: torch.device,
    epoch: int = 0,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0

    for batch in loader:
        target = batch["target"].to(device)
        base_path = batch["base_path"].to(device)
        global_cond = batch["global_cond"].to(device)
        mask = batch["mask"].to(device).unsqueeze(1)
        timesteps = schedule.sample_timesteps(target.shape[0], device)
        x_t, noise = schedule.q_sample(target, timesteps)
        eps_hat, _ = model(
            x_t,
            timesteps,
            conditioning={"extra_channels": base_path, "global_cond": global_cond},
        )
        loss, _ = compute_stage2_loss(
            eps_hat, noise, target, x_t, timesteps, mask, schedule, config, device,
            epoch=epoch, total_epochs=int(config.stage2.training.num_epochs)
        )
        total_loss += float(loss.detach().cpu())
        batches += 1

    return total_loss / max(batches, 1)


def save_checkpoint(
    path: Path,
    model: UnifiedTG2SDARU,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    schedule: DiffusionSchedule,
    scaler: Optional[amp.GradScaler],
    ema: Optional[EMAHelper],
    global_step: int = 0,
) -> None:
    """
    Save training checkpoint with global step for token weighting warm-up.
    
    Args:
        ...existing args...
        global_step: Current global training step (for token weighting warm-up)
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "betas": schedule.betas.cpu(),
        "global_step": global_step,  # Save global step for token weighting
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    config = load_config_from_yaml(args.config)

    data_dir = Path(config.data.processed_dir)
    metadata = load_metadata(config, data_dir)

    # Build project-based output directory: outputs/{project_name}_{date}/stage2
    output_root = Path(args.output_dir) if args.output_dir else Path(config.stage2.training.output_dir)
    project_name = config.project.name
    date_str = dt.datetime.now().strftime("%Y%m%d")
    run_dir = output_root / f"{project_name}_{date_str}" / "stage2"
    ckpt_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    device = prepare_device(config, args.device)
    set_seed(int(getattr(getattr(config, "project", None), "seed", 42)))
    torch.backends.cudnn.benchmark = device.type == "cuda"

    logger = Logger(
        name="stage2_train",
        colorize=True,
        log_path=run_dir / "train.log",
    )
    logger.info(f"Using device: {device}")

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f_out:
        with open(args.config, "r", encoding="utf-8") as f_in:
            f_out.write(f_in.read())

    num_workers = int(args.num_workers if args.num_workers is not None else config.data.num_workers)
    train_loader = create_dataloader(
        data_dir / config.data.stage_files.stage2.train,
        batch_size=int(config.stage2.training.batch_size),
        num_workers=num_workers,
        device=device,
        shuffle=True,
    )
    val_loader = None
    val_path = data_dir / config.data.stage_files.stage2.val
    if val_path.exists() and int(metadata["split_sizes"]["stage2"]["val"]) > 0:
        val_loader = create_dataloader(
            val_path,
            batch_size=int(config.stage2.training.batch_size),
            num_workers=num_workers,
            device=device,
            shuffle=False,
        )

    sample_batch = next(iter(train_loader))
    model = build_model(config, metadata, sample_batch, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.stage2.training.learning_rate),
        weight_decay=float(getattr(config.stage2.training, "weight_decay", 0.0)),
    )
    schedule = setup_diffusion(config, device)

    amp_enabled = bool(config.stage2.training.amp) and device.type == "cuda"
    scaler = amp.GradScaler(enabled=amp_enabled)

    ema = None
    if getattr(config.stage2.training, "ema", False):
        ema = EMAHelper(mu=float(config.stage2.training.ema_decay))
        ema.register(model)

    start_epoch = 1
    best_val = float("inf")
    global_step = 0  # Track global step for token weighting warm-up

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            logger.info(f"Resuming from checkpoint {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            schedule = DiffusionSchedule(state["betas"]).to(device)
            if scaler is not None and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            if ema is not None and "ema" in state:
                ema.load_state_dict(state["ema"])
            start_epoch = state.get("epoch", 0) + 1
            best_val = state.get("best_val", float("inf"))
            global_step = state.get("global_step", 0)  # Restore global step
        else:
            logger.warning(f"Checkpoint {ckpt_path} not found; starting from scratch.")

    # Calculate total steps for token weighting warm-up
    # total_steps = num_epochs * batches_per_epoch
    batches_per_epoch = len(train_loader)
    num_epochs = int(config.stage2.training.num_epochs)
    total_steps = num_epochs * batches_per_epoch
    logger.info(f"Training for {num_epochs} epochs, {batches_per_epoch} batches/epoch = {total_steps} total steps")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            schedule,
            optimizer,
            scaler if amp_enabled else None,
            ema,
            device,
            epoch,
            config,
            logger,
            global_step=global_step,
            total_steps=total_steps,
        )
        logger.info(f"[Train] Epoch {epoch} | Loss {train_loss:.4f} | Global Step {global_step}")

        if val_loader and epoch % int(config.stage2.training.val_interval) == 0:
            val_loss = evaluate(model, val_loader, schedule, config, device, epoch)
            logger.info(f"[Val]   Epoch {epoch} | Loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(best_path, model, optimizer, epoch, best_val, schedule, scaler, ema, global_step)
                logger.info(f"Saved new best checkpoint to {best_path}")

        if epoch % int(config.stage2.training.save_interval) == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val, schedule, scaler, ema, global_step)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    final_path = ckpt_dir / "last.pt"
    save_checkpoint(final_path, model, optimizer, epoch, best_val, schedule, scaler, ema, global_step)
    logger.info(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
