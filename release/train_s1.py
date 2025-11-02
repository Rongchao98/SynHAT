#!/usr/bin/env python3
"""Stage-1 training for the unified TG-2S-DARU diffuser."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Dict, Optional
from types import SimpleNamespace

import numpy as np
try:
    import torch
    import torch.nn.functional as F
    from torch.cuda import amp
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise ModuleNotFoundError(
        "PyTorch is required for train_s1.py. Install torch in the active environment."
    ) from exc

from model.EMA import EMAHelper
from model.diffusion_schedule import DiffusionSchedule, make_beta_schedule
from model.tg2s_daru import UnifiedTG2SDARU
from model.simple_unet import SimpleUNet
from utils.general_utils import load_config_from_yaml
from utils.logger import Logger
from utils.stage1_dataset import Stage1SMTDataset
from utils.token_weighting import build_token_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 coarse SMT diffuser")
    parser.add_argument("--config", default="config/sample_synhat.yaml", type=str, help="YAML config file")
    parser.add_argument("--device", type=str, help="Override device (e.g., cuda:0 or cpu)")
    parser.add_argument("--output-dir", type=str, help="Override Stage-1 output directory")
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
    if isinstance(config.data.stage_files, dict):
        meta_file = config.data.stage_files.get("metadata", "stage_metadata.json")
    else:
        meta_file = getattr(config.data.stage_files, "metadata", "stage_metadata.json")
    meta_path = data_dir / meta_file
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_dataloaders(config, metadata: Dict, device: torch.device, override_workers: Optional[int]) -> DataLoader:
    data_dir = Path(config.data.processed_dir)
    stage1_files = config.data.stage_files.stage1
    stay_bce_weight = float(getattr(config.stage1.training, "stay_bce_weight", 0.0))

    dataset = Stage1SMTDataset(
        data_dir / stage1_files.train,
        use_stay=stay_bce_weight > 0.0,
    )
    num_workers = int(override_workers if override_workers is not None else config.data.num_workers)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=int(config.stage1.training.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader


def build_model(config, metadata: Dict, device: torch.device) -> UnifiedTG2SDARU:
    model_cfg = config.stage1.model
    data_cfg = config.data
    
    # Check if we should use simple U-Net architecture
    architecture = getattr(model_cfg, "architecture", "tg2sdaru")
    
    if architecture == "simple_unet":
        model = SimpleUNet(
            input_channels=3,
            sequence_length=int(metadata["num_slots"]),
            cycle_length=int(metadata["slots_per_day"]),
            base_channels=int(model_cfg.base_channels),
            scales=int(model_cfg.scales),
            dropout=float(model_cfg.dropout),
            time_embed_dim=int(model_cfg.time_embed_dim),
            stay_head=False,
            extra_channels=int(getattr(model_cfg, "extra_channels", 0)),
        )
    else:
        model = UnifiedTG2SDARU(
            input_channels=3,
            sequence_length=int(metadata["num_slots"]),
            cycle_length=int(metadata["slots_per_day"]),
            base_channels=int(model_cfg.base_channels),
            scales=int(model_cfg.scales),
            jitter_blocks=int(model_cfg.jitter_blocks),
            drift_blocks=int(model_cfg.drift_blocks),
            dropout=float(model_cfg.dropout),
            time_embed_dim=int(model_cfg.time_embed_dim),
            cond_embed_dim=int(model_cfg.embedding_dim),
            stay_head=False,
            extra_channels=int(getattr(model_cfg, "extra_channels", 0)),
            global_cond_dim=int(getattr(model_cfg, "global_cond_dim", 0)),
            norm_groups=int(model_cfg.norm_groups),
        )
    return model.to(device)


def setup_diffusion(config, device: torch.device) -> DiffusionSchedule:
    diff_cfg = config.stage1.diffusion
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
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    channel_accum = torch.zeros(3, device=device)
    bce_accum = 0.0
    batches = 0

    grad_clip = float(getattr(config.stage1.training, "grad_clip", 0.0))
    amp_enabled = bool(config.stage1.training.amp) and device.type == "cuda"
    stay_bce_weight = float(getattr(config.stage1.training, "stay_bce_weight", 0.0))
    use_bce = stay_bce_weight > 0.0
    
    # Token weighting configuration
    use_token_weighting = bool(getattr(config.stage1.training, "use_token_weighting", True))
    tw_config = getattr(config.stage1.training, "token_weighting", {})
    alpha0 = float(getattr(tw_config, "alpha0", 0.15))
    alpha1_max = float(getattr(tw_config, "alpha1_max", 6.0))
    alpha_near = float(getattr(tw_config, "alpha_near", 0.6))
    r_max_stage = getattr(tw_config, "r_max_stage", {1: 1, 2: 2})
    # Convert SimpleNamespace or dict to dict with integer keys
    if hasattr(r_max_stage, '__dict__'):
        # SimpleNamespace from YAML
        r_max_stage = {int(k): int(v) for k, v in vars(r_max_stage).items()}
    elif isinstance(r_max_stage, dict):
        r_max_stage = {int(k): int(v) for k, v in r_max_stage.items()}
    ratio_cap = getattr(tw_config, "ratio_cap", None)
    if ratio_cap is not None:
        ratio_cap = float(ratio_cap)
    warmup_epochs = getattr(tw_config, "warmup_epochs", None)
    if warmup_epochs is not None:
        warmup_epochs = int(warmup_epochs)
    total_epochs = int(config.stage1.training.num_epochs)

    for step, batch in enumerate(loader, start=1):
        clean = batch["sequence"].to(device)  # (B, 3, L)
        timesteps = schedule.sample_timesteps(clean.shape[0], device)
        x_t, noise = schedule.q_sample(clean, timesteps)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=amp_enabled):
            eps_hat, _ = model(x_t, timesteps)
            diff = eps_hat - noise  # (B, 3, L)
            
            # Extract event mask from indicator channel (B, L)
            event_mask = (clean[:, 2, :] > 0.5).float()  # (B, L)
            
            # Build token weights with warm-up, soft neighbor, and normalization
            if use_token_weighting:
                token_weights = build_token_weights(
                    m=event_mask,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    stage=1,
                    alpha0=alpha0,
                    alpha1_max=alpha1_max,
                    alpha_near=alpha_near,
                    r_max_stage=r_max_stage,
                    ratio_cap=ratio_cap,
                    warmup_epochs=warmup_epochs,
                )  # (B, L)
            else:
                # Uniform weighting (all tokens have equal weight)
                token_weights = torch.ones_like(event_mask)  # (B, L)
            
            # Compute per-token MSE for spatial channels (lat, lon)
            mse_spatial = (diff[:, :2, :] ** 2).sum(dim=1)  # (B, L)
            
            # Compute per-token MSE for indicator channel
            mse_indicator = diff[:, 2, :] ** 2  # (B, L)
            
            # Apply token weights
            loss = (token_weights * (mse_spatial + mse_indicator)).sum() / (token_weights.sum() + 1e-8)
            
            channel_mse = (diff ** 2).mean(dim=(0, 2))
            if use_bce:
                stay_target = batch["stay"].to(device)
                x0_pred = schedule.predict_start_from_noise(x_t, timesteps, eps_hat)
                stay_pred = x0_pred[:, 2, :].clamp(0.0, 1.0)
                bce_loss = F.binary_cross_entropy(stay_pred, stay_target, reduction="mean")
                loss = loss + stay_bce_weight * bce_loss
            else:
                bce_loss = torch.tensor(0.0, device=device)

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
        channel_accum += channel_mse.detach()
        if use_bce:
            bce_accum += float(bce_loss.detach().cpu())
        batches += 1
        global_step += 1

        if step % int(config.stage1.training.log_interval) == 0:
            msg = (
                f"Epoch {epoch} Step {step}/{len(loader)} | "
                f"Loss: {total_loss / batches:.4f} | "
                f"Channel MSE: X {channel_accum[0] / batches:.4f} | "
                f"Y {channel_accum[1] / batches:.4f} | Stay {channel_accum[2] / batches:.4f}"
            )
            if use_bce:
                msg += f" | Stay BCE {bce_accum / batches:.4f}"
            logger.info(msg)

    metrics = {
        "loss": total_loss / max(batches, 1),
        "channel_mse": (channel_accum / max(batches, 1)).detach().cpu().tolist(),
        "global_step": global_step,
    }
    if use_bce:
        metrics["stay_bce"] = bce_accum / max(batches, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: UnifiedTG2SDARU,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    config,
    epoch: int = 0,
    global_step: int = 0,
) -> Dict[str, float]:
    """Evaluate model using token weighting system."""
    model.eval()
    total_loss = 0.0
    channel_accum = torch.zeros(3, device=device)
    bce_accum = 0.0
    batches = 0
    stay_bce_weight = float(getattr(config.stage1.training, "stay_bce_weight", 0.0))
    use_bce = stay_bce_weight > 0.0
    
    # Token weighting configuration (same as training)
    use_token_weighting = bool(getattr(config.stage1.training, "use_token_weighting", True))
    tw_config = getattr(config.stage1.training, "token_weighting", {})
    alpha0 = float(getattr(tw_config, "alpha0", 0.15))
    alpha1_max = float(getattr(tw_config, "alpha1_max", 6.0))
    alpha_near = float(getattr(tw_config, "alpha_near", 0.6))
    r_max_stage = getattr(tw_config, "r_max_stage", {1: 1, 2: 2})
    # Convert SimpleNamespace or dict to dict with integer keys
    if hasattr(r_max_stage, '__dict__'):
        # SimpleNamespace from YAML
        r_max_stage = {int(k): int(v) for k, v in vars(r_max_stage).items()}
    elif isinstance(r_max_stage, dict):
        r_max_stage = {int(k): int(v) for k, v in r_max_stage.items()}
    ratio_cap = getattr(tw_config, "ratio_cap", None)
    if ratio_cap is not None:
        ratio_cap = float(ratio_cap)
    warmup_epochs = getattr(tw_config, "warmup_epochs", None)
    if warmup_epochs is not None:
        warmup_epochs = int(warmup_epochs)
    total_epochs = int(config.stage1.training.num_epochs)

    for batch in loader:
        clean = batch["sequence"].to(device)
        timesteps = schedule.sample_timesteps(clean.shape[0], device)
        x_t, noise = schedule.q_sample(clean, timesteps)

        eps_hat, _ = model(x_t, timesteps)
        diff = eps_hat - noise  # (B, 3, L)
        
        # Extract event mask from indicator channel (B, L)
        event_mask = (clean[:, 2, :] > 0.5).float()
        
        # Build token weights (using current epoch)
        if use_token_weighting:
            token_weights = build_token_weights(
                m=event_mask,
                epoch=epoch,
                total_epochs=total_epochs,
                stage=1,
                alpha0=alpha0,
                alpha1_max=alpha1_max,
                alpha_near=alpha_near,
                r_max_stage=r_max_stage,
                ratio_cap=ratio_cap,
                warmup_epochs=warmup_epochs,
            )  # (B, L)
        else:
            # Uniform weighting (all tokens have equal weight)
            token_weights = torch.ones_like(event_mask)  # (B, L)
        
        # Compute per-token MSE
        mse_spatial = (diff[:, :2, :] ** 2).sum(dim=1)  # (B, L)
        mse_indicator = diff[:, 2, :] ** 2  # (B, L)
        
        # Apply token weights
        loss = (token_weights * (mse_spatial + mse_indicator)).sum() / (token_weights.sum() + 1e-8)
        
        channel_mse = (diff ** 2).mean(dim=(0, 2))
        if use_bce:
            stay_target = batch["stay"].to(device)
            x0_pred = schedule.predict_start_from_noise(x_t, timesteps, eps_hat)
            stay_pred = x0_pred[:, 2, :].clamp(0.0, 1.0)
            bce_loss = F.binary_cross_entropy(stay_pred, stay_target, reduction="mean")
            loss = loss + stay_bce_weight * bce_loss
        else:
            bce_loss = torch.tensor(0.0, device=device)

        total_loss += float(loss.detach().cpu())
        channel_accum += channel_mse.detach()
        if use_bce:
            bce_accum += float(bce_loss.detach().cpu())
        batches += 1

    metrics = {
        "loss": total_loss / max(batches, 1),
        "channel_mse": (channel_accum / max(batches, 1)).detach().cpu().tolist(),
    }
    if use_bce:
        metrics["stay_bce"] = bce_accum / max(batches, 1)
    return metrics


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
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "betas": schedule.betas.cpu(),
        "global_step": global_step,
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

    # Build project-based output directory: outputs/{project_name}_{date}/stage1
    output_root = Path(args.output_dir) if args.output_dir else Path(config.stage1.training.output_dir)
    project_name = config.project.name
    date_str = dt.datetime.now().strftime("%Y%m%d")
    run_dir = output_root / f"{project_name}_{date_str}" / "stage1"
    ckpt_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    device = prepare_device(config, args.device)
    set_seed(int(getattr(getattr(config, "project", None), "seed", 42)))
    torch.backends.cudnn.benchmark = device.type == "cuda"

    logger = Logger(
        name="stage1_train",
        colorize=True,
        log_path=run_dir / "train.log",
    )
    logger.info(f"Using device: {device}")
    logger.info(f"Outputs will be stored in {run_dir}")

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f_out:
        with open(args.config, "r", encoding="utf-8") as f_in:
            f_out.write(f_in.read())

    train_loader = create_dataloaders(config, metadata, device, args.num_workers)
    val_loader = None
    stay_bce_weight = float(getattr(config.stage1.training, "stay_bce_weight", 0.0))
    stage1_val_file = data_dir / config.data.stage_files.stage1.val
    if stage1_val_file.exists() and int(metadata["split_sizes"]["stage1"]["val"]) > 0:
        val_dataset = Stage1SMTDataset(
            stage1_val_file,
            use_stay=stay_bce_weight > 0.0,
        )
        num_workers = int(args.num_workers if args.num_workers is not None else config.data.num_workers)
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config.stage1.training.batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

    model = build_model(config, metadata, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.stage1.training.learning_rate),
        weight_decay=float(getattr(config.stage1.training, "weight_decay", 0.0)),
    )
    schedule = setup_diffusion(config, device)

    amp_enabled = bool(config.stage1.training.amp) and device.type == "cuda"
    scaler = amp.GradScaler(enabled=amp_enabled)

    ema = None
    if getattr(config.stage1.training, "ema", False):
        ema = EMAHelper(mu=float(config.stage1.training.ema_decay))
        ema.register(model)

    start_epoch = 1
    best_val = float("inf")
    global_step = 0

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
            global_step = state.get("global_step", 0)
        else:
            logger.warning(f"Checkpoint {ckpt_path} not found; starting from scratch")

    for epoch in range(start_epoch, int(config.stage1.training.num_epochs) + 1):
        train_metrics = train_one_epoch(
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
        )
        global_step = train_metrics["global_step"]
        train_channels = train_metrics["channel_mse"]
        train_msg = (
            f"[Train] Epoch {epoch} | Loss {train_metrics['loss']:.4f} | "
            f"Channel MSE X {train_channels[0]:.4f} | "
            f"Y {train_channels[1]:.4f} | Stay {train_channels[2]:.4f}"
        )
        if "stay_bce" in train_metrics:
            train_msg += f" | Stay BCE {train_metrics['stay_bce']:.4f}"
        logger.info(train_msg)

        if val_loader and epoch % int(config.stage1.training.val_interval) == 0:
            val_metrics = evaluate(model, val_loader, schedule, device, config, epoch, global_step)
            val_channels = val_metrics["channel_mse"]
            val_msg = (
                f"[Val]   Epoch {epoch} | Loss {val_metrics['loss']:.4f} | "
                f"Channel MSE X {val_channels[0]:.4f} | "
                f"Y {val_channels[1]:.4f} | Stay {val_channels[2]:.4f}"
            )
            if "stay_bce" in val_metrics:
                val_msg += f" | Stay BCE {val_metrics['stay_bce']:.4f}"
            logger.info(val_msg)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(best_path, model, optimizer, epoch, best_val, schedule, scaler, ema, global_step)
                logger.info(f"Saved new best checkpoint to {best_path}")

        if epoch % int(config.stage1.training.save_interval) == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val, schedule, scaler, ema, global_step)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    final_path = ckpt_dir / "last.pt"
    save_checkpoint(final_path, model, optimizer, epoch, best_val, schedule, scaler, ema, global_step)
    logger.info(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
