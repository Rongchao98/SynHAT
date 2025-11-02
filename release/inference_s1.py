#!/usr/bin/env python3
"""
Inference script for TG-2S-DARU-Lite Stage-1 model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from model.diffusion_schedule import DiffusionSchedule, make_beta_schedule
from model.tg2s_daru import UnifiedTG2SDARU
from model.simple_unet import SimpleUNet
from utils.general_utils import load_config_from_yaml
# Removed: latlon_to_local_xy, local_xy_to_latlon (no longer needed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 inference using TG-2S-DARU-Lite.")
    parser.add_argument("--config", default="config/sample_synhat.yaml", type=str, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Model checkpoint (.pt) to load.")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of trajectories to generate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for sampling.")
    parser.add_argument("--device", type=str, help="Device override, e.g. cpu or cuda:0.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter (0.0 = deterministic).")
    parser.add_argument("--output-dir", type=str, default="stage1_samples", help="Directory to store outputs.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for sampling.")
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also export generated arrays as .txt (flattened per sample).",
    )
    parser.add_argument(
        "--stay-temperature",
        type=float,
        default=1.0,
        help="Temperature scaling applied to stay logits before sigmoid (must be > 0).",
    )
    parser.add_argument(
        "--stay-bias",
        type=float,
        default=0.0,
        help="Additive bias applied to stay logits before sigmoid.",
    )
    parser.add_argument(
        "--stay-threshold",
        type=float,
        help="Threshold applied to calibrated stay probabilities when extracting stay events.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum trajectory length (number of stay events). Trajectories shorter than this will be filtered out.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(config) -> Dict:
    meta_path = Path(config.data.processed_dir) / config.data.stage_files.metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_device(config, override: str | None) -> torch.device:
    device_str = override or getattr(getattr(config, "project", None), "device", "cuda")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def build_model(config, metadata: Dict, device: torch.device) -> UnifiedTG2SDARU:
    model_cfg = config.stage1.model
    architecture = getattr(model_cfg, "architecture", "tg2sdaru").lower()

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
        ).to(device)
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
        ).to(device)
    return model


def setup_diffusion(config, device: torch.device) -> DiffusionSchedule:
    diff_cfg = config.stage1.diffusion
    betas = make_beta_schedule(
        schedule=diff_cfg.beta_schedule,
        timesteps=int(diff_cfg.timesteps),
        beta_start=float(diff_cfg.beta_start),
        beta_end=float(diff_cfg.beta_end),
    )
    return DiffusionSchedule(betas).to(device)


@torch.no_grad()
def ddim_sample(
    model: UnifiedTG2SDARU,
    schedule: DiffusionSchedule,
    num_steps: int,
    batch_size: int,
    num_slots: int,
    device: torch.device,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # CRITICAL FIX: Don't use t=999 where alpha_cumprod≈0 (causes 20,000x amplification!)
    # Cosine schedule has sqrt(alpha_cumprod[999])≈0.00005, leading to division by near-zero
    # Use t=950 as max where sqrt(alpha_cumprod)≈0.076 (stable)
    max_t = min(950, schedule.timesteps - 1)
    indices = torch.linspace(0, max_t, num_steps, device=device).long()
    x = torch.randn(batch_size, 3, num_slots, device=device)

    for idx in reversed(range(num_steps)):
        t = torch.full((batch_size,), indices[idx], dtype=torch.long, device=device)
        if idx == 0:
            eps_pred, _ = model(x, t)
            x = schedule.predict_start_from_noise(x, t, eps_pred)
        else:
            t_prev = torch.full((batch_size,), indices[idx - 1], dtype=torch.long, device=device)
            x = schedule.ddim_step(model, x, t, t_prev, eta=eta)

    stay_channel = x[:, 2:3, :]
    return x, stay_channel


def main() -> None:
    args = parse_args()
    config = load_config_from_yaml(args.config)
    metadata = load_metadata(config)
    device = prepare_device(config, args.device)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config, metadata, device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    schedule = setup_diffusion(config, device)

    # Use single lat/lon normalization (no XY or anchor-centering)
    norm_cfg = metadata.get("normalization")
    if norm_cfg is None:
        raise KeyError("Metadata missing normalization statistics.")
    latlon_mean = np.array(norm_cfg["mean"], dtype=np.float32)
    latlon_std = np.array(norm_cfg["std"], dtype=np.float32)

    num_samples = int(args.num_samples)
    batch_size = int(args.batch_size)
    sample_steps = int(getattr(config.stage1.diffusion, "ddim_steps", min(50, schedule.timesteps)))
    eta = float(args.eta)
    stay_temp = float(args.stay_temperature)
    stay_bias = float(args.stay_bias)
    if stay_temp <= 0:
        raise ValueError("stay_temperature must be > 0.")
    slot_minutes = int(metadata["slot_minutes"])

    if args.stay_threshold is not None:
        stay_threshold = float(args.stay_threshold)
    elif getattr(getattr(getattr(config, "inference", None), "stage1", None), "stay_threshold", None) is not None:
        stay_threshold = float(config.inference.stage1.stay_threshold)
    else:
        stay_threshold = 0.5

    # Result arrays (removed XY-related arrays: local_xy, coarse_norm, anchors_used)
    raw_samples = []
    raw_stay = []
    stay_probs = []
    latlon_list = []
    latlon_norm_list = []

    # Note: origin is no longer needed for XY conversion, but kept for potential compatibility
    if "origin" in metadata:
        origin = metadata["origin"]
    else:
        catalog = np.load(Path(config.data.processed_dir) / config.data.stage_files.poi_catalog)
        latlon_centroid = catalog["poi_latlon"].mean(axis=0)
        origin = {"lat": float(latlon_centroid[0]), "lon": float(latlon_centroid[1])}
    origin_lat = float(origin["lat"])
    origin_lon = float(origin["lon"])

    for start in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - start)
        x, stay_channel = ddim_sample(
            model=model,
            schedule=schedule,
            num_steps=sample_steps,
            batch_size=current_batch,
            num_slots=int(metadata["num_slots"]),
            device=device,
            eta=eta,
        )

        x_cpu = x.detach().cpu().numpy()
        stay_raw = stay_channel.detach().cpu().numpy()

        # Transpose to (batch, T, 3)
        x_norm = x_cpu.transpose(0, 2, 1)
        
        # Denormalize lat/lon coordinates directly (no XY conversion, no anchor-centering)
        latlon_coords = x_norm[:, :, :2] * latlon_std.reshape(1, 1, -1) + latlon_mean.reshape(1, 1, -1)
        
        # Calibrate stay probabilities from raw stay channel
        # stay_raw shape: (batch, 1, T) -> transpose to (batch, T, 1)
        stay_raw_transposed = stay_raw.transpose(0, 2, 1)
        raw_samples.append(x_cpu.astype(np.float32))
        raw_stay.append(stay_raw.astype(np.float32))
        calibrated = 1.0 / (1.0 + np.exp(-((stay_raw_transposed + stay_bias) / stay_temp)))
        stay_probs_calibrated = np.clip(calibrated, 0.0, 1.0).astype(np.float32)
        
        # Combine denormalized coords with calibrated stay probs (N, T, 3)
        latlon = np.concatenate([latlon_coords, stay_probs_calibrated], axis=2)
        latlon_list.append(latlon)
        latlon_norm_list.append(x_norm.astype(np.float32))
        stay_probs.append(stay_probs_calibrated)

    raw_samples = np.concatenate(raw_samples, axis=0)
    raw_stay_arr = np.concatenate(raw_stay, axis=0)
    latlon_arr = np.concatenate(latlon_list, axis=0)  # Already (N, T, 3) with calibrated stay probs
    latlon_norm_arr = np.concatenate(latlon_norm_list, axis=0)
    stay_probs = np.concatenate(stay_probs, axis=0).squeeze(-1)  # Remove last dim: (N, T, 1) -> (N, T)

    # latlon_arr and latlon_norm_arr already contain stay probabilities as 3rd channel
    # Use normalized coordinates with normalized stay for raw_sequence
    raw_sequence = latlon_norm_arr.astype(np.float32)
    # Use denormalized coordinates with calibrated stay for latlon_sequence  
    latlon_sequence = latlon_arr.astype(np.float32)

    # Removed XY-related arrays: local_xy, anchors, coarse_norm
    stay_mask = stay_probs >= stay_threshold
    if stay_mask.ndim != 2:
        raise ValueError(f"Expected stay probabilities with shape (N, T); got {stay_probs.shape}.")
    fallback_indices = np.argmax(stay_probs, axis=1)
    for idx, mask in enumerate(stay_mask):
        if not np.any(mask):
            mask[fallback_indices[idx]] = True

    num_slots = stay_mask.shape[1]
    slot_times = (np.arange(num_slots, dtype=np.float32) * float(slot_minutes)).astype(np.float32)
    stay_event_trajs = np.full((stay_mask.shape[0], num_slots, 3), np.nan, dtype=np.float32)
    stay_event_counts = np.zeros(stay_mask.shape[0], dtype=np.int32)
    stay_traj_list = []

    for sample_idx, mask in enumerate(stay_mask):
        selected = np.nonzero(mask)[0]
        stay_event_counts[sample_idx] = selected.size
        if selected.size > 0:
            stay_event_trajs[sample_idx, selected, 0] = slot_times[selected]
            stay_event_trajs[sample_idx, selected, 1:] = latlon_sequence[sample_idx, selected, :2]
        traj_serialised = [
            [
                float(slot_times[slot]),
                float(latlon_sequence[sample_idx, slot, 0]),
                float(latlon_sequence[sample_idx, slot, 1]),
            ]
            for slot in selected
        ]
        stay_traj_list.append(traj_serialised)

    # Filter trajectories by minimum length
    min_length = int(args.min_length)
    valid_indices = np.where(stay_event_counts >= min_length)[0]
    
    if valid_indices.size == 0:
        raise RuntimeError(
            f"No trajectories meet minimum length requirement ({min_length} events). "
            f"Max trajectory length found: {stay_event_counts.max()}. "
            f"Consider lowering --min-length or --stay-threshold."
        )
    
    num_filtered = stay_mask.shape[0] - valid_indices.size
    print(f"\nFiltering trajectories by min-length={min_length}:")
    print(f"  Generated:  {stay_mask.shape[0]}")
    print(f"  Valid:      {valid_indices.size}")
    print(f"  Filtered:   {num_filtered} ({100 * num_filtered / stay_mask.shape[0]:.1f}%)")
    
    # Apply filtering to all arrays
    raw_samples = raw_samples[valid_indices]
    raw_stay_arr = raw_stay_arr[valid_indices]
    latlon_arr = latlon_arr[valid_indices]
    latlon_norm_arr = latlon_norm_arr[valid_indices]
    stay_probs = stay_probs[valid_indices]
    raw_sequence = raw_sequence[valid_indices]
    latlon_sequence = latlon_sequence[valid_indices]
    stay_mask = stay_mask[valid_indices]
    stay_event_trajs = stay_event_trajs[valid_indices]
    stay_event_counts = stay_event_counts[valid_indices]
    stay_traj_list = [stay_traj_list[i] for i in valid_indices]

    arrays = {
        "latlon": latlon_arr,
        "latlon_norm": latlon_norm_arr,
        "stay_probs": stay_probs,
        "raw_samples": raw_samples,
        "raw_stay": raw_stay_arr,
        "raw_sequence": raw_sequence,
        "latlon_sequence": latlon_sequence,
        "stay_event_mask": stay_mask.astype(np.uint8),
        "stay_event_trajs": stay_event_trajs,
        "stay_event_counts": stay_event_counts,
        "coarse_norm": latlon_norm_arr[..., :2],
        "anchors": latlon_arr[:, 0, :2],
    }
    np.savez_compressed(output_dir / "stage1_samples.npz", **arrays)

    stay_json_path = output_dir / "stay_trajectories.json"
    with open(stay_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "slot_minutes": slot_minutes,
                "stay_threshold": stay_threshold,
                "trajectories": stay_traj_list,
            },
            f,
            indent=2,
        )

    txt_exports = None
    if args.save_txt:
        txt_dir = output_dir / "txt_exports"
        txt_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            txt_dir / "raw_sequence.txt",
            raw_sequence.reshape(raw_sequence.shape[0], -1),
            fmt="%.6f",
        )
        np.savetxt(
            txt_dir / "latlon_sequence.txt",
            latlon_sequence.reshape(latlon_sequence.shape[0], -1),
            fmt="%.6f",
        )
        txt_exports = str(txt_dir)

    summary = {
        "num_samples": int(latlon_arr.shape[0]),
        "num_samples_generated": int(num_samples),
        "num_samples_filtered": int(num_filtered),
        "filter_rate": float(num_filtered / num_samples),
        "min_length": int(min_length),
        "num_slots": int(metadata["num_slots"]),
        "slot_minutes": metadata["slot_minutes"],
        "eta": eta,
        "ddim_steps": sample_steps,
        "checkpoint": str(args.checkpoint),
        "stay_temperature": stay_temp,
        "stay_bias": stay_bias,
        "raw_keys": ["raw_samples", "raw_stay", "latlon_norm"],
        "stay_threshold": stay_threshold,
    }
    if txt_exports is not None:
        summary["txt_exports_dir"] = txt_exports
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Generated {latlon_arr.shape[0]} trajectories. "
        f"Outputs saved to {output_dir} (stay events: {stay_json_path.name})."
    )


if __name__ == "__main__":
    main()
