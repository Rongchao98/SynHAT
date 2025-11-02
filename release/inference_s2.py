#!/usr/bin/env python3
"""Stage-2 inference: sample fine-grained blocks conditioned on Stage-1 outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from model.diffusion_schedule import DiffusionSchedule, make_beta_schedule
from model.tg2s_daru import UnifiedTG2SDARU
from utils.general_utils import load_config_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 block inference")
    parser.add_argument("--config", default="config/sample_synhat.yaml", type=str, help="YAML config")
    parser.add_argument("--checkpoint", required=True, type=str, help="Stage-2 checkpoint path")
    parser.add_argument("--stage1-samples", required=True, type=str, help="Stage-1 samples NPZ")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0.0 deterministic)")
    parser.add_argument("--output-dir", type=str, default="stage2_blocks", help="Output directory")
    parser.add_argument("--stay-threshold", type=float, help="Override stay probability threshold")
    parser.add_argument("--min-sep", type=int, default=3, help="Minimum step separation between picked events")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for block sampling")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--indicator-threshold", type=float, default=0.7, help="Threshold for event indicator channel")
    parser.add_argument("--per-slot-max-events", type=int, default=4, help="Max events per time slot")
    parser.add_argument("--max-events-per-day", type=int, default=16, help="Max events per day")
    parser.add_argument("--min-global-mins", type=int, default=30, help="Minimum minutes between events globally")
    parser.add_argument("--min-length", type=int, default=3, help="Minimum trajectory length in events")
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


def build_model(
    config,
    metadata: Dict,
    sample_length: int,
    input_channels: int,
    extra_channels: int,
    global_dim: int,
    device: torch.device,
) -> UnifiedTG2SDARU:
    model_cfg = config.stage2.model
    model = UnifiedTG2SDARU(
        input_channels=input_channels,
        sequence_length=sample_length,
        cycle_length=sample_length,
        base_channels=int(model_cfg.base_channels),
        scales=int(model_cfg.scales),
        jitter_blocks=int(model_cfg.jitter_blocks),
        drift_blocks=int(model_cfg.drift_blocks),
        dropout=float(model_cfg.dropout),
        time_embed_dim=int(model_cfg.time_embed_dim),
        cond_embed_dim=int(model_cfg.embedding_dim),
        stay_head=False,
        extra_channels=extra_channels,
        global_cond_dim=global_dim,
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


def build_global_features(
    coarse_norm: np.ndarray,
    stay: np.ndarray,
    slot_index: int,
    num_slots: int,
    slots_per_day: int,
    duration_days: int,
) -> np.ndarray:
    prev_idx = max(slot_index - 1, 0)
    next_idx = min(slot_index + 1, num_slots - 1)
    mean_xy = coarse_norm.mean(axis=0)
    std_xy = coarse_norm.std(axis=0)
    curr = coarse_norm[slot_index]
    prev = coarse_norm[prev_idx]
    nxt = coarse_norm[next_idx]
    slot_frac = slot_index / (num_slots - 1 + 1e-6)
    tod_frac = (slot_index % slots_per_day) / slots_per_day
    dow_frac = (slot_index // slots_per_day) / duration_days
    stay_score = stay[slot_index]
    features = np.concatenate(
        [
            mean_xy,
            std_xy,
            curr,
            prev,
            nxt,
            np.array(
                [
                    slot_frac,
                    np.sin(2 * np.pi * tod_frac),
                    np.cos(2 * np.pi * tod_frac),
                    dow_frac,
                    stay_score,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return features.astype(np.float32)


def detect_events(
    fine_path: np.ndarray,
    slot_index: int,
    slot_minutes: int,
    fine_minutes: int,
    stay_score: float,
) -> List[int]:
    d1 = np.diff(fine_path, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    if len(fine_path) > 2:
        d2 = np.diff(fine_path, n=2, axis=0)
        curvature = np.linalg.norm(d2, axis=1)
        curvature = np.pad(curvature, (1, 1), mode="edge")
    else:
        curvature = np.zeros(len(fine_path))
    if speed.size > 0:
        speed = np.pad(speed, (0, 1), mode="edge")
    else:
        speed = np.zeros(len(fine_path))

    def zscore(x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        if std < 1e-6:
            return np.zeros_like(x)
        return (x - mean) / std

    score = 1 / (1 + np.exp(-(zscore(curvature) - zscore(speed))))
    threshold = max(0.4, min(0.85, 0.5 + 0.3 * stay_score))
    candidates = np.where(score >= threshold)[0]
    if candidates.size == 0:
        candidates = np.array([int(np.argmax(score))])
    return candidates.tolist()


@torch.no_grad()
def sample_residual(
    model: UnifiedTG2SDARU,
    schedule: DiffusionSchedule,
    base_path: torch.Tensor,
    global_cond: torch.Tensor,
    ddim_steps: int,
    eta: float,
    input_channels: int,
) -> torch.Tensor:
    device = base_path.device
    B, _, L = base_path.shape
    x = torch.randn(B, input_channels, L, device=device)
    
    # CRITICAL FIX: Avoid numerical instability at extreme timesteps
    # Cosine schedule has alpha ≈ 0 at t=999, causing 20,000x amplification
    # Cap max_t to avoid near-zero alphas (similar to Stage-1 fix)
    max_t = min(950, schedule.timesteps - 1)
    indices = torch.linspace(0, max_t, ddim_steps, device=device).long()

    for idx in reversed(range(ddim_steps)):
        t = torch.full((B,), indices[idx], dtype=torch.long, device=device)
        cond = {"extra_channels": base_path, "global_cond": global_cond}
        if idx == 0:
            eps_hat, _ = model(x, t, conditioning=cond)
            x = schedule.predict_start_from_noise(x, t, eps_hat)
        else:
            t_prev = torch.full((B,), indices[idx - 1], dtype=torch.long, device=device)
            x = schedule.ddim_step(lambda xt, ts: model(xt, ts, conditioning=cond), x, t, t_prev, eta=eta)
    return x


def main() -> None:
    args = parse_args()
    config = load_config_from_yaml(args.config)
    metadata = load_metadata(config)
    device = prepare_device(config, args.device)
    set_seed(args.seed)

    coord_mean = np.array(metadata["normalization"]["mean"], dtype=np.float32)
    coord_std = np.array(metadata["normalization"]["std"], dtype=np.float32)

    with np.load(args.stage1_samples) as stage1_data:
        latlon_abs = None
        if "coarse_norm" in stage1_data:
            coarse_norm = stage1_data["coarse_norm"].astype(np.float32)
            if "latlon" in stage1_data:
                latlon_abs = stage1_data["latlon"][..., :2].astype(np.float32)
            elif "latlon_sequence" in stage1_data:
                latlon_abs = stage1_data["latlon_sequence"][..., :2].astype(np.float32)
        else:
            if "latlon_norm" in stage1_data:
                coarse_norm = stage1_data["latlon_norm"][..., :2].astype(np.float32)
            else:
                if "latlon_sequence" in stage1_data:
                    latlon_abs = stage1_data["latlon_sequence"][..., :2].astype(np.float32)
                elif "latlon" in stage1_data:
                    latlon_abs = stage1_data["latlon"][..., :2].astype(np.float32)
                else:
                    raise KeyError("Stage-1 samples must contain 'coarse_norm', 'latlon_norm', or 'latlon_sequence'.")
                coarse_norm = (latlon_abs - coord_mean.reshape(1, 1, -1)) / coord_std.reshape(1, 1, -1)

        if latlon_abs is None:
            latlon_abs = (coarse_norm * coord_std.reshape(1, 1, -1)) + coord_mean.reshape(1, 1, -1)

        if "anchors" in stage1_data:
            anchors = stage1_data["anchors"].astype(np.float32)
        else:
            anchors = latlon_abs[:, 0, :].astype(np.float32)

        if "stay_probs" in stage1_data:
            stay_probs = stage1_data["stay_probs"].astype(np.float32)
        elif "latlon_sequence" in stage1_data:
            stay_probs = stage1_data["latlon_sequence"][..., 2].astype(np.float32)
        else:
            raise KeyError("Stage-1 samples missing stay probabilities ('stay_probs').")

    stay_threshold = args.stay_threshold if args.stay_threshold is not None else float(config.inference.stage1.stay_threshold)
    stage2_infer_cfg = getattr(getattr(config, "inference", None), "stage2", None)
    if args.batch_size is not None:
        block_batch_size = int(args.batch_size)
    elif stage2_infer_cfg is not None and getattr(stage2_infer_cfg, "batch_size", None) is not None:
        block_batch_size = int(stage2_infer_cfg.batch_size)
    else:
        block_batch_size = 64

    fine_steps = int(metadata["fine_steps"])
    slot_minutes = int(metadata["slot_minutes"])
    fine_minutes = int(metadata["fine_minutes"])
    num_slots = int(metadata["num_slots"])
    slots_per_day = int(metadata["slots_per_day"])
    duration_days = int(metadata["duration_days"])
    half_steps = 0  # EFFICIENCY FIX: No padding, matching training data
    window_len = fine_steps  # Now just 60 instead of 120

    def _lerp_batch(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
        if steps <= 0:
            return np.zeros((start.shape[0], 0, start.shape[1]), dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float32)
        return start[:, None, :] + alpha[None, :, None] * (end - start)[:, None, :]

    # Dummy values for model init
    sample_length = window_len
    input_channels = 3
    extra_channels = 3
    global_dim = build_global_features(
        coarse_norm[0],
        stay_probs[0],
        0,
        num_slots,
        slots_per_day,
        duration_days,
    ).shape[0]

    model = build_model(
        config,
        metadata,
        sample_length,
        input_channels,
        extra_channels,
        global_dim,
        device,
    )
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    schedule = setup_diffusion(config, device)
    ddim_steps = int(getattr(config.stage2.diffusion, "ddim_steps", min(50, schedule.timesteps)))

    indicator_threshold = args.indicator_threshold  # Threshold for event detection (default: 0.80)
    min_sep = max(int(args.min_sep), 1)
    per_slot_max = max(int(args.per_slot_max_events), 1)
    max_events_day = max(int(args.max_events_per_day), 1)  # Max events per day
    min_global_mins = max(int(args.min_global_mins), 1)  # Minimum minutes between events

    event_xy_norm: List[np.ndarray] = []
    event_xy_abs: List[np.ndarray] = []
    event_times: List[np.ndarray] = []
    event_slots: List[np.ndarray] = []
    event_days: List[np.ndarray] = []
    event_sequences: List[np.ndarray] = []
    
    # DEBUG: Save raw generated data per day/slot for inspection
    total_events = 0
    total_days = coarse_norm.shape[0]
    progress_interval = max(1, total_days // 10)

    for day_idx in range(total_days):
        coarse_day_norm = coarse_norm[day_idx]
        stay_day = stay_probs[day_idx]
        coarse_day_abs = coarse_day_norm * coord_std[None, :] + coord_mean[None, :]

        visited_slots = np.where(stay_day >= stay_threshold)[0]
        if visited_slots.size == 0:
            visited_slots = np.array([int(np.argmax(stay_day))])

        day_events_xy_abs: List[np.ndarray] = []
        day_events_xy_norm: List[np.ndarray] = []
        day_events_time: List[float] = []
        day_event_slots: List[int] = []
        day_event_scores: List[float] = []

        for start in range(0, visited_slots.size, block_batch_size):
            batch_slots = visited_slots[start : start + block_batch_size]
            prev_indices = np.clip(batch_slots - 1, 0, num_slots - 1)
            next_indices = np.clip(batch_slots + 1, 0, num_slots - 1)

            prev_coords = coarse_day_abs[prev_indices]
            curr_coords = coarse_day_abs[batch_slots]
            next_coords = coarse_day_abs[next_indices]

            prev_path = _lerp_batch(prev_coords, curr_coords, half_steps)
            slot_base = _lerp_batch(curr_coords, next_coords, fine_steps)
            next_path = _lerp_batch(curr_coords, next_coords, half_steps)

            base_paths = np.concatenate([prev_path, slot_base, next_path], axis=1).astype(np.float32)
            base_paths_norm = (base_paths - coord_mean) / coord_std
            base_paths_full = np.concatenate(
                [base_paths_norm, np.zeros((base_paths.shape[0], base_paths.shape[1], 1), dtype=np.float32)],
                axis=2,
            )
            base_tensor = torch.from_numpy(base_paths_full.transpose(0, 2, 1)).to(device)

            global_feats = np.stack(
                [
                    build_global_features(coarse_day_norm, stay_day, int(slot), num_slots, slots_per_day, duration_days)
                    for slot in batch_slots
                ],
                axis=0,
            ).astype(np.float32)
            global_tensor = torch.from_numpy(global_feats).to(device)

            # Model predicts absolute coordinates (not residuals)
            predicted = sample_residual(
                model,
                schedule,
                base_tensor,
                global_tensor,
                ddim_steps=ddim_steps,
                eta=float(args.eta),
                input_channels=input_channels,
            )
            predicted_np = predicted.cpu().numpy().transpose(0, 2, 1)
            coords_norm = predicted_np[..., :2]  # Already absolute, no need to add base_path
            coords_abs = coords_norm * coord_std[None, None, :] + coord_mean[None, None, :]
            indicator_batch = np.clip(predicted_np[..., 2], 0.0, 1.0)

            for idx_batch, slot in enumerate(batch_slots):
                coords_abs_slot = coords_abs[idx_batch]
                indicator_slot = indicator_batch[idx_batch]

                central_coords_abs = coords_abs_slot[half_steps:half_steps + fine_steps]
                central_indicator = indicator_slot[half_steps:half_steps + fine_steps]

                cand_idx = np.where(central_indicator >= indicator_threshold)[0]
                if cand_idx.size == 0:
                    # Fallback: take the single best step
                    best = int(np.argmax(central_indicator))
                    cand_idx = np.array([best], dtype=np.int64)
                # Greedy non-max suppression by value with min separation
                order = np.argsort(central_indicator[cand_idx])[::-1]
                selected = []
                for j in order:
                    t = int(cand_idx[j])
                    if all(abs(t - s) >= min_sep for s in selected):
                        selected.append(t)
                selected.sort()
                for idx_micro in selected[:per_slot_max]:
                    time_offset = slot * slot_minutes + idx_micro * fine_minutes
                    day_event_slots.append(int(slot))
                    day_events_xy_abs.append(central_coords_abs[idx_micro])
                    day_events_xy_norm.append(coords_norm[idx_batch, half_steps + idx_micro])
                    day_events_time.append(time_offset)
                    day_event_scores.append(float(central_indicator[idx_micro]))

        if not day_events_xy_norm:
            continue

        # Global pruning by score with minimum time separation and daily cap
        times_arr = np.array(day_events_time, dtype=np.float32)
        scores_arr = np.array(day_event_scores, dtype=np.float32)
        abs_arr = np.array(day_events_xy_abs, dtype=np.float32)
        norm_arr = np.array(day_events_xy_norm, dtype=np.float32)
        slots_arr = np.array(day_event_slots, dtype=np.int32)

        order_by_score = np.argsort(scores_arr)[::-1]
        kept_idx: List[int] = []
        for j in order_by_score:
            t = times_arr[j]
            if all(abs(t - times_arr[k]) >= min_global_mins for k in kept_idx):
                kept_idx.append(int(j))
            if len(kept_idx) >= max_events_day:
                break
        kept_idx.sort(key=lambda k: times_arr[k])

        events_xy_abs_sorted = abs_arr[kept_idx]
        events_xy_norm_sorted = norm_arr[kept_idx]
        events_time_sorted = times_arr[kept_idx]
        events_slots_sorted = slots_arr[kept_idx]
        day_ids_sorted = np.full(events_slots_sorted.shape, day_idx, dtype=np.int32)

        event_xy_abs.append(events_xy_abs_sorted)
        event_xy_norm.append(events_xy_norm_sorted)
        event_times.append(events_time_sorted)
        event_slots.append(events_slots_sorted)
        event_days.append(day_ids_sorted)
        day_sequence = np.column_stack(
            (events_time_sorted.astype(np.float32), events_xy_abs_sorted.astype(np.float32))
        )
        event_sequences.append(day_sequence)
        total_events += events_xy_norm_sorted.shape[0]

        if (day_idx + 1) % progress_interval == 0 or day_idx == total_days - 1:
            print(
                f"[Stage-2] processed {day_idx + 1}/{total_days} days | slots >= thr: {len(visited_slots)} | "
                f"cumulative events: {total_events}",
                flush=True,
            )

    if not event_xy_norm:
        raise RuntimeError("Stage-2 inference produced no events; adjust threshold or sampling parameters.")

    event_xy_abs_arr = np.concatenate(event_xy_abs, axis=0).astype(np.float32)
    event_xy_norm_arr = np.concatenate(event_xy_norm, axis=0).astype(np.float32)
    event_times_arr = np.concatenate(event_times, axis=0).astype(np.float32)
    event_slots_arr = np.concatenate(event_slots, axis=0).astype(np.int32)
    event_days_arr = np.concatenate(event_days, axis=0).astype(np.int32)
    event_sequences_arr = np.array(event_sequences, dtype=object)

    # Filter trajectories by minimum length
    min_length = int(args.min_length)
    sequence_lengths = np.array([len(seq) for seq in event_sequences_arr], dtype=np.int32)
    valid_traj_indices = np.where(sequence_lengths >= min_length)[0]
    
    if valid_traj_indices.size == 0:
        raise RuntimeError(
            f"No trajectories meet minimum length requirement ({min_length} events). "
            f"Max trajectory length found: {sequence_lengths.max()}. "
            f"Consider lowering --min-length or adjusting thresholds."
        )
    
    num_filtered_trajs = len(event_sequences_arr) - valid_traj_indices.size
    print(f"\nFiltering trajectories by min-length={min_length}:")
    print(f"  Generated:  {len(event_sequences_arr)}")
    print(f"  Valid:      {valid_traj_indices.size}")
    print(f"  Filtered:   {num_filtered_trajs} ({100 * num_filtered_trajs / len(event_sequences_arr):.1f}%)")
    
    # Filter event_sequences_arr
    event_sequences_arr = event_sequences_arr[valid_traj_indices]
    
    # Filter individual event arrays by trajectory ID
    valid_event_mask = np.isin(event_days_arr, valid_traj_indices)
    event_xy_abs_arr = event_xy_abs_arr[valid_event_mask]
    event_xy_norm_arr = event_xy_norm_arr[valid_event_mask]
    event_times_arr = event_times_arr[valid_event_mask]
    event_slots_arr = event_slots_arr[valid_event_mask]
    event_days_arr = event_days_arr[valid_event_mask]
    
    # Remap day IDs to be contiguous (0, 1, 2, ...) after filtering
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(valid_traj_indices)}
    event_days_arr = np.array([old_to_new[old_id] for old_id in event_days_arr], dtype=np.int32)
    
    # Update total_events after filtering
    total_events = event_xy_norm_arr.shape[0]

    # Compute delta_next per day for Stage-3 features
    delta_next = np.zeros_like(event_times_arr, dtype=np.float32)
    for idx in range(len(event_times_arr) - 1):
        if event_days_arr[idx] == event_days_arr[idx + 1]:
            delta_next[idx] = event_times_arr[idx + 1] - event_times_arr[idx]
        else:
            delta_next[idx] = slot_minutes
    delta_next[-1] = slot_minutes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter anchors and coarse_norm to match filtered trajectories
    anchors_filtered = anchors[valid_traj_indices]
    coarse_norm_filtered = coarse_norm[valid_traj_indices]
    
    # Save main output
    np.savez_compressed(
        output_dir / "stage2_events.npz",
        event_xy_norm=event_xy_norm_arr,
        event_xy_abs=event_xy_abs_arr,
        event_time=event_times_arr,
        slot_index=event_slots_arr,
        day_id=event_days_arr,
        delta_next=delta_next,
        anchors=anchors_filtered,
        coarse_norm=coarse_norm_filtered,
        event_sequences=event_sequences_arr,
        trajectory_ids=event_days_arr,  # Add explicit trajectory IDs
    )
    
    # Compute detailed statistics
    event_counts_per_seq = [len(seq) for seq in event_sequences_arr]
    import collections
    count_dist = collections.Counter(event_counts_per_seq)
    
    summary = {
        "num_days": int(coarse_norm.shape[0]),
        "num_days_generated": int(total_days),
        "num_days_filtered": int(num_filtered_trajs),
        "filter_rate": float(num_filtered_trajs / total_days),
        "min_length": int(min_length),
        "num_events": int(event_xy_norm_arr.shape[0]),
        "num_sequences": int(event_sequences_arr.shape[0]),
        "mean_events_per_sequence": float(event_xy_norm_arr.shape[0] / max(event_sequences_arr.shape[0], 1)),
        "ddim_steps": ddim_steps,
        "eta": float(args.eta),
        "stay_threshold": stay_threshold,
        "indicator_threshold": indicator_threshold,
        "max_events_per_day": max_events_day,
        "min_global_mins": min_global_mins,
        "per_slot_max_events": per_slot_max,
        "checkpoint": str(args.checkpoint),
        "stage1_samples": str(args.stage1_samples),
        "coordinate_stats": {
            "lat_range": [float(event_xy_abs_arr[:,0].min()), float(event_xy_abs_arr[:,0].max())],
            "lon_range": [float(event_xy_abs_arr[:,1].min()), float(event_xy_abs_arr[:,1].max())],
            "lat_mean": float(event_xy_abs_arr[:,0].mean()),
            "lon_mean": float(event_xy_abs_arr[:,1].mean()),
        },
        "event_distribution": {
            "min": int(min(event_counts_per_seq)) if event_counts_per_seq else 0,
            "max": int(max(event_counts_per_seq)) if event_counts_per_seq else 0,
            "sequences_at_max_cap": int(count_dist.get(max_events_day, 0)),
            "pct_at_max_cap": float(100 * count_dist.get(max_events_day, 0) / max(len(event_counts_per_seq), 1)),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Stage-2 Inference Complete")
    print(f"{'='*70}")
    print(f"Events saved to: {output_dir / 'stage2_events.npz'}")
    print(f"Summary:         {output_dir / 'summary.json'}")
    print(f"\nKey Statistics:")
    print(f"  Trajectories:     {summary['num_sequences']}")
    print(f"  Total events:     {summary['num_events']}")
    print(f"  Events/sequence:  {summary['mean_events_per_sequence']:.2f}")
    print(f"  Sequences at cap: {summary['event_distribution']['sequences_at_max_cap']} ({summary['event_distribution']['pct_at_max_cap']:.1f}%)")
    print(f"\nCoordinate Ranges:")
    print(f"  Latitude:  [{summary['coordinate_stats']['lat_range'][0]:.4f}, {summary['coordinate_stats']['lat_range'][1]:.4f}]")
    print(f"  Longitude: [{summary['coordinate_stats']['lon_range'][0]:.4f}, {summary['coordinate_stats']['lon_range'][1]:.4f}]")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
