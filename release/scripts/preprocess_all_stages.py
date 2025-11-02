"""
Preprocess raw check-in data to produce datasets for:
  * Stage-1 (coarse SMT diffusion)
  * Stage-2 (fine-grained block diffusion)
  * Stage-3 (POI recovery)

The script consumes a YAML config describing data locations and preprocessing
parameters, then materialises train/val/test splits for all three stages plus
metadata and POI catalog files required by the training and inference pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from types import SimpleNamespace

import numpy as np
import pandas as pd

# Ensure repository modules are importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.general_utils import load_config_from_yaml
from utils.smt_utils import build_smt


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageFiles:
    train: str
    val: str
    test: str


@dataclass
class PreprocessConfig:
    city: str
    raw_path: Path
    output_dir: Path
    slot_minutes: int
    fine_minutes: int
    duration_days: int
    min_points_per_traj: int
    stay_neighbor_value: float
    train_ratio: float
    val_ratio: float
    test_ratio: float
    start_date: Optional[str]
    end_date: Optional[str]
    seed: int
    max_trajectories: Optional[int]
    reference_lat: Optional[float]
    reference_lon: Optional[float]
    candidate_topk: int
    poi_time_bins: int
    num_workers: int
    stage1_files: StageFiles
    stage2_files: StageFiles
    stage3_files: StageFiles
    metadata_file: str
    poi_catalog_file: str
    time_columns: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_minutes(self) -> int:
        return self.duration_days * 24 * 60

    @property
    def num_slots(self) -> int:
        return self.duration_minutes // self.slot_minutes

    @property
    def slots_per_day(self) -> int:
        return (24 * 60) // self.slot_minutes

    @property
    def fine_steps(self) -> int:
        if self.slot_minutes % self.fine_minutes != 0:
            raise ValueError("slot_minutes must be divisible by fine_minutes.")
        return self.slot_minutes // self.fine_minutes

    @property
    def time_bin_minutes(self) -> float:
        return (24 * 60) / float(self.poi_time_bins)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RawEvent:
    time_offset: float
    slot_index: int
    rel_time: float
    latlon: np.ndarray  # (2,) [lat, lon] in degrees
    poi_idx: int
    cat_idx: int


@dataclass
class EventRecord:
    time_offset: float
    slot_index: int
    rel_time: float
    latlon: np.ndarray  # (2,) [lat, lon] in degrees
    poi_idx: int
    cat_idx: int
    delta_next: float = 0.0


@dataclass
class SlotRecord:
    slot_index: int
    entry_latlon: np.ndarray  # (2,) [lat, lon]
    exit_latlon: np.ndarray  # (2,) [lat, lon]
    events: List[EventRecord]


@dataclass
class DayRecord:
    traj_id: int
    coarse: np.ndarray  # (T, 2) [lat, lon] coordinates
    stay: np.ndarray  # (T,) stay probabilities
    anchor: np.ndarray  # (2,) first slot [lat, lon] 
    counts: np.ndarray  # (T,) visit counts per slot
    slot_c1: np.ndarray  # (T, 2) first visit [lat, lon]
    slot_cn: np.ndarray  # (T, 2) last visit [lat, lon]
    slots: List[SlotRecord]
    events: List[EventRecord]


# ---------------------------------------------------------------------------
# Argument parsing and config loading
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw check-ins into Stage-1/2/3 datasets."
    )
    parser.add_argument("--config", required=True, type=str, help="YAML config.")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory (default: config.data.processed_dir).",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        help="Override raw CSV path (default: config.data.raw_path).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override RNG seed (default: config.project.seed or 42).",
    )
    return parser.parse_args()


def _load_stage_files(stage_cfg) -> StageFiles:
    if isinstance(stage_cfg, dict):
        get_val = stage_cfg.get
    else:
        get_val = lambda key: getattr(stage_cfg, key)
    return StageFiles(
        train=get_val("train"),
        val=get_val("val"),
        test=get_val("test"),
    )


def load_preprocess_config(args: argparse.Namespace) -> PreprocessConfig:
    cfg = load_config_from_yaml(args.config)
    data_cfg = cfg.data

    time_columns_cfg = getattr(data_cfg, "time_columns", {})
    if isinstance(time_columns_cfg, SimpleNamespace):
        time_columns = vars(time_columns_cfg).copy()
    else:
        time_columns = dict(time_columns_cfg)
    required_columns = ["utc", "latitude", "longitude", "venue_id", "user_id"]
    optional_columns = ["venue_category_name", "timezone_offset"]
    defaults = {
        "local": "local_time",
        "time_offset": "time_offset",
        "traj_id": "traj_id",
    }
    for col in required_columns:
        if col not in time_columns:
            raise ValueError(f"Missing '{col}' in data.time_columns of {args.config}")
    for key, default in defaults.items():
        time_columns.setdefault(key, default)

    project_seed = getattr(getattr(cfg, "project", None), "seed", None)
    seed = args.seed if args.seed is not None else (project_seed if project_seed is not None else 42)

    stage_files_cfg = getattr(data_cfg, "stage_files", None)
    if stage_files_cfg is None:
        raise ValueError("Config must define data.stage_files with stage1/stage2/stage3 entries.")
    if isinstance(stage_files_cfg, dict):
        stage1_cfg = stage_files_cfg.get("stage1")
        stage2_cfg = stage_files_cfg.get("stage2")
        stage3_cfg = stage_files_cfg.get("stage3")
        metadata_file = stage_files_cfg.get("metadata", "stage_metadata.json")
        poi_catalog_file = stage_files_cfg.get("poi_catalog", "poi_catalog.npz")
    else:
        stage1_cfg = getattr(stage_files_cfg, "stage1")
        stage2_cfg = getattr(stage_files_cfg, "stage2")
        stage3_cfg = getattr(stage_files_cfg, "stage3")
        metadata_file = getattr(stage_files_cfg, "metadata", "stage_metadata.json")
        poi_catalog_file = getattr(stage_files_cfg, "poi_catalog", "poi_catalog.npz")
    if stage1_cfg is None or stage2_cfg is None or stage3_cfg is None:
        raise ValueError("stage_files must contain stage1, stage2, and stage3 entries.")

    return PreprocessConfig(
        city=getattr(data_cfg, "city"),
        raw_path=Path(args.raw_path) if args.raw_path else Path(getattr(data_cfg, "raw_path")),
        output_dir=Path(args.output_dir) if args.output_dir else Path(getattr(data_cfg, "processed_dir")),
        slot_minutes=int(getattr(data_cfg, "slot_minutes")),
        fine_minutes=int(getattr(data_cfg, "fine_minutes")),
        duration_days=int(getattr(data_cfg, "duration_days")),
        min_points_per_traj=int(getattr(data_cfg, "min_points_per_traj", 1)),
        stay_neighbor_value=float(getattr(data_cfg, "stay_neighbor_value", 0.5)),
        train_ratio=float(getattr(data_cfg, "train_ratio", 0.8)),
        val_ratio=float(getattr(data_cfg, "val_ratio", 0.1)),
        test_ratio=float(getattr(data_cfg, "test_ratio", 0.1)),
        start_date=getattr(data_cfg, "start_date", None),
        end_date=getattr(data_cfg, "end_date", None),
        seed=int(seed),
        max_trajectories=getattr(data_cfg, "max_trajectories", None),
        reference_lat=getattr(data_cfg, "reference_lat", None),
        reference_lon=getattr(data_cfg, "reference_lon", None),
        candidate_topk=int(getattr(data_cfg, "candidate_topk", 20)),
        poi_time_bins=int(getattr(data_cfg, "poi_time_bins", 24)),
        num_workers=int(getattr(data_cfg, "num_workers", 4)),
        stage1_files=_load_stage_files(stage1_cfg),
        stage2_files=_load_stage_files(stage2_cfg),
        stage3_files=_load_stage_files(stage3_cfg),
        metadata_file=str(metadata_file),
        poi_catalog_file=str(poi_catalog_file),
        time_columns=time_columns,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series.dt.tz_localize(None)
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def slice_trajectories(df: pd.DataFrame, cfg: PreprocessConfig) -> List[pd.DataFrame]:
    user_col = cfg.time_columns["user_id"]
    utc_col = cfg.time_columns["utc"]

    df = df.copy()
    df[utc_col] = ensure_datetime(df[utc_col])
    
    # Handle timezone offset if available
    if "timezone_offset" in cfg.time_columns and cfg.time_columns["timezone_offset"] in df.columns:
        tz_col = cfg.time_columns["timezone_offset"]
        df["local_time"] = df[utc_col] + pd.to_timedelta(df[tz_col], unit="m")
        df["local_time"] = df["local_time"].dt.tz_localize(None)
    else:
        # If no timezone offset, assume utc_col is already local_time
        df["local_time"] = df[utc_col]

    if cfg.start_date:
        start_dt = pd.to_datetime(cfg.start_date)
    else:
        start_dt = df["local_time"].min().floor("D")

    if cfg.end_date:
        end_dt = pd.to_datetime(cfg.end_date)
    else:
        end_dt = df["local_time"].max().ceil("D")

    duration = pd.Timedelta(days=cfg.duration_days)
    trajectories: List[pd.DataFrame] = []
    traj_id = 0

    for _, user_df in df.groupby(user_col):
        user_df = user_df.sort_values("local_time")
        current_start = start_dt
        while current_start < end_dt:
            current_end = current_start + duration
            mask = (user_df["local_time"] >= current_start) & (user_df["local_time"] < current_end)
            segment = user_df.loc[mask].copy()
            if len(segment) < cfg.min_points_per_traj:
                current_start = current_end
                continue
            segment["time_offset"] = (segment["local_time"] - current_start).dt.total_seconds() / 60.0
            segment["traj_id"] = traj_id
            trajectories.append(segment.reset_index(drop=True))
            traj_id += 1
            current_start = current_end

    if cfg.max_trajectories is not None:
        trajectories = trajectories[: int(cfg.max_trajectories)]

    return trajectories


def load_category_mapping(mapping_file: Path) -> Dict[str, str]:
    """Load fine-grained to macro category mapping from poi_categories.txt."""
    mapping = {}
    if not mapping_file.exists():
        print(f"[WARNING] Category mapping file not found: {mapping_file}")
        return mapping
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                fine_cat = parts[0].strip()
                macro_cat = parts[1].strip()
                mapping[fine_cat] = macro_cat
    
    unique_macros = len(set(mapping.values()))
    print(f"[INFO] Loaded category mapping: {len(mapping)} fine-grained → {unique_macros} macro categories")
    return mapping


def build_mappings(trajectories: Iterable[pd.DataFrame], cfg: PreprocessConfig) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build POI and category index mappings from trajectory data."""
    poi_mapping: Dict[str, int] = {}
    category_mapping: Dict[str, int] = {"UNKNOWN": 0}  # Default for missing values
    
    # Check if category column exists
    has_categories = "venue_category_name" in cfg.time_columns
    
    # Load macro category mapping from file if categories exist
    macro_category_map = {}
    if has_categories:
        macro_category_file = Path(__file__).parent / "poi_categories.txt"
        macro_category_map = load_category_mapping(macro_category_file)
    
    nan_count = 0
    for traj in trajectories:
        poi_vals = traj[cfg.time_columns["venue_id"]]
        
        # Build POI mapping
        for pid in poi_vals:
            if pid not in poi_mapping:
                poi_mapping[pid] = len(poi_mapping)
        
        # Build category mapping (fine → macro) if categories exist
        if has_categories and cfg.time_columns["venue_category_name"] in traj.columns:
            cat_vals = traj[cfg.time_columns["venue_category_name"]]
            for cat_name in cat_vals:
                if pd.isna(cat_name):
                    nan_count += 1
                    continue
                
                # Map to macro category
                macro_cat = macro_category_map.get(cat_name, cat_name)
                if macro_cat not in category_mapping:
                    category_mapping[macro_cat] = len(category_mapping)
    
    if has_categories:
        print(f"[INFO] Built mappings: {len(poi_mapping)} POIs, {len(category_mapping)} categories ({nan_count} NaN values)")
    else:
        print(f"[INFO] Built mappings: {len(poi_mapping)} POIs (no category data)")
    return poi_mapping, category_mapping


def _select_candidate_ids(
    xy_abs: np.ndarray,
    actual_idx: int,
    poi_positions: np.ndarray,
    topk: int,
) -> np.ndarray:
    """Return indices of top-k nearest POIs (ensure actual in list)."""
    if poi_positions.shape[0] <= topk:
        candidates = np.arange(poi_positions.shape[0], dtype=np.int32)
    else:
        dists = np.linalg.norm(poi_positions - xy_abs[None, :], axis=1)
        idx = np.argpartition(dists, topk - 1)[:topk]
        candidates = np.array(idx, dtype=np.int32)

    if actual_idx not in candidates:
        if len(candidates) < topk:
            candidates = np.concatenate([candidates, np.array([actual_idx], dtype=np.int32)])
        else:
            # Replace farthest candidate with actual index
            dists = np.linalg.norm(poi_positions[candidates] - xy_abs[None, :], axis=1)
            replace_idx = int(np.argmax(dists))
            candidates[replace_idx] = actual_idx

    # Pad if still shorter than topk (happens when num_pois < topk)
    if candidates.shape[0] < topk:
        pad_count = topk - candidates.shape[0]
        pad_vals = np.repeat(candidates[-1], pad_count)
        candidates = np.concatenate([candidates, pad_vals])
    return candidates.astype(np.int32)


def _interp_path(
    slot_minutes: int,
    fine_steps: int,
    entry_latlon: np.ndarray,
    exit_latlon: np.ndarray,
    events: Sequence[EventRecord],
) -> np.ndarray:
    """
    Construct fine-grained path with actual event coordinates filled at exact positions,
    and linear interpolation for unfilled gaps. This creates meaningful residuals.
    """
    if fine_steps <= 0:
        raise ValueError("fine_steps must be positive.")

    fine_minutes = slot_minutes / fine_steps
    
    # Initialize path with linear interpolation entry -> exit
    path = np.zeros((fine_steps, 2), dtype=np.float32)
    filled = np.zeros(fine_steps, dtype=bool)
    
    # Mark entry and exit
    path[0] = entry_latlon
    filled[0] = True
    # Note: We don't mark the last position as exit because it's the next slot's entry
    
    # Fill actual event positions with their exact coordinates
    for ev in events:
        idx = int(round(ev.rel_time / fine_minutes))
        idx = max(0, min(fine_steps - 1, idx))
        path[idx] = ev.latlon
        filled[idx] = True
    
    # Linear interpolation for unfilled positions
    # Find segments between filled positions and interpolate
    if not filled.any():
        # No events - just use linear from entry to exit
        alpha = np.linspace(0.0, 1.0, fine_steps, endpoint=False, dtype=np.float32)
        return entry_latlon[None, :] + alpha[:, None] * (exit_latlon - entry_latlon)[None, :]
    
    # Forward fill and backward fill to handle edges
    filled_indices = np.where(filled)[0]
    
    for i in range(fine_steps):
        if not filled[i]:
            # Find nearest filled positions before and after
            before_idx = filled_indices[filled_indices < i]
            after_idx = filled_indices[filled_indices > i]
            
            if len(before_idx) > 0 and len(after_idx) > 0:
                # Interpolate between before and after
                before = before_idx[-1]
                after = after_idx[0]
                alpha = (i - before) / (after - before)
                path[i] = (1 - alpha) * path[before] + alpha * path[after]
            elif len(before_idx) > 0:
                # Only before exists - extrapolate to exit
                before = before_idx[-1]
                alpha = (i - before) / (fine_steps - before)
                path[i] = (1 - alpha) * path[before] + alpha * exit_latlon
            elif len(after_idx) > 0:
                # Only after exists - extrapolate from entry
                after = after_idx[0]
                alpha = i / after
                path[i] = (1 - alpha) * entry_latlon + alpha * path[after]
    
    return path


def _linear_base_path(entry_latlon: np.ndarray, exit_latlon: np.ndarray, fine_steps: int) -> np.ndarray:
    """Linear baseline path between entry and exit (lat/lon coordinates)."""
    alpha = np.linspace(0.0, 1.0, fine_steps, endpoint=False, dtype=np.float32)
    return entry_latlon[None, :] + alpha[:, None] * (exit_latlon - entry_latlon)[None, :]


# ---------------------------------------------------------------------------
# Core preprocessing routines
# ---------------------------------------------------------------------------


def collect_day_records(
    trajectories: List[pd.DataFrame],
    cfg: PreprocessConfig,
    origin_lat: float,
    origin_lon: float,
    poi_mapping: Dict[str, int],
    category_mapping: Dict[str, int],
) -> Tuple[List[DayRecord], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    num_pois = len(poi_mapping)
    num_cats = len(category_mapping)

    poi_position_sum = np.zeros((num_pois, 2), dtype=np.float64)
    poi_counts = np.zeros((num_pois,), dtype=np.int64)
    poi_time_hist = np.zeros((num_pois, cfg.poi_time_bins), dtype=np.float64)
    poi_category_idx = np.full((num_pois,), -1, dtype=np.int32)
    category_transition = np.zeros((num_cats, num_cats), dtype=np.float64)
    
    # Weekly temporal density: 30-min slots for one week (2 * 24 * 7 = 336 slots)
    weekly_slot_minutes = 30
    weekly_slots_per_week = int((7 * 24 * 60) / weekly_slot_minutes)  # 336 slots
    poi_weekly_temporal_density = np.zeros((num_pois, weekly_slots_per_week), dtype=np.float64)

    day_records: List[DayRecord] = []

    for traj in trajectories:
        if traj.empty:
            continue

        visits: List[Tuple[float, np.ndarray]] = []
        raw_events: List[RawEvent] = []
        slot_events: Dict[int, List[RawEvent]] = {}

        for _, row in traj.iterrows():
            tau = float(row["time_offset"])
            slot_idx = int(tau // cfg.slot_minutes)
            rel_time = tau - slot_idx * cfg.slot_minutes

            # Use lat/lon directly (no XY conversion)
            lat = float(row[cfg.time_columns["latitude"]])
            lon = float(row[cfg.time_columns["longitude"]])
            latlon = np.array([lat, lon], dtype=np.float32)

            poi_idx = poi_mapping[row[cfg.time_columns["venue_id"]]]
            # Use category name to lookup macro category (if available)
            has_categories = "venue_category_name" in cfg.time_columns and cfg.time_columns["venue_category_name"] in row.index
            if has_categories:
                cat_name = row[cfg.time_columns["venue_category_name"]]
                cat_idx = category_mapping.get(cat_name, category_mapping["UNKNOWN"]) if not pd.isna(cat_name) else category_mapping["UNKNOWN"]
            else:
                cat_idx = category_mapping["UNKNOWN"]

            visits.append((tau, latlon))
            raw_ev = RawEvent(
                time_offset=tau,
                slot_index=slot_idx,
                rel_time=rel_time,
                latlon=latlon,
                poi_idx=poi_idx,
                cat_idx=cat_idx,
            )
            raw_events.append(raw_ev)
            slot_events.setdefault(slot_idx, []).append(raw_ev)

            poi_position_sum[poi_idx] += latlon
            poi_counts[poi_idx] += 1
            bin_idx = int((tau % (24 * 60)) // cfg.time_bin_minutes)
            bin_idx = min(bin_idx, cfg.poi_time_bins - 1)
            poi_time_hist[poi_idx, bin_idx] += 1.0
            if poi_category_idx[poi_idx] == -1:
                poi_category_idx[poi_idx] = cat_idx
            
            # Update weekly temporal density (group into weeks for long datasets)
            weekly_tau = tau % (7 * 24 * 60)  # Map to weekly cycle
            weekly_slot_idx = int(weekly_tau / weekly_slot_minutes)
            weekly_slot_idx = min(weekly_slot_idx, weekly_slots_per_week - 1)
            poi_weekly_temporal_density[poi_idx, weekly_slot_idx] += 1.0

        if not visits:
            continue

        coarse, stay, anchor, counts, slot_c1, slot_cn = build_smt(
            [(tau, latlon) for tau, latlon in visits],
            slot_minutes=cfg.slot_minutes,
            num_slots=cfg.num_slots,
            stay_neighbor_value=cfg.stay_neighbor_value,
        )

        # Convert raw events to EventRecords (no centering needed with lat/lon)
        events: List[EventRecord] = []
        for raw_ev in sorted(raw_events, key=lambda e: e.time_offset):
            events.append(
                EventRecord(
                    time_offset=raw_ev.time_offset,
                    slot_index=raw_ev.slot_index,
                    rel_time=raw_ev.rel_time,
                    latlon=raw_ev.latlon,
                    poi_idx=raw_ev.poi_idx,
                    cat_idx=raw_ev.cat_idx,
                )
            )

        # Compute delta_next for dwell-time features
        for idx, ev in enumerate(events):
            if idx < len(events) - 1:
                ev.delta_next = events[idx + 1].time_offset - ev.time_offset
            else:
                ev.delta_next = cfg.slot_minutes

        # Update category transitions
        for idx in range(1, len(events)):
            prev_cat = events[idx - 1].cat_idx
            curr_cat = events[idx].cat_idx
            category_transition[prev_cat, curr_cat] += 1.0

        # Build SlotRecords with lat/lon coordinates
        slots: List[SlotRecord] = []
        for slot_idx, raw_slot_events in slot_events.items():
            entry_latlon = slot_c1[slot_idx]
            exit_latlon = slot_cn[slot_idx]
            if np.isnan(entry_latlon).any() or np.isnan(exit_latlon).any():
                # Skip slots that have insufficient data (should be rare)
                continue
            slot_event_records = [
                ev for ev in events if ev.slot_index == slot_idx and not np.isnan(ev.latlon).any()
            ]
            if not slot_event_records:
                continue
            slots.append(
                SlotRecord(
                    slot_index=slot_idx,
                    entry_latlon=entry_latlon,
                    exit_latlon=exit_latlon,
                    events=slot_event_records,
                )
            )

        day_records.append(
            DayRecord(
                traj_id=int(traj["traj_id"].iloc[0]),
                coarse=coarse,
                stay=stay,
                anchor=anchor,
                counts=counts,
                slot_c1=slot_c1,
                slot_cn=slot_cn,
                slots=slots,
                events=events,
            )
        )

    # Normalize weekly temporal density to probability distribution (sum to 1.0 across all POIs and time slots)
    total_visits = poi_weekly_temporal_density.sum()
    if total_visits > 0:
        poi_weekly_temporal_density /= total_visits
    
    poi_stats = {
        "position_sum": poi_position_sum,
        "counts": poi_counts,
        "time_hist": poi_time_hist,
        "category_idx": poi_category_idx,
        "weekly_temporal_density": poi_weekly_temporal_density,
    }
    global_stats = {
        "category_transition": category_transition,
    }
    return day_records, poi_stats, global_stats


def compute_latlon_coord_stats(day_records: Sequence[DayRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalization statistics for lat/lon coordinates.
    
    Since coarse coordinates are now direct lat/lon (not anchor-centered),
    we compute the mean and std directly from all trajectory points.
    """
    all_coords = np.concatenate([day.coarse for day in day_records], axis=0)
    mean = all_coords.mean(axis=0).astype(np.float32)
    std = all_coords.std(axis=0).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def compute_global_poi_stats(
    poi_stats: Dict[str, np.ndarray],
    origin_lat: float,
    origin_lon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global normalization statistics from POI coordinates (lat/lon).

    Uses the mean position of each POI and treats all POIs equally.
    """
    counts = poi_stats["counts"]
    position_sum = poi_stats["position_sum"]
    valid = counts > 0
    if not np.any(valid):
        # Fallback to origin-centered stats
        return np.array([origin_lat, origin_lon], dtype=np.float32), np.ones(2, dtype=np.float32)

    # position_sum is already in lat/lon (no XY conversion)
    positions = np.divide(
        position_sum[valid],
        counts[valid, None].astype(np.float32),
        out=np.zeros_like(position_sum[valid], dtype=np.float32),
        where=counts[valid, None] > 0,
    )
    # positions is already in lat/lon
    lat_mean = positions.mean(axis=0).astype(np.float32)
    lat_std = positions.std(axis=0).astype(np.float32)
    lat_std = np.clip(lat_std, 1e-6, None)
    return lat_mean, lat_std


def split_day_indices(num_days: int, cfg: PreprocessConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(num_days, dtype=np.int32)
    rng.shuffle(indices)
    n_train = int(round(cfg.train_ratio * num_days))
    n_val = int(round(cfg.val_ratio * num_days))
    n_train = min(n_train, num_days)
    n_val = min(n_val, num_days - n_train)
    n_test = num_days - n_train - n_val
    if n_train == 0 or n_test == 0:
        raise ValueError("Train/test splits have zero samples; adjust ratios or gather more data.")
    train_ids = indices[:n_train]
    val_ids = indices[n_train : n_train + n_val]
    test_ids = indices[n_train + n_val :]
    return train_ids, val_ids, test_ids


def build_stage1_arrays(
    day_records: Sequence[DayRecord],
    indices: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    lat_mean: np.ndarray,
    lat_std: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Build Stage-1 arrays in simplified format with backward compatibility.
    
    Saves both new and old formats:
    - data: (N, T, 3) = [z_lat, z_lon, stay_prob] where z_lat/z_lon are normalized
      latitude/longitude values using global POI statistics.
    - anchors: (N, 2) - absolute latitude/longitude of the first slot (useful for
      reference, though not needed for denormalization with lat/lon approach).
    - coords: (N, T, 2) = [z_lat, z_lon] for backward compatibility.
    - stay: (N, T) stay probabilities.
    """
    data_list = []
    anchors_list = []
    
    for idx in indices:
        day = day_records[int(idx)]
        # day.coarse is already in lat/lon (no XY conversion)
        latlon = day.coarse.astype(np.float32)
        latlon_norm = (latlon - lat_mean.reshape(1, -1)) / lat_std.reshape(1, -1)
        stay = day.stay  # (T,)
        combined = np.concatenate([latlon_norm, stay[:, None]], axis=1)
        data_list.append(combined[None, :, :])  # (1, T, 3)
        anchors_list.append(np.array([latlon[0]], dtype=np.float32))  # store first slot lat/lon
    
    data_array = np.concatenate(data_list, axis=0).astype(np.float32)  # (N, T, 3)
    anchors_array = np.concatenate(anchors_list, axis=0).astype(np.float32)  # (N, 2)
    
    return {
        # New format (primary)
        "data": data_array,  # (N, T, 3)
        "anchors": anchors_array,  # (N, 2)
        # Old format (backward compatibility)
        "coords": data_array[:, :, :2],  # (N, T, 2) normalized coordinates
        "stay": data_array[:, :, 2],  # (N, T)
    }


def build_global_features(
    coarse_norm: np.ndarray,
    stay: np.ndarray,
    slot_index: int,
    cfg: PreprocessConfig,
) -> np.ndarray:
    prev_idx = max(slot_index - 1, 0)
    next_idx = min(slot_index + 1, coarse_norm.shape[0] - 1)
    mean_xy = coarse_norm.mean(axis=0)
    std_xy = coarse_norm.std(axis=0)
    curr = coarse_norm[slot_index]
    prev = coarse_norm[prev_idx]
    nxt = coarse_norm[next_idx]
    slot_frac = slot_index / (coarse_norm.shape[0] - 1 + 1e-6)
    tod_frac = (slot_index % cfg.slots_per_day) / cfg.slots_per_day
    dow_frac = (slot_index // cfg.slots_per_day) / cfg.duration_days
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
                    math.sin(2 * math.pi * tod_frac),
                    math.cos(2 * math.pi * tod_frac),
                    dow_frac,
                    stay_score,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return features.astype(np.float32)


def build_stage2_arrays(
    day_records: Sequence[DayRecord],
    indices: np.ndarray,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    cfg: PreprocessConfig,
    is_train: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Build Stage-2 arrays using fine-grained windows.

    Each sample uses L fine-grained steps (L = fine_steps = slot_minutes / fine_minutes).
    For 60-minute slots with 1-minute resolution, L = 60.
    
    NOTE: Previous version used 2L windows with padding (L/2 steps before/after for context).
    This was removed for 2x training efficiency - temporal context is already provided
    by the base_path conditioning which interpolates across neighboring slots.

    A binary stay indicator channel (1 for check-ins) is appended so the model 
    learns both coordinates and event occurrences.
    """
    fine_steps = int(cfg.fine_steps)
    half_steps = 0  # EFFICIENCY FIX: Remove padding for 2x speedup
    window_len = fine_steps  # Now just 60 instead of 120

    targets: List[np.ndarray] = []
    base_paths: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    global_cond: List[np.ndarray] = []

    # No coordinate augmentation during preprocessing - this should be done during training
    # to ensure different augmentations per epoch

    for idx in indices:
        day = day_records[int(idx)]
        coarse_latlon = day.coarse  # absolute lat/lon (no normalization yet)
        coarse_norm = (coarse_latlon - coord_mean) / coord_std
        
        # Use original coarse coordinates (no augmentation during preprocessing)
        coarse_norm_aug = coarse_norm
        
        for slot in day.slots:
            if len(slot.events) == 0:
                continue

            slot_idx = slot.slot_index
            prev_idx = max(slot_idx - 1, 0)
            next_idx = min(slot_idx + 1, coarse_latlon.shape[0] - 1)

            # Use augmented coarse coordinates for base path conditioning
            prev_coord_aug = coarse_norm_aug[prev_idx] * coord_std + coord_mean
            next_coord_aug = coarse_norm_aug[next_idx] * coord_std + coord_mean
            
            # Use original (non-augmented) for entry/exit and target
            prev_coord = coarse_latlon[prev_idx]
            curr_entry = slot.entry_latlon
            curr_exit = slot.exit_latlon
            next_coord = coarse_latlon[next_idx]

            # Fine-grained target for the current slot (L, 2)
            fine_path = _interp_path(
                cfg.slot_minutes,
                fine_steps,
                curr_entry,
                curr_exit,
                slot.events,
            )

            # Contextual interpolation toward neighbouring coarse slots
            def _lerp(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
                if steps <= 0:
                    return np.zeros((0, 2), dtype=np.float32)
                alpha = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float32)
                return start[None, :] + alpha[:, None] * (end - start)[None, :]

            prev_path = _lerp(prev_coord, curr_entry, half_steps)
            next_path = _lerp(curr_exit, next_coord, half_steps)

            # Ensure shapes align if fine_steps is odd (pad as needed)
            if prev_path.shape[0] < half_steps:
                prev_path = np.pad(
                    prev_path,
                    ((0, half_steps - prev_path.shape[0]), (0, 0)),
                    mode="edge",
                )
            if next_path.shape[0] < half_steps:
                next_path = np.pad(
                    next_path,
                    ((0, half_steps - next_path.shape[0]), (0, 0)),
                    mode="edge",
                )

            target_ext = np.concatenate([prev_path, fine_path, next_path], axis=0).astype(np.float32)

            # Baseline path: piecewise linear using augmented coarse coordinates for conditioning
            # This helps the model generalize to unseen coarse paths at inference
            base_prev = _lerp(prev_coord_aug, curr_entry, half_steps)
            base_slot = _linear_base_path(curr_entry, curr_exit, fine_steps)
            base_next = _lerp(curr_exit, next_coord_aug, half_steps)
            if base_prev.shape[0] < half_steps:
                base_prev = np.pad(
                    base_prev,
                    ((0, half_steps - base_prev.shape[0]), (0, 0)),
                    mode="edge",
                )
            if base_next.shape[0] < half_steps:
                base_next = np.pad(
                    base_next,
                    ((0, half_steps - base_next.shape[0]), (0, 0)),
                    mode="edge",
                )
            base_ext = np.concatenate([base_prev, base_slot, base_next], axis=0).astype(np.float32)

            # Stay indicator (binary) with ones at check-in minutes in the centre window
            stay_indicator = np.zeros((window_len,), dtype=np.float32)
            for ev in slot.events:
                rel = int(round(ev.rel_time / cfg.fine_minutes))
                rel = max(0, min(fine_steps - 1, rel))
                stay_indicator[half_steps + rel] = 1.0

            mask = np.zeros((window_len,), dtype=np.float32)
            mask[half_steps:half_steps + fine_steps] = 1.0

            # Normalise coordinates and build target/base arrays with indicator channel
            # Target = absolute fine-grained coordinates (not residuals!)
            base_norm = (base_ext - coord_mean) / coord_std
            target_norm = (target_ext - coord_mean) / coord_std

            base_full = np.concatenate(
                [base_norm, np.zeros((window_len, 1), dtype=np.float32)],
                axis=1,
            )
            target_full = np.concatenate(
                [target_norm, stay_indicator[:, None]],
                axis=1,
            )

            targets.append(target_full[None, :, :])
            base_paths.append(base_full[None, :, :])
            masks.append(mask[None, :])
            global_cond.append(
                build_global_features(coarse_norm, day.stay, slot_idx, cfg)[None, :]
            )

    if not targets:
        raise RuntimeError("No stage-2 blocks were produced; adjust preprocessing parameters.")

    return {
        "target": np.concatenate(targets, axis=0),
        "base_path": np.concatenate(base_paths, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "global_cond": np.concatenate(global_cond, axis=0),
    }


def build_stage3_arrays(
    day_records: Sequence[DayRecord],
    indices: np.ndarray,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    poi_positions: np.ndarray,
    poi_counts: np.ndarray,
    poi_time_hist: np.ndarray,
    poi_category_idx: np.ndarray,
    candidate_topk: int,
    cfg: PreprocessConfig,
) -> Dict[str, np.ndarray]:
    event_features = []
    candidate_features = []
    candidate_ids = []
    labels = []
    event_times = []
    slot_indices = []
    day_ids = []
    delta_next = []

    poi_time_prior = (poi_time_hist + 1.0) / (poi_counts[:, None] + cfg.poi_time_bins)
    log_time_prior = np.log(poi_time_prior + 1e-8)
    log_popularity = np.log1p(poi_counts.astype(np.float32))

    for idx in indices:
        day = day_records[int(idx)]
        for ev in day.events:
            # Normalize lat/lon coordinates (no anchor-centering)
            latlon_norm = (ev.latlon - coord_mean) / coord_std
            tod = (ev.time_offset % (24 * 60)) / (24 * 60)
            dow = (ev.time_offset // (24 * 60)) / cfg.duration_days
            event_feat = np.array(
                [
                    latlon_norm[0],
                    latlon_norm[1],
                    math.sin(2 * math.pi * tod),
                    math.cos(2 * math.pi * tod),
                    dow,
                    ev.delta_next / cfg.slot_minutes,
                ],
                dtype=np.float32,
            )
            event_features.append(event_feat[None, :])
            event_times.append(np.array([ev.time_offset], dtype=np.float32))
            slot_indices.append(np.array([ev.slot_index], dtype=np.int32))
            day_ids.append(np.array([day.traj_id], dtype=np.int64))
            delta_next.append(np.array([ev.delta_next], dtype=np.float32))

            # Use lat/lon coordinates for candidate selection
            cand_ids = _select_candidate_ids(ev.latlon, ev.poi_idx, poi_positions, candidate_topk)
            candidate_ids.append(cand_ids[None, :])
            label_idx = np.where(cand_ids == ev.poi_idx)[0]
            if len(label_idx) == 0:
                label = 0
            else:
                label = int(label_idx[0])
            labels.append(np.array([label], dtype=np.int64))

            cand_xy = poi_positions[cand_ids]
            # Compute deltas in lat/lon space (degrees)
            deltas = cand_xy - ev.latlon[None, :]
            # For distance, convert to approximate meters using simple scaling
            # (This is approximate; for better accuracy could use Haversine, but this maintains consistency)
            lat_to_m = 111000.0  # approximate meters per degree latitude
            lon_to_m = 111000.0 * np.cos(np.radians(ev.latlon[0]))  # meters per degree longitude at this latitude
            deltas_m = deltas * np.array([lat_to_m, lon_to_m])
            dist = np.linalg.norm(deltas_m, axis=1)
            bearing = np.arctan2(deltas[:, 1], deltas[:, 0])
            time_bin = int((ev.time_offset % (24 * 60)) // cfg.time_bin_minutes)
            time_bin = min(time_bin, cfg.poi_time_bins - 1)
            time_prior = log_time_prior[cand_ids, time_bin]
            pop_log = log_popularity[cand_ids]
            cat_match = (cand_ids * 0)  # placeholder overwritten below
            # `poi_category_idx` needed; we reconstruct using event.cat_idx? can't easily here.
            # We pass dummy zeros; actual match computed later using metadata.

            # Compute heuristic STC score
            stc_score = -dist + 0.5 * time_prior + 0.2 * pop_log

            cat_idx_candidates = poi_category_idx[cand_ids]
            cat_match = (cat_idx_candidates == ev.cat_idx).astype(np.float32)

            cand_feat = np.stack(
                [
                    dist / 1000.0,
                    np.sin(bearing),
                    np.cos(bearing),
                    cat_match,
                    pop_log,
                    time_prior,
                    stc_score,
                ],
                axis=1,
            ).astype(np.float32)
            candidate_features.append(cand_feat[None, :, :])

    if not event_features:
        raise RuntimeError("No Stage-3 events were produced; adjust preprocessing parameters.")

    return {
        "event_features": np.concatenate(event_features, axis=0),
        "candidate_features": np.concatenate(candidate_features, axis=0),
        "candidate_ids": np.concatenate(candidate_ids, axis=0),
        "labels": np.concatenate(labels, axis=0).reshape(-1),
        "event_time": np.concatenate(event_times, axis=0).reshape(-1),
        "slot_index": np.concatenate(slot_indices, axis=0).reshape(-1),
        "day_id": np.concatenate(day_ids, axis=0).reshape(-1),
        "delta_next": np.concatenate(delta_next, axis=0).reshape(-1),
    }


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# Metadata + catalog writing
# ---------------------------------------------------------------------------


def write_metadata(
    cfg: PreprocessConfig,
    latlon_mean: np.ndarray,
    latlon_std: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    split_sizes: Dict[str, Dict[str, int]],
    poi_catalog_path: Path,
    poi_mapping: Dict[str, int],
    category_mapping: Dict[str, int],
    global_stats: Dict[str, np.ndarray],
) -> None:
    metadata = {
        "city": cfg.city,
        "slot_minutes": cfg.slot_minutes,
        "fine_minutes": cfg.fine_minutes,
        "fine_steps": cfg.fine_steps,
        "duration_days": cfg.duration_days,
        "duration_minutes": cfg.duration_minutes,
        "num_slots": cfg.num_slots,
        "slots_per_day": cfg.slots_per_day,
        "candidate_topk": cfg.candidate_topk,
        "poi_time_bins": cfg.poi_time_bins,
        "normalization": {
            "mean": latlon_mean.tolist(),
            "std": latlon_std.tolist(),
        },
        "origin": {"lat": float(origin_lat), "lon": float(origin_lon)},
        "split_sizes": split_sizes,
        "poi_catalog": str(poi_catalog_path.name),
        "poi_mapping_size": len(poi_mapping),
        "category_mapping_size": len(category_mapping),
        "category_transition_shape": list(global_stats["category_transition"].shape),
    }
    metadata_path = cfg.output_dir / cfg.metadata_file
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def write_poi_catalog(
    cfg: PreprocessConfig,
    poi_mapping: Dict[str, int],
    category_mapping: Dict[str, int],
    poi_stats: Dict[str, np.ndarray],
    global_stats: Dict[str, np.ndarray],
    origin_lat: float,
    origin_lon: float,
) -> Path:
    num_pois = len(poi_mapping)
    poi_ids = [""] * num_pois
    for key, idx in poi_mapping.items():
        poi_ids[idx] = key

    cat_ids = [""] * len(category_mapping)
    for key, idx in category_mapping.items():
        cat_ids[idx] = key

    counts = poi_stats["counts"]
    position_sum = poi_stats["position_sum"]
    position_mean = np.divide(
        position_sum,
        counts[:, None] + 1e-6,
        out=np.zeros_like(position_sum),
        where=(counts[:, None] > 0),
    )
    # position_mean is already in lat/lon (no XY conversion)
    latlon = position_mean.astype(np.float32)
    time_hist = poi_stats["time_hist"].astype(np.float32)
    category_idx = poi_stats["category_idx"].astype(np.int32)
    weekly_temporal_density = poi_stats["weekly_temporal_density"].astype(np.float32)

    catalog_path = cfg.output_dir / cfg.poi_catalog_file
    np.savez_compressed(
        catalog_path,
        poi_ids=np.array(poi_ids, dtype=object),
        category_ids=np.array(cat_ids, dtype=object),
        poi_latlon=latlon,  # Store lat/lon directly
        poi_counts=counts.astype(np.int32),
        poi_time_hist=time_hist,
        poi_category_idx=category_idx,
        category_transition=global_stats["category_transition"].astype(np.float32),
        poi_weekly_temporal_density=weekly_temporal_density,  # (N_poi, 336) normalized probability
    )
    return catalog_path


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_preprocess_config(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading raw data from {cfg.raw_path}")
    df = pd.read_csv(cfg.raw_path)
    trajectories = slice_trajectories(df, cfg)
    if not trajectories:
        raise RuntimeError("No trajectories were extracted; adjust date range or thresholds.")
    print(f"[INFO] Extracted {len(trajectories)} trajectories.")

    origin_lat = cfg.reference_lat if cfg.reference_lat is not None else float(df[cfg.time_columns["latitude"]].mean())
    origin_lon = cfg.reference_lon if cfg.reference_lon is not None else float(df[cfg.time_columns["longitude"]].mean())

    poi_mapping, category_mapping = build_mappings(trajectories, cfg)
    print(f"[INFO] Mapped {len(poi_mapping)} POIs and {len(category_mapping)} categories.")

    day_records, poi_stats, global_stats = collect_day_records(
        trajectories,
        cfg,
        origin_lat,
        origin_lon,
        poi_mapping,
        category_mapping,
    )
    if not day_records:
        raise RuntimeError("No day records were produced; check preprocessing assumptions.")
    print(f"[INFO] Built day records: {len(day_records)} sequences.")

    # Compute lat/lon normalization statistics
    latlon_mean, latlon_std = compute_latlon_coord_stats(day_records)
    poi_positions = np.divide(
        poi_stats["position_sum"],
        poi_stats["counts"][:, None] + 1e-6,
        out=np.zeros_like(poi_stats["position_sum"]),
        where=(poi_stats["counts"][:, None] > 0),
    )
    # Also compute from POI stats for comparison (should be similar)
    poi_latlon_mean, poi_latlon_std = compute_global_poi_stats(
        poi_stats,
        origin_lat,
        origin_lon,
    )
    print(f"[INFO] Trajectory lat/lon stats: mean={latlon_mean}, std={latlon_std}")
    print(f"[INFO] POI lat/lon stats: mean={poi_latlon_mean}, std={poi_latlon_std}")
    
    # Use trajectory-based stats (more representative of actual data distribution)
    stage1_mean, stage1_std = latlon_mean, latlon_std
    
    train_ids, val_ids, test_ids = split_day_indices(len(day_records), cfg)

    # Merge validation split into training set as requested
    if len(val_ids):
        train_ids = np.concatenate([train_ids, val_ids])
        val_ids = np.zeros((0,), dtype=np.int32)

    print(f"[INFO] Splits - train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    stage1_train = build_stage1_arrays(day_records, train_ids, origin_lat, origin_lon, stage1_mean, stage1_std)
    stage1_val = (
        build_stage1_arrays(day_records, val_ids, origin_lat, origin_lon, stage1_mean, stage1_std)
        if len(val_ids)
        else None
    )
    stage1_test = build_stage1_arrays(day_records, test_ids, origin_lat, origin_lon, stage1_mean, stage1_std)

    stage2_train = build_stage2_arrays(day_records, train_ids, stage1_mean, stage1_std, cfg, is_train=True)
    stage2_val = build_stage2_arrays(day_records, val_ids, stage1_mean, stage1_std, cfg, is_train=False) if len(val_ids) else None
    stage2_test = build_stage2_arrays(day_records, test_ids, stage1_mean, stage1_std, cfg, is_train=False)

    stage3_train = build_stage3_arrays(
        day_records,
        train_ids,
        stage1_mean,
        stage1_std,
        poi_positions,
        poi_stats["counts"],
        poi_stats["time_hist"],
        poi_stats["category_idx"],
        cfg.candidate_topk,
        cfg,
    )
    stage3_val = (
        build_stage3_arrays(
            day_records,
            val_ids,
            stage1_mean,
            stage1_std,
            poi_positions,
            poi_stats["counts"],
            poi_stats["time_hist"],
            poi_stats["category_idx"],
            cfg.candidate_topk,
            cfg,
        )
        if len(val_ids)
        else None
    )
    stage3_test = build_stage3_arrays(
        day_records,
        test_ids,
        stage1_mean,
        stage1_std,
        poi_positions,
        poi_stats["counts"],
        poi_stats["time_hist"],
        poi_stats["category_idx"],
        cfg.candidate_topk,
        cfg,
    )

    # Save NPZ files
    s1_train_path = save_npz(cfg.output_dir / cfg.stage1_files.train, stage1_train)
    s1_val_path = save_npz(cfg.output_dir / cfg.stage1_files.val, stage1_val) if stage1_val else None
    s1_test_path = save_npz(cfg.output_dir / cfg.stage1_files.test, stage1_test)

    s2_train_path = save_npz(cfg.output_dir / cfg.stage2_files.train, stage2_train)
    s2_val_path = save_npz(cfg.output_dir / cfg.stage2_files.val, stage2_val) if stage2_val else None
    s2_test_path = save_npz(cfg.output_dir / cfg.stage2_files.test, stage2_test)

    s3_train_path = save_npz(cfg.output_dir / cfg.stage3_files.train, stage3_train)
    s3_val_path = save_npz(cfg.output_dir / cfg.stage3_files.val, stage3_val) if stage3_val else None
    s3_test_path = save_npz(cfg.output_dir / cfg.stage3_files.test, stage3_test)

    poi_catalog_path = write_poi_catalog(
        cfg,
        poi_mapping,
        category_mapping,
        poi_stats,
        global_stats,
        origin_lat,
        origin_lon,
    )

    split_sizes = {
        "stage1": {
            "train": int(stage1_train["data"].shape[0]),
            "val": int(stage1_val["data"].shape[0]) if stage1_val else 0,
            "test": int(stage1_test["data"].shape[0]),
        },
        "stage2": {
            "train": int(stage2_train["target"].shape[0]),
            "val": int(stage2_val["target"].shape[0]) if stage2_val else 0,
            "test": int(stage2_test["target"].shape[0]),
        },
        "stage3": {
            "train": int(stage3_train["event_features"].shape[0]),
            "val": int(stage3_val["event_features"].shape[0]) if stage3_val else 0,
            "test": int(stage3_test["event_features"].shape[0]),
        },
    }
    write_metadata(
        cfg,
        stage1_mean,
        stage1_std,
        origin_lat,
        origin_lon,
        split_sizes,
        poi_catalog_path,
        poi_mapping,
        category_mapping,
        global_stats,
    )

    print("[INFO] Preprocessing completed.")
    print(f"  Stage-1 train saved to: {s1_train_path}")
    if s1_val_path:
        print(f"  Stage-1 val saved to:   {s1_val_path}")
    print(f"  Stage-1 test saved to:  {s1_test_path}")
    print(f"  Stage-2 train saved to: {s2_train_path}")
    if s2_val_path:
        print(f"  Stage-2 val saved to:   {s2_val_path}")
    print(f"  Stage-2 test saved to:  {s2_test_path}")
    print(f"  Stage-3 train saved to: {s3_train_path}")
    if s3_val_path:
        print(f"  Stage-3 val saved to:   {s3_val_path}")
    print(f"  Stage-3 test saved to:  {s3_test_path}")
    print(f"  POI catalog saved to:   {poi_catalog_path}")
    print(f"  Metadata saved to:      {cfg.output_dir / cfg.metadata_file}")


if __name__ == "__main__":
    main()
