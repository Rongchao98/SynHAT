#!/usr/bin/env python3
"""Generate a tiny synthetic dataset for SynHAT demos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_stage1(root: Path, rng: np.random.Generator, num_slots: int) -> None:
    def save_split(name: str, samples: int) -> None:
        lat = rng.normal(0.0, 0.7, size=(samples, num_slots)).astype(np.float32)
        lon = rng.normal(0.0, 0.7, size=(samples, num_slots)).astype(np.float32)
        stay = np.clip(rng.normal(0.6, 0.2, size=(samples, num_slots)), 0.0, 1.0).astype(np.float32)
        data = np.stack([lat, lon, stay], axis=-1)
        np.savez_compressed(root / f"stage1_{name}.npz", data=data)

    save_split("train", 12)
    save_split("val", 4)
    save_split("test", 4)


def write_stage2(root: Path, rng: np.random.Generator, window_len: int) -> None:
    def save_split(name: str, samples: int) -> None:
        target = rng.normal(0.0, 0.5, size=(samples, window_len, 3)).astype(np.float32)
        base = target + rng.normal(0.0, 0.1, size=(samples, window_len, 3)).astype(np.float32)
        indicator = rng.uniform(0.0, 1.0, size=(samples, window_len)).astype(np.float32)
        target[..., 2] = indicator
        base[..., 2] = indicator * 0.8
        global_cond = rng.normal(0.0, 1.0, size=(samples, 16)).astype(np.float32)
        mask = np.ones((samples, window_len), dtype=np.float32)
        np.savez_compressed(
            root / f"stage2_{name}.npz",
            target=target,
            base_path=base,
            global_cond=global_cond,
            mask=mask,
        )

    save_split("train", 16)
    save_split("val", 4)
    save_split("test", 4)


def write_poi_catalog(root: Path, rng: np.random.Generator) -> None:
    poi_count = 12
    latlon = np.column_stack([
        40.0 + rng.normal(0.0, 0.02, size=poi_count),
        -73.0 + rng.normal(0.0, 0.03, size=poi_count),
    ]).astype(np.float32)
    poi_ids = np.array([f"POI_{i:03d}" for i in range(poi_count)], dtype="<U16")
    weekly = rng.random((poi_count, 336)).astype(np.float32)
    weekly /= weekly.sum(axis=1, keepdims=True) + 1e-6
    np.savez_compressed(
        root / "poi_catalog.npz",
        poi_latlon=latlon,
        poi_ids=poi_ids,
        poi_weekly_temporal_density=weekly,
    )


def write_stage2_events(root: Path, rng: np.random.Generator, num_slots: int, slot_minutes: int) -> None:
    events_per_day = 5
    coords = np.column_stack([
        40.0 + rng.normal(0.0, 0.02, size=events_per_day),
        -73.0 + rng.normal(0.0, 0.03, size=events_per_day),
    ]).astype(np.float32)
    times = np.sort(rng.integers(0, num_slots, size=events_per_day)) * slot_minutes
    day_ids = np.zeros(events_per_day, dtype=np.int32)
    np.savez_compressed(
        root / "stage2_events_sample.npz",
        event_xy_abs=coords,
        event_time=times.astype(np.float32),
        day_id=day_ids,
    )


def write_metadata(root: Path) -> None:
    metadata = {
        "city": "demo",
        "slot_minutes": 60,
        "fine_minutes": 15,
        "fine_steps": 4,
        "duration_days": 1,
        "num_slots": 24,
        "slots_per_day": 24,
        "normalization": {"mean": [40.0, -73.0], "std": [0.02, 0.02]},
        "split_sizes": {
            "stage1": {"train": 12, "val": 4, "test": 4},
            "stage2": {"train": 16, "val": 4, "test": 4},
        },
        "origin": {"lat": 40.0, "lon": -73.0},
    }
    (root / "stage_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "sample_synhat"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    num_slots = 24
    slot_minutes = 60
    window_len = 32

    write_stage1(root, rng, num_slots)
    write_stage2(root, rng, window_len)
    write_poi_catalog(root, rng)
    write_stage2_events(root, rng, num_slots, slot_minutes)
    write_metadata(root)
    print(f"Sample dataset written to {root}")


if __name__ == "__main__":
    main()
