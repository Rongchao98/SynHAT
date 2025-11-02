#!/usr/bin/env python3
"""Stage-3 inference: assign POIs to Stage-2 events via spatial filtering."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.general_utils import load_config_from_yaml


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres."""
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


class QuadTreeNode:
    """Minimal quadtree for spatial lookups."""

    def __init__(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        max_items: int = 32,
        max_depth: int = 8,
        depth: int = 0,
    ) -> None:
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.max_items = max_items
        self.max_depth = max_depth
        self.depth = depth
        self.points: List[tuple[int, float, float]] = []
        self.children: Optional[List[QuadTreeNode]] = None

    def insert(self, poi_idx: int, lat: float, lon: float) -> None:
        if not (self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon):
            return
        if self.children is not None:
            self.children[self._child_index(lat, lon)].insert(poi_idx, lat, lon)
            return
        self.points.append((poi_idx, lat, lon))
        if len(self.points) > self.max_items and self.depth < self.max_depth:
            self._subdivide()

    def _subdivide(self) -> None:
        mid_lat = (self.min_lat + self.max_lat) / 2.0
        mid_lon = (self.min_lon + self.max_lon) / 2.0
        self.children = [
            QuadTreeNode(self.min_lat, mid_lat, self.min_lon, mid_lon, self.max_items, self.max_depth, self.depth + 1),
            QuadTreeNode(self.min_lat, mid_lat, mid_lon, self.max_lon, self.max_items, self.max_depth, self.depth + 1),
            QuadTreeNode(mid_lat, self.max_lat, self.min_lon, mid_lon, self.max_items, self.max_depth, self.depth + 1),
            QuadTreeNode(mid_lat, self.max_lat, mid_lon, self.max_lon, self.max_items, self.max_depth, self.depth + 1),
        ]
        for poi_idx, lat, lon in self.points:
            self.children[self._child_index(lat, lon)].insert(poi_idx, lat, lon)
        self.points = []

    def _child_index(self, lat: float, lon: float) -> int:
        mid_lat = (self.min_lat + self.max_lat) / 2.0
        mid_lon = (self.min_lon + self.max_lon) / 2.0
        southern = lat < mid_lat
        western = lon < mid_lon
        return (0 if southern else 2) + (0 if western else 1)

    def query_radius(self, lat: float, lon: float, radius_km: float, poi_positions: np.ndarray) -> List[int]:
        lat_delta = radius_km / 111.0
        denom = max(math.cos(math.radians(lat)), 1e-6)
        lon_delta = radius_km / (111.0 * denom)
        if lat + lat_delta < self.min_lat or lat - lat_delta > self.max_lat:
            return []
        if lon + lon_delta < self.min_lon or lon - lon_delta > self.max_lon:
            return []
        if self.children is not None:
            hits: List[int] = []
            for child in self.children:
                hits.extend(child.query_radius(lat, lon, radius_km, poi_positions))
            return hits
        hits = []
        for poi_idx, poi_lat, poi_lon in self.points:
            if haversine(lat, lon, poi_lat, poi_lon) <= radius_km:
                hits.append(poi_idx)
        return hits


def build_quadtree(poi_positions: np.ndarray) -> QuadTreeNode:
    lat_min = float(np.min(poi_positions[:, 0]))
    lat_max = float(np.max(poi_positions[:, 0]))
    lon_min = float(np.min(poi_positions[:, 1]))
    lon_max = float(np.max(poi_positions[:, 1]))
    padding = 1e-3
    root = QuadTreeNode(lat_min - padding, lat_max + padding, lon_min - padding, lon_max + padding)
    for idx in range(poi_positions.shape[0]):
        root.insert(idx, float(poi_positions[idx, 0]), float(poi_positions[idx, 1]))
    return root


def probabilistic_poi_selection(
    event_xy_abs: np.ndarray,
    event_time: np.ndarray,
    poi_positions: np.ndarray,
    poi_weekly_temporal_density: np.ndarray,
    quadtree: QuadTreeNode,
    spatial_radius_km: float,
    weekly_slot_minutes: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    weekly_slots = max(1, int(round((7 * 24 * 60) / weekly_slot_minutes)))
    selected = np.zeros(event_xy_abs.shape[0], dtype=np.int32)
    for idx, (lat, lon) in enumerate(event_xy_abs):
        candidates = quadtree.query_radius(float(lat), float(lon), spatial_radius_km, poi_positions)
        if not candidates:
            dists = np.linalg.norm(poi_positions - event_xy_abs[idx], axis=1)
            selected[idx] = int(np.argmin(dists))
            continue
        slot = int((event_time[idx] % (7 * 24 * 60)) / weekly_slot_minutes)
        slot = max(0, min(slot, weekly_slots - 1))
        candidate_ids = np.asarray(candidates, dtype=np.int32)
        weights = poi_weekly_temporal_density[candidate_ids, slot]
        total = float(weights.sum())
        if total <= 1e-9:
            probs = np.full(candidate_ids.shape[0], 1.0 / candidate_ids.shape[0], dtype=np.float32)
        else:
            probs = (weights / total).astype(np.float32)
        selected[idx] = int(candidate_ids[rng.choice(candidate_ids.shape[0], p=probs)])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 POI recovery")
    parser.add_argument("--config", default="config/sample_synhat.yaml", type=str, help="Config YAML")
    parser.add_argument(
        "--stage2-events",
        default="data/sample_synhat/stage2_events_sample.npz",
        type=str,
        help="Stage-2 events NPZ",
    )
    parser.add_argument("--spatial-radius", type=float, help="Override spatial search radius in km")
    parser.add_argument("--output-dir", type=str, default="stage3_poi", help="Output directory")
    return parser.parse_args()


def load_metadata(config) -> Dict:
    meta_path = Path(config.data.processed_dir) / config.data.stage_files.metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_poi_catalog(config) -> Dict[str, np.ndarray]:
    catalog_path = Path(config.data.processed_dir) / config.data.stage_files.poi_catalog
    with np.load(catalog_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def group_events_by_day(day_ids: np.ndarray, event_time: np.ndarray, poi_ids: np.ndarray, selected: np.ndarray) -> List[Dict]:
    output: List[Dict] = []
    current_day: Optional[int] = None
    events: List[Dict] = []
    for idx, day in enumerate(day_ids):
        day = int(day)
        if current_day is None:
            current_day = day
        elif day != current_day:
            output.append({"day_id": current_day, "events": events})
            events = []
            current_day = day
        events.append({"poi_id": str(poi_ids[selected[idx]]), "time_offset": float(event_time[idx])})
    if current_day is not None:
        output.append({"day_id": current_day, "events": events})
    return output


def main() -> None:
    args = parse_args()
    config = load_config_from_yaml(args.config)
    metadata = load_metadata(config)
    catalog = load_poi_catalog(config)

    events = np.load(args.stage2_events, allow_pickle=False)
    event_xy_abs = events["event_xy_abs"].astype(np.float32)
    event_time = events["event_time"].astype(np.float32)
    day_id = events["day_id"].astype(np.int32)

    poi_positions = catalog["poi_latlon"].astype(np.float32)
    poi_ids = catalog["poi_ids"]
    if "poi_weekly_temporal_density" not in catalog:
        raise KeyError("poi_weekly_temporal_density missing from POI catalog")
    poi_weekly_temporal_density = catalog["poi_weekly_temporal_density"].astype(np.float32)

    default_radius = 2.0
    if getattr(getattr(config, "inference", None), "stage3", None) is not None:
        default_radius = float(getattr(config.inference.stage3, "spatial_radius_km", default_radius))
    spatial_radius_km = float(args.spatial_radius) if args.spatial_radius is not None else default_radius

    quadtree = build_quadtree(poi_positions)
    selected = probabilistic_poi_selection(
        event_xy_abs=event_xy_abs,
        event_time=event_time,
        poi_positions=poi_positions,
        poi_weekly_temporal_density=poi_weekly_temporal_density,
        quadtree=quadtree,
        spatial_radius_km=spatial_radius_km,
    )

    day_trajectories = group_events_by_day(day_id, event_time, poi_ids, selected)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "greedy_checkins.json", "w", encoding="utf-8") as f:
        json.dump(day_trajectories, f, indent=2)
    summary = {
        "num_events": int(event_xy_abs.shape[0]),
        "num_days": int(len(day_trajectories)),
        "spatial_radius_km": spatial_radius_km,
        "stage2_events": str(args.stage2_events),
        "poi_count": int(poi_positions.shape[0]),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Stage-3 inference complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
