import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def latlon_to_local_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude pairs to local tangent plane (meters) using an
    equirectangular approximation.

    Args:
        lat: Array of latitudes in degrees.
        lon: Array of longitudes in degrees.
        origin_lat: Reference latitude (degrees).
        origin_lon: Reference longitude (degrees).

    Returns:
        Tuple (x, y) representing offsets in meters relative to the origin.
    """
    # Earth radius in meters
    r_earth = 6_371_000.0

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)

    x = (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad) * r_earth
    y = (lat_rad - origin_lat_rad) * r_earth
    return x.astype(np.float32), y.astype(np.float32)


def local_xy_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform from local tangent plane (meters) back to latitude/longitude.
    """
    r_earth = 6_371_000.0
    origin_lat_rad = math.radians(origin_lat)

    lat = y / r_earth + origin_lat_rad
    lon = x / (r_earth * math.cos(origin_lat_rad)) + math.radians(origin_lon)

    return np.degrees(lat).astype(np.float32), np.degrees(lon).astype(np.float32)


def build_smt(
    visits: Sequence[Tuple[float, Sequence[float]]],
    slot_minutes: int,
    num_slots: int,
    stay_neighbor_value: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build Synchronous Movement Trajectory (SMT) representation for a sequence
    of asynchronous visits.

    Args:
        visits: Iterable of (timestamp_minutes, (x, y)) sorted by timestamp.
        slot_minutes: Slot duration in minutes (Δ).
        num_slots: Total number of slots (T).
        stay_neighbor_value: Target probability assigned to slots at distance 1 from a check-in.

    Returns:
        Tuple (X, stay, anchor, visit_counts, slot_c1, slot_cn) or None if
        the trajectory has no valid visits within the horizon.
            X: (T, 2) absolute coordinates per slot (float32).
            stay: (T,) stay score per slot (float32).
            anchor: (2,) starting anchor (float32).
            visit_counts: (T,) integer visit counts per slot (int16).
            slot_c1: (T, 2) first visit coordinates per slot (float32, NaN if empty).
            slot_cn: (T, 2) last visit coordinates per slot (float32, NaN if empty).
    """
    if len(visits) == 0:
        return None

    X = np.full((num_slots, 2), np.nan, dtype=np.float32)
    stay = np.zeros((num_slots,), dtype=np.float32)
    counts = np.zeros((num_slots,), dtype=np.int16)
    slot_c1 = np.full((num_slots, 2), np.nan, dtype=np.float32)
    slot_cn = np.full((num_slots, 2), np.nan, dtype=np.float32)

    by_slot: List[List[Tuple[float, Sequence[float]]]] = [[] for _ in range(num_slots)]

    def slot_index(tau_min: float) -> int:
        return int(tau_min // slot_minutes)

    for tau, xy in visits:
        idx = slot_index(tau)
        if 0 <= idx < num_slots and np.isfinite(xy[0]) and np.isfinite(xy[1]):
            by_slot[idx].append((tau, xy))
            counts[idx] += 1

    if not any(by_slot):
        return None

    exit_anchor: List[Optional[np.ndarray]] = [None] * num_slots
    hit_slots: List[int] = []
    for slot_idx, slot_visits in enumerate(by_slot):
        if not slot_visits:
            continue
        slot_visits.sort(key=lambda pair: pair[0])
        first_xy = np.asarray(slot_visits[0][1], dtype=np.float32)
        last_xy = np.asarray(slot_visits[-1][1], dtype=np.float32)
        X[slot_idx] = first_xy
        slot_c1[slot_idx] = first_xy
        slot_cn[slot_idx] = last_xy
        exit_anchor[slot_idx] = last_xy
        hit_slots.append(slot_idx)

    slot = 0
    while slot < num_slots:
        if np.isnan(X[slot, 0]):
            slot += 1
            continue

        next_slot = slot + 1
        while next_slot < num_slots and np.isnan(X[next_slot, 0]):
            next_slot += 1

        if next_slot < num_slots:
            start = exit_anchor[slot] if exit_anchor[slot] is not None else X[slot]
            # Next slot has at least one visit by construction.
            next_first_xy = np.asarray(by_slot[next_slot][0][1], dtype=np.float32)
            span = next_slot - slot
            for gap_idx in range(1, span):
                ratio = gap_idx / span
                X[slot + gap_idx] = (1 - ratio) * start + ratio * next_first_xy
        else:
            # Extend tail using last known coordinate.
            for fill_idx in range(slot + 1, num_slots):
                X[fill_idx] = X[slot]
            break

        slot = next_slot

    # Back-fill leading NaNs if the first slots had no visits.
    first_valid = np.argmax(~np.isnan(X[:, 0]))
    if np.isnan(X[first_valid, 0]):
        # The whole trajectory could not be filled (should not happen if any slot had visits).
        return None

    for lead_idx in range(0, first_valid):
        X[lead_idx] = X[first_valid]

    # Store anchor for reference, but DON'T subtract it (keep absolute lat/lon)
    anchor = X[0].copy()
    
    # For slot entry/exit coordinates, keep them as absolute (no centering)
    slot_c1_absolute = slot_c1.copy()
    slot_cn_absolute = slot_cn.copy()

    # X remains in absolute coordinates (no anchor subtraction)
    # This maintains lat/lon values in degrees throughout

    if hit_slots:
        stay[:] = _compute_soft_stay_scores(
            num_slots,
            hit_slots,
            stay_neighbor_value,
        )

    return (
        X.astype(np.float32),
        stay,
        anchor.astype(np.float32),  # Still return anchor for compatibility
        counts,
        slot_c1_absolute.astype(np.float32),  # Entry coordinates (absolute)
        slot_cn_absolute.astype(np.float32),  # Exit coordinates (absolute)
    )


def _compute_soft_stay_scores(
    num_slots: int,
    hit_slots: Sequence[int],
    neighbor_value: float,
    floor: float = 0.0,
) -> np.ndarray:
    """
    Compute smoothed stay scores by applying a Gaussian kernel over slot distance.

    Args:
        num_slots: Total number of slots.
        hit_slots: Indices with actual check-ins.
        neighbor_value: Desired probability at distance 1 (interpreted as alpha).
        floor: Minimum value assigned to any slot.

    Returns:
        Soft stay scores of shape (num_slots,) in [0, 1].
    """
    if not hit_slots:
        return np.zeros(num_slots, dtype=np.float32)

    # Interpret neighbor_value as target probability at distance 1.
    alpha = float(np.clip(neighbor_value, 1e-3, 0.999))
    sigma = math.sqrt(1.0 / (2.0 * math.log(1.0 / alpha)))
    indices = np.arange(num_slots, dtype=np.float32)
    stay_scores = np.full(num_slots, floor, dtype=np.float32)

    for slot in hit_slots:
        dist = np.abs(indices - slot)
        kernel = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        np.maximum(stay_scores, kernel.astype(np.float32), out=stay_scores)

    np.clip(stay_scores, 0.0, 1.0, out=stay_scores)
    if floor > 0.0:
        stay_scores = np.maximum(stay_scores, floor)
    return stay_scores


def compute_sequence_time_metadata(
    num_slots: int,
    slots_per_day: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time-of-day and day-of-week indices for each slot.

    Args:
        num_slots: Total number of slots in the sequence.
        slots_per_day: Number of slots per day (e.g., 24 for 60-minute slots).

    Returns:
        (tod_indices, dow_indices) each of shape (num_slots,) and dtype np.int64.
    """
    indices = np.arange(num_slots, dtype=np.int64)
    tod = indices % slots_per_day
    dow = indices // slots_per_day
    return tod, dow
