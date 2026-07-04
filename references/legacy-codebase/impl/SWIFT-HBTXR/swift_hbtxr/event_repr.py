from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_SENSOR_SIZE_WH = (346, 240)


def _normalize_timestamps(timestamps: np.ndarray, start_timestamp_us: int, end_timestamp_us: int, n_time_bins: int) -> np.ndarray:
    if end_timestamp_us <= start_timestamp_us:
        return np.zeros_like(timestamps, dtype=np.float32)
    return n_time_bins * (timestamps.astype(np.float32) - float(start_timestamp_us)) / float(end_timestamp_us - start_timestamp_us)


def _bilinear_bins(positions: np.ndarray, max_bin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weight_next = positions % 1.0
    weight_now = 1.0 - weight_next
    current = np.floor(positions).clip(0, max_bin).astype(np.int64)
    nxt = (current + 1).clip(0, max_bin).astype(np.int64)
    return current, nxt, weight_now.astype(np.float32), weight_next.astype(np.float32)


def to_facet_event_frame(
    events: np.ndarray,
    *,
    sensor_size_wh: tuple[int, int] = DEFAULT_SENSOR_SIZE_WH,
    n_time_bins: int = 1,
    mode: str = "causal_linear",
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    weight: float = 1.0,
    polarity_split: bool = True,
) -> np.ndarray:
    if start_timestamp_us is None:
        start_timestamp_us = int(events["t"][0])
    if end_timestamp_us is None:
        end_timestamp_us = int(events["t"][-1])
    positions = _normalize_timestamps(events["t"], start_timestamp_us, end_timestamp_us, n_time_bins)
    t_now, _, weight_now, weight_next = _bilinear_bins(positions, n_time_bins)
    channels = 2 if polarity_split else 1
    frame_stack = np.zeros((n_time_bins, channels, sensor_size_wh[1], sensor_size_wh[0]), dtype=np.float32)
    xs = events["x"].astype(np.int64)
    ys = events["y"].astype(np.int64)
    ps = events["p"].astype(np.int64)
    valid = (xs >= 0) & (xs < sensor_size_wh[0]) & (ys >= 0) & (ys < sensor_size_wh[1]) & (t_now >= 0) & (t_now < n_time_bins)
    if not np.any(valid):
        return frame_stack

    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]
    t_now = t_now[valid]
    weight_now = weight_now[valid]
    weight_next = weight_next[valid]

    if polarity_split:
        neg_mask = ps <= 0
        pos_mask = ~neg_mask
        if mode == "nearest":
            channel = np.where(ps <= 0, 0, 1)
            np.add.at(frame_stack, (t_now, channel, ys, xs), float(weight))
        elif mode == "bilinear":
            prev = np.clip(t_now - 1, 0, n_time_bins - 1)
            np.add.at(frame_stack, (t_now[neg_mask], np.zeros(np.count_nonzero(neg_mask), dtype=np.int64), ys[neg_mask], xs[neg_mask]), weight_next[neg_mask] * float(weight))
            np.add.at(frame_stack, (t_now[pos_mask], np.ones(np.count_nonzero(pos_mask), dtype=np.int64), ys[pos_mask], xs[pos_mask]), weight_next[pos_mask] * float(weight))
            np.add.at(frame_stack, (prev[neg_mask], np.zeros(np.count_nonzero(neg_mask), dtype=np.int64), ys[neg_mask], xs[neg_mask]), weight_now[neg_mask] * float(weight))
            np.add.at(frame_stack, (prev[pos_mask], np.ones(np.count_nonzero(pos_mask), dtype=np.int64), ys[pos_mask], xs[pos_mask]), weight_now[pos_mask] * float(weight))
        elif mode == "causal_linear":
            np.add.at(frame_stack, (t_now[neg_mask], np.zeros(np.count_nonzero(neg_mask), dtype=np.int64), ys[neg_mask], xs[neg_mask]), weight_next[neg_mask] * float(weight))
            np.add.at(frame_stack, (t_now[pos_mask], np.ones(np.count_nonzero(pos_mask), dtype=np.int64), ys[pos_mask], xs[pos_mask]), weight_next[pos_mask] * float(weight))
        else:
            raise ValueError(f"Unsupported FACET mode: {mode}")
        return frame_stack

    signed = np.where(ps <= 0, -1.0, 1.0).astype(np.float32)
    np.add.at(frame_stack, (t_now, np.zeros_like(t_now), ys, xs), signed * weight_next * float(weight))
    return frame_stack


def _event_indices_for_window(
    timestamps_us: np.ndarray,
    *,
    end_timestamp_us: int,
    policy: str,
    time_bin_us: int,
    event_count_target: int,
    start_timestamp_us: int | None = None,
) -> slice:
    end_idx = int(np.searchsorted(timestamps_us, end_timestamp_us, side="right"))
    if policy == "fixed_count":
        start_idx = max(0, end_idx - int(event_count_target))
        return slice(start_idx, end_idx)
    if start_timestamp_us is None:
        start_timestamp_us = int(end_timestamp_us) - int(time_bin_us)
    start_idx = int(np.searchsorted(timestamps_us, start_timestamp_us, side="left"))
    return slice(start_idx, end_idx)


def _accumulation_weights(
    timestamps_us: np.ndarray,
    *,
    mode: str,
    start_timestamp_us: int | None,
    end_timestamp_us: int | None,
    causal_weight_power: float,
) -> np.ndarray:
    count = int(len(timestamps_us))
    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    if mode == "plain":
        return np.ones((count,), dtype=np.float32)
    if mode in {"causal_linear", "causal_linear_ori"}:
        if start_timestamp_us is None:
            start_timestamp_us = int(timestamps_us[0])
        if end_timestamp_us is None:
            end_timestamp_us = int(timestamps_us[-1])
        duration = float(end_timestamp_us) - float(start_timestamp_us)
        if duration <= 1e-6:
            return np.ones((count,), dtype=np.float32)
        positions = (timestamps_us.astype(np.float32) - float(start_timestamp_us)) / duration
        positions = np.clip(positions, 0.0, 1.0)
        return np.power(positions, float(causal_weight_power)).astype(np.float32)
    raise ValueError(f"Unsupported accumulation mode: {mode}")


def build_event_frame(
    events: dict[str, np.ndarray],
    *,
    sensor_size_wh: tuple[int, int],
    end_timestamp_us: int,
    event_window: dict[str, Any],
) -> tuple[np.ndarray, int]:
    timestamps = np.asarray(events["t"], dtype=np.int64)
    policy = str(event_window.get("policy", "fixed_count"))
    time_bin_us = int(event_window.get("time_bin_us", 5000))
    event_count_target = int(event_window.get("event_count_target", 5000))
    accumulation = str(event_window.get("accumulation", "causal_linear"))
    causal_weight_power = float(event_window.get("causal_weight_power", 1.0))
    start_timestamp_us = event_window.get("start_timestamp_us")
    polarity_split = bool(event_window.get("polarity_split", True))

    selection = _event_indices_for_window(
        timestamps,
        end_timestamp_us=int(end_timestamp_us),
        policy=policy,
        time_bin_us=time_bin_us,
        event_count_target=event_count_target,
        start_timestamp_us=None if start_timestamp_us is None else int(start_timestamp_us),
    )
    xs = np.asarray(events["x"][selection], dtype=np.int64)
    ys = np.asarray(events["y"][selection], dtype=np.int64)
    ps = np.asarray(events["p"][selection], dtype=np.int64)
    ts = np.asarray(events["t"][selection], dtype=np.int64)
    channels = 2 if polarity_split else 1
    voxel = np.zeros((channels, int(sensor_size_wh[1]), int(sensor_size_wh[0])), dtype=np.float32)
    valid = (xs >= 0) & (xs < sensor_size_wh[0]) & (ys >= 0) & (ys < sensor_size_wh[1])
    if not np.any(valid):
        return voxel if polarity_split else np.vstack([voxel, np.zeros_like(voxel)]), 0

    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]
    ts = ts[valid]
    weights = _accumulation_weights(
        ts,
        mode=accumulation,
        start_timestamp_us=None if start_timestamp_us is None else int(start_timestamp_us),
        end_timestamp_us=int(end_timestamp_us),
        causal_weight_power=causal_weight_power,
    )
    if polarity_split:
        neg_mask = ps <= 0
        pos_mask = ~neg_mask
        np.add.at(voxel[0], (ys[neg_mask], xs[neg_mask]), weights[neg_mask])
        np.add.at(voxel[1], (ys[pos_mask], xs[pos_mask]), weights[pos_mask])
        return voxel, int(len(xs))

    signed = np.where(ps <= 0, -1.0, 1.0).astype(np.float32)
    np.add.at(voxel[0], (ys, xs), signed * weights)
    return np.vstack([voxel, np.zeros_like(voxel)]), int(len(xs))
