from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.cache import build_cache_file, load_or_build_npz_array

from .contracts import BuiltEventInput
from .transform import SpatialTransform, apply_transform_to_event
from .utils import extract_valid_event_points, safe_float, safe_text


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


class EventInputBuilder:
    def __init__(
        self,
        *,
        input_size: tuple[int, int],
        event_builder: dict[str, Any],
        reader: Any,
        use_cache: bool,
        cache_root: Path | None,
        data_mode: str,
        default_event_builder: dict[str, Any],
    ) -> None:
        self.input_size = tuple(int(v) for v in input_size)
        self.event_builder = dict(event_builder or {})
        self.reader = reader
        self.use_cache = bool(use_cache)
        self.cache_root = cache_root
        self.data_mode = safe_text(data_mode, "mode1")
        self.default_event_builder = dict(default_event_builder)
        self.synthetic_event_generation_strategy = safe_text(
            self.event_builder.get("generation_strategy"),
            "raw_window",
        ).lower()
        self.synthetic_event_source_pair_scope = safe_text(
            self.event_builder.get("source_pair_scope"),
            "anchor_window",
        ).lower()
        self.synthetic_event_average_weighting = safe_text(
            self.event_builder.get("average_weighting"),
            "mean",
        ).lower()

    @staticmethod
    def event_indices_for_window(
        timestamps: np.ndarray,
        *,
        end_timestamp_us: int,
        policy: str,
        time_bin_us: int,
        event_count_target: int,
        start_timestamp_us: int | None = None,
    ) -> slice:
        end_idx = int(np.searchsorted(timestamps, end_timestamp_us, side="right"))
        if policy == "fixed_count":
            start_idx = max(0, end_idx - int(event_count_target))
            return slice(start_idx, end_idx)
        if policy == "interval_all":
            if start_timestamp_us is None:
                start_timestamp_us = int(end_timestamp_us)
            start_idx = int(np.searchsorted(timestamps, start_timestamp_us, side="right"))
            return slice(start_idx, end_idx)
        if start_timestamp_us is None:
            start_timestamp_us = int(end_timestamp_us) - int(time_bin_us)
        start_idx = int(np.searchsorted(timestamps, start_timestamp_us, side="left"))
        return slice(start_idx, end_idx)

    @staticmethod
    def accumulation_weights(
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
        if mode in {"causal_linear", "causal_linear_ori", "fast_causal_linear"}:
            if start_timestamp_us is None:
                start_timestamp_us = int(timestamps_us[0])
            if end_timestamp_us is None:
                end_timestamp_us = int(timestamps_us[-1])
            duration = float(end_timestamp_us) - float(start_timestamp_us)
            if duration <= 1e-6:
                return np.ones((count,), dtype=np.float32)
            positions = (timestamps_us.astype(np.float64) - float(start_timestamp_us)) / duration
            positions = np.clip(positions, 0.0, 1.0)
            return np.power(positions, float(causal_weight_power)).astype(np.float32)
        raise ValueError(f"Unsupported accumulation mode: {mode}")

    @staticmethod
    def accumulate_events(
        voxel: np.ndarray,
        *,
        xs: np.ndarray,
        ys: np.ndarray,
        ps: np.ndarray,
        weights: np.ndarray,
        mode: str,
        fast_causal_limit: float,
    ) -> None:
        if mode == "fast_causal_linear":
            limit = max(0.0, float(fast_causal_limit))
            if limit <= 0.0:
                return
            for x, y, p, weight in zip(xs.tolist(), ys.tolist(), ps.tolist(), weights.tolist(), strict=False):
                channel = 0 if int(p) <= 0 else 1
                current = float(voxel[channel, y, x])
                if current >= limit:
                    continue
                voxel[channel, y, x] = min(limit, current + float(weight))
            return

        neg_mask = ps <= 0
        pos_mask = ~neg_mask
        np.add.at(voxel[0], (ys[neg_mask], xs[neg_mask]), weights[neg_mask])
        np.add.at(voxel[1], (ys[pos_mask], xs[pos_mask]), weights[pos_mask])

    def build_event_frame(
        self,
        events: dict[str, np.ndarray],
        *,
        sensor_size_wh: tuple[int, int],
        end_timestamp_us: int,
        event_window: dict[str, Any],
    ) -> tuple[np.ndarray, int]:
        timestamps = np.asarray(events["t"], dtype=np.int64)
        policy = str(event_window.get("policy", "fixed_count"))
        time_bin_us = int(event_window.get("time_bin_us", self.default_event_builder["time_bin_us"]))
        event_count_target = int(event_window.get("event_count_target", self.default_event_builder["event_count_target"]))
        accumulation = str(event_window.get("accumulation", self.default_event_builder["accumulation"]))
        causal_weight_power = float(event_window.get("causal_weight_power", self.default_event_builder["causal_weight_power"]))
        fast_causal_limit = float(event_window.get("fast_causal_limit", self.default_event_builder["fast_causal_limit"]))
        start_timestamp_us = event_window.get("start_timestamp_us")

        selection = self.event_indices_for_window(
            timestamps,
            end_timestamp_us=int(end_timestamp_us),
            policy=policy,
            time_bin_us=time_bin_us,
            event_count_target=event_count_target,
            start_timestamp_us=None if start_timestamp_us is None else int(start_timestamp_us),
        )
        return self.build_event_frame_from_selected_events(
            events,
            selection=selection,
            sensor_size_wh=sensor_size_wh,
            event_window=event_window,
            end_timestamp_us=end_timestamp_us,
        )

    def build_event_frame_from_selected_events(
        self,
        events: dict[str, np.ndarray],
        *,
        selection: slice,
        sensor_size_wh: tuple[int, int],
        event_window: dict[str, Any],
        end_timestamp_us: int,
    ) -> tuple[np.ndarray, int]:
        accumulation = str(event_window.get("accumulation", self.default_event_builder["accumulation"]))
        causal_weight_power = float(event_window.get("causal_weight_power", self.default_event_builder["causal_weight_power"]))
        fast_causal_limit = float(event_window.get("fast_causal_limit", self.default_event_builder["fast_causal_limit"]))
        start_timestamp_us = event_window.get("start_timestamp_us")

        xs = np.asarray(events["x"][selection], dtype=np.int64)
        ys = np.asarray(events["y"][selection], dtype=np.int64)
        ps = np.asarray(events["p"][selection], dtype=np.int64)
        ts = np.asarray(events["t"][selection], dtype=np.int64)
        voxel = np.zeros((2, int(sensor_size_wh[1]), int(sensor_size_wh[0])), dtype=np.float32)
        valid = (xs >= 0) & (xs < sensor_size_wh[0]) & (ys >= 0) & (ys < sensor_size_wh[1])
        if not np.any(valid):
            return voxel, 0

        xs = xs[valid]
        ys = ys[valid]
        ps = ps[valid]
        ts = ts[valid]
        if start_timestamp_us is None and len(ts) > 0:
            start_timestamp_us = int(ts[0])
        weights = self.accumulation_weights(
            ts,
            mode=accumulation,
            start_timestamp_us=None if start_timestamp_us is None else int(start_timestamp_us),
            end_timestamp_us=int(end_timestamp_us),
            causal_weight_power=causal_weight_power,
        )
        self.accumulate_events(
            voxel,
            xs=xs,
            ys=ys,
            ps=ps,
            weights=weights,
            mode=accumulation,
            fast_causal_limit=fast_causal_limit,
        )
        return voxel, int(len(xs))

    @staticmethod
    def extract_valid_event_points(
        events: dict[str, np.ndarray],
        *,
        selection: slice,
        sensor_size_wh: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return extract_valid_event_points(events, selection=selection, sensor_size_wh=sensor_size_wh)

    def normalize_event_input(self, event: np.ndarray, *, per_channel_normalize: bool) -> np.ndarray:
        built = np.asarray(event, dtype=np.float32).copy()
        if per_channel_normalize:
            for channel in range(built.shape[0]):
                denom = float(np.max(np.abs(built[channel])))
                if denom > 1e-6:
                    built[channel] /= denom
        return built

    def resolve_event_generation_strategy(self, row: dict[str, Any]) -> str:
        if self.reader.resolve_session_store_ref(row) is not None and row.get("event_index_range") not in (None, []):
            return "target_fps_session_store"
        if self.data_mode != "mode2" or not bool(row.get("synthetic_frame_flag", False)):
            return "raw_window"
        strategy = self.synthetic_event_generation_strategy
        return strategy if strategy in {"raw_window", "source_pair_average"} else "raw_window"

    def resolve_source_pair_scope(self) -> str:
        scope = self.synthetic_event_source_pair_scope
        return scope if scope in {"anchor_window", "pair_local"} else "anchor_window"

    def resolve_source_pair_timestamps(self, row: dict[str, Any]) -> tuple[int, int] | None:
        synthetic_event_window = row.get("synthetic_event_window") or {}
        pair = (
            synthetic_event_window.get("source_timestamp_pair_us")
            or row.get("source_timestamp_pair_us")
            or row.get("interp_source_timestamps_us")
        )
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        return int(pair[0]), int(pair[1])

    def resolve_anchor_event_window(self, anchor_timestamp_us: int) -> dict[str, Any]:
        effective = dict(self.event_builder)
        effective["end_timestamp_us"] = int(anchor_timestamp_us)
        policy = str(effective.get("policy", self.default_event_builder["policy"]))
        if policy == "time_bin":
            effective["start_timestamp_us"] = int(
                int(anchor_timestamp_us) - int(effective.get("time_bin_us", self.default_event_builder["time_bin_us"]))
            )
        else:
            effective.pop("start_timestamp_us", None)
        return effective

    def resolve_interval_event_window(self, *, start_timestamp_us: int, end_timestamp_us: int) -> dict[str, Any]:
        effective = dict(self.event_builder)
        effective["start_timestamp_us"] = int(start_timestamp_us)
        effective["end_timestamp_us"] = int(end_timestamp_us)
        return effective

    @staticmethod
    def resolve_timestamp_selection(
        timestamps: np.ndarray,
        *,
        start_timestamp_us: int,
        end_timestamp_us: int,
        include_start: bool,
    ) -> slice:
        start_side = "left" if include_start else "right"
        start_idx = int(np.searchsorted(timestamps, int(start_timestamp_us), side=start_side))
        end_idx = int(np.searchsorted(timestamps, int(end_timestamp_us), side="right"))
        return slice(max(0, start_idx), max(start_idx, end_idx))

    def resolve_average_weights(self, row: dict[str, Any]) -> tuple[float, float]:
        weighting = self.synthetic_event_average_weighting
        if weighting == "alpha":
            alpha = float(np.clip(safe_float(row.get("interp_alpha", 0.5), 0.5), 0.0, 1.0))
            return 1.0 - alpha, alpha
        return 0.5, 0.5

    def build_event_input_from_session_store(
        self,
        row: dict[str, Any],
        transform: SpatialTransform,
        *,
        sensor_size_wh: tuple[int, int],
        end_timestamp_us: int,
        per_channel_normalize: bool,
    ) -> tuple[np.ndarray, int]:
        _, _, store = self.reader.load_session_store(row)
        event_index_range = row.get("event_index_range")
        if not isinstance(event_index_range, (list, tuple)) or len(event_index_range) != 2:
            raise KeyError("Target-fps manifest row is missing event_index_range")
        start_idx = max(0, int(event_index_range[0]))
        end_idx = max(start_idx, int(event_index_range[1]))
        event_window = self.resolve_event_window(row, end_timestamp_us=end_timestamp_us)
        voxel_sensor, selected_count = self.build_event_frame_from_selected_events(
            {
                "t": np.asarray(store["event_t"]),
                "x": np.asarray(store["event_x"]),
                "y": np.asarray(store["event_y"]),
                "p": np.asarray(store["event_p"]),
            },
            selection=slice(start_idx, end_idx),
            sensor_size_wh=sensor_size_wh,
            event_window=event_window,
            end_timestamp_us=end_timestamp_us,
        )
        built = apply_transform_to_event(voxel_sensor, transform).astype(np.float32)
        return self.normalize_event_input(built, per_channel_normalize=per_channel_normalize), selected_count

    def build_transformed_event_from_window(
        self,
        *,
        events_path: str | Path,
        transform: SpatialTransform,
        sensor_size_wh: tuple[int, int],
        event_window: dict[str, Any],
        end_timestamp_us: int,
    ) -> tuple[np.ndarray, int]:
        voxel_sensor, selected_count = self.build_event_frame(
            self.reader.load_events(events_path),
            sensor_size_wh=sensor_size_wh,
            end_timestamp_us=end_timestamp_us,
            event_window=event_window,
        )
        built = apply_transform_to_event(voxel_sensor, transform).astype(np.float32)
        return built, selected_count

    def build_transformed_event_from_selection(
        self,
        *,
        events: dict[str, np.ndarray],
        transform: SpatialTransform,
        sensor_size_wh: tuple[int, int],
        selection: slice,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[np.ndarray, int]:
        voxel_sensor, selected_count = self.build_event_frame_from_selected_events(
            events,
            selection=selection,
            sensor_size_wh=sensor_size_wh,
            event_window=self.resolve_interval_event_window(
                start_timestamp_us=int(start_timestamp_us),
                end_timestamp_us=int(end_timestamp_us),
            ),
            end_timestamp_us=int(end_timestamp_us),
        )
        built = apply_transform_to_event(voxel_sensor, transform).astype(np.float32)
        return built, selected_count

    def resolve_event_window(self, row: dict[str, Any], *, end_timestamp_us: int) -> dict[str, Any]:
        event_key = "synthetic_event_window" if self.data_mode == "mode2" and row.get("synthetic_event_window") else "event_window"
        manifest_event_window = dict(row.get(event_key) or row.get("event_window") or {})
        effective = {**manifest_event_window, **self.event_builder}
        effective_end = int(manifest_event_window.get("end_timestamp_us", end_timestamp_us))
        effective["end_timestamp_us"] = effective_end
        policy = str(effective.get("policy", self.default_event_builder["policy"]))
        if policy == "time_bin":
            effective["start_timestamp_us"] = int(effective_end - int(effective.get("time_bin_us", self.default_event_builder["time_bin_us"])))
        elif policy == "interval_all":
            effective["start_timestamp_us"] = int(
                row.get(
                    "prev_sample_timestamp_us",
                    manifest_event_window.get("start_timestamp_us", effective_end),
                )
            )
        elif policy == "fixed_count":
            adaptive_count = effective.get("adaptive_count")
            if isinstance(adaptive_count, dict) and _safe_bool(adaptive_count.get("enabled"), False):
                base_count = int(effective.get("event_count_target", self.default_event_builder["event_count_target"]))
                reference_us = max(1.0, safe_float(adaptive_count.get("reference_us"), 10_000.0))
                min_count = max(1, int(safe_float(adaptive_count.get("min_event_count"), base_count)))
                max_count = max(min_count, int(safe_float(adaptive_count.get("max_event_count"), base_count)))
                scale_power = max(0.0, safe_float(adaptive_count.get("scale_power"), 1.0))
                previous_us = row.get("prev_sample_timestamp_us", manifest_event_window.get("start_timestamp_us"))
                if previous_us is not None:
                    delta_us = max(1.0, float(effective_end) - float(previous_us))
                    scaled = float(base_count) * ((delta_us / reference_us) ** scale_power)
                    resolved_count = int(np.clip(round(scaled), min_count, max_count))
                    effective["event_count_target"] = resolved_count
                    effective["adaptive_count_resolved"] = {
                        "base_event_count": int(base_count),
                        "delta_us": float(delta_us),
                        "reference_us": float(reference_us),
                        "scale_power": float(scale_power),
                        "min_event_count": int(min_count),
                        "max_event_count": int(max_count),
                        "resolved_event_count": int(resolved_count),
                    }
            effective.pop("start_timestamp_us", None)
        elif manifest_event_window.get("start_timestamp_us") is not None:
            effective["start_timestamp_us"] = int(manifest_event_window["start_timestamp_us"])
        return effective

    def build(
        self,
        row: dict[str, Any],
        transform: SpatialTransform,
        *,
        sensor_size_wh: tuple[int, int],
        end_timestamp_us: int,
        per_channel_normalize: bool,
    ) -> BuiltEventInput:
        effective_event_window = self.resolve_event_window(row, end_timestamp_us=end_timestamp_us)
        generation_strategy = self.resolve_event_generation_strategy(row)
        source_pair_timestamps = self.resolve_source_pair_timestamps(row) if generation_strategy == "source_pair_average" else None
        source_pair_scope = self.resolve_source_pair_scope() if generation_strategy == "source_pair_average" else None
        cache_payload = {
            "sample_id": row["sample_id"],
            "resize_policy": transform.policy,
            "target_size": list(self.input_size),
            "event_window": effective_event_window,
            "event_generation_strategy": generation_strategy,
            "event_generation_source_pair_scope": source_pair_scope,
            "event_average_weighting": self.synthetic_event_average_weighting,
            "source_timestamp_pair_us": list(source_pair_timestamps) if source_pair_timestamps is not None else None,
            "event_index_range": row.get("event_index_range"),
        }
        events_path = row.get("events_npz")

        def _builder() -> np.ndarray:
            if generation_strategy == "target_fps_session_store":
                built, selected_count = self.build_event_input_from_session_store(
                    row,
                    transform,
                    sensor_size_wh=sensor_size_wh,
                    end_timestamp_us=end_timestamp_us,
                    per_channel_normalize=per_channel_normalize,
                )
            elif generation_strategy == "source_pair_average" and source_pair_timestamps is not None:
                prev_timestamp_us, next_timestamp_us = source_pair_timestamps
                if source_pair_scope == "pair_local":
                    events = self.reader.load_events(events_path)
                    timestamps = np.asarray(events["t"], dtype=np.int64)
                    pair_start_us, pair_end_us = sorted((int(prev_timestamp_us), int(next_timestamp_us)))
                    split_timestamp_us = int(np.clip(int(row.get("sample_timestamp_us", end_timestamp_us)), pair_start_us, pair_end_us))
                    prev_event, prev_count = self.build_transformed_event_from_selection(
                        events=events,
                        transform=transform,
                        sensor_size_wh=sensor_size_wh,
                        selection=self.resolve_timestamp_selection(
                            timestamps,
                            start_timestamp_us=pair_start_us,
                            end_timestamp_us=split_timestamp_us,
                            include_start=True,
                        ),
                        start_timestamp_us=pair_start_us,
                        end_timestamp_us=split_timestamp_us,
                    )
                    next_event, next_count = self.build_transformed_event_from_selection(
                        events=events,
                        transform=transform,
                        sensor_size_wh=sensor_size_wh,
                        selection=self.resolve_timestamp_selection(
                            timestamps,
                            start_timestamp_us=split_timestamp_us,
                            end_timestamp_us=pair_end_us,
                            include_start=False,
                        ),
                        start_timestamp_us=split_timestamp_us,
                        end_timestamp_us=pair_end_us,
                    )
                    pair_selection = self.resolve_timestamp_selection(
                        timestamps,
                        start_timestamp_us=pair_start_us,
                        end_timestamp_us=pair_end_us,
                        include_start=True,
                    )
                    pair_local_count = max(0, int(pair_selection.stop) - int(pair_selection.start))
                else:
                    prev_event, prev_count = self.build_transformed_event_from_window(
                        events_path=events_path,
                        transform=transform,
                        sensor_size_wh=sensor_size_wh,
                        event_window=self.resolve_anchor_event_window(prev_timestamp_us),
                        end_timestamp_us=prev_timestamp_us,
                    )
                    next_event, next_count = self.build_transformed_event_from_window(
                        events_path=events_path,
                        transform=transform,
                        sensor_size_wh=sensor_size_wh,
                        event_window=self.resolve_anchor_event_window(next_timestamp_us),
                        end_timestamp_us=next_timestamp_us,
                    )
                    pair_local_count = None
                prev_weight, next_weight = self.resolve_average_weights(row)
                built = self.normalize_event_input(prev_weight * prev_event + next_weight * next_event, per_channel_normalize=per_channel_normalize)
                if pair_local_count is not None:
                    selected_count = int(pair_local_count)
                else:
                    selected_count = int(round(prev_weight * float(prev_count) + next_weight * float(next_count)))
            else:
                built, selected_count = self.build_transformed_event_from_window(
                    events_path=events_path,
                    transform=transform,
                    sensor_size_wh=sensor_size_wh,
                    event_window=effective_event_window,
                    end_timestamp_us=end_timestamp_us,
                )
                built = self.normalize_event_input(built, per_channel_normalize=per_channel_normalize)
            return np.concatenate([built, np.asarray([[selected_count]], dtype=np.float32)], axis=None)

        if self.use_cache and self.cache_root is not None:
            cache_file = build_cache_file(self.cache_root, "event", row["sample_id"], cache_payload)
            packed, _ = load_or_build_npz_array(cache_file, "array", _builder)
        else:
            packed = _builder()

        event = np.asarray(packed[:-1], dtype=np.float32).reshape(2, self.input_size[1], self.input_size[0])
        selected_count = int(round(float(packed[-1])))
        meta = {
            "effective_event_window": effective_event_window,
            "event_generation_strategy": generation_strategy,
            "source_pair_scope": source_pair_scope,
            "average_weighting": self.synthetic_event_average_weighting if generation_strategy == "source_pair_average" else None,
            "source_pair_timestamps": list(source_pair_timestamps) if source_pair_timestamps is not None else None,
        }
        return BuiltEventInput(
            event=event,
            selected_count=selected_count,
            meta=meta,
        )
