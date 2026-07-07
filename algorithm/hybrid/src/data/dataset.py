from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.config.runtime_config import DEFAULT_EVENT_BUILDER
from src.utils.io import read_json, read_jsonl
from src.utils.paths import resolve_canonical_dataset_root
from src.utils.component_registry import resolve_component

from .components import AdaptiveRoiResolver, AnnotationTargetBuilder, SampleAssembler
from .event_builder import EventInputBuilder
from .pipeline import DatasetPipeline
from .session_reader import SessionFrameEventReader
from .transform import (
    SpatialTransform,
    SpatialTransformResolver,
    apply_transform_to_event,
    apply_transform_to_frame,
    apply_transform_to_mask,
    build_transform,
)
from .utils import (
    normalize_uv,
    resolve_xywh,
    safe_text,
    state6_from_annotation,
    xywht_from_state6,
)


DEFAULT_INPUT_SIZE = (256, 256)


def load_track_target_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load sample-wise pseudo/teacher target overrides keyed by sample id."""

    if path is None or not str(path).strip():
        return {}
    raw_path = Path(path)
    if raw_path.suffix.lower() == ".jsonl":
        rows = read_jsonl(raw_path)
    else:
        payload = read_json(raw_path)
        rows = payload.get("overrides", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"track target override file must contain a list or {{'overrides': [...]}}: {raw_path}")
    overrides: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"track target override row must be an object: {row!r}")
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"track target override row missing sample_id: {row!r}")
        state = row.get("track_target_override_state", row.get("track_state"))
        if not isinstance(state, list) or len(state) < 6:
            raise ValueError(f"track target override row requires 6D state for sample_id={sample_id}")
        overrides[sample_id] = row
    return overrides


DATA_COMPONENT_VARIANTS = {
    "reader": {
        "default": SessionFrameEventReader,
        "split_v1": SessionFrameEventReader,
    },
    "transform": {
        "default": SpatialTransformResolver,
        "split_v1": SpatialTransformResolver,
    },
    "event_builder": {
        "default": EventInputBuilder,
        "split_v1": EventInputBuilder,
    },
    "roi": {
        "default": AdaptiveRoiResolver,
        "split_v1": AdaptiveRoiResolver,
    },
    "targets": {
        "default": AnnotationTargetBuilder,
        "split_v1": AnnotationTargetBuilder,
    },
    "assembler": {
        "default": SampleAssembler,
        "split_v1": SampleAssembler,
    },
}


class EVEyeHBTXRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        *,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        frame_input_size: tuple[int, int] | None = None,
        event_input_size: tuple[int, int] | None = None,
        resize_policy: str | None = "facet_square_direct",
        event_builder: dict[str, Any] | None = None,
        canonical_root: str | None = None,
        cache_root: str | None = None,
        use_cache: bool = True,
        per_channel_normalize: bool = True,
        data_mode: str = "mode1",
        canonical_name: str = "canonical1",
        manifest_name: str = "manifest1",
        frame_source: str = "original",
        mode2_execution: str = "materialized",
        component_cfg: dict[str, Any] | None = None,
        augmentation: dict[str, Any] | None = None,
        track_target_override_path: str | Path | None = None,
        allow_test_target_override: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        if track_target_override_path and "test" in self.manifest_path.name.lower() and not bool(allow_test_target_override):
            raise ValueError(
                "track_target_override_path is disabled for test manifests unless "
                "data.allow_test_target_override=true is set for diagnostics only"
            )
        self.track_target_overrides = load_track_target_overrides(track_target_override_path)
        self.input_size = tuple(int(v) for v in input_size)
        self.frame_input_size = tuple(int(v) for v in (frame_input_size or self.input_size))
        self.event_input_size = tuple(int(v) for v in (event_input_size or self.input_size))
        self.resize_policy = None if resize_policy is None or not str(resize_policy).strip() else str(resize_policy)
        self.event_builder = {**DEFAULT_EVENT_BUILDER, **(event_builder or {})}
        self.canonical_root = None if canonical_root is None else resolve_canonical_dataset_root(Path(canonical_root), canonical_name, prefer_nested=False)
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.use_cache = bool(use_cache)
        self.per_channel_normalize = bool(per_channel_normalize)
        self.data_mode = safe_text(data_mode, "mode1")
        self.canonical_name = safe_text(canonical_name, "canonical1")
        self.manifest_name = safe_text(manifest_name, "manifest1")
        self.frame_source = safe_text(frame_source, "original")
        self.mode2_execution = safe_text(mode2_execution, "materialized")
        self.component_cfg = dict(component_cfg or {})
        self.augmentation = dict(augmentation or {})

        reader_cls, reader_kwargs = resolve_component(
            self.component_cfg,
            key="reader",
            variants=DATA_COMPONENT_VARIANTS,
            group_name="data",
        )
        transform_cls, transform_kwargs = resolve_component(
            self.component_cfg,
            key="transform",
            variants=DATA_COMPONENT_VARIANTS,
            group_name="data",
        )
        event_builder_cls, event_builder_kwargs = resolve_component(
            self.component_cfg,
            key="event_builder",
            variants=DATA_COMPONENT_VARIANTS,
            group_name="data",
        )
        roi_cls, roi_kwargs = resolve_component(
            self.component_cfg,
            key="roi",
            variants=DATA_COMPONENT_VARIANTS,
            group_name="data",
        )
        target_cls, target_kwargs = resolve_component(
            self.component_cfg,
            key="targets",
            variants=DATA_COMPONENT_VARIANTS,
            group_name="data",
        )
        assembler_cls, assembler_kwargs = resolve_component(
            self.component_cfg,
            key="assembler",
            variants=DATA_COMPONENT_VARIANTS,
            group_name="data",
        )

        self.reader = reader_cls(
            manifest_path=self.manifest_path,
            canonical_root=self.canonical_root,
            **reader_kwargs,
        )
        self.transform_resolver = transform_cls(
            input_size=self.input_size,
            resize_policy=self.resize_policy,
            **transform_kwargs,
        )
        self.event_input_builder = event_builder_cls(
            input_size=self.input_size,
            event_builder=self.event_builder,
            reader=self.reader,
            use_cache=self.use_cache,
            cache_root=self.cache_root,
            data_mode=self.data_mode,
            default_event_builder=DEFAULT_EVENT_BUILDER,
            **event_builder_kwargs,
        )
        self.roi_resolver = roi_cls(event_builder=self.event_input_builder, **roi_kwargs)
        self.target_builder = target_cls(**target_kwargs)
        self.sample_assembler = assembler_cls(
            data_mode=self.data_mode,
            canonical_name=self.canonical_name,
            manifest_name=self.manifest_name,
            frame_source=self.frame_source,
            **assembler_kwargs,
        )
        self.pipeline = DatasetPipeline(
            reader=self.reader,
            transform_resolver=self.transform_resolver,
            event_input_builder=self.event_input_builder,
            roi_resolver=self.roi_resolver,
            target_builder=self.target_builder,
            sample_assembler=self.sample_assembler,
            per_channel_normalize=self.per_channel_normalize,
            augmentation=self.augmentation,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, raw_path: str | Path) -> Path:
        return self.reader.resolve_path(raw_path)

    def _load_events(self, events_path: str | Path):
        return self.reader.load_events(events_path)

    def _normalize_event_input(self, event):
        return self.event_input_builder.normalize_event_input(event, per_channel_normalize=self.per_channel_normalize)

    def _resolve_source_pair_scope(self) -> str:
        return self.event_input_builder.resolve_source_pair_scope()

    def _resolve_source_pair_timestamps(self, row: dict[str, Any]):
        return self.event_input_builder.resolve_source_pair_timestamps(row)

    def _resolve_anchor_event_window(self, anchor_timestamp_us: int):
        return self.event_input_builder.resolve_anchor_event_window(anchor_timestamp_us)

    def _resolve_timestamp_selection(self, timestamps, *, start_timestamp_us: int, end_timestamp_us: int, include_start: bool):
        return self.event_input_builder.resolve_timestamp_selection(
            timestamps,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            include_start=include_start,
        )

    def _build_transformed_event_from_window(self, *, events_path: str | Path, transform: SpatialTransform, sensor_size_wh: tuple[int, int], event_window: dict[str, Any], end_timestamp_us: int):
        return self.event_input_builder.build_transformed_event_from_window(
            events_path=events_path,
            transform=transform,
            sensor_size_wh=sensor_size_wh,
            event_window=event_window,
            end_timestamp_us=end_timestamp_us,
        )

    def _build_transformed_event_from_selection(self, *, events, transform: SpatialTransform, sensor_size_wh: tuple[int, int], selection: slice, start_timestamp_us: int, end_timestamp_us: int):
        return self.event_input_builder.build_transformed_event_from_selection(
            events=events,
            transform=transform,
            sensor_size_wh=sensor_size_wh,
            selection=selection,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
        )

    def _resolve_event_window(self, row: dict[str, Any], *, end_timestamp_us: int):
        return self.event_input_builder.resolve_event_window(row, end_timestamp_us=end_timestamp_us)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample = self.pipeline.build_sample(row)
        override = self.track_target_overrides.get(str(sample["sample_id"]))
        if self.track_target_overrides:
            if override is None:
                state = sample["cur_state"].detach().clone()
                weight = 0.0
                source = "missing"
            else:
                state = torch.tensor(override.get("track_target_override_state", override.get("track_state"))[:6], dtype=torch.float32)
                weight = float(override.get("track_target_override_weight", 1.0))
                source = str(override.get("track_target_override_source", override.get("selected_teacher", "override")))
            sample["track_target_override_state"] = state
            sample["track_target_override_weight"] = torch.tensor(weight, dtype=torch.float32)
            meta = dict(sample.get("meta") or {})
            meta["track_target_override_source"] = source
            meta["track_target_override_present"] = override is not None
            sample["meta"] = meta
        return sample


class _ModeSpecializedDataset(EVEyeHBTXRDataset):
    MODE_DEFAULTS: dict[str, str] = {}

    def __init__(self, manifest_path: str, **kwargs: Any) -> None:
        for key, value in self.MODE_DEFAULTS.items():
            kwargs.setdefault(key, value)
        super().__init__(manifest_path, **kwargs)


class Mode1Dataset(_ModeSpecializedDataset):
    MODE_DEFAULTS = {
        "data_mode": "mode1",
        "canonical_name": "canonical1",
        "manifest_name": "manifest1",
        "frame_source": "original",
    }


class Mode0Dataset(_ModeSpecializedDataset):
    MODE_DEFAULTS = {
        "data_mode": "mode0",
        "canonical_name": "canonical0",
        "manifest_name": "manifest0",
        "frame_source": "original",
    }


class Mode2Dataset(_ModeSpecializedDataset):
    MODE_DEFAULTS = {
        "data_mode": "mode2",
        "canonical_name": "canonical2",
        "manifest_name": "manifest2",
        "frame_source": "interpolated",
    }


_build_transform = build_transform
_apply_transform_to_frame = apply_transform_to_frame
_apply_transform_to_mask = apply_transform_to_mask
_apply_transform_to_event = apply_transform_to_event
_resolve_xywh = resolve_xywh


def _build_event_frame(
    events,
    *,
    sensor_size_wh: tuple[int, int],
    end_timestamp_us: int,
    event_window: dict[str, Any],
):
    builder = EventInputBuilder(
        input_size=DEFAULT_INPUT_SIZE,
        event_builder=event_window,
        reader=None,
        use_cache=False,
        cache_root=None,
        data_mode="mode1",
        default_event_builder=DEFAULT_EVENT_BUILDER,
    )
    return builder.build_event_frame(
        events,
        sensor_size_wh=sensor_size_wh,
        end_timestamp_us=end_timestamp_us,
        event_window=event_window,
    )


def _build_event_frame_from_selected_events(
    events,
    *,
    selection: slice,
    sensor_size_wh: tuple[int, int],
    event_window: dict[str, Any],
    end_timestamp_us: int,
):
    builder = EventInputBuilder(
        input_size=DEFAULT_INPUT_SIZE,
        event_builder=event_window,
        reader=None,
        use_cache=False,
        cache_root=None,
        data_mode="mode1",
        default_event_builder=DEFAULT_EVENT_BUILDER,
    )
    return builder.build_event_frame_from_selected_events(
        events,
        selection=selection,
        sensor_size_wh=sensor_size_wh,
        event_window=event_window,
        end_timestamp_us=end_timestamp_us,
    )


__all__ = [
    "DEFAULT_EVENT_BUILDER",
    "EVEyeHBTXRDataset",
    "load_track_target_overrides",
    "Mode0Dataset",
    "Mode1Dataset",
    "Mode2Dataset",
    "SpatialTransform",
    "_apply_transform_to_event",
    "_apply_transform_to_frame",
    "_apply_transform_to_mask",
    "_build_event_frame",
    "_build_event_frame_from_selected_events",
    "_build_transform",
    "_resolve_xywh",
    "normalize_uv",
    "state6_from_annotation",
    "xywht_from_state6",
]
