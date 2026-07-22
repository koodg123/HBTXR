from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import BuiltEventInput


class DatasetPipeline:
    def __init__(
        self,
        *,
        reader,
        transform_resolver,
        event_input_builder,
        roi_resolver,
        target_builder,
        sample_assembler,
        per_channel_normalize: bool,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.reader = reader
        self.transform_resolver = transform_resolver
        self.event_input_builder = event_input_builder
        self.roi_resolver = roi_resolver
        self.target_builder = target_builder
        self.sample_assembler = sample_assembler
        self.per_channel_normalize = bool(per_channel_normalize)
        self.augmentation = dict(augmentation or {})

    @staticmethod
    def _uniform(rng: np.random.Generator, value: Any, default: float) -> float:
        if value is None:
            return float(default)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(rng.uniform(float(value[0]), float(value[1])))
        return float(value)

    def _augment_arrays(self, frame: np.ndarray, event_input: BuiltEventInput, row: dict[str, Any]) -> tuple[np.ndarray, BuiltEventInput]:
        cfg = self.augmentation
        if not bool(cfg.get("enabled", False)):
            return frame, event_input
        seed = cfg.get("seed")
        if seed is None:
            rng = np.random.default_rng()
        else:
            sample_key = str(row.get("sample_id") or row.get("index") or "")
            sample_hash = sum(ord(ch) for ch in sample_key)
            rng = np.random.default_rng(int(seed) + sample_hash)

        prob = float(cfg.get("probability", 1.0))
        if prob < 1.0 and float(rng.random()) > prob:
            return frame, event_input

        frame_aug = frame.astype("float32", copy=True)
        event_aug = event_input.event.astype("float32", copy=True)

        frame_gain = self._uniform(rng, cfg.get("frame_gain"), 1.0)
        frame_bias = self._uniform(rng, cfg.get("frame_bias"), 0.0)
        frame_noise_std = float(cfg.get("frame_noise_std", 0.0))
        if frame_gain != 1.0 or frame_bias != 0.0:
            frame_aug = frame_aug * frame_gain + frame_bias
        if frame_noise_std > 0.0:
            frame_aug = frame_aug + rng.normal(0.0, frame_noise_std, size=frame_aug.shape).astype("float32")
        frame_aug = np.clip(frame_aug, 0.0, 1.0).astype("float32")

        event_gain = self._uniform(rng, cfg.get("event_gain"), 1.0)
        event_noise_std = float(cfg.get("event_noise_std", 0.0))
        event_dropout = float(cfg.get("event_dropout", 0.0))
        if event_gain != 1.0:
            event_aug = event_aug * event_gain
        if event_dropout > 0.0:
            keep = rng.random(event_aug.shape) >= min(max(event_dropout, 0.0), 1.0)
            event_aug = event_aug * keep.astype("float32")
        if event_noise_std > 0.0:
            event_aug = event_aug + rng.normal(0.0, event_noise_std, size=event_aug.shape).astype("float32")
        return frame_aug, BuiltEventInput(
            event=event_aug.astype("float32"),
            selected_count=event_input.selected_count,
            meta=event_input.meta,
        )

    def build_sample(self, row: dict[str, Any]) -> dict[str, Any]:
        assets = self.reader.read_sample_assets(row)
        resolved_roi = self.roi_resolver.resolve_effective_roi(
            row,
            sensor_size_wh=assets.sensor_size_wh,
            end_timestamp_us=assets.end_timestamp_us,
        )
        transform = self.transform_resolver.build(
            row,
            sensor_size_wh=assets.sensor_size_wh,
            roi_xywh=resolved_roi.roi_xywh,
        )
        mask_sensor = self.reader.load_mask(
            assets.annotation,
            sensor_size_wh=assets.sensor_size_wh,
            roi_xywh=resolved_roi.roi_xywh,
        )
        frame = self.transform_resolver.apply_frame(assets.frame_sensor, transform).astype("float32") / 255.0
        mask = self.transform_resolver.apply_mask(mask_sensor, transform)
        mask = (mask > 127).astype("float32")
        event_input = self.event_input_builder.build(
            row,
            transform,
            sensor_size_wh=assets.sensor_size_wh,
            end_timestamp_us=assets.end_timestamp_us,
            per_channel_normalize=self.per_channel_normalize,
        )
        frame, event_input = self._augment_arrays(frame, event_input, row)
        targets = self.target_builder.build(row, assets=assets, transform=transform)
        return self.sample_assembler.assemble(
            row=row,
            assets=assets,
            transform=transform,
            frame=frame,
            mask=mask,
            event_input=event_input,
            resolved_roi=resolved_roi,
            targets=targets,
        )


__all__ = [
    "DatasetPipeline",
]
