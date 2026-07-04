from __future__ import annotations

from typing import Any


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
    ) -> None:
        self.reader = reader
        self.transform_resolver = transform_resolver
        self.event_input_builder = event_input_builder
        self.roi_resolver = roi_resolver
        self.target_builder = target_builder
        self.sample_assembler = sample_assembler
        self.per_channel_normalize = bool(per_channel_normalize)

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
