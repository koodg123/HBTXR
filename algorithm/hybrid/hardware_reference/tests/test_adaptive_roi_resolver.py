from __future__ import annotations

from hbtxr.config.runtime_config import DEFAULT_EVENT_BUILDER
from hbtxr.data.components import AdaptiveRoiResolver
from hbtxr.data.event_builder import EventInputBuilder


class _Reader:
    def __init__(self, annotation):
        self.annotation = annotation

    def load_annotation(self, ref):
        assert ref["ann_id"] == "prev"
        return self.annotation


def _resolver(annotation, event_builder=None):
    builder = EventInputBuilder(
        input_size=(256, 256),
        event_builder={**DEFAULT_EVENT_BUILDER, **(event_builder or {})},
        reader=_Reader(annotation),
        use_cache=False,
        cache_root=None,
        data_mode="mode1",
        default_event_builder=DEFAULT_EVENT_BUILDER,
    )
    return AdaptiveRoiResolver(event_builder=builder)


def test_prev_pupil_anchor_crop_uses_previous_annotation_bbox():
    resolver = _resolver(
        {"pupil_region_bbox_xywh_sensor": [120, 70, 20, 10]},
        {
            "crop_policy": "prev_pupil_anchor",
            "crop_scope": "within_roi",
            "crop_margin_px": 16,
            "crop_min_side_px": 64,
        },
    )
    row = {
        "roi_xywh": [80, 40, 160, 120],
        "prev_annotation_ref": {"ann_id": "prev", "annotation_store_path": "unused.jsonl"},
    }

    resolved = resolver.resolve_effective_roi(row, sensor_size_wh=(346, 240), end_timestamp_us=123)

    assert resolved.meta["event_crop_policy"] == "prev_pupil_anchor"
    assert resolved.meta["anchor_roi_applied"] is True
    assert resolved.roi_xywh == (98.0, 43.0, 64.0, 64.0)


def test_prev_pupil_anchor_crop_falls_back_to_manifest_roi_without_annotation():
    resolver = _resolver(
        {},
        {
            "crop_policy": "prev_pupil_anchor",
            "crop_scope": "within_roi",
            "crop_margin_px": 16,
            "crop_min_side_px": 64,
        },
    )
    row = {"roi_xywh": [80, 40, 160, 120]}

    resolved = resolver.resolve_effective_roi(row, sensor_size_wh=(346, 240), end_timestamp_us=123)

    assert resolved.roi_xywh == (80.0, 40.0, 160.0, 120.0)
    assert resolved.meta["anchor_roi_applied"] is False
    assert resolved.meta["anchor_roi_reason"] == "missing_annotation_ref"
