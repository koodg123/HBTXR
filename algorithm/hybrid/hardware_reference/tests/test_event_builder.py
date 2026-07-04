from __future__ import annotations

import numpy as np

from hbtxr.config.runtime_config import DEFAULT_EVENT_BUILDER
from hbtxr.data.event_builder import EventInputBuilder


def _builder(event_builder=None) -> EventInputBuilder:
    return EventInputBuilder(
        input_size=(256, 256),
        event_builder={**DEFAULT_EVENT_BUILDER, **(event_builder or {})},
        reader=None,
        use_cache=False,
        cache_root=None,
        data_mode="mode1",
        default_event_builder=DEFAULT_EVENT_BUILDER,
    )


def test_accumulation_weights_preserve_us_timestamp_precision():
    base = 1_657_712_066_853_503
    timestamps = np.asarray([base, base + 2_500, base + 5_000], dtype=np.int64)

    weights = EventInputBuilder.accumulation_weights(
        timestamps,
        mode="fast_causal_linear",
        start_timestamp_us=base,
        end_timestamp_us=base + 5_000,
        causal_weight_power=1.0,
    )

    np.testing.assert_allclose(weights, np.asarray([0.0, 0.5, 1.0], dtype=np.float32), rtol=1e-6)


def test_fixed_count_window_drops_manifest_start_timestamp():
    builder = _builder({"policy": "fixed_count", "event_count_target": 1000})
    row = {
        "event_window": {
            "policy": "fixed_count",
            "event_count_target": 5000,
            "start_timestamp_us": 10,
            "end_timestamp_us": 20,
        }
    }

    resolved = builder.resolve_event_window(row, end_timestamp_us=20)

    assert resolved["policy"] == "fixed_count"
    assert resolved["event_count_target"] == 1000
    assert resolved["end_timestamp_us"] == 20
    assert "start_timestamp_us" not in resolved


def test_fixed_count_adaptive_count_scales_with_timestamp_delta():
    builder = _builder(
        {
            "policy": "fixed_count",
            "event_count_target": 10000,
            "adaptive_count": {
                "enabled": True,
                "reference_us": 10000,
                "min_event_count": 5000,
                "max_event_count": 20000,
                "scale_power": 1.0,
            },
        }
    )
    row = {
        "prev_sample_timestamp_us": 1000,
        "event_window": {
            "policy": "fixed_count",
            "event_count_target": 10000,
            "start_timestamp_us": 10,
            "end_timestamp_us": 21000,
        },
    }

    resolved = builder.resolve_event_window(row, end_timestamp_us=21000)

    assert resolved["event_count_target"] == 20000
    assert resolved["adaptive_count_resolved"]["delta_us"] == 20000.0
    assert "start_timestamp_us" not in resolved


def test_fixed_count_adaptive_count_disabled_keeps_event_count_target():
    builder = _builder(
        {
            "policy": "fixed_count",
            "event_count_target": 10000,
            "adaptive_count": {
                "enabled": False,
                "reference_us": 10000,
                "min_event_count": 5000,
                "max_event_count": 20000,
            },
        }
    )
    row = {
        "prev_sample_timestamp_us": 1000,
        "event_window": {
            "policy": "fixed_count",
            "event_count_target": 5000,
            "end_timestamp_us": 21000,
        },
    }

    resolved = builder.resolve_event_window(row, end_timestamp_us=21000)

    assert resolved["event_count_target"] == 10000
    assert "adaptive_count_resolved" not in resolved
