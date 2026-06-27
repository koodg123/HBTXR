from __future__ import annotations

import numpy as np

from fecet_hbtxr.dataset import FECETHBTXRDataset
from fecet_hbtxr.event_repr import build_event_frame
from prepare_dataset import build_manifests, build_session_index


def test_dataset_supports_fixed_count_and_time_bin(synthetic_workspace):
    build_session_index(canonical_root=synthetic_workspace["canonical_root"], indexes_root=synthetic_workspace["indexes_root"])
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
    )
    manifest_path = synthetic_workspace["manifests_root"] / "train_manifest.jsonl"

    fixed_dataset = FECETHBTXRDataset(
        str(manifest_path),
        canonical_root=str(synthetic_workspace["canonical_root"]),
        event_builder={"policy": "fixed_count", "event_count_target": 4},
        resize_policy="facet_square_direct",
        use_cache=False,
    )
    time_dataset = FECETHBTXRDataset(
        str(manifest_path),
        canonical_root=str(synthetic_workspace["canonical_root"]),
        event_builder={"policy": "time_bin", "time_bin_us": 250},
        resize_policy="letterbox_square",
        use_cache=False,
    )

    fixed_sample = fixed_dataset[1]
    time_sample = time_dataset[1]
    assert tuple(fixed_sample["frame"].shape) == (1, 256, 256)
    assert tuple(fixed_sample["event"].shape) == (2, 256, 256)
    assert tuple(fixed_sample["mask_target"].shape) == (1, 256, 256)
    assert fixed_sample["pupil_track_target"].shape[-1] == 8
    assert fixed_sample["meta"]["event_window"]["policy"] == "fixed_count"
    assert fixed_sample["meta"]["selected_event_count"] == 4
    assert time_sample["meta"]["event_window"]["policy"] == "time_bin"
    assert time_sample["meta"]["event_window"]["time_bin_us"] == 250
    assert time_sample["meta"]["selected_event_count"] == 3
    assert time_sample["meta"]["transform"]["policy"] == "letterbox_square"
    assert time_sample["meta"]["manifest_resize_policy"] == "facet_square_direct"
    assert time_sample["meta"]["manifest_event_window"]["policy"] == "fixed_count"
    assert not (fixed_sample["event"] == time_sample["event"]).all()


def test_causal_linear_is_timestamp_aware():
    events = {
        "t": np.asarray([0, 1, 99, 100], dtype=np.int64),
        "x": np.asarray([0, 1, 2, 3], dtype=np.int16),
        "y": np.asarray([0, 0, 0, 0], dtype=np.int16),
        "p": np.asarray([1, 1, 1, 1], dtype=np.int8),
    }
    voxel, selected = build_event_frame(
        events,
        sensor_size_wh=(4, 2),
        end_timestamp_us=100,
        event_window={
            "policy": "time_bin",
            "time_bin_us": 100,
            "start_timestamp_us": 0,
            "end_timestamp_us": 100,
            "accumulation": "causal_linear",
            "causal_weight_power": 1.0,
            "polarity_split": True,
        },
    )
    assert selected == 4
    assert np.isclose(voxel[1, 0, 0], 0.0, atol=1e-6)
    assert np.isclose(voxel[1, 0, 1], 0.01, atol=1e-4)
    assert np.isclose(voxel[1, 0, 2], 0.99, atol=1e-4)
    assert np.isclose(voxel[1, 0, 3], 1.0, atol=1e-6)
