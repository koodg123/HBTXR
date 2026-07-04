from __future__ import annotations

import numpy as np
from PIL import Image

from src.data.contracts import BuiltEventInput, ResolvedRoi
from src.data.dataset import EVEyeHBTXRDataset, Mode2Dataset, _build_event_frame, _build_transform, _resolve_xywh
from src.data.loader import build_dataset_kwargs
from src.preprocess.build_manifests import build_manifests
from src.preprocess.canonicalize import canonicalize_dataset


def test_dataset_supports_fixed_count_and_time_bin(synthetic_workspace):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    manifest_path = synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl"

    fixed_dataset = EVEyeHBTXRDataset(
        str(manifest_path),
        canonical_root=str(synthetic_workspace["canonical_root"]),
        event_builder={"policy": "fixed_count", "event_count_target": 4},
        resize_policy="facet_square_direct",
        use_cache=False,
    )
    time_dataset = EVEyeHBTXRDataset(
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
    assert fixed_sample["meta"]["data_mode"] == "mode1"
    assert fixed_sample["meta"]["canonical_name"] == "canonical1"
    assert fixed_sample["meta"]["manifest_name"] == "manifest1"
    assert time_sample["meta"]["event_window"]["policy"] == "time_bin"
    assert time_sample["meta"]["event_window"]["time_bin_us"] == 250
    assert time_sample["meta"]["selected_event_count"] == 3
    assert time_sample["meta"]["transform"]["policy"] == "letterbox_square"
    assert time_sample["meta"]["manifest_resize_policy"] == "facet_square_direct"
    assert time_sample["meta"]["manifest_event_window"]["policy"] == "fixed_count"
    assert not (fixed_sample["event"] == time_sample["event"]).all()


def test_dataset_split_components_can_be_selected_from_kwargs(synthetic_workspace):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    manifest_path = synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl"
    dataset = EVEyeHBTXRDataset(
        str(manifest_path),
        canonical_root=str(synthetic_workspace["canonical_root"]),
        event_builder={"policy": "fixed_count", "event_count_target": 4},
        resize_policy="facet_square_direct",
        use_cache=False,
        component_cfg={
            "reader": {"variant": "split_v1"},
            "transform": {"variant": "split_v1"},
            "event_builder": {"variant": "split_v1"},
            "roi": {"variant": "split_v1"},
            "targets": {"variant": "split_v1"},
            "assembler": {"variant": "split_v1"},
        },
    )

    sample = dataset[0]
    assert sample["meta"]["selected_event_count"] >= 1
    assert dataset.reader.__class__.__name__ == "SessionFrameEventReader"
    assert dataset.transform_resolver.__class__.__name__ == "SpatialTransformResolver"
    assert dataset.event_input_builder.__class__.__name__ == "EventInputBuilder"
    assert dataset.roi_resolver.__class__.__name__ == "AdaptiveRoiResolver"
    assert dataset.target_builder.__class__.__name__ == "AnnotationTargetBuilder"
    assert dataset.sample_assembler.__class__.__name__ == "SampleAssembler"
    assert dataset.pipeline.__class__.__name__ == "DatasetPipeline"


def test_dataset_contracts_expose_structured_roi_and_event_bundles(synthetic_workspace):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    manifest_path = synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl"
    dataset = EVEyeHBTXRDataset(
        str(manifest_path),
        canonical_root=str(synthetic_workspace["canonical_root"]),
        event_builder={"policy": "fixed_count", "event_count_target": 4},
        resize_policy="facet_square_direct",
        use_cache=False,
    )

    row = dataset.rows[0]
    assets = dataset.reader.read_sample_assets(row)
    resolved_roi = dataset.roi_resolver.resolve_effective_roi(
        row,
        sensor_size_wh=assets.sensor_size_wh,
        end_timestamp_us=assets.end_timestamp_us,
    )
    transform = dataset.transform_resolver.build(
        row,
        sensor_size_wh=assets.sensor_size_wh,
        roi_xywh=resolved_roi.roi_xywh,
    )
    event_input = dataset.event_input_builder.build(
        row,
        transform,
        sensor_size_wh=assets.sensor_size_wh,
        end_timestamp_us=assets.end_timestamp_us,
        per_channel_normalize=dataset.per_channel_normalize,
    )

    assert isinstance(resolved_roi, ResolvedRoi)
    assert isinstance(event_input, BuiltEventInput)
    assert len(resolved_roi.roi_xywh) == 4
    assert "effective_event_window" in event_input.meta
    assert tuple(event_input.event.shape) == (2, 256, 256)


def test_build_dataset_kwargs_merges_component_overrides():
    kwargs = build_dataset_kwargs(
        {
            "mode": "mode2",
            "components": {
                "reader": {"variant": "default"},
                "roi": {"variant": "default"},
            },
            "mode2": {
                "components": {
                    "reader": {"variant": "split_v1"},
                    "event_builder": {"variant": "split_v1"},
                }
            },
        }
    )

    assert kwargs["component_cfg"]["reader"]["variant"] == "split_v1"
    assert kwargs["component_cfg"]["roi"]["variant"] == "default"
    assert kwargs["component_cfg"]["event_builder"]["variant"] == "split_v1"


def test_causal_linear_is_timestamp_aware():
    events = {
        "t": np.asarray([0, 1, 99, 100], dtype=np.int64),
        "x": np.asarray([0, 1, 2, 3], dtype=np.int16),
        "y": np.asarray([0, 0, 0, 0], dtype=np.int16),
        "p": np.asarray([1, 1, 1, 1], dtype=np.int8),
    }
    voxel, selected = _build_event_frame(
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
        },
    )
    assert selected == 4
    assert np.isclose(voxel[1, 0, 0], 0.0, atol=1e-6)
    assert np.isclose(voxel[1, 0, 1], 0.01, atol=1e-4)
    assert np.isclose(voxel[1, 0, 2], 0.99, atol=1e-4)
    assert np.isclose(voxel[1, 0, 3], 1.0, atol=1e-6)


def test_fast_causal_linear_saturates_per_pixel():
    events = {
        "t": np.asarray([10, 50, 100], dtype=np.int64),
        "x": np.asarray([0, 0, 0], dtype=np.int16),
        "y": np.asarray([0, 0, 0], dtype=np.int16),
        "p": np.asarray([1, 1, 1], dtype=np.int8),
    }
    voxel, selected = _build_event_frame(
        events,
        sensor_size_wh=(2, 2),
        end_timestamp_us=100,
        event_window={
            "policy": "time_bin",
            "time_bin_us": 100,
            "start_timestamp_us": 0,
            "end_timestamp_us": 100,
            "accumulation": "fast_causal_linear",
            "causal_weight_power": 1.0,
            "fast_causal_limit": 1.0,
        },
    )
    assert selected == 3
    assert np.isclose(voxel[1, 0, 0], 1.0, atol=1e-6)


def test_mode2_dataset_reads_synthetic_frame_metadata(synthetic_raw_workspace):
    canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_alpha=0.5,
        interpolation_model="linear_blend",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    build_manifests(
        canonical_root=synthetic_raw_workspace["canonical_root"],
        manifests_root=synthetic_raw_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    manifest_path = synthetic_raw_workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl"
    dataset = Mode2Dataset(
        str(manifest_path),
        canonical_root=str(synthetic_raw_workspace["canonical_root"]),
        event_builder={"policy": "fixed_count", "event_count_target": 4},
        resize_policy="facet_square_direct",
        use_cache=False,
    )

    sample = dataset[0]
    assert tuple(sample["frame"].shape) == (1, 256, 256)
    assert tuple(sample["event"].shape) == (2, 256, 256)
    assert sample["meta"]["data_mode"] == "mode2"
    assert sample["meta"]["frame_source"] == "interpolated"
    assert sample["meta"]["synthetic_frame_flag"] is True
    assert sample["meta"]["interp_source_pair"] == ["000000_1000.png", "000001_2000.png"]
    assert sample["meta"]["interp_alpha"] == 0.5
    assert sample["meta"]["interp_model"] == "linear_blend"


def test_mode2_dataset_can_average_source_pair_events(synthetic_raw_workspace):
    canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_fixed_insert=1,
        interpolation_model="linear_blend",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    build_manifests(
        canonical_root=synthetic_raw_workspace["canonical_root"],
        manifests_root=synthetic_raw_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    manifest_path = synthetic_raw_workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl"
    dataset = Mode2Dataset(
        str(manifest_path),
        canonical_root=str(synthetic_raw_workspace["canonical_root"]),
        event_builder={
            "policy": "fixed_count",
            "event_count_target": 4,
            "generation_strategy": "source_pair_average",
            "average_weighting": "mean",
        },
        resize_policy="facet_square_direct",
        use_cache=False,
    )

    row = dataset.rows[0]
    frame_sensor = np.asarray(Image.open(dataset._resolve_path(row["frame_path"])).convert("L"))
    sensor_size_wh = tuple(int(v) for v in row.get("sensor_size_wh", [frame_sensor.shape[1], frame_sensor.shape[0]]))
    roi_xywh = tuple(_resolve_xywh(row, "roi_xywh", default=[0, 0, sensor_size_wh[0], sensor_size_wh[1]]))
    transform = _build_transform(
        resize_policy="facet_square_direct",
        source_shape_hw=(sensor_size_wh[1], sensor_size_wh[0]),
        roi_xywh=roi_xywh,
        target_size_wh=dataset.input_size,
    )
    source_pair = dataset._resolve_source_pair_timestamps(row)
    assert source_pair == (1000, 2000)

    prev_event, prev_count = dataset._build_transformed_event_from_window(
        events_path=row["events_npz"],
        transform=transform,
        sensor_size_wh=sensor_size_wh,
        event_window=dataset._resolve_anchor_event_window(source_pair[0]),
        end_timestamp_us=source_pair[0],
    )
    next_event, next_count = dataset._build_transformed_event_from_window(
        events_path=row["events_npz"],
        transform=transform,
        sensor_size_wh=sensor_size_wh,
        event_window=dataset._resolve_anchor_event_window(source_pair[1]),
        end_timestamp_us=source_pair[1],
    )
    expected_event = dataset._normalize_event_input(0.5 * prev_event + 0.5 * next_event)

    sample = dataset[0]
    assert np.allclose(sample["event"].numpy(), expected_event, atol=1e-6)
    assert sample["meta"]["event_generation_strategy"] == "source_pair_average"
    assert sample["meta"]["event_generation_source_timestamps"] == [1000, 2000]
    assert sample["meta"]["selected_event_count"] == int(round(0.5 * float(prev_count) + 0.5 * float(next_count)))


def test_mode2_dataset_source_pair_average_alpha_weighting_uses_interp_alpha(synthetic_raw_workspace):
    canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_fixed_insert=3,
        interpolation_model="linear_blend",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    build_manifests(
        canonical_root=synthetic_raw_workspace["canonical_root"],
        manifests_root=synthetic_raw_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    manifest_path = synthetic_raw_workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl"
    dataset = Mode2Dataset(
        str(manifest_path),
        canonical_root=str(synthetic_raw_workspace["canonical_root"]),
        event_builder={
            "policy": "fixed_count",
            "event_count_target": 4,
            "generation_strategy": "source_pair_average",
            "average_weighting": "alpha",
        },
        resize_policy="facet_square_direct",
        use_cache=False,
    )

    row = dataset.rows[0]
    assert np.isclose(float(row["interp_alpha"]), 0.25)
    frame_sensor = np.asarray(Image.open(dataset._resolve_path(row["frame_path"])).convert("L"))
    sensor_size_wh = tuple(int(v) for v in row.get("sensor_size_wh", [frame_sensor.shape[1], frame_sensor.shape[0]]))
    roi_xywh = tuple(_resolve_xywh(row, "roi_xywh", default=[0, 0, sensor_size_wh[0], sensor_size_wh[1]]))
    transform = _build_transform(
        resize_policy="facet_square_direct",
        source_shape_hw=(sensor_size_wh[1], sensor_size_wh[0]),
        roi_xywh=roi_xywh,
        target_size_wh=dataset.input_size,
    )
    source_pair = dataset._resolve_source_pair_timestamps(row)
    assert source_pair == (1000, 2000)
    prev_event, prev_count = dataset._build_transformed_event_from_window(
        events_path=row["events_npz"],
        transform=transform,
        sensor_size_wh=sensor_size_wh,
        event_window=dataset._resolve_anchor_event_window(source_pair[0]),
        end_timestamp_us=source_pair[0],
    )
    next_event, next_count = dataset._build_transformed_event_from_window(
        events_path=row["events_npz"],
        transform=transform,
        sensor_size_wh=sensor_size_wh,
        event_window=dataset._resolve_anchor_event_window(source_pair[1]),
        end_timestamp_us=source_pair[1],
    )
    alpha = float(row["interp_alpha"])
    expected_event = dataset._normalize_event_input((1.0 - alpha) * prev_event + alpha * next_event)

    sample = dataset[0]
    assert np.allclose(sample["event"].numpy(), expected_event, atol=1e-6)
    assert sample["meta"]["event_average_weighting"] == "alpha"
    assert sample["meta"]["selected_event_count"] == int(round((1.0 - alpha) * float(prev_count) + alpha * float(next_count)))


def test_mode2_dataset_source_pair_average_pair_local_scope_uses_pair_interval_and_alpha(synthetic_raw_workspace):
    canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_fixed_insert=3,
        interpolation_model="linear_blend",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    build_manifests(
        canonical_root=synthetic_raw_workspace["canonical_root"],
        manifests_root=synthetic_raw_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    manifest_path = synthetic_raw_workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl"
    dataset = Mode2Dataset(
        str(manifest_path),
        canonical_root=str(synthetic_raw_workspace["canonical_root"]),
        event_builder={
            "policy": "fixed_count",
            "event_count_target": 4,
            "generation_strategy": "source_pair_average",
            "source_pair_scope": "pair_local",
            "average_weighting": "alpha",
        },
        resize_policy="facet_square_direct",
        use_cache=False,
    )

    row = dataset.rows[0]
    assert np.isclose(float(row["interp_alpha"]), 0.25)
    frame_sensor = np.asarray(Image.open(dataset._resolve_path(row["frame_path"])).convert("L"))
    sensor_size_wh = tuple(int(v) for v in row.get("sensor_size_wh", [frame_sensor.shape[1], frame_sensor.shape[0]]))
    roi_xywh = tuple(_resolve_xywh(row, "roi_xywh", default=[0, 0, sensor_size_wh[0], sensor_size_wh[1]]))
    transform = _build_transform(
        resize_policy="facet_square_direct",
        source_shape_hw=(sensor_size_wh[1], sensor_size_wh[0]),
        roi_xywh=roi_xywh,
        target_size_wh=dataset.input_size,
    )
    source_pair = dataset._resolve_source_pair_timestamps(row)
    assert source_pair == (1000, 2000)
    assert dataset._resolve_source_pair_scope() == "pair_local"

    events = dataset._load_events(row["events_npz"])
    timestamps = np.asarray(events["t"], dtype=np.int64)
    split_timestamp_us = int(row["sample_timestamp_us"])
    prev_event, prev_count = dataset._build_transformed_event_from_selection(
        events=events,
        transform=transform,
        sensor_size_wh=sensor_size_wh,
        selection=dataset._resolve_timestamp_selection(
            timestamps,
            start_timestamp_us=source_pair[0],
            end_timestamp_us=split_timestamp_us,
            include_start=True,
        ),
        start_timestamp_us=source_pair[0],
        end_timestamp_us=split_timestamp_us,
    )
    next_event, next_count = dataset._build_transformed_event_from_selection(
        events=events,
        transform=transform,
        sensor_size_wh=sensor_size_wh,
        selection=dataset._resolve_timestamp_selection(
            timestamps,
            start_timestamp_us=split_timestamp_us,
            end_timestamp_us=source_pair[1],
            include_start=False,
        ),
        start_timestamp_us=split_timestamp_us,
        end_timestamp_us=source_pair[1],
    )
    pair_selection = dataset._resolve_timestamp_selection(
        timestamps,
        start_timestamp_us=source_pair[0],
        end_timestamp_us=source_pair[1],
        include_start=True,
    )
    alpha = float(row["interp_alpha"])
    expected_event = dataset._normalize_event_input((1.0 - alpha) * prev_event + alpha * next_event)
    expected_pair_count = int(pair_selection.stop) - int(pair_selection.start)

    sample = dataset[0]
    assert np.allclose(sample["event"].numpy(), expected_event, atol=1e-6)
    assert sample["meta"]["event_generation_strategy"] == "source_pair_average"
    assert sample["meta"]["event_generation_source_pair_scope"] == "pair_local"
    assert sample["meta"]["event_average_weighting"] == "alpha"
    assert sample["meta"]["event_generation_source_timestamps"] == [1000, 2000]
    assert prev_count == 1
    assert next_count == 6
    assert sample["meta"]["selected_event_count"] == expected_pair_count == 7


def test_mode2_dataset_raw_window_strategy_uses_synthetic_window_directly(synthetic_raw_workspace):
    canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_fixed_insert=1,
        interpolation_model="linear_blend",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    build_manifests(
        canonical_root=synthetic_raw_workspace["canonical_root"],
        manifests_root=synthetic_raw_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    manifest_path = synthetic_raw_workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl"
    dataset = Mode2Dataset(
        str(manifest_path),
        canonical_root=str(synthetic_raw_workspace["canonical_root"]),
        event_builder={
            "policy": "fixed_count",
            "event_count_target": 4,
            "generation_strategy": "raw_window",
        },
        resize_policy="facet_square_direct",
        use_cache=False,
    )

    row = dataset.rows[0]
    frame_sensor = np.asarray(Image.open(dataset._resolve_path(row["frame_path"])).convert("L"))
    sensor_size_wh = tuple(int(v) for v in row.get("sensor_size_wh", [frame_sensor.shape[1], frame_sensor.shape[0]]))
    roi_xywh = tuple(_resolve_xywh(row, "roi_xywh", default=[0, 0, sensor_size_wh[0], sensor_size_wh[1]]))
    transform = _build_transform(
        resize_policy="facet_square_direct",
        source_shape_hw=(sensor_size_wh[1], sensor_size_wh[0]),
        roi_xywh=roi_xywh,
        target_size_wh=dataset.input_size,
    )
    synthetic_window = dataset._resolve_event_window(row, end_timestamp_us=int(row["synthetic_event_window"]["end_timestamp_us"]))
    expected_event, expected_count = dataset._build_transformed_event_from_window(
        events_path=row["events_npz"],
        transform=transform,
        sensor_size_wh=sensor_size_wh,
        event_window=synthetic_window,
        end_timestamp_us=int(synthetic_window["end_timestamp_us"]),
    )
    expected_event = dataset._normalize_event_input(expected_event)

    sample = dataset[0]
    assert np.allclose(sample["event"].numpy(), expected_event, atol=1e-6)
    assert sample["meta"]["event_generation_strategy"] == "raw_window"
    assert sample["meta"]["event_generation_source_timestamps"] is None
    assert sample["meta"]["selected_event_count"] == expected_count
