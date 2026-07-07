from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hbtxr.data.dataset import Mode2Dataset
from hbtxr.preprocess.build_manifests import build_manifests
from hbtxr.preprocess.canonicalize import canonicalize_dataset
from hbtxr.preprocess.event_generation import generate_event_packets, resolve_event_generation_backend
from hbtxr.preprocess.interpolation import (
    build_alpha_schedule,
    compute_insert_count,
    interpolate_pair,
    resolve_timelens_checkpoint,
    resolve_interpolation_backend,
    unwrap_timelens_checkpoint_state_dict,
)
from hbtxr.utils.io import read_json, read_jsonl, write_jsonl
from hbtxr.utils.state6 import xywht_to_xyabuv


def _build_target_fps_raw_workspace(tmp_path: Path) -> dict[str, Path]:
    project_root = tmp_path / "project"
    raw_root = project_root / "raw"
    canonical_root = project_root / "canonical"
    manifests_root = project_root / "manifests"
    annotation_root = project_root / "annotations"
    groundedsam_root = project_root / "groundedsam_repo"
    session_dir = raw_root / "user01" / "left" / "session_1_0_1"
    frames_dir = session_dir / "frames"
    events_dir = session_dir / "events"

    frames_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    canonical_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)
    groundedsam_root.mkdir(parents=True, exist_ok=True)

    sensor_size_wh = [346, 240]
    timestamps = [1000, 41000]
    ellipses = [
        [160.0, 120.0, 40.0, 50.0, 0.10],
        [168.0, 124.0, 42.0, 52.0, 0.12],
    ]
    frame_names = [f"{idx:06d}_{ts}.png" for idx, ts in enumerate(timestamps)]

    for idx, (frame_name, ellipse) in enumerate(zip(frame_names, ellipses)):
        canvas = Image.new("L", tuple(sensor_size_wh), color=25 + idx * 10)
        draw = ImageDraw.Draw(canvas)
        bbox = (
            ellipse[0] - ellipse[2] / 2.0,
            ellipse[1] - ellipse[3] / 2.0,
            ellipse[0] + ellipse[2] / 2.0,
            ellipse[1] + ellipse[3] / 2.0,
        )
        draw.ellipse(bbox, fill=180)
        canvas.save(frames_dir / frame_name)

    event_times = np.arange(1000, 41001, 500, dtype=np.int64)
    np.savez_compressed(
        events_dir / "events.npz",
        t=event_times,
        x=np.full_like(event_times, 160, dtype=np.int16),
        y=np.full_like(event_times, 120, dtype=np.int16),
        p=np.where(np.arange(len(event_times)) % 2 == 0, 1, -1).astype(np.int8),
    )

    annotation_session_dir = annotation_root / "sessions" / "user01" / "left" / "session_101"
    annotation_session_dir.mkdir(parents=True, exist_ok=True)
    annotation_rows = []
    for idx, (frame_name, ts, ellipse) in enumerate(zip(frame_names, timestamps, ellipses)):
        annotation_rows.append(
            {
                "ann_id": f"user01__left__session_101__{idx:06d}",
                "frame_filename": frame_name,
                "timestamp_us": ts,
                "eye_region_bbox_xywh_sensor": [80.0, 40.0, 160.0, 160.0],
                "pupil_mask_path": None,
                "mask_path": None,
                "pupil_region_bbox_xywh_sensor": [ellipse[0] - ellipse[2] / 2.0, ellipse[1] - ellipse[3] / 2.0, ellipse[2], ellipse[3]],
                "pupil_ellipse_xywht_sensor": ellipse,
                "ellipse_sensor_xywht": ellipse,
                "state_xyabuv": xywht_to_xyabuv(np.asarray(ellipse, dtype=np.float32)).tolist(),
                "annotation_source": "gsa_reviewed",
                "annotation_quality": 0.95,
                "closed_eye_flag": False,
                "mask_valid": True,
            }
        )
    write_jsonl(annotation_rows, annotation_session_dir / "frame_annotations.jsonl")

    return {
        "project_root": project_root,
        "raw_root": raw_root,
        "canonical_root": canonical_root,
        "manifests_root": manifests_root,
        "annotation_root": annotation_root,
        "groundedsam_root": groundedsam_root,
    }


def test_compute_insert_count_matches_target_fps_examples():
    assert compute_insert_count(timestamp0_us=1000, timestamp1_us=41000, target_fps=2000.0) == 79
    assert compute_insert_count(timestamp0_us=1000, timestamp1_us=41000, target_fps=5000.0) == 199
    assert compute_insert_count(timestamp0_us=1000, timestamp1_us=41000, fixed_insert=12, target_fps=2000.0) == 12


def test_build_alpha_schedule_for_multi_frame_interpolation():
    schedule = build_alpha_schedule(79)
    assert len(schedule) == 79
    assert np.isclose(schedule[0], 1.0 / 80.0)
    assert np.isclose(schedule[-1], 79.0 / 80.0)


def test_backend_resolvers_accept_extended_runtime_keys():
    assert resolve_interpolation_backend(interpolation_backend="timelens_xl") == "timelens_xl"
    assert resolve_event_generation_backend("v2e") == "v2e"


def test_timelens_backend_requires_checkpoint_when_not_provided(tmp_path: Path):
    events_path = tmp_path / "events.npz"
    np.savez_compressed(
        events_path,
        t=np.asarray([0, 1000], dtype=np.int64),
        x=np.asarray([1, 1], dtype=np.int16),
        y=np.asarray([1, 1], dtype=np.int16),
        p=np.asarray([1, -1], dtype=np.int8),
    )
    timelens_root = tmp_path / "timelens_root"
    timelens_root.mkdir(parents=True, exist_ok=True)
    with np.testing.assert_raises_regex(FileNotFoundError, "requires a checkpoint"):
        interpolate_pair(
            backend="timelens",
            frame0=np.zeros((4, 4), dtype=np.uint8),
            frame1=np.ones((4, 4), dtype=np.uint8),
            alpha=0.5,
            timestamp0_us=0,
            timestamp1_us=1000,
            events_path=events_path,
            timelens_root=timelens_root,
        )


def test_timelens_backend_uses_runtime_adapter_when_available(tmp_path: Path, monkeypatch):
    from hbtxr.preprocess import interpolation as interpolation_module

    events_path = tmp_path / "events.npz"
    np.savez_compressed(
        events_path,
        t=np.asarray([0, 1000], dtype=np.int64),
        x=np.asarray([1, 1], dtype=np.int16),
        y=np.asarray([1, 1], dtype=np.int16),
        p=np.asarray([1, -1], dtype=np.int8),
    )
    timelens_root = tmp_path / "timelens_root"
    checkpoint_path = timelens_root / "refined_model" / "attention.bin"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"fake")

    class _FakeRuntime:
        def interpolate(self, **kwargs):
            return np.full((4, 4), 123, dtype=np.uint8)

    monkeypatch.setattr(interpolation_module, "_get_timelens_runtime", lambda **kwargs: _FakeRuntime())
    output = interpolate_pair(
        backend="timelens",
        frame0=np.zeros((4, 4), dtype=np.uint8),
        frame1=np.ones((4, 4), dtype=np.uint8),
        alpha=0.5,
        timestamp0_us=0,
        timestamp1_us=1000,
        events_path=events_path,
        timelens_root=timelens_root,
    )
    assert output.shape == (4, 4)
    assert int(output[0, 0]) == 123


def test_timelens_xl_backend_uses_runtime_adapter_when_available(tmp_path: Path, monkeypatch):
    from hbtxr.preprocess import interpolation as interpolation_module

    events_path = tmp_path / "events.npz"
    np.savez_compressed(
        events_path,
        t=np.asarray([0, 1000], dtype=np.int64),
        x=np.asarray([1, 1], dtype=np.int16),
        y=np.asarray([1, 1], dtype=np.int16),
        p=np.asarray([1, -1], dtype=np.int8),
    )
    timelens_xl_root = tmp_path / "timelens_xl_root"
    checkpoint_path = timelens_xl_root / "models" / "timelens" / "checkpoint.bin"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"fake")

    class _FakeRuntime:
        def interpolate(self, **kwargs):
            return np.full((4, 4), 231, dtype=np.uint8)

    monkeypatch.setattr(interpolation_module, "_get_timelens_runtime", lambda **kwargs: _FakeRuntime())
    output = interpolate_pair(
        backend="timelens_xl",
        frame0=np.zeros((4, 4), dtype=np.uint8),
        frame1=np.ones((4, 4), dtype=np.uint8),
        alpha=0.5,
        timestamp0_us=0,
        timestamp1_us=1000,
        events_path=events_path,
        timelens_xl_root=timelens_xl_root,
    )
    assert output.shape == (4, 4)
    assert int(output[0, 0]) == 231


def test_timelens_xl_backend_accepts_raw_events_txt_when_runtime_is_available(tmp_path: Path, monkeypatch):
    from hbtxr.preprocess import interpolation as interpolation_module

    raw_root = tmp_path / "raw" / "user01" / "left" / "session_1_0_1" / "events"
    raw_root.mkdir(parents=True, exist_ok=True)
    events_path = raw_root / "events.txt"
    events_path.write_text("1000 1 2 1\n1200 3 4 -1\n", encoding="utf-8")

    timelens_xl_root = tmp_path / "timelens_xl_root"
    checkpoint_path = timelens_xl_root / "models" / "timelens" / "checkpoint.bin"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"fake")

    class _FakeRuntime:
        def interpolate(self, **kwargs):
            return np.full((4, 4), 177, dtype=np.uint8)

    monkeypatch.setattr(interpolation_module, "_get_timelens_runtime", lambda **kwargs: _FakeRuntime())
    output = interpolate_pair(
        backend="timelens_xl",
        frame0=np.zeros((4, 4), dtype=np.uint8),
        frame1=np.ones((4, 4), dtype=np.uint8),
        alpha=0.5,
        timestamp0_us=1000,
        timestamp1_us=2000,
        events_path=events_path,
        timelens_xl_root=timelens_xl_root,
    )
    assert output.shape == (4, 4)
    assert int(output[0, 0]) == 177


def test_resolve_timelens_xl_checkpoint_prefers_shared_workspace_checkpoint(tmp_path: Path, monkeypatch):
    from hbtxr.preprocess import interpolation as interpolation_module

    repo_root = tmp_path / "Third" / "FI"
    repo_root.mkdir(parents=True, exist_ok=True)
    shared_root = tmp_path / "workspace" / "third_party_checkpoints"
    shared_checkpoint = shared_root / "FI" / "TimeLens-XL" / "TimeLens_1.pt"
    shared_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shared_checkpoint.write_bytes(b"fake-shared")

    monkeypatch.setattr(interpolation_module, "_default_third_party_checkpoint_root", lambda: shared_root)

    resolved = resolve_timelens_checkpoint(
        backend="timelens_xl",
        timelens_root=repo_root,
        checkpoint_path=None,
    )

    assert resolved == shared_checkpoint.resolve()


def test_unwrap_timelens_checkpoint_state_dict_accepts_native_finetune_format():
    payload = {
        "model_state": {
            "net.encoder.weight": np.asarray([1.0], dtype=np.float32),
            "net.decoder.bias": np.asarray([2.0], dtype=np.float32),
        },
        "epoch": 1,
        "metrics": {"psnr": [42.0]},
    }

    state_dict = unwrap_timelens_checkpoint_state_dict(payload)

    assert sorted(state_dict.keys()) == ["decoder.bias", "encoder.weight"]
    assert float(state_dict["encoder.weight"][0]) == 1.0
    assert float(state_dict["decoder.bias"][0]) == 2.0


def test_v2e_event_generation_backend_uses_runtime_adapter_when_available(monkeypatch):
    from hbtxr.preprocess import event_generation as event_generation_module

    captured_kwargs = {}

    class _FakeRuntime:
        def generate_sequence_events(self, **kwargs):
            return (
                {
                    "t": np.asarray([10, 20], dtype=np.int64),
                    "x": np.asarray([1, 2], dtype=np.int16),
                    "y": np.asarray([3, 4], dtype=np.int16),
                    "p": np.asarray([1, -1], dtype=np.int8),
                },
                np.asarray([[0, 1], [1, 2]], dtype=np.int64),
            )

    def _build_runtime(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeRuntime()

    monkeypatch.setattr(event_generation_module, "build_event_generation_runtime", _build_runtime)
    events, packets = generate_event_packets(
        event_generation_backend="v2e",
        frames=np.zeros((2, 4, 4), dtype=np.uint8),
        frame_timestamps_us=np.asarray([1000, 2000], dtype=np.int64),
        v2e_root=Path("/tmp/unused"),
        v2e_kwargs={"pos_thres": 0.35, "neg_thres": 0.35},
    )
    assert events["t"].tolist() == [10, 20]
    assert packets.tolist() == [[0, 1], [1, 2]]
    assert captured_kwargs["v2e_kwargs"] == {"pos_thres": 0.35, "neg_thres": 0.35}


def test_mode2_target_fps_generates_expected_synthetic_count(tmp_path: Path):
    workspace = _build_target_fps_raw_workspace(tmp_path)
    canonical_summary = canonicalize_dataset(
        raw_root=workspace["raw_root"],
        canonical_root=workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_alpha=0.5,
        interpolation_model="linear_blend",
        interpolation_backend="linear_blend",
        interpolation_target_fps=2000.0,
        interpolation_count_policy="round",
        interpolation_max_insert=255,
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )

    assert canonical_summary["n_sessions_ok"] == 1
    assert canonical_summary["interpolation_target_fps"] == 2000.0
    session_dir = workspace["canonical_root"] / "canonical2" / "sessions" / "user01" / "left" / "session_101"
    annotation_rows = read_jsonl(session_dir / "labels" / "frame_annotations.jsonl")
    assert len(annotation_rows) == 79
    assert annotation_rows[0]["interp_rank"] == 1
    assert annotation_rows[-1]["interp_rank"] == 79
    assert annotation_rows[0]["interp_insert_count"] == 79
    assert annotation_rows[0]["interp_target_fps"] == 2000.0
    assert annotation_rows[0]["synthetic_event_window"]["start_timestamp_us"] == 1000
    assert annotation_rows[0]["synthetic_event_window"]["end_timestamp_us"] == 1500
    assert annotation_rows[-1]["synthetic_event_window"]["end_timestamp_us"] == 40500

    interpolation_summary = read_json(session_dir / "labels" / "interpolation_index.json")
    assert interpolation_summary["n_synthetic_frames"] == 79
    assert interpolation_summary["interp_target_fps"] == 2000.0
    assert interpolation_summary["items"][0]["interp_alpha"] == annotation_rows[0]["interp_alpha"]
    assert interpolation_summary["items"][-1]["interp_rank"] == 79

    manifest_summary = build_manifests(
        canonical_root=workspace["canonical_root"],
        manifests_root=workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    assert manifest_summary["counts"]["train"] == 79
    manifest_rows = read_jsonl(workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl")
    assert manifest_rows[0]["interp_target_fps"] == 2000.0
    assert manifest_rows[0]["interp_insert_count"] == 79

    dataset = Mode2Dataset(
        str(workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl"),
        canonical_root=str(workspace["canonical_root"]),
        event_builder={"policy": "fixed_count", "event_count_target": 4},
        resize_policy="facet_square_direct",
        use_cache=False,
    )
    sample = dataset[0]
    assert sample["meta"]["interp_target_fps"] == 2000.0
    assert sample["meta"]["interp_insert_count"] == 79


def test_mode2_timelens_backend_wires_into_canonicalize(tmp_path: Path, monkeypatch):
    from hbtxr.preprocess import interpolation as interpolation_module

    workspace = _build_target_fps_raw_workspace(tmp_path)
    checkpoint_path = workspace["groundedsam_root"] / "refined_model" / "attention.bin"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"fake")

    class _FakeRuntime:
        def interpolate(self, **kwargs):
            frame0 = kwargs["frame0"]
            frame1 = kwargs["frame1"]
            return ((frame0.astype(np.float32) + frame1.astype(np.float32)) * 0.5).round().astype(np.uint8)

    monkeypatch.setattr(interpolation_module, "_get_timelens_runtime", lambda **kwargs: _FakeRuntime())
    canonicalize_dataset(
        raw_root=workspace["raw_root"],
        canonical_root=workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_alpha=0.5,
        interpolation_model="timelens",
        interpolation_backend="timelens",
        interpolation_fixed_insert=1,
        interpolation_timelens_root=workspace["groundedsam_root"],
        interpolation_timelens_device="cpu",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    session_dir = workspace["canonical_root"] / "canonical2" / "sessions" / "user01" / "left" / "session_101"
    annotation_rows = read_jsonl(session_dir / "labels" / "frame_annotations.jsonl")
    assert len(annotation_rows) == 1
    assert annotation_rows[0]["interp_model"] == "timelens"
