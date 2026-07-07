from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from hbtxr.preprocess.target_fps_build import build_target_fps_dataset, resolve_target_fps_root
from hbtxr.preprocess.target_fps_canonical import canonicalize_target_fps_dataset
from hbtxr.utils.io import read_json, read_jsonl, write_jsonl
from hbtxr.utils.paths import resolve_canonical_dataset_root

from tests.test_target_fps_build_v3 import _build_target_fps_annotation_workspace


def _build_target_fps_canonical_workspace(tmp_path: Path) -> dict[str, Path]:
    project_root = tmp_path / "project"
    raw_root = project_root / "raw"
    target_root = project_root / "target_data"
    annotation_root = project_root / "annotations"
    canonical_workspace_root = project_root / "canonical_workspace"

    session_dir = raw_root / "user01" / "left" / "session_1_0_1"
    frames_dir = session_dir / "frames"
    events_dir = session_dir / "events"
    bbox_store_dir = annotation_root / "eye_region_bbox_only" / "sessions" / "user01" / "left" / "session_101"

    frames_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    bbox_store_dir.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)
    canonical_workspace_root.mkdir(parents=True, exist_ok=True)

    for idx, ts in enumerate((1000, 41000)):
        Image.new("L", (346, 240), color=40 + idx * 20).save(frames_dir / f"{idx:06d}_{ts}.png")

    np.savez_compressed(
        events_dir / "events.npz",
        t=np.asarray([900, 1000, 1500, 2000, 40500, 41000], dtype=np.int64),
        x=np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int16),
        y=np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int16),
        p=np.asarray([1, -1, 1, -1, 1, -1], dtype=np.int8),
    )

    csv_path = session_dir / "user_1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "region_count", "region_id", "region_shape_attributes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "000000_1000.png",
                "region_count": 1,
                "region_id": 0,
                "region_shape_attributes": '{"name":"ellipse","cx":173.0,"cy":120.0,"rx":21.0,"ry":20.0,"theta":0.0}',
            }
        )
        writer.writerow(
            {
                "filename": "000001_41000.png",
                "region_count": 1,
                "region_id": 0,
                "region_shape_attributes": '{"name":"ellipse","cx":175.0,"cy":122.0,"rx":19.0,"ry":18.0,"theta":0.1}',
            }
        )

    write_jsonl(
        [
            {
                "frame_filename": "000000_1000.png",
                "frame_idx": 0,
                "timestamp_us": 1000,
                "eye_region_bbox_xywh_sensor": [80.0, 40.0, 160.0, 120.0],
            },
            {
                "frame_filename": "000001_41000.png",
                "frame_idx": 1,
                "timestamp_us": 41000,
                "eye_region_bbox_xywh_sensor": [84.0, 44.0, 158.0, 118.0],
            },
        ],
        bbox_store_dir / "eye_region_bboxes.jsonl",
    )

    build_target_fps_dataset(
        raw_root=raw_root,
        target_root=target_root,
        target_fps=2000.0,
        annotation_root=annotation_root,
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )

    return {
        "project_root": project_root,
        "raw_root": raw_root,
        "target_root": target_root,
        "annotation_root": annotation_root,
        "canonical_workspace_root": canonical_workspace_root,
    }


def test_target_fps_canonical_mode1_bridge_writes_session_store_references(tmp_path: Path):
    workspace = _build_target_fps_canonical_workspace(tmp_path)
    summary = canonicalize_target_fps_dataset(
        target_root=workspace["target_root"],
        target_fps=2000.0,
        canonical_root=workspace["canonical_workspace_root"],
        data_mode="mode1",
        overwrite=True,
    )

    canonical_root = resolve_canonical_dataset_root(workspace["canonical_workspace_root"], "canonical1", prefer_nested=True)
    indexes_root = canonical_root / "indexes"
    session_rows = read_jsonl(indexes_root / "sessions.jsonl")
    frame_rows = read_jsonl(canonical_root / "sessions" / "user01" / "left" / "session_101" / "labels" / "frame_annotations.jsonl")
    session_package = read_json(canonical_root / "sessions" / "user01" / "left" / "session_101" / "labels" / "session_package.json")

    assert summary["stage"] == "target_fps_canonicalized"
    assert summary["n_sessions_canonicalized"] == 1
    assert summary["n_rows_written"] == 81
    assert session_rows[0]["session_store_format"] == "npz"
    assert session_rows[0]["session_h5_path"] is None
    assert session_rows[0]["annotation_store_path"].endswith("frame_annotations.jsonl")
    assert session_rows[0]["frame_index_path"].endswith("frame_index.jsonl")
    assert session_package["session_store_path"].endswith("session_arrays.npz")
    assert session_package["session_store_format"] == "npz"
    assert frame_rows[0]["data_mode"] == "mode1"
    assert frame_rows[0]["canonical_name"] == "canonical1"
    assert frame_rows[0]["frame_source"] == "target_fps_grid"
    assert frame_rows[0]["frame_index"] == 0
    assert frame_rows[0]["annotation_status"] == frame_rows[0]["label_status"]
    assert frame_rows[0]["session_store_path"].endswith("session_arrays.npz")
    assert frame_rows[0]["session_h5_path"] is None
    assert frame_rows[0]["frame_path"] is None
    assert frame_rows[0]["event_index_range"] == [0, 2]


def test_target_fps_canonical_mode2_bridge_adds_interpolation_provenance(tmp_path: Path):
    workspace = _build_target_fps_canonical_workspace(tmp_path)
    summary = canonicalize_target_fps_dataset(
        target_root=workspace["target_root"],
        target_fps=2000.0,
        canonical_root=workspace["canonical_workspace_root"],
        data_mode="mode2",
        overwrite=True,
    )

    canonical_root = resolve_canonical_dataset_root(workspace["canonical_workspace_root"], "canonical2", prefer_nested=True)
    frame_rows = read_jsonl(canonical_root / "sessions" / "user01" / "left" / "session_101" / "labels" / "frame_annotations.jsonl")
    session_store_link = canonical_root / "sessions" / "user01" / "left" / "session_101" / "store" / "session_arrays.npz"

    assert summary["stage"] == "target_fps_canonicalized"
    assert summary["data_mode"] == "mode2"
    assert session_store_link.exists()
    assert frame_rows[0]["data_mode"] == "mode2"
    assert frame_rows[0]["synthetic_frame_flag"] is False
    assert frame_rows[1]["synthetic_frame_flag"] is True
    assert frame_rows[1]["interp_source_pair"] == ["000000_1000.png", "000001_41000.png"]
    assert frame_rows[1]["interp_source_pair_index"] == [0, 1]
    assert frame_rows[1]["interp_model"] == "linear_blend"
    assert frame_rows[1]["interp_target_fps"] == 2000.0
    assert frame_rows[1]["synthetic_event_window"]["event_count"] == 1


def test_target_fps_canonical_writes_annotation_failure_index(tmp_path: Path):
    workspace = _build_target_fps_annotation_workspace(
        tmp_path,
        with_manual_csv=True,
        with_bbox_store=False,
    )
    build_target_fps_dataset(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        annotation_root=workspace["annotation_root"],
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )

    canonical_workspace_root = workspace["project_root"] / "canonical_workspace"
    summary = canonicalize_target_fps_dataset(
        target_root=workspace["target_root"],
        target_fps=2000.0,
        canonical_root=canonical_workspace_root,
        data_mode="mode1",
        overwrite=True,
    )

    canonical_root = resolve_canonical_dataset_root(canonical_workspace_root, "canonical1", prefer_nested=True)
    failures = read_jsonl(canonical_root / "indexes" / "canonical_annotation_failures.jsonl")

    assert summary["n_annotation_failures"] == 81
    assert failures[0]["annotation_status"] == "pending_annotation_sources"
    assert failures[0]["annotation_reason"] == "groundedsam_eye_roi_missing"
