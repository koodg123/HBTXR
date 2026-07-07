from __future__ import annotations

from pathlib import Path

from hbtxr.preprocess.build_manifests import build_manifests
from hbtxr.preprocess.target_fps_build import build_target_fps_dataset
from hbtxr.preprocess.target_fps_canonical import canonicalize_target_fps_dataset
from hbtxr.utils.io import read_json, read_jsonl

from tests.test_target_fps_build_v3 import _build_target_fps_annotation_workspace
from tests.test_target_fps_canonical_v3 import _build_target_fps_canonical_workspace


def test_target_fps_manifest_mode1_uses_session_store_rows(tmp_path: Path):
    workspace = _build_target_fps_canonical_workspace(tmp_path)
    manifests_root = workspace["project_root"] / "manifests"
    canonicalize_target_fps_dataset(
        target_root=workspace["target_root"],
        target_fps=2000.0,
        canonical_root=workspace["canonical_workspace_root"],
        data_mode="mode1",
        overwrite=True,
    )

    summary = build_manifests(
        canonical_root=workspace["canonical_workspace_root"],
        manifests_root=manifests_root,
        data_mode="mode1",
    )

    train_rows = read_jsonl(manifests_root / "manifest1" / "train_manifest.jsonl")
    skipped_rows = read_jsonl(manifests_root / "manifest1" / "skipped_sessions.jsonl")
    manifest_summary = read_json(manifests_root / "manifest1" / "manifest_summary.json")

    assert summary["counts"]["train"] == 81
    assert summary["n_skipped_sessions"] == 0
    assert manifest_summary["n_skipped_sessions"] == 0
    assert skipped_rows == []
    assert train_rows[0]["data_mode"] == "mode1"
    assert train_rows[0]["frame_path"] is None
    assert train_rows[0]["events_npz"] is None
    assert train_rows[0]["session_store_path"].endswith("session_arrays.npz")
    assert train_rows[0]["session_store_format"] == "npz"
    assert train_rows[0]["frame_index"] == 0
    assert train_rows[0]["event_index_range"] == [0, 2]
    assert train_rows[0]["annotation_status"] == train_rows[0]["label_status"]
    assert train_rows[0]["eye_state_flag"] == "open"


def test_target_fps_manifest_mode2_preserves_interpolation_metadata(tmp_path: Path):
    workspace = _build_target_fps_canonical_workspace(tmp_path)
    manifests_root = workspace["project_root"] / "manifests"
    canonicalize_target_fps_dataset(
        target_root=workspace["target_root"],
        target_fps=2000.0,
        canonical_root=workspace["canonical_workspace_root"],
        data_mode="mode2",
        overwrite=True,
    )

    summary = build_manifests(
        canonical_root=workspace["canonical_workspace_root"],
        manifests_root=manifests_root,
        data_mode="mode2",
    )

    train_rows = read_jsonl(manifests_root / "manifest2" / "train_manifest.jsonl")

    assert summary["counts"]["train"] == 81
    assert train_rows[0]["data_mode"] == "mode2"
    assert train_rows[1]["synthetic_frame_flag"] is True
    assert train_rows[1]["interp_source_pair"] == ["000000_1000.png", "000001_41000.png"]
    assert train_rows[1]["interp_model"] == "linear_blend"
    assert train_rows[1]["interp_target_fps"] == 2000.0
    assert train_rows[1]["synthetic_event_window"]["event_count"] == 1
    assert train_rows[1]["session_store_path"].endswith("session_arrays.npz")


def test_target_fps_manifest_preserves_skips_and_annotation_failures(tmp_path: Path):
    workspace = _build_target_fps_annotation_workspace(
        tmp_path,
        with_manual_csv=True,
        with_bbox_store=False,
    )
    manifests_root = workspace["project_root"] / "manifests"
    canonical_workspace_root = workspace["project_root"] / "canonical_workspace"

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
    canonicalize_target_fps_dataset(
        target_root=workspace["target_root"],
        target_fps=2000.0,
        canonical_root=canonical_workspace_root,
        data_mode="mode1",
        overwrite=True,
    )

    summary = build_manifests(
        canonical_root=canonical_workspace_root,
        manifests_root=manifests_root,
        data_mode="mode1",
    )

    skipped_rows = read_jsonl(manifests_root / "manifest1" / "skipped_sessions.jsonl")
    failure_rows = read_jsonl(manifests_root / "manifest1" / "annotation_failures.jsonl")

    assert summary["n_skipped_sessions"] == 1
    assert summary["n_annotation_failures"] == 81
    assert skipped_rows[0]["skip_reason"] == "event_file_missing"
    assert failure_rows[0]["annotation_status"] == "pending_annotation_sources"
    assert failure_rows[0]["annotation_reason"] == "groundedsam_eye_roi_missing"
    assert failure_rows[0]["split"] == "train"
