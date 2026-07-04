from __future__ import annotations

from pathlib import Path

from src.data.dataset import EVEyeHBTXRDataset, Mode2Dataset
from src.preprocess.build_manifests import build_manifests
from src.preprocess.target_fps_canonical import canonicalize_target_fps_dataset

from tests.test_target_fps_canonical import _build_target_fps_canonical_workspace


def _build_target_fps_dataset_runtime_workspace(tmp_path: Path, *, data_mode: str) -> tuple[dict[str, Path], Path]:
    workspace = _build_target_fps_canonical_workspace(tmp_path)
    manifests_root = workspace["project_root"] / "manifests"
    canonicalize_target_fps_dataset(
        target_root=workspace["target_root"],
        target_fps=2000.0,
        canonical_root=workspace["canonical_workspace_root"],
        data_mode=data_mode,
        overwrite=True,
    )
    build_manifests(
        canonical_root=workspace["canonical_workspace_root"],
        manifests_root=manifests_root,
        data_mode=data_mode,
    )
    manifest_name = "manifest2" if data_mode == "mode2" else "manifest1"
    return workspace, manifests_root / manifest_name / "train_manifest.jsonl"


def test_target_fps_mode1_dataset_reads_frames_and_events_from_session_store(tmp_path: Path):
    workspace, manifest_path = _build_target_fps_dataset_runtime_workspace(tmp_path, data_mode="mode1")
    dataset = EVEyeHBTXRDataset(
        str(manifest_path),
        canonical_root=str(workspace["canonical_workspace_root"]),
        resize_policy="facet_square_direct",
        use_cache=False,
    )

    sample = dataset[1]

    assert tuple(sample["frame"].shape) == (1, 256, 256)
    assert tuple(sample["event"].shape) == (2, 256, 256)
    assert sample["meta"]["data_mode"] == "mode1"
    assert sample["meta"]["frame_source"] == "target_fps_grid"
    assert sample["meta"]["frame_load_source"] == "session_store"
    assert sample["meta"]["frame_path"] is None
    assert sample["meta"]["session_store_path"].endswith("session_arrays.npz")
    assert sample["meta"]["event_generation_strategy"] == "target_fps_session_store"
    assert sample["meta"]["selected_event_count"] == 1


def test_target_fps_mode2_dataset_reads_interpolation_metadata_from_session_store_manifest(tmp_path: Path):
    workspace, manifest_path = _build_target_fps_dataset_runtime_workspace(tmp_path, data_mode="mode2")
    dataset = Mode2Dataset(
        str(manifest_path),
        canonical_root=str(workspace["canonical_workspace_root"]),
        resize_policy="facet_square_direct",
        use_cache=False,
    )

    sample = dataset[1]

    assert tuple(sample["frame"].shape) == (1, 256, 256)
    assert tuple(sample["event"].shape) == (2, 256, 256)
    assert sample["meta"]["data_mode"] == "mode2"
    assert sample["meta"]["frame_load_source"] == "session_store"
    assert sample["meta"]["event_generation_strategy"] == "target_fps_session_store"
    assert sample["meta"]["selected_event_count"] == 1
    assert sample["meta"]["synthetic_frame_flag"] is True
    assert sample["meta"]["interp_source_pair"] == ["000000_1000.png", "000001_41000.png"]
    assert sample["meta"]["interp_model"] == "linear_blend"
    assert sample["meta"]["interp_target_fps"] == 2000.0
