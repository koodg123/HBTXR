from __future__ import annotations

import pytest
import numpy as np

from src.preprocess import canonicalize as canonicalize_module
from src.preprocess.build_manifests import build_manifests
from src.preprocess.canonicalize import canonicalize_dataset
from src.preprocess.canonicalize_pipeline import CanonicalSessionProcessor
from src.preprocess.groundedsam_build import _resolve_runtime_device, _session_belongs_to_shard, parse_groundedsam_devices
from src.preprocess.groundedsam_pipeline import GroundedSamRuntimeFactory, GroundedSamSummaryWriter
from src.utils.io import read_json, read_jsonl


def test_build_manifests_v3_schema(synthetic_workspace):
    summary = build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        split_scheme="exgaze_with_val",
        resize_policy="facet_square_direct",
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        canonical_name="canonical1",
        manifest_name="manifest1",
    )
    assert summary["counts"]["train"] == 2
    rows = read_jsonl(synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl")
    assert rows[0]["resize_policy"] == "facet_square_direct"
    assert rows[0]["event_window"]["policy"] == "fixed_count"
    assert rows[0]["event_window"]["accumulation"] == "fast_causal_linear"
    assert rows[0]["event_window"]["fast_causal_limit"] == 25.0
    assert rows[0]["data_mode"] == "mode1"
    assert rows[0]["canonical_name"] == "canonical1"
    assert rows[0]["manifest_name"] == "manifest1"
    assert rows[0]["annotation_ref"]["ann_id"]
    assert rows[0]["prev_annotation_ref"]["ann_id"]
    manifest_summary = read_json(synthetic_workspace["manifests_root"] / "manifest1" / "manifest_summary.json")
    assert manifest_summary["event_count_target"] == 4
    assert manifest_summary["accumulation"] == "fast_causal_linear"
    assert manifest_summary["fast_causal_limit"] == 25.0


def test_groundedsam_device_list_uses_first_entry():
    torch = pytest.importorskip("torch")

    device, warning_message = _resolve_runtime_device(torch, "cuda:0,1")

    assert str(device) == "cuda:0"
    assert warning_message is not None
    assert "single torch device" in warning_message


def test_parse_groundedsam_devices_normalizes_multi_gpu_list():
    assert parse_groundedsam_devices("cuda:0,1") == ["cuda:0", "cuda:1"]
    assert parse_groundedsam_devices("0, 1") == ["cuda:0", "cuda:1"]


def test_groundedsam_two_shards_do_not_overlap():
    shard0 = {idx for idx in range(12) if _session_belongs_to_shard(idx, num_shards=2, shard_index=0)}
    shard1 = {idx for idx in range(12) if _session_belongs_to_shard(idx, num_shards=2, shard_index=1)}

    assert shard0 & shard1 == set()
    assert shard0 | shard1 == set(range(12))


class _DummyGroundedSamConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _DummyGroundedSamRuntime:
    def __init__(self, cfg):
        self.cfg = cfg


def test_groundedsam_runtime_factory_prefers_existing_model_zoo_checkpoints(tmp_path):
    repo_root = tmp_path / "groundedsam_repo"
    config_path = repo_root / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    dino_ckpt = repo_root / "model_zoo" / "groundingdino_swint_ogc.pth"
    sam_ckpt = repo_root / "model_zoo" / "sam_vit_h_4b8939.pth"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    dino_ckpt.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# dummy config\n", encoding="utf-8")
    dino_ckpt.write_bytes(b"")
    sam_ckpt.write_bytes(b"")

    factory = GroundedSamRuntimeFactory(
        config_cls=_DummyGroundedSamConfig,
        runtime_cls=_DummyGroundedSamRuntime,
        default_classes=("pupil", "iris", "eye"),
        validate_shard_spec=lambda num_shards, shard_index: (int(num_shards or 1), int(shard_index or 0)),
    )

    runtime = factory.build_runtime(groundedsam_root=repo_root)

    assert runtime.cfg.groundingdino_config == config_path.resolve()
    assert runtime.cfg.groundingdino_checkpoint == dino_ckpt.resolve()
    assert runtime.cfg.sam_checkpoint == sam_ckpt.resolve()
    assert runtime.cfg.classes == ("pupil", "iris", "eye")


def test_groundedsam_summary_writer_uses_shard_suffix_for_multi_device_outputs(tmp_path):
    writer = GroundedSamSummaryWriter(tmp_path)

    summary_path, sessions_path = writer.summary_paths(num_shards=2, shard_index=1)

    assert summary_path.name == "annotation_summary.shard_1_of_2.json"
    assert sessions_path.name == "annotation_sessions.shard_1_of_2.jsonl"


def test_canonical_session_processor_delegates_job_to_session_callable():
    captured = {}

    def _session_callable(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "session_key": kwargs["session_key"]}

    processor = CanonicalSessionProcessor(session_callable=_session_callable)
    result = processor.process_job({"session_key": "user01/left/session_101", "user_id": 1})

    assert result == {"ok": True, "session_key": "user01/left/session_101"}
    assert captured["user_id"] == 1


def test_groundedsam_device_invalid_value_raises_helpful_error():
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="Unsupported Grounded-SAM device value"):
        _resolve_runtime_device(torch, "cuda:abc")


def test_mode2_canonicalize_builds_interpolated_frames_and_manifest(synthetic_raw_workspace):
    canonical_summary = canonicalize_dataset(
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
    assert canonical_summary["n_sessions_ok"] == 1
    assert canonical_summary["data_mode"] == "mode2"
    assert canonical_summary["canonical_root"].replace("\\", "/").endswith("/canonical/canonical2")

    session_dir = synthetic_raw_workspace["canonical_root"] / "canonical2" / "sessions" / "user01" / "left" / "session_101"
    annotation_rows = read_jsonl(session_dir / "labels" / "frame_annotations.jsonl")
    assert len(annotation_rows) == 1
    row = annotation_rows[0]
    assert row["frame_source"] == "interpolated"
    assert row["synthetic_frame_flag"] is True
    assert row["interp_source_pair"] == ["000000_1000.png", "000001_2000.png"]
    assert row["interp_alpha"] == 0.5
    assert row["interp_model"] == "linear_blend"
    assert row["interpolation_ref"]
    assert row["synthetic_event_window"]["source_timestamp_pair_us"] == [1000, 2000]

    interpolation_summary = read_json(session_dir / "labels" / "interpolation_index.json")
    assert interpolation_summary["n_synthetic_frames"] == 1
    assert interpolation_summary["items"][0]["synthetic_timestamp_us"] == 1500

    session_package = read_json(session_dir / "labels" / "session_package.json")
    assert session_package["frame_source"] == "interpolated"
    assert session_package["interpolation_ref"]

    manifest_summary = build_manifests(
        canonical_root=synthetic_raw_workspace["canonical_root"],
        manifests_root=synthetic_raw_workspace["manifests_root"],
        split_scheme="exgaze_with_val",
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    assert manifest_summary["counts"]["train"] == 1
    manifest_rows = read_jsonl(synthetic_raw_workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl")
    assert len(manifest_rows) == 1
    manifest_row = manifest_rows[0]
    assert manifest_row["data_mode"] == "mode2"
    assert manifest_row["frame_source"] == "interpolated"
    assert manifest_row["synthetic_frame_flag"] is True
    assert manifest_row["interp_source_pair"] == ["000000_1000.png", "000001_2000.png"]
    assert manifest_row["synthetic_overlap_policy"] == "reuse_event_window"
    assert manifest_summary["canonical_root"].replace("\\", "/").endswith("/canonical/canonical2")


def test_mode1_canonicalize_uses_physical_canonical1_root(synthetic_raw_workspace):
    canonical_summary = canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        data_mode="mode1",
        canonical_name="canonical1",
        frame_source="original",
        link_mode="copy",
        num_workers=1,
    )
    assert canonical_summary["n_sessions_ok"] == 1
    assert canonical_summary["data_mode"] == "mode1"
    assert canonical_summary["canonical_root"].replace("\\", "/").endswith("/canonical/canonical1")
    assert canonical_summary["indexes_root"].replace("\\", "/").endswith("/canonical/canonical1/indexes")

    session_dir = synthetic_raw_workspace["canonical_root"] / "canonical1" / "sessions" / "user01" / "left" / "session_101"
    assert session_dir.exists()
    rows = read_jsonl(session_dir / "labels" / "frame_annotations.jsonl")
    assert len(rows) == 2
    assert rows[0]["data_mode"] == "mode1"
    assert rows[0]["canonical_name"] == "canonical1"
    assert rows[0]["frame_source"] == "original"


class _FakeGroundedSamRuntime:
    def annotate_image(self, image_rgb: np.ndarray) -> dict[str, object]:
        height, width = image_rgb.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[96:152, 140:196] = 1
        return {
            "mask": mask,
            "bbox_xywh": [140.0, 96.0, 56.0, 56.0],
            "ellipse_xywht": [168.0, 124.0, 44.0, 46.0, 0.05],
            "eye_region_xywh": [96.0, 56.0, 144.0, 144.0],
            "class_name": "pupil",
            "box_xyxy": [140.0, 96.0, 196.0, 152.0],
            "box_confidence": 0.9,
            "mask_score": 0.8,
        }


def test_mode2_groundedsam_rerun_updates_synthetic_annotations(synthetic_raw_workspace, monkeypatch):
    monkeypatch.setattr(canonicalize_module, "_build_mode2_groundedsam_runtime", lambda **kwargs: _FakeGroundedSamRuntime())
    canonical_summary = canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        groundedsam_root=synthetic_raw_workspace["groundedsam_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_alpha=0.5,
        interpolation_model="linear_blend",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    assert canonical_summary["n_sessions_ok"] == 1
    session_dir = synthetic_raw_workspace["canonical_root"] / "canonical2" / "sessions" / "user01" / "left" / "session_101"
    annotation_rows = read_jsonl(session_dir / "labels" / "frame_annotations.jsonl")
    assert annotation_rows[0]["annotation_source"] == "gsa_synthetic_rerun"
    assert annotation_rows[0]["interp_model"] == "linear_blend"
    assert annotation_rows[0]["mask_path"]
    eye_region = read_json(session_dir / "labels" / "eye_region.json")
    assert eye_region["interpolation_summary"]["groundedsam_rerun"]["status"] == "completed"
