from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocess.io_utils import FrameRecord
from src.preprocess.target_fps_build import (
    build_target_fps_dataset,
    build_target_fps_dataset_plan,
    build_target_frame_plan,
    build_target_timestamp_grid,
    discover_target_fps_session_jobs,
    resolve_target_fps_root,
)
from src.utils.io import read_json, read_jsonl, write_jsonl


def _build_target_fps_raw_workspace(tmp_path: Path) -> dict[str, Path]:
    project_root = tmp_path / "project"
    raw_root = project_root / "raw"
    target_root = project_root / "target_data"
    valid_session_dir = raw_root / "user01" / "left" / "session_1_0_1"
    valid_frames_dir = valid_session_dir / "frames"
    valid_events_dir = valid_session_dir / "events"
    skipped_session_dir = raw_root / "user01" / "right" / "session_2_0_1"
    skipped_frames_dir = skipped_session_dir / "frames"

    valid_frames_dir.mkdir(parents=True, exist_ok=True)
    valid_events_dir.mkdir(parents=True, exist_ok=True)
    skipped_frames_dir.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    for idx, ts in enumerate((1000, 41000)):
        image = Image.new("L", (346, 240), color=40 + idx * 20)
        image.save(valid_frames_dir / f"{idx:06d}_{ts}.png")

    np.savez_compressed(
        valid_events_dir / "events.npz",
        t=np.asarray([900, 1000, 1500, 2000, 40500, 41000], dtype=np.int64),
        x=np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int16),
        y=np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int16),
        p=np.asarray([1, -1, 1, -1, 1, -1], dtype=np.int8),
    )

    for idx, ts in enumerate((5000, 9000)):
        image = Image.new("L", (346, 240), color=20 + idx * 10)
        image.save(skipped_frames_dir / f"{idx:06d}_{ts}.png")

    return {
        "project_root": project_root,
        "raw_root": raw_root,
        "target_root": target_root,
    }


def _build_target_fps_annotation_workspace(
    tmp_path: Path,
    *,
    with_manual_csv: bool = False,
    with_bbox_store: bool = False,
    with_full_groundedsam_store: bool = False,
) -> dict[str, Path]:
    workspace = _build_target_fps_raw_workspace(tmp_path)
    valid_session_dir = workspace["raw_root"] / "user01" / "left" / "session_1_0_1"
    annotation_root = workspace["project_root"] / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)

    if with_manual_csv:
        csv_path = valid_session_dir / "user_1.csv"
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

    if with_bbox_store:
        bbox_store_dir = annotation_root / "eye_region_bbox_only" / "sessions" / "user01" / "left" / "session_101"
        bbox_store_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            [
                {
                    "frame_filename": "000000_1000.png",
                    "frame_idx": 0,
                    "timestamp_us": 1000,
                    "eye_region_bbox_xywh_sensor": [80.0, 40.0, 160.0, 120.0],
                    "annotation_source": "gsa_eye_bbox_only",
                },
                {
                    "frame_filename": "000001_41000.png",
                    "frame_idx": 1,
                    "timestamp_us": 41000,
                    "eye_region_bbox_xywh_sensor": [84.0, 44.0, 158.0, 118.0],
                    "annotation_source": "gsa_eye_bbox_only",
                },
            ],
            bbox_store_dir / "eye_region_bboxes.jsonl",
        )

    if with_full_groundedsam_store:
        full_store_dir = annotation_root / "sessions" / "user01" / "left" / "session_101"
        full_mask_dir = full_store_dir / "masks"
        full_mask_dir.mkdir(parents=True, exist_ok=True)

        mask0 = np.zeros((240, 346), dtype=np.uint8)
        mask0[102:138, 154:190] = 1
        mask1 = np.zeros((240, 346), dtype=np.uint8)
        mask1[104:140, 158:194] = 1
        Image.fromarray((mask0 * 255).astype(np.uint8)).save(full_mask_dir / "000000_1000.png")
        Image.fromarray((mask1 * 255).astype(np.uint8)).save(full_mask_dir / "000001_41000.png")

        write_jsonl(
            [
                {
                    "frame_filename": "000000_1000.png",
                    "frame_idx": 0,
                    "timestamp_us": 1000,
                    "eye_region_bbox_xywh_sensor": [80.0, 40.0, 160.0, 120.0],
                    "pupil_mask_path": "sessions/user01/left/session_101/masks/000000_1000.png",
                    "mask_path": "sessions/user01/left/session_101/masks/000000_1000.png",
                    "pupil_region_bbox_xywh_sensor": [154.0, 102.0, 36.0, 36.0],
                    "pupil_ellipse_xywht_sensor": [171.5, 119.5, 36.0, 36.0, 0.0],
                    "annotation_source": "gsa_auto",
                    "annotation_quality": 0.92,
                    "closed_eye_flag": False,
                    "mask_valid": True,
                },
                {
                    "frame_filename": "000001_41000.png",
                    "frame_idx": 1,
                    "timestamp_us": 41000,
                    "eye_region_bbox_xywh_sensor": [84.0, 44.0, 158.0, 118.0],
                    "pupil_mask_path": "sessions/user01/left/session_101/masks/000001_41000.png",
                    "mask_path": "sessions/user01/left/session_101/masks/000001_41000.png",
                    "pupil_region_bbox_xywh_sensor": [158.0, 104.0, 36.0, 36.0],
                    "pupil_ellipse_xywht_sensor": [175.5, 121.5, 36.0, 36.0, 0.0],
                    "annotation_source": "gsa_auto",
                    "annotation_quality": 0.95,
                    "closed_eye_flag": False,
                    "mask_valid": True,
                },
            ],
            full_store_dir / "frame_annotations.jsonl",
        )

    workspace["annotation_root"] = annotation_root
    return workspace


def test_build_target_timestamp_grid_anchors_to_first_raw_frame():
    grid = build_target_timestamp_grid([1000, 41000], target_fps=2000.0)
    assert grid[0] == 1000
    assert grid[-1] == 41000
    assert len(grid) == 81
    assert grid[1] - grid[0] == 500


def test_build_target_frame_plan_marks_raw_reuse_and_interpolation():
    frame_records = [
        FrameRecord(filename="000000_1000.png", timestamp_us=1000, frame_idx=0),
        FrameRecord(filename="000001_41000.png", timestamp_us=41000, frame_idx=1),
    ]
    plan = build_target_frame_plan(frame_records, target_fps=2000.0)

    assert len(plan) == 81
    assert plan[0]["source_kind"] == "raw"
    assert plan[0]["matched_raw_frame_filename"] == "000000_1000.png"
    assert plan[1]["source_kind"] == "interpolated"
    assert plan[1]["source_pair_index"] == [0, 1]
    assert np.isclose(plan[1]["interp_alpha"], 0.0125)
    assert plan[-1]["source_kind"] == "raw"
    assert plan[-1]["matched_raw_frame_filename"] == "000001_41000.png"


def test_discover_target_fps_session_jobs_collects_valid_and_skipped_sessions(tmp_path: Path):
    workspace = _build_target_fps_raw_workspace(tmp_path)
    jobs, skipped = discover_target_fps_session_jobs(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        include_nonstandard_sessions=True,
    )

    assert len(jobs) == 1
    assert len(skipped) == 1
    assert jobs[0]["session_code"] == "101"
    assert jobs[0]["n_target_frames"] == 81
    assert jobs[0]["n_interpolated_frames"] == 79
    assert jobs[0]["planned_artifacts"]["session_h5"].endswith("session.h5")
    assert jobs[0]["planned_artifacts"]["session_npz"].endswith("session_arrays.npz")
    assert skipped[0]["skip_reason"] == "event_file_missing"


def test_build_target_fps_dataset_plan_writes_indexes_and_summary(tmp_path: Path):
    workspace = _build_target_fps_raw_workspace(tmp_path)
    summary = build_target_fps_dataset_plan(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        include_nonstandard_sessions=True,
    )

    target_fps_root = resolve_target_fps_root(workspace["target_root"], 2000.0)
    sessions = read_jsonl(target_fps_root / "indexes" / "sessions.jsonl")
    skipped = read_jsonl(target_fps_root / "indexes" / "skipped_sessions.jsonl")
    build_summary = read_json(target_fps_root / "build_summary.json")

    assert summary["stage"] == "scan_grid"
    assert summary["n_sessions_planned"] == 1
    assert summary["n_sessions_skipped"] == 1
    assert sessions[0]["target_step_us"] == 500
    assert sessions[0]["n_raw_frame_reuse"] == 2
    assert sessions[0]["n_interpolated_frames"] == 79
    assert skipped[0]["skip_reason"] == "event_file_missing"
    assert build_summary["n_target_frames_planned"] == 81
    assert (target_fps_root / "sessions" / "user01" / "left" / "session_101").exists()


def test_build_target_fps_dataset_materializes_npz_session_store(tmp_path: Path):
    workspace = _build_target_fps_raw_workspace(tmp_path)
    summary = build_target_fps_dataset(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )

    target_fps_root = resolve_target_fps_root(workspace["target_root"], 2000.0)
    session_dir = target_fps_root / "sessions" / "user01" / "left" / "session_101"
    session_meta = read_json(session_dir / "session_meta.json")
    frame_labels = read_jsonl(session_dir / "frame_labels.jsonl")
    session_arrays = np.load(session_dir / "session_arrays.npz")

    assert summary["stage"] == "materialized"
    assert summary["n_sessions_materialized"] == 1
    assert session_meta["session_store_format"] == "npz"
    assert session_arrays["frame_images"].shape == (81, 240, 346)
    assert int(session_arrays["frame_images"][0].mean()) == 40
    assert int(session_arrays["frame_images"][40].mean()) == 50
    assert int(session_arrays["frame_images"][-1].mean()) == 60
    assert session_arrays["frame_event_ranges"].shape == (81, 2)
    assert session_arrays["frame_event_ranges"][0].tolist() == [0, 2]
    assert session_arrays["frame_event_ranges"][1].tolist() == [2, 3]
    assert session_arrays["frame_event_ranges"][-1].tolist() == [5, 6]
    assert frame_labels[0]["label_status"] == "pending_annotation_sources"
    assert frame_labels[0]["annotation_reason"] == "groundedsam_no_csv_eye_mask_pipeline_pending"
    assert frame_labels[0]["pupil_status"] == "pending_groundedsam_roi_mask_pipeline"
    assert frame_labels[0]["event_index_range"] == [0, 2]
    assert frame_labels[0]["session_store_format"] == "npz"
    assert frame_labels[40]["source_kind"] == "interpolated"
    assert session_meta["n_rows_pending_annotation"] == 81
    assert session_meta["annotation_report"]["no_csv_groundedsam_step_order"] == [
        "eye_region_mask",
        "eye_region_bbox_from_eye_region_mask",
        "pupil_mask_inside_eye_roi_from_eye_region_bbox",
        "pupil_bbox_from_pupil_mask",
        "pupil_state_xywht_from_pupil_mask",
    ]
    annotation_failures = read_jsonl(target_fps_root / "indexes" / "annotation_failures.jsonl")
    assert summary["n_annotation_failures"] == 81
    assert session_meta["n_annotation_failures"] == 81
    assert annotation_failures[0]["label_status"] == "pending_annotation_sources"
    assert annotation_failures[0]["annotation_reason"] == "groundedsam_no_csv_eye_mask_pipeline_pending"


def test_build_target_fps_dataset_fuses_manual_csv_and_eye_bbox_annotations(tmp_path: Path):
    workspace = _build_target_fps_annotation_workspace(
        tmp_path,
        with_manual_csv=True,
        with_bbox_store=True,
    )
    summary = build_target_fps_dataset(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        annotation_root=workspace["annotation_root"],
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )

    target_fps_root = resolve_target_fps_root(workspace["target_root"], 2000.0)
    session_dir = target_fps_root / "sessions" / "user01" / "left" / "session_101"
    session_meta = read_json(session_dir / "session_meta.json")
    frame_labels = read_jsonl(session_dir / "frame_labels.jsonl")

    assert summary["stage"] == "materialized"
    assert summary["n_rows_annotated_complete"] == 81
    assert session_meta["n_rows_annotated_complete"] == 81
    assert frame_labels[0]["label_status"] == "annotated_complete"
    assert frame_labels[0]["annotation_sources_used"] == {
        "eye": "groundedsam_eye_bbox_only",
        "pupil": "manual_csv",
    }
    assert frame_labels[0]["eye_state_flag"] == "open"
    assert frame_labels[0]["eye_region_bbox_xywh_sensor"] == [80.0, 40.0, 160.0, 120.0]
    assert frame_labels[0]["eye_region_source_bbox_xywh_sensor"] == [80.0, 40.0, 160.0, 120.0]
    assert frame_labels[0]["eye_region_bbox_source"] == "derived_from_eye_region_mask"
    assert frame_labels[0]["eye_region_stage_validation_basis"] == "validated_eye_region_bbox_only_preview"
    assert frame_labels[0]["pupil_mask_path"].endswith("000000_1000.png")
    assert frame_labels[1]["label_status"] == "annotated_complete"
    assert frame_labels[1]["pupil_status"] == "interpolated_manual_csv"
    assert frame_labels[1]["roi_status"] == "interpolated_groundedsam_eye_bbox_only"
    assert np.isclose(frame_labels[1]["pupil_ellipse_xywht_sensor"][0], 173.025)
    assert np.isclose(frame_labels[1]["eye_region_source_bbox_xywh_sensor"][0], 80.05)
    assert frame_labels[1]["eye_region_bbox_xywh_sensor"] == [80.0, 40.0, 161.0, 121.0]


def test_build_target_fps_dataset_manual_csv_does_not_backfill_eye_labels(tmp_path: Path):
    workspace = _build_target_fps_annotation_workspace(
        tmp_path,
        with_manual_csv=True,
        with_bbox_store=False,
    )
    summary = build_target_fps_dataset(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        annotation_root=workspace["annotation_root"],
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )

    target_fps_root = resolve_target_fps_root(workspace["target_root"], 2000.0)
    session_dir = target_fps_root / "sessions" / "user01" / "left" / "session_101"
    session_meta = read_json(session_dir / "session_meta.json")
    frame_labels = read_jsonl(session_dir / "frame_labels.jsonl")

    assert summary["stage"] == "materialized"
    assert summary["n_rows_annotated_complete"] == 0
    assert session_meta["n_rows_pending_annotation"] == 81
    assert session_meta["annotation_report"]["manual_csv_available"] is True
    assert session_meta["annotation_report"]["eye_region_source_contract"] == "groundedsam_only"
    assert session_meta["annotation_report"]["pupil_source_contract"] == "raw_csv_state_or_groundedsam_in_roi"
    assert frame_labels[0]["label_status"] == "pending_annotation_sources"
    assert frame_labels[0]["annotation_reason"] == "groundedsam_eye_roi_missing"
    assert frame_labels[0]["annotation_sources_used"] == {
        "eye": None,
        "pupil": "manual_csv",
    }
    assert frame_labels[0]["eye_region_bbox_xywh_sensor"] is None
    assert frame_labels[0]["eye_region_mask_path"] is None
    assert frame_labels[0]["pupil_bbox_xywh_sensor"] is not None
    assert frame_labels[0]["pupil_mask_path"].endswith("000000_1000.png")
    assert frame_labels[0]["annotation_sources_expected"] == {
        "pupil_raw_csv": True,
        "eye_groundedsam": True,
        "pupil_groundedsam_in_roi": True,
    }
    assert frame_labels[1]["pupil_status"] == "interpolated_manual_csv"


def test_build_target_fps_dataset_no_csv_uses_groundedsam_full_mask_pipeline(tmp_path: Path):
    workspace = _build_target_fps_annotation_workspace(
        tmp_path,
        with_manual_csv=False,
        with_bbox_store=True,
        with_full_groundedsam_store=True,
    )
    summary = build_target_fps_dataset(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        annotation_root=workspace["annotation_root"],
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )

    target_fps_root = resolve_target_fps_root(workspace["target_root"], 2000.0)
    session_dir = target_fps_root / "sessions" / "user01" / "left" / "session_101"
    session_meta = read_json(session_dir / "session_meta.json")
    frame_labels = read_jsonl(session_dir / "frame_labels.jsonl")

    assert summary["stage"] == "materialized"
    assert summary["n_rows_annotated_complete"] == 81
    assert session_meta["n_rows_annotated_complete"] == 81
    assert session_meta["annotation_report"]["groundedsam_full_available"] is True
    assert frame_labels[0]["label_status"] == "annotated_complete"
    assert frame_labels[0]["annotation_sources_used"] == {
        "eye": "groundedsam_eye_bbox_only",
        "pupil": "groundedsam_full",
    }
    assert frame_labels[0]["pupil_status"] == "exact_groundedsam_full"
    assert frame_labels[0]["pupil_mask_source"] == "groundedsam_full"
    assert frame_labels[0]["pupil_bbox_source"] == "derived_from_pupil_mask"
    assert frame_labels[0]["pupil_state_source"] == "derived_from_pupil_mask"
    assert frame_labels[0]["pupil_mask_stage"] == "groundedsam_pupil_mask_inside_eye_roi"
    assert frame_labels[0]["pupil_mask_path"].endswith("000000_1000.png")
    assert frame_labels[0]["pupil_bbox_xywh_sensor"] is not None
    assert frame_labels[0]["state_xyabuv"] is not None
    assert frame_labels[1]["label_status"] == "annotated_complete"
    assert frame_labels[1]["pupil_status"] == "interpolated_groundedsam_full"
    assert frame_labels[1]["pupil_bbox_xywh_sensor"] is not None


def test_build_target_fps_dataset_prefers_eye_bbox_preview_for_no_csv_sessions(tmp_path: Path):
    workspace = _build_target_fps_annotation_workspace(
        tmp_path,
        with_manual_csv=False,
        with_bbox_store=True,
    )
    summary = build_target_fps_dataset(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        annotation_root=workspace["annotation_root"],
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )

    target_fps_root = resolve_target_fps_root(workspace["target_root"], 2000.0)
    session_dir = target_fps_root / "sessions" / "user01" / "left" / "session_101"
    session_meta = read_json(session_dir / "session_meta.json")
    frame_labels = read_jsonl(session_dir / "frame_labels.jsonl")

    assert summary["stage"] == "materialized"
    assert summary["n_rows_annotated_eye_only"] == 81
    assert session_meta["n_rows_annotated_eye_only"] == 81
    assert session_meta["annotation_report"]["groundedsam_eye_bbox_available"] is True
    assert session_meta["annotation_report"]["eye_region_preferred_source_order"] == [
        "groundedsam_eye_bbox_only",
        "groundedsam_full",
    ]
    assert frame_labels[0]["label_status"] == "annotated_eye_only"
    assert frame_labels[0]["annotation_reason"] == "groundedsam_no_csv_pupil_pipeline_pending"
    assert frame_labels[0]["annotation_sources_used"] == {
        "eye": "groundedsam_eye_bbox_only",
        "pupil": None,
    }
    assert frame_labels[0]["roi_status"] == "exact_groundedsam_eye_bbox_only"
    assert frame_labels[0]["eye_region_mask_source"] == "groundedsam_eye_bbox_only"
    assert frame_labels[0]["eye_region_bbox_source"] == "derived_from_eye_region_mask"
    assert frame_labels[0]["eye_region_stage_validation_basis"] == "validated_eye_region_bbox_only_preview"
    assert frame_labels[0]["eye_region_mask_path"].endswith("000000_1000.png")
    assert frame_labels[0]["eye_region_bbox_xywh_sensor"] == [80.0, 40.0, 160.0, 120.0]
    assert frame_labels[0]["eye_region_source_bbox_xywh_sensor"] == [80.0, 40.0, 160.0, 120.0]
    assert frame_labels[1]["roi_status"] == "interpolated_groundedsam_eye_bbox_only"
    assert np.isclose(frame_labels[1]["eye_region_source_bbox_xywh_sensor"][0], 80.05)
    assert frame_labels[1]["eye_region_bbox_xywh_sensor"] == [80.0, 40.0, 161.0, 121.0]


def test_build_target_fps_dataset_eye_only_blink_sets_eye_state_closed(tmp_path: Path):
    workspace = _build_target_fps_annotation_workspace(
        tmp_path,
        with_manual_csv=False,
        with_bbox_store=True,
    )
    bbox_store_path = (
        workspace["annotation_root"]
        / "eye_region_bbox_only"
        / "sessions"
        / "user01"
        / "left"
        / "session_101"
        / "eye_region_bboxes.jsonl"
    )
    write_jsonl(
        [
            {
                "frame_filename": "000000_1000.png",
                "frame_idx": 0,
                "timestamp_us": 1000,
                "eye_region_bbox_xywh_sensor": [80.0, 40.0, 160.0, 120.0],
                "closed_eye_flag": True,
                "blink_candidate_score": 0.91,
                "blink_candidate_reasons": ["rejected_large_box"],
                "blink_candidate_source": "groundedsam_ellipse_heuristic",
            },
            {
                "frame_filename": "000001_41000.png",
                "frame_idx": 1,
                "timestamp_us": 41000,
                "eye_region_bbox_xywh_sensor": [84.0, 44.0, 158.0, 118.0],
                "closed_eye_flag": False,
                "blink_candidate_score": 0.0,
                "blink_candidate_reasons": [],
                "blink_candidate_source": None,
            },
        ],
        bbox_store_path,
    )

    summary = build_target_fps_dataset(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        annotation_root=workspace["annotation_root"],
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )

    target_fps_root = resolve_target_fps_root(workspace["target_root"], 2000.0)
    session_dir = target_fps_root / "sessions" / "user01" / "left" / "session_101"
    frame_labels = read_jsonl(session_dir / "frame_labels.jsonl")

    assert summary["stage"] == "materialized"
    assert frame_labels[0]["label_status"] == "annotated_eye_only"
    assert frame_labels[0]["eye_state_flag"] == "closed"
    assert frame_labels[0]["closed_eye_flag"] is True
    assert frame_labels[0]["blink_candidate_score"] == 0.91
    assert frame_labels[0]["blink_candidate_reasons"] == ["rejected_large_box"]
    assert frame_labels[0]["blink_candidate_source"] == "groundedsam_ellipse_heuristic"
