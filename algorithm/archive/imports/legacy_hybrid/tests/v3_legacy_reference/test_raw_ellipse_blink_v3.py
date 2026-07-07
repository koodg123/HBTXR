from __future__ import annotations

from hbtxr.preprocess.io_utils import EllipseAnnotation
from hbtxr.preprocess.raw_ellipse_blink import (
    RawEllipseBlinkHeuristicConfig,
    apply_groundedsam_store_blink_metadata,
    analyze_raw_ellipse_blink_candidates,
    build_groundedsam_store_blink_metadata,
    build_raw_ellipse_blink_metadata,
)


def _ann(frame_idx: int, major: float, minor: float) -> EllipseAnnotation:
    return EllipseAnnotation(
        frame_filename=f"{frame_idx:06d}_{1000000 + frame_idx}.png",
        frame_idx=frame_idx,
        timestamp_us=1000000 + frame_idx,
        ellipse_xywht=(173.0, 120.0, float(major), float(minor), 0.0),
    )


def test_raw_ellipse_blink_candidate_flags_transient_minor_axis_collapse():
    annotations = [
        _ann(10, 42.0, 39.0),
        _ann(20, 43.0, 40.0),
        _ann(30, 42.0, 22.0),
        _ann(40, 44.0, 40.0),
        _ann(50, 43.0, 39.0),
    ]

    analysis = analyze_raw_ellipse_blink_candidates(annotations)
    rows = {row.frame_idx: row for row in analysis["rows"]}

    assert rows[30].blink_candidate is True
    assert "minor_axis_collapse" in rows[30].blink_candidate_reasons
    assert "local_minor_drop" in rows[30].blink_candidate_reasons
    assert analysis["n_blink_candidates"] == 1


def test_raw_ellipse_blink_candidate_ignores_sustained_narrow_sequence_without_local_drop():
    annotations = [
        _ann(10, 43.0, 40.0),
        _ann(20, 42.0, 39.0),
        _ann(30, 39.0, 27.0),
        _ann(40, 38.0, 26.0),
        _ann(50, 38.0, 25.5),
        _ann(60, 39.0, 27.0),
        _ann(70, 42.0, 39.0),
    ]

    analysis = analyze_raw_ellipse_blink_candidates(annotations)
    rows = {row.frame_idx: row for row in analysis["rows"]}

    assert rows[40].blink_candidate is False
    assert rows[50].blink_candidate is False


def test_raw_ellipse_blink_candidate_respects_custom_thresholds():
    annotations = [
        _ann(10, 42.0, 39.0),
        _ann(20, 42.0, 39.0),
        _ann(30, 42.0, 28.0),
        _ann(40, 42.0, 39.0),
        _ann(50, 42.0, 39.0),
    ]
    conservative = RawEllipseBlinkHeuristicConfig(
        global_minor_ratio_threshold=0.60,
        global_area_ratio_threshold=0.60,
    )
    permissive = RawEllipseBlinkHeuristicConfig(
        global_minor_ratio_threshold=0.80,
        global_area_ratio_threshold=0.75,
    )

    conservative_result = analyze_raw_ellipse_blink_candidates(annotations, heuristic=conservative)
    permissive_result = analyze_raw_ellipse_blink_candidates(annotations, heuristic=permissive)

    assert conservative_result["n_blink_candidates"] == 0
    assert permissive_result["n_blink_candidates"] == 1


def test_build_raw_ellipse_blink_metadata_exports_closed_eye_flag_payload():
    annotations = [
        _ann(10, 42.0, 39.0),
        _ann(20, 43.0, 40.0),
        _ann(30, 42.0, 22.0),
        _ann(40, 44.0, 40.0),
        _ann(50, 43.0, 39.0),
    ]

    metadata, report = build_raw_ellipse_blink_metadata(annotations)

    assert metadata["000030_1000030.png"]["closed_eye_flag"] is True
    assert metadata["000030_1000030.png"]["blink_candidate_source"] == "raw_ellipse_heuristic"
    assert report["source"] == "raw_ellipse_heuristic"
    assert report["n_blink_candidates"] == 1


def test_build_groundedsam_store_blink_metadata_exports_store_source():
    rows = [
        {"frame_filename": "000010_1000010.png", "frame_idx": 10, "timestamp_us": 1000010, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 42.0, 39.0, 0.0]},
        {"frame_filename": "000020_1000020.png", "frame_idx": 20, "timestamp_us": 1000020, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 43.0, 40.0, 0.0]},
        {"frame_filename": "000030_1000030.png", "frame_idx": 30, "timestamp_us": 1000030, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 42.0, 22.0, 0.0]},
        {"frame_filename": "000040_1000040.png", "frame_idx": 40, "timestamp_us": 1000040, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 44.0, 40.0, 0.0]},
        {"frame_filename": "000050_1000050.png", "frame_idx": 50, "timestamp_us": 1000050, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 43.0, 39.0, 0.0]},
    ]

    metadata, report = build_groundedsam_store_blink_metadata(rows)

    assert metadata["000030_1000030.png"]["closed_eye_flag"] is True
    assert metadata["000030_1000030.png"]["blink_candidate_source"] == "groundedsam_ellipse_heuristic"
    assert report["source"] == "groundedsam_ellipse_heuristic"
    assert report["n_blink_candidates"] == 1
    assert report["n_rows_with_ellipse"] == 5


def test_apply_groundedsam_store_blink_metadata_backfills_existing_store_rows():
    rows = [
        {"frame_filename": "000010_1000010.png", "frame_idx": 10, "timestamp_us": 1000010, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 42.0, 39.0, 0.0], "closed_eye_flag": False},
        {"frame_filename": "000020_1000020.png", "frame_idx": 20, "timestamp_us": 1000020, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 43.0, 40.0, 0.0], "closed_eye_flag": False},
        {"frame_filename": "000030_1000030.png", "frame_idx": 30, "timestamp_us": 1000030, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 42.0, 22.0, 0.0], "closed_eye_flag": False},
        {"frame_filename": "000040_1000040.png", "frame_idx": 40, "timestamp_us": 1000040, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 44.0, 40.0, 0.0], "closed_eye_flag": False},
        {"frame_filename": "000050_1000050.png", "frame_idx": 50, "timestamp_us": 1000050, "pupil_ellipse_xywht_sensor": [173.0, 120.0, 43.0, 39.0, 0.0], "closed_eye_flag": False},
    ]

    updated_rows, report = apply_groundedsam_store_blink_metadata(rows, overwrite_existing=False)
    by_frame = {row["frame_filename"]: row for row in updated_rows}

    assert by_frame["000030_1000030.png"]["closed_eye_flag"] is True
    assert by_frame["000030_1000030.png"]["blink_candidate_source"] == "groundedsam_ellipse_heuristic"
    assert report["n_rows_updated"] == 5


def test_apply_groundedsam_store_blink_metadata_preserves_existing_metadata_without_overwrite():
    rows = [
        {
            "frame_filename": "000030_1000030.png",
            "frame_idx": 30,
            "timestamp_us": 1000030,
            "pupil_ellipse_xywht_sensor": [173.0, 120.0, 42.0, 22.0, 0.0],
            "closed_eye_flag": False,
            "blink_candidate_score": 0.123,
            "blink_candidate_reasons": ["preexisting"],
            "blink_candidate_source": "manual_override",
        }
    ]

    updated_rows, report = apply_groundedsam_store_blink_metadata(rows, overwrite_existing=False)

    assert updated_rows[0]["closed_eye_flag"] is False
    assert updated_rows[0]["blink_candidate_source"] == "manual_override"
    assert report["n_rows_updated"] == 0
