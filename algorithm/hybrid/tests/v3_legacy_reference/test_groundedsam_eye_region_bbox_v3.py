from __future__ import annotations

import pytest

from pathlib import Path

from hbtxr.preprocess.groundedsam_eye_region_bbox import (
    _box_area_ratio_xyxy,
    _count_touched_borders,
    _eye_bbox_rejection_reason,
    _normalize_session_code,
    _resolve_raw_user_dir,
    _resolve_requested_eyes,
    _xyxy_to_xywh_sensor,
)


def test_resolve_requested_eyes_supports_both_and_single_eye():
    assert _resolve_requested_eyes("both") == ("left", "right")
    assert _resolve_requested_eyes("left") == ("left",)
    assert _resolve_requested_eyes("right") == ("right",)


def test_resolve_requested_eyes_rejects_invalid_selector():
    with pytest.raises(ValueError, match="Unsupported eye selector"):
        _resolve_requested_eyes("center")


def test_xyxy_to_xywh_sensor_clips_to_image_bounds():
    bbox = _xyxy_to_xywh_sensor([-5.0, 10.0, 400.0, 250.0], image_size_wh=(346, 240))

    assert bbox == [0.0, 10.0, 346.0, 230.0]


def test_box_area_ratio_xyxy_computes_frame_coverage():
    ratio = _box_area_ratio_xyxy([0.0, 0.0, 173.0, 120.0], image_size_wh=(346, 240))

    assert ratio == pytest.approx(0.25, rel=1.0e-6)


def test_count_touched_borders_detects_full_frame_like_box():
    touches = _count_touched_borders([0.0, 0.0, 346.0, 240.0], image_size_wh=(346, 240), border_margin_px=3)

    assert touches == 4


def test_eye_bbox_rejection_reason_rejects_large_box():
    reason = _eye_bbox_rejection_reason(
        [0.0, 0.0, 346.0, 240.0],
        image_size_wh=(346, 240),
        max_box_area_ratio=0.85,
        border_margin_px=3,
        max_border_touches=3,
    )

    assert reason is not None
    assert "rejected_large_box" in reason


def test_normalize_session_code_accepts_session_prefix_variants():
    assert _normalize_session_code("101") == "101"
    assert _normalize_session_code("session101") == "101"
    assert _normalize_session_code("session_1_0_1") == "101"


def test_resolve_raw_user_dir_accepts_unpadded_names(tmp_path: Path):
    (tmp_path / "user1").mkdir()
    (tmp_path / "user2").mkdir()

    assert _resolve_raw_user_dir(tmp_path, 1) == tmp_path / "user1"
