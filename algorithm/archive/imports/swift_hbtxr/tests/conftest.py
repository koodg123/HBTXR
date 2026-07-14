from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"

for path in (PROJECT_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swift_hbtxr.io import write_json, write_jsonl, xywht_to_xyabuv


@pytest.fixture
def tmp_path() -> Path:
    temp_root = PROJECT_ROOT / "temp_tests"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"pytest-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def synthetic_workspace(tmp_path: Path) -> dict[str, Path]:
    project_root = tmp_path / "project"
    canonical_root = project_root / "canonical"
    indexes_root = canonical_root / "indexes"
    manifests_root = project_root / "manifests"
    splits_root = project_root / "splits"
    session_dir = canonical_root / "sessions" / "user01" / "left" / "session_101"
    frames_dir = session_dir / "frames"
    masks_dir = session_dir / "labels" / "masks"
    labels_dir = session_dir / "labels"
    events_dir = session_dir / "events"

    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    indexes_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    sensor_size_wh = [346, 240]
    roi_xywh = [80, 40, 160, 160]
    timestamps = [1000, 2000]
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

        mask = Image.new("L", (roi_xywh[2], roi_xywh[3]), color=0)
        mask_draw = ImageDraw.Draw(mask)
        cx = ellipse[0] - roi_xywh[0]
        cy = ellipse[1] - roi_xywh[1]
        mask_bbox = (
            cx - ellipse[2] / 2.0,
            cy - ellipse[3] / 2.0,
            cx + ellipse[2] / 2.0,
            cy + ellipse[3] / 2.0,
        )
        mask_draw.ellipse(mask_bbox, fill=255)
        mask.save(masks_dir / Path(frame_name).with_suffix(".png").name)

    events = {
        "t": np.asarray([900, 950, 1000, 1500, 1600, 1700, 1800, 1900, 2000], dtype=np.int64),
        "x": np.asarray([150, 155, 160, 162, 164, 166, 168, 170, 172], dtype=np.int16),
        "y": np.asarray([118, 119, 120, 121, 122, 123, 124, 125, 126], dtype=np.int16),
        "p": np.asarray([1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=np.int8),
    }
    np.savez_compressed(events_dir / "events.npz", **events)

    annotations = []
    for idx, (frame_name, ts, ellipse) in enumerate(zip(frame_names, timestamps, ellipses)):
        annotations.append(
            {
                "ann_id": f"user01__left__session_101__{idx:06d}",
                "frame_filename": frame_name,
                "frame_idx": idx,
                "timestamp_us": ts,
                "frame_path": f"sessions/user01/left/session_101/frames/{frame_name}",
                "eye_region_xywh": roi_xywh,
                "eye_region_bbox_xywh_sensor": roi_xywh,
                "pupil_mask_path": f"sessions/user01/left/session_101/labels/masks/{Path(frame_name).with_suffix('.png').name}",
                "mask_path": f"sessions/user01/left/session_101/labels/masks/{Path(frame_name).with_suffix('.png').name}",
                "pupil_region_bbox_xywh_sensor": [ellipse[0] - ellipse[2] / 2.0, ellipse[1] - ellipse[3] / 2.0, ellipse[2], ellipse[3]],
                "pupil_ellipse_xywht_sensor": ellipse,
                "ellipse_sensor_xywht": ellipse,
                "state_xyabuv": xywht_to_xyabuv(np.asarray(ellipse, dtype=np.float32)).tolist(),
                "annotation_source": "gsa_reviewed",
                "annotation_quality": 0.9,
                "closed_eye_flag": False,
                "mask_valid": True,
            }
        )

    annotation_store_rel = "sessions/user01/left/session_101/labels/frame_annotations.jsonl"
    session_package_rel = "sessions/user01/left/session_101/labels/session_package.json"
    write_jsonl(annotations, labels_dir / "frame_annotations.jsonl")
    write_json(
        {
            "session_key": "user01/left/session_101",
            "subject_id": 1,
            "user_id": 1,
            "eye": "left",
            "session_code": "101",
            "sensor_size_wh": sensor_size_wh,
            "eye_region_xywh": roi_xywh,
            "frame_source_size_wh": [roi_xywh[2], roi_xywh[3]],
            "event_source_size_wh": [roi_xywh[2], roi_xywh[3]],
            "events_npz": "sessions/user01/left/session_101/events/events.npz",
            "annotation_store_path": annotation_store_rel,
        },
        labels_dir / "session_package.json",
    )

    return {
        "project_root": project_root,
        "canonical_root": canonical_root,
        "indexes_root": indexes_root,
        "manifests_root": manifests_root,
        "splits_root": splits_root,
        "session_package_path": canonical_root / session_package_rel,
    }
