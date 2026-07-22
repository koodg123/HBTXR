from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.io import read_jsonl, write_jsonl
from utils.paths import normalize_user_path, resolve_stored_path
from utils.state6 import xywht_to_xyabuv


@dataclass
class GroundedSamAnnotation:
    ann_id: str
    frame_filename: str
    timestamp_us: int
    eye_region_bbox_xywh_sensor: list[float]
    pupil_mask_path: str
    pupil_region_bbox_xywh_sensor: list[float]
    pupil_ellipse_xywht_sensor: list[float]
    annotation_source: str = "gsa_auto"
    annotation_quality: float = 1.0
    closed_eye_flag: bool = False
    mask_valid: bool = True

    def to_row(self) -> dict[str, Any]:
        ellipse = np.asarray(self.pupil_ellipse_xywht_sensor, dtype=np.float32)
        return {
            "ann_id": self.ann_id,
            "frame_filename": self.frame_filename,
            "timestamp_us": int(self.timestamp_us),
            "eye_region_bbox_xywh_sensor": [float(v) for v in self.eye_region_bbox_xywh_sensor],
            "pupil_mask_path": self.pupil_mask_path,
            "mask_path": self.pupil_mask_path,
            "pupil_region_bbox_xywh_sensor": [float(v) for v in self.pupil_region_bbox_xywh_sensor],
            "pupil_ellipse_xywht_sensor": ellipse.tolist(),
            "ellipse_sensor_xywht": ellipse.tolist(),
            "state_xyabuv": xywht_to_xyabuv(ellipse).tolist(),
            "annotation_source": str(self.annotation_source),
            "annotation_quality": float(self.annotation_quality),
            "closed_eye_flag": bool(self.closed_eye_flag),
            "mask_valid": bool(self.mask_valid),
        }


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(out)
    return out


def export_annotation_store(rows: list[GroundedSamAnnotation], path: str | Path) -> Path:
    out = Path(path)
    write_jsonl([row.to_row() for row in rows], out)
    return out


def groundedsam_session_dir(annotation_root: str | Path, *, user_id: int, eye: str, session_code: str) -> Path:
    root = normalize_user_path(annotation_root)
    return root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}"


def groundedsam_annotation_store_path(annotation_root: str | Path, *, user_id: int, eye: str, session_code: str) -> Path:
    return groundedsam_session_dir(annotation_root, user_id=user_id, eye=eye, session_code=session_code) / "frame_annotations.jsonl"


def load_groundedsam_annotation_store(annotation_root: str | Path, *, user_id: int, eye: str, session_code: str) -> list[dict[str, Any]]:
    return read_jsonl(groundedsam_annotation_store_path(annotation_root, user_id=user_id, eye=eye, session_code=session_code))


def resolve_groundedsam_mask_path(annotation_root: str | Path, stored_path: str | Path | None) -> Path | None:
    if stored_path in (None, ""):
        return None
    return resolve_stored_path(annotation_root, stored_path)


def mask_bbox_xywh(mask: np.ndarray) -> list[float] | None:
    coords = np.argwhere(np.asarray(mask) > 0)
    if coords.size == 0:
        return None
    y0 = float(coords[:, 0].min())
    y1 = float(coords[:, 0].max())
    x0 = float(coords[:, 1].min())
    x1 = float(coords[:, 1].max())
    return [x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0]


def mask_to_ellipse_xywht(mask: np.ndarray) -> list[float] | None:
    coords = np.argwhere(np.asarray(mask) > 0)
    if coords.shape[0] < 5:
        return None
    ys = coords[:, 0].astype(np.float32)
    xs = coords[:, 1].astype(np.float32)
    cx = float(xs.mean())
    cy = float(ys.mean())
    centered = np.stack([xs - cx, ys - cy], axis=1)
    cov = (centered.T @ centered) / max(centered.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major = max(float(eigvals[0]), 1.0e-6)
    minor = max(float(eigvals[1]), 1.0e-6)
    width = 4.0 * float(np.sqrt(major))
    height = 4.0 * float(np.sqrt(minor))
    theta = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return [cx, cy, max(width, 1.0), max(height, 1.0), theta]


def expand_eye_region_from_bbox(
    bbox_xywh: list[float] | tuple[float, float, float, float],
    *,
    image_size_wh: tuple[int, int],
    scale: float = 3.0,
    min_size: int = 96,
) -> list[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    image_w, image_h = [int(v) for v in image_size_wh]
    size = max(float(min_size), max(w, h) * float(scale))
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    x0 = max(0.0, cx - 0.5 * size)
    y0 = max(0.0, cy - 0.5 * size)
    x1 = min(float(image_w), cx + 0.5 * size)
    y1 = min(float(image_h), cy + 0.5 * size)
    return [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]
