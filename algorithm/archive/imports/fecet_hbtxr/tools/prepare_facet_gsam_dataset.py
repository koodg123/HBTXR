from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from _bootstrap import PROJECT_ROOT

from fecet_hbtxr.io import read_json, read_jsonl, relativize_to, resolve_stored_path, write_json, write_jsonl, xywht_to_xyabuv
from prepare_dataset import build_manifests, build_session_index, build_split_views


SENSOR_WIDTH = 346
SENSOR_HEIGHT = 240
OFFICIAL_SESSION_CODES = {"101", "102", "201", "202"}
LEFT_BAD_PIXELS = ((158, 27), (324, 27))


@dataclass
class FrameRecord:
    filename: str
    timestamp_us: int
    frame_idx: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "timestamp_us": int(self.timestamp_us),
            "frame_idx": self.frame_idx,
        }


@dataclass
class EllipseAnnotation:
    frame_filename: str
    frame_idx: int | None
    timestamp_us: int
    ellipse_xywht: tuple[float, float, float, float, float]
    region_id: int = 0
    label: str = "pupil"
    source: str = "manual"
    quality: float = 1.0


@dataclass
class EyeRegion:
    x: int
    y: int
    w: int
    h: int

    def to_list(self) -> list[int]:
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def to_box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class RawSession:
    user_id: int
    eye: str
    session_code: str
    raw_session_dir: Path
    frames_dir: Path
    events_txt: Path
    annotation_csv: Path | None

    @property
    def session_key(self) -> str:
        return f"{canonical_user_name(self.user_id)}/{self.eye}/session_{self.session_code}"


def canonical_user_name(user_id: int) -> str:
    return f"user{user_id:02d}"


def session_dir_to_code(session_name: str) -> str:
    parts = session_name.replace("session_", "").split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected session name: {session_name}")
    return "".join(parts)


def is_official_session_code(session_code: str) -> bool:
    return session_code in OFFICIAL_SESSION_CODES


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_link_or_copy(src: Path, dst: Path, *, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    ensure_dir(dst.parent)
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
            return
        except OSError:
            mode = "copy"
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported link mode: {mode}")


def assert_output_isolated(path: Path, *roots: Path) -> None:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise ValueError(f"Output path must not be nested under input root: {resolved} in {root}")


def parse_frame_filename(filename: str) -> tuple[int | None, int]:
    stem = Path(filename).stem
    if "_" in stem:
        idx_str, ts_str = stem.split("_", 1)
        frame_idx = None
        try:
            frame_idx = int(idx_str)
        except ValueError:
            frame_idx = None
        return frame_idx, int(float(ts_str))
    return None, int(float(stem))


def collect_frame_records(frames_dir: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for image_path in sorted(frames_dir.glob("*.png")):
        frame_idx, timestamp_us = parse_frame_filename(image_path.name)
        records.append(FrameRecord(filename=image_path.name, timestamp_us=timestamp_us, frame_idx=frame_idx))
    return records


def load_events_from_txt(path: Path, *, eye: str) -> dict[str, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Unexpected event txt shape: {arr.shape} from {path}")
    t = arr[:, 0].astype(np.int64)
    x = arr[:, 1].astype(np.int16)
    y = arr[:, 2].astype(np.int16)
    p = arr[:, 3].astype(np.int8)
    if eye == "left":
        keep = np.ones_like(t, dtype=bool)
        for bad_x, bad_y in LEFT_BAD_PIXELS:
            keep &= ~((x == bad_x) & (y == bad_y))
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
    return {"t": t, "x": x, "y": y, "p": p}


def save_events_npz(events: dict[str, np.ndarray], dst_path: Path) -> None:
    ensure_dir(dst_path.parent)
    np.savez_compressed(
        dst_path,
        t=np.asarray(events["t"], dtype=np.int64),
        x=np.asarray(events["x"], dtype=np.int16),
        y=np.asarray(events["y"], dtype=np.int16),
        p=np.asarray(events["p"], dtype=np.int8),
    )


def parse_via_csv_annotations(csv_path: Path, *, known_frame_filenames: Sequence[str]) -> tuple[list[EllipseAnnotation], dict[str, int]]:
    report = {
        "csv_exists": int(csv_path.exists()),
        "rows_total": 0,
        "rows_region_positive": 0,
        "groups_total": 0,
        "annotations_kept": 0,
        "groups_skipped_unknown_frame": 0,
        "groups_skipped_nonellipse": 0,
        "groups_skipped_bad_shape": 0,
        "groups_skipped_bad_values": 0,
    }
    if not csv_path.exists():
        return [], report

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report["rows_total"] = len(rows)
    if not rows or "region_count" not in rows[0]:
        return [], report

    known = set(known_frame_filenames)
    valid_rows = []
    for row in rows:
        try:
            if int(float(row.get("region_count", 0))) > 0:
                valid_rows.append(row)
        except (TypeError, ValueError):
            continue
    report["rows_region_positive"] = len(valid_rows)

    annotations: list[EllipseAnnotation] = []
    index = 0
    while index < len(valid_rows):
        row = valid_rows[index]
        frame_filename = str(row.get("filename", "")).strip()
        report["groups_total"] += 1
        try:
            region_count = max(1, int(float(row.get("region_count", 1))))
        except (TypeError, ValueError):
            region_count = 1
        if frame_filename not in known:
            report["groups_skipped_unknown_frame"] += 1
            index += region_count
            continue

        frame_idx, timestamp_us = parse_frame_filename(frame_filename)
        chosen = None
        region_id = 0
        saw_bad_shape = False
        for offset in range(region_count):
            if index + offset >= len(valid_rows):
                break
            candidate = valid_rows[index + offset]
            try:
                shape = json.loads(candidate.get("region_shape_attributes", "{}"))
            except Exception:
                saw_bad_shape = True
                continue
            if shape.get("name") not in (None, "ellipse"):
                continue
            chosen = shape
            try:
                region_id = int(float(candidate.get("region_id", 0)))
            except (TypeError, ValueError):
                region_id = 0
            break

        if chosen is None:
            if saw_bad_shape:
                report["groups_skipped_bad_shape"] += 1
            else:
                report["groups_skipped_nonellipse"] += 1
            index += region_count
            continue

        try:
            annotations.append(
                EllipseAnnotation(
                    frame_filename=frame_filename,
                    frame_idx=frame_idx,
                    timestamp_us=timestamp_us,
                    ellipse_xywht=(
                        float(chosen["cx"]),
                        float(chosen["cy"]),
                        float(chosen["rx"]) * 2.0,
                        float(chosen["ry"]) * 2.0,
                        float(chosen.get("theta", 0.0)),
                    ),
                    region_id=region_id,
                    source="manual_csv",
                    quality=1.0,
                )
            )
            report["annotations_kept"] += 1
        except Exception:
            report["groups_skipped_bad_values"] += 1
        index += region_count

    annotations.sort(key=lambda ann: ann.timestamp_us)
    return annotations, report


def derive_eye_region(
    annotations: Sequence[EllipseAnnotation],
    *,
    sensor_size: tuple[int, int] = (SENSOR_WIDTH, SENSOR_HEIGHT),
    margin_px: int = 24,
    target_aspect_wh: float = 256.0 / 160.0,
) -> EyeRegion:
    sensor_w, sensor_h = sensor_size
    if not annotations:
        return EyeRegion(0, 0, sensor_w, sensor_h)

    xs0, ys0, xs1, ys1 = [], [], [], []
    for ann in annotations:
        x, y, w, h, _ = ann.ellipse_xywht
        xs0.append(x - w / 2.0 - margin_px)
        ys0.append(y - h / 2.0 - margin_px)
        xs1.append(x + w / 2.0 + margin_px)
        ys1.append(y + h / 2.0 + margin_px)

    x0 = max(0.0, min(xs0))
    y0 = max(0.0, min(ys0))
    x1 = min(float(sensor_w), max(xs1))
    y1 = min(float(sensor_h), max(ys1))
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)

    current_ratio = width / height
    if current_ratio < target_aspect_wh:
        new_width = height * target_aspect_wh
        delta = new_width - width
        x0 -= delta / 2.0
        x1 += delta / 2.0
    else:
        new_height = width / target_aspect_wh
        delta = new_height - height
        y0 -= delta / 2.0
        y1 += delta / 2.0

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > sensor_w:
        x0 -= x1 - sensor_w
        x1 = sensor_w
    if y1 > sensor_h:
        y0 -= y1 - sensor_h
        y1 = sensor_h

    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(float(sensor_w), x1)
    y1 = min(float(sensor_h), y1)
    return EyeRegion(x=int(round(x0)), y=int(round(y0)), w=max(1, int(round(x1 - x0))), h=max(1, int(round(y1 - y0))))


def ellipse_sensor_to_roi(ellipse_xywht: Sequence[float], eye_region: EyeRegion) -> list[float]:
    x, y, w, h, theta = ellipse_xywht
    return [float(x) - eye_region.x, float(y) - eye_region.y, float(w), float(h), float(theta)]


def rasterize_ellipse_mask(ellipse_xywht: Sequence[float], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy, ew, eh, theta = ellipse_xywht
    cos_t = math.cos(float(theta))
    sin_t = math.sin(float(theta))
    x = xx - float(cx)
    y = yy - float(cy)
    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t
    mask = (xr / max(float(ew) / 2.0, 1e-6)) ** 2 + (yr / max(float(eh) / 2.0, 1e-6)) ** 2 <= 1.0
    return mask.astype(np.uint8)


def crop_mask(mask: np.ndarray, eye_region: EyeRegion) -> np.ndarray:
    x0, y0, x1, y1 = eye_region.to_box()
    return mask[y0:y1, x0:x1].astype(np.uint8)


def nearest_prior_ellipse(frame_timestamp_us: int, annotations: Sequence[EllipseAnnotation]) -> tuple[float, float, float, float, float] | None:
    if not annotations:
        return None
    return min(annotations, key=lambda ann: abs(int(ann.timestamp_us) - int(frame_timestamp_us))).ellipse_xywht


def fit_ellipse_from_mask(mask_roi: np.ndarray, *, roi_offset_xy: tuple[int, int]) -> tuple[float, float, float, float, float] | None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for Grounded-SAM ellipse fitting") from exc

    mask_u8 = (np.asarray(mask_roi, dtype=np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cnt is not None and len(cnt) >= 5]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 1.0:
        return None
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (major, minor), angle_deg = ellipse
    if major < minor:
        major, minor = minor, major
        angle_deg += 90.0
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return (
        float(cx + roi_offset_xy[0]),
        float(cy + roi_offset_xy[1]),
        float(major),
        float(minor),
        float(np.deg2rad(angle_deg)),
    )


class GroundedSamAnnotator:
    def __init__(
        self,
        *,
        grounded_sam_root: str | Path,
        grounding_config: str | Path,
        grounded_checkpoint: str | Path,
        sam_checkpoint: str | Path,
        sam_version: str = "vit_h",
        device: str = "cpu",
        text_prompt: str = "pupil",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        min_mask_area_px: int = 40,
    ) -> None:
        self.root = Path(grounded_sam_root).resolve()
        self.grounding_config = Path(grounding_config).resolve()
        self.grounded_checkpoint = Path(grounded_checkpoint).resolve()
        self.sam_checkpoint = Path(sam_checkpoint).resolve()
        self.sam_version = sam_version
        self.device = device
        self.text_prompt = text_prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.min_mask_area_px = int(min_mask_area_px)
        self._init_models()

    def _init_models(self) -> None:
        missing = [path for path in (self.root, self.grounding_config, self.grounded_checkpoint, self.sam_checkpoint) if not path.exists()]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Grounded-SAM resources are missing: {joined}")
        root_str = str(self.root)
        dino_str = str(self.root / "GroundingDINO")
        for item in (root_str, dino_str):
            if item not in sys.path:
                sys.path.insert(0, item)

        try:
            import cv2
            from groundingdino.util.inference import Model
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("Grounded-SAM dependencies are not importable in the current Python environment") from exc

        self.cv2 = cv2
        self.grounding_model = Model(
            model_config_path=str(self.grounding_config),
            model_checkpoint_path=str(self.grounded_checkpoint),
            device=self.device,
        )
        sam_model = sam_model_registry[self.sam_version](checkpoint=str(self.sam_checkpoint)).to(self.device)
        self.predictor = SamPredictor(sam_model)

    def _score_candidate(
        self,
        *,
        mask_roi: np.ndarray,
        box_confidence: float,
        sam_score: float,
        gray_roi: np.ndarray,
        prior_center_roi: tuple[float, float] | None,
    ) -> float:
        area = float(mask_roi.sum())
        if area < self.min_mask_area_px:
            return float("-inf")
        ys, xs = np.nonzero(mask_roi)
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        darkness = 1.0 - float(gray_roi[mask_roi > 0].mean()) / 255.0
        prior_score = 0.5
        if prior_center_roi is not None:
            distance = math.sqrt((center_x - prior_center_roi[0]) ** 2 + (center_y - prior_center_roi[1]) ** 2)
            diagonal = math.sqrt(float(mask_roi.shape[0] ** 2 + mask_roi.shape[1] ** 2))
            prior_score = max(0.0, 1.0 - distance / max(diagonal, 1.0))
        return 0.45 * float(box_confidence) + 0.35 * float(sam_score) + 0.15 * darkness + 0.05 * prior_score

    def annotate_frame(
        self,
        *,
        image_path: Path,
        eye_region: EyeRegion | None,
        prior_ellipse_sensor: tuple[float, float, float, float, float] | None,
    ) -> dict[str, Any] | None:
        image_bgr = self.cv2.imread(str(image_path))
        if image_bgr is None:
            return None

        if eye_region is None:
            roi_offset = (0, 0)
            roi_image = image_bgr
        else:
            x0, y0, x1, y1 = eye_region.to_box()
            roi_offset = (x0, y0)
            roi_image = image_bgr[y0:y1, x0:x1]
        if roi_image.size == 0:
            return None

        detections, _ = self.grounding_model.predict_with_caption(
            image=roi_image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        boxes = np.asarray(detections.xyxy) if getattr(detections, "xyxy", None) is not None else np.zeros((0, 4), dtype=np.float32)
        confidences = np.asarray(detections.confidence) if getattr(detections, "confidence", None) is not None else np.ones((len(boxes),), dtype=np.float32)
        if len(boxes) == 0:
            return None

        self.predictor.set_image(self.cv2.cvtColor(roi_image, self.cv2.COLOR_BGR2RGB))
        gray_roi = self.cv2.cvtColor(roi_image, self.cv2.COLOR_BGR2GRAY)
        prior_center_roi = None
        if prior_ellipse_sensor is not None:
            prior_center_roi = (
                float(prior_ellipse_sensor[0] - roi_offset[0]),
                float(prior_ellipse_sensor[1] - roi_offset[1]),
            )

        best: dict[str, Any] | None = None
        for box, confidence in zip(boxes, confidences):
            masks, scores, _ = self.predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=True)
            for mask_roi, sam_score in zip(masks, scores):
                candidate_score = self._score_candidate(
                    mask_roi=mask_roi.astype(np.uint8),
                    box_confidence=float(confidence),
                    sam_score=float(sam_score),
                    gray_roi=gray_roi,
                    prior_center_roi=prior_center_roi,
                )
                if best is None or candidate_score > best["score"]:
                    best = {
                        "mask_roi": mask_roi.astype(np.uint8),
                        "score": float(candidate_score),
                        "box_confidence": float(confidence),
                        "sam_score": float(sam_score),
                    }
        if best is None:
            return None

        ellipse = fit_ellipse_from_mask(best["mask_roi"], roi_offset_xy=roi_offset)
        if ellipse is None:
            return None
        quality = float(np.clip(0.5 * best["box_confidence"] + 0.5 * best["sam_score"], 0.0, 1.0))
        return {
            "ellipse_xywht": ellipse,
            "mask_roi": best["mask_roi"],
            "quality": quality,
        }


def annotation_row_from_ellipse(
    *,
    ann: EllipseAnnotation,
    session_key: str,
    user_id: int,
    eye: str,
    session_code: str,
    eye_region: EyeRegion,
    canonical_root: Path,
    session_dir: Path,
    roi_mask: np.ndarray,
) -> dict[str, Any]:
    ellipse_sensor = [float(v) for v in ann.ellipse_xywht]
    ellipse_roi = ellipse_sensor_to_roi(ellipse_sensor, eye_region)
    state6 = [float(v) for v in xywht_to_xyabuv(np.asarray(ellipse_sensor, dtype=np.float32))]
    pupil_region_bbox = [
        float(ellipse_sensor[0] - 0.5 * ellipse_sensor[2]),
        float(ellipse_sensor[1] - 0.5 * ellipse_sensor[3]),
        float(ellipse_sensor[2]),
        float(ellipse_sensor[3]),
    ]
    mask_name = Path(ann.frame_filename).with_suffix(".png").name
    mask_path = session_dir / "labels" / "masks" / mask_name
    ensure_dir(mask_path.parent)
    Image.fromarray((roi_mask > 0).astype(np.uint8) * 255).save(mask_path)
    frame_rel = relativize_to(canonical_root, session_dir / "frames" / ann.frame_filename)
    mask_rel = relativize_to(canonical_root, mask_path)
    return {
        "ann_id": f"{session_key.replace('/', '__')}__{Path(ann.frame_filename).stem}",
        "frame_filename": ann.frame_filename,
        "frame_idx": ann.frame_idx,
        "timestamp_us": int(ann.timestamp_us),
        "frame_path": frame_rel,
        "session_key": session_key,
        "user_id": int(user_id),
        "subject_id": int(user_id),
        "eye": eye,
        "session_code": session_code,
        "session_official": bool(is_official_session_code(session_code)),
        "eye_region_xywh": eye_region.to_list(),
        "eye_region_bbox_xywh_sensor": eye_region.to_list(),
        "roi_size_wh": [int(eye_region.w), int(eye_region.h)],
        "frame_source_size_wh": [int(eye_region.w), int(eye_region.h)],
        "event_source_size_wh": [int(eye_region.w), int(eye_region.h)],
        "pupil_region_bbox_xywh_sensor": pupil_region_bbox,
        "pupil_ellipse_xywht_sensor": ellipse_sensor,
        "ellipse_sensor_xywht": ellipse_sensor,
        "ellipse_roi_xywht": ellipse_roi,
        "ellipse_frame_xywht": ellipse_roi,
        "ellipse_event_xywht": ellipse_roi,
        "state_xyabuv": state6,
        "sensor_size_wh": [SENSOR_WIDTH, SENSOR_HEIGHT],
        "annotation_source": ann.source,
        "annotation_quality": float(ann.quality),
        "closed_eye_flag": False,
        "mask_valid": True,
        "mask_path": mask_rel,
        "pupil_mask_path": mask_rel,
        "region_id": int(ann.region_id),
        "label": ann.label,
    }


def build_session_package(
    *,
    session_key: str,
    user_id: int,
    eye: str,
    session_code: str,
    eye_region_xywh: Sequence[int],
    events_npz: str,
    frame_index_path: str,
    annotation_store_path: str,
    n_frames: int,
    n_labelled_frames: int,
    raw_session_dir: str,
    source_events_txt: str,
    annotation_mode: str,
) -> dict[str, Any]:
    return {
        "session_key": session_key,
        "user_id": int(user_id),
        "subject_id": int(user_id),
        "eye": eye,
        "session_code": str(session_code),
        "sensor_size_wh": [SENSOR_WIDTH, SENSOR_HEIGHT],
        "eye_region_xywh": [int(v) for v in eye_region_xywh],
        "frame_source_size_wh": [int(eye_region_xywh[2]), int(eye_region_xywh[3])],
        "event_source_size_wh": [int(eye_region_xywh[2]), int(eye_region_xywh[3])],
        "events_npz": events_npz,
        "frame_index_path": frame_index_path,
        "annotation_store_path": annotation_store_path,
        "split_policy": "subject_independent_exgaze_with_val",
        "n_frames": int(n_frames),
        "n_labelled_frames": int(n_labelled_frames),
        "raw_session_dir": str(raw_session_dir),
        "source_events_txt": str(source_events_txt),
        "annotation_mode": annotation_mode,
    }


def discover_raw_sessions(
    raw_root: Path,
    *,
    include_users: set[int] | None = None,
    eyes: Sequence[str] = ("left", "right"),
    official_only: bool = True,
) -> list[RawSession]:
    sessions: list[RawSession] = []
    for user_dir in sorted([path for path in raw_root.glob("user*") if path.is_dir()]):
        try:
            user_id = int(user_dir.name.replace("user", ""))
        except ValueError:
            continue
        if include_users is not None and user_id not in include_users:
            continue
        for eye in eyes:
            eye_dir = user_dir / eye
            if not eye_dir.exists():
                continue
            for session_dir in sorted([path for path in eye_dir.glob("session_*_*_*") if path.is_dir()]):
                session_code = session_dir_to_code(session_dir.name)
                if official_only and not is_official_session_code(session_code):
                    continue
                frames_dir = session_dir / "frames"
                events_txt = session_dir / "events" / "events.txt"
                annotation_csv = None
                for candidate in (session_dir / f"user_{user_id}.csv", session_dir / f"user{user_id}.csv"):
                    if candidate.exists():
                        annotation_csv = candidate
                        break
                if not frames_dir.exists() or not events_txt.exists():
                    continue
                sessions.append(
                    RawSession(
                        user_id=user_id,
                        eye=eye,
                        session_code=session_code,
                        raw_session_dir=session_dir,
                        frames_dir=frames_dir,
                        events_txt=events_txt,
                        annotation_csv=annotation_csv,
                    )
                )
    return sessions


def should_select_for_annotation(index: int, *, stride: int, offset: int) -> bool:
    if stride <= 1:
        return True
    return (index - offset) % stride == 0


def build_canonical_from_raw(
    *,
    raw_root: Path,
    canonical_root: Path,
    grounded_sam_root: Path,
    annotation_mode: str,
    link_mode: str,
    overwrite: bool,
    official_only: bool,
    annotation_stride: int,
    annotation_offset: int,
    frame_limit: int | None,
    include_users: set[int] | None,
    eyes: Sequence[str],
    gsam_kwargs: dict[str, Any],
) -> dict[str, Any]:
    sessions = discover_raw_sessions(raw_root, include_users=include_users, eyes=eyes, official_only=official_only)
    summary_rows: list[dict[str, Any]] = []
    annotator = None
    if annotation_mode in {"csv_then_gsam", "grounded_sam"}:
        annotator = GroundedSamAnnotator(grounded_sam_root=grounded_sam_root, **gsam_kwargs)

    for session in sessions:
        session_dir = canonical_root / "sessions" / canonical_user_name(session.user_id) / session.eye / f"session_{session.session_code}"
        package_path = session_dir / "labels" / "session_package.json"
        if package_path.exists() and not overwrite:
            summary_rows.append({"session_key": session.session_key, "status": "skipped_existing", "annotation_mode": annotation_mode})
            continue

        ensure_dir(session_dir / "frames")
        ensure_dir(session_dir / "events")
        ensure_dir(session_dir / "labels" / "masks")
        frame_records = collect_frame_records(session.frames_dir)
        if frame_limit is not None:
            frame_records = frame_records[: int(frame_limit)]
        frame_names = [record.filename for record in frame_records]

        for record in frame_records:
            maybe_link_or_copy(session.frames_dir / record.filename, session_dir / "frames" / record.filename, mode=link_mode, overwrite=overwrite)

        events = load_events_from_txt(session.events_txt, eye=session.eye)
        save_events_npz(events, session_dir / "events" / "events.npz")

        manual_annotations, parse_report = ([], {"csv_exists": 0, "annotations_kept": 0})
        if session.annotation_csv is not None:
            manual_annotations, parse_report = parse_via_csv_annotations(session.annotation_csv, known_frame_filenames=frame_names)

        manual_by_name = {ann.frame_filename: ann for ann in manual_annotations}
        eye_region = derive_eye_region(manual_annotations)
        rows: list[dict[str, Any]] = []

        for manual_ann in manual_annotations:
            mask_sensor = rasterize_ellipse_mask(manual_ann.ellipse_xywht, image_size=(SENSOR_WIDTH, SENSOR_HEIGHT))
            rows.append(
                annotation_row_from_ellipse(
                    ann=manual_ann,
                    session_key=session.session_key,
                    user_id=session.user_id,
                    eye=session.eye,
                    session_code=session.session_code,
                    eye_region=eye_region,
                    canonical_root=canonical_root,
                    session_dir=session_dir,
                    roi_mask=crop_mask(mask_sensor, eye_region),
                )
            )

        if annotator is not None:
            for index, record in enumerate(frame_records):
                if not should_select_for_annotation(index, stride=annotation_stride, offset=annotation_offset):
                    continue
                if annotation_mode == "csv_then_gsam" and record.filename in manual_by_name:
                    continue
                image_path = session_dir / "frames" / record.filename
                prior_ellipse = nearest_prior_ellipse(record.timestamp_us, manual_annotations)
                result = annotator.annotate_frame(image_path=image_path, eye_region=eye_region, prior_ellipse_sensor=prior_ellipse)
                if result is None:
                    continue
                gsam_ann = EllipseAnnotation(
                    frame_filename=record.filename,
                    frame_idx=record.frame_idx,
                    timestamp_us=record.timestamp_us,
                    ellipse_xywht=tuple(float(v) for v in result["ellipse_xywht"]),
                    source="grounded_sam",
                    quality=float(result["quality"]),
                )
                rows.append(
                    annotation_row_from_ellipse(
                        ann=gsam_ann,
                        session_key=session.session_key,
                        user_id=session.user_id,
                        eye=session.eye,
                        session_code=session.session_code,
                        eye_region=eye_region,
                        canonical_root=canonical_root,
                        session_dir=session_dir,
                        roi_mask=np.asarray(result["mask_roi"], dtype=np.uint8),
                    )
                )

        rows.sort(key=lambda row: (int(row.get("frame_idx") or -1), int(row["timestamp_us"])))
        ann_path = session_dir / "labels" / "frame_annotations.jsonl"
        frame_index_path = session_dir / "labels" / "frame_index.jsonl"
        write_jsonl([record.to_dict() for record in frame_records], frame_index_path)
        write_jsonl(rows, ann_path)

        session_package = build_session_package(
            session_key=session.session_key,
            user_id=session.user_id,
            eye=session.eye,
            session_code=session.session_code,
            eye_region_xywh=eye_region.to_list(),
            events_npz=relativize_to(canonical_root, session_dir / "events" / "events.npz"),
            frame_index_path=relativize_to(canonical_root, frame_index_path),
            annotation_store_path=relativize_to(canonical_root, ann_path),
            n_frames=len(frame_records),
            n_labelled_frames=len(rows),
            raw_session_dir=str(session.raw_session_dir),
            source_events_txt=str(session.events_txt),
            annotation_mode=annotation_mode,
        )
        write_json(session_package, package_path)
        write_json(
            {
                "session_key": session.session_key,
                "raw_session_dir": str(session.raw_session_dir),
                "annotation_mode": annotation_mode,
                "parse_report": parse_report,
                "n_frames": len(frame_records),
                "n_annotations": len(rows),
                "manual_csv_annotations": int(parse_report.get("annotations_kept", 0)),
                "grounded_sam_annotations": int(sum(1 for row in rows if row["annotation_source"] == "grounded_sam")),
            },
            session_dir / "labels" / "annotation_summary.json",
        )
        summary_rows.append(
            {
                "session_key": session.session_key,
                "status": "prepared",
                "n_frames": len(frame_records),
                "n_annotations": len(rows),
                "manual_csv_annotations": int(parse_report.get("annotations_kept", 0)),
                "grounded_sam_annotations": int(sum(1 for row in rows if row["annotation_source"] == "grounded_sam")),
            }
        )

    summary = {
        "raw_root": str(raw_root),
        "canonical_root": str(canonical_root),
        "annotation_mode": annotation_mode,
        "sessions_total": len(sessions),
        "sessions_prepared": int(sum(1 for row in summary_rows if row["status"] == "prepared")),
        "sessions_skipped_existing": int(sum(1 for row in summary_rows if row["status"] == "skipped_existing")),
        "rows": summary_rows,
    }
    write_json(summary, canonical_root / "prepare_facet_gsam_canonical_summary.json")
    return summary


def write_events_txt_from_npz(events_npz_path: Path, output_path: Path) -> None:
    events = np.load(events_npz_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for t, x, y, p in zip(events["t"], events["x"], events["y"], events["p"]):
            handle.write(f"{int(t)} {int(x)} {int(y)} {int(p)}\n")


def write_center_labels_txt(annotation_rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in sorted(annotation_rows, key=lambda item: int(item["timestamp_us"])):
            ellipse = row["ellipse_sensor_xywht"]
            close = 1 if bool(row.get("closed_eye_flag", False)) else 0
            handle.write(f"{int(row['timestamp_us'])},{float(ellipse[0]):.2f},{float(ellipse[1]):.2f},{close}\n")


def write_ellipse_labels_txt(annotation_rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in sorted(annotation_rows, key=lambda item: int(item["timestamp_us"])):
            ellipse = row["ellipse_sensor_xywht"]
            theta_deg = float(np.rad2deg(float(ellipse[4])))
            handle.write(
                f"{int(row['timestamp_us'])} {float(ellipse[0]):.2f} {float(ellipse[1]):.2f} "
                f"{float(ellipse[2]):.2f} {float(ellipse[3]):.2f} {theta_deg:.2f}\n"
            )


def export_facet_style_dataset(
    *,
    canonical_root: Path,
    indexes_root: Path,
    manifests_root: Path,
    facet_root: Path,
    export_frames: str,
    overwrite: bool,
) -> dict[str, Any]:
    split_by_session: dict[str, str] = {}
    for split in ("train", "val", "test"):
        manifest_path = manifests_root / f"{split}_manifest.jsonl"
        if not manifest_path.exists():
            continue
        for row in read_jsonl(manifest_path):
            split_by_session.setdefault(str(row["session_key"]), split)

    session_rows = read_jsonl(indexes_root / "sessions.jsonl")
    exported_sessions = 0
    for session_row in session_rows:
        session_key = str(session_row["session_key"])
        split = split_by_session.get(session_key)
        if split is None:
            continue
        annotation_rows = read_jsonl(resolve_stored_path(canonical_root, session_row["annotation_store_path"]))
        if not annotation_rows:
            continue

        package = read_json(resolve_stored_path(canonical_root, session_row["session_package_path"]))
        base_name = session_key.replace("/", "_")
        split_root = facet_root / split
        events_txt_path = split_root / "data" / f"{base_name}_events.txt"
        center_txt_path = split_root / "label" / f"{base_name}_centers.txt"
        ellipse_txt_path = split_root / "ellipse" / f"{base_name}_ellipses.txt"
        events_npz_path = resolve_stored_path(canonical_root, package["events_npz"])

        if overwrite or not events_txt_path.exists():
            write_events_txt_from_npz(events_npz_path, events_txt_path)
        if overwrite or not center_txt_path.exists():
            write_center_labels_txt(annotation_rows, center_txt_path)
        if overwrite or not ellipse_txt_path.exists():
            write_ellipse_labels_txt(annotation_rows, ellipse_txt_path)

        if export_frames != "none":
            frame_source = resolve_stored_path(canonical_root, annotation_rows[0]["frame_path"]).parent
            frame_dest = split_root / "frames" / base_name
            maybe_link_or_copy(frame_source, frame_dest, mode=export_frames, overwrite=overwrite)

        write_json(
            {
                "session_key": session_key,
                "split": split,
                "events_txt": str(events_txt_path),
                "centers_txt": str(center_txt_path),
                "ellipse_txt": str(ellipse_txt_path),
                "annotation_count": len(annotation_rows),
            },
            split_root / "metadata" / f"{base_name}.json",
        )
        exported_sessions += 1

    summary = {
        "canonical_root": str(canonical_root),
        "facet_root": str(facet_root),
        "exported_sessions": exported_sessions,
        "export_frames": export_frames,
    }
    write_json(summary, facet_root / "facet_export_summary.json")
    return summary


def prepare_facet_gsam_dataset(
    *,
    raw_root: str | Path,
    grounded_sam_root: str | Path,
    canonical_root: str | Path,
    manifests_root: str | Path,
    splits_root: str | Path,
    facet_root: str | Path,
    annotation_mode: str,
    link_mode: str,
    export_frames: str,
    overwrite: bool,
    official_only: bool,
    annotation_stride: int,
    annotation_offset: int,
    frame_limit: int | None,
    users: Sequence[int] | None,
    eyes: Sequence[str],
    split_scheme: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    gsam_kwargs: dict[str, Any],
) -> dict[str, Any]:
    raw_root = Path(raw_root).resolve()
    grounded_sam_root = Path(grounded_sam_root).resolve()
    canonical_root = Path(canonical_root).resolve()
    manifests_root = Path(manifests_root).resolve()
    splits_root = Path(splits_root).resolve()
    facet_root = Path(facet_root).resolve()

    assert_output_isolated(canonical_root, raw_root, grounded_sam_root)
    assert_output_isolated(manifests_root, raw_root, grounded_sam_root)
    assert_output_isolated(splits_root, raw_root, grounded_sam_root)
    assert_output_isolated(facet_root, raw_root, grounded_sam_root)

    ensure_dir(canonical_root)
    ensure_dir(manifests_root)
    ensure_dir(splits_root)
    ensure_dir(facet_root)
    include_users = None if not users else {int(value) for value in users}

    canonical_summary = build_canonical_from_raw(
        raw_root=raw_root,
        canonical_root=canonical_root,
        grounded_sam_root=grounded_sam_root,
        annotation_mode=annotation_mode,
        link_mode=link_mode,
        overwrite=overwrite,
        official_only=official_only,
        annotation_stride=annotation_stride,
        annotation_offset=annotation_offset,
        frame_limit=frame_limit,
        include_users=include_users,
        eyes=eyes,
        gsam_kwargs=gsam_kwargs,
    )
    index_summary = build_session_index(canonical_root=canonical_root, indexes_root=canonical_root / "indexes")
    manifest_summary = build_manifests(
        canonical_root=canonical_root,
        indexes_root=canonical_root / "indexes",
        manifests_root=manifests_root,
        split_scheme=split_scheme,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        resize_policy="facet_square_direct",
        target_size_wh=(256, 256),
        event_policy="fixed_count",
        time_bin_us=5000,
        event_count_target=5000,
        accumulation="causal_linear",
        causal_weight_power=1.0,
    )
    split_summary = build_split_views(manifests_root=manifests_root, splits_root=splits_root)
    facet_summary = export_facet_style_dataset(
        canonical_root=canonical_root,
        indexes_root=canonical_root / "indexes",
        manifests_root=manifests_root,
        facet_root=facet_root,
        export_frames=export_frames,
        overwrite=overwrite,
    )
    summary = {
        "raw_root": str(raw_root),
        "grounded_sam_root": str(grounded_sam_root),
        "canonical": canonical_summary,
        "index": index_summary,
        "manifests": manifest_summary,
        "splits": split_summary,
        "facet_export": facet_summary,
        "non_destructive_guarantee": {
            "raw_root_read_only": str(raw_root),
            "grounded_sam_root_read_only": str(grounded_sam_root),
        },
    }
    write_json(summary, canonical_root / "prepare_facet_gsam_summary.json")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a FACET-style dataset from raw EV-Eye data using Grounded-SAM without modifying input roots")
    parser.add_argument("--raw-root", type=str, default=r"E:\WSL\Shared\dataset\Eye\EV_Eye\raw_data\Data_davis")
    parser.add_argument("--grounded-sam-root", type=str, default=r"E:\WSL\Shared\ETRI_SYNC\HBTXR\annotation_tools\Grounded-Segment-Anything-main")
    parser.add_argument("--canonical-root", type=str, default=str(PROJECT_ROOT / "data" / "_internal" / "canonical"))
    parser.add_argument("--manifests-root", type=str, default=str(PROJECT_ROOT / "data" / "_internal" / "manifests"))
    parser.add_argument("--splits-root", type=str, default=str(PROJECT_ROOT / "data" / "splits"))
    parser.add_argument("--facet-root", type=str, default=str(PROJECT_ROOT / "data" / "facet_style"))
    parser.add_argument("--annotation-mode", choices=["csv_only", "csv_then_gsam", "grounded_sam"], default="csv_then_gsam")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--export-frames", choices=["none", "symlink", "copy"], default="none")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--all-sessions", action="store_true", help="Process non-standard sessions too")
    parser.add_argument("--annotation-stride", type=int, default=100, help="Annotate every Nth frame with Grounded-SAM; use 1 for dense annotation")
    parser.add_argument("--annotation-offset", type=int, default=0)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--user", action="append", type=int, default=[])
    parser.add_argument("--eye", action="append", choices=["left", "right"], default=[])
    parser.add_argument("--split-scheme", choices=["exgaze_with_val", "exgaze", "random"], default="exgaze_with_val")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--grounding-config", type=str, default=None)
    parser.add_argument("--grounded-checkpoint", type=str, default=None)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--sam-version", type=str, default="vit_h")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text-prompt", type=str, default="pupil")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--min-mask-area-px", type=int, default=40)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    grounded_sam_root = Path(args.grounded_sam_root).resolve()
    grounding_config = args.grounding_config or str(grounded_sam_root / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
    grounded_checkpoint = args.grounded_checkpoint or str(grounded_sam_root / "groundingdino_swint_ogc.pth")
    sam_checkpoint = args.sam_checkpoint or str(grounded_sam_root / "sam_vit_h_4b8939.pth")
    return prepare_facet_gsam_dataset(
        raw_root=args.raw_root,
        grounded_sam_root=grounded_sam_root,
        canonical_root=args.canonical_root,
        manifests_root=args.manifests_root,
        splits_root=args.splits_root,
        facet_root=args.facet_root,
        annotation_mode=args.annotation_mode,
        link_mode=args.link_mode,
        export_frames=args.export_frames,
        overwrite=bool(args.overwrite),
        official_only=not bool(args.all_sessions),
        annotation_stride=int(args.annotation_stride),
        annotation_offset=int(args.annotation_offset),
        frame_limit=args.frame_limit,
        users=args.user,
        eyes=args.eye or ["left", "right"],
        split_scheme=args.split_scheme,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        gsam_kwargs={
            "grounding_config": grounding_config,
            "grounded_checkpoint": grounded_checkpoint,
            "sam_checkpoint": sam_checkpoint,
            "sam_version": args.sam_version,
            "device": args.device,
            "text_prompt": args.text_prompt,
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "min_mask_area_px": int(args.min_mask_area_px),
        },
    )


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
