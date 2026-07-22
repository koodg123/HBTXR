from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.io import read_json, read_jsonl, write_json, write_jsonl

SENSOR_WIDTH = 346
SENSOR_HEIGHT = 240
OFFICIAL_SESSION_CODES = {"101", "102", "201", "202"}
LEFT_BAD_PIXELS = ((158, 27), (324, 27))


def canonical_user_name(user_id: int) -> str:
    return f"user{user_id:02d}"


def session_dir_to_code(session_name: str) -> str:
    parts = session_name.replace("session_", "").split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected session name: {session_name}")
    return "".join(parts)


def code_to_session_dir(session_code: str) -> str:
    if len(session_code) != 3:
        raise ValueError(f"Unexpected session code: {session_code}")
    return f"session_{session_code[0]}_{session_code[1]}_{session_code[2]}"


def is_official_session_code(session_code: str) -> bool:
    return session_code in OFFICIAL_SESSION_CODES


def parse_frame_filename(filename: str) -> Tuple[Optional[int], int]:
    stem = Path(filename).stem
    if "_" in stem:
        idx_str, ts_str = stem.split("_", 1)
        try:
            frame_idx = int(idx_str)
        except ValueError:
            frame_idx = None
        return frame_idx, int(float(ts_str))
    return None, int(float(stem))


def try_parse_frame_filename(filename: str) -> Optional[Tuple[Optional[int], int]]:
    try:
        return parse_frame_filename(filename)
    except (TypeError, ValueError):
        return None


@dataclass
class EllipseAnnotation:
    frame_filename: str
    frame_idx: Optional[int]
    timestamp_us: int
    ellipse_xywht: Tuple[float, float, float, float, float]
    region_id: int = 0
    label: str = "pupil"

    def to_dict(self) -> Dict:
        return {
            "frame_filename": self.frame_filename,
            "frame_idx": self.frame_idx,
            "timestamp_us": self.timestamp_us,
            "ellipse_xywht": list(self.ellipse_xywht),
            "region_id": self.region_id,
            "label": self.label,
        }


@dataclass
class EyeRegion:
    x: int
    y: int
    w: int
    h: int

    def to_list(self) -> List[int]:
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def to_box(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class FrameRecord:
    filename: str
    timestamp_us: int
    frame_idx: Optional[int]

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "timestamp_us": self.timestamp_us,
            "frame_idx": self.frame_idx,
        }


@dataclass
class SessionLayout:
    raw_session_dir: Path
    frames_dir: Optional[Path]
    events_dir: Optional[Path]
    event_file: Optional[Path]
    annotation_csv: Optional[Path]
    session_code: str
    is_official_session: bool
    warnings: List[str]

    def to_dict(self) -> Dict:
        return {
            "raw_session_dir": str(self.raw_session_dir),
            "frames_dir": str(self.frames_dir) if self.frames_dir else None,
            "events_dir": str(self.events_dir) if self.events_dir else None,
            "event_file": str(self.event_file) if self.event_file else None,
            "annotation_csv": str(self.annotation_csv) if self.annotation_csv else None,
            "session_code": self.session_code,
            "is_official_session": self.is_official_session,
            "warnings": list(self.warnings),
        }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_link_or_copy(src: Path, dst: Path, mode: str = "symlink", overwrite: bool = False) -> None:
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
            dst.symlink_to(src.resolve())
            return
        except OSError:
            mode = "copy"
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    elif mode == "skip":
        return
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def collect_users(raw_root: Path) -> List[Path]:
    return sorted([p for p in raw_root.glob("user*") if p.is_dir()])


def discover_frames_dir(session_dir: Path) -> Optional[Path]:
    candidates = [
        session_dir / "frames",
        session_dir / "events" / "frames",
        session_dir / "Frames",
        session_dir / "events" / "Frames",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    png_dirs = [p for p in session_dir.rglob("*") if p.is_dir() and any(p.glob("*.png"))]
    if not png_dirs:
        return None
    return sorted(png_dirs, key=lambda p: (len(p.parts), str(p)))[0]


def discover_events_dir(session_dir: Path) -> Optional[Path]:
    candidates = [session_dir / "events", session_dir]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            if (candidate / "events.npz").exists() or (candidate / "events.txt").exists():
                return candidate
    return None


def find_event_file(session_dir: Path) -> Path:
    candidates = [
        session_dir / "events" / "events.npz",
        session_dir / "events" / "events.txt",
        session_dir / "events.npz",
        session_dir / "events.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in session_dir.rglob("events.npz"):
        return candidate
    for candidate in session_dir.rglob("events.txt"):
        return candidate
    raise FileNotFoundError(f"No events file found under {session_dir}")


def find_annotation_csv(session_dir: Path, user_id: int) -> Optional[Path]:
    candidates = [
        session_dir / f"user_{user_id}.csv",
        session_dir / "events" / f"user_{user_id}.csv",
        session_dir / f"user{user_id}.csv",
        session_dir / "events" / f"user{user_id}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    csvs = sorted(session_dir.rglob("*.csv"))
    for candidate in csvs:
        if candidate.name in {f"user_{user_id}.csv", f"user{user_id}.csv"}:
            return candidate
    return csvs[0] if csvs else None


def discover_session_layout(session_dir: Path, user_id: int) -> SessionLayout:
    warnings: List[str] = []
    session_code = session_dir_to_code(session_dir.name)
    frames_dir = discover_frames_dir(session_dir)
    if frames_dir is None:
        warnings.append("frames_dir_missing")
    elif frames_dir.parent.name == "events":
        warnings.append("frames_nested_under_events")

    events_dir = discover_events_dir(session_dir)
    if events_dir is None:
        warnings.append("events_dir_missing")

    event_file = None
    try:
        event_file = find_event_file(session_dir)
    except FileNotFoundError:
        warnings.append("event_file_missing")

    annotation_csv = find_annotation_csv(session_dir, user_id=user_id)
    if annotation_csv is None:
        warnings.append("annotation_csv_missing")

    if not is_official_session_code(session_code):
        warnings.append("nonstandard_session_code")

    return SessionLayout(
        raw_session_dir=session_dir,
        frames_dir=frames_dir,
        events_dir=events_dir,
        event_file=event_file,
        annotation_csv=annotation_csv,
        session_code=session_code,
        is_official_session=is_official_session_code(session_code),
        warnings=warnings,
    )


def scan_dataset_layout(raw_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for user_dir in collect_users(raw_root):
        try:
            user_id = int(user_dir.name.replace("user", ""))
        except ValueError:
            continue
        for eye in ("left", "right"):
            eye_dir = user_dir / eye
            if not eye_dir.exists():
                continue
            for session_dir in sorted(eye_dir.glob("session_*_*_*")):
                layout = discover_session_layout(session_dir, user_id=user_id)
                row = layout.to_dict()
                row.update({"user_id": user_id, "eye": eye, "session_dir_name": session_dir.name})
                rows.append(row)
    return rows


def load_events_from_txt(path: Path, eye: Optional[str] = None) -> Dict[str, np.ndarray]:
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


def save_events_npz(events: Dict[str, np.ndarray], dst_path: Path) -> None:
    ensure_dir(dst_path.parent)
    np.savez_compressed(
        dst_path,
        t=np.asarray(events["t"], dtype=np.int64),
        x=np.asarray(events["x"], dtype=np.int16),
        y=np.asarray(events["y"], dtype=np.int16),
        p=np.asarray(events["p"], dtype=np.int8),
    )


def _int_from_row(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def parse_via_csv_with_report(
    csv_path: Path,
    known_frame_filenames: Optional[Sequence[str]] = None,
    known_frame_stems: Optional[Sequence[str]] = None,
) -> Tuple[List[EllipseAnnotation], Dict[str, int]]:
    report = {
        "csv_exists": int(csv_path.exists()),
        "rows_total": 0,
        "rows_region_positive": 0,
        "groups_total": 0,
        "annotations_kept": 0,
        "groups_skipped_bad_filename": 0,
        "groups_skipped_unknown_frame": 0,
        "groups_skipped_nonellipse": 0,
        "groups_skipped_bad_shape": 0,
        "groups_skipped_bad_values": 0,
    }
    if not csv_path.exists():
        return [], report

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    report["rows_total"] = len(rows)
    if not rows or "region_count" not in rows[0]:
        return [], report

    valid_rows = [row for row in rows if _int_from_row(row, "region_count", 0) > 0]
    report["rows_region_positive"] = len(valid_rows)
    annotations: List[EllipseAnnotation] = []
    known_frame_filenames = set(known_frame_filenames or [])
    known_frame_stems = set(known_frame_stems or [])

    i = 0
    while i < len(valid_rows):
        row = valid_rows[i]
        frame_filename = str(row.get("filename", "")).strip()
        region_count = max(1, _int_from_row(row, "region_count", 1))
        report["groups_total"] += 1
        parsed = try_parse_frame_filename(frame_filename)
        if parsed is None:
            stem = Path(frame_filename).stem
            if known_frame_filenames and frame_filename not in known_frame_filenames and stem not in known_frame_stems:
                report["groups_skipped_unknown_frame"] += 1
            else:
                report["groups_skipped_bad_filename"] += 1
            i += region_count
            continue

        frame_idx, timestamp_us = parsed
        if known_frame_filenames and frame_filename not in known_frame_filenames and Path(frame_filename).stem not in known_frame_stems:
            report["groups_skipped_unknown_frame"] += 1
            i += region_count
            continue

        chosen = None
        chosen_region_id = 0
        saw_bad_shape = False
        for j in range(region_count):
            if i + j >= len(valid_rows):
                break
            candidate = valid_rows[i + j]
            try:
                shape = json.loads(candidate["region_shape_attributes"])
            except Exception:
                saw_bad_shape = True
                continue
            if shape.get("name") not in (None, "ellipse"):
                continue
            chosen = shape
            chosen_region_id = _int_from_row(candidate, "region_id", 0)
            break

        if chosen is None:
            if saw_bad_shape:
                report["groups_skipped_bad_shape"] += 1
            else:
                report["groups_skipped_nonellipse"] += 1
            i += region_count
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
                    region_id=chosen_region_id,
                    label="pupil",
                )
            )
            report["annotations_kept"] += 1
        except Exception:
            report["groups_skipped_bad_values"] += 1
        i += region_count

    annotations.sort(key=lambda ann: ann.timestamp_us)
    return annotations, report


def collect_frame_records(frames_dir: Path) -> List[FrameRecord]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing frames directory: {frames_dir}")
    records: List[FrameRecord] = []
    for image_path in sorted(frames_dir.glob("*.png")):
        frame_idx, timestamp_us = parse_frame_filename(image_path.name)
        records.append(FrameRecord(image_path.name, timestamp_us, frame_idx))
    return records


def derive_eye_region(
    annotations: Sequence[EllipseAnnotation],
    sensor_size: Tuple[int, int] = (SENSOR_WIDTH, SENSOR_HEIGHT),
    margin_px: int = 24,
    margin_left_px: int | None = None,
    margin_right_px: int | None = None,
    margin_top_px: int | None = None,
    margin_bottom_px: int | None = None,
    target_aspect_wh: float = 256.0 / 160.0,
) -> EyeRegion:
    sensor_w, sensor_h = sensor_size
    if not annotations:
        return EyeRegion(0, 0, sensor_w, sensor_h)

    margin_left = int(margin_px if margin_left_px is None else margin_left_px)
    margin_right = int(margin_px if margin_right_px is None else margin_right_px)
    margin_top = int(margin_px if margin_top_px is None else margin_top_px)
    margin_bottom = int(margin_px if margin_bottom_px is None else margin_bottom_px)

    xs0, ys0, xs1, ys1 = [], [], [], []
    for ann in annotations:
        x, y, w, h, _ = ann.ellipse_xywht
        xs0.append(x - w / 2.0 - margin_left)
        ys0.append(y - h / 2.0 - margin_top)
        xs1.append(x + w / 2.0 + margin_right)
        ys1.append(y + h / 2.0 + margin_bottom)

    x0 = max(0.0, min(xs0))
    y0 = max(0.0, min(ys0))
    x1 = min(float(sensor_w), max(xs1))
    y1 = min(float(sensor_h), max(ys1))
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)

    current_ratio = w / h
    if current_ratio < target_aspect_wh:
        new_w = h * target_aspect_wh
        delta = new_w - w
        x0 -= delta / 2.0
        x1 += delta / 2.0
    else:
        new_h = w / target_aspect_wh
        delta = new_h - h
        y0 -= delta / 2.0
        y1 += delta / 2.0

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > sensor_w:
        x0 -= (x1 - sensor_w)
        x1 = sensor_w
    if y1 > sensor_h:
        y0 -= (y1 - sensor_h)
        y1 = sensor_h

    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(float(sensor_w), x1)
    y1 = min(float(sensor_h), y1)

    roi = EyeRegion(x=int(round(x0)), y=int(round(y0)), w=int(round(x1 - x0)), h=int(round(y1 - y0)))
    roi.w = max(1, roi.w)
    roi.h = max(1, roi.h)
    return roi


def ellipse_sensor_to_roi(ellipse_xywht: Sequence[float], eye_region: EyeRegion) -> List[float]:
    x, y, w, h, theta = ellipse_xywht
    return [
        float(x) - float(eye_region.x),
        float(y) - float(eye_region.y),
        float(w),
        float(h),
        float(theta),
    ]


def rasterize_ellipse_mask(ellipse_xywht: Sequence[float], image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy, ew, eh, theta = ellipse_xywht
    cos_t, sin_t = math.cos(float(theta)), math.sin(float(theta))
    x = xx - float(cx)
    y = yy - float(cy)
    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t
    mask = (xr / max(float(ew) / 2.0, 1e-6)) ** 2 + (yr / max(float(eh) / 2.0, 1e-6)) ** 2 <= 1.0
    return mask.astype(np.uint8)


def crop_mask(mask: np.ndarray, eye_region: EyeRegion) -> np.ndarray:
    x0, y0, x1, y1 = eye_region.to_box()
    return mask[y0:y1, x0:x1].astype(np.uint8)
