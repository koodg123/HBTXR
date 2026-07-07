#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.groundedsam_build import build_groundedsam_runtime
from hbtxr.preprocess.annotation_groundedsam import mask_bbox_xywh, mask_to_ellipse_xywht
from hbtxr.preprocess.groundedsam_eye_region_bbox import (
    _box_area_ratio_xyxy,
    _count_touched_borders,
    _eye_bbox_rejection_reason,
    _xyxy_to_xywh_sensor,
)
from hbtxr.preprocess.io_utils import (
    collect_frame_records,
    collect_users,
    discover_session_layout,
    is_official_session_code,
    maybe_link_or_copy,
    session_dir_to_code,
)
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.utils.io import read_json, write_json


SENSOR_WIDTH = 346
SENSOR_HEIGHT = 240
FULLFRAME_SIZE = (SENSOR_WIDTH, SENSOR_HEIGHT)
PAIR_GAP_PX = 8
PANEL_LABEL_HEIGHT_PX = 22
PAIR_COLUMNS = 4


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random raw frames per session, detect Eye ROI with bbox-only Grounded-SAM, "
            "rerun prompted pupil inference inside the cropped Eye ROI, and save overlay previews."
        ),
    )
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--options-config", type=str, default="exps/configs/groundedsam/roi_crop_prompted_pupil_v1.json")
    parser.add_argument("--raw-samples-root", type=str, default="dataset/raw_samples/roi_crop_prompted_all_sessions_v1")
    parser.add_argument("--workspace-root", type=str, default="workspace/roi_crop_prompted_preview_all_sessions_v1")
    parser.add_argument("--logs-root", type=str, default="dataset/logs")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="stable_random",
        choices=["stable_random", "consecutive_center", "consecutive_start"],
    )
    parser.add_argument("--seed-prefix", type=str, default="roi-crop-prompted-preview-v1")
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--groundingdino-config", type=str, default=None)
    parser.add_argument("--groundingdino-checkpoint", type=str, default=None)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--sam-encoder-version", type=str, default="vit_h")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--user-id", type=int, action="append", default=None)
    parser.add_argument("--eye", type=str, choices=["left", "right", "both"], default="both")
    parser.add_argument("--session-code", type=str, action="append", default=None)
    parser.add_argument("--include-nonstandard-sessions", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _normalize_session_codes(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if value.startswith("session_"):
            value = session_dir_to_code(value)
        elif value.startswith("session"):
            value = value[len("session") :]
            value = value.replace("_", "")
        if len(value) != 3 or not value.isdigit():
            raise ValueError(f"Unsupported session code: {raw}")
        out.add(value)
    return out


def _normalize_eyes(value: str) -> tuple[str, ...]:
    eye = str(value).strip().lower()
    if eye == "both":
        return ("left", "right")
    if eye in {"left", "right"}:
        return (eye,)
    raise ValueError(f"Unsupported eye selector: {value}")


def _iter_target_sessions(
    *,
    raw_root: Path,
    user_ids: set[int] | None,
    eyes: tuple[str, ...],
    session_codes: set[str] | None,
    include_nonstandard_sessions: bool,
) -> list[tuple[int, str, Path]]:
    rows: list[tuple[int, str, Path]] = []
    for user_dir in collect_users(raw_root):
        try:
            user_id = int(user_dir.name.replace("user", ""))
        except ValueError:
            continue
        if user_ids is not None and user_id not in user_ids:
            continue
        for eye in eyes:
            eye_dir = user_dir / eye
            if not eye_dir.exists():
                continue
            for session_dir in sorted(eye_dir.glob("session_*_*_*")):
                session_code = session_dir_to_code(session_dir.name)
                if session_codes is not None and session_code not in session_codes:
                    continue
                if not include_nonstandard_sessions and not is_official_session_code(session_code):
                    continue
                rows.append((user_id, eye, session_dir))
    return rows


def _stable_sample_indices(*, session_key: str, total_frames: int, sample_count: int, seed_prefix: str) -> tuple[list[int], str]:
    if total_frames <= 0:
        return [], f"{seed_prefix}|{session_key}|empty"
    seed_text = f"{seed_prefix}|{session_key}|n={sample_count}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    rng = np.random.default_rng(seed)
    k = min(int(sample_count), int(total_frames))
    indices = sorted(int(v) for v in rng.choice(total_frames, size=k, replace=False).tolist())
    return indices, seed_text


def _consecutive_sample_indices(*, total_frames: int, sample_count: int, mode: str) -> list[int]:
    if total_frames <= 0:
        return []
    k = min(int(sample_count), int(total_frames))
    if k <= 0:
        return []
    if mode == "consecutive_start" or total_frames <= k:
        start = 0
    elif mode == "consecutive_center":
        start = max(0, (int(total_frames) - k) // 2)
    else:
        raise ValueError(f"Unsupported consecutive sample mode: {mode}")
    return list(range(start, min(int(total_frames), start + k)))


def _resolve_sample_indices(
    *,
    session_key: str,
    total_frames: int,
    sample_count: int,
    seed_prefix: str,
    sample_mode: str,
) -> tuple[list[int], str]:
    if sample_mode == "stable_random":
        return _stable_sample_indices(
            session_key=session_key,
            total_frames=total_frames,
            sample_count=sample_count,
            seed_prefix=seed_prefix,
        )
    indices = _consecutive_sample_indices(total_frames=total_frames, sample_count=sample_count, mode=sample_mode)
    return indices, f"{sample_mode}|{session_key}|n={sample_count}"


def _clip_crop_rect(bbox_xywh: list[float], image_size_wh: tuple[int, int]) -> list[int]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    image_w, image_h = image_size_wh
    x0 = max(0, min(int(math.floor(x)), image_w - 1))
    y0 = max(0, min(int(math.floor(y)), image_h - 1))
    x1 = max(x0 + 1, min(int(math.ceil(x + w)), image_w))
    y1 = max(y0 + 1, min(int(math.ceil(y + h)), image_h))
    return [x0, y0, x1, y1]


def _sensor_bbox_from_local(local_xywh: list[float], crop_rect_xyxy: list[int]) -> list[float]:
    x0, y0, _, _ = crop_rect_xyxy
    x, y, w, h = [float(v) for v in local_xywh]
    return [float(x0) + x, float(y0) + y, float(w), float(h)]


def _sensor_ellipse_from_local(local_xywht: list[float], crop_rect_xyxy: list[int]) -> list[float]:
    x0, y0, _, _ = crop_rect_xyxy
    cx, cy, w, h, theta = [float(v) for v in local_xywht]
    return [float(x0) + cx, float(y0) + cy, float(w), float(h), float(theta)]


def _normalize_size_wh(value: Any) -> tuple[int, int] | None:
    if value in (None, [], ()):
        return None
    if isinstance(value, (int, float)):
        size = int(value)
        if size <= 0:
            return None
        return (size, size)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width = int(value[0])
        height = int(value[1])
        if width <= 0 or height <= 0:
            return None
        return (width, height)
    raise ValueError(f"Unsupported crop resize spec: {value}")


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _draw_rotated_ellipse(
    draw: ImageDraw.ImageDraw,
    ellipse_xywht: list[float],
    *,
    outline: str,
    width: int = 2,
    steps: int = 72,
) -> None:
    cx, cy, ew, eh, theta = [float(v) for v in ellipse_xywht]
    a = max(1.0, ew / 2.0)
    b = max(1.0, eh / 2.0)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    points: list[tuple[float, float]] = []
    for index in range(int(max(12, steps))):
        angle = 2.0 * math.pi * float(index) / float(max(12, steps))
        ex = a * math.cos(angle)
        ey = b * math.sin(angle)
        px = cx + ex * cos_t - ey * sin_t
        py = cy + ex * sin_t + ey * cos_t
        points.append((px, py))
    if points:
        points.append(points[0])
        draw.line(points, fill=outline, width=width)


def _apply_mask_overlay(
    image_rgb: Image.Image,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: int,
    offset_xy: tuple[int, int] = (0, 0),
) -> Image.Image:
    base = image_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_u8 = (mask > 0).astype(np.uint8) * int(alpha)
    mask_image = Image.fromarray(mask_u8, mode="L")
    color_patch = Image.new("RGBA", mask_image.size, tuple(color) + (int(alpha),))
    x0, y0 = [int(v) for v in offset_xy]
    overlay.paste(color_patch, (x0, y0), mask_image)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _resize_rgb(image_rgb: np.ndarray, size_wh: tuple[int, int] | None) -> np.ndarray:
    if size_wh is None:
        return image_rgb.copy()
    target_w, target_h = [int(v) for v in size_wh]
    if image_rgb.shape[1] == target_w and image_rgb.shape[0] == target_h:
        return image_rgb.copy()
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    return np.asarray(image.resize((target_w, target_h), resample=Image.BILINEAR))


def _resize_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    target_w, target_h = [int(v) for v in size_wh]
    if mask.shape[1] == target_w and mask.shape[0] == target_h:
        return (mask > 0).astype(np.uint8)
    image = Image.fromarray(((mask > 0).astype(np.uint8) * 255), mode="L")
    resized = image.resize((target_w, target_h), resample=Image.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def _scale_xyxy_between_sizes(
    xyxy: np.ndarray,
    *,
    from_size_wh: tuple[int, int],
    to_size_wh: tuple[int, int],
) -> np.ndarray:
    from_w, from_h = [float(max(1, v)) for v in from_size_wh]
    to_w, to_h = [float(max(1, v)) for v in to_size_wh]
    sx = to_w / from_w
    sy = to_h / from_h
    scaled = np.asarray(
        [
            float(xyxy[0]) * sx,
            float(xyxy[1]) * sy,
            float(xyxy[2]) * sx,
            float(xyxy[3]) * sy,
        ],
        dtype=np.float32,
    )
    scaled[0] = float(np.clip(scaled[0], 0.0, max(0.0, to_w - 1.0)))
    scaled[1] = float(np.clip(scaled[1], 0.0, max(0.0, to_h - 1.0)))
    scaled[2] = float(np.clip(scaled[2], scaled[0] + 1.0, to_w))
    scaled[3] = float(np.clip(scaled[3], scaled[1] + 1.0, to_h))
    return scaled


def _resolve_stage_crop_resize_wh(
    *,
    pupil_options: dict[str, Any],
    default_resize_wh: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if "crop_resize_wh" in pupil_options:
        return _normalize_size_wh(pupil_options.get("crop_resize_wh"))
    return default_resize_wh


def _project_candidate_to_source(
    candidate: dict[str, Any],
    *,
    source_size_wh: tuple[int, int],
) -> dict[str, Any] | None:
    if bool(candidate.get("segmented_on_source_image")):
        source_mask = np.asarray(candidate["mask"]).astype(np.uint8)
        source_bbox = mask_bbox_xywh(source_mask)
        source_ellipse = mask_to_ellipse_xywht(source_mask)
        if source_bbox is None or source_ellipse is None:
            return None
        input_size_wh = tuple(int(v) for v in candidate.get("input_size_wh", source_size_wh))
        input_mask = np.asarray(candidate.get("mask_input", _resize_mask(source_mask, input_size_wh))).astype(np.uint8)
        input_bbox = candidate.get("bbox_xywh_input")
        input_ellipse = candidate.get("ellipse_xywht_input")
        if input_bbox is None:
            input_bbox = mask_bbox_xywh(input_mask)
        if input_ellipse is None:
            input_ellipse = mask_to_ellipse_xywht(input_mask)
        projected = dict(candidate)
        projected["mask_input"] = input_mask
        projected["bbox_xywh_input"] = None if input_bbox is None else [float(v) for v in input_bbox]
        projected["ellipse_xywht_input"] = None if input_ellipse is None else [float(v) for v in input_ellipse]
        projected["mask"] = source_mask
        projected["bbox_xywh"] = source_bbox
        projected["ellipse_xywht"] = source_ellipse
        return projected
    input_mask = np.asarray(candidate["mask"]).astype(np.uint8)
    source_mask = _resize_mask(input_mask, source_size_wh)
    source_bbox = mask_bbox_xywh(source_mask)
    source_ellipse = mask_to_ellipse_xywht(source_mask)
    if source_bbox is None or source_ellipse is None:
        return None
    projected = dict(candidate)
    projected["mask_input"] = input_mask
    projected["bbox_xywh_input"] = [float(v) for v in candidate["bbox_xywh"]]
    projected["ellipse_xywht_input"] = [float(v) for v in candidate["ellipse_xywht"]]
    projected["mask"] = source_mask
    projected["bbox_xywh"] = source_bbox
    projected["ellipse_xywht"] = source_ellipse
    return projected


def _save_crop_overlay(
    *,
    crop_rgb: np.ndarray,
    pupil_mask: np.ndarray | None,
    pupil_bbox_xywh: list[float] | None,
    pupil_ellipse_xywht: list[float] | None,
    caption: str,
    output_path: Path,
) -> Path:
    image = Image.fromarray(crop_rgb.astype(np.uint8), mode="RGB")
    if pupil_mask is not None:
        image = _apply_mask_overlay(image, pupil_mask, color=(255, 64, 64), alpha=96)
    draw = ImageDraw.Draw(image)
    if pupil_bbox_xywh is not None:
        _draw_xywh(draw, pupil_bbox_xywh, outline="#ffd84d", width=2)
    if pupil_ellipse_xywht is not None:
        _draw_rotated_ellipse(draw, pupil_ellipse_xywht, outline="#ff4040", width=2)
    draw.text((6, 6), caption, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _save_full_overlay(
    *,
    image_rgb: np.ndarray,
    eye_region_bbox_xywh: list[float] | None,
    crop_mask: np.ndarray | None,
    crop_rect_xyxy: list[int] | None,
    pupil_bbox_xywh_sensor: list[float] | None,
    pupil_ellipse_xywht_sensor: list[float] | None,
    caption: str,
    output_path: Path,
) -> Path:
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    if crop_mask is not None and crop_rect_xyxy is not None:
        image = _apply_mask_overlay(
            image,
            crop_mask,
            color=(255, 64, 64),
            alpha=80,
            offset_xy=(int(crop_rect_xyxy[0]), int(crop_rect_xyxy[1])),
        )
    draw = ImageDraw.Draw(image)
    if eye_region_bbox_xywh is not None:
        _draw_xywh(draw, eye_region_bbox_xywh, outline="#00ff88", width=2)
    if pupil_bbox_xywh_sensor is not None:
        _draw_xywh(draw, pupil_bbox_xywh_sensor, outline="#ffd84d", width=2)
    if pupil_ellipse_xywht_sensor is not None:
        _draw_rotated_ellipse(draw, pupil_ellipse_xywht_sensor, outline="#ff4040", width=2)
    draw.text((6, 6), caption, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _fit_image_to_box(image: Image.Image, size_wh: tuple[int, int], *, background_rgb: tuple[int, int, int] = (16, 16, 16)) -> Image.Image:
    target_w, target_h = [int(v) for v in size_wh]
    canvas = Image.new("RGB", (target_w, target_h), background_rgb)
    src = image.copy()
    src.thumbnail((target_w, target_h))
    offset = ((target_w - src.width) // 2, (target_h - src.height) // 2)
    canvas.paste(src, offset)
    return canvas


def _compose_pair_panel(
    *,
    panel_index: int,
    full_overlay_path: Path,
    crop_overlay_path: Path | None,
    crop_rect_xyxy: list[int] | None,
) -> Image.Image:
    full_image = Image.open(full_overlay_path).convert("RGB")
    crop_image = (
        Image.open(crop_overlay_path).convert("RGB")
        if crop_overlay_path is not None and crop_overlay_path.exists()
        else Image.new("RGB", FULLFRAME_SIZE, (18, 18, 18))
    )
    crop_padded = _fit_image_to_box(crop_image, FULLFRAME_SIZE)
    panel_w = FULLFRAME_SIZE[0] * 2 + PAIR_GAP_PX
    panel_h = FULLFRAME_SIZE[1] + PANEL_LABEL_HEIGHT_PX
    panel = Image.new("RGB", (panel_w, panel_h), (10, 10, 10))
    panel.paste(full_image.resize(FULLFRAME_SIZE), (0, PANEL_LABEL_HEIGHT_PX))
    panel.paste(crop_padded, (FULLFRAME_SIZE[0] + PAIR_GAP_PX, PANEL_LABEL_HEIGHT_PX))
    draw = ImageDraw.Draw(panel)
    label = f"{panel_index:02d}"
    if crop_rect_xyxy is not None:
        crop_w = int(crop_rect_xyxy[2] - crop_rect_xyxy[0])
        crop_h = int(crop_rect_xyxy[3] - crop_rect_xyxy[1])
        label = f"{label} | crop {crop_w}x{crop_h}"
    draw.text((6, 4), label, fill=(255, 255, 255))
    draw.text((FULLFRAME_SIZE[0] + PAIR_GAP_PX + 6, 4), "crop", fill=(255, 255, 255))
    return panel


def _compose_contact_sheet_pairs(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    columns: int = PAIR_COLUMNS,
) -> Path:
    if not rows:
        raise ValueError("rows is required for contact sheet generation")
    panels = [
        _compose_pair_panel(
            panel_index=int(row["panel"]),
            full_overlay_path=Path(row["full_overlay_path"]),
            crop_overlay_path=None if row.get("crop_overlay_path") is None else Path(row["crop_overlay_path"]),
            crop_rect_xyxy=row.get("crop_rect_xyxy_sensor"),
        )
        for row in rows
    ]
    panel_w, panel_h = panels[0].size
    n_columns = max(1, int(columns))
    n_rows = int(math.ceil(len(panels) / float(n_columns)))
    sheet = Image.new("RGB", (panel_w * n_columns, panel_h * n_rows), (0, 0, 0))
    for index, panel in enumerate(panels):
        row_idx = index // n_columns
        col_idx = index % n_columns
        sheet.paste(panel, (col_idx * panel_w, row_idx * panel_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _detect_eye_region_bbox(
    *,
    image_rgb: np.ndarray,
    runtime: Any,
    eye_options: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    image_bgr = image_rgb[:, :, ::-1].copy()
    detections = runtime.model.predict_with_classes(
        image=image_bgr,
        classes=list(eye_options["classes"]),
        box_threshold=float(eye_options["box_threshold"]),
        text_threshold=float(eye_options["text_threshold"]),
    )
    xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4), dtype=np.float32)))
    conf = np.asarray(getattr(detections, "confidence", np.empty((0,), dtype=np.float32)))
    class_id = np.asarray(getattr(detections, "class_id", np.empty((0,), dtype=np.int64)))
    if xyxy.size == 0 or conf.size == 0:
        return None, "no_eye_detections"
    keep = runtime._torchvision.ops.nms(
        runtime._torch.from_numpy(xyxy),
        runtime._torch.from_numpy(conf.astype(np.float32)),
        float(eye_options["nms_threshold"]),
    ).cpu().numpy().tolist()
    image_size_wh = (image_rgb.shape[1], image_rgb.shape[0])
    priority_map = {name: idx for idx, name in enumerate(eye_options["classes"])}
    best_key = None
    best_idx = None
    best_class_name = None
    best_rejected_key = None
    best_rejected_reason = None
    for idx in keep:
        raw_class_id = class_id[idx] if idx < class_id.shape[0] else None
        if raw_class_id is None or int(raw_class_id) < 0 or int(raw_class_id) >= len(eye_options["classes"]):
            class_name = "unknown"
        else:
            class_name = str(eye_options["classes"][int(raw_class_id)])
        area_ratio = _box_area_ratio_xyxy(xyxy[idx], image_size_wh=image_size_wh)
        border_touches = _count_touched_borders(
            xyxy[idx],
            image_size_wh=image_size_wh,
            border_margin_px=int(eye_options["border_margin_px"]),
        )
        key = (priority_map.get(class_name, 999), -float(conf[idx]), border_touches, area_ratio)
        rejection_reason = _eye_bbox_rejection_reason(
            xyxy[idx],
            image_size_wh=image_size_wh,
            max_box_area_ratio=float(eye_options["max_box_area_ratio"]),
            border_margin_px=int(eye_options["border_margin_px"]),
            max_border_touches=int(eye_options["max_border_touches"]),
        )
        if rejection_reason is not None:
            if best_rejected_key is None or key < best_rejected_key:
                best_rejected_key = key
                best_rejected_reason = rejection_reason
            continue
        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(idx)
            best_class_name = class_name
    if best_idx is None:
        return None, best_rejected_reason or "no_valid_eye_box"
    xyxy_best = np.asarray(xyxy[best_idx], dtype=np.float32)
    return {
        "eye_region_bbox_xywh_sensor": _xyxy_to_xywh_sensor(xyxy_best, image_size_wh=image_size_wh),
        "gsam_box_xyxy": [float(v) for v in xyxy_best.tolist()],
        "gsam_box_confidence": float(conf[best_idx]),
        "gsam_class_name": best_class_name or "eye",
        "gsam_box_area_ratio": _box_area_ratio_xyxy(xyxy_best, image_size_wh=image_size_wh),
        "gsam_border_touches": _count_touched_borders(
            xyxy_best,
            image_size_wh=image_size_wh,
            border_margin_px=int(eye_options["border_margin_px"]),
        ),
    }, None


def _rank_detection_candidates(
    *,
    detections: Any,
    image_shape: tuple[int, int, int],
    classes: list[str],
    runtime: Any,
    nms_threshold: float,
) -> list[tuple[int, str]]:
    xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4), dtype=np.float32)))
    conf = np.asarray(getattr(detections, "confidence", np.empty((0,), dtype=np.float32)))
    class_id = np.asarray(getattr(detections, "class_id", np.empty((0,), dtype=np.int64)))
    if xyxy.size == 0 or conf.size == 0:
        return []
    keep = runtime._torchvision.ops.nms(
        runtime._torch.from_numpy(xyxy),
        runtime._torch.from_numpy(conf.astype(np.float32)),
        float(nms_threshold),
    ).cpu().numpy().tolist()
    priority_map = {name: idx for idx, name in enumerate(classes)}
    image_area = float(max(1, image_shape[0] * image_shape[1]))
    ranked: list[tuple[tuple[float, ...], int, str]] = []
    for idx in keep:
        raw_class_id = int(class_id[idx]) if idx < class_id.shape[0] else 0
        class_name = classes[raw_class_id] if 0 <= raw_class_id < len(classes) else "unknown"
        area = max(0.0, float((xyxy[idx][2] - xyxy[idx][0]) * (xyxy[idx][3] - xyxy[idx][1]))) / image_area
        key = (priority_map.get(class_name, 999), -float(conf[idx]), area)
        ranked.append((key, int(idx), class_name))
    ranked.sort(key=lambda item: item[0])
    return [(idx, class_name) for _key, idx, class_name in ranked]


def _annotate_crop_with_pupil_options(
    *,
    source_rgb: np.ndarray,
    runtime: Any,
    pupil_options: dict[str, Any],
    default_resize_wh: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    classes = [str(item).strip() for item in pupil_options.get("classes", []) if str(item).strip()]
    if not classes:
        raise ValueError("pupil stage classes are required")
    proposal_size_wh = _resolve_stage_crop_resize_wh(pupil_options=pupil_options, default_resize_wh=default_resize_wh)
    proposal_rgb = _resize_rgb(source_rgb, proposal_size_wh)
    proposal_size_wh = (int(proposal_rgb.shape[1]), int(proposal_rgb.shape[0]))
    source_size_wh = (int(source_rgb.shape[1]), int(source_rgb.shape[0]))
    segment_on_source_image = bool(pupil_options.get("segment_on_source_image", False))
    proposal_bgr = proposal_rgb[:, :, ::-1].copy()
    detections = runtime.model.predict_with_classes(
        image=proposal_bgr,
        classes=classes,
        box_threshold=float(pupil_options["box_threshold"]),
        text_threshold=float(pupil_options["text_threshold"]),
    )
    ranked = _rank_detection_candidates(
        detections=detections,
        image_shape=proposal_rgb.shape,
        classes=classes,
        runtime=runtime,
        nms_threshold=float(pupil_options["nms_threshold"]),
    )
    if not ranked:
        return []
    max_candidate_boxes = int(max(1, pupil_options.get("max_candidate_boxes", 1)))
    results: list[dict[str, Any]] = []
    for candidate_rank, (idx, class_name) in enumerate(ranked[:max_candidate_boxes], start=1):
        proposal_xyxy = np.asarray(detections.xyxy[idx], dtype=np.float32)
        source_xyxy = _scale_xyxy_between_sizes(
            proposal_xyxy,
            from_size_wh=proposal_size_wh,
            to_size_wh=source_size_wh,
        )
        confidence = float(detections.confidence[idx])
        segment_rgb = source_rgb if segment_on_source_image else proposal_rgb
        segment_xyxy = source_xyxy if segment_on_source_image else proposal_xyxy
        mask, mask_score = runtime._segment(segment_rgb, segment_xyxy)
        if int(mask.sum()) < int(pupil_options["min_mask_area"]):
            continue
        bbox = mask_bbox_xywh(mask)
        ellipse = mask_to_ellipse_xywht(mask)
        if bbox is None or ellipse is None:
            continue
        mask_input = np.asarray(mask).astype(np.uint8)
        bbox_input = bbox
        ellipse_input = ellipse
        if segment_on_source_image:
            mask_input = _resize_mask(mask, proposal_size_wh)
            bbox_input = mask_bbox_xywh(mask_input)
            ellipse_input = mask_to_ellipse_xywht(mask_input)
        results.append(
            {
                "mask": mask,
                "bbox_xywh": bbox,
                "ellipse_xywht": ellipse,
                "class_name": class_name or classes[0],
                "box_xyxy": [float(v) for v in proposal_xyxy.tolist()],
                "box_xyxy_source": [float(v) for v in source_xyxy.tolist()],
                "box_confidence": confidence,
                "mask_score": mask_score,
                "stage_name": str(pupil_options.get("name") or classes[0]),
                "classes_used": classes,
                "candidate_rank": int(candidate_rank),
                "max_candidate_boxes": int(max_candidate_boxes),
                "input_size_wh": [int(v) for v in proposal_size_wh],
                "segmented_on_source_image": bool(segment_on_source_image),
                "mask_input": mask_input,
                "bbox_xywh_input": None if bbox_input is None else [float(v) for v in bbox_input],
                "ellipse_xywht_input": None if ellipse_input is None else [float(v) for v in ellipse_input],
            }
        )
    return results


def _passes_geometry_gate(
    *,
    crop_mask_area_ratio: float,
    crop_bbox_fill_width_ratio: float,
    crop_bbox_fill_height_ratio: float,
    filters: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    gate = dict(filters or {})
    if not gate:
        return True, []
    reasons: list[str] = []
    mask_area_ratio_max = gate.get("crop_mask_area_ratio_max")
    if mask_area_ratio_max is not None and crop_mask_area_ratio > float(mask_area_ratio_max):
        reasons.append(f"crop_mask_area_ratio>{float(mask_area_ratio_max):.3f}")
    fill_max = gate.get("crop_bbox_fill_max")
    max_fill = max(float(crop_bbox_fill_width_ratio), float(crop_bbox_fill_height_ratio))
    if fill_max is not None and max_fill > float(fill_max):
        reasons.append(f"crop_bbox_fill_max>{float(fill_max):.3f}")
    return len(reasons) == 0, reasons


def _fail_like_preference_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    max_fill = max(
        float(candidate.get("crop_bbox_fill_width_ratio", 0.0)),
        float(candidate.get("crop_bbox_fill_height_ratio", 0.0)),
    )
    return (
        float(candidate.get("crop_mask_area_ratio", 0.0)),
        float(max_fill),
        -float(candidate.get("box_confidence", 0.0)),
        float(candidate.get("candidate_rank", 999)),
    )


def _session_paths(
    *,
    raw_samples_root: Path,
    workspace_root: Path,
    user_id: int,
    eye: str,
    session_code: str,
) -> dict[str, Path]:
    session_rel = Path("sessions") / f"user{int(user_id):02d}" / eye / f"session_{session_code}"
    sample_root = raw_samples_root / session_rel
    workspace_dir = workspace_root / session_rel
    return {
        "session_rel": session_rel,
        "sample_root": sample_root,
        "workspace_dir": workspace_dir,
        "sample_frames_dir": sample_root / "frames",
        "eye_masks_dir": workspace_dir / "eye_region_masks",
        "pupil_masks_dir": workspace_dir / "pupil_masks",
        "roi_crops_dir": workspace_dir / "roi_crops",
        "crop_overlays_dir": workspace_dir / "roi_crop_overlays",
        "full_overlays_dir": workspace_dir / "fullframe_backproj_overlays",
        "summary_path": workspace_dir / "summary.json",
        "contact_sheet_pairs": workspace_dir / "contact_sheet_pairs.png",
    }


def _rasterize_bbox_mask_xywh(bbox_xywh: list[float], *, image_size_wh: tuple[int, int]) -> np.ndarray:
    image_w, image_h = [int(v) for v in image_size_wh]
    x, y, w, h = [float(v) for v in bbox_xywh]
    x0 = max(0, min(int(math.floor(x)), image_w - 1))
    y0 = max(0, min(int(math.floor(y)), image_h - 1))
    x1 = max(x0 + 1, min(int(math.ceil(x + w)), image_w))
    y1 = max(y0 + 1, min(int(math.ceil(y + h)), image_h))
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _save_mask_png(mask: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(output_path)
    return output_path


def _process_session(
    *,
    runtime: Any,
    raw_root: Path,
    raw_samples_root: Path,
    workspace_root: Path,
    user_id: int,
    eye: str,
    session_dir: Path,
    sample_count: int,
    sample_mode: str,
    seed_prefix: str,
    link_mode: str,
    overwrite: bool,
    eye_options: dict[str, Any],
    pupil_option_chain: list[dict[str, Any]],
    optional_post_filters: dict[str, Any],
    retry_pupil_on_fail_like: bool,
    crop_resize_wh: tuple[int, int] | None,
) -> dict[str, Any]:
    session_code = session_dir_to_code(session_dir.name)
    session_key = f"user{int(user_id):02d}/{eye}/session_{session_code}"
    paths = _session_paths(
        raw_samples_root=raw_samples_root,
        workspace_root=workspace_root,
        user_id=user_id,
        eye=eye,
        session_code=session_code,
    )
    summary_path = paths["summary_path"]
    if summary_path.exists() and not overwrite:
        cached = read_json(summary_path)
        cached["reused_existing_summary"] = True
        return cached

    layout = discover_session_layout(session_dir, user_id=user_id)
    session_summary: dict[str, Any] = {
        "experiment": "groundedsam_eye_roi_to_prompted_pupil_preview_batch",
        "session_key": session_key,
        "user_id": int(user_id),
        "eye": eye,
        "session_code": session_code,
        "raw_root": str(raw_root),
        "raw_session_dir": str(session_dir.resolve()),
        "frames_dir": None if layout.frames_dir is None else str(layout.frames_dir.resolve()),
        "sample_root": str(paths["sample_root"].resolve()),
        "workspace_dir": str(paths["workspace_dir"].resolve()),
        "warnings": list(layout.warnings),
        "options": {
            "sample_count": int(sample_count),
            "sample_mode": str(sample_mode),
            "seed_prefix": seed_prefix,
            "link_mode": link_mode,
            "eye_stage": dict(eye_options),
            "pupil_stage_chain": [dict(item) for item in pupil_option_chain],
            "optional_post_filters": dict(optional_post_filters),
            "retry_pupil_on_fail_like": bool(retry_pupil_on_fail_like),
            "crop_resize_wh": None if crop_resize_wh is None else [int(v) for v in crop_resize_wh],
        },
    }
    if layout.frames_dir is None or not layout.frames_dir.exists():
        session_summary.update(
            {
                "status": "skipped_frames_dir_missing",
                "n_frames_total": 0,
                "n_sampled": 0,
                "n_eye_detected": 0,
                "n_completed": 0,
                "n_eye_failed": 0,
                "n_pupil_failed": 0,
                "n_fail_like_mask_ratio": 0,
                "n_fail_like_fill_ratio": 0,
                "n_fail_like_union": 0,
                "frames": [],
                "contact_sheet_pairs": None,
            }
        )
        write_json(session_summary, summary_path)
        return session_summary

    frame_records = collect_frame_records(layout.frames_dir)
    sample_indices, sample_seed = _resolve_sample_indices(
        session_key=session_key,
        total_frames=len(frame_records),
        sample_count=sample_count,
        seed_prefix=seed_prefix,
        sample_mode=sample_mode,
    )
    sampled_records = [frame_records[index] for index in sample_indices]
    session_summary["sample_seed"] = sample_seed
    session_summary["n_frames_total"] = len(frame_records)
    session_summary["sample_indices"] = sample_indices
    session_summary["sample_filenames"] = [record.filename for record in sampled_records]

    rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    eye_failures = 0
    pupil_failures = 0
    eye_detected = 0

    for panel_index, record in enumerate(sampled_records, start=1):
        frame_path = layout.frames_dir / record.filename
        sample_frame_dst = paths["sample_frames_dir"] / record.filename
        maybe_link_or_copy(frame_path, sample_frame_dst, mode=link_mode, overwrite=overwrite)

        frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
        eye_result, eye_status = _detect_eye_region_bbox(
            image_rgb=frame_rgb,
            runtime=runtime,
            eye_options=eye_options,
        )

        row: dict[str, Any] = {
            "panel": int(panel_index),
            "frame_filename": record.filename,
            "frame_idx": record.frame_idx,
            "timestamp_us": int(record.timestamp_us),
            "raw_frame_path": str(frame_path.resolve()),
            "sample_frame_path": str(sample_frame_dst.absolute()),
            "eye_region_mask_path": None,
            "pupil_mask_path": None,
            "eye_region_bbox_xywh_sensor": None,
            "crop_rect_xyxy_sensor": None,
            "crop_source_size_wh": None,
            "crop_input_size_wh": None,
            "crop_image_path": None,
            "crop_overlay_path": None,
            "full_overlay_path": None,
            "crop_pupil_bbox_xywh_local": None,
            "crop_pupil_ellipse_xywht_local": None,
            "crop_pupil_bbox_xywh_local_input": None,
            "crop_pupil_ellipse_xywht_local_input": None,
            "sensor_pupil_bbox_xywh": None,
            "sensor_pupil_ellipse_xywht": None,
            "crop_mask_area_ratio": None,
            "crop_bbox_fill_width_ratio": None,
            "crop_bbox_fill_height_ratio": None,
            "predicted_class_name": None,
            "predicted_box_confidence": None,
            "eye_detection_status": eye_status,
            "fail_like_mask_ratio": False,
            "fail_like_fill_ratio": False,
            "fail_like_union": False,
        }

        if eye_result is None:
            eye_failures += 1
            caption = f"{session_key} | {record.filename} | eye: {eye_status or 'failed'}"
            full_overlay_path = paths["full_overlays_dir"] / record.filename
            _save_full_overlay(
                image_rgb=frame_rgb,
                eye_region_bbox_xywh=None,
                crop_mask=None,
                crop_rect_xyxy=None,
                pupil_bbox_xywh_sensor=None,
                pupil_ellipse_xywht_sensor=None,
                caption=caption,
                output_path=full_overlay_path,
            )
            row["status"] = "eye_failed"
            row["full_overlay_path"] = str(full_overlay_path.resolve())
            rows.append(row)
            continue

        eye_detected += 1
        eye_bbox_xywh = [float(v) for v in eye_result["eye_region_bbox_xywh_sensor"]]
        eye_mask_path = paths["eye_masks_dir"] / Path(record.filename).with_suffix(".png").name
        eye_mask = _rasterize_bbox_mask_xywh(
            eye_bbox_xywh,
            image_size_wh=(frame_rgb.shape[1], frame_rgb.shape[0]),
        )
        _save_mask_png(eye_mask, eye_mask_path)
        crop_rect_xyxy = _clip_crop_rect(eye_bbox_xywh, image_size_wh=(frame_rgb.shape[1], frame_rgb.shape[0]))
        x0, y0, x1, y1 = crop_rect_xyxy
        crop_source_rgb = frame_rgb[y0:y1, x0:x1].copy()
        crop_source_size_wh = (int(crop_source_rgb.shape[1]), int(crop_source_rgb.shape[0]))
        crop_input_rgb = crop_source_rgb.copy()
        crop_input_size_wh = (int(crop_input_rgb.shape[1]), int(crop_input_rgb.shape[0]))
        crop_image_path = paths["roi_crops_dir"] / record.filename
        crop_image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop_input_rgb.astype(np.uint8), mode="RGB").save(crop_image_path)

        pupil_result = None
        accepted_pupil_result = None
        best_fail_like_result = None
        attempted_pupil_stage_names: list[str] = []
        row["eye_region_bbox_xywh_sensor"] = eye_bbox_xywh
        row["eye_region_mask_path"] = str(eye_mask_path.resolve())
        row["crop_rect_xyxy_sensor"] = crop_rect_xyxy
        row["crop_source_size_wh"] = [int(v) for v in crop_source_size_wh]
        row["crop_input_size_wh"] = [int(v) for v in crop_input_size_wh]
        row["crop_image_path"] = str(crop_image_path.resolve())

        for pupil_options in pupil_option_chain:
            candidates = _annotate_crop_with_pupil_options(
                source_rgb=crop_source_rgb,
                runtime=runtime,
                pupil_options=pupil_options,
                default_resize_wh=crop_resize_wh,
            )
            attempted_pupil_stage_names.append(str(pupil_options.get("name") or pupil_options.get("classes", ["pupil"])[0]))
            if not candidates:
                continue
            stage_best_fail_like = None
            for raw_candidate in candidates:
                candidate = _project_candidate_to_source(raw_candidate, source_size_wh=crop_source_size_wh)
                if candidate is None:
                    continue
                local_bbox = [float(v) for v in candidate["bbox_xywh"]]
                crop_mask = np.asarray(candidate["mask"]).astype(np.uint8)
                crop_area = float(max(1, crop_source_rgb.shape[0] * crop_source_rgb.shape[1]))
                crop_mask_area_ratio = float(crop_mask.sum()) / crop_area
                crop_bbox_fill_width_ratio = float(local_bbox[2]) / float(max(1, crop_source_rgb.shape[1]))
                crop_bbox_fill_height_ratio = float(local_bbox[3]) / float(max(1, crop_source_rgb.shape[0]))
                fail_like_mask_ratio = crop_mask_area_ratio > float(optional_post_filters.get("crop_mask_area_ratio_max", 0.30))
                fail_like_fill_ratio = max(crop_bbox_fill_width_ratio, crop_bbox_fill_height_ratio) > float(
                    optional_post_filters.get("crop_bbox_fill_max", 0.85),
                )
                fail_like_union = bool(fail_like_mask_ratio or fail_like_fill_ratio)
                candidate["crop_mask_area_ratio"] = crop_mask_area_ratio
                candidate["crop_bbox_fill_width_ratio"] = crop_bbox_fill_width_ratio
                candidate["crop_bbox_fill_height_ratio"] = crop_bbox_fill_height_ratio
                candidate["fail_like_mask_ratio"] = fail_like_mask_ratio
                candidate["fail_like_fill_ratio"] = fail_like_fill_ratio
                candidate["fail_like_union"] = fail_like_union
                acceptance_filters = dict(pupil_options.get("acceptance_filters") or {})
                stage_gate_passed, stage_gate_reasons = _passes_geometry_gate(
                    crop_mask_area_ratio=crop_mask_area_ratio,
                    crop_bbox_fill_width_ratio=crop_bbox_fill_width_ratio,
                    crop_bbox_fill_height_ratio=crop_bbox_fill_height_ratio,
                    filters=acceptance_filters,
                )
                candidate["acceptance_filters"] = acceptance_filters
                candidate["stage_gate_passed"] = bool(stage_gate_passed)
                candidate["stage_gate_reasons"] = list(stage_gate_reasons)
                if acceptance_filters and not stage_gate_passed:
                    continue
                if not fail_like_union:
                    accepted_pupil_result = candidate
                    break
                if stage_best_fail_like is None or _fail_like_preference_key(candidate) < _fail_like_preference_key(stage_best_fail_like):
                    stage_best_fail_like = candidate
            if accepted_pupil_result is not None:
                break
            if stage_best_fail_like is not None and (
                best_fail_like_result is None
                or _fail_like_preference_key(stage_best_fail_like) < _fail_like_preference_key(best_fail_like_result)
            ):
                best_fail_like_result = stage_best_fail_like
            if not retry_pupil_on_fail_like:
                break

        pupil_result = accepted_pupil_result or best_fail_like_result

        if pupil_result is None:
            pupil_failures += 1
            crop_caption = f"{session_key} | {record.filename} | pupil: failed"
            full_caption = f"{session_key} | {record.filename} | pupil: failed"
            crop_overlay_path = paths["crop_overlays_dir"] / record.filename
            full_overlay_path = paths["full_overlays_dir"] / record.filename
            _save_crop_overlay(
                crop_rgb=crop_input_rgb,
                pupil_mask=None,
                pupil_bbox_xywh=None,
                pupil_ellipse_xywht=None,
                caption=crop_caption,
                output_path=crop_overlay_path,
            )
            _save_full_overlay(
                image_rgb=frame_rgb,
                eye_region_bbox_xywh=eye_bbox_xywh,
                crop_mask=None,
                crop_rect_xyxy=crop_rect_xyxy,
                pupil_bbox_xywh_sensor=None,
                pupil_ellipse_xywht_sensor=None,
                caption=full_caption,
                output_path=full_overlay_path,
            )
            row["status"] = "pupil_failed"
            row["crop_overlay_path"] = str(crop_overlay_path.resolve())
            row["full_overlay_path"] = str(full_overlay_path.resolve())
            row["attempted_pupil_stage_names"] = attempted_pupil_stage_names
            rows.append(row)
            continue

        local_bbox = [float(v) for v in pupil_result["bbox_xywh"]]
        local_ellipse = [float(v) for v in pupil_result["ellipse_xywht"]]
        input_bbox = None if pupil_result.get("bbox_xywh_input") is None else [float(v) for v in pupil_result["bbox_xywh_input"]]
        input_ellipse = (
            None if pupil_result.get("ellipse_xywht_input") is None else [float(v) for v in pupil_result["ellipse_xywht_input"]]
        )
        accepted_input_size_wh = tuple(int(v) for v in pupil_result.get("input_size_wh", crop_source_size_wh))
        accepted_input_rgb = _resize_rgb(crop_source_rgb, accepted_input_size_wh)
        crop_input_rgb = accepted_input_rgb
        crop_input_size_wh = (int(crop_input_rgb.shape[1]), int(crop_input_rgb.shape[0]))
        Image.fromarray(crop_input_rgb.astype(np.uint8), mode="RGB").save(crop_image_path)
        sensor_bbox = _sensor_bbox_from_local(local_bbox, crop_rect_xyxy)
        sensor_ellipse = _sensor_ellipse_from_local(local_ellipse, crop_rect_xyxy)
        crop_mask = np.asarray(pupil_result["mask"]).astype(np.uint8)
        crop_input_mask = np.asarray(pupil_result.get("mask_input", pupil_result["mask"])).astype(np.uint8)
        sensor_mask = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)
        sensor_mask[y0:y1, x0:x1] = np.maximum(sensor_mask[y0:y1, x0:x1], crop_mask.astype(np.uint8) * 255)
        pupil_mask_path = paths["pupil_masks_dir"] / Path(record.filename).with_suffix(".png").name
        _save_mask_png(sensor_mask, pupil_mask_path)
        crop_mask_area_ratio = float(pupil_result["crop_mask_area_ratio"])
        crop_bbox_fill_width_ratio = float(pupil_result["crop_bbox_fill_width_ratio"])
        crop_bbox_fill_height_ratio = float(pupil_result["crop_bbox_fill_height_ratio"])
        fail_like_mask_ratio = bool(pupil_result["fail_like_mask_ratio"])
        fail_like_fill_ratio = bool(pupil_result["fail_like_fill_ratio"])
        fail_like_union = bool(pupil_result["fail_like_union"])
        class_name = str(pupil_result["class_name"])
        class_counter[class_name] += 1

        crop_caption = (
            f"{session_key} | {record.filename} | {class_name} "
            f"| conf={float(pupil_result['box_confidence']):.3f} | ratio={crop_mask_area_ratio:.3f}"
        )
        full_caption = (
            f"{session_key} | {record.filename} | eye+{class_name} "
            f"| conf={float(pupil_result['box_confidence']):.3f}"
        )
        crop_overlay_path = paths["crop_overlays_dir"] / record.filename
        full_overlay_path = paths["full_overlays_dir"] / record.filename
        _save_crop_overlay(
            crop_rgb=crop_input_rgb,
            pupil_mask=crop_input_mask,
            pupil_bbox_xywh=input_bbox,
            pupil_ellipse_xywht=input_ellipse,
            caption=crop_caption,
            output_path=crop_overlay_path,
        )
        _save_full_overlay(
            image_rgb=frame_rgb,
            eye_region_bbox_xywh=eye_bbox_xywh,
            crop_mask=crop_mask,
            crop_rect_xyxy=crop_rect_xyxy,
            pupil_bbox_xywh_sensor=sensor_bbox,
            pupil_ellipse_xywht_sensor=sensor_ellipse,
            caption=full_caption,
            output_path=full_overlay_path,
        )

        row.update(
            {
                "status": "completed",
                "crop_overlay_path": str(crop_overlay_path.resolve()),
                "full_overlay_path": str(full_overlay_path.resolve()),
                "crop_input_size_wh": [int(v) for v in crop_input_size_wh],
                "crop_image_path": str(crop_image_path.resolve()),
                "crop_pupil_bbox_xywh_local": local_bbox,
                "crop_pupil_ellipse_xywht_local": local_ellipse,
                "crop_pupil_bbox_xywh_local_input": input_bbox,
                "crop_pupil_ellipse_xywht_local_input": input_ellipse,
                "pupil_mask_path": str(pupil_mask_path.resolve()),
                "sensor_pupil_bbox_xywh": sensor_bbox,
                "sensor_pupil_ellipse_xywht": sensor_ellipse,
                "crop_mask_area_ratio": crop_mask_area_ratio,
                "crop_bbox_fill_width_ratio": crop_bbox_fill_width_ratio,
                "crop_bbox_fill_height_ratio": crop_bbox_fill_height_ratio,
                "predicted_class_name": class_name,
                "predicted_box_confidence": float(pupil_result["box_confidence"]),
                "predicted_pupil_stage_name": str(pupil_result.get("stage_name") or class_name),
                "predicted_candidate_rank": int(pupil_result.get("candidate_rank", 1)),
                "predicted_max_candidate_boxes": int(pupil_result.get("max_candidate_boxes", 1)),
                "attempted_pupil_stage_names": attempted_pupil_stage_names,
                "stage_gate_passed": bool(pupil_result.get("stage_gate_passed", True)),
                "stage_gate_reasons": list(pupil_result.get("stage_gate_reasons") or []),
                "eye_detection_status": "ok",
                "fail_like_mask_ratio": fail_like_mask_ratio,
                "fail_like_fill_ratio": fail_like_fill_ratio,
                "fail_like_union": fail_like_union,
            }
        )
        rows.append(row)

    contact_sheet_pairs = None
    if rows:
        contact_sheet_path = _compose_contact_sheet_pairs(
            rows=rows,
            output_path=paths["contact_sheet_pairs"],
        )
        contact_sheet_pairs = str(contact_sheet_path.resolve())

    n_completed = sum(1 for row in rows if row.get("status") == "completed")
    n_fail_like_mask = sum(1 for row in rows if bool(row.get("fail_like_mask_ratio")))
    n_fail_like_fill = sum(1 for row in rows if bool(row.get("fail_like_fill_ratio")))
    n_fail_like_union = sum(1 for row in rows if bool(row.get("fail_like_union")))

    session_summary.update(
        {
            "status": "completed",
            "n_sampled": len(rows),
            "n_eye_detected": int(eye_detected),
            "n_completed": int(n_completed),
            "n_eye_failed": int(eye_failures),
            "n_pupil_failed": int(pupil_failures),
            "n_fail_like_mask_ratio": int(n_fail_like_mask),
            "n_fail_like_fill_ratio": int(n_fail_like_fill),
            "n_fail_like_union": int(n_fail_like_union),
            "predicted_class_name_counts": dict(sorted(class_counter.items())),
            "frames": rows,
            "contact_sheet_pairs": contact_sheet_pairs,
        }
    )
    write_json(session_summary, summary_path)
    return session_summary


def _build_markdown_report(aggregate: dict[str, Any]) -> str:
    lines = [
        f"# ROI Crop Prompted Preview Batch Report ({aggregate['run_started_at']})",
        "",
        "## Summary",
        f"- sessions_discovered: {aggregate['n_sessions_discovered']}",
        f"- sessions_processed: {aggregate['n_sessions_processed']}",
        f"- sessions_reused: {aggregate['n_sessions_reused']}",
        f"- sampled_frames: {aggregate['n_sampled_frames_total']}",
        f"- completed_frames: {aggregate['n_completed_frames_total']}",
        f"- eye_failed_frames: {aggregate['n_eye_failed_frames_total']}",
        f"- pupil_failed_frames: {aggregate['n_pupil_failed_frames_total']}",
        f"- fail_like_union_frames: {aggregate['n_fail_like_union_frames_total']}",
        "",
        "## Worst Sessions By Fail-like Frames",
    ]
    for row in aggregate.get("top_sessions_by_fail_like_union", []):
        lines.append(
            f"- {row['session_key']}: fail_like_union={row['n_fail_like_union']}, "
            f"pupil_failed={row['n_pupil_failed']}, eye_failed={row['n_eye_failed']}",
        )
        if row.get("contact_sheet_pairs"):
            lines.append(f"  - preview: {row['contact_sheet_pairs']}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    if paths.groundedsam_root is None:
        raise ValueError("groundedsam_root is required. Pass --groundedsam-root or provide it in the paths config.")

    project_root = Path(paths.project_root).resolve()
    raw_root = Path(paths.raw_root).resolve()
    options_config = (project_root / args.options_config).resolve() if not Path(args.options_config).is_absolute() else Path(args.options_config).resolve()
    options = read_json(options_config)
    eye_options = dict(options["eye_stage"])
    pupil_options = dict(options["pupil_stage"])
    pupil_stage_fallbacks = [dict(item) for item in (options.get("pupil_stage_fallbacks") or [])]
    pupil_option_chain = [pupil_options] + pupil_stage_fallbacks
    optional_post_filters = dict(options.get("optional_post_filters") or {})
    retry_pupil_on_fail_like = bool(options.get("retry_pupil_on_fail_like", True))
    crop_resize_wh = _normalize_size_wh(options.get("crop_resize_wh"))

    raw_samples_root = (project_root / args.raw_samples_root).resolve() if not Path(args.raw_samples_root).is_absolute() else Path(args.raw_samples_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve() if not Path(args.workspace_root).is_absolute() else Path(args.workspace_root).resolve()
    logs_root = (project_root / args.logs_root).resolve() if not Path(args.logs_root).is_absolute() else Path(args.logs_root).resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    run_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_stamp = datetime.now().strftime("%Y-%m-%d")
    eye_tuple = _normalize_eyes(args.eye)
    user_ids = None if not args.user_id else {int(v) for v in args.user_id}
    session_codes = _normalize_session_codes(args.session_code)

    sessions = _iter_target_sessions(
        raw_root=raw_root,
        user_ids=user_ids,
        eyes=eye_tuple,
        session_codes=session_codes,
        include_nonstandard_sessions=bool(args.include_nonstandard_sessions),
    )
    if args.max_sessions is not None:
        sessions = sessions[: max(0, int(args.max_sessions))]

    run_options_snapshot = {
        "run_started_at": run_started_at,
        "options_config": str(options_config),
        "raw_root": str(raw_root),
        "raw_samples_root": str(raw_samples_root),
        "workspace_root": str(workspace_root),
        "logs_root": str(logs_root),
        "sample_count": int(args.sample_count),
        "sample_mode": str(args.sample_mode),
        "seed_prefix": args.seed_prefix,
        "link_mode": args.link_mode,
        "device": args.device,
        "user_ids": None if user_ids is None else sorted(user_ids),
        "eyes": list(eye_tuple),
        "session_codes": None if session_codes is None else sorted(session_codes),
        "include_nonstandard_sessions": bool(args.include_nonstandard_sessions),
        "max_sessions": args.max_sessions,
        "overwrite": bool(args.overwrite),
        "eye_stage": eye_options,
        "pupil_stage_chain": pupil_option_chain,
        "optional_post_filters": optional_post_filters,
        "retry_pupil_on_fail_like": bool(retry_pupil_on_fail_like),
        "crop_resize_wh": None if crop_resize_wh is None else [int(v) for v in crop_resize_wh],
    }
    options_snapshot_path = logs_root / f"{run_stamp}_roi_crop_prompted_preview_all_sessions_options.json"
    write_json(run_options_snapshot, options_snapshot_path)

    runtime = build_groundedsam_runtime(
        groundedsam_root=paths.groundedsam_root,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_encoder_version=args.sam_encoder_version,
        classes=tuple(pupil_options["classes"]),
        box_threshold=float(pupil_options["box_threshold"]),
        text_threshold=float(pupil_options["text_threshold"]),
        nms_threshold=float(pupil_options["nms_threshold"]),
        min_mask_area=int(pupil_options["min_mask_area"]),
        device=args.device,
    )

    started_perf = time.perf_counter()
    session_summaries: list[dict[str, Any]] = []
    n_reused = 0
    interrupted = False
    last_started_session_key: str | None = None

    try:
        for index, (user_id, eye, session_dir) in enumerate(sessions, start=1):
            session_key = f"user{int(user_id):02d}/{eye}/session_{session_dir_to_code(session_dir.name)}"
            last_started_session_key = session_key
            print(f"[{index}/{len(sessions)}] {session_key}", flush=True)
            summary = _process_session(
                runtime=runtime,
                raw_root=raw_root,
                raw_samples_root=raw_samples_root,
                workspace_root=workspace_root,
                user_id=user_id,
                eye=eye,
                session_dir=session_dir,
                sample_count=args.sample_count,
                sample_mode=str(args.sample_mode),
                seed_prefix=args.seed_prefix,
                link_mode=args.link_mode,
                overwrite=bool(args.overwrite),
                eye_options=eye_options,
                pupil_option_chain=pupil_option_chain,
                optional_post_filters=optional_post_filters,
                retry_pupil_on_fail_like=retry_pupil_on_fail_like,
                crop_resize_wh=crop_resize_wh,
            )
            if summary.get("reused_existing_summary"):
                n_reused += 1
            print(
                "  "
                + f"sampled={summary.get('n_sampled', 0)} "
                + f"completed={summary.get('n_completed', 0)} "
                + f"eye_failed={summary.get('n_eye_failed', 0)} "
                + f"pupil_failed={summary.get('n_pupil_failed', 0)} "
                + f"fail_like={summary.get('n_fail_like_union', 0)}",
                flush=True,
            )
            session_summaries.append(summary)
    except KeyboardInterrupt:
        interrupted = True
        print(f"[INTERRUPTED] last_started_session={last_started_session_key}", flush=True)

    elapsed_sec = time.perf_counter() - started_perf
    class_counter: Counter[str] = Counter()
    for summary in session_summaries:
        class_counter.update(summary.get("predicted_class_name_counts") or {})

    top_sessions = sorted(
        [
            {
                "session_key": summary["session_key"],
                "n_fail_like_union": int(summary.get("n_fail_like_union", 0)),
                "n_pupil_failed": int(summary.get("n_pupil_failed", 0)),
                "n_eye_failed": int(summary.get("n_eye_failed", 0)),
                "contact_sheet_pairs": summary.get("contact_sheet_pairs"),
            }
            for summary in session_summaries
        ],
        key=lambda item: (-item["n_fail_like_union"], -item["n_pupil_failed"], -item["n_eye_failed"], item["session_key"]),
    )[:20]

    aggregate = {
        "experiment": "groundedsam_eye_roi_to_prompted_pupil_preview_batch",
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_sec": elapsed_sec,
        "interrupted": bool(interrupted),
        "last_started_session_key": last_started_session_key,
        "options_config": str(options_config),
        "options_snapshot_path": str(options_snapshot_path),
        "raw_root": str(raw_root),
        "raw_samples_root": str(raw_samples_root),
        "workspace_root": str(workspace_root),
        "logs_root": str(logs_root),
        "device": args.device,
        "n_sessions_discovered": len(sessions),
        "n_sessions_processed": len(session_summaries),
        "n_sessions_reused": int(n_reused),
        "n_sessions_with_eye_failures": sum(1 for s in session_summaries if int(s.get("n_eye_failed", 0)) > 0),
        "n_sessions_with_pupil_failures": sum(1 for s in session_summaries if int(s.get("n_pupil_failed", 0)) > 0),
        "n_sessions_with_fail_like_union": sum(1 for s in session_summaries if int(s.get("n_fail_like_union", 0)) > 0),
        "n_sampled_frames_total": sum(int(s.get("n_sampled", 0)) for s in session_summaries),
        "n_completed_frames_total": sum(int(s.get("n_completed", 0)) for s in session_summaries),
        "n_eye_failed_frames_total": sum(int(s.get("n_eye_failed", 0)) for s in session_summaries),
        "n_pupil_failed_frames_total": sum(int(s.get("n_pupil_failed", 0)) for s in session_summaries),
        "n_fail_like_mask_ratio_frames_total": sum(int(s.get("n_fail_like_mask_ratio", 0)) for s in session_summaries),
        "n_fail_like_fill_ratio_frames_total": sum(int(s.get("n_fail_like_fill_ratio", 0)) for s in session_summaries),
        "n_fail_like_union_frames_total": sum(int(s.get("n_fail_like_union", 0)) for s in session_summaries),
        "predicted_class_name_counts": dict(sorted(class_counter.items())),
        "top_sessions_by_fail_like_union": top_sessions,
        "session_summary_paths": [str(Path(s["workspace_dir"]) / "summary.json") for s in session_summaries],
    }
    summary_suffix = "partial_summary" if interrupted else "summary"
    aggregate_path = logs_root / f"{run_stamp}_roi_crop_prompted_preview_all_sessions_{summary_suffix}.json"
    aggregate_md_path = logs_root / f"{run_stamp}_roi_crop_prompted_preview_all_sessions_{summary_suffix}.md"
    write_json(aggregate, aggregate_path)
    aggregate_md_path.write_text(_build_markdown_report(aggregate), encoding="utf-8")

    status_tag = "[DONE]" if not interrupted else "[PARTIAL]"
    print(
        status_tag
        + " "
        + f"sessions={aggregate['n_sessions_processed']} "
        + f"sampled_frames={aggregate['n_sampled_frames_total']} "
        + f"completed={aggregate['n_completed_frames_total']} "
        + f"eye_failed={aggregate['n_eye_failed_frames_total']} "
        + f"pupil_failed={aggregate['n_pupil_failed_frames_total']} "
        + f"fail_like={aggregate['n_fail_like_union_frames_total']} "
        + f"summary={aggregate_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
