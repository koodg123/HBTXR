#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from build_tsgss_frame_event_visualization import _load_frame_rgb, _read_session_summaries
from hbtxr.preprocess.io_utils import SENSOR_HEIGHT, SENSOR_WIDTH, load_events_from_txt, maybe_link_or_copy
from hbtxr.utils.io import write_json, write_jsonl


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build frame-guided derived event-feature interpolation for tsgss sampled sequences. "
            "This script does not synthesize raw event tuples; it generates polarity-split "
            "count-map features only."
        ),
    )
    parser.add_argument("--batch-root", type=str, default="tsgss/frame_interpolation")
    parser.add_argument(
        "--output-root",
        type=str,
        default="tsgss/guided_event_feature_interpolation",
    )
    parser.add_argument("--feature-width", type=int, default=128)
    parser.add_argument("--feature-height", type=int, default=96)
    parser.add_argument("--anchor-event-threshold", type=int, default=600)
    parser.add_argument("--guidance-strength", type=float, default=0.35)
    parser.add_argument("--guidance-blur-radius", type=float, default=1.4)
    parser.add_argument("--crop-mode", type=str, default="none", choices=["none", "density_adaptive"])
    parser.add_argument("--crop-min-events", type=int, default=64)
    parser.add_argument("--crop-quantile-low", type=float, default=0.05)
    parser.add_argument("--crop-quantile-high", type=float, default=0.95)
    parser.add_argument("--crop-margin-px", type=float, default=12.0)
    parser.add_argument("--crop-min-side-px", type=float, default=96.0)
    parser.add_argument("--sensor-width", type=int, default=SENSOR_WIDTH)
    parser.add_argument("--sensor-height", type=int, default=SENSOR_HEIGHT)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _sequence_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["sequence_index"]), int(row["timestamp_us"])


def _session_output_dirs(output_root: Path, *, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
    base = output_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}"
    return {
        "base": base,
        "assets": base / "assets",
        "manifests": base / "manifests",
    }


def _resolve_sequence_frame_path(batch_root: Path, row: dict[str, Any]) -> Path:
    path_text = str(row["sequence_frame_path"])
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (batch_root / path).resolve()


def _interval_indices(timestamps: np.ndarray, *, start_timestamp_us: int, end_timestamp_us: int) -> tuple[int, int]:
    start_idx = int(np.searchsorted(timestamps, int(start_timestamp_us), side="right"))
    end_idx = int(np.searchsorted(timestamps, int(end_timestamp_us), side="right"))
    return max(0, start_idx), max(start_idx, end_idx)


def _clip_xywh_to_sensor(
    xywh: tuple[float, float, float, float],
    *,
    sensor_width: int,
    sensor_height: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in xywh]
    max_w = max(1.0, float(sensor_width))
    max_h = max(1.0, float(sensor_height))
    w = min(max(1.0, w), max_w)
    h = min(max(1.0, h), max_h)
    x = min(max(0.0, x), max_w - w)
    y = min(max(0.0, y), max_h - h)
    return (x, y, w, h)


def _pad_crop_bounds(
    xyxy: tuple[float, float, float, float],
    *,
    sensor_width: int,
    sensor_height: int,
    margin_px: float,
    min_side_px: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    x0 -= float(margin_px)
    y0 -= float(margin_px)
    x1 += float(margin_px)
    y1 += float(margin_px)
    width = x1 - x0
    height = y1 - y0
    target_w = max(float(min_side_px), width)
    target_h = max(float(min_side_px), height)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    return _clip_xywh_to_sensor(
        (cx - target_w * 0.5, cy - target_h * 0.5, target_w, target_h),
        sensor_width=int(sensor_width),
        sensor_height=int(sensor_height),
    )


def _crop_image_to_roi(image: Image.Image, roi_xywh: tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = [float(v) for v in roi_xywh]
    left = int(max(0, math.floor(x)))
    top = int(max(0, math.floor(y)))
    right = int(max(left + 1, math.ceil(x + w)))
    bottom = int(max(top + 1, math.ceil(y + h)))
    return image.crop((left, top, right, bottom))


def _crop_points_to_roi(
    xs: np.ndarray,
    ys: np.ndarray,
    roi_xywh: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(xs) <= 0 or len(ys) <= 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=bool)
    x, y, w, h = [float(v) for v in roi_xywh]
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    inside = (
        (xs >= x)
        & (xs < x + w)
        & (ys >= y)
        & (ys < y + h)
    )
    if not np.any(inside):
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, inside
    return xs[inside] - x, ys[inside] - y, inside


def _frame_guidance_bbox(
    start_frame: Image.Image,
    end_frame: Image.Image,
    *,
    sensor_width: int,
    sensor_height: int,
    crop_margin_px: float,
    crop_min_side_px: float,
) -> tuple[tuple[float, float, float, float], dict[str, Any]] | None:
    start_gray = start_frame.convert("L").resize((int(sensor_width), int(sensor_height)), resample=Image.Resampling.BILINEAR)
    end_gray = end_frame.convert("L").resize((int(sensor_width), int(sensor_height)), resample=Image.Resampling.BILINEAR)
    diff = np.abs(np.asarray(end_gray, dtype=np.float32) - np.asarray(start_gray, dtype=np.float32))
    if float(diff.max()) <= 0.0:
        return None
    positive = diff[diff > 0.0]
    if positive.size <= 0:
        return None
    threshold = float(np.quantile(positive, 0.85))
    active = diff >= threshold
    if int(np.count_nonzero(active)) < 32:
        return None
    ys, xs = np.nonzero(active)
    roi_xywh = _pad_crop_bounds(
        (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
        sensor_width=int(sensor_width),
        sensor_height=int(sensor_height),
        margin_px=float(crop_margin_px),
        min_side_px=float(crop_min_side_px),
    )
    return roi_xywh, {
        "adaptive_crop_applied": True,
        "adaptive_crop_reason": "frame_guidance_bbox",
        "adaptive_crop_source": "frame_guidance",
        "adaptive_crop_support": int(np.count_nonzero(active)),
    }


def _resolve_crop_xywh(
    *,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    start_frame: Image.Image,
    end_frame: Image.Image,
    sensor_width: int,
    sensor_height: int,
    crop_mode: str,
    crop_min_events: int,
    crop_quantile_low: float,
    crop_quantile_high: float,
    crop_margin_px: float,
    crop_min_side_px: float,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    full_sensor = (0.0, 0.0, float(sensor_width), float(sensor_height))
    if str(crop_mode) != "density_adaptive":
        return full_sensor, {
            "adaptive_crop_applied": False,
            "adaptive_crop_reason": "crop_mode_disabled",
            "adaptive_crop_source": "full_sensor",
            "adaptive_crop_support": int(len(raw_x)),
        }

    if len(raw_x) >= int(crop_min_events):
        lo = float(min(max(float(crop_quantile_low), 0.0), 1.0))
        hi = float(min(max(float(crop_quantile_high), 0.0), 1.0))
        if hi <= lo:
            hi = min(1.0, lo + 0.9)
        qx0, qx1 = np.quantile(np.asarray(raw_x, dtype=np.float32), [lo, hi])
        qy0, qy1 = np.quantile(np.asarray(raw_y, dtype=np.float32), [lo, hi])
        roi_xywh = _pad_crop_bounds(
            (float(qx0), float(qy0), float(qx1), float(qy1)),
            sensor_width=int(sensor_width),
            sensor_height=int(sensor_height),
            margin_px=float(crop_margin_px),
            min_side_px=float(crop_min_side_px),
        )
        return roi_xywh, {
            "adaptive_crop_applied": True,
            "adaptive_crop_reason": "raw_density_quantile_box",
            "adaptive_crop_source": "raw_density",
            "adaptive_crop_support": int(len(raw_x)),
        }

    guidance_bbox = _frame_guidance_bbox(
        start_frame,
        end_frame,
        sensor_width=int(sensor_width),
        sensor_height=int(sensor_height),
        crop_margin_px=float(crop_margin_px),
        crop_min_side_px=float(crop_min_side_px),
    )
    if guidance_bbox is not None:
        return guidance_bbox

    return full_sensor, {
        "adaptive_crop_applied": False,
        "adaptive_crop_reason": "insufficient_support",
        "adaptive_crop_source": "full_sensor",
        "adaptive_crop_support": int(len(raw_x)),
    }


def _build_polarity_feature(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    sensor_width: int,
    sensor_height: int,
    feature_width: int,
    feature_height: int,
) -> np.ndarray:
    neg = np.zeros((int(feature_height), int(feature_width)), dtype=np.float32)
    pos = np.zeros((int(feature_height), int(feature_width)), dtype=np.float32)
    if len(xs) <= 0:
        return np.stack([neg, pos], axis=0)

    grid_x = np.clip(
        np.floor(np.asarray(xs, dtype=np.float32) * float(feature_width) / float(sensor_width)).astype(np.int64),
        0,
        max(0, int(feature_width) - 1),
    )
    grid_y = np.clip(
        np.floor(np.asarray(ys, dtype=np.float32) * float(feature_height) / float(sensor_height)).astype(np.int64),
        0,
        max(0, int(feature_height) - 1),
    )
    pol = np.asarray(ps, dtype=np.int8)
    neg_mask = pol <= 0
    pos_mask = ~neg_mask
    if np.any(neg_mask):
        np.add.at(neg, (grid_y[neg_mask], grid_x[neg_mask]), 1.0)
    if np.any(pos_mask):
        np.add.at(pos, (grid_y[pos_mask], grid_x[pos_mask]), 1.0)
    return np.stack([np.log1p(neg), np.log1p(pos)], axis=0)


def _feature_to_rgb(feature: np.ndarray) -> np.ndarray:
    neg = np.asarray(feature[0], dtype=np.float32)
    pos = np.asarray(feature[1], dtype=np.float32)
    denom = float(max(float(neg.max()), float(pos.max()), 1.0))
    neg = np.clip(neg / denom, 0.0, 1.0)
    pos = np.clip(pos / denom, 0.0, 1.0)
    rgb = np.zeros((feature.shape[1], feature.shape[2], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(pos * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb[..., 1] = np.clip(np.minimum(pos, neg) * 84.0, 0.0, 255.0).astype(np.uint8)
    rgb[..., 2] = np.clip(neg * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb


def _frame_guidance_map(
    start_frame: Image.Image,
    end_frame: Image.Image,
    *,
    feature_width: int,
    feature_height: int,
    blur_radius: float,
) -> np.ndarray:
    start_gray = start_frame.convert("L").resize((int(feature_width), int(feature_height)), resample=Image.Resampling.BILINEAR)
    end_gray = end_frame.convert("L").resize((int(feature_width), int(feature_height)), resample=Image.Resampling.BILINEAR)
    diff = np.abs(
        np.asarray(end_gray, dtype=np.float32) - np.asarray(start_gray, dtype=np.float32)
    )
    diff_img = Image.fromarray(np.clip(diff, 0.0, 255.0).astype(np.uint8), mode="L")
    if float(blur_radius) > 0.0:
        diff_img = diff_img.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    diff = np.asarray(diff_img, dtype=np.float32)
    denom = float(max(float(diff.max()), 1.0))
    return np.clip(diff / denom, 0.0, 1.0)


def _find_prev_anchor(intervals: list[dict[str, Any]], current_index: int, threshold: int) -> int | None:
    for idx in range(current_index - 1, -1, -1):
        if int(intervals[idx]["feature_event_count"]) >= int(threshold):
            return idx
    return None


def _find_next_anchor(intervals: list[dict[str, Any]], current_index: int, threshold: int) -> int | None:
    for idx in range(current_index + 1, len(intervals)):
        if int(intervals[idx]["feature_event_count"]) >= int(threshold):
            return idx
    return None


def _interval_midpoint_us(interval: dict[str, Any]) -> float:
    return 0.5 * (float(interval["start_timestamp_us"]) + float(interval["end_timestamp_us"]))


def _guided_feature_for_interval(
    *,
    intervals: list[dict[str, Any]],
    current_index: int,
    anchor_event_threshold: int,
    guidance_strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = intervals[current_index]
    observed = np.asarray(current["observed_feature"], dtype=np.float32)
    feature_event_count = int(current["feature_event_count"])

    if feature_event_count >= int(anchor_event_threshold):
        return observed, {
            "event_feature_source": "observed_event_feature",
            "event_feature_valid": bool(observed.max() > 0.0),
            "interp_gap_us": 0,
            "anchor_prev_interval_index": int(current_index),
            "anchor_next_interval_index": int(current_index),
            "guidance_mode": "observed_anchor",
            "sparsity_alpha": 0.0,
        }

    prev_anchor = _find_prev_anchor(intervals, current_index, int(anchor_event_threshold))
    next_anchor = _find_next_anchor(intervals, current_index, int(anchor_event_threshold))
    if prev_anchor is None and next_anchor is None:
        return observed, {
            "event_feature_source": "observed_event_feature",
            "event_feature_valid": bool(observed.max() > 0.0),
            "interp_gap_us": 0,
            "anchor_prev_interval_index": None,
            "anchor_next_interval_index": None,
            "guidance_mode": "no_anchor_fallback",
            "sparsity_alpha": 0.0,
        }

    if prev_anchor is not None and next_anchor is not None and prev_anchor != next_anchor:
        prev_feature = np.asarray(intervals[prev_anchor]["observed_feature"], dtype=np.float32)
        next_feature = np.asarray(intervals[next_anchor]["observed_feature"], dtype=np.float32)
        prev_mid = _interval_midpoint_us(intervals[prev_anchor])
        next_mid = _interval_midpoint_us(intervals[next_anchor])
        current_mid = _interval_midpoint_us(current)
        denom = max(next_mid - prev_mid, 1.0)
        alpha = float(np.clip((current_mid - prev_mid) / denom, 0.0, 1.0))
        interpolated = (1.0 - alpha) * prev_feature + alpha * next_feature
        interp_gap_us = int(round(next_mid - prev_mid))
        guidance_mode = "bilateral_linear"
    elif prev_anchor is not None:
        interpolated = np.asarray(intervals[prev_anchor]["observed_feature"], dtype=np.float32)
        interp_gap_us = int(round(abs(_interval_midpoint_us(current) - _interval_midpoint_us(intervals[prev_anchor]))))
        guidance_mode = "carry_prev"
    else:
        interpolated = np.asarray(intervals[next_anchor]["observed_feature"], dtype=np.float32)
        interp_gap_us = int(round(abs(_interval_midpoint_us(intervals[next_anchor]) - _interval_midpoint_us(current))))
        guidance_mode = "carry_next"

    pos_sum = float(observed[1].sum())
    neg_sum = float(observed[0].sum())
    if (pos_sum + neg_sum) <= 1e-6:
        pos_sum = float(interpolated[1].sum())
        neg_sum = float(interpolated[0].sum())
    if (pos_sum + neg_sum) <= 1e-6:
        pos_ratio = 0.5
    else:
        pos_ratio = pos_sum / (pos_sum + neg_sum)
    neg_ratio = 1.0 - pos_ratio

    guidance_map = np.asarray(current["guidance_map"], dtype=np.float32)
    guidance_scale = float(max(float(interpolated.max()), float(observed.max()), 1.0))
    guidance_prior = np.stack(
        [
            guidance_map * neg_ratio * guidance_scale,
            guidance_map * pos_ratio * guidance_scale,
        ],
        axis=0,
    )
    sparse_alpha = float(np.clip(1.0 - (float(feature_event_count) / float(max(anchor_event_threshold, 1))), 0.0, 1.0))
    guided_feature = (1.0 - sparse_alpha) * observed + sparse_alpha * (
        (1.0 - float(guidance_strength)) * interpolated + float(guidance_strength) * guidance_prior
    )
    return np.asarray(guided_feature, dtype=np.float32), {
        "event_feature_source": "interpolated_event_feature",
        "event_feature_valid": bool(guided_feature.max() > 0.0),
        "interp_gap_us": int(interp_gap_us),
        "anchor_prev_interval_index": None if prev_anchor is None else int(prev_anchor),
        "anchor_next_interval_index": None if next_anchor is None else int(next_anchor),
        "guidance_mode": str(guidance_mode),
        "sparsity_alpha": float(round(sparse_alpha, 6)),
    }


def _render_feature_tile(feature: np.ndarray, *, label: str, target_size: tuple[int, int] = (220, 156)) -> Image.Image:
    tile = Image.fromarray(_feature_to_rgb(feature), mode="RGB")
    tile = ImageOps.pad(tile, target_size, color=(14, 14, 16))
    tile = ImageOps.expand(tile, border=3, fill=(58, 68, 84))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, tile.width - 1, 18), fill=(18, 21, 28))
    draw.text((8, 3), str(label), fill=(236, 236, 236))
    return tile


def _compose_quad(
    *,
    row: dict[str, Any],
    start_frame: Image.Image,
    observed_tile: Image.Image,
    guided_tile: Image.Image,
    end_frame: Image.Image,
) -> Image.Image:
    frame_size = (200, 142)
    start_tile = ImageOps.pad(start_frame.convert("RGB"), frame_size, color=(234, 234, 234))
    end_tile = ImageOps.pad(end_frame.convert("RGB"), frame_size, color=(234, 234, 234))
    start_tile = ImageOps.expand(start_tile, border=3, fill=(66, 128, 82))
    end_tile = ImageOps.expand(end_tile, border=3, fill=(132, 74, 40))

    gap = 14
    title_h = 52
    footer_h = 24
    width = start_tile.width + observed_tile.width + guided_tile.width + end_tile.width + gap * 5
    height = title_h + max(start_tile.height, observed_tile.height, guided_tile.height, end_tile.height) + footer_h
    canvas = Image.new("RGB", (width, height), color=(246, 243, 237))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 8),
        (
            f"Interval {int(row['interval_index']):02d} | {int(row['interval_us'])/1000.0:.1f} ms | "
            f"raw={int(row['raw_event_count'])} roi={int(row['feature_event_count'])} | {row['event_feature_source']}"
        ),
        fill=(31, 29, 27),
    )
    draw.text(
        (gap, 28),
        (
            f"{row['guidance_mode']} | crop={row['adaptive_crop_source']} | "
            f"sparsity_alpha={float(row['sparsity_alpha']):.2f}"
        ),
        fill=(48, 46, 44),
    )
    y = title_h
    x = gap
    for tile in [start_tile, observed_tile, guided_tile, end_tile]:
        canvas.paste(tile, (x, y))
        x += tile.width + gap
    draw.text((gap, height - 18), "Start Frame", fill=(48, 72, 56))
    draw.text((gap + start_tile.width + gap + 8, height - 18), "Observed Feature", fill=(56, 60, 88))
    draw.text((gap + start_tile.width + gap + observed_tile.width + gap + 8, height - 18), "Guided Feature", fill=(96, 58, 78))
    draw.text((width - end_tile.width - gap, height - 18), "End Frame", fill=(88, 56, 42))
    return canvas


def _build_contact_sheet(images: list[Image.Image], *, save_path: Path) -> None:
    if not images:
        return
    thumb_size = (380, 122)
    gap = 18
    cols = 2
    rows = int(math.ceil(len(images) / cols))
    canvas = Image.new(
        "RGB",
        (gap + cols * (thumb_size[0] + gap), gap + rows * (thumb_size[1] + gap)),
        color=(252, 248, 242),
    )
    for idx, image in enumerate(images):
        tile = ImageOps.pad(image, thumb_size, color=(236, 233, 227))
        x = gap + (idx % cols) * (thumb_size[0] + gap)
        y = gap + (idx // cols) * (thumb_size[1] + gap)
        canvas.paste(tile, (x, y))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def _write_readme(output_root: Path, quick_preview_root: Path, summary: dict[str, Any], mode_examples: dict[str, dict[str, str]]) -> None:
    lines = [
        "# TSGSS Guided Event Feature Interpolation",
        "",
        "This artifact stores derived event features only.",
        "No raw event tuples are synthesized.",
        "",
        "## Quick Preview",
        f"- summary: `{summary['n_sessions']} sessions`, `{summary['n_intervals']} intervals`",
        f"- quick preview root: `{quick_preview_root}`",
        f"- anchor threshold: `{summary['anchor_event_threshold']}`",
        f"- feature size: `{summary['feature_width']}x{summary['feature_height']}`",
        f"- crop mode: `{summary['crop_mode']}`",
    ]
    for key in ["observed_event_feature", "interpolated_event_feature"]:
        example = mode_examples.get(key)
        if example is not None:
            lines.append(f"- {key}: `{example['session_key']}`")
    lines.extend(
        [
            "",
            "## Representative PNGs",
            f"- `tsgss/{quick_preview_root.name}/01_interpolated_feature_triptych.png`",
            f"- `tsgss/{quick_preview_root.name}/02_observed_feature_triptych.png`",
            f"- `tsgss/{quick_preview_root.name}/03_first_session_contact_sheet.png`",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    batch_root = _resolve_path(args.batch_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    quick_preview_root = output_root.parent / f"{output_root.name}_quick_preview"
    quick_preview_root.mkdir(parents=True, exist_ok=True)

    session_summaries = _read_session_summaries(batch_root)
    if args.limit_sessions is not None:
        session_summaries = session_summaries[: max(0, int(args.limit_sessions))]

    n_intervals_total = 0
    source_counter: Counter[str] = Counter()
    guidance_mode_counter: Counter[str] = Counter()
    crop_source_counter: Counter[str] = Counter()
    mode_examples: dict[str, dict[str, str]] = {}
    first_contact_sheet: Path | None = None

    for session in session_summaries:
        session_key = str(session["session_key"])
        user_id = int(session["user_id"])
        eye = str(session["eye"])
        session_code = str(session["session_code"])
        if session.get("event_file") is None:
            print(f"[SKIP] {session_key}: missing event_file", flush=True)
            continue

        sequence_manifest_path = Path(str(session["sequence_manifest_path"])).resolve()
        sequence_rows = sorted(_read_jsonl_rows(sequence_manifest_path), key=_sequence_sort_key)
        raw_events = load_events_from_txt(Path(str(session["event_file"])).resolve(), eye=eye)
        timestamps = np.asarray(raw_events["t"], dtype=np.int64)
        dirs = _session_output_dirs(output_root, user_id=user_id, eye=eye, session_code=session_code)
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        frame_cache: dict[str, Image.Image] = {}
        interval_records: list[dict[str, Any]] = []

        for interval_index, (row_start, row_end) in enumerate(zip(sequence_rows[:-1], sequence_rows[1:], strict=False)):
            start_timestamp_us = int(row_start["timestamp_us"])
            end_timestamp_us = int(row_end["timestamp_us"])
            start_idx, end_idx = _interval_indices(
                timestamps,
                start_timestamp_us=start_timestamp_us,
                end_timestamp_us=end_timestamp_us,
            )
            raw_x = np.asarray(raw_events["x"][start_idx:end_idx], dtype=np.int16)
            raw_y = np.asarray(raw_events["y"][start_idx:end_idx], dtype=np.int16)
            raw_p = np.asarray(raw_events["p"][start_idx:end_idx], dtype=np.int8)
            start_frame_path = _resolve_sequence_frame_path(batch_root, row_start)
            end_frame_path = _resolve_sequence_frame_path(batch_root, row_end)
            if str(start_frame_path) not in frame_cache:
                frame_cache[str(start_frame_path)] = _load_frame_rgb(start_frame_path)
            if str(end_frame_path) not in frame_cache:
                frame_cache[str(end_frame_path)] = _load_frame_rgb(end_frame_path)
            start_frame_full = frame_cache[str(start_frame_path)]
            end_frame_full = frame_cache[str(end_frame_path)]
            crop_roi_xywh, crop_meta = _resolve_crop_xywh(
                raw_x=raw_x,
                raw_y=raw_y,
                start_frame=start_frame_full,
                end_frame=end_frame_full,
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
                crop_mode=str(args.crop_mode),
                crop_min_events=int(args.crop_min_events),
                crop_quantile_low=float(args.crop_quantile_low),
                crop_quantile_high=float(args.crop_quantile_high),
                crop_margin_px=float(args.crop_margin_px),
                crop_min_side_px=float(args.crop_min_side_px),
            )
            crop_width = int(max(1, math.ceil(float(crop_roi_xywh[2]))))
            crop_height = int(max(1, math.ceil(float(crop_roi_xywh[3]))))
            start_frame = _crop_image_to_roi(start_frame_full, crop_roi_xywh)
            end_frame = _crop_image_to_roi(end_frame_full, crop_roi_xywh)
            raw_x_local, raw_y_local, inside_mask = _crop_points_to_roi(raw_x, raw_y, crop_roi_xywh)
            raw_p_local = np.asarray(raw_p[inside_mask], dtype=np.int8) if len(raw_p) > 0 else np.zeros((0,), dtype=np.int8)

            observed_feature = _build_polarity_feature(
                xs=raw_x_local,
                ys=raw_y_local,
                ps=raw_p_local,
                sensor_width=int(crop_width),
                sensor_height=int(crop_height),
                feature_width=int(args.feature_width),
                feature_height=int(args.feature_height),
            )
            guidance_map = _frame_guidance_map(
                start_frame,
                end_frame,
                feature_width=int(args.feature_width),
                feature_height=int(args.feature_height),
                blur_radius=float(args.guidance_blur_radius),
            )
            interval_records.append(
                {
                    "interval_index": int(interval_index),
                    "start_timestamp_us": int(start_timestamp_us),
                    "end_timestamp_us": int(end_timestamp_us),
                    "interval_us": int(end_timestamp_us - start_timestamp_us),
                    "raw_event_index_range": [int(start_idx), int(end_idx)],
                    "raw_event_count": int(len(raw_x)),
                    "feature_event_count": int(len(raw_x_local)),
                    "start_frame_path": str(start_frame_path),
                    "end_frame_path": str(end_frame_path),
                    "start_frame_kind": str(row_start["frame_kind"]),
                    "end_frame_kind": str(row_end["frame_kind"]),
                    "crop_roi_xywh": [float(v) for v in crop_roi_xywh],
                    "crop_width_height": [int(crop_width), int(crop_height)],
                    "start_frame_preview": start_frame,
                    "end_frame_preview": end_frame,
                    "observed_feature": observed_feature,
                    "guidance_map": guidance_map,
                    **crop_meta,
                },
            )

        observed_stack = np.zeros(
            (len(interval_records), 2, int(args.feature_height), int(args.feature_width)),
            dtype=np.float16,
        )
        guided_stack = np.zeros_like(observed_stack)
        guidance_stack = np.zeros(
            (len(interval_records), int(args.feature_height), int(args.feature_width)),
            dtype=np.float16,
        )
        interval_rows: list[dict[str, Any]] = []
        previews: list[Image.Image] = []

        for interval in interval_records:
            guided_feature, meta = _guided_feature_for_interval(
                intervals=interval_records,
                current_index=int(interval["interval_index"]),
                anchor_event_threshold=int(args.anchor_event_threshold),
                guidance_strength=float(args.guidance_strength),
            )
            interval_index = int(interval["interval_index"])
            observed_feature = np.asarray(interval["observed_feature"], dtype=np.float32)
            guidance_map = np.asarray(interval["guidance_map"], dtype=np.float32)
            observed_stack[interval_index] = observed_feature.astype(np.float16)
            guided_stack[interval_index] = guided_feature.astype(np.float16)
            guidance_stack[interval_index] = guidance_map.astype(np.float16)

            row = {
                "session_key": session_key,
                "user_id": int(user_id),
                "eye": str(eye),
                "session_code": str(session_code),
                "interval_index": int(interval_index),
                "start_timestamp_us": int(interval["start_timestamp_us"]),
                "end_timestamp_us": int(interval["end_timestamp_us"]),
                "interval_us": int(interval["interval_us"]),
                "raw_event_index_range": [int(v) for v in interval["raw_event_index_range"]],
                "raw_event_count": int(interval["raw_event_count"]),
                "feature_event_count": int(interval["feature_event_count"]),
                "start_frame_path": str(interval["start_frame_path"]),
                "end_frame_path": str(interval["end_frame_path"]),
                "start_frame_kind": str(interval["start_frame_kind"]),
                "end_frame_kind": str(interval["end_frame_kind"]),
                "crop_mode": str(args.crop_mode),
                "crop_roi_xywh": [float(v) for v in interval["crop_roi_xywh"]],
                "crop_width_height": [int(v) for v in interval["crop_width_height"]],
                "adaptive_crop_applied": bool(interval["adaptive_crop_applied"]),
                "adaptive_crop_reason": str(interval["adaptive_crop_reason"]),
                "adaptive_crop_source": str(interval["adaptive_crop_source"]),
                "adaptive_crop_support": int(interval["adaptive_crop_support"]),
                **meta,
            }
            interval_rows.append(row)
            source_counter[str(row["event_feature_source"])] += 1
            guidance_mode_counter[str(row["guidance_mode"])] += 1
            crop_source_counter[str(row["adaptive_crop_source"])] += 1

            observed_tile = _render_feature_tile(observed_feature, label="Observed")
            guided_tile = _render_feature_tile(guided_feature, label="Guided")
            preview = _compose_quad(
                row=row,
                start_frame=interval["start_frame_preview"],
                observed_tile=observed_tile,
                guided_tile=guided_tile,
                end_frame=interval["end_frame_preview"],
            )
            previews.append(preview)
            preview_path = dirs["assets"] / f"interval_{interval_index:03d}_guided_feature_quad.png"
            preview.save(preview_path)

            source_name = str(row["event_feature_source"])
            if source_name not in mode_examples:
                mode_examples[source_name] = {
                    "session_key": session_key,
                    "preview_path": str(preview_path),
                }

        feature_npz_path = dirs["manifests"] / "guided_event_features.npz"
        np.savez_compressed(
            feature_npz_path,
            observed_feature=observed_stack,
            guided_feature=guided_stack,
            guidance_map=guidance_stack,
        )
        manifest_path = dirs["manifests"] / "guided_feature_manifest.jsonl"
        write_jsonl(interval_rows, manifest_path)
        contact_sheet_path = dirs["base"] / "guided_feature_contact_sheet.png"
        _build_contact_sheet(previews, save_path=contact_sheet_path)
        if first_contact_sheet is None:
            first_contact_sheet = contact_sheet_path

        session_summary = {
            "experiment": "tsgss_guided_event_feature_interpolation",
            "session_key": session_key,
            "user_id": int(user_id),
            "eye": str(eye),
            "session_code": str(session_code),
            "feature_width": int(args.feature_width),
            "feature_height": int(args.feature_height),
            "anchor_event_threshold": int(args.anchor_event_threshold),
            "guidance_strength": float(args.guidance_strength),
            "guidance_blur_radius": float(args.guidance_blur_radius),
            "crop_mode": str(args.crop_mode),
            "crop_min_events": int(args.crop_min_events),
            "crop_quantile_low": float(args.crop_quantile_low),
            "crop_quantile_high": float(args.crop_quantile_high),
            "crop_margin_px": float(args.crop_margin_px),
            "crop_min_side_px": float(args.crop_min_side_px),
            "n_intervals": len(interval_rows),
            "guided_feature_path": str(feature_npz_path),
            "guided_feature_manifest_path": str(manifest_path),
            "contact_sheet_path": str(contact_sheet_path),
            "event_feature_source_distribution": dict(Counter(str(row["event_feature_source"]) for row in interval_rows)),
            "guidance_mode_distribution": dict(Counter(str(row["guidance_mode"]) for row in interval_rows)),
            "adaptive_crop_source_distribution": dict(Counter(str(row["adaptive_crop_source"]) for row in interval_rows)),
        }
        write_json(session_summary, dirs["base"] / "summary.json")
        n_intervals_total += len(interval_rows)
        print(
            f"{session_key}: intervals={len(interval_rows)} sources={session_summary['event_feature_source_distribution']}",
            flush=True,
        )

    preview_specs = [
        ("interpolated_event_feature", "01_interpolated_feature_triptych.png"),
        ("observed_event_feature", "02_observed_feature_triptych.png"),
    ]
    for source_name, preview_name in preview_specs:
        example = mode_examples.get(source_name)
        if example is None:
            continue
        maybe_link_or_copy(
            Path(example["preview_path"]),
            quick_preview_root / preview_name,
            mode=str(args.link_mode),
            overwrite=True,
        )
    if first_contact_sheet is not None:
        maybe_link_or_copy(
            first_contact_sheet,
            quick_preview_root / "03_first_session_contact_sheet.png",
            mode=str(args.link_mode),
            overwrite=True,
        )

    summary = {
        "experiment": "tsgss_guided_event_feature_interpolation",
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "quick_preview_root": str(quick_preview_root),
        "n_sessions": len(session_summaries),
        "n_intervals": int(n_intervals_total),
        "feature_width": int(args.feature_width),
        "feature_height": int(args.feature_height),
        "anchor_event_threshold": int(args.anchor_event_threshold),
        "guidance_strength": float(args.guidance_strength),
        "guidance_blur_radius": float(args.guidance_blur_radius),
        "crop_mode": str(args.crop_mode),
        "crop_min_events": int(args.crop_min_events),
        "crop_quantile_low": float(args.crop_quantile_low),
        "crop_quantile_high": float(args.crop_quantile_high),
        "crop_margin_px": float(args.crop_margin_px),
        "crop_min_side_px": float(args.crop_min_side_px),
        "event_feature_source_distribution": {str(k): int(v) for k, v in sorted(source_counter.items())},
        "guidance_mode_distribution": {str(k): int(v) for k, v in sorted(guidance_mode_counter.items())},
        "adaptive_crop_source_distribution": {str(k): int(v) for k, v in sorted(crop_source_counter.items())},
    }
    write_json(summary, output_root / "experiment_summary.json")
    _write_readme(output_root, quick_preview_root, summary, mode_examples)
    print(
        f"[DONE] sessions={summary['n_sessions']} intervals={summary['n_intervals']} "
        f"output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
