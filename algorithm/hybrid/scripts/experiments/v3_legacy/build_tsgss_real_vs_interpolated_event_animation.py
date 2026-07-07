#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from build_tsgss_event_alignment import _read_jsonl_rows, _resolve_path
from build_tsgss_frame_event_visualization import _load_frame_rgb, _read_session_summaries
from hbtxr.preprocess.io_utils import SENSOR_HEIGHT, SENSOR_WIDTH, load_events_from_txt, maybe_link_or_copy
from hbtxr.utils.io import write_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frame-to-frame GIF where raw events and interpolated fill are "
            "rendered with different colors."
        ),
    )
    parser.add_argument("--alignment-root", type=str, default="tsgss/event_alignment")
    parser.add_argument(
        "--output-root",
        type=str,
        default="tsgss/frame_event_animation_real_vs_interpolated_example",
    )
    parser.add_argument("--session-key", type=str, default="user01/left/session_102")
    parser.add_argument("--sensor-width", type=int, default=SENSOR_WIDTH)
    parser.add_argument("--sensor-height", type=int, default=SENSOR_HEIGHT)
    parser.add_argument("--target-size", type=int, default=320)
    parser.add_argument("--duration-ms", type=int, default=220)
    parser.add_argument("--pause-ms", type=int, default=900)
    parser.add_argument("--blur-radius", type=float, default=0.9)
    parser.add_argument("--layout", type=str, default="side_by_side", choices=["overlay", "side_by_side"])
    parser.add_argument("--crop-mode", type=str, default="density_adaptive", choices=["none", "density_adaptive"])
    parser.add_argument("--crop-min-events", type=int, default=64)
    parser.add_argument("--crop-quantile-low", type=float, default=0.05)
    parser.add_argument("--crop-quantile-high", type=float, default=0.95)
    parser.add_argument("--crop-margin-px", type=float, default=12.0)
    parser.add_argument("--crop-min-side-px", type=float, default=96.0)
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _session_output_dirs(output_root: Path, *, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
    base = output_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}"
    return {
        "base": base,
        "frames": base / "frames",
    }


def _save_gif(frames: list[Image.Image], *, dst: Path, duration_ms: int, pause_ms: int) -> None:
    if not frames:
        raise ValueError("Cannot save GIF with zero frames")
    durations = [int(duration_ms)] * len(frames)
    durations[-1] = int(max(duration_ms, pause_ms))
    palette_frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    dst.parent.mkdir(parents=True, exist_ok=True)
    palette_frames[0].save(
        dst,
        save_all=True,
        append_images=palette_frames[1:],
        loop=0,
        duration=durations,
        optimize=False,
        disposal=2,
    )


def _density_map(xs: np.ndarray, ys: np.ndarray, *, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((int(height), int(width)), dtype=np.float32)
    if len(xs) > 0:
        xs = np.clip(np.asarray(xs, dtype=np.int64), 0, max(0, int(width) - 1))
        ys = np.clip(np.asarray(ys, dtype=np.int64), 0, max(0, int(height) - 1))
        np.add.at(canvas, (ys, xs), 1.0)
    return np.log1p(canvas)


def _apply_blur(channel: np.ndarray, radius: float) -> np.ndarray:
    if float(radius) <= 0.0:
        return np.asarray(channel, dtype=np.float32)
    if float(channel.max()) <= 0.0:
        return np.asarray(channel, dtype=np.float32)
    image = Image.fromarray(np.clip(channel / float(channel.max()) * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    image = image.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(image, dtype=np.float32) / 255.0 * float(channel.max())


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
    xywh: tuple[float, float, float, float],
    *,
    sensor_width: int,
    sensor_height: int,
    margin_px: float,
    min_side_px: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in xywh]
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


def _resolve_density_crop_xywh(
    *,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    interp_x: np.ndarray,
    interp_y: np.ndarray,
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
            "adaptive_roi_applied": False,
            "adaptive_roi_reason": "crop_mode_disabled",
            "adaptive_roi_source": "full_sensor",
            "adaptive_roi_event_count": int(len(raw_x) + len(interp_x)),
        }

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("raw", np.asarray(raw_x, dtype=np.float32), np.asarray(raw_y, dtype=np.float32)),
        (
            "combined",
            np.concatenate([np.asarray(raw_x, dtype=np.float32), np.asarray(interp_x, dtype=np.float32)]),
            np.concatenate([np.asarray(raw_y, dtype=np.float32), np.asarray(interp_y, dtype=np.float32)]),
        ),
    ]

    for source_name, xs, ys in candidates:
        if xs.size < int(crop_min_events) or ys.size < int(crop_min_events):
            continue
        lo = float(min(max(float(crop_quantile_low), 0.0), 1.0))
        hi = float(min(max(float(crop_quantile_high), 0.0), 1.0))
        if hi <= lo:
            hi = min(1.0, lo + 0.9)
        qx0, qx1 = np.quantile(xs, [lo, hi])
        qy0, qy1 = np.quantile(ys, [lo, hi])
        roi_xywh = _pad_crop_bounds(
            (float(qx0), float(qy0), float(qx1), float(qy1)),
            sensor_width=int(sensor_width),
            sensor_height=int(sensor_height),
            margin_px=float(crop_margin_px),
            min_side_px=float(crop_min_side_px),
        )
        return roi_xywh, {
            "adaptive_roi_applied": True,
            "adaptive_roi_reason": "density_quantile_box",
            "adaptive_roi_source": source_name,
            "adaptive_roi_event_count": int(xs.size),
        }

    return full_sensor, {
        "adaptive_roi_applied": False,
        "adaptive_roi_reason": "insufficient_events",
        "adaptive_roi_source": "full_sensor",
        "adaptive_roi_event_count": int(len(raw_x) + len(interp_x)),
    }


def _blend_color(base: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return base * (1.0 - alpha[..., None]) + color_arr * alpha[..., None]


def _interpolated_only_mask(*, raw_count: int, target_count: int, alignment_mode: str) -> np.ndarray:
    if int(target_count) <= 0:
        return np.zeros((0,), dtype=bool)
    if alignment_mode in {"subsampled", "identity", "empty_pad"}:
        return np.zeros((int(target_count),), dtype=bool)
    if alignment_mode == "interp_single":
        mask = np.ones((int(target_count),), dtype=bool)
        if target_count > 0:
            mask[0] = False
        return mask
    if alignment_mode != "interpolated_dense" or raw_count <= 1:
        return np.zeros((int(target_count),), dtype=bool)
    positions = np.linspace(0.0, float(raw_count - 1), num=int(target_count), endpoint=True)
    lo = np.floor(positions)
    alpha = positions - lo
    return np.abs(alpha) > 1e-9


def _render_comparison_plane(
    *,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    interp_x: np.ndarray,
    interp_y: np.ndarray,
    width: int,
    height: int,
    blur_radius: float,
    target_size: int,
) -> Image.Image:
    raw_map = _apply_blur(_density_map(raw_x, raw_y, width=width, height=height), radius=float(blur_radius))
    interp_map = _apply_blur(_density_map(interp_x, interp_y, width=width, height=height), radius=float(blur_radius))
    denom = float(max(float(raw_map.max()), float(interp_map.max()), 1.0))
    raw_alpha = np.clip(raw_map / denom, 0.0, 1.0) * 0.96
    interp_alpha = np.clip(interp_map / denom, 0.0, 1.0) * 0.92

    base = np.full((int(height), int(width), 3), 255.0, dtype=np.float32)
    composed = _blend_color(base, raw_alpha, (52, 96, 255))
    composed = _blend_color(composed, interp_alpha, (236, 72, 56))
    image = Image.fromarray(np.clip(composed, 0.0, 255.0).astype(np.uint8), mode="RGB")
    return ImageOps.pad(image, (int(target_size), int(target_size)), color=(255, 255, 255))


def _crop_image_to_roi(image: Image.Image, roi_xywh: tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = [float(v) for v in roi_xywh]
    left = int(max(0, math.floor(x)))
    top = int(max(0, math.floor(y)))
    right = int(max(left + 1, math.ceil(x + w)))
    bottom = int(max(top + 1, math.ceil(y + h)))
    return image.crop((left, top, right, bottom))


def _render_single_source_plane(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    blur_radius: float,
    target_size: int,
    color: tuple[int, int, int],
) -> Image.Image:
    density = _apply_blur(_density_map(xs, ys, width=width, height=height), radius=float(blur_radius))
    denom = float(max(float(density.max()), 1.0))
    alpha = np.clip(density / denom, 0.0, 1.0) * 0.96
    base = np.full((int(height), int(width), 3), 255.0, dtype=np.float32)
    composed = _blend_color(base, alpha, color)
    image = Image.fromarray(np.clip(composed, 0.0, 255.0).astype(np.uint8), mode="RGB")
    return ImageOps.pad(image, (int(target_size), int(target_size)), color=(255, 255, 255))


def _crop_points_to_roi(
    xs: np.ndarray,
    ys: np.ndarray,
    roi_xywh: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(xs) == 0 or len(ys) == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
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
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return xs[inside] - x, ys[inside] - y


def _compose_triptych(
    *,
    row: dict[str, Any],
    start_frame: Image.Image,
    compare_plane: Image.Image,
    end_frame: Image.Image,
) -> Image.Image:
    frame_size = (220, 156)
    plane_size = (320, 320)
    start_tile = ImageOps.pad(start_frame.convert("RGB"), frame_size, color=(236, 236, 236))
    end_tile = ImageOps.pad(end_frame.convert("RGB"), frame_size, color=(236, 236, 236))
    start_tile = ImageOps.expand(start_tile, border=3, fill=(74, 126, 83))
    end_tile = ImageOps.expand(end_tile, border=3, fill=(128, 79, 54))
    plane_tile = ImageOps.pad(compare_plane.convert("RGB"), plane_size, color=(255, 255, 255))
    plane_tile = ImageOps.expand(plane_tile, border=3, fill=(74, 86, 112))

    gap = 18
    title_h = 56
    footer_h = 26
    width = start_tile.width + plane_tile.width + end_tile.width + gap * 4
    height = title_h + max(start_tile.height, plane_tile.height, end_tile.height) + footer_h
    canvas = Image.new("RGB", (width, height), color=(245, 241, 235))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 8),
        (
            f"Interval {int(row['interval_index']):02d} | {int(row['interval_us']) / 1000.0:.1f} ms | "
            f"{row['alignment_mode']}"
        ),
        fill=(34, 31, 29),
    )
    draw.text(
        (gap, 28),
        (
            f"Raw events: {int(row['raw_event_count'])} | "
            f"Interpolated fill: {int(row['interpolated_event_count'])} | "
            f"Target: {int(row['target_event_count'])}"
        ),
        fill=(48, 46, 43),
    )
    y = title_h
    x = gap
    canvas.paste(start_tile, (x, y))
    x += start_tile.width + gap
    canvas.paste(plane_tile, (x, y))
    x += plane_tile.width + gap
    canvas.paste(end_tile, (x, y))
    draw.text((gap, height - 18), "Start Frame", fill=(48, 72, 56))
    draw.text((gap + start_tile.width + gap + 8, height - 18), "Blue=Raw  Red=Interpolated", fill=(56, 64, 94))
    draw.text((width - end_tile.width - gap, height - 18), "End Frame", fill=(92, 58, 42))
    return canvas


def _compose_side_by_side(
    *,
    row: dict[str, Any],
    start_frame: Image.Image,
    raw_plane: Image.Image,
    interp_plane: Image.Image,
    end_frame: Image.Image,
) -> Image.Image:
    frame_size = (220, 156)
    plane_size = (250, 250)
    start_tile = ImageOps.pad(start_frame.convert("RGB"), frame_size, color=(236, 236, 236))
    end_tile = ImageOps.pad(end_frame.convert("RGB"), frame_size, color=(236, 236, 236))
    start_tile = ImageOps.expand(start_tile, border=3, fill=(74, 126, 83))
    end_tile = ImageOps.expand(end_tile, border=3, fill=(128, 79, 54))
    raw_tile = ImageOps.pad(raw_plane.convert("RGB"), plane_size, color=(255, 255, 255))
    interp_tile = ImageOps.pad(interp_plane.convert("RGB"), plane_size, color=(255, 255, 255))
    raw_tile = ImageOps.expand(raw_tile, border=3, fill=(66, 98, 196))
    interp_tile = ImageOps.expand(interp_tile, border=3, fill=(184, 78, 58))

    gap = 16
    title_h = 56
    footer_h = 26
    width = start_tile.width + raw_tile.width + interp_tile.width + end_tile.width + gap * 5
    height = title_h + max(start_tile.height, raw_tile.height, interp_tile.height, end_tile.height) + footer_h
    canvas = Image.new("RGB", (width, height), color=(245, 241, 235))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 8),
        (
            f"Interval {int(row['interval_index']):02d} | {int(row['interval_us']) / 1000.0:.1f} ms | "
            f"{row['alignment_mode']}"
        ),
        fill=(34, 31, 29),
    )
    draw.text(
        (gap, 28),
        (
            f"Raw events: {int(row['raw_event_count'])} | "
            f"Interpolated fill: {int(row['interpolated_event_count'])} | "
            f"Target: {int(row['target_event_count'])}"
        ),
        fill=(48, 46, 43),
    )

    x = gap
    y = title_h
    canvas.paste(start_tile, (x, y))
    x += start_tile.width + gap
    canvas.paste(raw_tile, (x, y))
    x += raw_tile.width + gap
    canvas.paste(interp_tile, (x, y))
    x += interp_tile.width + gap
    canvas.paste(end_tile, (x, y))

    draw.text((gap, height - 18), "Start Frame", fill=(48, 72, 56))
    draw.text((gap + start_tile.width + gap + 12, height - 18), "Raw Event", fill=(56, 78, 154))
    draw.text((gap + start_tile.width + gap + raw_tile.width + gap + 12, height - 18), "Interpolated Fill", fill=(154, 72, 58))
    draw.text((width - end_tile.width - gap, height - 18), "End Frame", fill=(92, 58, 42))
    return canvas


def _build_contact_sheet(images: list[Image.Image], *, save_path: Path) -> None:
    if not images:
        return
    thumb_size = (360, 220)
    gap = 18
    cols = 2
    rows = int(math.ceil(len(images) / cols))
    canvas = Image.new("RGB", (gap + cols * (thumb_size[0] + gap), gap + rows * (thumb_size[1] + gap)), color=(251, 247, 241))
    for idx, image in enumerate(images):
        tile = ImageOps.pad(image, thumb_size, color=(235, 231, 224))
        x = gap + (idx % cols) * (thumb_size[0] + gap)
        y = gap + (idx // cols) * (thumb_size[1] + gap)
        canvas.paste(tile, (x, y))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def _find_session_summary(alignment_root: Path, session_key: str) -> dict[str, Any]:
    for session in _read_session_summaries(alignment_root):
        if str(session.get("session_key")) == str(session_key):
            return session
    raise FileNotFoundError(f"Session not found under {alignment_root}: {session_key}")


def main() -> None:
    args = build_argparser().parse_args()
    alignment_root = _resolve_path(args.alignment_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    session = _find_session_summary(alignment_root, str(args.session_key))
    session_key = str(session["session_key"])
    user_id = int(session["user_id"])
    eye = str(session["eye"])
    session_code = str(session["session_code"])

    dirs = _session_output_dirs(output_root, user_id=user_id, eye=eye, session_code=session_code)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_jsonl_rows(Path(str(session["frame_event_manifest_path"])).resolve())
    aligned = np.load(Path(str(session["aligned_events_path"])).resolve())
    raw_events = load_events_from_txt(Path(str(session["event_file"])).resolve(), eye=eye)

    frame_cache: dict[str, Image.Image] = {}
    animation_frames: list[Image.Image] = []
    interval_stats: list[dict[str, Any]] = []

    for row in manifest_rows:
        interval_index = int(row["aligned_interval_index"])
        start_frame_path = Path(str(row["start_frame_path"])).resolve()
        end_frame_path = Path(str(row["end_frame_path"])).resolve()
        if str(start_frame_path) not in frame_cache:
            frame_cache[str(start_frame_path)] = _load_frame_rgb(start_frame_path)
        if str(end_frame_path) not in frame_cache:
            frame_cache[str(end_frame_path)] = _load_frame_rgb(end_frame_path)

        raw_start, raw_end = [int(v) for v in row["raw_event_index_range"]]
        raw_x = np.asarray(raw_events["x"][raw_start:raw_end], dtype=np.int16)
        raw_y = np.asarray(raw_events["y"][raw_start:raw_end], dtype=np.int16)

        valid_mask = np.asarray(aligned["valid_mask"][interval_index], dtype=bool)
        interp_mask = _interpolated_only_mask(
            raw_count=int(row["raw_event_count"]),
            target_count=int(row["target_event_count"]),
            alignment_mode=str(row["alignment_mode"]),
        ) & valid_mask
        interp_x = np.asarray(aligned["x"][interval_index][interp_mask], dtype=np.int16)
        interp_y = np.asarray(aligned["y"][interval_index][interp_mask], dtype=np.int16)
        crop_roi_xywh, crop_meta = _resolve_density_crop_xywh(
            raw_x=raw_x,
            raw_y=raw_y,
            interp_x=interp_x,
            interp_y=interp_y,
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
        start_frame = _crop_image_to_roi(frame_cache[str(start_frame_path)], crop_roi_xywh)
        end_frame = _crop_image_to_roi(frame_cache[str(end_frame_path)], crop_roi_xywh)
        raw_x_local, raw_y_local = _crop_points_to_roi(raw_x, raw_y, crop_roi_xywh)
        interp_x_local, interp_y_local = _crop_points_to_roi(interp_x, interp_y, crop_roi_xywh)

        if str(args.layout) == "overlay":
            compare_plane = _render_comparison_plane(
                raw_x=raw_x_local,
                raw_y=raw_y_local,
                interp_x=interp_x_local,
                interp_y=interp_y_local,
                width=crop_width,
                height=crop_height,
                blur_radius=float(args.blur_radius),
                target_size=int(args.target_size),
            )
            rendered = _compose_triptych(
                row=row,
                start_frame=start_frame,
                compare_plane=compare_plane,
                end_frame=end_frame,
            )
        else:
            raw_plane = _render_single_source_plane(
                xs=raw_x_local,
                ys=raw_y_local,
                width=crop_width,
                height=crop_height,
                blur_radius=float(args.blur_radius),
                target_size=int(args.target_size),
                color=(52, 96, 255),
            )
            interp_plane = _render_single_source_plane(
                xs=interp_x_local,
                ys=interp_y_local,
                width=crop_width,
                height=crop_height,
                blur_radius=float(args.blur_radius),
                target_size=int(args.target_size),
                color=(236, 72, 56),
            )
            rendered = _compose_side_by_side(
                row=row,
                start_frame=start_frame,
                raw_plane=raw_plane,
                interp_plane=interp_plane,
                end_frame=end_frame,
            )
        animation_frames.append(rendered)
        rendered.save(dirs["frames"] / f"frame_{interval_index:03d}.png")
        interval_stats.append(
            {
                "interval_index": int(row["interval_index"]),
                "alignment_mode": str(row["alignment_mode"]),
                "raw_event_count": int(row["raw_event_count"]),
                "interpolated_fill_count_rendered": int(len(interp_x)),
                "raw_event_count_in_crop": int(len(raw_x_local)),
                "interpolated_fill_count_in_crop": int(len(interp_x_local)),
                "frame_kinds": [str(row["start_frame_kind"]), str(row["end_frame_kind"])],
                "crop_mode": str(args.crop_mode),
                "crop_roi_xywh": [float(v) for v in crop_roi_xywh],
                "crop_width_height": [int(crop_width), int(crop_height)],
                **crop_meta,
            },
        )

    gif_path = dirs["base"] / "session_animation.gif"
    poster_path = dirs["base"] / "animation_poster.png"
    contact_sheet_path = dirs["base"] / "comparison_contact_sheet.png"
    _save_gif(animation_frames, dst=gif_path, duration_ms=int(args.duration_ms), pause_ms=int(args.pause_ms))
    animation_frames[0].save(poster_path)
    _build_contact_sheet(animation_frames, save_path=contact_sheet_path)

    quick_preview_root = output_root.parent / f"{output_root.name}_quick_preview"
    quick_preview_root.mkdir(parents=True, exist_ok=True)
    maybe_link_or_copy(gif_path, quick_preview_root / "01_real_vs_interpolated_animation.gif", mode=str(args.link_mode), overwrite=True)
    maybe_link_or_copy(poster_path, quick_preview_root / "02_real_vs_interpolated_poster.png", mode=str(args.link_mode), overwrite=True)
    maybe_link_or_copy(contact_sheet_path, quick_preview_root / "03_real_vs_interpolated_contact_sheet.png", mode=str(args.link_mode), overwrite=True)

    summary = {
        "experiment": "tsgss_real_vs_interpolated_event_animation",
        "alignment_root": str(alignment_root),
        "output_root": str(output_root),
        "session_key": session_key,
        "layout": str(args.layout),
        "crop_mode": str(args.crop_mode),
        "crop_min_events": int(args.crop_min_events),
        "crop_quantile_low": float(args.crop_quantile_low),
        "crop_quantile_high": float(args.crop_quantile_high),
        "crop_margin_px": float(args.crop_margin_px),
        "crop_min_side_px": float(args.crop_min_side_px),
        "gif_path": str(gif_path),
        "poster_path": str(poster_path),
        "contact_sheet_path": str(contact_sheet_path),
        "duration_ms": int(args.duration_ms),
        "pause_ms": int(args.pause_ms),
        "quick_preview_root": str(quick_preview_root),
        "interval_stats": interval_stats,
        "legend": {
            "raw_event": "blue",
            "interpolated_fill": "red",
        },
    }
    write_json(summary, dirs["base"] / "summary.json")
    write_json(summary, output_root / "experiment_summary.json")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# TSGSS Real-vs-Interpolated Event Animation",
                "",
                f"Layout: {str(args.layout)}",
                f"Crop mode: {str(args.crop_mode)}",
                "Blue shows raw events from the original interval.",
                "Red shows interpolated fill introduced to reach the fixed event count.",
                "",
                "## Quick Preview",
                f"- `{quick_preview_root}/01_real_vs_interpolated_animation.gif`",
                f"- `{quick_preview_root}/02_real_vs_interpolated_poster.png`",
                f"- `{quick_preview_root}/03_real_vs_interpolated_contact_sheet.png`",
            ]
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[DONE] session={session_key} intervals={len(animation_frames)} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
