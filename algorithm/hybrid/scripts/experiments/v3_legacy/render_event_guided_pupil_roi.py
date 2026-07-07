from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hbtxr.preprocess.io_utils import collect_frame_records  # noqa: E402
from hbtxr.preprocess.target_fps_build import _load_event_arrays  # noqa: E402


def _session_dir(raw_root: Path, *, user_id: int, eye: str, session_code: str) -> Path:
    return raw_root / f"user{int(user_id)}" / str(eye) / f"session_{session_code[0]}_{session_code[1]}_{session_code[2]}"


def _v10_summary_path(anchor_root: Path, *, user_id: int, eye: str, session_code: str) -> Path:
    return anchor_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}" / "summary.json"


def _clip_xywh_to_bounds(xywh: list[float], *, width: int, height: int) -> list[float]:
    x, y, w, h = [float(v) for v in xywh]
    x = min(max(0.0, x), float(width - 1))
    y = min(max(0.0, y), float(height - 1))
    w = min(max(1.0, w), float(width) - x)
    h = min(max(1.0, h), float(height) - y)
    return [x, y, w, h]


def _bbox_center_xy(xywh: list[float]) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in xywh]
    return x + 0.5 * w, y + 0.5 * h


def _load_anchor_prior(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    eye_boxes = [frame["eye_region_bbox_xywh_sensor"] for frame in payload["frames"] if frame.get("eye_region_bbox_xywh_sensor")]
    pupil_boxes = [frame["sensor_pupil_bbox_xywh"] for frame in payload["frames"] if frame.get("sensor_pupil_bbox_xywh")]
    if not eye_boxes or not pupil_boxes:
        raise ValueError(f"Missing eye/pupil boxes in anchor summary: {summary_path}")

    eye_x = statistics.median(float(row[0]) for row in eye_boxes)
    eye_y = statistics.median(float(row[1]) for row in eye_boxes)
    eye_w = statistics.median(float(row[2]) for row in eye_boxes)
    eye_h = statistics.median(float(row[3]) for row in eye_boxes)
    eye_bbox = [eye_x, eye_y, eye_w, eye_h]

    rel_centers_x: list[float] = []
    rel_centers_y: list[float] = []
    rel_widths: list[float] = []
    rel_heights: list[float] = []
    for eye_box, pupil_box in zip(eye_boxes, pupil_boxes, strict=False):
        ex, ey, ew, eh = [float(v) for v in eye_box]
        px, py, pw, ph = [float(v) for v in pupil_box]
        pcx = px + 0.5 * pw
        pcy = py + 0.5 * ph
        rel_centers_x.append((pcx - ex) / max(ew, 1.0))
        rel_centers_y.append((pcy - ey) / max(eh, 1.0))
        rel_widths.append(pw / max(ew, 1.0))
        rel_heights.append(ph / max(eh, 1.0))

    prior = {
        "eye_bbox_xywh_sensor": eye_bbox,
        "relative_center_xy": [statistics.median(rel_centers_x), statistics.median(rel_centers_y)],
        "relative_size_wh": [statistics.median(rel_widths), statistics.median(rel_heights)],
    }
    prior["pupil_bbox_xywh_sensor"] = _prior_pupil_bbox_from_eye_bbox(prior)
    return prior


def _prior_pupil_bbox_from_eye_bbox(prior: dict[str, Any]) -> list[float]:
    ex, ey, ew, eh = [float(v) for v in prior["eye_bbox_xywh_sensor"]]
    rel_cx, rel_cy = [float(v) for v in prior["relative_center_xy"]]
    rel_w, rel_h = [float(v) for v in prior["relative_size_wh"]]
    pw = max(6.0, ew * rel_w)
    ph = max(6.0, eh * rel_h)
    pcx = ex + ew * rel_cx
    pcy = ey + eh * rel_cy
    return [pcx - 0.5 * pw, pcy - 0.5 * ph, pw, ph]


def _accumulate_counts(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    sensor_width: int,
    sensor_height: int,
) -> np.ndarray:
    counts = np.zeros((sensor_height, sensor_width), dtype=np.uint16)
    if xs.size == 0 or ys.size == 0:
        return counts
    valid = (xs >= 0) & (xs < sensor_width) & (ys >= 0) & (ys < sensor_height)
    if not np.any(valid):
        return counts
    np.add.at(counts, (ys[valid], xs[valid]), 1)
    return counts


def _box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32, copy=False)
    k = 2 * radius + 1
    padded = np.pad(arr.astype(np.float32), radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    out = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return out / float(k * k)


def _normalize_clipped(arr: np.ndarray, *, percentile: float = 99.0) -> np.ndarray:
    positive = arr[arr > 0]
    if positive.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    cap = float(np.percentile(positive, percentile))
    cap = max(cap, 1.0)
    return np.clip(arr.astype(np.float32), 0.0, cap) / cap


def _gaussian_prior(
    width: int,
    height: int,
    *,
    center_xy: tuple[float, float],
    sigma_xy: tuple[float, float],
) -> np.ndarray:
    cx, cy = center_xy
    sx, sy = sigma_xy
    sx = max(float(sx), 1.0)
    sy = max(float(sy), 1.0)
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    gx = ((xs - float(cx)) ** 2) / (2.0 * sx * sx)
    gy = ((ys - float(cy)) ** 2) / (2.0 * sy * sy)
    return np.exp(-(gx + gy), dtype=np.float32)


def _roi_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    inter_w = max(0.0, min(ax1, bx1) - max(ax, bx))
    inter_h = max(0.0, min(ay1, by1) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _build_event_guided_roi(
    *,
    neg_counts: np.ndarray,
    pos_counts: np.ndarray,
    eye_bbox_xywh: list[float],
    prior_pupil_xywh: list[float],
    prev_roi_xywh: list[float] | None,
    interval_events_total: int,
    low_event_threshold: float,
    high_event_threshold: float,
    sensor_width: int,
    sensor_height: int,
) -> dict[str, Any]:
    ex, ey, ew, eh = [float(v) for v in eye_bbox_xywh]
    x0 = max(0, int(math.floor(ex)))
    y0 = max(0, int(math.floor(ey)))
    x1 = min(sensor_width, int(math.ceil(ex + ew)))
    y1 = min(sensor_height, int(math.ceil(ey + eh)))
    neg_crop = neg_counts[y0:y1, x0:x1]
    pos_crop = pos_counts[y0:y1, x0:x1]
    prior_roi = prev_roi_xywh or prior_pupil_xywh
    prior_cx, prior_cy = _bbox_center_xy(prior_roi)
    prior_local_cx = prior_cx - x0
    prior_local_cy = prior_cy - y0
    prior_w = float(prior_roi[2])
    prior_h = float(prior_roi[3])

    neg_blur = _box_blur(neg_crop, radius=2)
    pos_blur = _box_blur(pos_crop, radius=2)
    neg_norm = _normalize_clipped(neg_blur, percentile=99.0)
    pos_norm = _normalize_clipped(pos_blur, percentile=99.0)
    consensus = np.minimum(neg_norm, pos_norm)
    balance = 1.0 - (np.abs(pos_norm - neg_norm) / np.maximum(pos_norm + neg_norm, 1e-6))
    score = consensus * balance

    inner_margin = 4
    if score.shape[0] > 2 * inner_margin and score.shape[1] > 2 * inner_margin:
        score[:inner_margin, :] = 0.0
        score[-inner_margin:, :] = 0.0
        score[:, :inner_margin] = 0.0
        score[:, -inner_margin:] = 0.0

    high_burst = bool(interval_events_total > high_event_threshold)
    sparse = bool(interval_events_total < low_event_threshold or float(score.max(initial=0.0)) <= 1e-6)
    source = "event_guided"
    if sparse:
        source = "carry_prev_sparse" if prev_roi_xywh is not None else "fallback_prior_sparse"
        return {
            "roi_xywh_sensor": [float(v) for v in prior_roi],
            "source": source,
            "score_peak": 0.0,
            "score_mass": 0.0,
            "interval_events_total": int(interval_events_total),
            "sparse_interval": True,
            "high_burst_interval": high_burst,
            "roi_iou_prev": _roi_iou(prior_roi, prev_roi_xywh or prior_roi),
            "consensus_bbox_xywh_sensor": None,
        }

    sigma_x = max(prior_w * (1.15 if high_burst else 1.5), 8.0)
    sigma_y = max(prior_h * (1.15 if high_burst else 1.5), 8.0)
    prior_map = _gaussian_prior(score.shape[1], score.shape[0], center_xy=(prior_local_cx, prior_local_cy), sigma_xy=(sigma_x, sigma_y))
    score = score * prior_map
    active = score[score > 0]
    if active.size == 0:
        source = "carry_prev_zero_score" if prev_roi_xywh is not None else "fallback_prior_zero_score"
        return {
            "roi_xywh_sensor": [float(v) for v in prior_roi],
            "source": source,
            "score_peak": 0.0,
            "score_mass": 0.0,
            "interval_events_total": int(interval_events_total),
            "sparse_interval": False,
            "high_burst_interval": high_burst,
            "roi_iou_prev": _roi_iou(prior_roi, prev_roi_xywh or prior_roi),
            "consensus_bbox_xywh_sensor": None,
        }

    threshold = max(float(active.max()) * 0.35, float(np.percentile(active, 90)))
    focus = np.where(score >= threshold, score, 0.0)
    if float(focus.sum()) <= 0.0:
        focus = score

    ys, xs = np.nonzero(focus > 0)
    weights = focus[ys, xs].astype(np.float32)
    weight_sum = float(weights.sum())
    local_cx = float(np.sum(xs * weights) / max(weight_sum, 1e-6))
    local_cy = float(np.sum(ys * weights) / max(weight_sum, 1e-6))
    var_x = float(np.sum(weights * ((xs - local_cx) ** 2)) / max(weight_sum, 1e-6))
    var_y = float(np.sum(weights * ((ys - local_cy) ** 2)) / max(weight_sum, 1e-6))
    std_x = math.sqrt(max(var_x, 1.0))
    std_y = math.sqrt(max(var_y, 1.0))

    new_cx = x0 + local_cx
    new_cy = y0 + local_cy
    if prev_roi_xywh is not None:
        prev_cx, prev_cy = _bbox_center_xy(prev_roi_xywh)
        center_blend = 0.45 if high_burst else 0.25
        new_cx = (1.0 - center_blend) * new_cx + center_blend * prev_cx
        new_cy = (1.0 - center_blend) * new_cy + center_blend * prev_cy

    min_roi_w = max(ew * 0.12, prior_w * 0.85)
    min_roi_h = max(eh * 0.12, prior_h * 0.85)
    max_roi_w = min(ew * 0.45, prior_w * 2.2)
    max_roi_h = min(eh * 0.45, prior_h * 2.2)
    roi_w = float(np.clip(max(6.0 * std_x, min_roi_w), min_roi_w, max_roi_w))
    roi_h = float(np.clip(max(6.0 * std_y, min_roi_h), min_roi_h, max_roi_h))

    eye_left = ex
    eye_top = ey
    eye_right = ex + ew
    eye_bottom = ey + eh
    roi_x = float(np.clip(new_cx - 0.5 * roi_w, eye_left, eye_right - roi_w))
    roi_y = float(np.clip(new_cy - 0.5 * roi_h, eye_top, eye_bottom - roi_h))
    roi_xywh = _clip_xywh_to_bounds([roi_x, roi_y, roi_w, roi_h], width=sensor_width, height=sensor_height)

    xs_global = xs + x0
    ys_global = ys + y0
    consensus_bbox = [
        float(xs_global.min(initial=x0)),
        float(ys_global.min(initial=y0)),
        float(xs_global.max(initial=x0) - xs_global.min(initial=x0) + 1),
        float(ys_global.max(initial=y0) - ys_global.min(initial=y0) + 1),
    ]
    if high_burst:
        source = "event_guided_high_burst"

    return {
        "roi_xywh_sensor": roi_xywh,
        "source": source,
        "score_peak": float(active.max(initial=0.0)),
        "score_mass": float(score.sum()),
        "interval_events_total": int(interval_events_total),
        "sparse_interval": False,
        "high_burst_interval": high_burst,
        "roi_iou_prev": _roi_iou(roi_xywh, prev_roi_xywh or prior_roi),
        "consensus_bbox_xywh_sensor": consensus_bbox,
        "score_map_local": score,
    }


def _gray_frame(frame_path: Path) -> Image.Image:
    return Image.open(frame_path).convert("L").convert("RGB")


def _score_map_to_image(score_map_local: np.ndarray) -> Image.Image:
    if score_map_local.size == 0:
        return Image.new("RGB", (1, 1), color=(0, 0, 0))
    peak = float(score_map_local.max(initial=0.0))
    if peak <= 0.0:
        arr = np.zeros(score_map_local.shape, dtype=np.uint8)
    else:
        arr = np.clip(np.round((score_map_local / peak) * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L").convert("RGB")
    return image


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _composite_panel(
    *,
    frame_path: Path,
    eye_bbox_xywh: list[float],
    roi_xywh: list[float],
    consensus_bbox_xywh: list[float] | None,
    score_map_local: np.ndarray | None,
    interval_index: int,
    events_total: int,
    source: str,
) -> Image.Image:
    frame = _gray_frame(frame_path)
    frame_draw = ImageDraw.Draw(frame)
    _draw_xywh(frame_draw, eye_bbox_xywh, outline="#00ff88", width=2)
    if consensus_bbox_xywh is not None:
        _draw_xywh(frame_draw, consensus_bbox_xywh, outline="#46a6ff", width=2)
    _draw_xywh(frame_draw, roi_xywh, outline="#ffd84d", width=2)
    frame_draw.text((6, 6), f"interval={interval_index} events={events_total}", fill=(255, 255, 255), font=ImageFont.load_default())
    frame_draw.text((6, 18), f"source={source}", fill=(255, 255, 255), font=ImageFont.load_default())

    if score_map_local is None:
        score_panel = Image.new("RGB", (frame.width, frame.height), color=(0, 0, 0))
    else:
        score_panel = Image.new("RGB", (frame.width, frame.height), color=(0, 0, 0))
        ex, ey, ew, eh = [float(v) for v in eye_bbox_xywh]
        x0 = max(0, int(math.floor(ex)))
        y0 = max(0, int(math.floor(ey)))
        x1 = min(frame.width, int(math.ceil(ex + ew)))
        y1 = min(frame.height, int(math.ceil(ey + eh)))
        local_image = _score_map_to_image(score_map_local)
        score_panel.paste(local_image, (x0, y0, x1, y1))
        draw = ImageDraw.Draw(score_panel)
        _draw_xywh(draw, eye_bbox_xywh, outline="#00ff88", width=2)
        if consensus_bbox_xywh is not None:
            _draw_xywh(draw, consensus_bbox_xywh, outline="#46a6ff", width=2)
        _draw_xywh(draw, roi_xywh, outline="#ffd84d", width=2)
        draw.text((6, 6), "event consensus", fill=(255, 255, 255), font=ImageFont.load_default())

    out = Image.new("RGB", (frame.width * 2, frame.height), color=(8, 8, 8))
    out.paste(frame, (0, 0))
    out.paste(score_panel, (frame.width, 0))
    return out


def _thumb_with_title(
    image: Image.Image,
    *,
    label: str,
    thumb_size_wh: tuple[int, int],
) -> Image.Image:
    thumb_w, thumb_h = thumb_size_wh
    title_h = 18
    out = Image.new("RGB", (thumb_w, thumb_h + title_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(out)
    draw.text((4, 2), label, fill=(230, 230, 230), font=ImageFont.load_default())
    thumb = image.copy()
    thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
    paste_x = (thumb_w - thumb.width) // 2
    paste_y = title_h + (thumb_h - thumb.height) // 2
    out.paste(thumb, (paste_x, paste_y))
    return out


def _save_pages(
    *,
    items: list[tuple[str, Image.Image]],
    out_dir: Path,
    title_prefix: str,
    cols: int = 2,
    rows: int = 3,
    thumb_size_wh: tuple[int, int] = (520, 220),
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[str] = []
    per_page = cols * rows
    for page_idx in range(0, len(items), per_page):
        chunk = items[page_idx : page_idx + per_page]
        thumb_w, thumb_h = thumb_size_wh
        tile_h = thumb_h + 18
        title_h = 28
        page = Image.new("RGB", (cols * thumb_w, rows * tile_h + title_h), color=(12, 12, 12))
        draw = ImageDraw.Draw(page)
        draw.text((8, 6), f"{title_prefix} page={page_idx // per_page + 1}", fill=(240, 240, 240), font=ImageFont.load_default())
        for local_idx, (label, image) in enumerate(chunk):
            r = local_idx // cols
            c = local_idx % cols
            x = c * thumb_w
            y = title_h + r * tile_h
            tile = _thumb_with_title(image, label=label, thumb_size_wh=thumb_size_wh)
            page.paste(tile, (x, y))
        out_path = out_dir / f"page_{page_idx // per_page + 1:03d}.png"
        page.save(out_path)
        page_paths.append(str(out_path))
    return page_paths


def _representative_indices(total: int, *, events: list[int]) -> dict[str, list[int]]:
    if total <= 0:
        return {"uniform": [], "burst": [], "sparse": []}
    uniform_count = min(24, total)
    uniform = sorted({int(round(v)) for v in np.linspace(0, total - 1, num=uniform_count)})
    ranked_hi = [idx for idx, _ in sorted(enumerate(events), key=lambda item: item[1], reverse=True)[:24]]
    ranked_lo = [idx for idx, _ in sorted(enumerate(events), key=lambda item: item[1])[:24]]
    return {
        "uniform": uniform,
        "burst": sorted(ranked_hi),
        "sparse": sorted(ranked_lo),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render event-guided pupil ROI proposals from interval event voxels.")
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "configs" / "paths" / "ev_eye_groundedsam_paths.json")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-code", type=str, default="201")
    parser.add_argument("--eyes", nargs="+", default=["left", "right"])
    parser.add_argument(
        "--anchor-summary-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_session_samples_7" / "roi_crop_prompted_all48_v10_native_then_crop128_rescue_same8samples",
    )
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=260)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_event_guided_pupil_roi",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = json.loads(args.paths_config.read_text(encoding="utf-8"))
    raw_root = Path(paths["raw_root"])
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for eye in args.eyes:
        summary_path = _v10_summary_path(args.anchor_summary_root, user_id=int(args.user_id), eye=str(eye), session_code=str(args.session_code))
        anchor = _load_anchor_prior(summary_path)
        eye_bbox = _clip_xywh_to_bounds(anchor["eye_bbox_xywh_sensor"], width=int(args.sensor_width), height=int(args.sensor_height))
        prior_pupil = _clip_xywh_to_bounds(anchor["pupil_bbox_xywh_sensor"], width=int(args.sensor_width), height=int(args.sensor_height))

        session_dir = _session_dir(raw_root, user_id=int(args.user_id), eye=str(eye), session_code=str(args.session_code))
        frames_dir = session_dir / "frames"
        events_path = session_dir / "events" / "events.txt"
        frame_records = collect_frame_records(frames_dir)
        events = _load_event_arrays(events_path)
        timestamps = np.asarray(events["t"], dtype=np.int64)
        xs = np.asarray(events["x"], dtype=np.int16)
        ys = np.asarray(events["y"], dtype=np.int16)
        ps = np.asarray(events["p"], dtype=np.int8)

        frame_ts = np.asarray([int(record.timestamp_us) for record in frame_records], dtype=np.int64)
        start_indices = np.searchsorted(timestamps, frame_ts[:-1], side="left")
        end_indices = np.searchsorted(timestamps, frame_ts[1:], side="left")
        interval_totals = [int(b - a) for a, b in zip(start_indices.tolist(), end_indices.tolist(), strict=False)]
        low_thr = float(np.percentile(interval_totals, 10))
        high_thr = float(np.percentile(interval_totals, 99))

        interval_rows: list[dict[str, Any]] = []
        prev_roi = prior_pupil
        preview_cache: dict[int, Image.Image] = {}
        rep_groups = _representative_indices(len(interval_totals), events=interval_totals)
        preview_needed = {idx for group in rep_groups.values() for idx in group}

        for interval_zero_idx, (start_idx, end_idx) in enumerate(zip(start_indices.tolist(), end_indices.tolist(), strict=False)):
            interval_x = xs[start_idx:end_idx]
            interval_y = ys[start_idx:end_idx]
            interval_p = ps[start_idx:end_idx]
            neg_mask = interval_p <= 0
            pos_mask = interval_p > 0
            neg_counts = _accumulate_counts(interval_x[neg_mask], interval_y[neg_mask], sensor_width=int(args.sensor_width), sensor_height=int(args.sensor_height))
            pos_counts = _accumulate_counts(interval_x[pos_mask], interval_y[pos_mask], sensor_width=int(args.sensor_width), sensor_height=int(args.sensor_height))
            result = _build_event_guided_roi(
                neg_counts=neg_counts,
                pos_counts=pos_counts,
                eye_bbox_xywh=eye_bbox,
                prior_pupil_xywh=prior_pupil,
                prev_roi_xywh=prev_roi,
                interval_events_total=interval_totals[interval_zero_idx],
                low_event_threshold=low_thr,
                high_event_threshold=high_thr,
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
            )
            row = {
                "interval_index": int(interval_zero_idx + 1),
                "frame_from": frame_records[interval_zero_idx].filename,
                "frame_to": frame_records[interval_zero_idx + 1].filename,
                "timestamp_from_us": int(frame_ts[interval_zero_idx]),
                "timestamp_to_us": int(frame_ts[interval_zero_idx + 1]),
                "events_total": int(interval_totals[interval_zero_idx]),
                "eye_bbox_xywh_sensor": eye_bbox,
                "prior_pupil_bbox_xywh_sensor": prior_pupil,
                "event_guided_roi_xywh_sensor": result["roi_xywh_sensor"],
                "event_guided_source": result["source"],
                "score_peak": result["score_peak"],
                "score_mass": result["score_mass"],
                "sparse_interval": result["sparse_interval"],
                "high_burst_interval": result["high_burst_interval"],
                "roi_iou_prev": result["roi_iou_prev"],
                "consensus_bbox_xywh_sensor": result["consensus_bbox_xywh_sensor"],
            }
            interval_rows.append(row)
            prev_roi = row["event_guided_roi_xywh_sensor"]

            if interval_zero_idx in preview_needed:
                preview_cache[interval_zero_idx] = _composite_panel(
                    frame_path=frames_dir / frame_records[interval_zero_idx].filename,
                    eye_bbox_xywh=eye_bbox,
                    roi_xywh=row["event_guided_roi_xywh_sensor"],
                    consensus_bbox_xywh=row["consensus_bbox_xywh_sensor"],
                    score_map_local=result.get("score_map_local"),
                    interval_index=int(interval_zero_idx + 1),
                    events_total=int(interval_totals[interval_zero_idx]),
                    source=str(result["source"]),
                )

        eye_root = output_root / str(eye)
        eye_root.mkdir(parents=True, exist_ok=True)
        stats_path = eye_root / "interval_event_guided_roi_stats.json"
        stats_path.write_text(json.dumps(interval_rows, indent=2), encoding="utf-8")

        preview_pages: dict[str, list[str]] = {}
        for group_name, indices in rep_groups.items():
            items = []
            for idx in indices:
                panel = preview_cache.get(idx)
                if panel is None:
                    continue
                items.append((f"{idx + 1:04d}", panel))
            preview_pages[group_name] = _save_pages(
                items=items,
                out_dir=eye_root / f"{group_name}_pages",
                title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} event-guided pupil ROI {group_name}",
            )

        roi_ws = [float(row["event_guided_roi_xywh_sensor"][2]) for row in interval_rows]
        roi_hs = [float(row["event_guided_roi_xywh_sensor"][3]) for row in interval_rows]
        source_counts: dict[str, int] = {}
        for row in interval_rows:
            source_counts[row["event_guided_source"]] = source_counts.get(row["event_guided_source"], 0) + 1

        summary_rows.append(
            {
                "user_id": int(args.user_id),
                "eye": str(eye),
                "session_code": str(args.session_code),
                "n_frames": len(frame_records),
                "n_intervals": len(interval_rows),
                "eye_bbox_xywh_sensor": eye_bbox,
                "prior_pupil_bbox_xywh_sensor": prior_pupil,
                "interval_event_count_p10": float(low_thr),
                "interval_event_count_p99": float(high_thr),
                "roi_width_mean": float(np.mean(roi_ws)) if roi_ws else 0.0,
                "roi_width_min": float(np.min(roi_ws)) if roi_ws else 0.0,
                "roi_width_max": float(np.max(roi_ws)) if roi_ws else 0.0,
                "roi_height_mean": float(np.mean(roi_hs)) if roi_hs else 0.0,
                "roi_height_min": float(np.min(roi_hs)) if roi_hs else 0.0,
                "roi_height_max": float(np.max(roi_hs)) if roi_hs else 0.0,
                "source_counts": source_counts,
                "preview_pages": preview_pages,
                "interval_stats_path": str(stats_path),
            }
        )
        print(f"[done] event-guided ROI rendered for {eye} intervals={len(interval_rows)}")

    summary_json_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    lines = [
        "# Event-Guided Pupil ROI Summary",
        "",
        f"- user_id: {int(args.user_id)}",
        f"- session_code: {str(args.session_code)}",
        f"- eyes: {', '.join(str(v) for v in args.eyes)}",
        "- method: consensus=min(blur(pos), blur(neg)) with prior-weighted robust centroid",
        "- anchor source: v10 sampled eye/pupil median per session eye",
        "- fallback policy: low-event intervals carry previous ROI or prior ROI",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['eye']}",
                f"- n_frames: {row['n_frames']}",
                f"- n_intervals: {row['n_intervals']}",
                f"- eye_bbox_xywh_sensor: {row['eye_bbox_xywh_sensor']}",
                f"- prior_pupil_bbox_xywh_sensor: {row['prior_pupil_bbox_xywh_sensor']}",
                f"- interval_event_count_p10: {row['interval_event_count_p10']:.2f}",
                f"- interval_event_count_p99: {row['interval_event_count_p99']:.2f}",
                f"- roi_width_mean/min/max: {row['roi_width_mean']:.2f} / {row['roi_width_min']:.2f} / {row['roi_width_max']:.2f}",
                f"- roi_height_mean/min/max: {row['roi_height_mean']:.2f} / {row['roi_height_min']:.2f} / {row['roi_height_max']:.2f}",
                f"- source_counts: {row['source_counts']}",
                f"- uniform_first_page: {row['preview_pages']['uniform'][0] if row['preview_pages']['uniform'] else ''}",
                f"- burst_first_page: {row['preview_pages']['burst'][0] if row['preview_pages']['burst'] else ''}",
                f"- sparse_first_page: {row['preview_pages']['sparse'][0] if row['preview_pages']['sparse'] else ''}",
                "",
            ]
        )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote outputs under {output_root}")


if __name__ == "__main__":
    main()
