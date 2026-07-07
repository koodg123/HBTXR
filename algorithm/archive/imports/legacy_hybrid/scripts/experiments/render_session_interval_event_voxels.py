from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hbtxr.preprocess.io_utils import collect_frame_records  # noqa: E402
from hbtxr.preprocess.target_fps_build import _load_event_arrays  # noqa: E402


def _session_dir(raw_root: Path, *, user_id: int, eye: str, session_code: str) -> Path:
    return raw_root / f"user{int(user_id)}" / str(eye) / f"session_{session_code[0]}_{session_code[1]}_{session_code[2]}"


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


def _counts_to_image(counts: np.ndarray, *, point_radius: int = 1) -> Image.Image:
    if counts.size == 0:
        return Image.new("L", (1, 1), color=0)
    if point_radius <= 0:
        scale = float(np.percentile(counts[counts > 0], 99.5)) if np.any(counts > 0) else 1.0
        scale = max(scale, 1.0)
        arr = np.zeros_like(counts, dtype=np.uint8)
        mask = counts > 0
        arr[mask] = np.clip(np.round((np.sqrt(counts[mask].astype(np.float32)) / math.sqrt(scale)) * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    canvas = Image.new("L", (counts.shape[1], counts.shape[0]), color=0)
    draw = ImageDraw.Draw(canvas)
    active_y, active_x = np.nonzero(counts > 0)
    if active_x.size == 0:
        return canvas
    active_counts = counts[active_y, active_x].astype(np.float32)
    scale = float(np.percentile(active_counts, 99.5)) if active_counts.size else 1.0
    scale = max(scale, 1.0)
    for x, y, value in zip(active_x.tolist(), active_y.tolist(), active_counts.tolist(), strict=False):
        intensity = int(np.clip(round((math.sqrt(float(value)) / math.sqrt(scale)) * 255.0), 48, 255))
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
            fill=intensity,
        )
    return canvas


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
    thumb = image.convert("RGB")
    thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.NEAREST)
    paste_x = (thumb_w - thumb.width) // 2
    paste_y = title_h + (thumb_h - thumb.height) // 2
    out.paste(thumb, (paste_x, paste_y))
    return out


def _save_pages(
    *,
    items: list[tuple[str, Image.Image]],
    out_dir: Path,
    title_prefix: str,
    cols: int = 5,
    rows: int = 4,
    thumb_size_wh: tuple[int, int] = (240, 180),
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


def _render_interval_images(
    *,
    timestamps: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    frame_records: list[Any],
    sensor_width: int,
    sensor_height: int,
    point_radius: int,
) -> tuple[list[tuple[str, Image.Image]], list[tuple[str, Image.Image]], list[dict[str, Any]]]:
    frame_ts = np.asarray([int(record.timestamp_us) for record in frame_records], dtype=np.int64)
    start_indices = np.searchsorted(timestamps, frame_ts[:-1], side="left")
    end_indices = np.searchsorted(timestamps, frame_ts[1:], side="left")
    neg_items: list[tuple[str, Image.Image]] = []
    pos_items: list[tuple[str, Image.Image]] = []
    stats: list[dict[str, Any]] = []

    for i, (start_idx, end_idx) in enumerate(zip(start_indices.tolist(), end_indices.tolist(), strict=False), start=1):
        interval_x = xs[start_idx:end_idx]
        interval_y = ys[start_idx:end_idx]
        interval_p = ps[start_idx:end_idx]
        neg_mask = interval_p <= 0
        pos_mask = interval_p > 0
        neg_counts = _accumulate_counts(interval_x[neg_mask], interval_y[neg_mask], sensor_width=sensor_width, sensor_height=sensor_height)
        pos_counts = _accumulate_counts(interval_x[pos_mask], interval_y[pos_mask], sensor_width=sensor_width, sensor_height=sensor_height)
        neg_image = _counts_to_image(neg_counts, point_radius=point_radius)
        pos_image = _counts_to_image(pos_counts, point_radius=point_radius)
        label = f"{i:04d}"
        neg_items.append((label, neg_image))
        pos_items.append((label, pos_image))
        stats.append(
            {
                "interval_index": i,
                "frame_from": frame_records[i - 1].filename,
                "frame_to": frame_records[i].filename,
                "timestamp_from_us": int(frame_ts[i - 1]),
                "timestamp_to_us": int(frame_ts[i]),
                "events_total": int(end_idx - start_idx),
                "events_negative": int(neg_mask.sum()),
                "events_positive": int(pos_mask.sum()),
                "neg_active_pixels": int(np.count_nonzero(neg_counts)),
                "pos_active_pixels": int(np.count_nonzero(pos_counts)),
            }
        )
    return neg_items, pos_items, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render per-frame-interval polarity-separated accumulated event images.")
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "configs" / "paths" / "ev_eye_groundedsam_paths.json")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-code", type=str, default="201")
    parser.add_argument("--eyes", nargs="+", default=["left", "right"])
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=260)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_interval_events")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = json.loads(args.paths_config.read_text(encoding="utf-8"))
    raw_root = Path(paths["raw_root"])
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for eye in args.eyes:
        session_dir = _session_dir(raw_root, user_id=int(args.user_id), eye=str(eye), session_code=str(args.session_code))
        frames_dir = session_dir / "frames"
        events_path = session_dir / "events" / "events.txt"
        frame_records = collect_frame_records(frames_dir)
        events = _load_event_arrays(events_path)
        timestamps = np.asarray(events["t"], dtype=np.int64)
        xs = np.asarray(events["x"], dtype=np.int16)
        ys = np.asarray(events["y"], dtype=np.int16)
        ps = np.asarray(events["p"], dtype=np.int8)

        eye_root = output_root / str(eye)
        neg_dir = eye_root / "negative_pages"
        pos_dir = eye_root / "positive_pages"
        neg_items, pos_items, interval_stats = _render_interval_images(
            timestamps=timestamps,
            xs=xs,
            ys=ys,
            ps=ps,
            frame_records=frame_records,
            sensor_width=int(args.sensor_width),
            sensor_height=int(args.sensor_height),
            point_radius=int(args.point_radius),
        )
        neg_pages = _save_pages(
            items=neg_items,
            out_dir=neg_dir,
            title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} negative polarity",
        )
        pos_pages = _save_pages(
            items=pos_items,
            out_dir=pos_dir,
            title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} positive polarity",
        )
        stats_path = eye_root / "interval_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(interval_stats, indent=2), encoding="utf-8")
        total_events = sum(int(row["events_total"]) for row in interval_stats)
        summary_rows.append(
            {
                "user_id": int(args.user_id),
                "eye": str(eye),
                "session_code": str(args.session_code),
                "n_frames": len(frame_records),
                "n_intervals": len(interval_stats),
                "events_total": int(total_events),
                "events_per_interval_mean": float(total_events / max(1, len(interval_stats))),
                "events_per_interval_min": int(min((row["events_total"] for row in interval_stats), default=0)),
                "events_per_interval_max": int(max((row["events_total"] for row in interval_stats), default=0)),
                "negative_pages": neg_pages,
                "positive_pages": pos_pages,
                "interval_stats_path": str(stats_path),
            }
        )
        print(f"[done] rendered {len(interval_stats)} intervals for {eye}")

    summary_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    lines = [
        "# Session Interval Event Accumulation Summary",
        "",
        f"- user_id: {int(args.user_id)}",
        f"- session_code: {str(args.session_code)}",
        f"- eyes: {', '.join(str(v) for v in args.eyes)}",
        f"- sensor_size_wh: ({int(args.sensor_width)}, {int(args.sensor_height)})",
        "- rendering policy: events between consecutive frame timestamps only",
        "- polarity output: separate negative / positive pages",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['eye']}",
                f"- n_frames: {row['n_frames']}",
                f"- n_intervals: {row['n_intervals']}",
                f"- events_total: {row['events_total']}",
                f"- events_per_interval_mean: {row['events_per_interval_mean']:.2f}",
                f"- events_per_interval_min: {row['events_per_interval_min']}",
                f"- events_per_interval_max: {row['events_per_interval_max']}",
                f"- negative_first_page: {row['negative_pages'][0] if row['negative_pages'] else ''}",
                f"- positive_first_page: {row['positive_pages'][0] if row['positive_pages'] else ''}",
                "",
            ]
        )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote outputs under {output_root}")


if __name__ == "__main__":
    main()
