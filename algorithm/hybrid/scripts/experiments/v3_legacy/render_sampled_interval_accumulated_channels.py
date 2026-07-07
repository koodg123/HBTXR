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

from hbtxr.preprocess.io_utils import collect_frame_records, load_events_from_txt  # noqa: E402


def _session_dir(raw_root: Path, *, user_id: int, eye: str, session_code: str) -> Path:
    return raw_root / f"user{int(user_id)}" / str(eye) / f"session_{session_code[0]}_{session_code[1]}_{session_code[2]}"


def _resolve_raw_root(explicit_raw_root: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_raw_root is not None:
        candidates.append(explicit_raw_root)

    paths_config = PROJECT_ROOT / "configs" / "paths" / "ev_eye_groundedsam_paths.json"
    if paths_config.exists():
        try:
            payload = json.loads(paths_config.read_text(encoding="utf-8"))
            raw_root = payload.get("raw_root")
            if raw_root:
                candidates.append(Path(str(raw_root)))
        except json.JSONDecodeError:
            pass

    candidates.extend(
        [
            PROJECT_ROOT.parent / "dataset" / "EYE" / "EV_Eye" / "raw_dataset" / "Data_davis",
            PROJECT_ROOT / "dataset" / "EYE" / "EV_Eye" / "raw_dataset" / "Data_davis",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve raw_root. Pass --raw-root explicitly or update configs/paths/ev_eye_groundedsam_paths.json."
    )


def _load_selection_summary(summary_path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]
    return {str(row["eye"]): row for row in rows}


def _load_event_arrays(event_file: Path, *, eye: str) -> dict[str, np.ndarray]:
    suffix = event_file.suffix.lower()
    if suffix == ".npz":
        raw = np.load(event_file)
        timestamp_key = "t" if "t" in raw.files else "timestamp_us"
        timestamps = np.asarray(raw[timestamp_key], dtype=np.int64)
        xs = np.asarray(raw["x"], dtype=np.int16)
        ys = np.asarray(raw["y"], dtype=np.int16)
        ps = np.asarray(raw["p"], dtype=np.int8)
    elif suffix == ".txt":
        raw = load_events_from_txt(event_file, eye=eye)
        timestamps = np.asarray(raw["t"], dtype=np.int64)
        xs = np.asarray(raw["x"], dtype=np.int16)
        ys = np.asarray(raw["y"], dtype=np.int16)
        ps = np.asarray(raw["p"], dtype=np.int8)
    else:
        raise ValueError(f"Unsupported event file format: {event_file}")
    order = np.argsort(timestamps, kind="stable")
    return {
        "t": timestamps[order],
        "x": xs[order],
        "y": ys[order],
        "p": ps[order],
    }


def _build_interval_rows(frame_timestamps_us: np.ndarray, event_timestamps_us: np.ndarray) -> list[dict[str, int]]:
    interval_rows: list[dict[str, int]] = []
    for idx in range(len(frame_timestamps_us) - 1):
        start_us = int(frame_timestamps_us[idx])
        end_us = int(frame_timestamps_us[idx + 1])
        start_idx = int(np.searchsorted(event_timestamps_us, start_us, side="left"))
        end_idx = int(np.searchsorted(event_timestamps_us, end_us, side="left"))
        interval_rows.append(
            {
                "interval_index": int(idx + 1),
                "start_us": start_us,
                "end_us": end_us,
                "events_total": int(end_idx - start_idx),
            }
        )
    return interval_rows


def _accumulate_counts(xs: np.ndarray, ys: np.ndarray, *, sensor_width: int, sensor_height: int) -> np.ndarray:
    counts = np.zeros((sensor_height, sensor_width), dtype=np.uint16)
    if xs.size == 0 or ys.size == 0:
        return counts
    valid = (xs >= 0) & (xs < sensor_width) & (ys >= 0) & (ys < sensor_height)
    if not np.any(valid):
        return counts
    np.add.at(counts, (ys[valid], xs[valid]), 1)
    return counts


def _normalize_counts(counts: np.ndarray, *, percentile: float = 99.5) -> tuple[np.ndarray, float]:
    active = counts[counts > 0].astype(np.float32)
    if active.size == 0:
        return np.zeros_like(counts, dtype=np.float32), 1.0
    vmax = float(np.percentile(active, percentile))
    vmax = max(vmax, 1.0)
    norm = np.zeros_like(counts, dtype=np.float32)
    mask = counts > 0
    norm[mask] = np.clip(np.sqrt(counts[mask].astype(np.float32) / vmax), 0.0, 1.0)
    return norm, vmax


def _colorize_counts(counts: np.ndarray, *, fg_rgb: tuple[int, int, int], bg_rgb: tuple[int, int, int]) -> tuple[Image.Image, float]:
    norm, vmax = _normalize_counts(counts)
    image = np.empty((counts.shape[0], counts.shape[1], 3), dtype=np.uint8)
    bg = np.asarray(bg_rgb, dtype=np.float32)
    fg = np.asarray(fg_rgb, dtype=np.float32)
    alpha = (0.10 + 0.90 * norm)[..., None]
    rgb = bg[None, None, :] * (1.0 - alpha) + fg[None, None, :] * alpha
    zero_mask = counts <= 0
    rgb[zero_mask] = bg
    image[...] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB"), vmax


def _compose_interval_panel(
    *,
    neg_image: Image.Image,
    pos_image: Image.Image,
    interval_index: int,
    events_total: int,
) -> Image.Image:
    gap = 16
    pad = 14
    title_h = 24
    subtitle_h = 16
    width = neg_image.width + pos_image.width + gap + pad * 2
    height = max(neg_image.height, pos_image.height) + title_h + subtitle_h + pad * 2
    panel = Image.new("RGB", (width, height), color=(12, 12, 12))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((pad, 6), f"interval={int(interval_index):04d} events={int(events_total)}", fill=(240, 240, 240), font=font)
    draw.text((pad, title_h), "negative", fill=(190, 220, 255), font=font)
    draw.text((pad + neg_image.width + gap, title_h), "positive", fill=(255, 195, 185), font=font)
    y0 = title_h + subtitle_h
    panel.paste(neg_image, (pad, y0))
    panel.paste(pos_image, (pad + neg_image.width + gap, y0))
    return panel


def _thumb_with_title(image: Image.Image, *, label: str, thumb_size_wh: tuple[int, int]) -> Image.Image:
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
            row = local_idx // cols
            col = local_idx % cols
            x = col * thumb_w
            y = title_h + row * tile_h
            tile = _thumb_with_title(image, label=label, thumb_size_wh=thumb_size_wh)
            page.paste(tile, (x, y))
        out_path = out_dir / f"page_{page_idx // per_page + 1:03d}.png"
        page.save(out_path)
        page_paths.append(str(out_path))
    return page_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render accumulated event-channel results for sampled interval event streams.")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-code", type=str, default="201")
    parser.add_argument("--eyes", nargs="+", default=["left", "right"])
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=260)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--selection-summary",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_interval_events_axis_swapped_views" / "summary.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_interval_events_accumulated_channels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = _resolve_raw_root(args.raw_root)
    selection_rows = _load_selection_summary(args.selection_summary)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for eye in args.eyes:
        selection_row = selection_rows[str(eye)]
        session_dir = _session_dir(raw_root, user_id=int(args.user_id), eye=str(eye), session_code=str(args.session_code))
        frames_dir = session_dir / "frames"
        event_file = session_dir / "events" / "events.txt"
        frame_records = collect_frame_records(frames_dir)
        events = _load_event_arrays(event_file, eye=str(eye))
        frame_timestamps_us = np.asarray([int(record.timestamp_us) for record in frame_records], dtype=np.int64)
        interval_rows = _build_interval_rows(frame_timestamps_us, np.asarray(events["t"], dtype=np.int64))

        selected_groups = {str(name): [int(v) for v in values] for name, values in selection_row["selected_groups"].items()}
        all_selected = sorted({interval_idx for values in selected_groups.values() for interval_idx in values})
        eye_root = output_root / str(eye)
        neg_dir = eye_root / "negative_images"
        pos_dir = eye_root / "positive_images"
        neg_dir.mkdir(parents=True, exist_ok=True)
        pos_dir.mkdir(parents=True, exist_ok=True)

        panel_by_interval: dict[int, Image.Image] = {}
        interval_stats: list[dict[str, Any]] = []

        for interval_idx in all_selected:
            row = interval_rows[int(interval_idx) - 1]
            start_idx = int(np.searchsorted(np.asarray(events["t"]), int(row["start_us"]), side="left"))
            end_idx = int(np.searchsorted(np.asarray(events["t"]), int(row["end_us"]), side="left"))
            interval_x = np.asarray(events["x"])[start_idx:end_idx]
            interval_y = np.asarray(events["y"])[start_idx:end_idx]
            interval_p = np.asarray(events["p"])[start_idx:end_idx]
            neg_mask = interval_p <= 0
            pos_mask = interval_p > 0

            neg_counts = _accumulate_counts(
                interval_x[neg_mask],
                interval_y[neg_mask],
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
            )
            pos_counts = _accumulate_counts(
                interval_x[pos_mask],
                interval_y[pos_mask],
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
            )

            neg_image, neg_vmax = _colorize_counts(neg_counts, fg_rgb=(34, 113, 220), bg_rgb=(245, 249, 253))
            pos_image, pos_vmax = _colorize_counts(pos_counts, fg_rgb=(200, 48, 34), bg_rgb=(251, 244, 239))

            neg_path = neg_dir / f"user{int(args.user_id):02d}_session{str(args.session_code)}_{eye}_interval_{int(interval_idx)}_negative_accum.png"
            pos_path = pos_dir / f"user{int(args.user_id):02d}_session{str(args.session_code)}_{eye}_interval_{int(interval_idx)}_positive_accum.png"
            neg_image.save(neg_path)
            pos_image.save(pos_path)

            panel_by_interval[int(interval_idx)] = _compose_interval_panel(
                neg_image=neg_image,
                pos_image=pos_image,
                interval_index=int(interval_idx),
                events_total=int(row["events_total"]),
            )
            interval_stats.append(
                {
                    "interval_index": int(interval_idx),
                    "frame_from": frame_records[int(interval_idx) - 1].filename,
                    "frame_to": frame_records[int(interval_idx)].filename,
                    "start_us": int(row["start_us"]),
                    "end_us": int(row["end_us"]),
                    "events_total": int(row["events_total"]),
                    "negative_events": int(neg_mask.sum()),
                    "positive_events": int(pos_mask.sum()),
                    "negative_nonzero_pixels": int(np.count_nonzero(neg_counts)),
                    "positive_nonzero_pixels": int(np.count_nonzero(pos_counts)),
                    "negative_max_count": int(np.max(neg_counts)),
                    "positive_max_count": int(np.max(pos_counts)),
                    "negative_render_vmax": float(neg_vmax),
                    "positive_render_vmax": float(pos_vmax),
                    "negative_image_path": str(neg_path),
                    "positive_image_path": str(pos_path),
                }
            )

        pages_by_group: dict[str, list[str]] = {}
        for group_name, interval_indices in selected_groups.items():
            items = [
                (f"{int(interval_idx):04d} | n={next(row['events_total'] for row in interval_stats if row['interval_index'] == int(interval_idx))}", panel_by_interval[int(interval_idx)])
                for interval_idx in interval_indices
            ]
            pages_by_group[group_name] = _save_pages(
                items=items,
                out_dir=eye_root / f"{group_name}_pages",
                title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} accumulated channels {group_name}",
            )

        stats_path = eye_root / "interval_stats.json"
        stats_path.write_text(json.dumps(interval_stats, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "user_id": int(args.user_id),
                "eye": str(eye),
                "session_code": str(args.session_code),
                "n_frames": len(frame_records),
                "n_intervals": len(interval_rows),
                "n_selected_intervals": len(all_selected),
                "selected_groups": selected_groups,
                "group_pages": pages_by_group,
                "interval_stats_path": str(stats_path),
            }
        )
        print(f"[done] rendered sampled accumulated channels for {eye} selected={len(all_selected)}")

    summary_json_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    lines = [
        "# Sampled Interval Accumulated Channels Summary",
        "",
        f"- user_id: {int(args.user_id)}",
        f"- session_code: {str(args.session_code)}",
        f"- eyes: {', '.join(str(v) for v in args.eyes)}",
        "- rendering policy: same sampled interval set as generated event streams",
        "- panel layout: negative accumulation + positive accumulation",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['eye']}",
                f"- n_frames: {row['n_frames']}",
                f"- n_intervals: {row['n_intervals']}",
                f"- n_selected_intervals: {row['n_selected_intervals']}",
                f"- uniform_first_page: {row['group_pages']['uniform'][0] if row['group_pages'].get('uniform') else ''}",
                f"- burst_first_page: {row['group_pages']['burst'][0] if row['group_pages'].get('burst') else ''}",
                f"- sparse_first_page: {row['group_pages']['sparse'][0] if row['group_pages'].get('sparse') else ''}",
                "",
            ]
        )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote sampled accumulated-channel outputs under {output_root}")


if __name__ == "__main__":
    main()
