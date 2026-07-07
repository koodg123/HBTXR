from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from hbtxr.preprocess.io_utils import collect_frame_records  # noqa: E402
from hbtxr.preprocess.target_fps_build import _load_event_arrays  # noqa: E402
from generate_time_sync_geometry_stable_result_panels import (  # noqa: E402
    _add_border,
    _crop_sparse_event_render,
    _horizontal_panel,
    _labeled_panel,
    _save_png,
)
from render_session_interval_event_3d import _render_3d_scatter  # noqa: E402
from render_session_interval_event_voxels import _accumulate_counts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild reference-style event interpolation result panels for the fi_ei_demo "
            "workspace using the actual helper code from the original figure pipeline."
        )
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT / "workspace" / "fi_ei_demo",
        help="Workspace root that contains raw_data, target_data, fi_target_data, and fi_ei_uniform5000.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "workspace" / "fi_ei_demo" / "visuals" / "reference_style_rebuilt",
        help="Output root for rebuilt panels.",
    )
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=260)
    parser.add_argument("--clean", action="store_true", help="Delete the output root before rendering.")
    return parser.parse_args()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_raw_session(workspace_root: Path) -> dict[str, Any]:
    session_dir = workspace_root / "raw_data" / "Data_davis" / "user2" / "right" / "session_1_0_2"
    frame_records = collect_frame_records(session_dir / "frames")
    events = _load_event_arrays(session_dir / "events" / "events.txt")
    return {
        "session_dir": session_dir,
        "frame_records": frame_records,
        "frame_timestamps_us": np.asarray([int(record.timestamp_us) for record in frame_records], dtype=np.int64),
        "event_t": np.asarray(events["t"], dtype=np.int64),
        "event_x": np.asarray(events["x"], dtype=np.int16),
        "event_y": np.asarray(events["y"], dtype=np.int16),
        "event_p": np.asarray(events["p"], dtype=np.int8),
    }


def _load_h5_session(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        return {
            "frame_timestamps_us": np.asarray(f["frames/timestamps_us"], dtype=np.int64),
            "event_ranges": np.asarray(f["frames/event_ranges"], dtype=np.int64),
            "event_t": np.asarray(f["events/t"], dtype=np.int64),
            "event_x": np.asarray(f["events/x"], dtype=np.int16),
            "event_y": np.asarray(f["events/y"], dtype=np.int16),
            "event_p": np.asarray(f["events/p"], dtype=np.int8),
        }


def _load_npz_session(path: Path) -> dict[str, Any]:
    with np.load(path) as d:
        return {
            "frame_timestamps_us": np.asarray(d["frame_timestamps_us"], dtype=np.int64),
            "event_ranges": np.asarray(d["frame_event_ranges"], dtype=np.int64),
            "event_t": np.asarray(d["event_t"], dtype=np.int64),
            "event_x": np.asarray(d["event_x"], dtype=np.int16),
            "event_y": np.asarray(d["event_y"], dtype=np.int16),
            "event_p": np.asarray(d["event_p"], dtype=np.int8),
        }


def _interval_rows_from_ranges(frame_timestamps_us: np.ndarray, event_ranges: np.ndarray) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for idx in range(len(frame_timestamps_us) - 1):
        start_idx, end_idx = [int(v) for v in event_ranges[idx + 1]]
        rows.append(
            {
                "interval_index": idx + 1,
                "timestamp_from_us": int(frame_timestamps_us[idx]),
                "timestamp_to_us": int(frame_timestamps_us[idx + 1]),
                "start_index": start_idx,
                "end_index": end_idx,
                "events_total": int(end_idx - start_idx),
            }
        )
    return rows


def _interval_rows_from_timestamps(frame_timestamps_us: np.ndarray, event_t: np.ndarray) -> list[dict[str, int]]:
    start_indices = np.searchsorted(event_t, frame_timestamps_us[:-1], side="left")
    end_indices = np.searchsorted(event_t, frame_timestamps_us[1:], side="left")
    rows: list[dict[str, int]] = []
    for idx, (start_idx, end_idx) in enumerate(zip(start_indices.tolist(), end_indices.tolist(), strict=False), start=1):
        rows.append(
            {
                "interval_index": idx,
                "timestamp_from_us": int(frame_timestamps_us[idx - 1]),
                "timestamp_to_us": int(frame_timestamps_us[idx]),
                "start_index": int(start_idx),
                "end_index": int(end_idx),
                "events_total": int(end_idx - start_idx),
            }
        )
    return rows


def _colorize_positive_counts(counts: np.ndarray) -> Image.Image:
    h, w = counts.shape
    rgb = np.full((h, w, 3), fill_value=(247, 241, 236), dtype=np.uint8)
    if np.any(counts > 0):
        mask = counts > 0
        scale = max(float(np.percentile(counts[mask], 99.5)), 1.0)
        red = np.clip(np.round((np.sqrt(counts[mask].astype(np.float32)) / math.sqrt(scale)) * 255.0), 0, 255).astype(np.uint8)
        rgb[..., 0][mask] = np.maximum(rgb[..., 0][mask], red)
        rgb[..., 1][mask] = np.minimum(rgb[..., 1][mask], 228)
        rgb[..., 2][mask] = np.minimum(rgb[..., 2][mask], 220)
    return Image.fromarray(rgb, mode="RGB")


def _colorize_polarity_overlay(pos_counts: np.ndarray, neg_counts: np.ndarray) -> Image.Image:
    h, w = pos_counts.shape
    rgb = np.full((h, w, 3), fill_value=(247, 247, 247), dtype=np.uint8)
    if np.any(pos_counts > 0):
        mask = pos_counts > 0
        scale = max(float(np.percentile(pos_counts[mask], 99.5)), 1.0)
        red = np.clip(np.round((np.sqrt(pos_counts[mask].astype(np.float32)) / math.sqrt(scale)) * 255.0), 0, 255).astype(np.uint8)
        rgb[..., 0][mask] = np.maximum(rgb[..., 0][mask], red)
        rgb[..., 1][mask] = np.minimum(rgb[..., 1][mask], 226)
        rgb[..., 2][mask] = np.minimum(rgb[..., 2][mask], 226)
    if np.any(neg_counts > 0):
        mask = neg_counts > 0
        scale = max(float(np.percentile(neg_counts[mask], 99.5)), 1.0)
        blue = np.clip(np.round((np.sqrt(neg_counts[mask].astype(np.float32)) / math.sqrt(scale)) * 255.0), 0, 255).astype(np.uint8)
        rgb[..., 2][mask] = np.maximum(rgb[..., 2][mask], blue)
        rgb[..., 0][mask] = np.minimum(rgb[..., 0][mask], 232)
        rgb[..., 1][mask] = np.minimum(rgb[..., 1][mask], 232)
    return Image.fromarray(rgb, mode="RGB")


def _positive_accum_crop(xs: np.ndarray, ys: np.ndarray, ps: np.ndarray, *, sensor_width: int, sensor_height: int) -> Image.Image:
    pos_mask = ps > 0
    counts = _accumulate_counts(xs[pos_mask], ys[pos_mask], sensor_width=sensor_width, sensor_height=sensor_height)
    image = _colorize_positive_counts(counts)
    return _add_border(_crop_sparse_event_render(image, threshold=244, pad_x=18, pad_y=18))


def _polarity_overlay_crop(xs: np.ndarray, ys: np.ndarray, ps: np.ndarray, *, sensor_width: int, sensor_height: int) -> Image.Image:
    pos_mask = ps > 0
    neg_mask = ~pos_mask
    pos_counts = _accumulate_counts(xs[pos_mask], ys[pos_mask], sensor_width=sensor_width, sensor_height=sensor_height)
    neg_counts = _accumulate_counts(xs[neg_mask], ys[neg_mask], sensor_width=sensor_width, sensor_height=sensor_height)
    image = _colorize_polarity_overlay(pos_counts, neg_counts)
    return _add_border(_crop_sparse_event_render(image, threshold=244, pad_x=18, pad_y=18))


def _render_event_3d_crop(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    sensor_width: int,
    sensor_height: int,
    title: str,
) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="hbtxr_ref_style_") as tmpdir:
        tmp_path = Path(tmpdir) / "event3d.png"
        _render_3d_scatter(
            t_us=np.asarray(ts, dtype=np.int64),
            x=np.asarray(xs, dtype=np.int16),
            y=np.asarray(ys, dtype=np.int16),
            p=np.asarray(ps, dtype=np.int8),
            title=title,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            output_path=tmp_path,
        )
        image = Image.open(tmp_path).convert("RGB")
    return _add_border(_crop_sparse_event_render(image, threshold=248, pad_x=28, pad_y=18))


def _save_gif(images: list[Path], output_path: Path, *, duration_ms: int) -> None:
    if not images:
        return
    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in images]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


def _save_pages(items: list[tuple[str, Path]], output_dir: Path, title_prefix: str, *, cols: int = 2, rows: int = 3) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[str] = []
    per_page = cols * rows
    thumb_w = 520
    thumb_h = 250
    for start in range(0, len(items), per_page):
        chunk = items[start : start + per_page]
        title_h = 28
        page = Image.new("RGB", (cols * thumb_w, rows * thumb_h + title_h), color=(10, 10, 14))
        draw = ImageDraw.Draw(page)
        draw.text((8, 8), f"{title_prefix} page={start // per_page + 1}", fill=(240, 240, 240))
        for local_idx, (_, image_path) in enumerate(chunk):
            row = local_idx // cols
            col = local_idx % cols
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h - 4), Image.Resampling.BILINEAR)
            x = col * thumb_w + (thumb_w - img.width) // 2
            y = title_h + row * thumb_h + (thumb_h - img.height) // 2
            page.paste(img, (x, y))
        page_path = output_dir / f"page_{start // per_page + 1:03d}.png"
        page.save(page_path)
        page_paths.append(str(page_path))
    return page_paths


def _render_reference_style_all_intervals(
    *,
    rows: list[dict[str, int]],
    event_t: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    event_p: np.ndarray,
    sensor_width: int,
    sensor_height: int,
    output_root: Path,
    label_prefix: str,
) -> dict[str, Any]:
    event3d_dir = _ensure_dir(output_root / "event_3d")
    positive_dir = _ensure_dir(output_root / "positive_accum")
    combo_dir = _ensure_dir(output_root / "combo")
    combo_paths: list[Path] = []
    summary_rows: list[dict[str, Any]] = []

    for row in rows:
        idx = int(row["interval_index"])
        start_idx = int(row["start_index"])
        end_idx = int(row["end_index"])
        xs = event_x[start_idx:end_idx]
        ys = event_y[start_idx:end_idx]
        ts = event_t[start_idx:end_idx]
        ps = event_p[start_idx:end_idx]
        title = f"{label_prefix} interval={idx:04d} events={int(end_idx - start_idx)}"
        event3d_crop = _render_event_3d_crop(
            xs=xs,
            ys=ys,
            ts=ts,
            ps=ps,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            title=title,
        )
        positive_crop = _positive_accum_crop(xs, ys, ps, sensor_width=sensor_width, sensor_height=sensor_height)
        event3d_path = _save_png(event3d_crop, event3d_dir / f"interval_{idx:04d}.png")
        positive_path = _save_png(positive_crop, positive_dir / f"interval_{idx:04d}.png")
        combo = _horizontal_panel([event3d_crop, positive_crop], padding=20, gap=24)
        combo_path = _save_png(combo, combo_dir / f"interval_{idx:04d}.png")
        combo_paths.append(combo_path)
        summary_rows.append(
            {
                "interval_index": idx,
                "timestamp_from_us": int(row["timestamp_from_us"]),
                "timestamp_to_us": int(row["timestamp_to_us"]),
                "events_total": int(end_idx - start_idx),
                "event_3d_path": str(event3d_path),
                "positive_accum_path": str(positive_path),
                "combo_path": str(combo_path),
            }
        )

    pages = _save_pages([(f"{int(row['interval_index']):04d}", path) for row, path in zip(summary_rows, combo_paths, strict=False)], output_root / "pages", f"{label_prefix} reference-style")
    gif_path = output_root / "all_intervals.gif"
    _save_gif(combo_paths, gif_path, duration_ms=120)
    summary = {
        "label_prefix": label_prefix,
        "interval_count": len(summary_rows),
        "gif_path": str(gif_path),
        "page_paths": pages,
        "rows": summary_rows,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _render_raw_timestamp_accum_compare(
    *,
    raw_rows: list[dict[str, int]],
    raw_event_x: np.ndarray,
    raw_event_y: np.ndarray,
    raw_event_p: np.ndarray,
    gen_event_t: np.ndarray,
    gen_event_x: np.ndarray,
    gen_event_y: np.ndarray,
    gen_event_p: np.ndarray,
    sensor_width: int,
    sensor_height: int,
    output_root: Path,
) -> dict[str, Any]:
    raw_accum_dir = _ensure_dir(output_root / "raw_accum")
    fi_accum_dir = _ensure_dir(output_root / "fi_accum_over_raw_intervals")
    compare_dir = _ensure_dir(output_root / "compare")
    compare_paths: list[Path] = []
    summary_rows: list[dict[str, Any]] = []

    for row in raw_rows:
        idx = int(row["interval_index"])
        raw_start_idx = int(row["start_index"])
        raw_end_idx = int(row["end_index"])
        t0 = int(row["timestamp_from_us"])
        t1 = int(row["timestamp_to_us"])
        gen_start_idx = int(np.searchsorted(gen_event_t, t0, side="left"))
        gen_end_idx = int(np.searchsorted(gen_event_t, t1, side="left"))

        raw_overlay = _polarity_overlay_crop(
            raw_event_x[raw_start_idx:raw_end_idx],
            raw_event_y[raw_start_idx:raw_end_idx],
            raw_event_p[raw_start_idx:raw_end_idx],
            sensor_width=sensor_width,
            sensor_height=sensor_height,
        )
        gen_overlay = _polarity_overlay_crop(
            gen_event_x[gen_start_idx:gen_end_idx],
            gen_event_y[gen_start_idx:gen_end_idx],
            gen_event_p[gen_start_idx:gen_end_idx],
            sensor_width=sensor_width,
            sensor_height=sensor_height,
        )
        raw_path = _save_png(raw_overlay, raw_accum_dir / f"interval_{idx:04d}.png")
        gen_path = _save_png(gen_overlay, fi_accum_dir / f"interval_{idx:04d}.png")
        compare = _horizontal_panel(
            [
                _labeled_panel(raw_overlay, "original test-session accumulation"),
                _labeled_panel(gen_overlay, "FI-session accumulation over same raw timestamps"),
            ],
            padding=22,
            gap=26,
        )
        compare_path = _save_png(compare, compare_dir / f"interval_{idx:04d}.png")
        compare_paths.append(compare_path)
        summary_rows.append(
            {
                "interval_index": idx,
                "timestamp_from_us": t0,
                "timestamp_to_us": t1,
                "raw_total": int(raw_end_idx - raw_start_idx),
                "fi_total": int(gen_end_idx - gen_start_idx),
                "raw_accum_path": str(raw_path),
                "fi_accum_path": str(gen_path),
                "compare_path": str(compare_path),
            }
        )

    pages = _save_pages([(f"{int(row['interval_index']):04d}", path) for row, path in zip(summary_rows, compare_paths, strict=False)], output_root / "pages", "raw timestamp accumulation compare")
    gif_path = output_root / "all_intervals.gif"
    _save_gif(compare_paths, gif_path, duration_ms=150)
    summary = {
        "interval_count": len(summary_rows),
        "polarity_colors": {"positive": "red", "negative": "blue"},
        "gif_path": str(gif_path),
        "page_paths": pages,
        "rows": summary_rows,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if bool(args.clean) and args.output_root.exists():
        shutil.rmtree(args.output_root)
    _ensure_dir(args.output_root)

    raw_session = _load_raw_session(args.workspace_root)
    fi_session = _load_npz_session(
        args.workspace_root
        / "fi_ei_uniform5000"
        / "fps_100"
        / "sessions"
        / "user02"
        / "right"
        / "session_102"
        / "session_arrays.npz"
    )

    raw_rows = _interval_rows_from_timestamps(raw_session["frame_timestamps_us"], raw_session["event_t"])
    fi_rows = _interval_rows_from_ranges(fi_session["frame_timestamps_us"], fi_session["event_ranges"])

    raw_summary = _render_reference_style_all_intervals(
        rows=raw_rows,
        event_t=raw_session["event_t"],
        event_x=raw_session["event_x"],
        event_y=raw_session["event_y"],
        event_p=raw_session["event_p"],
        sensor_width=int(args.sensor_width),
        sensor_height=int(args.sensor_height),
        output_root=args.output_root / "original_test_session_all_intervals",
        label_prefix="original_test_session",
    )
    fi_summary = _render_reference_style_all_intervals(
        rows=fi_rows,
        event_t=fi_session["event_t"],
        event_x=fi_session["event_x"],
        event_y=fi_session["event_y"],
        event_p=fi_session["event_p"],
        sensor_width=int(args.sensor_width),
        sensor_height=int(args.sensor_height),
        output_root=args.output_root / "fi_session_all_intervals",
        label_prefix="fi_session",
    )
    compare_summary = _render_raw_timestamp_accum_compare(
        raw_rows=raw_rows,
        raw_event_x=raw_session["event_x"],
        raw_event_y=raw_session["event_y"],
        raw_event_p=raw_session["event_p"],
        gen_event_t=fi_session["event_t"],
        gen_event_x=fi_session["event_x"],
        gen_event_y=fi_session["event_y"],
        gen_event_p=fi_session["event_p"],
        sensor_width=int(args.sensor_width),
        sensor_height=int(args.sensor_height),
        output_root=args.output_root / "raw_timestamp_accum_compare",
    )

    top_summary = {
        "workspace_root": str(args.workspace_root),
        "output_root": str(args.output_root),
        "original_test_session_all_intervals": raw_summary,
        "fi_session_all_intervals": fi_summary,
        "raw_timestamp_accum_compare": compare_summary,
    }
    (args.output_root / "summary.json").write_text(json.dumps(top_summary, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "original_interval_count": raw_summary["interval_count"],
            "fi_interval_count": fi_summary["interval_count"],
            "compare_interval_count": compare_summary["interval_count"],
            "original_first_page": raw_summary["page_paths"][0] if raw_summary["page_paths"] else None,
            "fi_first_page": fi_summary["page_paths"][0] if fi_summary["page_paths"] else None,
            "compare_first_page": compare_summary["page_paths"][0] if compare_summary["page_paths"] else None,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
