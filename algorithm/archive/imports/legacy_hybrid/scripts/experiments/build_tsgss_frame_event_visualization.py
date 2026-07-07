#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import SENSOR_HEIGHT, SENSOR_WIDTH, maybe_link_or_copy
from hbtxr.utils.io import read_json, write_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render frame-to-frame plus event-voxel visualizations from tsgss event-alignment outputs."
        ),
    )
    parser.add_argument(
        "--alignment-root",
        type=str,
        default="tsgss/event_alignment",
        help="Input root produced by build_tsgss_event_alignment.py",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="tsgss/frame_event_visualization",
        help="Output root for visualization assets",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Temporal bins for event voxel visualization",
    )
    parser.add_argument(
        "--sensor-width",
        type=int,
        default=SENSOR_WIDTH,
    )
    parser.add_argument(
        "--sensor-height",
        type=int,
        default=SENSOR_HEIGHT,
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        help="Optional limit for smoke-testing a subset of sessions",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy", "skip"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _read_session_summaries(alignment_root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((alignment_root / "sessions").rglob("summary.json"))]


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _session_output_dirs(output_root: Path, *, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
    base = output_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}"
    return {
        "base": base,
        "assets": base / "assets",
    }


def _load_frame_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("L").convert("RGB")
    return image


def _normalize_heatmap(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.log1p(np.maximum(values, 0.0))
    denom = float(max(values.max(), 1.0))
    return values / denom


def _build_temporal_voxel(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    ts: np.ndarray,
    valid_mask: np.ndarray,
    start_timestamp_us: int,
    end_timestamp_us: int,
    sensor_width: int,
    sensor_height: int,
    n_bins: int,
) -> np.ndarray:
    voxel = np.zeros((int(n_bins), int(sensor_height), int(sensor_width)), dtype=np.float32)
    if not np.any(valid_mask):
        return voxel

    valid_x = np.asarray(xs[valid_mask], dtype=np.int64)
    valid_y = np.asarray(ys[valid_mask], dtype=np.int64)
    valid_p = np.asarray(ps[valid_mask], dtype=np.int64)
    valid_t = np.asarray(ts[valid_mask], dtype=np.int64)

    duration = max(1, int(end_timestamp_us) - int(start_timestamp_us))
    rel = (valid_t.astype(np.float64) - float(start_timestamp_us)) / float(duration)
    rel = np.clip(rel, 0.0, 0.999999)
    bin_idx = np.floor(rel * float(n_bins)).astype(np.int64)
    weights = np.where(valid_p > 0, 1.0, -1.0).astype(np.float32)
    np.add.at(voxel, (bin_idx, valid_y, valid_x), weights)
    return voxel


def _render_voxel_strip(
    *,
    voxel: np.ndarray,
    target_bin_size: tuple[int, int] = (84, 120),
) -> Image.Image:
    bin_images: list[Image.Image] = []
    pos_global = _normalize_heatmap(np.maximum(voxel, 0.0))
    neg_global = _normalize_heatmap(np.maximum(-voxel, 0.0))

    for bin_index in range(voxel.shape[0]):
        pos = pos_global[bin_index]
        neg = neg_global[bin_index]
        rgb = np.zeros((voxel.shape[1], voxel.shape[2], 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(pos * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb[..., 1] = np.clip(np.minimum(pos, neg) * 92.0, 0.0, 255.0).astype(np.uint8)
        rgb[..., 2] = np.clip(neg * 255.0, 0.0, 255.0).astype(np.uint8)
        tile = Image.fromarray(rgb, mode="RGB")
        tile = ImageOps.pad(tile, target_bin_size, color=(10, 10, 10))
        tile = ImageOps.expand(tile, border=2, fill=(58, 68, 84))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile.width - 1, 18), fill=(19, 22, 28))
        draw.text((6, 3), f"B{bin_index}", fill=(235, 235, 235))
        bin_images.append(tile)

    gap = 6
    width = sum(image.width for image in bin_images) + gap * max(0, len(bin_images) - 1)
    height = max(image.height for image in bin_images)
    strip = Image.new("RGB", (width, height), color=(18, 18, 20))
    cursor_x = 0
    for image in bin_images:
        strip.paste(image, (cursor_x, 0))
        cursor_x += image.width + gap
    return strip


def _render_triptych(
    *,
    row: dict[str, Any],
    voxel_strip: Image.Image,
    start_frame: Image.Image,
    end_frame: Image.Image,
) -> Image.Image:
    frame_tile_size = (220, 156)
    start_tile = ImageOps.pad(start_frame, frame_tile_size, color=(225, 225, 225))
    end_tile = ImageOps.pad(end_frame, frame_tile_size, color=(225, 225, 225))
    start_tile = ImageOps.expand(start_tile, border=3, fill=(66, 128, 82))
    end_tile = ImageOps.expand(end_tile, border=3, fill=(132, 74, 40))

    target_voxel_height = start_tile.height
    if voxel_strip.height != target_voxel_height:
        target_width = max(1, int(round(voxel_strip.width * (target_voxel_height / float(voxel_strip.height)))))
        voxel_strip = voxel_strip.resize((target_width, target_voxel_height), resample=Image.Resampling.BILINEAR)

    gap = 14
    title_h = 34
    footer_h = 24
    width = start_tile.width + voxel_strip.width + end_tile.width + gap * 4
    height = title_h + max(start_tile.height, voxel_strip.height, end_tile.height) + footer_h
    canvas = Image.new("RGB", (width, height), color=(246, 243, 237))
    draw = ImageDraw.Draw(canvas)

    title = (
        f"Interval {int(row['interval_index']):02d} | {int(row['interval_us'])/1000.0:.1f} ms | "
        f"{int(row['raw_event_count'])} -> {int(row['target_event_count'])} | {row['alignment_mode']}"
    )
    draw.text((gap, 8), title, fill=(30, 29, 27))
    y = title_h
    x = gap
    canvas.paste(start_tile, (x, y))
    x += start_tile.width + gap
    canvas.paste(voxel_strip, (x, y))
    x += voxel_strip.width + gap
    canvas.paste(end_tile, (x, y))
    draw.text((gap, height - 18), "Start Frame", fill=(48, 72, 56))
    draw.text((gap + start_tile.width + gap + 8, height - 18), "Event Voxel", fill=(56, 60, 88))
    draw.text((width - end_tile.width - gap, height - 18), "End Frame", fill=(88, 56, 42))
    return canvas


def _build_contact_sheet(triptychs: list[Image.Image], *, save_path: Path) -> None:
    thumb_size = (360, 120)
    gap = 18
    cols = min(2, max(1, len(triptychs)))
    rows = int(math.ceil(len(triptychs) / cols))
    canvas = Image.new(
        "RGB",
        (gap + cols * (thumb_size[0] + gap), gap + rows * (thumb_size[1] + gap)),
        color=(252, 248, 242),
    )
    for index, triptych in enumerate(triptychs):
        tile = ImageOps.pad(triptych, thumb_size, color=(236, 233, 227))
        x = gap + (index % cols) * (thumb_size[0] + gap)
        y = gap + (index // cols) * (thumb_size[1] + gap)
        canvas.paste(tile, (x, y))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def _write_readme(output_root: Path, quick_preview_root: Path, summary: dict[str, Any], mode_examples: dict[str, dict[str, str]]) -> None:
    lines = [
        "# TSGSS Frame-Event Visualization",
        "",
        "## Quick Preview",
        f"- summary: `{summary['n_sessions']} sessions`, `{summary['n_intervals']} intervals`, `{summary['n_bins']} voxel bins`",
        f"- quick preview root: `{quick_preview_root}`",
    ]
    for label in [
        "interpolated_dense",
        "subsampled",
        "empty_pad",
        "interp_single",
    ]:
        example = mode_examples.get(label)
        if example is None:
            continue
        lines.append(f"- {label}: `{example['session_key']}`")
    lines.extend(
        [
            "",
            "## Representative PNGs",
            "- `tsgss/frame_event_visualization_quick_preview/01_interpolated_dense_triptych.png`",
            "- `tsgss/frame_event_visualization_quick_preview/02_subsampled_triptych.png`",
            "- `tsgss/frame_event_visualization_quick_preview/03_empty_pad_triptych.png`",
            "- `tsgss/frame_event_visualization_quick_preview/04_first_session_contact_sheet.png`",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    alignment_root = _resolve_path(args.alignment_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    quick_preview_root = output_root.parent / "frame_event_visualization_quick_preview"
    quick_preview_root.mkdir(parents=True, exist_ok=True)

    session_summaries = _read_session_summaries(alignment_root)
    if args.limit_sessions is not None:
        session_summaries = session_summaries[: max(0, int(args.limit_sessions))]

    n_intervals_total = 0
    mode_counter: Counter[str] = Counter()
    first_contact_sheet: Path | None = None
    mode_examples: dict[str, dict[str, str]] = {}

    for session in session_summaries:
        user_id = int(session["user_id"])
        eye = str(session["eye"])
        session_code = str(session["session_code"])
        session_key = str(session["session_key"])
        frame_event_manifest_path = Path(session["frame_event_manifest_path"]).resolve()
        aligned_events_path = Path(session["aligned_events_path"]).resolve()
        interval_rows = _read_jsonl_rows(frame_event_manifest_path)
        aligned = np.load(aligned_events_path)

        dirs = _session_output_dirs(output_root, user_id=user_id, eye=eye, session_code=session_code)
        for dst in dirs.values():
            dst.mkdir(parents=True, exist_ok=True)

        frame_cache: dict[str, Image.Image] = {}
        triptychs: list[Image.Image] = []
        representative_triptych_path: Path | None = None

        for row in interval_rows:
            interval_index = int(row["aligned_interval_index"])
            start_path = Path(str(row["start_frame_path"])).resolve()
            end_path = Path(str(row["end_frame_path"])).resolve()
            if str(start_path) not in frame_cache:
                frame_cache[str(start_path)] = _load_frame_rgb(start_path)
            if str(end_path) not in frame_cache:
                frame_cache[str(end_path)] = _load_frame_rgb(end_path)

            voxel = _build_temporal_voxel(
                xs=np.asarray(aligned["x"][interval_index]),
                ys=np.asarray(aligned["y"][interval_index]),
                ps=np.asarray(aligned["p"][interval_index]),
                ts=np.asarray(aligned["t"][interval_index]),
                valid_mask=np.asarray(aligned["valid_mask"][interval_index]),
                start_timestamp_us=int(row["start_timestamp_us"]),
                end_timestamp_us=int(row["end_timestamp_us"]),
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
                n_bins=int(args.n_bins),
            )
            voxel_strip = _render_voxel_strip(voxel=voxel)
            triptych = _render_triptych(
                row=row,
                voxel_strip=voxel_strip,
                start_frame=frame_cache[str(start_path)],
                end_frame=frame_cache[str(end_path)],
            )
            triptychs.append(triptych)

            if interval_index == 0:
                interval_zero_path = dirs["assets"] / "interval_000_triptych.png"
                triptych.save(interval_zero_path)
                if representative_triptych_path is None:
                    representative_triptych_path = interval_zero_path

            mode = str(row["alignment_mode"])
            mode_counter[mode] += 1
            if mode not in mode_examples:
                preview_path = dirs["assets"] / f"{mode}_triptych.png"
                triptych.save(preview_path)
                mode_examples[mode] = {
                    "session_key": session_key,
                    "triptych_path": str(preview_path),
                }
            n_intervals_total += 1

        contact_sheet_path = dirs["base"] / "frame_event_voxel_contact_sheet.png"
        _build_contact_sheet(triptychs, save_path=contact_sheet_path)
        if first_contact_sheet is None:
            first_contact_sheet = contact_sheet_path

        session_summary = {
            "experiment": "tsgss_frame_event_visualization",
            "session_key": session_key,
            "user_id": user_id,
            "eye": eye,
            "session_code": session_code,
            "n_intervals": len(interval_rows),
            "n_bins": int(args.n_bins),
            "contact_sheet_path": str(contact_sheet_path),
            "representative_triptych_path": None if representative_triptych_path is None else str(representative_triptych_path),
            "mode_distribution": dict(Counter(str(row["alignment_mode"]) for row in interval_rows)),
        }
        write_json(session_summary, dirs["base"] / "summary.json")
        print(
            f"{session_key}: intervals={len(interval_rows)} modes={session_summary['mode_distribution']}",
            flush=True,
        )

    preview_specs = [
        ("interpolated_dense", "01_interpolated_dense_triptych.png"),
        ("subsampled", "02_subsampled_triptych.png"),
        ("empty_pad", "03_empty_pad_triptych.png"),
        ("interp_single", "05_interp_single_triptych.png"),
    ]
    for mode, filename in preview_specs:
        example = mode_examples.get(mode)
        if example is None:
            continue
        maybe_link_or_copy(
            Path(example["triptych_path"]),
            quick_preview_root / filename,
            mode=str(args.link_mode),
            overwrite=True,
        )
    if first_contact_sheet is not None:
        maybe_link_or_copy(
            first_contact_sheet,
            quick_preview_root / "04_first_session_contact_sheet.png",
            mode=str(args.link_mode),
            overwrite=True,
        )

    summary = {
        "experiment": "tsgss_frame_event_visualization",
        "alignment_root": str(alignment_root),
        "output_root": str(output_root),
        "n_sessions": len(session_summaries),
        "n_intervals": int(n_intervals_total),
        "n_bins": int(args.n_bins),
        "mode_distribution": {str(key): int(value) for key, value in sorted(mode_counter.items())},
        "quick_preview_root": str(quick_preview_root),
    }
    write_json(summary, output_root / "experiment_summary.json")
    _write_readme(output_root, quick_preview_root, summary, mode_examples)
    print(
        f"[DONE] sessions={summary['n_sessions']} intervals={summary['n_intervals']} "
        f"bins={summary['n_bins']} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
