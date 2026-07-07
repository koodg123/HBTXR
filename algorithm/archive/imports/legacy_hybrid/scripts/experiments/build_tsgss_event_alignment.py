#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import SENSOR_HEIGHT, SENSOR_WIDTH, load_events_from_txt, maybe_link_or_copy
from hbtxr.utils.io import read_json, write_json, write_jsonl


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build synchronous frame-to-frame event alignment for tsgss sequences "
            "with an exact fixed event count per interval."
        ),
    )
    parser.add_argument(
        "--batch-root",
        type=str,
        default="tsgss/frame_interpolation",
        help="Input root produced by build_tsgss_frame_interpolation.py",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="tsgss/event_alignment",
        help="Output root for aligned frame-event intervals",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        default=1000,
        help="Exact aligned event count per adjacent frame interval",
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
        help="How to materialize quick-preview assets",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _relative_to(root: Path, path: Path) -> str:
    root_abs = root.resolve()
    path_abs = path.resolve()
    try:
        return str(path_abs.relative_to(root_abs))
    except ValueError:
        return str(path_abs)


def _read_session_summaries(batch_root: Path) -> list[dict[str, Any]]:
    summaries = [read_json(path) for path in sorted((batch_root / "sessions").rglob("summary.json"))]
    return summaries


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(__import__("json").loads(text))
    return rows


def _session_output_dirs(output_root: Path, *, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
    base = output_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}"
    return {
        "base": base,
        "assets": base / "assets",
        "manifests": base / "manifests",
    }


def _sequence_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["sequence_index"]), int(row["timestamp_us"])


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


def _uniform_interval_timestamps(start_timestamp_us: int, end_timestamp_us: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    low = int(start_timestamp_us) + 1
    high = int(end_timestamp_us)
    if high < low:
        low = int(start_timestamp_us)
        high = int(end_timestamp_us)
    if high < low:
        high = low
    if count == 1:
        return np.asarray([high], dtype=np.int64)
    values = np.rint(np.linspace(low, high, num=int(count), endpoint=True)).astype(np.int64)
    return np.maximum.accumulate(values)


def _subsample_indices(raw_count: int, target_count: int) -> np.ndarray:
    positions = (np.arange(int(target_count), dtype=np.float64) + 0.5) * float(raw_count) / float(target_count)
    indices = np.floor(positions).astype(np.int64)
    return np.clip(indices, 0, max(0, int(raw_count) - 1))


def _clip_xy(xs: np.ndarray, ys: np.ndarray, *, sensor_width: int, sensor_height: int) -> tuple[np.ndarray, np.ndarray]:
    clipped_x = np.clip(np.rint(xs).astype(np.int64), 0, max(0, int(sensor_width) - 1)).astype(np.int16)
    clipped_y = np.clip(np.rint(ys).astype(np.int64), 0, max(0, int(sensor_height) - 1)).astype(np.int16)
    return clipped_x, clipped_y


def _align_interval_events(
    *,
    raw_events: dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
    start_timestamp_us: int,
    end_timestamp_us: int,
    target_count: int,
    sensor_width: int,
    sensor_height: int,
) -> dict[str, Any]:
    raw_t = np.asarray(raw_events["t"][start_idx:end_idx], dtype=np.int64)
    raw_x = np.asarray(raw_events["x"][start_idx:end_idx], dtype=np.int16)
    raw_y = np.asarray(raw_events["y"][start_idx:end_idx], dtype=np.int16)
    raw_p = np.asarray(raw_events["p"][start_idx:end_idx], dtype=np.int8)
    raw_count = int(len(raw_t))

    if raw_count <= 0:
        aligned_t = _uniform_interval_timestamps(start_timestamp_us, end_timestamp_us, target_count)
        aligned_x = np.zeros((target_count,), dtype=np.int16)
        aligned_y = np.zeros((target_count,), dtype=np.int16)
        aligned_p = np.zeros((target_count,), dtype=np.int8)
        valid_mask = np.zeros((target_count,), dtype=bool)
        alignment_mode = "empty_pad"
        interpolated_count = int(target_count)
    elif raw_count == target_count:
        aligned_t = raw_t.copy()
        aligned_x = raw_x.copy()
        aligned_y = raw_y.copy()
        aligned_p = raw_p.copy()
        valid_mask = np.ones((target_count,), dtype=bool)
        alignment_mode = "identity"
        interpolated_count = 0
    elif raw_count > target_count:
        indices = _subsample_indices(raw_count, target_count)
        aligned_t = raw_t[indices].astype(np.int64, copy=False)
        aligned_x = raw_x[indices].astype(np.int16, copy=False)
        aligned_y = raw_y[indices].astype(np.int16, copy=False)
        aligned_p = raw_p[indices].astype(np.int8, copy=False)
        valid_mask = np.ones((target_count,), dtype=bool)
        alignment_mode = "subsampled"
        interpolated_count = 0
    elif raw_count == 1:
        aligned_t = _uniform_interval_timestamps(start_timestamp_us, end_timestamp_us, target_count)
        aligned_x = np.full((target_count,), int(raw_x[0]), dtype=np.int16)
        aligned_y = np.full((target_count,), int(raw_y[0]), dtype=np.int16)
        aligned_p = np.full((target_count,), int(raw_p[0]), dtype=np.int8)
        valid_mask = np.ones((target_count,), dtype=bool)
        alignment_mode = "interp_single"
        interpolated_count = int(target_count - raw_count)
    else:
        positions = np.linspace(0.0, float(raw_count - 1), num=target_count, endpoint=True)
        lo = np.floor(positions).astype(np.int64)
        hi = np.ceil(positions).astype(np.int64)
        alpha = positions - lo.astype(np.float64)
        aligned_t = np.rint((1.0 - alpha) * raw_t[lo] + alpha * raw_t[hi]).astype(np.int64)
        aligned_t = np.clip(
            aligned_t,
            _uniform_interval_timestamps(start_timestamp_us, end_timestamp_us, target_count)[0],
            max(int(end_timestamp_us), _uniform_interval_timestamps(start_timestamp_us, end_timestamp_us, target_count)[-1]),
        )
        aligned_t = np.maximum.accumulate(aligned_t)
        aligned_x, aligned_y = _clip_xy(
            (1.0 - alpha) * raw_x[lo].astype(np.float64) + alpha * raw_x[hi].astype(np.float64),
            (1.0 - alpha) * raw_y[lo].astype(np.float64) + alpha * raw_y[hi].astype(np.float64),
            sensor_width=sensor_width,
            sensor_height=sensor_height,
        )
        nearest = np.where(alpha <= 0.5, lo, hi)
        aligned_p = raw_p[nearest].astype(np.int8, copy=False)
        valid_mask = np.ones((target_count,), dtype=bool)
        alignment_mode = "interpolated_dense"
        interpolated_count = int(target_count - raw_count)

    if raw_count > 0:
        aligned_x, aligned_y = _clip_xy(
            aligned_x.astype(np.float64),
            aligned_y.astype(np.float64),
            sensor_width=sensor_width,
            sensor_height=sensor_height,
        )
        aligned_t = np.maximum.accumulate(np.asarray(aligned_t, dtype=np.int64))

    return {
        "t": np.asarray(aligned_t, dtype=np.int64),
        "x": np.asarray(aligned_x, dtype=np.int16),
        "y": np.asarray(aligned_y, dtype=np.int16),
        "p": np.asarray(aligned_p, dtype=np.int8),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "raw_count": raw_count,
        "interpolated_count": interpolated_count,
        "alignment_mode": alignment_mode,
    }


def _event_map_from_aligned(
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sensor_width: int,
    sensor_height: int,
) -> np.ndarray:
    pos = np.zeros((int(sensor_height), int(sensor_width)), dtype=np.float32)
    neg = np.zeros((int(sensor_height), int(sensor_width)), dtype=np.float32)
    if np.any(valid_mask):
        valid_x = np.asarray(xs[valid_mask], dtype=np.int64)
        valid_y = np.asarray(ys[valid_mask], dtype=np.int64)
        valid_p = np.asarray(ps[valid_mask], dtype=np.int64)
        negative = valid_p <= 0
        positive = ~negative
        np.add.at(neg, (valid_y[negative], valid_x[negative]), 1.0)
        np.add.at(pos, (valid_y[positive], valid_x[positive]), 1.0)
    pos = np.log1p(pos)
    neg = np.log1p(neg)
    denom = float(max(float(pos.max()), float(neg.max()), 1.0))
    rgb = np.zeros((int(sensor_height), int(sensor_width), 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((pos / denom) * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb[..., 1] = np.clip((np.minimum(pos, neg) / denom) * 80.0, 0.0, 255.0).astype(np.uint8)
    rgb[..., 2] = np.clip((neg / denom) * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb


def _interval_badge_text(row: dict[str, Any]) -> str:
    return (
        f"{int(row['interval_index']):02d} "
        f"{int(row['raw_event_count'])}->{int(row['target_event_count'])} "
        f"{str(row['alignment_mode'])}"
    )


def _build_contact_sheet(
    interval_rows: list[dict[str, Any]],
    interval_maps: list[np.ndarray],
    *,
    save_root: Path,
) -> Path:
    thumb_w = 220
    thumb_h = 150
    pad = 16
    cols = min(4, max(1, len(interval_maps)))
    rows = int(math.ceil(len(interval_maps) / cols))
    canvas = Image.new(
        "RGB",
        (pad + cols * (thumb_w + pad), pad + rows * (thumb_h + 40 + pad)),
        color=(247, 244, 239),
    )
    draw = ImageDraw.Draw(canvas)

    for idx, (row, image) in enumerate(zip(interval_rows, interval_maps, strict=False)):
        x = pad + (idx % cols) * (thumb_w + pad)
        y = pad + (idx // cols) * (thumb_h + 40 + pad)
        tile = ImageOps.pad(Image.fromarray(image, mode="RGB"), (thumb_w, thumb_h), color=(228, 225, 219))
        bordered = ImageOps.expand(tile, border=4, fill=(60, 72, 96))
        canvas.paste(bordered, (x, y))
        draw.text((x, y + thumb_h + 10), _interval_badge_text(row), fill=(34, 31, 28))

    dst = save_root / "event_alignment_contact_sheet.png"
    canvas.save(dst)
    return dst


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TSGSS Event Alignment Summary",
        "",
        "## Aggregate",
        f"- sessions: {summary['n_sessions']}",
        f"- intervals: {summary['n_intervals']}",
        f"- target_event_count: {summary['target_event_count']}",
        f"- min_raw_event_count: {summary['min_raw_event_count']}",
        f"- max_raw_event_count: {summary['max_raw_event_count']}",
        f"- mean_raw_event_count: {summary['mean_raw_event_count']}",
        f"- zero_raw_intervals: {summary['n_zero_raw_intervals']}",
        "",
        "## Alignment Modes",
    ]
    for key, value in sorted((summary.get("alignment_mode_distribution") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Frame Gap (us)",
            f"- min: {summary['min_interval_us']}",
            f"- max: {summary['max_interval_us']}",
            f"- mean: {summary['mean_interval_us']}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_argparser().parse_args()
    batch_root = _resolve_path(args.batch_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    session_summaries = _read_session_summaries(batch_root)
    if args.limit_sessions is not None:
        session_summaries = session_summaries[: max(0, int(args.limit_sessions))]

    aggregate_rows: list[dict[str, Any]] = []
    session_briefs: list[dict[str, Any]] = []
    quick_preview_root = output_root.parent / "event_alignment_quick_preview"
    quick_preview_root.mkdir(parents=True, exist_ok=True)
    first_contact_sheet: Path | None = None
    first_interval_preview: Path | None = None

    alignment_counter: Counter[str] = Counter()
    raw_event_counter: list[int] = []
    interval_us_counter: list[int] = []
    zero_raw_intervals = 0

    for session in session_summaries:
        user_id = int(session["user_id"])
        eye = str(session["eye"])
        session_code = str(session["session_code"])
        session_key = str(session["session_key"])
        event_file = Path(session["event_file"]).resolve()
        sequence_manifest_path = Path(session["sequence_manifest_path"]).resolve()
        sequence_rows = sorted(_read_jsonl_rows(sequence_manifest_path), key=_sequence_sort_key)
        raw_events = load_events_from_txt(event_file, eye=eye)
        timestamps = np.asarray(raw_events["t"], dtype=np.int64)

        dirs = _session_output_dirs(output_root, user_id=user_id, eye=eye, session_code=session_code)
        for dst in dirs.values():
            dst.mkdir(parents=True, exist_ok=True)

        n_intervals = max(0, len(sequence_rows) - 1)
        aligned_t = np.zeros((n_intervals, int(args.event_count)), dtype=np.int64)
        aligned_x = np.zeros((n_intervals, int(args.event_count)), dtype=np.int16)
        aligned_y = np.zeros((n_intervals, int(args.event_count)), dtype=np.int16)
        aligned_p = np.zeros((n_intervals, int(args.event_count)), dtype=np.int8)
        aligned_valid = np.zeros((n_intervals, int(args.event_count)), dtype=bool)
        raw_counts = np.zeros((n_intervals,), dtype=np.int32)
        interval_starts = np.zeros((n_intervals,), dtype=np.int64)
        interval_ends = np.zeros((n_intervals,), dtype=np.int64)

        interval_rows: list[dict[str, Any]] = []
        interval_maps: list[np.ndarray] = []

        for interval_index, (row_start, row_end) in enumerate(zip(sequence_rows[:-1], sequence_rows[1:], strict=False)):
            start_timestamp_us = int(row_start["timestamp_us"])
            end_timestamp_us = int(row_end["timestamp_us"])
            start_idx, end_idx = _interval_indices(
                timestamps,
                start_timestamp_us=start_timestamp_us,
                end_timestamp_us=end_timestamp_us,
            )
            aligned = _align_interval_events(
                raw_events=raw_events,
                start_idx=start_idx,
                end_idx=end_idx,
                start_timestamp_us=start_timestamp_us,
                end_timestamp_us=end_timestamp_us,
                target_count=int(args.event_count),
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
            )

            aligned_t[interval_index] = aligned["t"]
            aligned_x[interval_index] = aligned["x"]
            aligned_y[interval_index] = aligned["y"]
            aligned_p[interval_index] = aligned["p"]
            aligned_valid[interval_index] = aligned["valid_mask"]
            raw_counts[interval_index] = int(aligned["raw_count"])
            interval_starts[interval_index] = start_timestamp_us
            interval_ends[interval_index] = end_timestamp_us

            raw_count = int(aligned["raw_count"])
            zero_raw_intervals += int(raw_count == 0)
            alignment_mode = str(aligned["alignment_mode"])
            alignment_counter[alignment_mode] += 1
            raw_event_counter.append(raw_count)
            interval_us_counter.append(int(end_timestamp_us - start_timestamp_us))

            start_frame_path = _resolve_sequence_frame_path(batch_root, row_start)
            end_frame_path = _resolve_sequence_frame_path(batch_root, row_end)
            interval_row = {
                "session_key": session_key,
                "user_id": user_id,
                "eye": eye,
                "session_code": session_code,
                "interval_index": int(interval_index),
                "frame_to_frame_sync": True,
                "target_event_count": int(args.event_count),
                "alignment_mode": alignment_mode,
                "raw_event_count": raw_count,
                "interpolated_event_count": int(aligned["interpolated_count"]),
                "valid_event_count": int(np.count_nonzero(aligned["valid_mask"])),
                "raw_event_index_range": [int(start_idx), int(end_idx)],
                "aligned_event_index_range": [0, int(args.event_count)],
                "start_sequence_index": int(row_start["sequence_index"]),
                "end_sequence_index": int(row_end["sequence_index"]),
                "start_frame_kind": str(row_start["frame_kind"]),
                "end_frame_kind": str(row_end["frame_kind"]),
                "start_timestamp_us": start_timestamp_us,
                "end_timestamp_us": end_timestamp_us,
                "interval_us": int(end_timestamp_us - start_timestamp_us),
                "start_frame_path": str(start_frame_path),
                "end_frame_path": str(end_frame_path),
                "aligned_events_path": str(dirs["manifests"] / "aligned_events.npz"),
                "aligned_interval_index": int(interval_index),
            }
            interval_rows.append(interval_row)
            aggregate_rows.append(interval_row)

            event_map = _event_map_from_aligned(
                aligned["x"],
                aligned["y"],
                aligned["p"],
                aligned["valid_mask"],
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
            )
            interval_maps.append(event_map)
            if interval_index == 0:
                preview_path = dirs["assets"] / "interval_000_event_map.png"
                Image.fromarray(event_map, mode="RGB").save(preview_path)
                if first_interval_preview is None:
                    first_interval_preview = preview_path

        aligned_events_path = dirs["manifests"] / "aligned_events.npz"
        np.savez_compressed(
            aligned_events_path,
            t=aligned_t,
            x=aligned_x,
            y=aligned_y,
            p=aligned_p,
            valid_mask=aligned_valid,
            raw_event_count=raw_counts,
            interval_start_timestamp_us=interval_starts,
            interval_end_timestamp_us=interval_ends,
        )
        contact_sheet_path = _build_contact_sheet(interval_rows, interval_maps, save_root=dirs["base"])
        if first_contact_sheet is None:
            first_contact_sheet = contact_sheet_path

        session_summary = {
            "experiment": "tsgss_event_alignment",
            "session_key": session_key,
            "user_id": user_id,
            "eye": eye,
            "session_code": session_code,
            "target_event_count": int(args.event_count),
            "n_sequence_frames": len(sequence_rows),
            "n_intervals": n_intervals,
            "event_file": str(event_file),
            "sequence_manifest_path": str(sequence_manifest_path),
            "aligned_events_path": str(aligned_events_path),
            "frame_event_manifest_path": str(dirs["manifests"] / "frame_event_manifest.jsonl"),
            "contact_sheet_path": str(contact_sheet_path),
            "raw_event_count_min": None if n_intervals <= 0 else int(raw_counts.min()),
            "raw_event_count_max": None if n_intervals <= 0 else int(raw_counts.max()),
            "raw_event_count_mean": None if n_intervals <= 0 else float(np.mean(raw_counts)),
            "alignment_mode_distribution": dict(Counter(str(row["alignment_mode"]) for row in interval_rows)),
        }
        write_jsonl(interval_rows, dirs["manifests"] / "frame_event_manifest.jsonl")
        write_json(session_summary, dirs["base"] / "summary.json")
        session_briefs.append(
            {
                "session_key": session_key,
                "n_intervals": n_intervals,
                "raw_event_count_min": session_summary["raw_event_count_min"],
                "raw_event_count_max": session_summary["raw_event_count_max"],
                "contact_sheet_path": str(contact_sheet_path),
            }
        )
        print(
            f"{session_key}: intervals={n_intervals} raw_min={session_summary['raw_event_count_min']} "
            f"raw_max={session_summary['raw_event_count_max']} modes={session_summary['alignment_mode_distribution']}",
            flush=True,
        )

    if first_contact_sheet is not None:
        maybe_link_or_copy(
            first_contact_sheet,
            quick_preview_root / "01_event_alignment_contact_sheet.png",
            mode=str(args.link_mode),
            overwrite=True,
        )
    if first_interval_preview is not None:
        maybe_link_or_copy(
            first_interval_preview,
            quick_preview_root / "02_first_interval_event_map.png",
            mode=str(args.link_mode),
            overwrite=True,
        )

    summary = {
        "experiment": "tsgss_event_alignment",
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "target_event_count": int(args.event_count),
        "sensor_width": int(args.sensor_width),
        "sensor_height": int(args.sensor_height),
        "n_sessions": len(session_summaries),
        "n_intervals": len(aggregate_rows),
        "n_zero_raw_intervals": int(zero_raw_intervals),
        "min_raw_event_count": None if not raw_event_counter else int(min(raw_event_counter)),
        "max_raw_event_count": None if not raw_event_counter else int(max(raw_event_counter)),
        "mean_raw_event_count": None if not raw_event_counter else float(round(float(np.mean(raw_event_counter)), 4)),
        "min_interval_us": None if not interval_us_counter else int(min(interval_us_counter)),
        "max_interval_us": None if not interval_us_counter else int(max(interval_us_counter)),
        "mean_interval_us": None if not interval_us_counter else float(round(float(np.mean(interval_us_counter)), 4)),
        "alignment_mode_distribution": {str(key): int(value) for key, value in sorted(alignment_counter.items())},
        "aggregate_frame_event_manifest_path": str(output_root / "aggregate_frame_event_manifest.jsonl"),
        "session_briefs_path": str(output_root / "session_briefs.jsonl"),
        "quick_preview_root": str(quick_preview_root),
    }
    write_jsonl(aggregate_rows, output_root / "aggregate_frame_event_manifest.jsonl")
    write_jsonl(session_briefs, output_root / "session_briefs.jsonl")
    write_json(summary, output_root / "experiment_summary.json")
    (output_root / "experiment_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(
        f"[DONE] sessions={summary['n_sessions']} intervals={summary['n_intervals']} "
        f"target={summary['target_event_count']} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
