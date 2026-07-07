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

from hbtxr.preprocess.interpolation import (
    DEFAULT_INTERPOLATION_COUNT_POLICY,
    build_alpha_schedule,
    compute_insert_count,
    interpolate_pair,
    synth_timestamp_us,
)
from hbtxr.preprocess.io_utils import discover_session_layout, maybe_link_or_copy
from hbtxr.utils.io import read_json, write_json, write_jsonl


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build frame-interpolated tsgss sequences from sampled raw-frame session summaries."
        ),
    )
    parser.add_argument(
        "--batch-root",
        type=str,
        default="tsgss/workspace/roi_crop_prompted_all_sessions",
        help="Workspace root produced by run_roi_crop_prompted_preview_batch.py",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="tsgss/frame_interpolation",
        help="Output root for interpolated-frame sequences",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="linear_blend",
        choices=["linear_blend", "timelens"],
        help="Interpolation backend",
    )
    parser.add_argument(
        "--fixed-insert",
        type=int,
        default=1,
        help="Number of synthetic frames inserted between adjacent sampled raw frames",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Optional target FPS used to derive the insert count from timestamp gaps",
    )
    parser.add_argument(
        "--count-policy",
        type=str,
        default=DEFAULT_INTERPOLATION_COUNT_POLICY,
        choices=["round", "floor", "ceil"],
    )
    parser.add_argument("--max-insert", type=int, default=None)
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--timelens-root", type=str, default=None)
    parser.add_argument("--timelens-checkpoint", type=str, default=None)
    parser.add_argument("--timelens-device", type=str, default="cpu")
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
    return [read_json(path) for path in sorted((batch_root / "sessions").rglob("summary.json"))]


def _frame_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    frame_idx = row.get("frame_idx")
    timestamp_us = row.get("timestamp_us")
    return (
        -1 if frame_idx is None else int(frame_idx),
        -1 if timestamp_us is None else int(timestamp_us),
    )


def _session_output_dirs(output_root: Path, *, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
    base = output_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}"
    return {
        "base": base,
        "sequence_frames": base / "sequence_frames",
        "interpolated_frames": base / "interpolated_frames",
        "manifests": base / "manifests",
    }


def _load_frame(path_text: str) -> np.ndarray:
    path = Path(path_text).resolve()
    return np.asarray(Image.open(path))


def _save_frame(image: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(dst)


def _sequence_badge_text(row: dict[str, Any]) -> str:
    kind = str(row["frame_kind"])
    seq_idx = int(row["sequence_index"])
    if kind == "raw":
        return f"{seq_idx:02d} RAW"
    return f"{seq_idx:02d} SYN a={float(row['interp_alpha']):.2f}"


def _border_color(row: dict[str, Any]) -> tuple[int, int, int]:
    return (38, 135, 84) if str(row["frame_kind"]) == "raw" else (194, 110, 32)


def _build_contact_sheet(sequence_rows: list[dict[str, Any]], *, asset_root: Path, save_root: Path) -> Path:
    thumb_w = 220
    thumb_h = 150
    pad = 16
    cols = min(5, max(1, len(sequence_rows)))
    rows = int(math.ceil(len(sequence_rows) / cols))
    canvas = Image.new("RGB", (pad + cols * (thumb_w + pad), pad + rows * (thumb_h + 40 + pad)), color=(250, 246, 238))
    draw = ImageDraw.Draw(canvas)

    for idx, row in enumerate(sequence_rows):
        x = pad + (idx % cols) * (thumb_w + pad)
        y = pad + (idx // cols) * (thumb_h + 40 + pad)
        img = Image.open(asset_root / row["sequence_frame_path"]).convert("L")
        tile = ImageOps.pad(img.convert("RGB"), (thumb_w, thumb_h), color=(232, 229, 220))
        border = _border_color(row)
        bordered = ImageOps.expand(tile, border=4, fill=border)
        canvas.paste(bordered, (x, y))
        draw.text((x, y + thumb_h + 10), _sequence_badge_text(row), fill=(36, 31, 28))

    dst = save_root / "sequence_contact_sheet.png"
    canvas.save(dst)
    return dst


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TSGSS Frame Interpolation Summary",
        "",
        "## Aggregate",
        f"- sessions: {summary['n_sessions']}",
        f"- raw_frames: {summary['n_raw_frames']}",
        f"- interpolated_frames: {summary['n_interpolated_frames']}",
        f"- total_sequence_frames: {summary['n_total_sequence_frames']}",
        f"- backend: {summary['backend']}",
        f"- fixed_insert: {summary['fixed_insert']}",
        "",
        "## Insert Count Distribution",
    ]
    for key, value in sorted((summary.get("insert_count_distribution") or {}).items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_argparser().parse_args()
    batch_root = _resolve_path(args.batch_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    session_summaries = _read_session_summaries(batch_root)
    aggregate_rows: list[dict[str, Any]] = []
    session_briefs: list[dict[str, Any]] = []
    insert_counter: Counter[int] = Counter()
    n_raw_total = 0
    n_interp_total = 0

    resolved_timelens_root = None if args.timelens_root is None else _resolve_path(args.timelens_root)
    resolved_timelens_checkpoint = None if args.timelens_checkpoint is None else _resolve_path(args.timelens_checkpoint)

    for session in session_summaries:
        user_id = int(session["user_id"])
        eye = str(session["eye"])
        session_code = str(session["session_code"])
        session_key = str(session["session_key"])
        raw_session_dir = Path(session["raw_session_dir"]).resolve()
        layout = discover_session_layout(raw_session_dir, user_id=user_id)

        dirs = _session_output_dirs(output_root, user_id=user_id, eye=eye, session_code=session_code)
        for dst in dirs.values():
            dst.mkdir(parents=True, exist_ok=True)

        frames = sorted(session.get("frames", []), key=_frame_sort_key)
        frame_cache: dict[str, np.ndarray] = {}
        sequence_rows: list[dict[str, Any]] = []
        interpolation_rows: list[dict[str, Any]] = []
        sequence_index = 0
        n_raw = 0
        n_interp = 0

        for idx, frame in enumerate(frames):
            raw_src = Path(frame.get("sample_frame_path") or frame.get("raw_frame_path")).resolve()
            raw_seq_name = f"{sequence_index:05d}_raw_{raw_src.name}"
            raw_seq_dst = dirs["sequence_frames"] / raw_seq_name
            maybe_link_or_copy(raw_src, raw_seq_dst, mode=args.link_mode, overwrite=bool(args.overwrite))

            raw_row = {
                "session_key": session_key,
                "user_id": user_id,
                "eye": eye,
                "session_code": session_code,
                "sequence_index": sequence_index,
                "frame_kind": "raw",
                "synthetic_frame_flag": False,
                "frame_filename": str(frame["frame_filename"]),
                "timestamp_us": int(frame["timestamp_us"]),
                "frame_idx": int(frame["frame_idx"]),
                "sequence_frame_path": _relative_to(output_root, raw_seq_dst),
                "raw_frame_path": str(raw_src),
                "source_pair": None,
                "interp_alpha": None,
                "interp_rank": None,
                "insert_count": 0,
                "backend": str(args.backend),
            }
            sequence_rows.append(raw_row)
            aggregate_rows.append(raw_row)
            n_raw += 1
            sequence_index += 1

            if idx >= len(frames) - 1:
                continue

            frame_next = frames[idx + 1]
            frame0_path = str(frame.get("raw_frame_path") or frame.get("sample_frame_path"))
            frame1_path = str(frame_next.get("raw_frame_path") or frame_next.get("sample_frame_path"))
            if frame0_path not in frame_cache:
                frame_cache[frame0_path] = _load_frame(frame0_path)
            if frame1_path not in frame_cache:
                frame_cache[frame1_path] = _load_frame(frame1_path)
            frame0_img = frame_cache[frame0_path]
            frame1_img = frame_cache[frame1_path]

            insert_count = compute_insert_count(
                timestamp0_us=int(frame["timestamp_us"]),
                timestamp1_us=int(frame_next["timestamp_us"]),
                target_fps=None if args.target_fps is None else float(args.target_fps),
                fixed_insert=int(args.fixed_insert) if args.target_fps is None else None,
                count_policy=str(args.count_policy),
                max_insert=None if args.max_insert is None else int(args.max_insert),
            )
            insert_counter[int(insert_count)] += 1
            alpha_schedule = build_alpha_schedule(insert_count)

            for rank, alpha in enumerate(alpha_schedule, start=1):
                synth_ts = synth_timestamp_us(int(frame["timestamp_us"]), int(frame_next["timestamp_us"]), alpha)
                interp_stem = (
                    f"{Path(frame['frame_filename']).stem}"
                    f"__a{int(round(alpha * 1000.0)):04d}__"
                    f"{Path(frame_next['frame_filename']).stem}"
                )
                interp_dst = dirs["interpolated_frames"] / f"{interp_stem}.png"
                synthetic = interpolate_pair(
                    backend=str(args.backend),
                    frame0=frame0_img,
                    frame1=frame1_img,
                    alpha=float(alpha),
                    timestamp0_us=int(frame["timestamp_us"]),
                    timestamp1_us=int(frame_next["timestamp_us"]),
                    events_path=None if layout.event_file is None else layout.event_file,
                    timelens_root=resolved_timelens_root,
                    timelens_checkpoint=resolved_timelens_checkpoint,
                    timelens_device=str(args.timelens_device),
                )
                _save_frame(synthetic, interp_dst)

                interp_seq_name = f"{sequence_index:05d}_interp_{synth_ts}.png"
                interp_seq_dst = dirs["sequence_frames"] / interp_seq_name
                maybe_link_or_copy(interp_dst, interp_seq_dst, mode=args.link_mode, overwrite=True)

                interp_row = {
                    "session_key": session_key,
                    "user_id": user_id,
                    "eye": eye,
                    "session_code": session_code,
                    "sequence_index": sequence_index,
                    "frame_kind": "interpolated",
                    "synthetic_frame_flag": True,
                    "frame_filename": interp_dst.name,
                    "timestamp_us": int(synth_ts),
                    "frame_idx": None,
                    "sequence_frame_path": _relative_to(output_root, interp_seq_dst),
                    "interpolated_frame_path": _relative_to(output_root, interp_dst),
                    "source_pair": [
                        str(frame["frame_filename"]),
                        str(frame_next["frame_filename"]),
                    ],
                    "source_pair_timestamps_us": [
                        int(frame["timestamp_us"]),
                        int(frame_next["timestamp_us"]),
                    ],
                    "interp_alpha": float(alpha),
                    "interp_rank": int(rank),
                    "insert_count": int(insert_count),
                    "backend": str(args.backend),
                }
                sequence_rows.append(interp_row)
                interpolation_rows.append(interp_row)
                aggregate_rows.append(interp_row)
                n_interp += 1
                sequence_index += 1

        contact_sheet_path = _build_contact_sheet(sequence_rows, asset_root=output_root, save_root=dirs["base"])
        session_summary = {
            "experiment": "tsgss_frame_interpolation",
            "session_key": session_key,
            "user_id": user_id,
            "eye": eye,
            "session_code": session_code,
            "backend": str(args.backend),
            "fixed_insert": int(args.fixed_insert),
            "target_fps": None if args.target_fps is None else float(args.target_fps),
            "count_policy": str(args.count_policy),
            "max_insert": None if args.max_insert is None else int(args.max_insert),
            "n_sampled_raw_frames": n_raw,
            "n_interpolated_frames": n_interp,
            "n_total_sequence_frames": n_raw + n_interp,
            "raw_session_dir": str(raw_session_dir),
            "event_file": None if layout.event_file is None else str(layout.event_file),
            "sequence_manifest_path": str(dirs["manifests"] / "sequence_manifest.jsonl"),
            "interpolation_rows_path": str(dirs["manifests"] / "interpolation_rows.jsonl"),
            "contact_sheet_path": str(contact_sheet_path),
        }
        write_jsonl(sequence_rows, dirs["manifests"] / "sequence_manifest.jsonl")
        write_jsonl(interpolation_rows, dirs["manifests"] / "interpolation_rows.jsonl")
        write_json(session_summary, dirs["base"] / "summary.json")

        n_raw_total += n_raw
        n_interp_total += n_interp
        session_briefs.append(
            {
                "session_key": session_key,
                "n_sampled_raw_frames": n_raw,
                "n_interpolated_frames": n_interp,
                "n_total_sequence_frames": n_raw + n_interp,
                "contact_sheet_path": str(contact_sheet_path),
            }
        )
        print(
            f"{session_key}: raw={n_raw} interp={n_interp} total={n_raw + n_interp}",
            flush=True,
        )

    summary = {
        "experiment": "tsgss_frame_interpolation",
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "backend": str(args.backend),
        "fixed_insert": int(args.fixed_insert),
        "target_fps": None if args.target_fps is None else float(args.target_fps),
        "count_policy": str(args.count_policy),
        "max_insert": None if args.max_insert is None else int(args.max_insert),
        "timelens_root": None if resolved_timelens_root is None else str(resolved_timelens_root),
        "timelens_checkpoint": None if resolved_timelens_checkpoint is None else str(resolved_timelens_checkpoint),
        "n_sessions": len(session_summaries),
        "n_raw_frames": n_raw_total,
        "n_interpolated_frames": n_interp_total,
        "n_total_sequence_frames": n_raw_total + n_interp_total,
        "insert_count_distribution": {str(key): int(value) for key, value in sorted(insert_counter.items())},
        "aggregate_sequence_manifest_path": str(output_root / "aggregate_sequence_manifest.jsonl"),
        "session_briefs_path": str(output_root / "session_briefs.jsonl"),
    }
    write_jsonl(aggregate_rows, output_root / "aggregate_sequence_manifest.jsonl")
    write_jsonl(session_briefs, output_root / "session_briefs.jsonl")
    write_json(summary, output_root / "experiment_summary.json")
    (output_root / "experiment_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(
        f"[DONE] sessions={summary['n_sessions']} raw={summary['n_raw_frames']} "
        f"interp={summary['n_interpolated_frames']} total={summary['n_total_sequence_frames']} "
        f"output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
