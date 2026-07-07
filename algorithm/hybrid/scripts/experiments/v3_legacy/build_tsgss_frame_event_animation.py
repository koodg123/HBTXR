#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from build_tsgss_frame_event_visualization import (
    _build_temporal_voxel,
    _load_frame_rgb,
    _read_jsonl_rows,
    _read_session_summaries,
    _render_triptych,
    _render_voxel_strip,
    _resolve_path,
)
from build_tsgss_event_xyz_preview import _compose_triptych as _compose_xyz_triptych
from build_tsgss_event_xyz_preview import _render_event_xyz_plot
from build_tsgss_event_plane_preview import _compose_triptych as _compose_plane_triptych
from build_tsgss_event_plane_preview import _render_event_plane
from hbtxr.preprocess.io_utils import SENSOR_HEIGHT, SENSOR_WIDTH, maybe_link_or_copy
from hbtxr.utils.io import write_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export animated frame-event visualizations from tsgss event-alignment outputs."
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
        default="tsgss/frame_event_animation",
        help="Output root for animated frame-event previews",
    )
    parser.add_argument(
        "--event-view",
        type=str,
        default="voxel_strip",
        choices=["voxel_strip", "xyz_plot", "event_plane"],
        help="Middle-panel event rendering mode for the animation",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Temporal bins for event voxel visualization",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=180,
        help="Per-interval GIF frame duration in milliseconds",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=700,
        help="Final-frame hold duration in milliseconds",
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
        "--session-key",
        type=str,
        default=None,
        help="Optional single session key filter such as user01/left/session_102",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy", "skip"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _session_output_dirs(output_root: Path, *, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
    base = output_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}"
    return {
        "base": base,
        "assets": base / "assets",
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


def _export_mp4_if_available(frame_dir: Path, *, mp4_path: Path, duration_ms: int) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False
    fps = 1000.0 / max(1, float(duration_ms))
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frame_dir / "frame_%03d.png"),
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def _write_readme(output_root: Path, quick_preview_root: Path, summary: dict[str, Any], mode_examples: dict[str, dict[str, str]]) -> None:
    lines = [
        "# TSGSS Frame-Event Animation",
        "",
        "## Summary",
        f"- sessions: `{summary['n_sessions']}`",
        f"- intervals: `{summary['n_intervals']}`",
        f"- event view: `{summary['event_view']}`",
        f"- voxel bins: `{summary['n_bins']}`",
        f"- GIF duration per interval: `{summary['duration_ms']} ms`",
        f"- MP4 enabled: `{summary['mp4_enabled']}`",
        f"- quick preview root: `{quick_preview_root}`",
        "",
        "## Representative GIFs",
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
    (output_root / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    alignment_root = _resolve_path(args.alignment_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    quick_preview_root = output_root.parent / f"{output_root.name}_quick_preview"
    quick_preview_root.mkdir(parents=True, exist_ok=True)

    session_summaries = _read_session_summaries(alignment_root)
    if args.session_key is not None:
        session_summaries = [
            session for session in session_summaries
            if str(session.get("session_key")) == str(args.session_key)
        ]
    if args.limit_sessions is not None:
        session_summaries = session_summaries[: max(0, int(args.limit_sessions))]

    mode_counter: Counter[str] = Counter()
    mode_examples: dict[str, dict[str, str]] = {}
    n_intervals_total = 0
    mp4_enabled = shutil.which("ffmpeg") is not None
    first_gif: Path | None = None
    first_poster: Path | None = None

    for session in session_summaries:
        user_id = int(session["user_id"])
        eye = str(session["eye"])
        session_code = str(session["session_code"])
        session_key = str(session["session_key"])
        frame_event_manifest_path = Path(session["frame_event_manifest_path"]).resolve()
        aligned_events_path = Path(session["aligned_events_path"]).resolve()
        interval_rows = _read_jsonl_rows(frame_event_manifest_path)

        dirs = _session_output_dirs(output_root, user_id=user_id, eye=eye, session_code=session_code)
        for dst in dirs.values():
            dst.mkdir(parents=True, exist_ok=True)

        aligned = __import__("numpy").load(aligned_events_path)
        frame_cache: dict[str, Image.Image] = {}
        animation_frames: list[Image.Image] = []

        for row in interval_rows:
            interval_index = int(row["aligned_interval_index"])
            start_path = Path(str(row["start_frame_path"])).resolve()
            end_path = Path(str(row["end_frame_path"])).resolve()
            if str(start_path) not in frame_cache:
                frame_cache[str(start_path)] = _load_frame_rgb(start_path)
            if str(end_path) not in frame_cache:
                frame_cache[str(end_path)] = _load_frame_rgb(end_path)

            if str(args.event_view) == "xyz_plot":
                event_plot = _render_event_xyz_plot(
                    row=row,
                    arrays={
                        "x": aligned["x"][interval_index],
                        "y": aligned["y"][interval_index],
                        "p": aligned["p"][interval_index],
                        "t": aligned["t"][interval_index],
                        "valid_mask": aligned["valid_mask"][interval_index],
                    },
                    sensor_width=int(args.sensor_width),
                    sensor_height=int(args.sensor_height),
                    point_size=7.0,
                )
                triptych = _compose_xyz_triptych(
                    row=row,
                    start_frame=frame_cache[str(start_path)],
                    event_plot=event_plot,
                    end_frame=frame_cache[str(end_path)],
                )
            elif str(args.event_view) == "event_plane":
                event_plane = _render_event_plane(
                    arrays={
                        "x": aligned["x"][interval_index],
                        "y": aligned["y"][interval_index],
                        "p": aligned["p"][interval_index],
                        "valid_mask": aligned["valid_mask"][interval_index],
                    },
                    width=int(args.sensor_width),
                    height=int(args.sensor_height),
                    background="white",
                    positive_color="green",
                    negative_color="blue",
                    blur_radius=1.1,
                    target_size=320,
                )
                triptych = _compose_plane_triptych(
                    row=row,
                    start_frame=frame_cache[str(start_path)],
                    event_plane=event_plane,
                    end_frame=frame_cache[str(end_path)],
                )
            else:
                voxel = _build_temporal_voxel(
                    xs=aligned["x"][interval_index],
                    ys=aligned["y"][interval_index],
                    ps=aligned["p"][interval_index],
                    ts=aligned["t"][interval_index],
                    valid_mask=aligned["valid_mask"][interval_index],
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
            animation_frames.append(triptych)

            frame_png = dirs["frames"] / f"frame_{interval_index:03d}.png"
            triptych.save(frame_png)

            mode = str(row["alignment_mode"])
            mode_counter[mode] += 1
            if mode not in mode_examples:
                mode_examples[mode] = {
                    "session_key": session_key,
                    "gif_path": str(dirs["base"] / "session_animation.gif"),
                }
            n_intervals_total += 1

        gif_path = dirs["base"] / "session_animation.gif"
        _save_gif(animation_frames, dst=gif_path, duration_ms=int(args.duration_ms), pause_ms=int(args.pause_ms))

        poster_path = dirs["base"] / "animation_poster.png"
        animation_frames[0].save(poster_path)
        if first_gif is None:
            first_gif = gif_path
        if first_poster is None:
            first_poster = poster_path

        mp4_path = dirs["base"] / "session_animation.mp4"
        mp4_created = _export_mp4_if_available(
            dirs["frames"],
            mp4_path=mp4_path,
            duration_ms=int(args.duration_ms),
        )
        session_summary = {
            "experiment": "tsgss_frame_event_animation",
            "session_key": session_key,
            "user_id": user_id,
            "eye": eye,
            "session_code": session_code,
            "n_intervals": len(interval_rows),
            "event_view": str(args.event_view),
            "n_bins": int(args.n_bins),
            "duration_ms": int(args.duration_ms),
            "pause_ms": int(args.pause_ms),
            "gif_path": str(gif_path),
            "poster_path": str(poster_path),
            "mp4_path": str(mp4_path) if mp4_created else None,
            "mode_distribution": dict(Counter(str(row["alignment_mode"]) for row in interval_rows)),
        }
        write_json(session_summary, dirs["base"] / "summary.json")
        print(
            f"{session_key}: intervals={len(interval_rows)} gif={gif_path.name} mp4={bool(mp4_created)} "
            f"modes={session_summary['mode_distribution']}",
            flush=True,
        )

    preview_specs = [
        ("interpolated_dense", "01_interpolated_dense_animation.gif"),
        ("subsampled", "02_subsampled_animation.gif"),
        ("empty_pad", "03_empty_pad_animation.gif"),
        ("interp_single", "04_interp_single_animation.gif"),
    ]
    for mode, filename in preview_specs:
        example = mode_examples.get(mode)
        if example is None:
            continue
        maybe_link_or_copy(
            Path(example["gif_path"]),
            quick_preview_root / filename,
            mode=str(args.link_mode),
            overwrite=True,
        )
    if first_gif is not None:
        maybe_link_or_copy(
            first_gif,
            quick_preview_root / "05_first_session_animation.gif",
            mode=str(args.link_mode),
            overwrite=True,
        )
    if first_poster is not None:
        maybe_link_or_copy(
            first_poster,
            quick_preview_root / "06_first_session_poster.png",
            mode=str(args.link_mode),
            overwrite=True,
        )

    summary = {
        "experiment": "tsgss_frame_event_animation",
        "alignment_root": str(alignment_root),
        "output_root": str(output_root),
        "event_view": str(args.event_view),
        "session_key_filter": None if args.session_key is None else str(args.session_key),
        "n_sessions": len(session_summaries),
        "n_intervals": int(n_intervals_total),
        "n_bins": int(args.n_bins),
        "duration_ms": int(args.duration_ms),
        "pause_ms": int(args.pause_ms),
        "mp4_enabled": bool(mp4_enabled),
        "mode_distribution": {str(key): int(value) for key, value in sorted(mode_counter.items())},
        "quick_preview_root": str(quick_preview_root),
    }
    write_json(summary, output_root / "experiment_summary.json")
    _write_readme(output_root, quick_preview_root, summary, mode_examples)
    print(
        f"[DONE] sessions={summary['n_sessions']} intervals={summary['n_intervals']} "
        f"gif_output={output_root} mp4_enabled={summary['mp4_enabled']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
