from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _select_indices(events_total: list[int], *, per_group: int) -> dict[str, list[int]]:
    total = len(events_total)
    if total <= 0:
        return {"uniform": [], "burst": [], "sparse": []}
    n = min(int(per_group), total)
    uniform = sorted({int(round(v)) for v in np.linspace(0, total - 1, num=n)})
    burst = sorted(idx for idx, _ in sorted(enumerate(events_total), key=lambda item: item[1], reverse=True)[:n])
    sparse = sorted(idx for idx, _ in sorted(enumerate(events_total), key=lambda item: item[1])[:n])
    return {"uniform": uniform, "burst": burst, "sparse": sparse}


def _sample_interval_events(
    *,
    timestamps: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    t_start_us: int,
    t_end_us: int,
    max_points: int,
) -> dict[str, np.ndarray]:
    start_idx = int(np.searchsorted(timestamps, int(t_start_us), side="left"))
    end_idx = int(np.searchsorted(timestamps, int(t_end_us), side="left"))
    interval_t = np.asarray(timestamps[start_idx:end_idx], dtype=np.int64)
    interval_x = np.asarray(xs[start_idx:end_idx], dtype=np.int16)
    interval_y = np.asarray(ys[start_idx:end_idx], dtype=np.int16)
    interval_p = np.asarray(ps[start_idx:end_idx], dtype=np.int8)
    total = int(interval_t.shape[0])
    if total <= int(max_points):
        chosen = np.arange(total, dtype=np.int64)
    elif total > 0:
        chosen = np.linspace(0, total - 1, num=int(max_points), dtype=np.int64)
    else:
        chosen = np.empty((0,), dtype=np.int64)
    return {
        "t": interval_t[chosen],
        "x": interval_x[chosen],
        "y": interval_y[chosen],
        "p": interval_p[chosen],
        "n_events_total": np.asarray([total], dtype=np.int64),
    }


def _render_3d_scatter(
    *,
    t_us: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    title: str,
    sensor_width: int,
    sensor_height: int,
    output_path: Path,
) -> Path:
    fig = plt.figure(figsize=(7.0, 5.4), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if t_us.size > 0:
        t0 = float(t_us.min())
        t_ms = (t_us.astype(np.float64) - t0) / 1000.0
        neg_mask = p <= 0
        pos_mask = p > 0
        if np.any(neg_mask):
            ax.scatter(
                x[neg_mask],
                y[neg_mask],
                t_ms[neg_mask],
                s=4,
                c="#4aa3ff",
                alpha=0.55,
                depthshade=False,
                label="negative",
            )
        if np.any(pos_mask):
            ax.scatter(
                x[pos_mask],
                y[pos_mask],
                t_ms[pos_mask],
                s=4,
                c="#ff6a5c",
                alpha=0.55,
                depthshade=False,
                label="positive",
            )
        ax.set_zlim(float(t_ms.min(initial=0.0)), float(t_ms.max(initial=0.0) + 1e-6))
    else:
        ax.text2D(0.15, 0.5, "No events in interval", transform=ax.transAxes)

    ax.set_xlim(0, int(sensor_width) - 1)
    ax.set_ylim(int(sensor_height) - 1, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("time (ms)")
    ax.set_title(title, pad=10)
    ax.view_init(elev=24, azim=-58)
    if t_us.size > 0:
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _thumb_with_title(image: Image.Image, *, label: str, thumb_size_wh: tuple[int, int]) -> Image.Image:
    thumb_w, thumb_h = thumb_size_wh
    title_h = 18
    out = Image.new("RGB", (thumb_w, thumb_h + title_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(out)
    draw.text((4, 2), label, fill=(230, 230, 230), font=ImageFont.load_default())
    thumb = image.convert("RGB")
    thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
    paste_x = (thumb_w - thumb.width) // 2
    paste_y = title_h + (thumb_h - thumb.height) // 2
    out.paste(thumb, (paste_x, paste_y))
    return out


def _save_pages(
    *,
    items: list[tuple[str, Path]],
    out_dir: Path,
    title_prefix: str,
    cols: int = 2,
    rows: int = 3,
    thumb_size_wh: tuple[int, int] = (420, 280),
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
        for local_idx, (label, image_path) in enumerate(chunk):
            r = local_idx // cols
            c = local_idx % cols
            x = c * thumb_w
            y = title_h + r * tile_h
            image = Image.open(image_path).convert("RGB")
            page.paste(_thumb_with_title(image, label=label, thumb_size_wh=thumb_size_wh), (x, y))
        out_path = out_dir / f"page_{page_idx // per_page + 1:03d}.png"
        page.save(out_path)
        page_paths.append(str(out_path))
    return page_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3D time-x-y scatter views for representative interval events.")
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "configs" / "paths" / "ev_eye_groundedsam_paths.json")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-code", type=str, default="201")
    parser.add_argument("--eyes", nargs="+", default=["left", "right"])
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=260)
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_interval_events_3d",
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
        interval_rows: list[dict[str, Any]] = []
        for idx in range(len(frame_records) - 1):
            start_us = int(frame_ts[idx])
            end_us = int(frame_ts[idx + 1])
            start_idx = int(np.searchsorted(timestamps, start_us, side="left"))
            end_idx = int(np.searchsorted(timestamps, end_us, side="left"))
            interval_rows.append(
                {
                    "interval_index": int(idx + 1),
                    "frame_from": frame_records[idx].filename,
                    "frame_to": frame_records[idx + 1].filename,
                    "timestamp_from_us": start_us,
                    "timestamp_to_us": end_us,
                    "events_total": int(end_idx - start_idx),
                }
            )

        selected = _select_indices([int(row["events_total"]) for row in interval_rows], per_group=int(args.per_group))
        all_selected = sorted({idx for group in selected.values() for idx in group})
        eye_root = output_root / str(eye)
        renders_dir = eye_root / "renders"
        pages_by_group: dict[str, list[str]] = {}
        render_items_by_group: dict[str, list[tuple[str, Path]]] = {name: [] for name in selected}

        for zero_idx in all_selected:
            row = interval_rows[zero_idx]
            sampled = _sample_interval_events(
                timestamps=timestamps,
                xs=xs,
                ys=ys,
                ps=ps,
                t_start_us=int(row["timestamp_from_us"]),
                t_end_us=int(row["timestamp_to_us"]),
                max_points=int(args.max_points),
            )
            title = (
                f"user{int(args.user_id):02d} {eye} interval={int(row['interval_index']):04d} "
                f"events={int(row['events_total'])}"
            )
            render_path = renders_dir / f"interval_{int(row['interval_index']):04d}.png"
            _render_3d_scatter(
                t_us=np.asarray(sampled["t"]),
                x=np.asarray(sampled["x"]),
                y=np.asarray(sampled["y"]),
                p=np.asarray(sampled["p"]),
                title=title,
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
                output_path=render_path,
            )
            label = f"{int(row['interval_index']):04d} | n={int(row['events_total'])}"
            for group_name, indices in selected.items():
                if zero_idx in indices:
                    render_items_by_group[group_name].append((label, render_path))

        for group_name, items in render_items_by_group.items():
            pages_by_group[group_name] = _save_pages(
                items=items,
                out_dir=eye_root / f"{group_name}_pages",
                title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} 3d {group_name}",
            )

        stats_path = eye_root / "interval_3d_selection.json"
        stats_payload = {
            "user_id": int(args.user_id),
            "eye": str(eye),
            "session_code": str(args.session_code),
            "n_frames": len(frame_records),
            "n_intervals": len(interval_rows),
            "per_group": int(args.per_group),
            "max_points": int(args.max_points),
            "selected_groups": {
                name: [int(interval_rows[idx]["interval_index"]) for idx in indices]
                for name, indices in selected.items()
            },
            "group_pages": pages_by_group,
        }
        stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

        summary_rows.append(stats_payload | {"selection_path": str(stats_path)})
        print(f"[done] rendered 3D interval views for {eye} selected={len(all_selected)}")

    summary_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    lines = [
        "# Session Interval Event 3D Summary",
        "",
        f"- user_id: {int(args.user_id)}",
        f"- session_code: {str(args.session_code)}",
        f"- eyes: {', '.join(str(v) for v in args.eyes)}",
        "- view: time-x-y 3D scatter",
        "- polarity colors: negative=blue, positive=red",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['eye']}",
                f"- n_frames: {row['n_frames']}",
                f"- n_intervals: {row['n_intervals']}",
                f"- per_group: {row['per_group']}",
                f"- max_points: {row['max_points']}",
                f"- uniform_first_page: {row['group_pages']['uniform'][0] if row['group_pages']['uniform'] else ''}",
                f"- burst_first_page: {row['group_pages']['burst'][0] if row['group_pages']['burst'] else ''}",
                f"- sparse_first_page: {row['group_pages']['sparse'][0] if row['group_pages']['sparse'] else ''}",
                "",
            ]
        )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote outputs under {output_root}")


if __name__ == "__main__":
    main()
