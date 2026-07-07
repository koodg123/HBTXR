from __future__ import annotations

import argparse
import json
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


def _load_interval_rows(summary_path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    return {str(row["eye"]): row for row in rows}


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


def _sample_interval_events(
    *,
    timestamps: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    start_us: int,
    end_us: int,
    max_points: int,
) -> dict[str, np.ndarray]:
    start_idx = int(np.searchsorted(timestamps, int(start_us), side="left"))
    end_idx = int(np.searchsorted(timestamps, int(end_us), side="left"))
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


def _render_axis_swapped_view(
    *,
    t_us: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    output_path: Path,
    sensor_width: int,
    sensor_height: int,
) -> Path:
    fig = plt.figure(figsize=(7.4, 5.6), dpi=160, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    if t_us.size > 0:
        t_ref_us = float(t_us.min())
        t_ms = (t_us.astype(np.float64) - t_ref_us) / 1000.0
        neg_mask = p <= 0
        pos_mask = p > 0
        if np.any(neg_mask):
            ax.scatter(t_ms[neg_mask], x[neg_mask], y[neg_mask], s=3, c="#4aa3ff", alpha=0.45, depthshade=False)
        if np.any(pos_mask):
            ax.scatter(t_ms[pos_mask], x[pos_mask], y[pos_mask], s=3, c="#ff6a5c", alpha=0.45, depthshade=False)
        t0 = float(t_ms.min(initial=0.0))
        t1 = float(t_ms.max(initial=0.0) + 1e-6)
    else:
        ax.text2D(0.12, 0.5, "No events in interval", transform=ax.transAxes)
        t0, t1 = 0.0, 1.0

    ax.set_xlim(t0, t1)
    ax.set_ylim(0, int(sensor_width) - 1)
    ax.set_zlim(int(sensor_height) - 1, 0)
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
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
    parser = argparse.ArgumentParser(description="Render axis-swapped event-only interval views.")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-code", type=str, default="201")
    parser.add_argument("--interval-index", type=int, default=5012)
    parser.add_argument("--eyes", nargs="+", default=["left", "right"])
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=260)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--interval-summary",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "interval_5012_accumulated_channels" / "summary.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "axis_swapped_views",
    )
    parser.add_argument(
        "--selection-summary",
        type=Path,
        default=None,
        help="Optional sampled-interval selection summary.json used to render all selected intervals.",
    )
    return parser.parse_args()


def _render_single_interval_views(args: argparse.Namespace, *, raw_root: Path) -> None:
    interval_rows = _load_interval_rows(args.interval_summary)

    for eye in args.eyes:
        interval_row = interval_rows[str(eye)]
        session_dir = _session_dir(raw_root, user_id=int(args.user_id), eye=str(eye), session_code=str(args.session_code))
        event_file = session_dir / "events" / "events.txt"
        events = _load_event_arrays(event_file, eye=str(eye))
        sampled = _sample_interval_events(
            timestamps=np.asarray(events["t"]),
            xs=np.asarray(events["x"]),
            ys=np.asarray(events["y"]),
            ps=np.asarray(events["p"]),
            start_us=int(interval_row["start_us"]),
            end_us=int(interval_row["end_us"]),
            max_points=int(args.max_points),
        )
        base_name = f"user{int(args.user_id):02d}_session{str(args.session_code)}_{eye}_interval_{int(args.interval_index)}_time_x_y"

        _render_axis_swapped_view(
            t_us=np.asarray(sampled["t"]),
            x=np.asarray(sampled["x"]),
            y=np.asarray(sampled["y"]),
            p=np.asarray(sampled["p"]),
            output_path=args.output_root / f"{base_name}.png",
            sensor_width=int(args.sensor_width),
            sensor_height=int(args.sensor_height),
        )
        _render_axis_swapped_view(
            t_us=np.asarray(sampled["t"]),
            x=np.asarray(sampled["x"]),
            y=np.asarray(sampled["y"]),
            p=np.asarray(sampled["p"]),
            output_path=args.output_root / f"{base_name}_with_rois.png",
            sensor_width=int(args.sensor_width),
            sensor_height=int(args.sensor_height),
        )
        print(f"[done] rendered axis_swapped_views for {eye}")


def _render_sampled_interval_views(args: argparse.Namespace, *, raw_root: Path) -> None:
    selection_rows = _load_selection_summary(args.selection_summary)
    summary_rows: list[dict[str, Any]] = []
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

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
        renders_dir = eye_root / "renders"
        rendered_paths: dict[int, Path] = {}
        event_counts: dict[int, int] = {}

        for interval_idx in all_selected:
            row = interval_rows[int(interval_idx) - 1]
            sampled = _sample_interval_events(
                timestamps=np.asarray(events["t"]),
                xs=np.asarray(events["x"]),
                ys=np.asarray(events["y"]),
                ps=np.asarray(events["p"]),
                start_us=int(row["start_us"]),
                end_us=int(row["end_us"]),
                max_points=int(args.max_points),
            )
            render_path = renders_dir / f"interval_{int(interval_idx):04d}.png"
            _render_axis_swapped_view(
                t_us=np.asarray(sampled["t"]),
                x=np.asarray(sampled["x"]),
                y=np.asarray(sampled["y"]),
                p=np.asarray(sampled["p"]),
                output_path=render_path,
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
            )
            rendered_paths[int(interval_idx)] = render_path
            event_counts[int(interval_idx)] = int(row["events_total"])

        pages_by_group: dict[str, list[str]] = {}
        for group_name, interval_indices in selected_groups.items():
            items = [
                (f"{int(interval_idx):04d} | n={event_counts[int(interval_idx)]}", rendered_paths[int(interval_idx)])
                for interval_idx in interval_indices
            ]
            pages_by_group[group_name] = _save_pages(
                items=items,
                out_dir=eye_root / f"{group_name}_pages",
                title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} axis-swapped event-only {group_name}",
            )

        stats_row = {
            "user_id": int(args.user_id),
            "eye": str(eye),
            "session_code": str(args.session_code),
            "n_frames": len(frame_records),
            "n_intervals": len(interval_rows),
            "per_group": int(selection_row.get("per_group", len(next(iter(selected_groups.values()), [])))),
            "max_points": int(args.max_points),
            "selected_groups": selected_groups,
            "group_pages": pages_by_group,
        }
        stats_path = eye_root / "axis_swapped_selection.json"
        stats_path.write_text(json.dumps(stats_row, indent=2), encoding="utf-8")
        summary_rows.append(stats_row | {"selection_path": str(stats_path)})
        print(f"[done] rendered sampled axis-swapped views for {eye} selected={len(all_selected)}")

    summary_json_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    lines = [
        "# Session Sampled Interval Axis-Swapped Views Summary",
        "",
        f"- user_id: {int(args.user_id)}",
        f"- session_code: {str(args.session_code)}",
        f"- eyes: {', '.join(str(v) for v in args.eyes)}",
        "- render: event-only 3D scatter with swapped axes layout and hidden axes",
        "- colors: negative=blue, positive=red",
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
                f"- uniform_first_page: {row['group_pages']['uniform'][0] if row['group_pages'].get('uniform') else ''}",
                f"- burst_first_page: {row['group_pages']['burst'][0] if row['group_pages'].get('burst') else ''}",
                f"- sparse_first_page: {row['group_pages']['sparse'][0] if row['group_pages'].get('sparse') else ''}",
                "",
            ]
        )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote sampled axis-swapped outputs under {output_root}")


def main() -> None:
    args = parse_args()
    raw_root = _resolve_raw_root(args.raw_root)
    if args.selection_summary is not None:
        _render_sampled_interval_views(args, raw_root=raw_root)
        return
    _render_single_interval_views(args, raw_root=raw_root)


if __name__ == "__main__":
    main()
