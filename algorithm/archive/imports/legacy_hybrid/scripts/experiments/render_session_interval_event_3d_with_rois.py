from __future__ import annotations

import argparse
import json
import statistics
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


def _v10_summary_path(anchor_root: Path, *, user_id: int, eye: str, session_code: str) -> Path:
    return anchor_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}" / "summary.json"


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


def _prior_pupil_bbox_from_eye_bbox(prior: dict[str, Any]) -> list[float]:
    ex, ey, ew, eh = [float(v) for v in prior["eye_bbox_xywh_sensor"]]
    rel_cx, rel_cy = [float(v) for v in prior["relative_center_xy"]]
    rel_w, rel_h = [float(v) for v in prior["relative_size_wh"]]
    pw = max(6.0, ew * rel_w)
    ph = max(6.0, eh * rel_h)
    pcx = ex + ew * rel_cx
    pcy = ey + eh * rel_cy
    return [pcx - 0.5 * pw, pcy - 0.5 * ph, pw, ph]


def _load_anchor_prior(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    eye_boxes = [frame["eye_region_bbox_xywh_sensor"] for frame in payload["frames"] if frame.get("eye_region_bbox_xywh_sensor")]
    pupil_boxes = [frame["sensor_pupil_bbox_xywh"] for frame in payload["frames"] if frame.get("sensor_pupil_bbox_xywh")]
    if not eye_boxes or not pupil_boxes:
        raise ValueError(f"Missing eye/pupil boxes in anchor summary: {summary_path}")

    eye_bbox = [
        statistics.median(float(row[0]) for row in eye_boxes),
        statistics.median(float(row[1]) for row in eye_boxes),
        statistics.median(float(row[2]) for row in eye_boxes),
        statistics.median(float(row[3]) for row in eye_boxes),
    ]

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


def _load_event_guided_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["interval_index"]): row for row in rows}


def _draw_box_wireframe(
    ax: Any,
    xywh: list[float],
    *,
    z0: float,
    z1: float,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.4,
    label: str | None = None,
) -> None:
    x, y, w, h = [float(v) for v in xywh]
    x0, x1 = x, x + w
    y0, y1 = y, y + h
    z_vals = [z0, z1]
    top = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    label_used = False
    for z in z_vals:
        xs = [p[0] for p in top]
        ys = [p[1] for p in top]
        zs = [z] * len(top)
        ax.plot(xs, ys, zs, color=color, linestyle=linestyle, linewidth=linewidth, label=label if not label_used else None)
        label_used = True
    for px, py in top[:-1]:
        ax.plot([px, px], [py, py], [z0, z1], color=color, linestyle=linestyle, linewidth=linewidth)


def _render_3d_scatter_with_boxes(
    *,
    t_us: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    eye_bbox_xywh: list[float],
    prior_pupil_bbox_xywh: list[float],
    event_guided_roi_xywh: list[float] | None,
    title: str,
    sensor_width: int,
    sensor_height: int,
    output_path: Path,
) -> Path:
    fig = plt.figure(figsize=(7.4, 5.6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    if t_us.size > 0:
        t0 = float(t_us.min())
        t_ms = (t_us.astype(np.float64) - t0) / 1000.0
        neg_mask = p <= 0
        pos_mask = p > 0
        if np.any(neg_mask):
            ax.scatter(x[neg_mask], y[neg_mask], t_ms[neg_mask], s=3, c="#4aa3ff", alpha=0.45, depthshade=False, label="negative")
        if np.any(pos_mask):
            ax.scatter(x[pos_mask], y[pos_mask], t_ms[pos_mask], s=3, c="#ff6a5c", alpha=0.45, depthshade=False, label="positive")
        z0 = float(t_ms.min(initial=0.0))
        z1 = float(t_ms.max(initial=0.0) + 1e-6)
    else:
        ax.text2D(0.12, 0.5, "No events in interval", transform=ax.transAxes)
        z0, z1 = 0.0, 1.0

    _draw_box_wireframe(ax, eye_bbox_xywh, z0=z0, z1=z1, color="#00ff88", linewidth=1.6, label="eye ROI")
    _draw_box_wireframe(ax, prior_pupil_bbox_xywh, z0=z0, z1=z1, color="#ffffff", linestyle="--", linewidth=1.2, label="pupil prior")
    if event_guided_roi_xywh is not None:
        _draw_box_wireframe(ax, event_guided_roi_xywh, z0=z0, z1=z1, color="#ffd84d", linewidth=1.5, label="event ROI")

    ax.set_xlim(0, int(sensor_width) - 1)
    ax.set_ylim(int(sensor_height) - 1, 0)
    ax.set_zlim(z0, z1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("time (ms)")
    ax.set_title(title, pad=10)
    ax.view_init(elev=24, azim=-58)
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
    thumb_size_wh: tuple[int, int] = (430, 290),
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
    parser = argparse.ArgumentParser(description="Render 3D time-x-y scatter views with eye/pupil ROI wireframes.")
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "configs" / "paths" / "ev_eye_groundedsam_paths.json")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-code", type=str, default="201")
    parser.add_argument("--eyes", nargs="+", default=["left", "right"])
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=260)
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument(
        "--anchor-summary-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_session_samples_7" / "roi_crop_prompted_all48_v10_native_then_crop128_rescue_same8samples",
    )
    parser.add_argument(
        "--event-guided-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_event_guided_pupil_roi",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_interval_events_3d_with_rois",
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
        prior = _load_anchor_prior(
            _v10_summary_path(
                args.anchor_summary_root,
                user_id=int(args.user_id),
                eye=str(eye),
                session_code=str(args.session_code),
            )
        )
        event_guided_rows = _load_event_guided_rows(args.event_guided_root / str(eye) / "interval_event_guided_roi_stats.json")

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
                    "timestamp_from_us": start_us,
                    "timestamp_to_us": end_us,
                    "events_total": int(end_idx - start_idx),
                }
            )

        selected = _select_indices([int(row["events_total"]) for row in interval_rows], per_group=int(args.per_group))
        all_selected = sorted({idx for group in selected.values() for idx in group})
        eye_root = output_root / str(eye)
        renders_dir = eye_root / "renders"
        render_items_by_group: dict[str, list[tuple[str, Path]]] = {name: [] for name in selected}

        for zero_idx in all_selected:
            row = interval_rows[zero_idx]
            interval_idx = int(row["interval_index"])
            event_guided_row = event_guided_rows.get(interval_idx)
            sampled = _sample_interval_events(
                timestamps=timestamps,
                xs=xs,
                ys=ys,
                ps=ps,
                t_start_us=int(row["timestamp_from_us"]),
                t_end_us=int(row["timestamp_to_us"]),
                max_points=int(args.max_points),
            )
            title = f"user{int(args.user_id):02d} {eye} interval={interval_idx:04d} events={int(row['events_total'])}"
            render_path = renders_dir / f"interval_{interval_idx:04d}.png"
            _render_3d_scatter_with_boxes(
                t_us=np.asarray(sampled["t"]),
                x=np.asarray(sampled["x"]),
                y=np.asarray(sampled["y"]),
                p=np.asarray(sampled["p"]),
                eye_bbox_xywh=[float(v) for v in prior["eye_bbox_xywh_sensor"]],
                prior_pupil_bbox_xywh=[float(v) for v in prior["pupil_bbox_xywh_sensor"]],
                event_guided_roi_xywh=None if event_guided_row is None else [float(v) for v in event_guided_row["event_guided_roi_xywh_sensor"]],
                title=title,
                sensor_width=int(args.sensor_width),
                sensor_height=int(args.sensor_height),
                output_path=render_path,
            )
            label = f"{interval_idx:04d} | n={int(row['events_total'])}"
            for group_name, indices in selected.items():
                if zero_idx in indices:
                    render_items_by_group[group_name].append((label, render_path))

        pages_by_group: dict[str, list[str]] = {}
        for group_name, items in render_items_by_group.items():
            pages_by_group[group_name] = _save_pages(
                items=items,
                out_dir=eye_root / f"{group_name}_pages",
                title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} 3d+roi {group_name}",
            )

        summary_row = {
            "user_id": int(args.user_id),
            "eye": str(eye),
            "session_code": str(args.session_code),
            "n_frames": len(frame_records),
            "n_intervals": len(interval_rows),
            "per_group": int(args.per_group),
            "max_points": int(args.max_points),
            "eye_bbox_xywh_sensor": prior["eye_bbox_xywh_sensor"],
            "prior_pupil_bbox_xywh_sensor": prior["pupil_bbox_xywh_sensor"],
            "selected_groups": {
                name: [int(interval_rows[idx]["interval_index"]) for idx in indices]
                for name, indices in selected.items()
            },
            "group_pages": pages_by_group,
        }
        stats_path = eye_root / "interval_3d_with_rois_selection.json"
        stats_path.write_text(json.dumps(summary_row, indent=2), encoding="utf-8")
        summary_rows.append(summary_row | {"selection_path": str(stats_path)})
        print(f"[done] rendered 3D ROI views for {eye} selected={len(all_selected)}")

    summary_json_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    lines = [
        "# Session Interval Event 3D With ROI Summary",
        "",
        f"- user_id: {int(args.user_id)}",
        f"- session_code: {str(args.session_code)}",
        f"- eyes: {', '.join(str(v) for v in args.eyes)}",
        "- view: time-x-y 3D scatter with eye ROI, pupil prior, and event-guided ROI wireframes",
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
                f"- eye_bbox_xywh_sensor: {row['eye_bbox_xywh_sensor']}",
                f"- prior_pupil_bbox_xywh_sensor: {row['prior_pupil_bbox_xywh_sensor']}",
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
