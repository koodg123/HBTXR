from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hbtxr.data.dataset import DEFAULT_EVENT_BUILDER, _build_event_frame  # noqa: E402
from hbtxr.preprocess.io_utils import (  # noqa: E402
    OFFICIAL_SESSION_CODES,
    collect_frame_records,
    discover_session_layout,
    load_events_from_txt,
)
from hbtxr.preprocess.target_fps_build import _load_event_arrays  # noqa: E402


@dataclass
class SampleWindow:
    frame_filename: str
    frame_idx: int | None
    timestamp_us: int
    width_px: int
    height_px: int
    area_px: int
    event_count: int
    window_t: np.ndarray
    window_x: np.ndarray
    window_y: np.ndarray
    window_p: np.ndarray


def _read_paths_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_user_name(user_id: int) -> str:
    return f"user{int(user_id):02d}"


def _session_key(*, user_id: int, eye: str, session_code: str) -> str:
    return f"{_canonical_user_name(user_id)}/{eye}/session_{session_code}"


def _bbox_size(xs: np.ndarray, ys: np.ndarray) -> tuple[int, int, int]:
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return width, height, int(width * height)


def _voxel_to_rgb(voxel: np.ndarray) -> Image.Image:
    neg = np.asarray(voxel[0], dtype=np.float32)
    pos = np.asarray(voxel[1], dtype=np.float32)
    scale = float(np.percentile(np.concatenate([neg.reshape(-1), pos.reshape(-1)]), 99.5)) if (neg.size + pos.size) > 0 else 0.0
    if scale <= 1e-6:
        scale = max(float(neg.max(initial=0.0)), float(pos.max(initial=0.0)), 1.0)
    neg_norm = np.clip(neg / scale, 0.0, 1.0)
    pos_norm = np.clip(pos / scale, 0.0, 1.0)
    rgb = np.zeros((neg.shape[0], neg.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = np.round(pos_norm * 255.0).astype(np.uint8)
    rgb[..., 2] = np.round(neg_norm * 255.0).astype(np.uint8)
    rgb[..., 1] = np.round(np.clip((pos_norm + neg_norm) * 0.25, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _voxel_to_point_rgb(voxel: np.ndarray) -> Image.Image:
    neg = np.asarray(voxel[0], dtype=np.float32)
    pos = np.asarray(voxel[1], dtype=np.float32)
    canvas = np.zeros((neg.shape[0], neg.shape[1], 3), dtype=np.uint8)
    canvas[..., :] = np.asarray([12, 12, 12], dtype=np.uint8)

    neg_active = neg > 1e-6
    pos_active = pos > 1e-6
    both_active = neg_active & pos_active
    only_neg = neg_active & ~pos_active
    only_pos = pos_active & ~neg_active

    canvas[only_neg] = np.asarray([70, 120, 255], dtype=np.uint8)
    canvas[only_pos] = np.asarray([255, 90, 90], dtype=np.uint8)
    canvas[both_active] = np.asarray([255, 255, 180], dtype=np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def _draw_bbox(image: Image.Image, xs: np.ndarray, ys: np.ndarray) -> Image.Image:
    if xs.size == 0 or ys.size == 0:
        return image
    draw = ImageDraw.Draw(image)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)
    return image


def _representative_index(samples: list[SampleWindow]) -> int:
    if not samples:
        return 0
    areas = np.asarray([sample.area_px for sample in samples], dtype=np.float64)
    target = float(areas.mean()) if areas.size else 0.0
    distances = np.abs(areas - target)
    return int(np.argmin(distances))


def _build_preview_image(
    *,
    sample: SampleWindow,
    session_key: str,
    sensor_size_wh: tuple[int, int],
) -> Image.Image:
    voxel, _ = _build_event_frame(
        {
            "t": sample.window_t,
            "x": sample.window_x,
            "y": sample.window_y,
            "p": sample.window_p,
        },
        sensor_size_wh=sensor_size_wh,
        end_timestamp_us=int(sample.timestamp_us),
        event_window=DEFAULT_EVENT_BUILDER,
    )
    heatmap = _draw_bbox(_voxel_to_rgb(voxel), sample.window_x, sample.window_y)
    pointmap = _draw_bbox(_voxel_to_point_rgb(voxel), sample.window_x, sample.window_y)
    spacer = 8
    panel = Image.new("RGB", (heatmap.width * 2 + spacer, heatmap.height), color=(8, 8, 8))
    panel.paste(heatmap, (0, 0))
    panel.paste(pointmap, (heatmap.width + spacer, 0))
    title_h = 34
    canvas = Image.new("RGB", (panel.width, panel.height + title_h), color=(16, 16, 16))
    canvas.paste(panel, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = f"{session_key}  frame={sample.frame_idx}  size={sample.width_px}x{sample.height_px}  left=heatmap right=points"
    draw.text((6, 6), title, fill=(240, 240, 240), font=font)
    return canvas


def _build_contact_sheet(
    *,
    items: list[tuple[str, Image.Image]],
    page_title: str,
    out_path: Path,
    cols: int = 12,
    cell_wh: tuple[int, int] = (160, 140),
) -> None:
    if not items:
        return
    cell_w, cell_h = cell_wh
    rows = int(math.ceil(len(items) / cols))
    title_h = 28
    page = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()
    draw.text((8, 6), page_title, fill=(240, 240, 240), font=font)
    for idx, (label, image) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = title_h + row * cell_h
        thumb = image.copy()
        thumb.thumbnail((cell_w, cell_h - 16))
        page.paste(thumb, (x, y + 16))
        draw.text((x + 4, y + 2), label, fill=(220, 220, 220), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


def _compute_samples_for_txt(
    *,
    event_file: Path,
    eye: str,
    frame_records: list[Any],
    frames_per_session: int,
    event_count_target: int,
) -> list[SampleWindow]:
    targets = frame_records[:frames_per_session]
    if not targets:
        return []
    target_timestamps = [int(record.timestamp_us) for record in targets]
    history_t: deque[int] = deque()
    history_x: deque[int] = deque()
    history_y: deque[int] = deque()
    history_p: deque[int] = deque()
    samples: list[SampleWindow] = []
    target_idx = 0

    with event_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                t_s, x_s, y_s, p_s = stripped.split()[:4]
                t = int(float(t_s))
                x = int(float(x_s))
                y = int(float(y_s))
                p = int(float(p_s))
            except ValueError:
                continue

            if eye == "left" and ((x == 158 and y == 27) or (x == 324 and y == 27)):
                continue

            while target_idx < len(target_timestamps) and t > target_timestamps[target_idx]:
                xs = np.asarray(history_x, dtype=np.int16)
                ys = np.asarray(history_y, dtype=np.int16)
                width, height, area = _bbox_size(xs, ys)
                samples.append(
                    SampleWindow(
                        frame_filename=str(targets[target_idx].filename),
                        frame_idx=targets[target_idx].frame_idx,
                        timestamp_us=target_timestamps[target_idx],
                        width_px=width,
                        height_px=height,
                        area_px=area,
                        event_count=int(xs.size),
                        window_t=np.asarray(history_t, dtype=np.int64),
                        window_x=xs,
                        window_y=ys,
                        window_p=np.asarray(history_p, dtype=np.int8),
                    )
                )
                target_idx += 1
            if target_idx >= len(target_timestamps):
                break

            history_t.append(t)
            history_x.append(x)
            history_y.append(y)
            history_p.append(p)
            if len(history_t) > event_count_target:
                history_t.popleft()
                history_x.popleft()
                history_y.popleft()
                history_p.popleft()

    while target_idx < len(target_timestamps):
        xs = np.asarray(history_x, dtype=np.int16)
        ys = np.asarray(history_y, dtype=np.int16)
        width, height, area = _bbox_size(xs, ys)
        samples.append(
            SampleWindow(
                frame_filename=str(targets[target_idx].filename),
                frame_idx=targets[target_idx].frame_idx,
                timestamp_us=target_timestamps[target_idx],
                width_px=width,
                height_px=height,
                area_px=area,
                event_count=int(xs.size),
                window_t=np.asarray(history_t, dtype=np.int64),
                window_x=xs,
                window_y=ys,
                window_p=np.asarray(history_p, dtype=np.int8),
            )
        )
        target_idx += 1
    return samples


def _compute_samples_for_npz(
    *,
    event_file: Path,
    frame_records: list[Any],
    frames_per_session: int,
    event_count_target: int,
) -> list[SampleWindow]:
    targets = frame_records[:frames_per_session]
    if not targets:
        return []
    events = _load_event_arrays(event_file)
    timestamps = np.asarray(events["t"], dtype=np.int64)
    xs_all = np.asarray(events["x"], dtype=np.int16)
    ys_all = np.asarray(events["y"], dtype=np.int16)
    ps_all = np.asarray(events["p"], dtype=np.int8)
    frame_ts = np.asarray([int(record.timestamp_us) for record in targets], dtype=np.int64)
    end_indices = np.searchsorted(timestamps, frame_ts, side="right")
    samples: list[SampleWindow] = []
    for record, end_idx, target_ts in zip(targets, end_indices.tolist(), frame_ts.tolist(), strict=False):
        start_idx = max(0, int(end_idx) - int(event_count_target))
        window_t = timestamps[start_idx:end_idx]
        window_x = xs_all[start_idx:end_idx]
        window_y = ys_all[start_idx:end_idx]
        window_p = ps_all[start_idx:end_idx]
        width, height, area = _bbox_size(window_x, window_y)
        samples.append(
            SampleWindow(
                frame_filename=str(record.filename),
                frame_idx=record.frame_idx,
                timestamp_us=int(target_ts),
                width_px=width,
                height_px=height,
                area_px=area,
                event_count=int(window_x.size),
                window_t=np.asarray(window_t, dtype=np.int64),
                window_x=np.asarray(window_x, dtype=np.int16),
                window_y=np.asarray(window_y, dtype=np.int16),
                window_p=np.asarray(window_p, dtype=np.int8),
            )
        )
    return samples


def _compute_samples_for_session(
    *,
    session_dir: Path,
    user_id: int,
    eye: str,
    frames_per_session: int,
    event_count_target: int,
) -> tuple[list[SampleWindow], Path]:
    layout = discover_session_layout(session_dir, user_id=user_id)
    if layout.event_file is None:
        raise FileNotFoundError(f"Missing event file for {session_dir}")
    if layout.frames_dir is None:
        raise FileNotFoundError(f"Missing frames directory for {session_dir}")
    frame_records = collect_frame_records(layout.frames_dir)
    event_file = Path(layout.event_file)
    if event_file.suffix.lower() == ".npz":
        return (
            _compute_samples_for_npz(
                event_file=event_file,
                frame_records=frame_records,
                frames_per_session=frames_per_session,
                event_count_target=event_count_target,
            ),
            event_file,
        )
    return (
        _compute_samples_for_txt(
            event_file=event_file,
            eye=eye,
            frame_records=frame_records,
            frames_per_session=frames_per_session,
            event_count_target=event_count_target,
        ),
        event_file,
    )


def _session_dir_from_key(raw_root: Path, session_key: str) -> tuple[Path, int, str, str]:
    user_name, eye, session_name = session_key.split("/")
    user_id = int(user_name.replace("user", ""))
    session_code = session_name.replace("session_", "")
    session_dir = raw_root / f"user{user_id}" / eye / f"session_{session_code[0]}_{session_code[1]}_{session_code[2]}"
    return session_dir, user_id, eye, session_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sampled all-user event-voxel previews and spatial-size statistics.")
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=PROJECT_ROOT / "workspace_session_samples_10" / "all48_dataset_construction_same8samples" / "sample_manifest.jsonl",
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paths" / "ev_eye_groundedsam_paths.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "all48_first8_per_session",
    )
    parser.add_argument("--frames-per-session", type=int, default=8)
    parser.add_argument("--event-count-target", type=int, default=int(DEFAULT_EVENT_BUILDER["event_count_target"]))
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths_cfg = _read_paths_config(args.paths_config)
    raw_root = Path(paths_cfg["raw_root"])
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    representative_dir = output_root / "representatives"
    representative_dir.mkdir(parents=True, exist_ok=True)
    page_dir = output_root / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [json.loads(line) for line in args.sample_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    session_keys = sorted({row["session_key"] for row in manifest_rows if str(row.get("session_code")) in OFFICIAL_SESSION_CODES})
    sensor_size_wh = (int(args.sensor_width), int(args.sensor_height))

    session_rows: list[dict[str, Any]] = []
    code_rows: dict[str, list[dict[str, Any]]] = {code: [] for code in sorted(OFFICIAL_SESSION_CODES)}
    preview_items: dict[str, list[tuple[str, Image.Image]]] = {code: [] for code in sorted(OFFICIAL_SESSION_CODES)}

    for index, session_key in enumerate(session_keys, start=1):
        session_dir, user_id, eye, session_code = _session_dir_from_key(raw_root, session_key)
        samples, event_file = _compute_samples_for_session(
            session_dir=session_dir,
            user_id=user_id,
            eye=eye,
            frames_per_session=int(args.frames_per_session),
            event_count_target=int(args.event_count_target),
        )
        if not samples:
            continue
        widths = np.asarray([sample.width_px for sample in samples], dtype=np.float64)
        heights = np.asarray([sample.height_px for sample in samples], dtype=np.float64)
        areas = np.asarray([sample.area_px for sample in samples], dtype=np.float64)
        rep_idx = _representative_index(samples)
        representative = samples[rep_idx]
        preview = _build_preview_image(sample=representative, session_key=session_key, sensor_size_wh=sensor_size_wh)
        rep_name = f"{session_key.replace('/', '__')}.png"
        rep_path = representative_dir / rep_name
        preview.save(rep_path)
        preview_items[session_code].append((f"u{user_id:02d}-{eye[0].upper()}", preview))
        row = {
            "session_key": session_key,
            "user_id": user_id,
            "eye": eye,
            "session_code": session_code,
            "event_file": str(event_file),
            "n_voxel_samples": len(samples),
            "mean_width_px": float(widths.mean()),
            "min_width_px": int(widths.min(initial=0)),
            "max_width_px": int(widths.max(initial=0)),
            "mean_height_px": float(heights.mean()),
            "min_height_px": int(heights.min(initial=0)),
            "max_height_px": int(heights.max(initial=0)),
            "mean_area_px": float(areas.mean()),
            "min_area_px": int(areas.min(initial=0)),
            "max_area_px": int(areas.max(initial=0)),
            "representative_frame_filename": representative.frame_filename,
            "representative_frame_idx": representative.frame_idx,
            "representative_timestamp_us": representative.timestamp_us,
            "representative_width_px": representative.width_px,
            "representative_height_px": representative.height_px,
            "representative_area_px": representative.area_px,
            "representative_preview_path": str(rep_path),
        }
        session_rows.append(row)
        code_rows[session_code].append(row)
        if index % 24 == 0 or index == len(session_keys):
            print(f"[progress] processed {index}/{len(session_keys)} session instances")

    aggregate_rows: list[dict[str, Any]] = []
    for session_code in sorted(code_rows):
        rows = code_rows[session_code]
        if not rows:
            continue
        width_values = np.asarray([float(row["representative_width_px"]) for row in rows], dtype=np.float64)
        height_values = np.asarray([float(row["representative_height_px"]) for row in rows], dtype=np.float64)
        area_values = np.asarray([float(row["representative_area_px"]) for row in rows], dtype=np.float64)
        sample_width_values = np.asarray([float(row["mean_width_px"]) for row in rows], dtype=np.float64)
        sample_height_values = np.asarray([float(row["mean_height_px"]) for row in rows], dtype=np.float64)
        sample_area_values = np.asarray([float(row["mean_area_px"]) for row in rows], dtype=np.float64)
        aggregate_rows.append(
            {
                "session_code": session_code,
                "session_instances": len(rows),
                "voxel_samples_total": int(sum(int(row["n_voxel_samples"]) for row in rows)),
                "representative_width_mean_px": float(width_values.mean()),
                "representative_width_min_px": int(width_values.min(initial=0)),
                "representative_width_max_px": int(width_values.max(initial=0)),
                "representative_height_mean_px": float(height_values.mean()),
                "representative_height_min_px": int(height_values.min(initial=0)),
                "representative_height_max_px": int(height_values.max(initial=0)),
                "representative_area_mean_px": float(area_values.mean()),
                "representative_area_min_px": int(area_values.min(initial=0)),
                "representative_area_max_px": int(area_values.max(initial=0)),
                "session_mean_width_mean_px": float(sample_width_values.mean()),
                "session_mean_width_min_px": float(sample_width_values.min(initial=0.0)),
                "session_mean_width_max_px": float(sample_width_values.max(initial=0.0)),
                "session_mean_height_mean_px": float(sample_height_values.mean()),
                "session_mean_height_min_px": float(sample_height_values.min(initial=0.0)),
                "session_mean_height_max_px": float(sample_height_values.max(initial=0.0)),
                "session_mean_area_mean_px": float(sample_area_values.mean()),
                "session_mean_area_min_px": float(sample_area_values.min(initial=0.0)),
                "session_mean_area_max_px": float(sample_area_values.max(initial=0.0)),
            }
        )
        _build_contact_sheet(
            items=sorted(preview_items[session_code], key=lambda item: item[0]),
            page_title=f"Event Voxel Representatives - session_{session_code} - first {args.frames_per_session} raw frames per session instance",
            out_path=page_dir / f"session_{session_code}_overview.png",
        )

    session_rows_path = output_root / "event_voxel_session_instance_stats.json"
    session_rows_path.write_text(json.dumps(session_rows, indent=2), encoding="utf-8")
    aggregate_path = output_root / "event_voxel_session_code_summary.json"
    aggregate_path.write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Event Voxel Spatial Summary",
        "",
        "## Run Scope",
        f"- official session instances processed: {len(session_rows)}",
        f"- per-session frame count used: {int(args.frames_per_session)}",
        f"- total voxel samples analyzed: {sum(int(row['n_voxel_samples']) for row in session_rows)}",
        f"- sensor_size_wh used for rendered voxel previews: ({int(args.sensor_width)}, {int(args.sensor_height)})",
        f"- event builder policy: {DEFAULT_EVENT_BUILDER['policy']}",
        f"- event_count_target: {int(args.event_count_target)}",
        "- spatial size definition: occupied non-zero event bbox width x height within the voxel window",
        "",
        "## Session-Code Summary",
    ]
    for row in aggregate_rows:
        markdown_lines.extend(
            [
                f"### session_{row['session_code']}",
                f"- session_instances: {row['session_instances']}",
                f"- voxel_samples_total: {row['voxel_samples_total']}",
                f"- representative_width_px: mean={row['representative_width_mean_px']:.2f}, min={row['representative_width_min_px']}, max={row['representative_width_max_px']}",
                f"- representative_height_px: mean={row['representative_height_mean_px']:.2f}, min={row['representative_height_min_px']}, max={row['representative_height_max_px']}",
                f"- representative_area_px: mean={row['representative_area_mean_px']:.2f}, min={row['representative_area_min_px']}, max={row['representative_area_max_px']}",
                f"- session_mean_width_px: mean={row['session_mean_width_mean_px']:.2f}, min={row['session_mean_width_min_px']:.2f}, max={row['session_mean_width_max_px']:.2f}",
                f"- session_mean_height_px: mean={row['session_mean_height_mean_px']:.2f}, min={row['session_mean_height_min_px']:.2f}, max={row['session_mean_height_max_px']:.2f}",
                f"- session_mean_area_px: mean={row['session_mean_area_mean_px']:.2f}, min={row['session_mean_area_min_px']:.2f}, max={row['session_mean_area_max_px']:.2f}",
                f"- overview_page: {str((page_dir / ('session_' + str(row['session_code']) + '_overview.png')).resolve())}",
                "- overview_page layout: left=accumulated voxel heatmap, right=accumulated occupied-point map",
                "",
            ]
        )
    (output_root / "event_voxel_spatial_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"[done] wrote outputs under {output_root}")


if __name__ == "__main__":
    main()
