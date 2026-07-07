#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / "tsgss" / ".mplconfig").resolve()))

from build_tsgss_frame_event_visualization import _load_frame_rgb, _read_jsonl_rows, _resolve_path
from hbtxr.preprocess.io_utils import SENSOR_HEIGHT, SENSOR_WIDTH, maybe_link_or_copy
from hbtxr.utils.io import write_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build time-x-y event previews with red/blue polarity channels from tsgss event alignment."
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
        default="tsgss/event_xyz_preview",
        help="Output root for time-x-y event previews",
    )
    parser.add_argument(
        "--session-key",
        type=str,
        default="user01/left/session_101",
        help="Session used to build a full contact sheet of interval triptychs",
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
        "--point-size",
        type=float,
        default=7.0,
        help="Scatter point size in the 3D time-x-y event plot",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy", "skip"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_aggregate_rows(alignment_root: Path) -> list[dict[str, Any]]:
    return _read_jsonl_rows(alignment_root / "aggregate_frame_event_manifest.jsonl")


def _load_interval_arrays(row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(row["aligned_events_path"])).resolve()
    data = np.load(path)
    idx = int(row["aligned_interval_index"])
    return {
        "t": np.asarray(data["t"][idx]),
        "x": np.asarray(data["x"][idx]),
        "y": np.asarray(data["y"][idx]),
        "p": np.asarray(data["p"][idx]),
        "valid_mask": np.asarray(data["valid_mask"][idx]),
    }


def _render_event_xyz_plot(
    *,
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    sensor_width: int,
    sensor_height: int,
    point_size: float,
) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    valid_mask = np.asarray(arrays["valid_mask"], dtype=bool)
    t = np.asarray(arrays["t"][valid_mask], dtype=np.int64)
    x = np.asarray(arrays["x"][valid_mask], dtype=np.int64)
    y = np.asarray(arrays["y"][valid_mask], dtype=np.int64)
    p = np.asarray(arrays["p"][valid_mask], dtype=np.int64)

    start_timestamp_us = int(row["start_timestamp_us"])
    end_timestamp_us = int(row["end_timestamp_us"])
    duration_us = max(1, end_timestamp_us - start_timestamp_us)
    time_ms = (t.astype(np.float64) - float(start_timestamp_us)) / 1000.0
    duration_ms = float(duration_us) / 1000.0

    fig = Figure(figsize=(7.6, 5.6), facecolor="#0f1117")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f1117")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#e7edf7")
        axis.set_tick_params(colors="#c9d3e3")

    try:
        ax.xaxis.pane.set_facecolor((0.09, 0.11, 0.14, 1.0))
        ax.yaxis.pane.set_facecolor((0.09, 0.11, 0.14, 1.0))
        ax.zaxis.pane.set_facecolor((0.09, 0.11, 0.14, 1.0))
        ax.xaxis.pane.set_edgecolor("#6f7e94")
        ax.yaxis.pane.set_edgecolor("#6f7e94")
        ax.zaxis.pane.set_edgecolor("#6f7e94")
    except Exception:
        pass

    pos_mask = p > 0
    neg_mask = ~pos_mask
    if np.any(pos_mask):
        ax.scatter(
            time_ms[pos_mask],
            x[pos_mask],
            y[pos_mask],
            s=float(point_size),
            c="#ff4d5a",
            alpha=0.34,
            depthshade=False,
            label=f"positive (+): {int(np.count_nonzero(pos_mask))}",
        )
    if np.any(neg_mask):
        ax.scatter(
            time_ms[neg_mask],
            x[neg_mask],
            y[neg_mask],
            s=float(point_size),
            c="#3f7cff",
            alpha=0.34,
            depthshade=False,
            label=f"negative (-): {int(np.count_nonzero(neg_mask))}",
        )
    if len(t) == 0:
        ax.text(
            0.5,
            sensor_width * 0.5,
            sensor_height * 0.5,
            "No events",
            color="#f4f7fb",
        )

    ax.set_xlim(0.0, max(duration_ms, 0.1))
    ax.set_ylim(0.0, float(sensor_width - 1))
    ax.set_zlim(float(sensor_height - 1), 0.0)
    ax.set_xlabel("Time (ms)", labelpad=10)
    ax.set_ylabel("X", labelpad=10)
    ax.set_zlabel("Y", labelpad=10)
    ax.view_init(elev=22, azim=-64)
    ax.set_title(
        (
            f"Time-X-Y Event Plot | interval {int(row['interval_index']):02d} | "
            f"{int(row['raw_event_count'])}->{int(row['target_event_count'])} | {row['alignment_mode']}"
        ),
        color="#f4f7fb",
        pad=14,
        fontsize=11,
    )
    legend = None
    if np.any(pos_mask) or np.any(neg_mask):
        legend = ax.legend(loc="upper left", frameon=True, fontsize=8)
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor("#171b22")
        frame.set_edgecolor("#5f6b7b")
        for text in legend.get_texts():
            text.set_color("#edf2f7")

    fig.tight_layout()
    buf = BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    buf.close()
    return image


def _compose_triptych(
    *,
    row: dict[str, Any],
    start_frame: Image.Image,
    event_plot: Image.Image,
    end_frame: Image.Image,
) -> Image.Image:
    frame_size = (220, 156)
    event_size = (520, 360)
    start_tile = ImageOps.pad(start_frame.convert("RGB"), frame_size, color=(230, 230, 230))
    end_tile = ImageOps.pad(end_frame.convert("RGB"), frame_size, color=(230, 230, 230))
    start_tile = ImageOps.expand(start_tile, border=3, fill=(74, 126, 83))
    end_tile = ImageOps.expand(end_tile, border=3, fill=(128, 79, 54))
    event_tile = ImageOps.pad(event_plot.convert("RGB"), event_size, color=(14, 17, 23))
    event_tile = ImageOps.expand(event_tile, border=3, fill=(47, 64, 92))

    gap = 18
    title_h = 38
    footer_h = 24
    width = start_tile.width + event_tile.width + end_tile.width + gap * 4
    height = title_h + max(start_tile.height, event_tile.height, end_tile.height) + footer_h
    canvas = Image.new("RGB", (width, height), color=(245, 241, 235))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 10),
        (
            f"Interval {int(row['interval_index']):02d} | {int(row['interval_us']) / 1000.0:.1f} ms | "
            f"mode={row['alignment_mode']}"
        ),
        fill=(33, 31, 29),
    )
    y = title_h
    x = gap
    canvas.paste(start_tile, (x, y))
    x += start_tile.width + gap
    canvas.paste(event_tile, (x, y))
    x += event_tile.width + gap
    canvas.paste(end_tile, (x, y))
    draw.text((gap, height - 18), "Start Frame", fill=(50, 74, 56))
    draw.text((gap + start_tile.width + gap + 10, height - 18), "Event (Time-X-Y, red/blue channels)", fill=(56, 67, 96))
    draw.text((width - end_tile.width - gap, height - 18), "End Frame", fill=(92, 58, 42))
    return canvas


def _build_contact_sheet(images: list[Image.Image], *, save_path: Path) -> None:
    if not images:
        return
    thumb_size = (420, 150)
    cols = 2
    gap = 18
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new(
        "RGB",
        (gap + cols * (thumb_size[0] + gap), gap + rows * (thumb_size[1] + gap)),
        color=(251, 247, 241),
    )
    for idx, image in enumerate(images):
        tile = ImageOps.pad(image, thumb_size, color=(235, 231, 224))
        x = gap + (idx % cols) * (thumb_size[0] + gap)
        y = gap + (idx // cols) * (thumb_size[1] + gap)
        canvas.paste(tile, (x, y))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def _find_mode_examples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row["alignment_mode"])
        if mode not in examples:
            examples[mode] = row
    return examples


def _session_rows(rows: list[dict[str, Any]], session_key: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row["session_key"]) == str(session_key)]
    return sorted(selected, key=lambda row: int(row["interval_index"]))


def main() -> None:
    args = build_argparser().parse_args()
    alignment_root = _resolve_path(args.alignment_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    quick_preview_root = output_root.parent / "event_xyz_quick_preview"
    quick_preview_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = _read_aggregate_rows(alignment_root)
    mode_examples = _find_mode_examples(aggregate_rows)
    rendered_examples: dict[str, dict[str, str]] = {}

    representative_root = output_root / "representatives"
    representative_root.mkdir(parents=True, exist_ok=True)

    for mode, row in mode_examples.items():
        arrays = _load_interval_arrays(row)
        event_plot = _render_event_xyz_plot(
            row=row,
            arrays=arrays,
            sensor_width=int(args.sensor_width),
            sensor_height=int(args.sensor_height),
            point_size=float(args.point_size),
        )
        start_frame = _load_frame_rgb(Path(str(row["start_frame_path"])).resolve())
        end_frame = _load_frame_rgb(Path(str(row["end_frame_path"])).resolve())
        triptych = _compose_triptych(
            row=row,
            start_frame=start_frame,
            event_plot=event_plot,
            end_frame=end_frame,
        )

        event_plot_path = representative_root / f"{mode}_event_xyz.png"
        triptych_path = representative_root / f"{mode}_triptych.png"
        event_plot.save(event_plot_path)
        triptych.save(triptych_path)
        rendered_examples[mode] = {
            "session_key": str(row["session_key"]),
            "event_plot_path": str(event_plot_path),
            "triptych_path": str(triptych_path),
        }

    session_rows = _session_rows(aggregate_rows, session_key=str(args.session_key))
    session_root = output_root / "sessions" / str(args.session_key).replace("/", "__")
    session_root.mkdir(parents=True, exist_ok=True)
    session_triptychs: list[Image.Image] = []
    for row in session_rows:
        arrays = _load_interval_arrays(row)
        event_plot = _render_event_xyz_plot(
            row=row,
            arrays=arrays,
            sensor_width=int(args.sensor_width),
            sensor_height=int(args.sensor_height),
            point_size=float(args.point_size),
        )
        start_frame = _load_frame_rgb(Path(str(row["start_frame_path"])).resolve())
        end_frame = _load_frame_rgb(Path(str(row["end_frame_path"])).resolve())
        triptych = _compose_triptych(
            row=row,
            start_frame=start_frame,
            event_plot=event_plot,
            end_frame=end_frame,
        )
        interval_path = session_root / f"interval_{int(row['interval_index']):03d}_triptych.png"
        triptych.save(interval_path)
        if int(row["interval_index"]) == 0:
            event_plot.save(session_root / "interval_000_event_xyz.png")
        session_triptychs.append(triptych)

    contact_sheet_path = session_root / "session_contact_sheet.png"
    _build_contact_sheet(session_triptychs, save_path=contact_sheet_path)

    preview_order = [
        ("interpolated_dense", "01_interpolated_dense_triptych.png"),
        ("subsampled", "02_subsampled_triptych.png"),
        ("empty_pad", "03_empty_pad_triptych.png"),
        ("interp_single", "04_interp_single_triptych.png"),
    ]
    for mode, filename in preview_order:
        example = rendered_examples.get(mode)
        if example is None:
            continue
        maybe_link_or_copy(
            Path(example["triptych_path"]),
            quick_preview_root / filename,
            mode=str(args.link_mode),
            overwrite=True,
        )
    maybe_link_or_copy(
        contact_sheet_path,
        quick_preview_root / "05_session_contact_sheet.png",
        mode=str(args.link_mode),
        overwrite=True,
    )
    maybe_link_or_copy(
        session_root / "interval_000_event_xyz.png",
        quick_preview_root / "06_session_interval0_event_xyz.png",
        mode=str(args.link_mode),
        overwrite=True,
    )

    summary = {
        "experiment": "tsgss_event_xyz_preview",
        "alignment_root": str(alignment_root),
        "output_root": str(output_root),
        "session_key": str(args.session_key),
        "sensor_width": int(args.sensor_width),
        "sensor_height": int(args.sensor_height),
        "point_size": float(args.point_size),
        "mode_examples": rendered_examples,
        "session_contact_sheet_path": str(contact_sheet_path),
        "quick_preview_root": str(quick_preview_root),
    }
    write_json(summary, output_root / "experiment_summary.json")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# TSGSS Event XYZ Preview",
                "",
                "This preview renders aligned events in a true `time-x-y` 3D plot.",
                "",
                "Polarity channels:",
                "- red: positive (+)",
                "- blue: negative (-)",
                "",
                "Quick preview files:",
                "- `tsgss/event_xyz_quick_preview/01_interpolated_dense_triptych.png`",
                "- `tsgss/event_xyz_quick_preview/02_subsampled_triptych.png`",
                "- `tsgss/event_xyz_quick_preview/03_empty_pad_triptych.png`",
                "- `tsgss/event_xyz_quick_preview/04_interp_single_triptych.png`",
                "- `tsgss/event_xyz_quick_preview/05_session_contact_sheet.png`",
                "- `tsgss/event_xyz_quick_preview/06_session_interval0_event_xyz.png`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[DONE] mode_examples={len(rendered_examples)} session_rows={len(session_rows)} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
