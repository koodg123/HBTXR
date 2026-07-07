#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from build_tsgss_frame_event_visualization import _load_frame_rgb, _read_jsonl_rows, _resolve_path
from hbtxr.preprocess.io_utils import SENSOR_HEIGHT, SENSOR_WIDTH, maybe_link_or_copy
from hbtxr.utils.io import write_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render sensor-plane event previews with polarity channels on a white background.",
    )
    parser.add_argument("--alignment-root", type=str, default="tsgss/event_alignment")
    parser.add_argument("--output-root", type=str, default="tsgss/event_plane_preview")
    parser.add_argument("--session-key", type=str, default="user01/left/session_101")
    parser.add_argument("--sensor-width", type=int, default=SENSOR_WIDTH)
    parser.add_argument("--sensor-height", type=int, default=SENSOR_HEIGHT)
    parser.add_argument("--background", type=str, default="white", choices=["white", "black"])
    parser.add_argument("--positive-color", type=str, default="green", choices=["green", "red"])
    parser.add_argument("--negative-color", type=str, default="blue", choices=["blue", "red"])
    parser.add_argument("--blur-radius", type=float, default=1.1)
    parser.add_argument("--target-size", type=int, default=320)
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_aggregate_rows(alignment_root: Path) -> list[dict[str, Any]]:
    return _read_jsonl_rows(alignment_root / "aggregate_frame_event_manifest.jsonl")


def _session_rows(rows: list[dict[str, Any]], session_key: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row["session_key"]) == str(session_key)]
    return sorted(selected, key=lambda row: int(row["interval_index"]))


def _find_mode_examples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row["alignment_mode"])
        if mode not in examples:
            examples[mode] = row
    return examples


def _load_interval_arrays(row: dict[str, Any]) -> dict[str, np.ndarray]:
    data = np.load(Path(str(row["aligned_events_path"])).resolve())
    idx = int(row["aligned_interval_index"])
    return {
        "x": np.asarray(data["x"][idx]),
        "y": np.asarray(data["y"][idx]),
        "p": np.asarray(data["p"][idx]),
        "valid_mask": np.asarray(data["valid_mask"][idx]),
    }


def _color_rgb(name: str) -> tuple[int, int, int]:
    mapping = {
        "green": (64, 220, 96),
        "blue": (64, 92, 255),
        "red": (255, 72, 88),
    }
    return mapping[str(name)]


def _density_map(xs: np.ndarray, ys: np.ndarray, *, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((int(height), int(width)), dtype=np.float32)
    if len(xs) > 0:
        np.add.at(canvas, (ys.astype(np.int64), xs.astype(np.int64)), 1.0)
    return np.log1p(canvas)


def _apply_blur(channel: np.ndarray, radius: float) -> np.ndarray:
    if float(radius) <= 0.0:
        return np.asarray(channel, dtype=np.float32)
    img = Image.fromarray(np.clip(channel / max(float(channel.max()), 1.0) * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if float(channel.max()) > 0.0:
        arr *= float(channel.max())
    return arr


def _blend_color(
    base: np.ndarray,
    alpha: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return base * (1.0 - alpha[..., None]) + color_arr * alpha[..., None]


def _render_event_plane(
    *,
    arrays: dict[str, np.ndarray],
    width: int,
    height: int,
    background: str,
    positive_color: str,
    negative_color: str,
    blur_radius: float,
    target_size: int,
) -> Image.Image:
    valid_mask = np.asarray(arrays["valid_mask"], dtype=bool)
    x = np.asarray(arrays["x"][valid_mask], dtype=np.int64)
    y = np.asarray(arrays["y"][valid_mask], dtype=np.int64)
    p = np.asarray(arrays["p"][valid_mask], dtype=np.int64)

    pos_mask = p > 0
    neg_mask = ~pos_mask
    pos = _density_map(x[pos_mask], y[pos_mask], width=width, height=height)
    neg = _density_map(x[neg_mask], y[neg_mask], width=width, height=height)
    pos = _apply_blur(pos, radius=float(blur_radius))
    neg = _apply_blur(neg, radius=float(blur_radius))

    denom = float(max(float(pos.max()), float(neg.max()), 1.0))
    pos_alpha = np.clip(pos / denom, 0.0, 1.0) * 0.96
    neg_alpha = np.clip(neg / denom, 0.0, 1.0) * 0.96

    if str(background) == "black":
        base = np.zeros((int(height), int(width), 3), dtype=np.float32)
    else:
        base = np.full((int(height), int(width), 3), 255.0, dtype=np.float32)

    composed = _blend_color(base, pos_alpha, _color_rgb(positive_color))
    composed = _blend_color(composed, neg_alpha, _color_rgb(negative_color))
    image = Image.fromarray(np.clip(composed, 0.0, 255.0).astype(np.uint8), mode="RGB")
    return ImageOps.pad(image, (int(target_size), int(target_size)), color=(255, 255, 255) if background == "white" else (0, 0, 0))


def _compose_triptych(
    *,
    row: dict[str, Any],
    start_frame: Image.Image,
    event_plane: Image.Image,
    end_frame: Image.Image,
) -> Image.Image:
    frame_size = (220, 156)
    plane_size = (320, 320)
    start_tile = ImageOps.pad(start_frame.convert("RGB"), frame_size, color=(236, 236, 236))
    end_tile = ImageOps.pad(end_frame.convert("RGB"), frame_size, color=(236, 236, 236))
    start_tile = ImageOps.expand(start_tile, border=3, fill=(74, 126, 83))
    end_tile = ImageOps.expand(end_tile, border=3, fill=(128, 79, 54))
    plane_tile = ImageOps.pad(event_plane.convert("RGB"), plane_size, color=(255, 255, 255))
    plane_tile = ImageOps.expand(plane_tile, border=3, fill=(74, 86, 112))

    gap = 18
    title_h = 38
    footer_h = 24
    width = start_tile.width + plane_tile.width + end_tile.width + gap * 4
    height = title_h + max(start_tile.height, plane_tile.height, end_tile.height) + footer_h
    canvas = Image.new("RGB", (width, height), color=(245, 241, 235))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 10),
        (
            f"Interval {int(row['interval_index']):02d} | {int(row['interval_us']) / 1000.0:.1f} ms | "
            f"{row['alignment_mode']}"
        ),
        fill=(34, 31, 29),
    )
    y = title_h
    x = gap
    canvas.paste(start_tile, (x, y))
    x += start_tile.width + gap
    canvas.paste(plane_tile, (x, y))
    x += plane_tile.width + gap
    canvas.paste(end_tile, (x, y))
    draw.text((gap, height - 18), "Start Frame", fill=(48, 72, 56))
    draw.text((gap + start_tile.width + gap + 8, height - 18), "Event Plane", fill=(56, 64, 94))
    draw.text((width - end_tile.width - gap, height - 18), "End Frame", fill=(92, 58, 42))
    return canvas


def _build_contact_sheet(images: list[Image.Image], *, save_path: Path) -> None:
    if not images:
        return
    thumb_size = (360, 220)
    gap = 18
    cols = 2
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (gap + cols * (thumb_size[0] + gap), gap + rows * (thumb_size[1] + gap)), color=(251, 247, 241))
    for idx, image in enumerate(images):
        tile = ImageOps.pad(image, thumb_size, color=(235, 231, 224))
        x = gap + (idx % cols) * (thumb_size[0] + gap)
        y = gap + (idx // cols) * (thumb_size[1] + gap)
        canvas.paste(tile, (x, y))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def main() -> None:
    args = build_argparser().parse_args()
    alignment_root = _resolve_path(args.alignment_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    quick_preview_root = output_root.parent / "event_plane_quick_preview"
    quick_preview_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = _read_aggregate_rows(alignment_root)
    mode_examples = _find_mode_examples(aggregate_rows)
    rendered_examples: dict[str, dict[str, str]] = {}

    representative_root = output_root / "representatives"
    representative_root.mkdir(parents=True, exist_ok=True)
    for mode, row in mode_examples.items():
        arrays = _load_interval_arrays(row)
        plane = _render_event_plane(
            arrays=arrays,
            width=int(args.sensor_width),
            height=int(args.sensor_height),
            background=str(args.background),
            positive_color=str(args.positive_color),
            negative_color=str(args.negative_color),
            blur_radius=float(args.blur_radius),
            target_size=int(args.target_size),
        )
        plane_path = representative_root / f"{mode}_event_plane.png"
        triptych_path = representative_root / f"{mode}_triptych.png"
        plane.save(plane_path)
        triptych = _compose_triptych(
            row=row,
            start_frame=_load_frame_rgb(Path(str(row["start_frame_path"])).resolve()),
            event_plane=plane,
            end_frame=_load_frame_rgb(Path(str(row["end_frame_path"])).resolve()),
        )
        triptych.save(triptych_path)
        rendered_examples[mode] = {
            "session_key": str(row["session_key"]),
            "event_plane_path": str(plane_path),
            "triptych_path": str(triptych_path),
        }

    session_root = output_root / "sessions" / str(args.session_key).replace("/", "__")
    session_root.mkdir(parents=True, exist_ok=True)
    session_rows = _session_rows(aggregate_rows, session_key=str(args.session_key))
    session_triptychs: list[Image.Image] = []
    for row in session_rows:
        arrays = _load_interval_arrays(row)
        plane = _render_event_plane(
            arrays=arrays,
            width=int(args.sensor_width),
            height=int(args.sensor_height),
            background=str(args.background),
            positive_color=str(args.positive_color),
            negative_color=str(args.negative_color),
            blur_radius=float(args.blur_radius),
            target_size=int(args.target_size),
        )
        interval_idx = int(row["interval_index"])
        plane_path = session_root / f"interval_{interval_idx:03d}_event_plane.png"
        plane.save(plane_path)
        triptych = _compose_triptych(
            row=row,
            start_frame=_load_frame_rgb(Path(str(row["start_frame_path"])).resolve()),
            event_plane=plane,
            end_frame=_load_frame_rgb(Path(str(row["end_frame_path"])).resolve()),
        )
        triptych.save(session_root / f"interval_{interval_idx:03d}_triptych.png")
        session_triptychs.append(triptych)

    contact_sheet_path = session_root / "session_contact_sheet.png"
    _build_contact_sheet(session_triptychs, save_path=contact_sheet_path)

    preview_order = [
        ("interpolated_dense", "01_interpolated_dense_event_plane.png"),
        ("subsampled", "02_subsampled_event_plane.png"),
        ("empty_pad", "03_empty_pad_event_plane.png"),
        ("interp_single", "04_interp_single_event_plane.png"),
    ]
    for mode, filename in preview_order:
        example = rendered_examples.get(mode)
        if example is None:
            continue
        maybe_link_or_copy(Path(example["event_plane_path"]), quick_preview_root / filename, mode=str(args.link_mode), overwrite=True)
    maybe_link_or_copy(contact_sheet_path, quick_preview_root / "05_session_contact_sheet.png", mode=str(args.link_mode), overwrite=True)
    maybe_link_or_copy(session_root / "interval_000_event_plane.png", quick_preview_root / "06_session_interval0_event_plane.png", mode=str(args.link_mode), overwrite=True)

    summary = {
        "experiment": "tsgss_event_plane_preview",
        "alignment_root": str(alignment_root),
        "output_root": str(output_root),
        "session_key": str(args.session_key),
        "background": str(args.background),
        "positive_color": str(args.positive_color),
        "negative_color": str(args.negative_color),
        "blur_radius": float(args.blur_radius),
        "target_size": int(args.target_size),
        "mode_examples": rendered_examples,
        "session_contact_sheet_path": str(contact_sheet_path),
        "quick_preview_root": str(quick_preview_root),
    }
    write_json(summary, output_root / "experiment_summary.json")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# TSGSS Event Plane Preview",
                "",
                "This preview projects aligned events directly onto the sensor plane.",
                "",
                "Color convention:",
                f"- positive: {args.positive_color}",
                f"- negative: {args.negative_color}",
                "",
                "Quick preview files:",
                "- `tsgss/event_plane_quick_preview/01_interpolated_dense_event_plane.png`",
                "- `tsgss/event_plane_quick_preview/02_subsampled_event_plane.png`",
                "- `tsgss/event_plane_quick_preview/03_empty_pad_event_plane.png`",
                "- `tsgss/event_plane_quick_preview/04_interp_single_event_plane.png`",
                "- `tsgss/event_plane_quick_preview/05_session_contact_sheet.png`",
                "- `tsgss/event_plane_quick_preview/06_session_interval0_event_plane.png`",
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
