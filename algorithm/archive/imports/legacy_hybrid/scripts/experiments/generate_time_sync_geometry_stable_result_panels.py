from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace_event_voxel_analysis"
OUTPUT_ROOT = REPO_ROOT / "workspace_figures" / "time_sync_geometry_stable_result_panels"
RAW_DATA_ROOT = REPO_ROOT.parent / "dataset" / "EYE" / "EV_Eye" / "raw_dataset" / "Data_davis"
USER_ID = 1
SESSION_CODE = 201


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _save_png(image: Image.Image, path: Path) -> Path:
    _ensure_dir(path.parent)
    image.save(path)
    return path


def _save_sequence_frames(
    images: list[Image.Image],
    output_dir: Path,
    *,
    alphas: list[float | None] | None = None,
) -> Path:
    if not images:
        raise ValueError("images must not be empty")
    _ensure_dir(output_dir)
    alpha_values = list(alphas) if alphas is not None else [None] * len(images)
    if len(alpha_values) != len(images):
        raise ValueError("alphas length must match images length")

    for idx, (image, alpha) in enumerate(zip(images, alpha_values)):
        if alpha is None:
            suffix = "endpoint"
        else:
            suffix = f"alpha_{alpha:.3f}".replace(".", "p")
        image.save(output_dir / f"{idx:03d}_{suffix}.png")
    return output_dir


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval_stats_by_eye(summary_path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_json(summary_path)
    return {str(row["eye"]): row for row in rows}


def _clamp_crop(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> tuple[int, int, int, int]:
    ix0 = max(0, int(np.floor(x0)))
    iy0 = max(0, int(np.floor(y0)))
    ix1 = min(width, int(np.ceil(x1)))
    iy1 = min(height, int(np.ceil(y1)))
    if ix1 <= ix0:
        ix1 = min(width, ix0 + 1)
    if iy1 <= iy0:
        iy1 = min(height, iy0 + 1)
    return ix0, iy0, ix1, iy1


def _crop_with_bbox_margin(
    image: Image.Image,
    bbox_xywh: list[float],
    *,
    margin_x_ratio: float,
    margin_y_ratio: float,
) -> Image.Image:
    width, height = image.size
    x, y, w, h = [float(v) for v in bbox_xywh]
    mx = w * float(margin_x_ratio)
    my = h * float(margin_y_ratio)
    crop = _clamp_crop(x - mx, y - my, x + w + mx, y + h + my, width, height)
    return image.crop(crop)


def _crop_sparse_event_render(
    image: Image.Image,
    *,
    threshold: int = 248,
    pad_x: int = 26,
    pad_y: int = 18,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    non_bg = np.any(rgb < int(threshold), axis=2)
    ys, xs = np.nonzero(non_bg)
    if xs.size == 0 or ys.size == 0:
        return image.copy()

    # Ignore a tiny fraction of isolated outlier points so the crop follows the main event cloud.
    width, height = image.size
    x0 = float(np.percentile(xs, 1.5))
    x1 = float(np.percentile(xs, 98.5))
    y0 = float(np.percentile(ys, 2.0))
    y1 = float(np.percentile(ys, 98.0))
    crop = _clamp_crop(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, width, height)
    return image.crop(crop)


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    src_w, src_h = image.size
    if src_h == int(target_height):
        return image
    scale = float(target_height) / float(src_h)
    target_width = max(1, int(round(src_w * scale)))
    return image.resize((target_width, int(target_height)), Image.Resampling.LANCZOS)


def _add_border(image: Image.Image, *, border: int = 2, color: str = "#D8D8D8") -> Image.Image:
    return ImageOps.expand(image, border=border, fill=color)


def _colored_border(image: Image.Image, *, border: int = 3, color: str = "#3C63FF") -> Image.Image:
    return ImageOps.expand(image, border=border, fill=color)


def _horizontal_panel(
    images: list[Image.Image],
    *,
    padding: int = 24,
    gap: int = 28,
    background: str = "white",
) -> Image.Image:
    if not images:
        raise ValueError("images must not be empty")
    max_height = max(img.height for img in images)
    resized = [_resize_to_height(img, max_height) for img in images]
    total_width = padding * 2 + gap * (len(resized) - 1) + sum(img.width for img in resized)
    canvas = Image.new("RGB", (total_width, max_height + padding * 2), background)
    x = padding
    for img in resized:
        y = padding + (max_height - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + gap
    return canvas


def _grid_panel(
    rows: list[list[Image.Image]],
    *,
    padding: int = 24,
    gap_x: int = 28,
    gap_y: int = 28,
    background: str = "white",
) -> Image.Image:
    if not rows or not rows[0]:
        raise ValueError("rows must not be empty")

    row_heights = [max(img.height for img in row) for row in rows]
    col_count = max(len(row) for row in rows)
    col_widths: list[int] = []
    for col_idx in range(col_count):
        col_widths.append(max((row[col_idx].width for row in rows if col_idx < len(row)), default=0))

    total_width = padding * 2 + gap_x * (col_count - 1) + sum(col_widths)
    total_height = padding * 2 + gap_y * (len(rows) - 1) + sum(row_heights)
    canvas = Image.new("RGB", (total_width, total_height), background)

    y = padding
    for row_idx, row in enumerate(rows):
        x = padding
        row_height = row_heights[row_idx]
        for col_idx in range(col_count):
            if col_idx < len(row):
                img = row[col_idx]
                col_width = col_widths[col_idx]
                paste_x = x + (col_width - img.width) // 2
                paste_y = y + (row_height - img.height) // 2
                canvas.paste(img, (paste_x, paste_y))
            x += col_widths[col_idx] + gap_x
        y += row_height + gap_y
    return canvas


def _labeled_panel(image: Image.Image, label: str) -> Image.Image:
    header_h = 34
    canvas = Image.new("RGB", (image.width, image.height + header_h), "white")
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), label, fill="#222222")
    return canvas


def _session_dir_name(session_code: int) -> str:
    digits = str(int(session_code)).zfill(3)
    return f"session_{digits[0]}_{digits[1]}_{digits[2]}"


def _frame_sources(interval_index: int) -> dict[str, Path]:
    return {
        "left": WORKSPACE_ROOT
        / f"interval_{interval_index}_frame_eye_pupil_mask_overlay_blue"
        / f"user01_left_session201_interval{interval_index}_frame_eye_pupil_mask_overlay_blue.png",
        "right": WORKSPACE_ROOT
        / f"interval_{interval_index}_frame_eye_pupil_mask_overlay_blue"
        / f"user01_right_session201_interval{interval_index}_frame_eye_pupil_mask_overlay_blue.png",
    }


def _event_3d_sources(interval_index: int) -> dict[str, Path]:
    return {
        "left": WORKSPACE_ROOT
        / "user01_session201_interval_events_axis_swapped_views"
        / "left"
        / "renders"
        / f"interval_{interval_index:04d}.png",
        "right": WORKSPACE_ROOT
        / "user01_session201_interval_events_axis_swapped_views"
        / "right"
        / "renders"
        / f"interval_{interval_index:04d}.png",
    }


def _positive_accum_sources(interval_index: int) -> dict[str, Path]:
    return {
        "left": WORKSPACE_ROOT
        / "user01_session201_interval_events_accumulated_channels"
        / "left"
        / "positive_images"
        / f"user01_session201_left_interval_{interval_index}_positive_accum.png",
        "right": WORKSPACE_ROOT
        / "user01_session201_interval_events_accumulated_channels"
        / "right"
        / "positive_images"
        / f"user01_session201_right_interval_{interval_index}_positive_accum.png",
    }


def _raw_frame_pair_sources(interval_index: int) -> dict[str, dict[str, Path]]:
    stats = _interval_stats_by_eye(WORKSPACE_ROOT / f"interval_{interval_index}_accumulated_channels" / "summary.json")
    session_dir = _session_dir_name(SESSION_CODE)
    raw_user = f"user{int(USER_ID)}"
    out: dict[str, dict[str, Path]] = {}
    for eye, row in stats.items():
        base = RAW_DATA_ROOT / raw_user / eye / session_dir / "frames"
        out[eye] = {
            "frame_from": base / str(row["frame_from"]),
            "frame_to": base / str(row["frame_to"]),
        }
    return out


def _blend_images(image0: Image.Image, image1: Image.Image, *, alpha: float) -> Image.Image:
    t = float(np.clip(alpha, 0.0, 1.0))
    arr0 = np.asarray(image0.convert("RGB"), dtype=np.float32)
    arr1 = np.asarray(image1.convert("RGB"), dtype=np.float32)
    blended = ((1.0 - t) * arr0 + t * arr1).round().clip(0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _alpha_schedule(insert_count: int) -> list[float]:
    count = max(int(insert_count), 0)
    if count <= 0:
        return []
    denom = float(count + 1)
    return [float((idx + 1) / denom) for idx in range(count)]


def _sequence_stack(
    images: list[Image.Image],
    *,
    tile_height: int,
    x_step: int,
    y_step: int,
    border_color: str,
    background: str = "white",
    padding: int = 28,
) -> Image.Image:
    if not images:
        raise ValueError("images must not be empty")

    tiles = [_colored_border(_resize_to_height(img, tile_height), border=3, color=border_color) for img in images]
    x_offsets = [idx * int(x_step) for idx in range(len(tiles))]
    y_offsets = [idx * int(y_step) for idx in range(len(tiles))]
    min_y = min(y_offsets)
    max_y = max(y + img.height for y, img in zip(y_offsets, tiles))
    total_width = padding * 2 + max(x + img.width for x, img in zip(x_offsets, tiles))
    total_height = padding * 2 + (max_y - min_y)

    canvas = Image.new("RGB", (total_width, total_height), background)
    for x, y, img in zip(x_offsets, y_offsets, tiles):
        paste_x = padding + x
        paste_y = padding + (y - min_y)
        canvas.paste(img, (paste_x, paste_y))
    return canvas


def _render_interval(interval_index: int, output_root: Path) -> list[Path]:
    frame_stats = _interval_stats_by_eye(
        WORKSPACE_ROOT / f"interval_{interval_index}_accumulated_channels_with_rois" / "summary.json"
    )
    frame_sources = _frame_sources(interval_index)
    raw_frame_sources = _raw_frame_pair_sources(interval_index)
    event_3d_sources = _event_3d_sources(interval_index)
    positive_accum_sources = _positive_accum_sources(interval_index)

    outputs: list[Path] = []
    frame_crops: dict[str, Image.Image] = {}
    event_3d_crops: dict[str, Image.Image] = {}
    accum_crops: dict[str, Image.Image] = {}

    for eye in ("left", "right"):
        eye_bbox = list(frame_stats[eye]["eye_bbox_xywh_sensor"])

        frame_img = _load_rgb(frame_sources[eye])
        frame_crop = _add_border(
            _crop_with_bbox_margin(frame_img, eye_bbox, margin_x_ratio=0.10, margin_y_ratio=0.34)
        )
        frame_crops[eye] = frame_crop
        outputs.append(_save_png(frame_crop, output_root / f"frame_interpolation_result_{eye}.png"))

        event_3d_img = _load_rgb(event_3d_sources[eye])
        event_3d_crop = _add_border(_crop_sparse_event_render(event_3d_img, threshold=248, pad_x=28, pad_y=18))
        event_3d_crops[eye] = event_3d_crop
        outputs.append(_save_png(event_3d_crop, output_root / f"event_interpolation_result_3d_{eye}.png"))

        accum_img = _load_rgb(positive_accum_sources[eye])
        accum_crop = _add_border(
            _crop_with_bbox_margin(accum_img, eye_bbox, margin_x_ratio=0.10, margin_y_ratio=0.24)
        )
        accum_crops[eye] = accum_crop
        outputs.append(_save_png(accum_crop, output_root / f"event_accumulation_result_{eye}.png"))

        three_d_plus_accum = _horizontal_panel([event_3d_crop, accum_crop], padding=20, gap=24)
        event_3d_crops[f"{eye}_combo"] = three_d_plus_accum
        outputs.append(_save_png(three_d_plus_accum, output_root / f"event_interpolation_result_3d_plus_accumulate_{eye}.png"))

        raw_frame0 = _load_rgb(raw_frame_sources[eye]["frame_from"])
        raw_frame1 = _load_rgb(raw_frame_sources[eye]["frame_to"])
        raw_crop0 = _crop_with_bbox_margin(raw_frame0, eye_bbox, margin_x_ratio=0.12, margin_y_ratio=0.30)
        raw_crop1 = _crop_with_bbox_margin(raw_frame1, eye_bbox, margin_x_ratio=0.12, margin_y_ratio=0.30)

        source_stack = _sequence_stack(
            [raw_crop0, raw_crop1],
            tile_height=132,
            x_step=120,
            y_step=-18,
            border_color="#335CFF",
        )
        outputs.append(_save_png(source_stack, output_root / f"frame_to_frame_source_{eye}.png"))

        dense_frames = [raw_crop0]
        dense_alphas = _alpha_schedule(6)
        dense_frames.extend(_blend_images(raw_crop0, raw_crop1, alpha=alpha) for alpha in dense_alphas)
        dense_frames.append(raw_crop1)
        dense_frame_dir = output_root / f"frame_to_frame_dense_{eye}_frames"
        dense_frame_alphas: list[float | None] = [None, *dense_alphas, None]
        outputs.append(_save_sequence_frames(dense_frames, dense_frame_dir, alphas=dense_frame_alphas))
        dense_stack = _sequence_stack(
            dense_frames,
            tile_height=112,
            x_step=72,
            y_step=-8,
            border_color="#86A5FF",
        )
        outputs.append(_save_png(dense_stack, output_root / f"frame_to_frame_dense_{eye}.png"))

        before_after = _horizontal_panel(
            [
                _labeled_panel(source_stack, "source frame-to-frame"),
                _labeled_panel(dense_stack, "after interpolation: dense frame-to-frame"),
            ],
            padding=24,
            gap=30,
        )
        outputs.append(_save_png(before_after, output_root / f"frame_interpolation_before_after_{eye}.png"))

    frame_pair = _horizontal_panel(
        [_labeled_panel(frame_crops["left"], "left"), _labeled_panel(frame_crops["right"], "right")],
        padding=28,
        gap=32,
    )
    outputs.append(_save_png(frame_pair, output_root / "frame_interpolation_results_pair.png"))

    event_3d_pair = _horizontal_panel(
        [_labeled_panel(event_3d_crops["left"], "left"), _labeled_panel(event_3d_crops["right"], "right")],
        padding=28,
        gap=32,
    )
    outputs.append(_save_png(event_3d_pair, output_root / "event_interpolation_results_3d_pair.png"))

    event_3d_plus_accum_pair = _grid_panel(
        [
            [_labeled_panel(event_3d_crops["left_combo"], "left")],
            [_labeled_panel(event_3d_crops["right_combo"], "right")],
        ],
        padding=28,
        gap_x=0,
        gap_y=28,
    )
    outputs.append(_save_png(event_3d_plus_accum_pair, output_root / "event_interpolation_results_3d_plus_accumulate_pair.png"))

    frame_before_after_pair = _grid_panel(
        [
            [_labeled_panel(_load_rgb(output_root / "frame_interpolation_before_after_left.png"), "left")],
            [_labeled_panel(_load_rgb(output_root / "frame_interpolation_before_after_right.png"), "right")],
        ],
        padding=28,
        gap_x=0,
        gap_y=24,
    )
    outputs.append(_save_png(frame_before_after_pair, output_root / "frame_interpolation_before_after_pair.png"))
    return outputs


def _write_readme(output_root: Path, interval_index: int, outputs: list[Path]) -> Path:
    readme_path = output_root / "README.md"
    lines = [
        "# Time-Synchronous Geometry-Stable Result Panels",
        "",
        f"- interval: `{interval_index}`",
        "- example: `user01 / session_201`",
        "",
        "Figure-ready outputs:",
        "",
    ]
    for path in outputs:
        suffix = "/" if path.is_dir() else ""
        lines.append(f"- `{path.name}{suffix}`")
    lines += [
        "",
        "Source assets:",
        "- frame interpolation result base: `interval_5012_frame_eye_pupil_mask_overlay_blue`",
        "- raw frame pair base: `/dataset/EYE/EV_Eye/raw_dataset/Data_davis/user1/*/session_2_0_1/frames`",
        "- event 3D base: `user01_session201_interval_events_axis_swapped_views`",
        "- accumulation base: `user01_session201_interval_events_accumulated_channels`",
        "",
        "Notes:",
        "- frame and accumulation panels are cropped around the eye ROI with a small margin",
        "- frame-to-frame interpolation previews use fixed-crop raw frame pairs around interval 5012",
        "- dense frame-to-frame previews are generated with linear blending because a TimeLens checkpoint is not present in this workspace",
        "- dense per-frame outputs are stored under `frame_to_frame_dense_left_frames/` and `frame_to_frame_dense_right_frames/`",
        "- event 3D panels are cropped from the sparse axis-swapped render while ignoring isolated outlier pixels",
        "- pair sheets keep `left` first, then `right`",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def main() -> None:
    interval_index = 5012
    output_root = _ensure_dir(OUTPUT_ROOT)
    outputs = _render_interval(interval_index=interval_index, output_root=output_root)
    readme = _write_readme(output_root=output_root, interval_index=interval_index, outputs=outputs)
    for path in outputs + [readme]:
        print(path)


if __name__ == "__main__":
    main()
