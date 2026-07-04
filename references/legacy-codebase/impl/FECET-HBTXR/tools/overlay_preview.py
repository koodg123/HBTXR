from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from _bootstrap import PROJECT_ROOT

from fecet_hbtxr.io import ensure_dir, read_json, read_jsonl, resolve_stored_path, write_json


def _sanitize_session_key(session_key: str) -> str:
    return session_key.replace("/", "_").replace("\\", "_")


def _load_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return ImageFont.load_default()


def _ellipse_points(xywht: list[float], *, steps: int = 72) -> list[tuple[float, float]]:
    cx, cy, width, height, theta = [float(value) for value in xywht]
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rx = max(width / 2.0, 1e-6)
    ry = max(height / 2.0, 1e-6)
    points: list[tuple[float, float]] = []
    for index in range(steps + 1):
        alpha = 2.0 * math.pi * index / steps
        px = rx * math.cos(alpha)
        py = ry * math.sin(alpha)
        x = cx + px * cos_t - py * sin_t
        y = cy + px * sin_t + py * cos_t
        points.append((x, y))
    return points


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: tuple[int, int, int, int], width: int = 2) -> None:
    x, y, w, h = [float(value) for value in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _render_mask_overlay(base: Image.Image, mask_path: Path, roi_xywh: list[float]) -> Image.Image:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask = Image.open(mask_path).convert("L")
    x, y, _, _ = [int(round(float(value))) for value in roi_xywh]
    tint = Image.new("RGBA", mask.size, (48, 210, 110, 96))
    overlay.paste(tint, (x, y), mask=mask)
    return overlay


def _annotation_text(annotation: dict[str, Any]) -> str:
    lines = [
        str(annotation.get("session_key", "unknown_session")),
        f"user={annotation.get('user_id', annotation.get('subject_id', '?'))} eye={annotation.get('eye', '?')} session={annotation.get('session_code', '?')}",
        f"frame={annotation.get('frame_filename', '?')} ts={annotation.get('timestamp_us', '?')}",
        f"source={annotation.get('annotation_source', '?')} quality={float(annotation.get('annotation_quality', 0.0)):.3f}",
        f"closed={bool(annotation.get('closed_eye_flag', False))} mask_valid={bool(annotation.get('mask_valid', True))}",
    ]
    return "\n".join(lines)


def render_annotation_overlay(
    *,
    canonical_root: Path,
    annotation: dict[str, Any],
    tile_width: int,
    show_mask: bool,
) -> Image.Image:
    frame_path = resolve_stored_path(canonical_root, annotation["frame_path"])
    image = Image.open(frame_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font()

    roi_xywh = [float(value) for value in annotation.get("eye_region_bbox_xywh_sensor", annotation.get("eye_region_xywh", [0, 0, image.width, image.height]))]
    pupil_bbox = [float(value) for value in annotation.get("pupil_region_bbox_xywh_sensor", roi_xywh)]
    ellipse = [float(value) for value in annotation.get("ellipse_sensor_xywht", annotation.get("pupil_ellipse_xywht_sensor", annotation.get("ellipse_xywht", [])))]

    if show_mask and annotation.get("pupil_mask_path"):
        mask_path = resolve_stored_path(canonical_root, annotation["pupil_mask_path"])
        if mask_path.exists():
            overlay = Image.alpha_composite(overlay, _render_mask_overlay(image, mask_path, roi_xywh))
            draw = ImageDraw.Draw(overlay)

    _draw_xywh(draw, roi_xywh, outline=(255, 196, 0, 255), width=2)
    _draw_xywh(draw, pupil_bbox, outline=(0, 196, 255, 255), width=2)
    if len(ellipse) == 5:
        draw.line(_ellipse_points(ellipse), fill=(255, 64, 64, 255), width=3)
        cx, cy = ellipse[0], ellipse[1]
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(255, 255, 255, 255))

    text = _annotation_text(annotation)
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    text_box = (
        8,
        8,
        8 + (text_bbox[2] - text_bbox[0]) + 12,
        8 + (text_bbox[3] - text_bbox[1]) + 12,
    )
    draw.rounded_rectangle(text_box, radius=6, fill=(0, 0, 0, 168))
    draw.multiline_text((14, 14), text, fill=(255, 255, 255, 255), font=font, spacing=2)

    composed = Image.alpha_composite(image, overlay).convert("RGB")
    if tile_width > 0 and composed.width != tile_width:
        scale = float(tile_width) / float(composed.width)
        tile_height = max(1, int(round(composed.height * scale)))
        composed = composed.resize((tile_width, tile_height), Image.Resampling.BILINEAR)
    return composed


def compose_session_sheet(
    *,
    session_key: str,
    rendered_tiles: list[Image.Image],
    columns: int,
) -> Image.Image:
    if not rendered_tiles:
        raise ValueError("rendered_tiles must not be empty")
    font = _load_font()
    columns = max(1, min(int(columns), len(rendered_tiles)))
    rows = int(math.ceil(len(rendered_tiles) / columns))
    pad = 12
    header_h = 44
    tile_width, tile_height = rendered_tiles[0].size
    sheet_width = columns * tile_width + (columns + 1) * pad
    sheet_height = header_h + rows * tile_height + (rows + 1) * pad
    canvas = Image.new("RGB", (sheet_width, sheet_height), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    title = f"{session_key} | random overlay preview | samples={len(rendered_tiles)}"
    draw.text((pad, 12), title, fill=(255, 255, 255), font=font)
    for index, tile in enumerate(rendered_tiles):
        row = index // columns
        col = index % columns
        x = pad + col * (tile_width + pad)
        y = header_h + pad + row * (tile_height + pad)
        canvas.paste(tile, (x, y))
    return canvas


def _discover_session_rows(canonical_root: Path, indexes_root: Path | None) -> list[dict[str, Any]]:
    if indexes_root is None:
        indexes_root = canonical_root / "indexes"
    sessions_index = indexes_root / "sessions.jsonl"
    if sessions_index.exists():
        return [row for row in read_jsonl(sessions_index) if not bool(row.get("skipped", False))]

    rows: list[dict[str, Any]] = []
    for package_path in sorted(canonical_root.glob("**/labels/session_package.json")):
        package = read_json(package_path)
        rows.append(
            {
                "session_key": str(package.get("session_key")),
                "subject_id": int(package.get("subject_id", package.get("user_id", -1))),
                "user_id": int(package.get("user_id", package.get("subject_id", -1))),
                "eye": str(package.get("eye", "left")),
                "session_code": str(package.get("session_code", "")),
                "annotation_store_path": str(package.get("annotation_store_path", package_path.parent / "frame_annotations.jsonl")),
            }
        )
    return rows


def build_overlay_preview(
    *,
    canonical_root: str | Path,
    indexes_root: str | Path | None,
    output_dir: str | Path,
    samples_per_session: int,
    seed: int,
    users: list[int],
    eyes: list[str],
    session_keys: list[str],
    max_sessions: int | None,
    tile_width: int,
    columns: int,
    show_mask: bool,
) -> dict[str, Any]:
    canonical_root = Path(canonical_root).resolve()
    indexes_root = None if indexes_root is None else Path(indexes_root).resolve()
    output_dir = ensure_dir(Path(output_dir).resolve())
    rng = random.Random(int(seed))

    include_users = {int(value) for value in users} if users else None
    include_eyes = {str(value) for value in eyes} if eyes else None
    include_session_keys = {str(value) for value in session_keys} if session_keys else None

    session_rows = _discover_session_rows(canonical_root, indexes_root)
    filtered_rows: list[dict[str, Any]] = []
    for row in session_rows:
        if include_users is not None and int(row.get("user_id", row.get("subject_id", -1))) not in include_users:
            continue
        if include_eyes is not None and str(row.get("eye", "")) not in include_eyes:
            continue
        if include_session_keys is not None and str(row.get("session_key", "")) not in include_session_keys:
            continue
        filtered_rows.append(row)

    filtered_rows.sort(key=lambda row: (int(row.get("user_id", row.get("subject_id", -1))), str(row.get("session_key", ""))))
    if max_sessions is not None:
        filtered_rows = filtered_rows[: max(0, int(max_sessions))]

    session_summaries: list[dict[str, Any]] = []
    for row in filtered_rows:
        annotation_store_path = resolve_stored_path(canonical_root, row["annotation_store_path"])
        annotations = read_jsonl(annotation_store_path)
        if not annotations:
            continue
        sample_count = min(max(1, int(samples_per_session)), len(annotations))
        selected_annotations = rng.sample(annotations, k=sample_count)
        selected_annotations.sort(key=lambda item: (int(item.get("frame_idx") or -1), int(item.get("timestamp_us") or 0)))

        rendered_tiles = [
            render_annotation_overlay(
                canonical_root=canonical_root,
                annotation=annotation,
                tile_width=tile_width,
                show_mask=show_mask,
            )
            for annotation in selected_annotations
        ]
        sheet = compose_session_sheet(
            session_key=str(row["session_key"]),
            rendered_tiles=rendered_tiles,
            columns=columns,
        )
        output_path = output_dir / f"{_sanitize_session_key(str(row['session_key']))}_overlay.png"
        sheet.save(output_path)
        session_summaries.append(
            {
                "session_key": str(row["session_key"]),
                "user_id": int(row.get("user_id", row.get("subject_id", -1))),
                "eye": str(row.get("eye", "")),
                "selected_count": len(selected_annotations),
                "selected_ann_ids": [str(annotation.get("ann_id", "")) for annotation in selected_annotations],
                "output": str(output_path),
            }
        )

    summary = {
        "canonical_root": str(canonical_root),
        "output_dir": str(output_dir),
        "samples_per_session": int(samples_per_session),
        "seed": int(seed),
        "session_count": len(session_summaries),
        "rows": session_summaries,
    }
    write_json(summary, output_dir / "overlay_preview_summary.json")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build per-session random overlay previews with annotation overlays")
    parser.add_argument("--canonical-root", type=str, default=str(PROJECT_ROOT / "data" / "_internal" / "canonical"))
    parser.add_argument("--indexes-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "runs" / "overlay_preview"))
    parser.add_argument("--samples-per-session", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user", action="append", type=int, default=[])
    parser.add_argument("--eye", action="append", choices=["left", "right"], default=[])
    parser.add_argument("--session-key", action="append", default=[])
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--tile-width", type=int, default=512)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--hide-mask", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    return build_overlay_preview(
        canonical_root=args.canonical_root,
        indexes_root=args.indexes_root,
        output_dir=args.output_dir,
        samples_per_session=args.samples_per_session,
        seed=args.seed,
        users=args.user,
        eyes=args.eye,
        session_keys=args.session_key,
        max_sessions=args.max_sessions,
        tile_width=args.tile_width,
        columns=args.columns,
        show_mask=not bool(args.hide_mask),
    )


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
