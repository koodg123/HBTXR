#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import code_to_session_dir
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.utils.io import read_jsonl
from hbtxr.utils.paths import resolve_stored_path


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _draw_rotated_ellipse(
    draw: ImageDraw.ImageDraw,
    ellipse_xywht: list[float],
    *,
    outline: str,
    width: int = 2,
) -> None:
    cx, cy, ew, eh, theta = [float(v) for v in ellipse_xywht]
    points = []
    for idx in range(72):
        ang = (2.0 * math.pi * idx) / 72.0
        px = 0.5 * ew * math.cos(ang)
        py = 0.5 * eh * math.sin(ang)
        x = cx + px * math.cos(theta) - py * math.sin(theta)
        y = cy + px * math.sin(theta) + py * math.cos(theta)
        points.append((x, y))
    if points:
        draw.line(points + [points[0]], fill=outline, width=width)


def _overlay_mask(base: Image.Image, mask_path: Path | None, *, color: str = "#ff5050", alpha: int = 96) -> Image.Image:
    image = base.convert("RGBA")
    if mask_path is None or not mask_path.exists():
        return image.convert("RGB")
    mask = Image.open(mask_path).convert("L")
    rgba = Image.new("RGBA", image.size, color=ImageColor.getrgb(color) + (0,))
    rgba.putalpha(mask.point(lambda px: alpha if px > 0 else 0))
    return Image.alpha_composite(image, rgba).convert("RGB")


def _resolve_raw_frame_path(raw_root: Path, row: dict) -> Path:
    user_id = int(row["user_id"])
    eye = str(row["eye"])
    session_code = str(row["session_code"])
    frame_filename = str(row["frame_filename"])
    return raw_root / f"user{user_id}" / eye / code_to_session_dir(session_code) / "frames" / frame_filename


def _annotated_panel(frame_path: Path, row: dict, mask_path: Path | None) -> Image.Image:
    base = Image.open(frame_path).convert("RGB")
    overlay = _overlay_mask(base, mask_path)
    draw = ImageDraw.Draw(overlay)
    eye_region = row.get("eye_region_bbox_xywh_sensor") or row.get("eye_region_xywh")
    pupil_region = row.get("pupil_region_bbox_xywh_sensor")
    ellipse = row.get("pupil_ellipse_xywht_sensor") or row.get("ellipse_sensor_xywht")
    if eye_region:
        _draw_xywh(draw, eye_region, outline="#00ff88", width=2)
    if pupil_region:
        _draw_xywh(draw, pupil_region, outline="#ffd54f", width=2)
    if ellipse:
        _draw_rotated_ellipse(draw, ellipse, outline="#ff4040", width=2)

    caption_parts = [
        str(row.get("session_key", "unknown")),
        str(row.get("frame_filename", "n/a")),
        f"class={row.get('gsam_class_name', 'n/a')}",
    ]
    if row.get("mask_area_ratio") is not None:
        caption_parts.append(f"mask={float(row['mask_area_ratio']):.3f}")
    if row.get("gsam_box_confidence") is not None:
        caption_parts.append(f"box={float(row['gsam_box_confidence']):.2f}")
    if row.get("gsam_mask_score") is not None:
        caption_parts.append(f"sam={float(row['gsam_mask_score']):.2f}")
    draw.text((6, 6), " | ".join(caption_parts), fill=(255, 255, 255))
    return overlay


def _make_contact_sheet(panels: list[Image.Image], *, cols: int) -> Image.Image:
    if not panels:
        raise ValueError("panels are required")
    cols = max(1, int(cols))
    width, height = panels[0].size
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), color=(0, 0, 0))
    for index, panel in enumerate(panels):
        x = (index % cols) * width
        y = (index // cols) * height
        sheet.paste(panel, (x, y))
    return sheet


def _iter_annotation_stores(annotation_root: Path) -> list[Path]:
    return sorted(annotation_root.glob("sessions/user*/left/session_*/frame_annotations.jsonl")) + sorted(
        annotation_root.glob("sessions/user*/right/session_*/frame_annotations.jsonl")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-session spot-check contact sheets directly from Grounded-SAM annotation exports")
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--count-per-session", type=int, default=64)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    if paths.annotation_root is None:
        raise ValueError("annotation_root is required. Pass --annotation-root or provide it in the paths config.")
    annotation_root = paths.annotation_root.resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (paths.preview_root / "annotation_spotcheck").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    store_paths = _iter_annotation_stores(annotation_root)
    if args.max_sessions is not None:
        store_paths = store_paths[: max(1, int(args.max_sessions))]

    rng = random.Random(int(args.seed))
    summary_rows = []
    for store_path in store_paths:
        rows = read_jsonl(store_path)
        if not rows:
            continue
        picked = rows if len(rows) <= int(args.count_per_session) else rng.sample(rows, int(args.count_per_session))
        panels = []
        session_key = str(picked[0].get("session_key", store_path.parent.name))
        for row in picked:
            frame_path = _resolve_raw_frame_path(paths.raw_root, row)
            if not frame_path.exists():
                continue
            stored_mask = row.get("pupil_mask_path") or row.get("mask_path")
            mask_path = resolve_stored_path(annotation_root, stored_mask) if stored_mask else None
            panels.append(_annotated_panel(frame_path, row, mask_path))
        if not panels:
            continue
        sheet = _make_contact_sheet(panels, cols=int(args.cols))
        out_name = f"{session_key.replace('/', '__')}__spotcheck.png"
        sheet.save(output_dir / out_name)
        summary_rows.append(
            {
                "session_key": session_key,
                "annotation_store_path": str(store_path),
                "n_rows_total": len(rows),
                "n_panels": len(panels),
                "output_path": str(output_dir / out_name),
            }
        )

    summary = {
        "raw_root": str(paths.raw_root),
        "annotation_root": str(annotation_root),
        "output_dir": str(output_dir),
        "count_per_session": int(args.count_per_session),
        "max_sessions": args.max_sessions,
        "seed": int(args.seed),
        "n_sessions": len(summary_rows),
    }
    (output_dir / "spotcheck_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "spotcheck_sessions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summary_rows),
        encoding="utf-8",
    )
    print(f"[DONE] sessions={len(summary_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
