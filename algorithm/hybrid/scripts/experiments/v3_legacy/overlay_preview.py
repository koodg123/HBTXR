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

from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.utils.io import read_jsonl
from hbtxr.utils.paths import resolve_canonical_dataset_root, resolve_canonical_indexes_root, resolve_stored_path


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _draw_rotated_ellipse(draw: ImageDraw.ImageDraw, ellipse_xywht: list[float], *, outline: str, width: int = 2) -> None:
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
    caption = f"{row.get('session_key', 'unknown')} | {row.get('ann_id', 'n/a')}"
    draw.text((6, 6), caption, fill=(255, 255, 255))
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create random per-session overlay previews from canonical annotations")
    add_common_path_args(parser, need_raw=False, need_canonical=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["mode0", "mode1", "mode2"], default="mode1")
    parser.add_argument("--canonical-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args, need_raw=False, need_canonical=True)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (paths.preview_root / "overlay_random").resolve()
    mode = str(args.mode).strip().lower()
    canonical_name = str(
        args.canonical_name
        or ("canonical2" if mode == "mode2" else ("canonical0" if mode == "mode0" else "canonical1"))
    )
    indexes_root = resolve_canonical_indexes_root(
        paths.canonical_root,
        canonical_name=canonical_name,
        indexes_root=paths.indexes_root,
        prefer_nested=False,
    )
    session_rows = [row for row in read_jsonl(indexes_root / "sessions.jsonl") if not row.get("skipped")]
    rng = random.Random(int(args.seed))
    rng.shuffle(session_rows)
    chosen_sessions = session_rows[: max(1, int(args.count))]
    output_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    for session in chosen_sessions:
        session_canonical_name = str(session.get("canonical_name", canonical_name))
        session_root = resolve_canonical_dataset_root(paths.canonical_root, session_canonical_name, prefer_nested=False)
        annotation_store = resolve_stored_path(session_root, session["annotation_store_path"])
        rows = read_jsonl(annotation_store)
        if not rows:
            continue
        row = rng.choice(rows)
        row_root = resolve_canonical_dataset_root(
            paths.canonical_root,
            str(row.get("canonical_name", session_canonical_name)),
            prefer_nested=False,
        )
        frame_path = resolve_stored_path(row_root, row["frame_path"])
        stored_mask = row.get("pupil_mask_path") or row.get("mask_path")
        mask_path = resolve_stored_path(row_root, stored_mask) if stored_mask else None
        panel = _annotated_panel(frame_path, row, mask_path)
        name = f"{row['session_key'].replace('/', '__')}__{Path(row['frame_filename']).stem}.png"
        panel.save(output_dir / name)
        produced += 1

    summary = {
        "canonical_root": str(paths.canonical_root),
        "canonical_name": canonical_name,
        "indexes_root": str(indexes_root),
        "output_dir": str(output_dir),
        "requested_count": int(args.count),
        "produced": int(produced),
        "seed": int(args.seed),
    }
    (output_dir / "overlay_preview_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] produced={produced} output={output_dir}")


if __name__ == "__main__":
    main()
