#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import (
    canonical_user_name,
    collect_users,
    derive_eye_region,
    discover_session_layout,
    is_official_session_code,
    parse_via_csv_with_report,
    rasterize_ellipse_mask,
)
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths


def _comma_split(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


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


def _overlay_mask_array(base: Image.Image, mask: np.ndarray, *, color: str = "#ff5050", alpha: int = 96) -> Image.Image:
    image = base.convert("RGBA")
    rgba = Image.new("RGBA", image.size, color=ImageColor.getrgb(color) + (0,))
    mask_image = Image.fromarray(np.where(mask > 0, alpha, 0).astype(np.uint8), mode="L")
    rgba.putalpha(mask_image)
    return Image.alpha_composite(image, rgba).convert("RGB")


def _pick_annotations(annotations: list, *, count_per_session: int | None, sample_mode: str, seed: int) -> list:
    if not annotations:
        return []
    if sample_mode == "all" or count_per_session is None or len(annotations) <= int(count_per_session):
        return list(annotations)
    limit = max(1, int(count_per_session))
    if sample_mode == "first":
        return list(annotations[:limit])
    if sample_mode == "random":
        rng = random.Random(int(seed))
        picked = rng.sample(annotations, limit)
        return sorted(picked, key=lambda ann: (int(ann.frame_idx or -1), int(ann.timestamp_us), str(ann.frame_filename)))
    if sample_mode == "uniform":
        if limit == 1:
            return [annotations[0]]
        indices = sorted({int(round(idx * (len(annotations) - 1) / (limit - 1))) for idx in range(limit)})
        return [annotations[idx] for idx in indices]
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def _ellipse_to_bbox(ellipse_xywht: list[float]) -> list[float]:
    cx, cy, ew, eh, _ = [float(v) for v in ellipse_xywht]
    return [cx - 0.5 * ew, cy - 0.5 * eh, ew, eh]


def _iter_raw_sessions(raw_root: Path) -> list[tuple[int, str, Path]]:
    rows = []
    for user_dir in collect_users(raw_root):
        try:
            user_id = int(user_dir.name.replace("user", ""))
        except ValueError:
            continue
        for eye in ("left", "right"):
            eye_dir = user_dir / eye
            if not eye_dir.exists():
                continue
            for session_dir in sorted(eye_dir.glob("session_*_*_*")):
                rows.append((user_id, eye, session_dir))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw-frame overlay previews using the original manual CSV labels from the raw EV-Eye dataset"
    )
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--user-ids", type=str, default=None, help="Comma-separated user ids, for example: 1,2")
    parser.add_argument("--eye", type=str, default=None, choices=["left", "right"])
    parser.add_argument("--session-codes", type=str, default=None, help="Comma-separated session codes, for example: 101,102")
    parser.add_argument("--count-per-session", type=int, default=16)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--sample-mode", type=str, default="uniform", choices=["uniform", "first", "random", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin-px", type=int, default=24)
    parser.add_argument("--margin-left-px", type=int, default=None)
    parser.add_argument("--margin-right-px", type=int, default=None)
    parser.add_argument("--margin-top-px", type=int, default=None)
    parser.add_argument("--margin-bottom-px", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (paths.preview_root / "raw_overlay_manual_csv").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_user_ids = {int(item) for item in _comma_split(args.user_ids)}
    allowed_eyes = {str(args.eye).lower()} if args.eye else set()
    allowed_session_codes = {str(item) for item in _comma_split(args.session_codes)}

    session_rows = []
    processed_sessions = 0
    for user_id, eye, session_dir in _iter_raw_sessions(paths.raw_root):
        layout = discover_session_layout(session_dir, user_id=user_id)
        session_code = layout.session_code
        if allowed_user_ids and int(user_id) not in allowed_user_ids:
            continue
        if allowed_eyes and str(eye).lower() not in allowed_eyes:
            continue
        if allowed_session_codes and str(session_code) not in allowed_session_codes:
            continue
        if args.max_sessions is not None and processed_sessions >= max(1, int(args.max_sessions)):
            break
        processed_sessions += 1

        session_key = f"{canonical_user_name(user_id)}/{eye}/session_{session_code}"
        if layout.frames_dir is None:
            session_rows.append(
                {
                    "session_key": session_key,
                    "raw_session_dir": str(session_dir),
                    "annotation_csv": None,
                    "skipped": True,
                    "message": "frames_dir_missing",
                }
            )
            continue
        if layout.annotation_csv is None:
            session_rows.append(
                {
                    "session_key": session_key,
                    "raw_session_dir": str(session_dir),
                    "annotation_csv": None,
                    "skipped": True,
                    "message": "annotation_csv_missing",
                }
            )
            continue

        frame_paths = sorted(layout.frames_dir.glob("*.png"))
        frame_names = [path.name for path in frame_paths]
        frame_stems = [path.stem for path in frame_paths]
        if not frame_paths:
            session_rows.append(
                {
                    "session_key": session_key,
                    "raw_session_dir": str(session_dir),
                    "annotation_csv": str(layout.annotation_csv),
                    "skipped": True,
                    "message": "no_frame_png_found",
                }
            )
            continue

        frame_width, frame_height = Image.open(frame_paths[0]).size
        annotations, parse_report = parse_via_csv_with_report(
            layout.annotation_csv,
            known_frame_filenames=frame_names,
            known_frame_stems=frame_stems,
        )
        annotations.sort(key=lambda ann: (int(ann.frame_idx or -1), int(ann.timestamp_us), str(ann.frame_filename)))
        picked = _pick_annotations(
            annotations,
            count_per_session=None if args.sample_mode == "all" else int(args.count_per_session),
            sample_mode=str(args.sample_mode),
            seed=int(args.seed),
        )
        if not picked:
            session_rows.append(
                {
                    "session_key": session_key,
                    "raw_session_dir": str(session_dir),
                    "annotation_csv": str(layout.annotation_csv),
                    "skipped": True,
                    "message": "no_annotations_kept",
                    "parse_report": parse_report,
                }
            )
            continue

        eye_region = derive_eye_region(
            picked,
            sensor_size=(int(frame_width), int(frame_height)),
            margin_px=int(args.margin_px),
            margin_left_px=args.margin_left_px,
            margin_right_px=args.margin_right_px,
            margin_top_px=args.margin_top_px,
            margin_bottom_px=args.margin_bottom_px,
        )
        session_output_dir = output_dir / session_key
        session_output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = []
        for ann in picked:
            frame_path = layout.frames_dir / ann.frame_filename
            if not frame_path.exists():
                continue
            base = Image.open(frame_path).convert("RGB")
            mask = rasterize_ellipse_mask(ann.ellipse_xywht, image_size=base.size)
            overlay = _overlay_mask_array(base, mask)
            draw = ImageDraw.Draw(overlay)
            ellipse = [float(v) for v in ann.ellipse_xywht]
            eye_region_xywh = eye_region.to_list()
            pupil_bbox = _ellipse_to_bbox(ellipse)
            _draw_xywh(draw, eye_region_xywh, outline="#00ff88", width=2)
            _draw_xywh(draw, pupil_bbox, outline="#ffd54f", width=2)
            _draw_rotated_ellipse(draw, ellipse, outline="#ff4040", width=2)
            caption = (
                f"{session_key} | {ann.frame_filename} | source=manual_csv | "
                f"official={str(is_official_session_code(session_code)).lower()}"
            )
            draw.text((6, 6), caption, fill=(255, 255, 255))
            output_name = f"{Path(ann.frame_filename).stem}__overlay.png"
            output_path = session_output_dir / output_name
            overlay.save(output_path)
            exported_files.append(str(output_path))

        session_rows.append(
            {
                "session_key": session_key,
                "raw_session_dir": str(session_dir),
                "annotation_csv": str(layout.annotation_csv),
                "skipped": not bool(exported_files),
                "message": "manual_csv_overlay_export_complete" if exported_files else "frames_missing_for_annotations",
                "n_annotations_total": len(annotations),
                "n_exported": len(exported_files),
                "sample_mode": str(args.sample_mode),
                "count_per_session": None if args.sample_mode == "all" else int(args.count_per_session),
                "margin_px": int(args.margin_px),
                "margin_left_px": args.margin_left_px,
                "margin_right_px": args.margin_right_px,
                "margin_top_px": args.margin_top_px,
                "margin_bottom_px": args.margin_bottom_px,
                "parse_report": parse_report,
                "output_dir": str(session_output_dir),
                "files": exported_files,
                "eye_region_xywh": eye_region.to_list(),
            }
        )

    summary = {
        "raw_root": str(paths.raw_root),
        "output_dir": str(output_dir),
        "user_ids": sorted(allowed_user_ids),
        "eye": args.eye,
        "session_codes": sorted(allowed_session_codes),
        "count_per_session": None if args.sample_mode == "all" else int(args.count_per_session),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "margin_px": int(args.margin_px),
        "margin_left_px": args.margin_left_px,
        "margin_right_px": args.margin_right_px,
        "margin_top_px": args.margin_top_px,
        "margin_bottom_px": args.margin_bottom_px,
        "n_sessions_seen": len(session_rows),
        "n_sessions_exported": int(sum(1 for row in session_rows if not row.get("skipped"))),
        "n_panels": int(sum(int(row.get("n_exported", 0)) for row in session_rows)),
    }
    (output_dir / "raw_overlay_manual_csv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "raw_overlay_manual_csv_sessions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in session_rows),
        encoding="utf-8",
    )
    print(
        f"[DONE] sessions_seen={summary['n_sessions_seen']} "
        f"sessions_exported={summary['n_sessions_exported']} "
        f"panels={summary['n_panels']} output={output_dir}"
    )


if __name__ == "__main__":
    main()
