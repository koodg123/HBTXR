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
    draw.text((6, 6), " | ".join(caption_parts), fill=(255, 255, 255))
    return overlay


def _iter_annotation_stores(annotation_root: Path) -> list[Path]:
    return sorted(annotation_root.glob("sessions/user*/left/session_*/frame_annotations.jsonl")) + sorted(
        annotation_root.glob("sessions/user*/right/session_*/frame_annotations.jsonl")
    )


def _match_row_filters(
    row: dict,
    *,
    allowed_user_ids: set[int],
    allowed_eyes: set[str],
    allowed_session_codes: set[str],
    session_key_contains: str | None,
) -> bool:
    if allowed_user_ids and int(row.get("user_id", -1)) not in allowed_user_ids:
        return False
    if allowed_eyes and str(row.get("eye", "")).lower() not in allowed_eyes:
        return False
    if allowed_session_codes and str(row.get("session_code", "")) not in allowed_session_codes:
        return False
    if session_key_contains and session_key_contains not in str(row.get("session_key", "")):
        return False
    return True


def _pick_rows(
    rows: list[dict],
    *,
    count_per_session: int | None,
    sample_mode: str,
    seed: int,
) -> list[dict]:
    if not rows:
        return []
    if sample_mode == "all" or count_per_session is None or len(rows) <= int(count_per_session):
        return list(rows)
    limit = max(1, int(count_per_session))
    if sample_mode == "first":
        return list(rows[:limit])
    if sample_mode == "random":
        rng = random.Random(int(seed))
        picked = rng.sample(rows, limit)
        return sorted(
            picked,
            key=lambda row: (
                int(row.get("frame_idx") or -1),
                int(row.get("timestamp_us") or 0),
                str(row.get("ann_id") or ""),
            ),
        )
    if sample_mode == "uniform":
        if limit == 1:
            return [rows[0]]
        indices = sorted({int(round(idx * (len(rows) - 1) / (limit - 1))) for idx in range(limit)})
        return [rows[idx] for idx in indices]
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-frame raw overlay previews directly from Grounded-SAM annotation stores"
    )
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--user-ids", type=str, default=None, help="Comma-separated user ids, for example: 1,2")
    parser.add_argument("--eye", type=str, default=None, choices=["left", "right"])
    parser.add_argument("--session-codes", type=str, default=None, help="Comma-separated session codes, for example: 101,102")
    parser.add_argument("--session-key-contains", type=str, default=None)
    parser.add_argument("--count-per-session", type=int, default=16)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--sample-mode", type=str, default="uniform", choices=["uniform", "first", "random", "all"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    if paths.annotation_root is None:
        raise ValueError("annotation_root is required. Pass --annotation-root or provide it in the paths config.")

    annotation_root = paths.annotation_root.resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (paths.preview_root / "raw_overlay_groundedsam").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_user_ids = {int(item) for item in _comma_split(args.user_ids)}
    allowed_eyes = {str(args.eye).lower()} if args.eye else set()
    allowed_session_codes = {str(item) for item in _comma_split(args.session_codes)}
    session_key_contains = str(args.session_key_contains).strip() if args.session_key_contains else None

    store_paths = _iter_annotation_stores(annotation_root)
    if args.max_sessions is not None:
        store_paths = store_paths[: max(1, int(args.max_sessions))]

    summary_rows = []
    total_panels = 0
    for store_path in store_paths:
        rows = read_jsonl(store_path)
        rows = [
            row
            for row in rows
            if _match_row_filters(
                row,
                allowed_user_ids=allowed_user_ids,
                allowed_eyes=allowed_eyes,
                allowed_session_codes=allowed_session_codes,
                session_key_contains=session_key_contains,
            )
        ]
        if not rows:
            continue

        rows.sort(
            key=lambda row: (
                int(row.get("frame_idx") or -1),
                int(row.get("timestamp_us") or 0),
                str(row.get("ann_id") or ""),
            )
        )
        picked = _pick_rows(
            rows,
            count_per_session=None if args.sample_mode == "all" else int(args.count_per_session),
            sample_mode=str(args.sample_mode),
            seed=int(args.seed),
        )
        if not picked:
            continue

        session_key = str(picked[0].get("session_key", store_path.parent.name))
        session_dir = output_dir / session_key
        session_dir.mkdir(parents=True, exist_ok=True)
        exported_files = []
        for row in picked:
            frame_path = _resolve_raw_frame_path(paths.raw_root, row)
            if not frame_path.exists():
                continue
            stored_mask = row.get("pupil_mask_path") or row.get("mask_path")
            mask_path = resolve_stored_path(annotation_root, stored_mask) if stored_mask else None
            panel = _annotated_panel(frame_path, row, mask_path)
            output_name = f"{Path(str(row.get('frame_filename', 'frame'))).stem}__overlay.png"
            output_path = session_dir / output_name
            panel.save(output_path)
            exported_files.append(str(output_path))

        if not exported_files:
            continue

        total_panels += len(exported_files)
        summary_rows.append(
            {
                "session_key": session_key,
                "annotation_store_path": str(store_path),
                "output_dir": str(session_dir),
                "n_rows_total": len(rows),
                "n_exported": len(exported_files),
                "sample_mode": str(args.sample_mode),
                "count_per_session": None if args.sample_mode == "all" else int(args.count_per_session),
                "files": exported_files,
            }
        )

    summary = {
        "raw_root": str(paths.raw_root),
        "annotation_root": str(annotation_root),
        "output_dir": str(output_dir),
        "user_ids": sorted(allowed_user_ids),
        "eye": args.eye,
        "session_codes": sorted(allowed_session_codes),
        "session_key_contains": session_key_contains,
        "count_per_session": None if args.sample_mode == "all" else int(args.count_per_session),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "n_sessions": len(summary_rows),
        "n_panels": int(total_panels),
    }
    (output_dir / "raw_overlay_preview_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "raw_overlay_preview_sessions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summary_rows),
        encoding="utf-8",
    )
    print(f"[DONE] sessions={len(summary_rows)} panels={total_panels} output={output_dir}")


if __name__ == "__main__":
    main()
