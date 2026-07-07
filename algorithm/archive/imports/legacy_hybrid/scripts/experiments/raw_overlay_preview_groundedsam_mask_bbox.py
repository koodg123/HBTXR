#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.annotation_groundedsam import mask_bbox_xywh
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


def _overlay_mask(base: Image.Image, mask_path: Path | None, *, color: str = "#ff5050", alpha: int = 112) -> Image.Image:
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
) -> bool:
    if allowed_user_ids and int(row.get("user_id", -1)) not in allowed_user_ids:
        return False
    if allowed_eyes and str(row.get("eye", "")).lower() not in allowed_eyes:
        return False
    if allowed_session_codes and str(row.get("session_code", "")) not in allowed_session_codes:
        return False
    return True


def _pick_rows(rows: list[dict], *, count_per_session: int | None, sample_mode: str, seed: int) -> list[dict]:
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


def _load_mask_bbox(mask_path: Path | None) -> list[float] | None:
    if mask_path is None or not mask_path.exists():
        return None
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    bbox = mask_bbox_xywh(mask.astype(np.uint8))
    if bbox is None:
        return None
    return [float(v) for v in bbox]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export two raw overlay previews from Grounded-SAM eye masks: mask overlay and mask-derived bbox overlay"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    if paths.annotation_root is None:
        raise ValueError("annotation_root is required. Pass --annotation-root or provide it in the paths config.")

    annotation_root = paths.annotation_root.resolve()
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (paths.preview_root / "raw_overlay_groundedsam_mask_bbox").resolve()
    )
    mask_root = output_root / "mask_overlay"
    bbox_root = output_root / "bbox_overlay"
    mask_root.mkdir(parents=True, exist_ok=True)
    bbox_root.mkdir(parents=True, exist_ok=True)

    allowed_user_ids = {int(item) for item in _comma_split(args.user_ids)}
    allowed_eyes = {str(args.eye).lower()} if args.eye else set()
    allowed_session_codes = {str(item) for item in _comma_split(args.session_codes)}

    store_paths = _iter_annotation_stores(annotation_root)
    if args.max_sessions is not None:
        store_paths = store_paths[: max(1, int(args.max_sessions))]

    session_rows = []
    total_pairs = 0
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
        session_mask_dir = mask_root / session_key
        session_bbox_dir = bbox_root / session_key
        session_mask_dir.mkdir(parents=True, exist_ok=True)
        session_bbox_dir.mkdir(parents=True, exist_ok=True)

        exported_mask_files: list[str] = []
        exported_bbox_files: list[str] = []
        for row in picked:
            frame_path = _resolve_raw_frame_path(paths.raw_root, row)
            if not frame_path.exists():
                continue
            stored_mask = row.get("pupil_mask_path") or row.get("mask_path")
            mask_path = resolve_stored_path(annotation_root, stored_mask) if stored_mask else None
            mask_bbox = _load_mask_bbox(mask_path)

            base = Image.open(frame_path).convert("RGB")
            mask_panel = _overlay_mask(base, mask_path)
            mask_draw = ImageDraw.Draw(mask_panel)
            mask_caption = [
                str(row.get("session_key", "unknown")),
                str(row.get("frame_filename", "n/a")),
                f"class={row.get('gsam_class_name', 'n/a')}",
            ]
            if row.get("mask_area_ratio") is not None:
                mask_caption.append(f"mask={float(row['mask_area_ratio']):.3f}")
            mask_draw.text((6, 6), " | ".join(mask_caption), fill=(255, 255, 255))

            bbox_panel = base.copy()
            bbox_draw = ImageDraw.Draw(bbox_panel)
            if mask_bbox is not None:
                _draw_xywh(bbox_draw, mask_bbox, outline="#00ff88", width=2)
            bbox_caption = [
                str(row.get("session_key", "unknown")),
                str(row.get("frame_filename", "n/a")),
                "bbox=mask",
            ]
            if mask_bbox is not None:
                bbox_caption.append(
                    f"xywh={[round(float(v), 1) for v in mask_bbox]}"
                )
            bbox_draw.text((6, 6), " | ".join(bbox_caption), fill=(255, 255, 255))

            stem = Path(str(row.get("frame_filename", "frame"))).stem
            mask_output = session_mask_dir / f"{stem}__mask_overlay.png"
            bbox_output = session_bbox_dir / f"{stem}__mask_bbox_overlay.png"
            mask_panel.save(mask_output)
            bbox_panel.save(bbox_output)
            exported_mask_files.append(str(mask_output))
            exported_bbox_files.append(str(bbox_output))

        if not exported_mask_files:
            continue

        total_pairs += len(exported_mask_files)
        session_rows.append(
            {
                "session_key": session_key,
                "annotation_store_path": str(store_path),
                "mask_output_dir": str(session_mask_dir),
                "bbox_output_dir": str(session_bbox_dir),
                "n_rows_total": len(rows),
                "n_exported": len(exported_mask_files),
                "sample_mode": str(args.sample_mode),
                "count_per_session": None if args.sample_mode == "all" else int(args.count_per_session),
                "mask_files": exported_mask_files,
                "bbox_files": exported_bbox_files,
            }
        )

    summary = {
        "raw_root": str(paths.raw_root),
        "annotation_root": str(annotation_root),
        "output_root": str(output_root),
        "user_ids": sorted(allowed_user_ids),
        "eye": args.eye,
        "session_codes": sorted(allowed_session_codes),
        "count_per_session": None if args.sample_mode == "all" else int(args.count_per_session),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "n_sessions": len(session_rows),
        "n_pairs": int(total_pairs),
    }
    (output_root / "raw_overlay_groundedsam_mask_bbox_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "raw_overlay_groundedsam_mask_bbox_sessions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in session_rows),
        encoding="utf-8",
    )
    print(f"[DONE] sessions={len(session_rows)} pairs={total_pairs} output={output_root}")


if __name__ == "__main__":
    main()
