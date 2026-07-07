#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import code_to_session_dir
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.utils.io import read_jsonl


def _comma_split(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _resolve_raw_frame_path(raw_root: Path, row: dict) -> Path:
    user_id = int(row["user_id"])
    eye = str(row["eye"])
    session_code = str(row["session_code"])
    frame_filename = str(row["frame_filename"])
    return raw_root / f"user{user_id}" / eye / code_to_session_dir(session_code) / "frames" / frame_filename


def _resolve_bbox_xywh(row: dict, *, bbox_source: str) -> list[float] | None:
    if bbox_source == "eye_region":
        return row.get("eye_region_bbox_xywh_sensor") or row.get("eye_region_xywh")
    if bbox_source == "detected":
        return row.get("pupil_region_bbox_xywh_sensor")
    class_name = str(row.get("gsam_class_name", "")).lower()
    if class_name == "eye":
        return row.get("pupil_region_bbox_xywh_sensor")
    return row.get("eye_region_bbox_xywh_sensor") or row.get("eye_region_xywh")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw overlay previews showing only Grounded-SAM eye-region bounding boxes")
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--user-ids", type=str, default=None, help="Comma-separated user ids, for example: 1,2")
    parser.add_argument("--eye", type=str, default=None, choices=["left", "right"])
    parser.add_argument("--session-codes", type=str, default=None, help="Comma-separated session codes, for example: 101,102")
    parser.add_argument("--count-per-session", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--bbox-source", type=str, default="auto", choices=["auto", "eye_region", "detected"])
    parser.add_argument("--sample-mode", type=str, default="all", choices=["uniform", "first", "random", "all"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    if paths.annotation_root is None:
        raise ValueError("annotation_root is required. Pass --annotation-root or provide it in the paths config.")

    annotation_root = paths.annotation_root.resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (paths.preview_root / "raw_overlay_groundedsam_eye_region").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_user_ids = {int(item) for item in _comma_split(args.user_ids)}
    allowed_eyes = {str(args.eye).lower()} if args.eye else set()
    allowed_session_codes = {str(item) for item in _comma_split(args.session_codes)}

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
            count_per_session=None if args.sample_mode == "all" else args.count_per_session,
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
            image = Image.open(frame_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            bbox_xywh = _resolve_bbox_xywh(row, bbox_source=str(args.bbox_source))
            if bbox_xywh:
                _draw_xywh(draw, bbox_xywh, outline="#00ff88", width=2)
            caption_parts = [
                str(row.get("session_key", "unknown")),
                str(row.get("frame_filename", "n/a")),
            ]
            if bbox_xywh:
                caption_parts.append(f"bbox={[round(float(v), 2) for v in bbox_xywh]}")
            caption_parts.append(f"bbox_source={args.bbox_source}")
            draw.text((6, 6), " | ".join(caption_parts), fill=(255, 255, 255))
            output_name = f"{Path(str(row.get('frame_filename', 'frame'))).stem}__eye_region.png"
            output_path = session_dir / output_name
            image.save(output_path)
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
                "count_per_session": None if args.sample_mode == "all" else args.count_per_session,
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
        "bbox_source": str(args.bbox_source),
        "count_per_session": None if args.sample_mode == "all" else args.count_per_session,
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "n_sessions": len(summary_rows),
        "n_panels": int(total_panels),
    }
    (output_dir / "raw_overlay_groundedsam_eye_region_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "raw_overlay_groundedsam_eye_region_sessions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summary_rows),
        encoding="utf-8",
    )
    print(f"[DONE] sessions={len(summary_rows)} panels={total_panels} output={output_dir}")


if __name__ == "__main__":
    main()
