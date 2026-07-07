#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import SENSOR_HEIGHT, SENSOR_WIDTH, canonical_user_name, code_to_session_dir
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _pick_frame_paths(
    frame_paths: list[Path],
    *,
    count: int | None,
    sample_mode: str,
    seed: int,
) -> list[Path]:
    if not frame_paths:
        return []
    if sample_mode == "all" or count is None or len(frame_paths) <= int(count):
        return list(frame_paths)
    limit = max(1, int(count))
    if sample_mode == "first":
        return list(frame_paths[:limit])
    if sample_mode == "random":
        rng = random.Random(int(seed))
        picked = rng.sample(frame_paths, limit)
        return sorted(picked)
    if sample_mode == "uniform":
        if limit == 1:
            return [frame_paths[0]]
        indices = sorted({int(round(idx * (len(frame_paths) - 1) / (limit - 1))) for idx in range(limit)})
        return [frame_paths[idx] for idx in indices]
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def _parse_bbox_xywh(text: str | None, *, sensor_width: int, sensor_height: int) -> list[int]:
    if text is None:
        return [0, 0, int(sensor_width), int(sensor_height)]
    parts = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError(f"--bbox-xywh expects 4 comma-separated values, got: {text}")
    return [int(round(float(item))) for item in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw overlay previews using an eye-region bounding box only, without canonical annotations"
    )
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--eye", type=str, required=True, choices=["left", "right"])
    parser.add_argument("--session-code", type=str, required=True, help="Session code such as 101, 102, 201, 202")
    parser.add_argument("--bbox-xywh", type=str, default=None, help="Explicit eye-region box as x,y,w,h. Defaults to sensor-aligned fallback.")
    parser.add_argument("--sensor-width", type=int, default=int(SENSOR_WIDTH))
    parser.add_argument("--sensor-height", type=int, default=int(SENSOR_HEIGHT))
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--sample-mode", type=str, default="uniform", choices=["uniform", "first", "random", "all"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)

    user_name = canonical_user_name(int(args.user_id))
    session_dir_name = code_to_session_dir(str(args.session_code))
    session_key = f"{user_name}/{str(args.eye)}/session_{str(args.session_code)}"
    raw_session_dir = paths.raw_root / f"user{int(args.user_id)}" / str(args.eye) / session_dir_name
    frames_dir = raw_session_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (paths.preview_root / "raw_overlay_eye_region_bbox" / session_key).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    eye_region_xywh = _parse_bbox_xywh(
        args.bbox_xywh,
        sensor_width=int(args.sensor_width),
        sensor_height=int(args.sensor_height),
    )
    frame_paths = sorted(frames_dir.glob("*.png"))
    picked = _pick_frame_paths(
        frame_paths,
        count=None if args.sample_mode == "all" else int(args.count),
        sample_mode=str(args.sample_mode),
        seed=int(args.seed),
    )
    if not picked:
        raise ValueError(f"No PNG frames found under: {frames_dir}")

    exported_files = []
    for frame_path in picked:
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        _draw_xywh(draw, eye_region_xywh, outline="#00ff88", width=2)
        caption = (
            f"{session_key} | {frame_path.name} | "
            f"eye_region={eye_region_xywh} | source=fallback_sensor_bbox"
        )
        draw.text((6, 6), caption, fill=(255, 255, 255))
        output_path = output_dir / f"{frame_path.stem}__eye_region.png"
        image.save(output_path)
        exported_files.append(str(output_path))

    summary = {
        "raw_root": str(paths.raw_root),
        "raw_session_dir": str(raw_session_dir),
        "frames_dir": str(frames_dir),
        "session_key": session_key,
        "user_id": int(args.user_id),
        "eye": str(args.eye),
        "session_code": str(args.session_code),
        "eye_region_xywh": [int(v) for v in eye_region_xywh],
        "sensor_size_wh": [int(args.sensor_width), int(args.sensor_height)],
        "frame_count_total": len(frame_paths),
        "sample_mode": str(args.sample_mode),
        "count": None if args.sample_mode == "all" else int(args.count),
        "seed": int(args.seed),
        "n_exported": len(exported_files),
        "output_dir": str(output_dir),
        "files": exported_files,
    }
    (output_dir / "eye_region_bbox_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[DONE] session={session_key} "
        f"frame_count_total={len(frame_paths)} "
        f"exported={len(exported_files)} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
