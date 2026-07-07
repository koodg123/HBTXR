#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import canonical_user_name, is_official_session_code
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths

_SESSION_RE = re.compile(r"^user(?P<user_id>\d+)_session_(?P<a>\d+)_(?P<b>\d+)_(?P<c>\d+)$")


def _load_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "h5py is required for raw_overlay_preview_h5_mask.py. "
            "Install it in the active environment, for example: pip install h5py"
        ) from exc
    return h5py


def _comma_split(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _overlay_mask_array(base: Image.Image, mask: np.ndarray, *, color: str = "#ff5050", alpha: int = 96) -> Image.Image:
    image = base.convert("RGBA")
    rgba = Image.new("RGBA", image.size, color=ImageColor.getrgb(color) + (0,))
    mask_image = Image.fromarray(np.where(mask > 0, alpha, 0).astype(np.uint8), mode="L")
    rgba.putalpha(mask_image)
    return Image.alpha_composite(image, rgba).convert("RGB")


def _pick_indices(total: int, *, count_per_session: int | None, sample_mode: str, seed: int) -> list[int]:
    if total <= 0:
        return []
    if sample_mode == "all" or count_per_session is None or total <= int(count_per_session):
        return list(range(total))
    limit = max(1, int(count_per_session))
    if sample_mode == "first":
        return list(range(limit))
    if sample_mode == "random":
        rng = random.Random(int(seed))
        return sorted(rng.sample(list(range(total)), limit))
    if sample_mode == "uniform":
        if limit == 1:
            return [0]
        return sorted({int(round(idx * (total - 1) / (limit - 1))) for idx in range(limit)})
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def _parse_h5_session(path: Path) -> dict | None:
    match = _SESSION_RE.match(path.stem)
    if match is None:
        return None
    session_code = f"{match.group('a')}{match.group('b')}{match.group('c')}"
    return {
        "user_id": int(match.group("user_id")),
        "session_code": session_code,
        "session_dir_name": f"session_{match.group('a')}_{match.group('b')}_{match.group('c')}",
    }


def _iter_h5_sessions(h5_root: Path) -> list[tuple[int, str, str, Path]]:
    rows = []
    for eye in ("left", "right"):
        eye_dir = h5_root / eye
        if not eye_dir.exists():
            continue
        for h5_path in sorted(eye_dir.glob("*.h5")):
            parsed = _parse_h5_session(h5_path)
            if parsed is None:
                continue
            rows.append((parsed["user_id"], eye, parsed["session_code"], h5_path))
    return rows


def _resolve_h5_root(args: argparse.Namespace, paths) -> Path:
    if getattr(args, "h5_root", None):
        return Path(args.h5_root).expanduser().resolve()
    if paths.raw_root is not None:
        raw_root = paths.raw_root.resolve()
        if raw_root.name == "Data_davis_labelled_with_mask":
            return raw_root
        sibling = raw_root.parent / "Data_davis_labelled_with_mask"
        if sibling.exists():
            return sibling.resolve()
    raise ValueError(
        "h5_root could not be resolved. Pass --h5-root or supply a raw_root whose sibling "
        "directory is Data_davis_labelled_with_mask."
    )


def _ensure_hwn_stack(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D HDF5 dataset shaped like HxWxN or WxHxN, got {arr.shape}")
    no_transpose_score = abs(int(arr.shape[0]) - 260) + abs(int(arr.shape[1]) - 346)
    transpose_score = abs(int(arr.shape[1]) - 260) + abs(int(arr.shape[0]) - 346)
    if transpose_score < no_transpose_score:
        arr = arr.transpose(1, 0, 2)
    return arr


def _normalize_grayscale_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale frame, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi <= 1.0 and lo >= 0.0:
        scaled = arr * 255.0
    elif hi > lo:
        scaled = (arr - lo) / (hi - lo)
        scaled = scaled * 255.0
    else:
        scaled = np.zeros(arr.shape, dtype=np.float32)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _to_positive_mask(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D label frame, got shape {arr.shape}")
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def _mask_bbox_xywh(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export overlay previews directly from Data_davis_labelled_with_mask HDF5 files"
    )
    add_common_path_args(parser, need_raw=False, need_canonical=False)
    parser.add_argument("--h5-root", type=str, default=None, help="Root directory of Data_davis_labelled_with_mask")
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
    paths = resolve_paths(args, need_raw=False, need_canonical=False)
    h5_root = _resolve_h5_root(args, paths)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (paths.preview_root / "raw_overlay_h5_mask").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_user_ids = {int(item) for item in _comma_split(args.user_ids)}
    allowed_eyes = {str(args.eye).lower()} if args.eye else set()
    allowed_session_codes = {str(item) for item in _comma_split(args.session_codes)}

    h5py = _load_h5py()
    session_rows = []
    processed_sessions = 0
    for user_id, eye, session_code, h5_path in _iter_h5_sessions(h5_root):
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
        try:
            with h5py.File(h5_path, "r") as handle:
                dataset_names = list(handle.keys())
                if "data" not in handle or "label" not in handle:
                    session_rows.append(
                        {
                            "session_key": session_key,
                            "h5_path": str(h5_path),
                            "skipped": True,
                            "message": "missing_data_or_label_dataset",
                            "datasets": dataset_names,
                        }
                    )
                    continue
                data_stack = _ensure_hwn_stack(handle["data"][:])
                label_stack = _ensure_hwn_stack(handle["label"][:])
        except Exception as exc:
            session_rows.append(
                {
                    "session_key": session_key,
                    "h5_path": str(h5_path),
                    "skipped": True,
                    "message": f"h5_read_failed: {type(exc).__name__}: {exc}",
                }
            )
            continue

        if data_stack.shape[:2] != label_stack.shape[:2]:
            session_rows.append(
                {
                    "session_key": session_key,
                    "h5_path": str(h5_path),
                    "skipped": True,
                    "message": "data_label_spatial_shape_mismatch",
                    "data_shape": [int(v) for v in data_stack.shape],
                    "label_shape": [int(v) for v in label_stack.shape],
                }
            )
            continue

        frame_count = min(int(data_stack.shape[-1]), int(label_stack.shape[-1]))
        picked_indices = _pick_indices(
            frame_count,
            count_per_session=None if args.sample_mode == "all" else int(args.count_per_session),
            sample_mode=str(args.sample_mode),
            seed=int(args.seed),
        )
        if not picked_indices:
            session_rows.append(
                {
                    "session_key": session_key,
                    "h5_path": str(h5_path),
                    "skipped": True,
                    "message": "no_frames_kept",
                    "n_frames_total": frame_count,
                }
            )
            continue

        session_output_dir = output_dir / session_key
        session_output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = []
        mask_area_ratios = []
        image_height, image_width = int(data_stack.shape[0]), int(data_stack.shape[1])
        for frame_index in picked_indices:
            image_frame = _normalize_grayscale_frame(data_stack[..., frame_index])
            mask = _to_positive_mask(label_stack[..., frame_index])
            mask_bbox = _mask_bbox_xywh(mask)
            mask_area_ratio = float(mask.mean())
            base = Image.fromarray(image_frame, mode="L").convert("RGB")
            overlay = _overlay_mask_array(base, mask)
            draw = ImageDraw.Draw(overlay)
            if mask_bbox is not None:
                _draw_xywh(draw, mask_bbox, outline="#ffd54f", width=2)
            caption = (
                f"{session_key} | frame={frame_index:04d} | source=h5_mask | "
                f"official={str(is_official_session_code(session_code)).lower()} | "
                f"mask={mask_area_ratio:.3f}"
            )
            draw.text((6, 6), caption, fill=(255, 255, 255))
            output_path = session_output_dir / f"{frame_index:04d}__overlay.png"
            overlay.save(output_path)
            exported_files.append(str(output_path))
            mask_area_ratios.append(mask_area_ratio)

        session_rows.append(
            {
                "session_key": session_key,
                "h5_path": str(h5_path),
                "skipped": not bool(exported_files),
                "message": "h5_mask_overlay_export_complete" if exported_files else "no_frames_exported",
                "datasets": ["data", "label"],
                "n_frames_total": frame_count,
                "n_exported": len(exported_files),
                "image_shape_hw": [image_height, image_width],
                "sample_mode": str(args.sample_mode),
                "count_per_session": None if args.sample_mode == "all" else int(args.count_per_session),
                "output_dir": str(session_output_dir),
                "mask_area_ratio_min": float(min(mask_area_ratios)) if mask_area_ratios else None,
                "mask_area_ratio_median": float(np.median(mask_area_ratios)) if mask_area_ratios else None,
                "mask_area_ratio_max": float(max(mask_area_ratios)) if mask_area_ratios else None,
                "files": exported_files,
            }
        )

    summary = {
        "h5_root": str(h5_root),
        "output_dir": str(output_dir),
        "user_ids": sorted(allowed_user_ids),
        "eye": args.eye,
        "session_codes": sorted(allowed_session_codes),
        "count_per_session": None if args.sample_mode == "all" else int(args.count_per_session),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "n_sessions_seen": len(session_rows),
        "n_sessions_exported": int(sum(1 for row in session_rows if not row.get("skipped"))),
        "n_panels": int(sum(int(row.get("n_exported", 0)) for row in session_rows)),
    }
    (output_dir / "raw_overlay_h5_mask_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "raw_overlay_h5_mask_sessions.jsonl").write_text(
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
