#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.groundedsam_build import (
    _patch_groundingdino_ms_deform_attn_compat,
    _patch_transformers_groundingdino_compat,
)
from hbtxr.preprocess.io_utils import canonical_user_name, code_to_session_dir, collect_frame_records
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths


def _draw_xyxy(draw: ImageDraw.ImageDraw, xyxy: list[float], *, outline: str, width: int = 2) -> None:
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    draw.rectangle((x0, y0, x1, y1), outline=outline, width=width)


def _resolve_default_groundingdino_config(repo_root: Path) -> Path:
    return repo_root / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"


def _prepare_model(*, repo_root: Path, config_path: Path, checkpoint_path: Path, device: str):
    extra_paths = [repo_root, repo_root / "GroundingDINO"]
    for extra in extra_paths:
        text = str(extra)
        if text not in sys.path:
            sys.path.insert(0, text)

    import torch  # noqa: WPS433
    import torchvision  # noqa: WPS433

    _patch_transformers_groundingdino_compat()
    _patch_groundingdino_ms_deform_attn_compat()
    from groundingdino.util.inference import Model  # noqa: WPS433

    model = Model(
        model_config_path=str(config_path),
        model_checkpoint_path=str(checkpoint_path),
        device=str(device),
    )
    return torch, torchvision, model


def _pick_best_eye_detection(*, torch_mod, torchvision_mod, detections, nms_threshold: float) -> tuple[list[float], float] | None:
    xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4), dtype=np.float32)))
    conf = np.asarray(getattr(detections, "confidence", np.empty((0,), dtype=np.float32)))
    if xyxy.size == 0:
        return None
    keep = torchvision_mod.ops.nms(
        torch_mod.from_numpy(xyxy),
        torch_mod.from_numpy(conf.astype(np.float32)),
        float(nms_threshold),
    ).cpu().numpy().tolist()
    if not keep:
        return None
    best_idx = max(keep, key=lambda idx: float(conf[idx]))
    return [float(v) for v in xyxy[int(best_idx)].tolist()], float(conf[int(best_idx)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPU eye-region bbox detection on all frames of one raw session and save overlays")
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--groundingdino-config", type=str, default=None)
    parser.add_argument("--groundingdino-checkpoint", type=str, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--eye", type=str, required=True, choices=["left", "right"])
    parser.add_argument("--session-code", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--annotation-jsonl", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.18)
    parser.add_argument("--text-threshold", type=float, default=0.12)
    parser.add_argument("--nms-threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    if paths.groundedsam_root is None:
        raise ValueError("groundedsam_root is required. Pass --groundedsam-root or provide it in the paths config.")

    repo_root = paths.groundedsam_root.resolve()
    config_path = Path(args.groundingdino_config).resolve() if args.groundingdino_config else _resolve_default_groundingdino_config(repo_root)
    checkpoint_path = Path(args.groundingdino_checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")

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
        else (paths.preview_root / "groundedsam_eye_bbox_overlay" / session_key).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_jsonl = (
        Path(args.annotation_jsonl).resolve()
        if args.annotation_jsonl
        else (output_dir / "groundedsam_eye_bbox_rows.jsonl").resolve()
    )

    torch_mod, torchvision_mod, model = _prepare_model(
        repo_root=repo_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=str(args.device),
    )

    frame_records = collect_frame_records(frames_dir)
    rows = []
    detected = 0
    for record in frame_records:
        frame_path = frames_dir / record.filename
        image_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
        image_bgr = image_rgb[:, :, ::-1].copy()
        detections = model.predict_with_classes(
            image=image_bgr,
            classes=["eye"],
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
        )
        picked = _pick_best_eye_detection(
            torch_mod=torch_mod,
            torchvision_mod=torchvision_mod,
            detections=detections,
            nms_threshold=float(args.nms_threshold),
        )

        overlay = Image.fromarray(image_rgb).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        row = {
            "session_key": session_key,
            "user_id": int(args.user_id),
            "eye": str(args.eye),
            "session_code": str(args.session_code),
            "frame_filename": record.filename,
            "frame_idx": record.frame_idx,
            "timestamp_us": int(record.timestamp_us),
            "gsam_class_name": "eye",
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "nms_threshold": float(args.nms_threshold),
            "detected": False,
            "eye_region_bbox_xyxy_sensor": None,
            "eye_region_bbox_xywh_sensor": None,
            "gsam_box_confidence": None,
        }
        if picked is not None:
            xyxy, confidence = picked
            x0, y0, x1, y1 = xyxy
            xywh = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]
            _draw_xyxy(draw, xyxy, outline="#00ff88", width=2)
            row["detected"] = True
            row["eye_region_bbox_xyxy_sensor"] = xyxy
            row["eye_region_bbox_xywh_sensor"] = xywh
            row["gsam_box_confidence"] = float(confidence)
            detected += 1
        caption_parts = [
            session_key,
            record.filename,
            f"detected={str(row['detected']).lower()}",
        ]
        if row["gsam_box_confidence"] is not None:
            caption_parts.append(f"box={float(row['gsam_box_confidence']):.2f}")
        draw.text((6, 6), " | ".join(caption_parts), fill=(255, 255, 255))
        output_path = output_dir / f"{Path(record.filename).stem}__eye_bbox.png"
        overlay.save(output_path)
        row["overlay_path"] = str(output_path)
        rows.append(row)

    annotation_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "raw_root": str(paths.raw_root),
        "groundedsam_root": str(repo_root),
        "groundingdino_config": str(config_path),
        "groundingdino_checkpoint": str(checkpoint_path),
        "raw_session_dir": str(raw_session_dir),
        "frames_dir": str(frames_dir),
        "session_key": session_key,
        "device": str(args.device),
        "n_frames_total": len(frame_records),
        "n_detected": int(detected),
        "output_dir": str(output_dir),
        "annotation_jsonl": str(annotation_jsonl),
    }
    (output_dir / "groundedsam_eye_bbox_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[DONE] session={session_key} "
        f"frames={len(frame_records)} "
        f"detected={detected} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
