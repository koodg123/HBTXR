#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from src.preprocess.groundedsam_build import (
    DEFAULT_CLASSES,
    annotate_dataset_with_groundedsam,
    annotate_dataset_with_groundedsam_multi,
    parse_groundedsam_devices,
)
from src.preprocess.annotation_backends import (
    DEFAULT_ANNOTATION_BACKEND,
    SUPPORTED_ANNOTATION_BACKENDS,
)
from src.preprocess.path_utils import add_common_path_args, resolve_paths


def _comma_split(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate EV-Eye frames with Grounded-SAM without modifying the raw dataset")
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--annotation-backend", type=str, default=DEFAULT_ANNOTATION_BACKEND, choices=list(SUPPORTED_ANNOTATION_BACKENDS))
    parser.add_argument("--groundingdino-config", type=str, default=None)
    parser.add_argument("--groundingdino-checkpoint", type=str, default=None)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--ultralytics-sam3-checkpoint", type=str, default=None)
    parser.add_argument("--groundedsam2-config", type=str, default=None)
    parser.add_argument("--groundedsam2-checkpoint", type=str, default=None)
    parser.add_argument("--groundedsam2-groundingdino-config", type=str, default=None)
    parser.add_argument("--groundedsam2-groundingdino-checkpoint", type=str, default=None)
    parser.add_argument("--sam-encoder-version", type=str, default="vit_h")
    parser.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.8)
    parser.add_argument("--min-mask-area", type=int, default=16)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames-per-session", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Single torch device such as cpu, cuda, or cuda:0. Pass comma-separated devices such as cuda:0,cuda:1 to launch non-overlapping annotation shards.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    resolved_backend = str(args.annotation_backend)
    if paths.annotation_root is None:
        raise ValueError("annotation_root is required. Pass --annotation-root or provide it in the paths config.")
    if resolved_backend == "groundedsam" and paths.groundedsam_root is None:
        raise ValueError("groundedsam_root is required for annotation_backend=groundedsam.")
    if resolved_backend == "ultralytics_sam3" and paths.ultralytics_root is None:
        raise ValueError("ultralytics_root is required for annotation_backend=ultralytics_sam3.")
    if resolved_backend == "groundedsam2" and paths.groundedsam2_root is None:
        raise ValueError("groundedsam2_root is required for annotation_backend=groundedsam2.")

    devices = parse_groundedsam_devices(args.device)
    common_kwargs = {
        "raw_root": paths.raw_root,
        "annotation_root": paths.annotation_root,
        "annotation_backend": resolved_backend,
        "groundedsam_root": paths.groundedsam_root,
        "ultralytics_root": paths.ultralytics_root,
        "groundedsam2_root": paths.groundedsam2_root,
        "groundingdino_config": args.groundingdino_config,
        "groundingdino_checkpoint": args.groundingdino_checkpoint,
        "sam_checkpoint": args.sam_checkpoint,
        "ultralytics_sam3_checkpoint": args.ultralytics_sam3_checkpoint,
        "groundedsam2_config": args.groundedsam2_config,
        "groundedsam2_checkpoint": args.groundedsam2_checkpoint,
        "groundedsam2_groundingdino_config": args.groundedsam2_groundingdino_config,
        "groundedsam2_groundingdino_checkpoint": args.groundedsam2_groundingdino_checkpoint,
        "sam_encoder_version": args.sam_encoder_version,
        "classes": _comma_split(args.classes),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "nms_threshold": args.nms_threshold,
        "min_mask_area": args.min_mask_area,
        "frame_step": args.frame_step,
        "max_frames_per_session": args.max_frames_per_session,
        "max_sessions": args.max_sessions,
        "overwrite": args.overwrite,
    }
    if len(devices) > 1:
        summary = annotate_dataset_with_groundedsam_multi(devices=devices, **common_kwargs)
    else:
        summary = annotate_dataset_with_groundedsam(
            **common_kwargs,
            device=devices[0] if devices else args.device,
        )
    print(f"[DONE] sessions={summary['n_sessions']} annotations={summary['n_annotations']} root={summary['annotation_root']}")


if __name__ == "__main__":
    main()
