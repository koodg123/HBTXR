#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.build_manifests import build_manifests
from hbtxr.preprocess.canonicalize import DEFAULT_EVENT_SIZE, DEFAULT_FRAME_SIZE, canonicalize_dataset
from hbtxr.preprocess.annotation_backends import (
    DEFAULT_ANNOTATION_BACKEND,
    SUPPORTED_ANNOTATION_BACKENDS,
)
from hbtxr.preprocess.groundedsam_build import (
    DEFAULT_CLASSES,
    annotate_dataset_with_groundedsam,
    annotate_dataset_with_groundedsam_multi,
    parse_groundedsam_devices,
)
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths


def _comma_split(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Grounded-SAM annotation, canonicalization, and manifest generation in one step")
    add_common_path_args(parser, need_raw=True, need_canonical=True)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_SIZE[0])
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_SIZE[1])
    parser.add_argument("--event-width", type=int, default=DEFAULT_EVENT_SIZE[0])
    parser.add_argument("--event-height", type=int, default=DEFAULT_EVENT_SIZE[1])
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--overwrite-links", action="store_true")
    parser.add_argument("--no-export-masks", action="store_true")
    parser.add_argument("--strict-layout", action="store_true")
    parser.add_argument("--include-nonstandard-sessions", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--split-scheme", type=str, default="exgaze_with_val", choices=["random", "exgaze", "exgaze_with_val"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--resize-policy", type=str, default="facet_square_direct", choices=["facet_square_direct", "letterbox_square", "sensor_full_square", "sensor_full_letterbox"])
    parser.add_argument("--event-policy", type=str, default="fixed_count", choices=["fixed_count", "time_bin"])
    parser.add_argument("--time-bin-us", type=int, default=5000)
    parser.add_argument("--event-count-target", type=int, default=5000)
    parser.add_argument("--accumulation", type=str, default="fast_causal_linear", choices=["plain", "causal_linear", "fast_causal_linear"])
    parser.add_argument("--causal-weight-power", type=float, default=1.0)
    parser.add_argument("--fast-causal-limit", type=float, default=25.0)
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
        help="Single torch device such as cpu, cuda, or cuda:0. Pass comma-separated devices such as cuda:0,cuda:1 to shard annotation across GPUs without overlap.",
    )
    parser.add_argument("--overwrite-annotations", action="store_true")
    parser.add_argument("--skip-annotation", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=True)
    resolved_backend = str(args.annotation_backend)
    if paths.annotation_root is None:
        raise ValueError("annotation_root is required. Pass --annotation-root or provide it in the paths config.")
    if (not args.skip_annotation) and resolved_backend == "groundedsam" and paths.groundedsam_root is None:
        raise ValueError("groundedsam_root is required when annotation_backend=groundedsam.")
    if (not args.skip_annotation) and resolved_backend == "ultralytics_sam3" and paths.ultralytics_root is None:
        raise ValueError("ultralytics_root is required when annotation_backend=ultralytics_sam3.")
    if (not args.skip_annotation) and resolved_backend == "groundedsam2" and paths.groundedsam2_root is None:
        raise ValueError("groundedsam2_root is required when annotation_backend=groundedsam2.")

    annotation_summary = None
    if not args.skip_annotation:
        devices = parse_groundedsam_devices(args.device)
        annotation_kwargs = {
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
            "overwrite": args.overwrite_annotations,
        }
        if len(devices) > 1:
            annotation_summary = annotate_dataset_with_groundedsam_multi(devices=devices, **annotation_kwargs)
        else:
            annotation_summary = annotate_dataset_with_groundedsam(
                **annotation_kwargs,
                device=devices[0] if devices else args.device,
            )

    canonical_summary = canonicalize_dataset(
        raw_root=paths.raw_root,
        canonical_root=paths.canonical_root,
        indexes_root=None if getattr(args, "indexes_root", None) is None else paths.indexes_root,
        default_frame_size=(int(args.frame_width), int(args.frame_height)),
        default_event_size=(int(args.event_width), int(args.event_height)),
        export_masks=not bool(args.no_export_masks),
        link_mode=str(args.link_mode),
        overwrite_links=bool(args.overwrite_links),
        strict_layout=bool(args.strict_layout),
        skip_nonstandard_sessions=not bool(args.include_nonstandard_sessions),
        annotation_mode="groundedsam",
        annotation_root=paths.annotation_root,
        groundedsam_root=paths.groundedsam_root,
        annotation_backend=resolved_backend,
        ultralytics_root=paths.ultralytics_root,
        groundedsam2_root=paths.groundedsam2_root,
        num_workers=int(args.num_workers),
        log_every=int(args.log_every),
    )

    manifest_summary = build_manifests(
        canonical_root=paths.canonical_root,
        indexes_root=None if getattr(args, "indexes_root", None) is None else paths.indexes_root,
        manifests_root=paths.manifests_root,
        split_scheme=str(args.split_scheme),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        resize_policy=str(args.resize_policy),
        target_size_wh=(int(args.frame_width), int(args.frame_height)),
        event_policy=str(args.event_policy),
        time_bin_us=int(args.time_bin_us),
        event_count_target=int(args.event_count_target),
        accumulation=str(args.accumulation),
        causal_weight_power=float(args.causal_weight_power),
        fast_causal_limit=float(args.fast_causal_limit),
    )

    summary = {
        "annotation": annotation_summary,
        "canonical": canonical_summary,
        "manifest": manifest_summary,
    }
    summary_path = Path(paths.manifests_root) / "build_groundedsam_dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
