#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.groundedsam_eye_region_bbox import export_groundedsam_eye_region_bbox_overlays
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Grounded-SAM eye-region bounding boxes only and save per-frame overlay previews",
    )
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--groundingdino-config", type=str, default=None)
    parser.add_argument("--groundingdino-checkpoint", type=str, default=None)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-code", type=str, default="101")
    parser.add_argument("--eye", type=str, default="both", choices=["left", "right", "both"])
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.8)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.85)
    parser.add_argument("--border-margin-px", type=int, default=3)
    parser.add_argument("--max-border-touches", type=int, default=3)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Single torch device such as cpu, cuda, or cuda:0.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    if paths.annotation_root is None:
        raise ValueError("annotation_root is required. Pass --annotation-root or provide it in the paths config.")
    if paths.groundedsam_root is None:
        raise ValueError("groundedsam_root is required. Pass --groundedsam-root or provide it in the paths config.")
    if paths.preview_root is None:
        raise ValueError("preview_root is required. Pass --preview-root or provide it in the paths config.")

    summary = export_groundedsam_eye_region_bbox_overlays(
        raw_root=paths.raw_root,
        annotation_root=paths.annotation_root,
        preview_root=paths.preview_root,
        groundedsam_root=paths.groundedsam_root,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        user_id=args.user_id,
        session_code=args.session_code,
        eye=args.eye,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
        max_box_area_ratio=args.max_box_area_ratio,
        border_margin_px=args.border_margin_px,
        max_border_touches=args.max_border_touches,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(
        f"[DONE] sessions={summary['n_sessions']} frames={summary['n_frames']} "
        f"detected={summary['n_detected']} bbox_root={summary['bbox_root']} "
        f"preview_root={summary['preview_output_root']}"
    )


if __name__ == "__main__":
    main()
