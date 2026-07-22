#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hybrid.preprocess.build_manifests import build_manifests
from hybrid.preprocess.annotation_backends import (
    DEFAULT_ANNOTATION_BACKEND,
    SUPPORTED_ANNOTATION_BACKENDS,
)
from hybrid.preprocess.canonicalize import DEFAULT_EVENT_SIZE, DEFAULT_FRAME_SIZE, canonicalize_dataset
from hybrid.preprocess.event_generation import (
    DEFAULT_EVENT_GENERATION_BACKEND,
    SUPPORTED_EVENT_GENERATION_BACKENDS,
)
from hybrid.preprocess.path_utils import add_common_path_args, resolve_paths
from hybrid.preprocess.progress import print_progress


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EV-Eye canonicalization and HBTXR v3 manifest generation in one step")
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
    parser.add_argument("--annotation-mode", choices=["auto", "manual_csv", "groundedsam"], default="auto")
    parser.add_argument("--annotation-backend", choices=list(SUPPORTED_ANNOTATION_BACKENDS), default=DEFAULT_ANNOTATION_BACKEND)
    parser.add_argument("--data-mode", choices=["mode0", "mode1", "mode2"], default="mode1")
    parser.add_argument("--canonical-name", type=str, default="canonical1")
    parser.add_argument("--manifest-name", type=str, default="manifest1")
    parser.add_argument("--frame-source", type=str, default="original")
    parser.add_argument("--split-scheme", type=str, default="exgaze_with_val", choices=["random", "exgaze", "exgaze_with_val"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--resize-policy", type=str, default="facet_square_direct", choices=["facet_square_direct", "letterbox_square", "sensor_full_square"])
    parser.add_argument("--event-policy", type=str, default="fixed_count", choices=["fixed_count", "time_bin"])
    parser.add_argument("--time-bin-us", type=int, default=5000)
    parser.add_argument("--event-count-target", type=int, default=5000)
    parser.add_argument("--accumulation", type=str, default="fast_causal_linear", choices=["plain", "causal_linear", "fast_causal_linear"])
    parser.add_argument("--causal-weight-power", type=float, default=1.0)
    parser.add_argument("--fast-causal-limit", type=float, default=25.0)
    parser.add_argument("--interp-alpha", type=float, default=0.5)
    parser.add_argument("--interp-model", type=str, default="linear_blend")
    parser.add_argument("--interp-backend", type=str, default=None, choices=["linear_blend", "timelens", "timelens_xl"])
    parser.add_argument("--interp-target-fps", type=float, default=None)
    parser.add_argument("--interp-fixed-insert", type=int, default=None)
    parser.add_argument("--interp-count-policy", type=str, default="round", choices=["round", "floor", "ceil"])
    parser.add_argument("--interp-max-insert", type=int, default=None)
    parser.add_argument("--interp-timelens-root", type=str, default=None, help="TimeLens repo root. Falls back to --timelens-root / paths-config / packages/timelens when available")
    parser.add_argument("--interp-timelens-checkpoint", type=str, default=None)
    parser.add_argument("--interp-timelens-device", type=str, default="cpu")
    parser.add_argument("--interp-timelens-xl-root", type=str, default=None, help="TimeLens-XL repo root. Falls back to --timelens-xl-root / paths-config / Third/FI when available")
    parser.add_argument("--interp-timelens-xl-checkpoint", type=str, default=None)
    parser.add_argument("--interp-timelens-xl-device", type=str, default="cpu")
    parser.add_argument("--event-generation-backend", type=str, default=DEFAULT_EVENT_GENERATION_BACKEND, choices=list(SUPPORTED_EVENT_GENERATION_BACKENDS))
    parser.add_argument("--v2e-device", type=str, default="cpu")
    parser.add_argument("--synthetic-overlap-policy", type=str, default="reuse_event_window")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if str(args.data_mode) == "mode0":
        if str(args.canonical_name) == "canonical1":
            args.canonical_name = "canonical0"
        if str(args.manifest_name) == "manifest1":
            args.manifest_name = "manifest0"
        if str(args.frame_source) != "original":
            args.frame_source = "original"
        if str(args.annotation_mode) == "auto":
            args.annotation_mode = "manual_csv"
    elif str(args.data_mode) == "mode2":
        if str(args.canonical_name) == "canonical1":
            args.canonical_name = "canonical2"
        if str(args.manifest_name) == "manifest1":
            args.manifest_name = "manifest2"
        if str(args.frame_source) == "original":
            args.frame_source = "interpolated"
    paths = resolve_paths(args, need_raw=True, need_canonical=True)
    resolved_timelens_root = paths.timelens_root if args.interp_timelens_root is None else args.interp_timelens_root
    resolved_timelens_xl_root = paths.timelens_xl_root if args.interp_timelens_xl_root is None else args.interp_timelens_xl_root
    started_at = time.perf_counter()
    print_progress(
        name="prepare_ev_eye",
        status="start",
        started_at=started_at,
        completed=0,
        total=2,
        message="prepare pipeline started",
        raw_root=str(paths.raw_root),
        canonical_root=str(paths.canonical_root),
        manifests_root=str(paths.manifests_root),
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
        annotation_mode=str(args.annotation_mode),
        annotation_root=paths.annotation_root,
        groundedsam_root=paths.groundedsam_root,
        annotation_backend=str(args.annotation_backend),
        ultralytics_root=paths.ultralytics_root,
        groundedsam2_root=paths.groundedsam2_root,
        data_mode=str(args.data_mode),
        canonical_name=str(args.canonical_name),
        frame_source=str(args.frame_source),
        interpolation_alpha=float(args.interp_alpha),
        interpolation_model=str(args.interp_model),
        interpolation_backend=None if args.interp_backend is None else str(args.interp_backend),
        interpolation_target_fps=None if args.interp_target_fps is None else float(args.interp_target_fps),
        interpolation_fixed_insert=None if args.interp_fixed_insert is None else int(args.interp_fixed_insert),
        interpolation_count_policy=str(args.interp_count_policy),
        interpolation_max_insert=None if args.interp_max_insert is None else int(args.interp_max_insert),
        interpolation_timelens_root=resolved_timelens_root,
        interpolation_timelens_checkpoint=None if args.interp_timelens_checkpoint is None else args.interp_timelens_checkpoint,
        interpolation_timelens_device=str(args.interp_timelens_device),
        interpolation_timelens_xl_root=resolved_timelens_xl_root,
        interpolation_timelens_xl_checkpoint=None if args.interp_timelens_xl_checkpoint is None else args.interp_timelens_xl_checkpoint,
        interpolation_timelens_xl_device=str(args.interp_timelens_xl_device),
        event_generation_backend=str(args.event_generation_backend),
        v2e_root=paths.v2e_root,
        v2e_device=str(args.v2e_device),
        synthetic_overlap_policy=str(args.synthetic_overlap_policy),
        num_workers=int(args.num_workers),
        log_every=int(args.log_every),
    )
    print_progress(
        name="prepare_ev_eye",
        status="progress",
        started_at=started_at,
        completed=1,
        total=2,
        message="canonicalization complete",
        n_sessions_ok=canonical_summary["n_sessions_ok"],
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
        data_mode=str(args.data_mode),
        canonical_name=str(args.canonical_name),
        manifest_name=str(args.manifest_name),
        frame_source=str(args.frame_source),
        synthetic_overlap_policy=str(args.synthetic_overlap_policy),
    )
    print_progress(
        name="prepare_ev_eye",
        status="done",
        started_at=started_at,
        completed=2,
        total=2,
        message="prepare pipeline complete",
        n_sessions_ok=canonical_summary["n_sessions_ok"],
        manifest_counts=manifest_summary["counts"],
    )
    print(f"[DONE] canonical={canonical_summary['n_sessions_ok']} manifests={manifest_summary['counts']} root={manifest_summary['manifests_root']}")
