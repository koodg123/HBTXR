#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.event_generation import (
    DEFAULT_EVENT_GENERATION_BACKEND,
    SUPPORTED_EVENT_GENERATION_BACKENDS,
)
from hbtxr.preprocess.interpolation import (
    DEFAULT_INTERPOLATION_BACKEND,
    SUPPORTED_INTERPOLATION_BACKENDS,
)
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.preprocess.target_fps_build import build_target_fps_dataset


def _comma_split(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or plan the integrated v3.2 target-FPS dataset surface for EV-Eye")
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--target-root", type=str, default=None, help="Target dataset workspace root. Defaults to <project_root>/workspace/target_data")
    parser.add_argument("--target-fps", type=float, required=True, help="Target FPS used to build the target timestamp grid")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--eye", choices=["left", "right", "both"], default="both")
    parser.add_argument("--session-code", type=str, default=None, help="Optional comma-separated session filters such as 101,102")
    parser.add_argument("--include-nonstandard-sessions", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Materialize session artifacts instead of planning only")
    parser.add_argument(
        "--interpolation-backend",
        type=str,
        default=DEFAULT_INTERPOLATION_BACKEND,
        choices=list(SUPPORTED_INTERPOLATION_BACKENDS),
        help="Interpolation backend used for non-raw target frames",
    )
    parser.add_argument("--timelens-checkpoint", type=str, default=None, help="Optional TimeLens checkpoint when --interpolation-backend timelens is used")
    parser.add_argument("--timelens-device", type=str, default="cpu")
    parser.add_argument("--timelens-xl-checkpoint", type=str, default=None, help="Optional TimeLens-XL checkpoint when --interpolation-backend timelens_xl is used")
    parser.add_argument("--timelens-xl-device", type=str, default="cpu")
    parser.add_argument(
        "--event-generation-backend",
        type=str,
        default=DEFAULT_EVENT_GENERATION_BACKEND,
        choices=list(SUPPORTED_EVENT_GENERATION_BACKENDS),
        help="Synthetic event-generation backend used after frame interpolation",
    )
    parser.add_argument("--v2e-device", type=str, default="cpu")
    parser.add_argument("--v2e-pos-thres", type=float, default=None)
    parser.add_argument("--v2e-neg-thres", type=float, default=None)
    parser.add_argument("--v2e-sigma-thres", type=float, default=None)
    parser.add_argument("--v2e-cutoff-hz", type=float, default=None)
    parser.add_argument("--v2e-leak-rate-hz", type=float, default=None)
    parser.add_argument("--v2e-shot-noise-rate-hz", type=float, default=None)
    parser.add_argument("--v2e-refractory-period-s", type=float, default=None)
    parser.add_argument("--v2e-seed", type=int, default=None)
    parser.add_argument("--session-store-format", choices=["auto", "h5", "npz"], default="auto")
    parser.add_argument(
        "--frame-storage-mode",
        choices=["materialized_target_frames", "lazy_source_frames"],
        default="materialized_target_frames",
        help="How to store target-fps frames inside the session store",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    target_root = Path(args.target_root).resolve() if args.target_root else (paths.project_root / "workspace" / "target_data").resolve()
    v2e_kwargs = {
        key: value
        for key, value in {
            "pos_thres": args.v2e_pos_thres,
            "neg_thres": args.v2e_neg_thres,
            "sigma_thres": args.v2e_sigma_thres,
            "cutoff_hz": args.v2e_cutoff_hz,
            "leak_rate_hz": args.v2e_leak_rate_hz,
            "shot_noise_rate_hz": args.v2e_shot_noise_rate_hz,
            "refractory_period_s": args.v2e_refractory_period_s,
            "seed": args.v2e_seed,
        }.items()
        if value is not None
    }
    summary = build_target_fps_dataset(
        raw_root=paths.raw_root,
        target_root=target_root,
        target_fps=float(args.target_fps),
        annotation_root=paths.annotation_root,
        user_id=args.user_id,
        eye=str(args.eye),
        session_codes=_comma_split(args.session_code),
        include_nonstandard_sessions=bool(args.include_nonstandard_sessions),
        max_sessions=args.max_sessions,
        execute=bool(args.execute),
        interpolation_backend=str(args.interpolation_backend),
        timelens_root=paths.timelens_root,
        timelens_checkpoint=args.timelens_checkpoint,
        timelens_device=str(args.timelens_device),
        timelens_xl_root=paths.timelens_xl_root,
        timelens_xl_checkpoint=args.timelens_xl_checkpoint,
        timelens_xl_device=str(args.timelens_xl_device),
        event_generation_backend=str(args.event_generation_backend),
        v2e_root=paths.v2e_root,
        v2e_device=str(args.v2e_device),
        v2e_kwargs=v2e_kwargs,
        session_store_format=str(args.session_store_format),
        frame_storage_mode=str(args.frame_storage_mode),
        overwrite=bool(args.overwrite),
    )
    print(
        f"[DONE] stage={summary['stage']} sessions_planned={summary['n_sessions_planned']} "
        f"target_frames={summary['n_target_frames_planned']} root={summary['target_root']}"
    )


if __name__ == "__main__":
    main()
