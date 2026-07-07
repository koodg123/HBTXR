#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.event_generation import DEFAULT_EVENT_GENERATION_BACKEND
from hbtxr.preprocess.interpolation import DEFAULT_INTERPOLATION_BACKEND, SUPPORTED_INTERPOLATION_BACKENDS
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.preprocess.target_fps_build import build_target_fps_dataset
from hbtxr.preprocess.v2e_experiment import compare_target_fps_roots


def _comma_split(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paired target-FPS experiment that compares baseline rebinned events against v2e-generated events"
    )
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--experiment-root", type=str, default=None, help="Workspace root for the paired none/v2e builds")
    parser.add_argument("--target-fps", type=float, required=True)
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--eye", choices=["left", "right", "both"], default="both")
    parser.add_argument("--session-code", type=str, default=None, help="Optional comma-separated session filters such as 101,102")
    parser.add_argument("--include-nonstandard-sessions", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--interpolation-backend", type=str, default=DEFAULT_INTERPOLATION_BACKEND, choices=list(SUPPORTED_INTERPOLATION_BACKENDS))
    parser.add_argument("--timelens-checkpoint", type=str, default=None)
    parser.add_argument("--timelens-device", type=str, default="cpu")
    parser.add_argument("--timelens-xl-checkpoint", type=str, default=None)
    parser.add_argument("--timelens-xl-device", type=str, default="cpu")
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
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset builds and only compare existing paired roots")
    return parser


def _build_one(
    *,
    raw_root: Path,
    annotation_root: Path | None,
    target_root: Path,
    target_fps: float,
    user_id: int | None,
    eye: str,
    session_codes: list[str],
    include_nonstandard_sessions: bool,
    max_sessions: int | None,
    interpolation_backend: str,
    timelens_root: Path | None,
    timelens_checkpoint: str | None,
    timelens_device: str,
    timelens_xl_root: Path | None,
    timelens_xl_checkpoint: str | None,
    timelens_xl_device: str,
    event_generation_backend: str,
    v2e_root: Path | None,
    v2e_device: str,
    v2e_kwargs: dict | None,
    session_store_format: str,
    frame_storage_mode: str,
    overwrite: bool,
) -> dict:
    return build_target_fps_dataset(
        raw_root=raw_root,
        target_root=target_root,
        target_fps=target_fps,
        annotation_root=annotation_root,
        user_id=user_id,
        eye=eye,
        session_codes=session_codes,
        include_nonstandard_sessions=include_nonstandard_sessions,
        max_sessions=max_sessions,
        execute=True,
        interpolation_backend=interpolation_backend,
        timelens_root=timelens_root,
        timelens_checkpoint=timelens_checkpoint,
        timelens_device=timelens_device,
        timelens_xl_root=timelens_xl_root,
        timelens_xl_checkpoint=timelens_xl_checkpoint,
        timelens_xl_device=timelens_xl_device,
        event_generation_backend=event_generation_backend,
        v2e_root=v2e_root,
        v2e_device=v2e_device,
        v2e_kwargs=v2e_kwargs,
        session_store_format=session_store_format,
        frame_storage_mode=frame_storage_mode,
        overwrite=overwrite,
    )


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    experiment_root = Path(args.experiment_root).resolve() if args.experiment_root else (paths.project_root / "workspace" / "v2e_experiments" / "default").resolve()
    baseline_root = experiment_root / "none_target_data"
    candidate_root = experiment_root / "v2e_target_data"
    report_root = experiment_root / "reports"
    session_codes = _comma_split(args.session_code)
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

    if not args.skip_build:
        baseline_summary = _build_one(
            raw_root=paths.raw_root,
            annotation_root=paths.annotation_root,
            target_root=baseline_root,
            target_fps=float(args.target_fps),
            user_id=args.user_id,
            eye=str(args.eye),
            session_codes=session_codes,
            include_nonstandard_sessions=bool(args.include_nonstandard_sessions),
            max_sessions=args.max_sessions,
            interpolation_backend=str(args.interpolation_backend),
            timelens_root=paths.timelens_root,
            timelens_checkpoint=args.timelens_checkpoint,
            timelens_device=str(args.timelens_device),
            timelens_xl_root=paths.timelens_xl_root,
            timelens_xl_checkpoint=args.timelens_xl_checkpoint,
            timelens_xl_device=str(args.timelens_xl_device),
            event_generation_backend=DEFAULT_EVENT_GENERATION_BACKEND,
            v2e_root=paths.v2e_root,
            v2e_device=str(args.v2e_device),
            v2e_kwargs=v2e_kwargs,
            session_store_format=str(args.session_store_format),
            frame_storage_mode=str(args.frame_storage_mode),
            overwrite=bool(args.overwrite),
        )
        candidate_summary = _build_one(
            raw_root=paths.raw_root,
            annotation_root=paths.annotation_root,
            target_root=candidate_root,
            target_fps=float(args.target_fps),
            user_id=args.user_id,
            eye=str(args.eye),
            session_codes=session_codes,
            include_nonstandard_sessions=bool(args.include_nonstandard_sessions),
            max_sessions=args.max_sessions,
            interpolation_backend=str(args.interpolation_backend),
            timelens_root=paths.timelens_root,
            timelens_checkpoint=args.timelens_checkpoint,
            timelens_device=str(args.timelens_device),
            timelens_xl_root=paths.timelens_xl_root,
            timelens_xl_checkpoint=args.timelens_xl_checkpoint,
            timelens_xl_device=str(args.timelens_xl_device),
            event_generation_backend="v2e",
            v2e_root=paths.v2e_root,
            v2e_device=str(args.v2e_device),
            v2e_kwargs=v2e_kwargs,
            session_store_format=str(args.session_store_format),
            frame_storage_mode=str(args.frame_storage_mode),
            overwrite=bool(args.overwrite),
        )
        print(
            f"[BUILD] baseline_events={baseline_summary['n_rebinned_events_materialized']} "
            f"candidate_events={candidate_summary['n_rebinned_events_materialized']}"
        )

    summary = compare_target_fps_roots(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        target_fps=float(args.target_fps),
        output_dir=report_root,
    )
    print(
        f"[DONE] experiment_root={experiment_root} shared_sessions={summary['n_shared_sessions']} "
        f"baseline_total_events={summary['baseline_total_events']} "
        f"candidate_total_events={summary['candidate_total_events']} "
        f"ratio={summary['event_ratio_candidate_vs_baseline']}"
    )


if __name__ == "__main__":
    main()
