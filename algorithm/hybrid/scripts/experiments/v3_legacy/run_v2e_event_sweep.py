#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.event_generation import DEFAULT_EVENT_GENERATION_BACKEND
from hbtxr.preprocess.interpolation import DEFAULT_INTERPOLATION_BACKEND, SUPPORTED_INTERPOLATION_BACKENDS
from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.preprocess.target_fps_build import build_target_fps_dataset
from hbtxr.preprocess.v2e_experiment import compare_target_fps_roots


def _comma_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _comma_split(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _format_threshold_tag(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep v2e threshold settings against a fixed target-FPS baseline")
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--experiment-root", type=str, required=True, help="Workspace root for baseline/candidate sweep outputs")
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
    parser.add_argument("--v2e-pos-thres-values", type=str, required=True, help="Comma-separated threshold values, e.g. 0.50,0.45,0.40")
    parser.add_argument("--v2e-neg-thres-values", type=str, default=None, help="Optional comma-separated OFF thresholds. Defaults to the ON list.")
    parser.add_argument("--v2e-sigma-thres", type=float, default=0.02)
    parser.add_argument("--v2e-cutoff-hz", type=float, default=30.0)
    parser.add_argument("--v2e-leak-rate-hz", type=float, default=0.0)
    parser.add_argument("--v2e-shot-noise-rate-hz", type=float, default=0.0)
    parser.add_argument("--v2e-refractory-period-s", type=float, default=None)
    parser.add_argument("--v2e-seed", type=int, default=None)
    parser.add_argument("--session-store-format", choices=["auto", "h5", "npz"], default="auto")
    parser.add_argument(
        "--frame-storage-mode",
        choices=["materialized_target_frames", "lazy_source_frames"],
        default="materialized_target_frames",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-baseline-build", action="store_true")
    return parser


def _build_target_dataset(
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
    experiment_root = Path(args.experiment_root).resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)
    baseline_root = experiment_root / "none_target_data"
    reports_root = experiment_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    session_codes = _comma_split(args.session_code)

    pos_values = _comma_float_list(args.v2e_pos_thres_values)
    neg_values = _comma_float_list(args.v2e_neg_thres_values) if args.v2e_neg_thres_values else list(pos_values)
    if len(pos_values) != len(neg_values):
        raise ValueError("v2e_pos_thres_values and v2e_neg_thres_values must have the same number of entries")

    common_v2e_kwargs = {
        key: value
        for key, value in {
            "sigma_thres": args.v2e_sigma_thres,
            "cutoff_hz": args.v2e_cutoff_hz,
            "leak_rate_hz": args.v2e_leak_rate_hz,
            "shot_noise_rate_hz": args.v2e_shot_noise_rate_hz,
            "refractory_period_s": args.v2e_refractory_period_s,
            "seed": args.v2e_seed,
        }.items()
        if value is not None
    }

    if not args.skip_baseline_build:
        _build_target_dataset(
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
            v2e_kwargs=common_v2e_kwargs,
            session_store_format=str(args.session_store_format),
            frame_storage_mode=str(args.frame_storage_mode),
            overwrite=bool(args.overwrite),
        )

    results: list[dict] = []
    for pos_thres, neg_thres in zip(pos_values, neg_values, strict=False):
        label = f"p{_format_threshold_tag(pos_thres)}_n{_format_threshold_tag(neg_thres)}"
        candidate_root = experiment_root / f"v2e_{label}_target_data"
        v2e_kwargs = dict(common_v2e_kwargs)
        v2e_kwargs["pos_thres"] = float(pos_thres)
        v2e_kwargs["neg_thres"] = float(neg_thres)
        try:
            _build_target_dataset(
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
            compare_summary = compare_target_fps_roots(
                baseline_root=baseline_root,
                candidate_root=candidate_root,
                target_fps=float(args.target_fps),
                output_dir=reports_root / label,
            )
            results.append(
                {
                    "label": label,
                    "status": "ok",
                    "pos_thres": float(pos_thres),
                    "neg_thres": float(neg_thres),
                    "candidate_root": str(candidate_root),
                    "event_ratio_candidate_vs_baseline": compare_summary["event_ratio_candidate_vs_baseline"],
                    "baseline_total_events": compare_summary["baseline_total_events"],
                    "candidate_total_events": compare_summary["candidate_total_events"],
                    "baseline_pos_fraction": compare_summary["baseline_pos_fraction"],
                    "candidate_pos_fraction": compare_summary["candidate_pos_fraction"],
                    "report_dir": str((reports_root / label).resolve()),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "label": label,
                    "status": "error",
                    "pos_thres": float(pos_thres),
                    "neg_thres": float(neg_thres),
                    "candidate_root": str(candidate_root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    successful = [item for item in results if item.get("status") == "ok"]
    ranked = sorted(
        successful,
        key=lambda item: abs((item["event_ratio_candidate_vs_baseline"] or 0.0) - 1.0),
    )
    sweep_summary = {
        "target_fps": float(args.target_fps),
        "experiment_root": str(experiment_root),
        "interpolation_backend": str(args.interpolation_backend),
        "frame_storage_mode": str(args.frame_storage_mode),
        "session_store_format": str(args.session_store_format),
        "results": results,
        "ranked_by_ratio_distance_to_1": ranked,
        "best_label": None if not ranked else ranked[0]["label"],
    }
    summary_path = reports_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(sweep_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[DONE] candidates={len(results)} best={sweep_summary['best_label']} "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
