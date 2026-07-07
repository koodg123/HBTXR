#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.preprocess.raw_ellipse_blink import RawEllipseBlinkHeuristicConfig, export_raw_ellipse_blink_candidates


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract blink-candidate frames from raw EV-Eye VIA CSV ellipse annotations",
    )
    add_common_path_args(parser, need_raw=True, need_canonical=False)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--eye", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--session-code", type=str, default="102")
    parser.add_argument("--baseline-minor-quantile", type=float, default=0.90)
    parser.add_argument("--baseline-area-quantile", type=float, default=0.90)
    parser.add_argument("--baseline-major-quantile", type=float, default=0.50)
    parser.add_argument("--local-window", type=int, default=2)
    parser.add_argument("--global-minor-ratio-threshold", type=float, default=0.75)
    parser.add_argument("--global-area-ratio-threshold", type=float, default=0.70)
    parser.add_argument("--aspect-ratio-threshold", type=float, default=0.70)
    parser.add_argument("--local-minor-ratio-threshold", type=float, default=0.80)
    parser.add_argument("--local-area-ratio-threshold", type=float, default=0.75)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=True, need_canonical=False)
    if paths.raw_root is None:
        raise ValueError("raw_root is required. Pass --raw-root or provide it in the paths config.")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (paths.project_root / "workspace" / "raw_ellipse_blink_candidates").resolve()
    )
    heuristic = RawEllipseBlinkHeuristicConfig(
        baseline_minor_quantile=float(args.baseline_minor_quantile),
        baseline_area_quantile=float(args.baseline_area_quantile),
        baseline_major_quantile=float(args.baseline_major_quantile),
        local_window=max(0, int(args.local_window)),
        global_minor_ratio_threshold=float(args.global_minor_ratio_threshold),
        global_area_ratio_threshold=float(args.global_area_ratio_threshold),
        aspect_ratio_threshold=float(args.aspect_ratio_threshold),
        local_minor_ratio_threshold=float(args.local_minor_ratio_threshold),
        local_area_ratio_threshold=float(args.local_area_ratio_threshold),
    )
    summary = export_raw_ellipse_blink_candidates(
        raw_root=paths.raw_root,
        output_root=output_root,
        user_id=args.user_id,
        eye=args.eye,
        session_code=args.session_code,
        heuristic=heuristic,
    )
    print(
        f"[DONE] annotations={summary['n_annotations']} candidates={summary['n_blink_candidates']} "
        f"rate={summary['candidate_rate']:.3f} summary={summary['summary_path']}"
    )


if __name__ == "__main__":
    main()
