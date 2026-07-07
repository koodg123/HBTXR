#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.v2e_experiment import compare_target_fps_roots


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline event packets against v2e-generated event packets on target-FPS session stores"
    )
    parser.add_argument("--baseline-root", type=str, required=True, help="Baseline target_data root built with event_generation_backend=none")
    parser.add_argument("--candidate-root", type=str, required=True, help="Candidate target_data root built with event_generation_backend=v2e")
    parser.add_argument("--target-fps", type=float, required=True)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for summary.json and event-count plots")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = compare_target_fps_roots(
        baseline_root=Path(args.baseline_root).resolve(),
        candidate_root=Path(args.candidate_root).resolve(),
        target_fps=float(args.target_fps),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(
        f"[DONE] shared_sessions={summary['n_shared_sessions']} "
        f"baseline_total_events={summary['baseline_total_events']} "
        f"candidate_total_events={summary['candidate_total_events']} "
        f"ratio={summary['event_ratio_candidate_vs_baseline']}"
    )


if __name__ == "__main__":
    main()
