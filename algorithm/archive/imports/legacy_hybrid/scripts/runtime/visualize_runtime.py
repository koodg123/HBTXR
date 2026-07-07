#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from _config import (
    apply_config_overrides,
    load_config,
    resolve_project_path,
    resolve_resume_root,
    resolve_run_contract,
    write_run_artifacts,
)
from _viz import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize runtime/inference trace summaries")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--mode", choices=["mode0", "mode1", "mode2"], default=None)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is None:
        if args.results is None or args.output_dir is None:
            raise ValueError("visualize_runtime requires --config for run-contract mode, or both --results and --output-dir for manual mode.")
        results_path = Path(args.results)
        output_dir = args.output_dir
    else:
        cfg = load_config(args.config)
        overrides = list(args.override or [])
        if args.mode:
            overrides.append(f"data.mode={args.mode}")
        if args.stage:
            overrides.append(f"training.stage={args.stage}")
        cfg = apply_config_overrides(
            cfg,
            overrides=overrides,
            device_override=args.device,
            experiment_name_override=args.experiment_name,
        )
        experiment_name = str((cfg.get("experiment") or {}).get("name") or args.config.stem)
        resume_root = resolve_resume_root(PROJECT_ROOT, experiment_name, args.resume)
        run_contract = resolve_run_contract(
            cfg,
            config_path=args.config,
            project_root=PROJECT_ROOT,
            action="vis",
            resume_root=resume_root,
        )
        cfg.setdefault("run", {})
        cfg["run"]["materialized_experiment_name"] = run_contract["materialized_experiment_name"]
        write_run_artifacts(
            run_contract=run_contract,
            cfg=cfg,
            config_path=args.config,
            cli_args=sys.argv[1:],
            overrides=overrides,
            device=str((cfg.get("training") or {}).get("device", "")),
        )
        results_path = resolve_project_path(args.results, project_root=PROJECT_ROOT)
        if results_path is None:
            results_path = Path(run_contract["infer_dir"]) / str(args.split) / "infer_rows.jsonl"
        output_dir = args.output_dir or (Path(run_contract["vis_dir"]) / "runtime" / str(args.split))
    if not results_path.exists():
        raise FileNotFoundError(f"runtime input results not found: {results_path}")
    rows = read_jsonl(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = []
    for row in rows[: args.limit]:
        summary_lines.append(
            f"{row.get('sample_id')} search={row.get('search_state')} track={row.get('track_state')}"
        )
    (output_dir / "runtime_trace.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
