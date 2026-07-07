#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import sys
import time
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from _config import (
    apply_config_overrides,
    load_config,
    resolve_run_contract,
    resolve_resume_root,
    resolve_training_entry,
    write_run_artifacts,
)
from hbtxr.optim.pool import expand_optimizer_pool_candidates, write_optimizer_pool_report
from hbtxr.training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HBTXR v3 with mode-aware run contract")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["mode0", "mode1", "mode2"], default=None)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--val-manifest", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _validate_path(path_str: str | None, *, label: str, optional: bool = False) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.exists():
        return str(path)
    if optional:
        print(f"[train_hbtxr] {label} not found, continuing without it: {path}")
        return None
    raise FileNotFoundError(f"{label} not found: {path}")


def _resume_checkpoint_from_root(run_root: Path) -> str | None:
    candidate = run_root / "train" / "last.pt"
    return str(candidate) if candidate.exists() else None


def _default_best_metric(stage: str) -> str:
    return "metric_search_p10_pct" if stage == "stage1" else "metric_track_p10_pct"


def _candidate_run_contract(candidate_root: Path, *, label: str) -> dict[str, str]:
    return {
        "action": "train_pool_candidate",
        "root": str(candidate_root),
        "materialized_experiment_name": label,
        "train_dir": str(candidate_root / "train"),
        "eval_dir": str(candidate_root / "eval"),
        "infer_dir": str(candidate_root / "infer"),
        "vis_dir": str(candidate_root / "vis"),
        "hypers_dir": str(candidate_root / "hypers"),
    }


def main() -> None:
    args = parse_args()
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
        output_dir_override=args.output,
        train_manifest_override=args.train_manifest,
        val_manifest_override=args.val_manifest,
        checkpoint_override=args.init_checkpoint,
    )
    experiment_name = str((cfg.get("experiment") or {}).get("name") or args.config.stem)
    resume_root = resolve_resume_root(PROJECT_ROOT, experiment_name, args.resume)
    run_contract = resolve_run_contract(
        cfg,
        config_path=args.config,
        project_root=PROJECT_ROOT,
        action="train",
        resume_root=resume_root,
    )
    cfg.setdefault("run", {})
    cfg["run"]["materialized_experiment_name"] = run_contract["materialized_experiment_name"]
    resolved = resolve_training_entry(
        cfg,
        config_path=args.config,
        project_root=PROJECT_ROOT,
        train_manifest_override=args.train_manifest,
        val_manifest_override=args.val_manifest,
        output_override=run_contract["train_dir"] if args.output is None else args.output,
        init_checkpoint_override=args.init_checkpoint,
    )
    train_manifest = _validate_path(resolved["train_manifest"], label="train manifest")
    val_manifest = _validate_path(resolved["val_manifest"], label="val manifest", optional=True)
    init_checkpoint = _validate_path(resolved["init_checkpoint"], label="init checkpoint", optional=False) if resolved["init_checkpoint"] else None
    resume_checkpoint = None
    if resume_root is not None:
        resume_checkpoint = _resume_checkpoint_from_root(resume_root)
        if resume_checkpoint is None:
            raise FileNotFoundError(f"Resume checkpoint not found under run root: {resume_root}")
    write_run_artifacts(
        run_contract=run_contract,
        cfg=cfg,
        config_path=args.config,
        cli_args=sys.argv[1:],
        overrides=overrides,
        device=str((cfg.get("training") or {}).get("device", "")),
        pretrained_report_name="pretrained_report_stage1.json" if resolved["stage"] == "stage1" else "pretrained_report.json",
    )

    optimizer_pool_cfg = ((cfg.get("training") or {}).get("optimizer_pool") or {})
    if bool(optimizer_pool_cfg.get("enabled", False)):
        if resume_checkpoint is not None:
            raise ValueError("optimizer_pool mode does not support --resume yet.")
        pool_root = Path(run_contract["root"]) / "pool"
        rows: list[dict[str, object]] = []
        candidates = expand_optimizer_pool_candidates(cfg)
        metric_name = str((cfg.get("training") or {}).get("best_metric_name") or _default_best_metric(resolved["stage"]))

        for candidate in candidates:
            candidate_cfg = deepcopy(cfg)
            candidate_cfg.setdefault("training", {})
            candidate_cfg["training"]["optimizer"] = deepcopy(candidate["resolved"])
            candidate_cfg["training"]["optimizer_modifiers"] = deepcopy(candidate["modifiers"])
            candidate_root = pool_root / candidate["label"]
            candidate_contract = _candidate_run_contract(candidate_root, label=str(candidate["label"]))
            write_run_artifacts(
                run_contract=candidate_contract,
                cfg=candidate_cfg,
                config_path=args.config,
                cli_args=sys.argv[1:],
                overrides=overrides,
                device=str((candidate_cfg.get("training") or {}).get("device", "")),
                pretrained_report_name="pretrained_report_stage1.json" if resolved["stage"] == "stage1" else "pretrained_report.json",
            )
            started_at = time.time()
            result = train(
                cfg=candidate_cfg,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                output_dir=candidate_contract["train_dir"],
                stage1_checkpoint=init_checkpoint,
                resume_checkpoint=None,
            )
            elapsed = time.time() - started_at
            best_metrics = result.get("best_metrics") or {}
            best_entry = best_metrics.get(metric_name) or {}
            rows.append(
                {
                    "label": candidate["label"],
                    "optimizer": candidate["resolved"]["name"],
                    "modifiers": ",".join(key for key, enabled in candidate["modifiers"].items() if enabled),
                    "stage": resolved["stage"],
                    "best_metric_name": metric_name,
                    metric_name: float(best_entry.get("value", float("nan"))),
                    "best_epoch": best_entry.get("epoch"),
                    "elapsed_sec": elapsed,
                    "output_dir": result["output_dir"],
                }
            )

        report_paths = write_optimizer_pool_report(pool_root, rows, metric_name=metric_name)
        print(
            {
                "config": str(args.config),
                "mode": resolved["mode"],
                "stage": resolved["stage"],
                "experiment_name": run_contract["materialized_experiment_name"],
                "pool_root": str(pool_root),
                "candidates": [row["label"] for row in rows],
                "reports": report_paths,
            }
        )
        return

    print(
        {
            "config": str(args.config),
            "mode": resolved["mode"],
            "stage": resolved["stage"],
            "experiment_name": run_contract["materialized_experiment_name"],
            "train_manifest": train_manifest,
            "val_manifest": val_manifest,
            "run_root": run_contract["root"],
            "output_dir": run_contract["train_dir"],
            "init_checkpoint": init_checkpoint,
            "resume_checkpoint": resume_checkpoint,
        }
    )
    train(
        cfg=cfg,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=run_contract["train_dir"],
        stage1_checkpoint=init_checkpoint,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == "__main__":
    main()
