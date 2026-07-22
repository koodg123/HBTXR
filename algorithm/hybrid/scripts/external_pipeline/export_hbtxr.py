#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from _config import (
    apply_config_overrides,
    default_checkpoint_for_stage,
    load_config,
    resolve_project_path,
    resolve_resume_root,
    resolve_run_contract,
    write_run_artifacts,
)
from hybrid.models.export_pruned import export_structural_student
from hybrid.training.trainer import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a structural slim-student checkpoint from an HBTXR checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["mode0", "mode1", "mode2"], default=None)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _resolve_device(cfg: dict) -> torch.device:
    raw = str((cfg.get("training") or {}).get("device", "cpu")).strip()
    if raw == "multi-gpu":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    first = raw.split(",")[0].strip()
    if first.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(first or "cpu")


def _extract_model_state(payload: dict) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        return payload["model"]
    raise ValueError("checkpoint payload does not contain a top-level 'model' state_dict")


def export_from_checkpoint(
    *,
    cfg: dict,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    stage: str,
) -> dict:
    device = _resolve_device(cfg)
    model = build_model(cfg, role="student").to(device)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_model_state(payload)
    model.load_state_dict(state_dict, strict=False)
    return export_structural_student(
        model=model,
        cfg=cfg,
        output_dir=output_dir,
        stage=stage,
    )


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
    )
    experiment_name = str((cfg.get("experiment") or {}).get("name") or args.config.stem)
    resume_root = resolve_resume_root(PROJECT_ROOT, experiment_name, args.resume)
    run_contract = resolve_run_contract(
        cfg,
        config_path=args.config,
        project_root=PROJECT_ROOT,
        action="export",
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
    checkpoint_path = resolve_project_path(args.checkpoint, project_root=PROJECT_ROOT)
    stage = str((cfg.get("training") or {}).get("stage", "stage1")).strip().lower()
    if checkpoint_path is None:
        if resume_root is None:
            raise ValueError("Export requires --checkpoint or --resume to resolve an existing run root.")
        checkpoint_path = default_checkpoint_for_stage(resume_root, stage)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    output_dir = args.output or Path(run_contract["train_dir"])
    report = export_from_checkpoint(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        stage=stage,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
