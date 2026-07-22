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
    resolve_manifest_path,
    resolve_project_path,
    resolve_resume_root,
    resolve_run_contract,
    write_run_artifacts,
)
from hybrid.training.losses import compute_metrics
from hybrid.training.trainer import build_model, checkpoint_summary_metadata, make_loader, move_to_device


def resolve_device(cfg: dict) -> torch.device:
    raw = str(cfg.get("training", {}).get("device", "cpu"))
    if raw == "multi-gpu":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    first = raw.split(",")[0].strip()
    if first.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(first or "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HBTXR v3 checkpoint with mode-aware run contract")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["mode0", "mode1", "mode2"], default=None)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _require_existing_path(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


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
        action="eval",
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
    manifest_path = _require_existing_path(
        resolve_manifest_path(cfg, project_root=PROJECT_ROOT, split=args.split, manifest_override=args.manifest),
        label="manifest",
    )
    checkpoint_path = resolve_project_path(args.checkpoint, project_root=PROJECT_ROOT)
    if checkpoint_path is None:
        if resume_root is None:
            raise ValueError("Evaluation requires --checkpoint or --resume to resolve an existing run root.")
        checkpoint_path = default_checkpoint_for_stage(
            resume_root,
            str((cfg.get("training") or {}).get("stage", "stage1")),
        )
    checkpoint_path = _require_existing_path(checkpoint_path, label="checkpoint")
    device = resolve_device(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    loader = make_loader(str(manifest_path), cfg, shuffle=False)

    rows = []
    metrics_acc: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch)
            metrics = compute_metrics(batch, outputs)
            for key, value in metrics.items():
                metrics_acc.setdefault(key, []).append(float(value.detach().cpu().item()))
            for sample_idx, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "search_state": outputs.get("search/state", torch.empty(0)).detach().cpu()[sample_idx].tolist() if "search/state" in outputs else None,
                        "event_state": outputs.get("event/state", torch.empty(0)).detach().cpu()[sample_idx].tolist() if "event/state" in outputs else None,
                        "track_state": outputs.get("track/state", torch.empty(0)).detach().cpu()[sample_idx].tolist() if "track/state" in outputs else None,
                    }
                )

    summary = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "mode": str((cfg.get("data") or {}).get("mode", "mode1")),
        "split": str(args.split),
        **checkpoint_summary_metadata(state if isinstance(state, dict) else None),
        **{key: (sum(values) / max(1, len(values))) for key, values in metrics_acc.items()},
    }
    output_dir = args.output or (Path(run_contract["eval_dir"]) / str(args.split))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "eval_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
