"""Evaluation entrypoint for the reimplemented HBTXR models (P5.7).

Replaces the old Lightning ``apps/evaluate.py`` with a thin launcher over the custom
evaluator:

    load_config -> make_model -> load_checkpoint -> build_dataloader(val/test,
    shuffle=False) -> evaluate -> print/return metrics.

Run from the ``algorithm/`` root::

    python -m engine.eval.entrypoint -c configs/experiment/frame_hbtxr.yaml --ckpt runs/.../final.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from engine.data.adapter import resolve_modality
from engine.data.factory import build_dataloader
from engine.eval.evaluator import evaluate
from engine.model_factory import make_model
from engine.runspec.run_contract import resolve_manifest_path, resolve_training_entry
from engine.tools.checkpoint import load_checkpoint
from engine.tools.load_config import load_config


def _resolve_project_root(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    root = (cfg.get("experiment") or {}).get("project_root")
    return Path(root).expanduser().resolve() if root else Path.cwd()


def _resolve_eval_manifest(cfg: dict[str, Any], root: Path, config_path: str) -> str:
    entry = resolve_training_entry(cfg, config_path=config_path, project_root=root)
    if entry.get("val_manifest"):
        return entry["val_manifest"]
    return str(resolve_manifest_path(cfg, project_root=root, split="test"))


def _resolve_ckpt(cfg: dict[str, Any], override: str | None) -> str | None:
    if override:
        return override
    for section in ("eval", "val", "test"):
        ckpt = (cfg.get(section) or {}).get("ckpt_path")
        if ckpt:
            return str(ckpt)
    return None


def run_eval(
    config_path: str,
    *,
    ckpt: str | None = None,
    project_root: str | None = None,
    device: str | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint on the val/test manifest and return averaged metrics."""
    cfg = load_config(config_path)
    root = _resolve_project_root(cfg, project_root)
    modality = resolve_modality(cfg)
    dev = device or str((cfg.get("training") or {}).get("device", "cpu"))

    model = make_model(cfg["model"])
    ckpt_path = _resolve_ckpt(cfg, ckpt)
    if ckpt_path:
        load_checkpoint(model, ckpt_path, map_location=dev)

    manifest = _resolve_eval_manifest(cfg, root, config_path)
    loader = build_dataloader(manifest, cfg, shuffle=False, modality=modality)
    weights = (cfg.get("training") or {}).get("loss_weights") or (cfg.get("loss") or {}).get("weights")
    metrics = evaluate(model, loader, modality=modality, device=dev, loss_weights=weights if isinstance(weights, dict) else None)
    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a reimplemented HBTXR model.")
    parser.add_argument("-c", "--config", required=True, help="experiment config path (yaml/json)")
    parser.add_argument("--ckpt", default=None, help="checkpoint path (overrides config)")
    parser.add_argument("--project-root", default=None, help="root for manifest resolution")
    parser.add_argument("--device", default=None, help="override eval device")
    args = parser.parse_args(argv)
    run_eval(args.config, ckpt=args.ckpt, project_root=args.project_root, device=args.device)


if __name__ == "__main__":
    main()
