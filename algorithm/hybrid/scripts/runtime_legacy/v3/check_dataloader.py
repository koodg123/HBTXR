#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from _config import apply_config_overrides, load_config, resolve_manifest_path
from hbtxr.training.trainer import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one HBTXR dataloader batch and report the ABI")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--mode", choices=["mode0", "mode1", "mode2"], default=None)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _summarize_value(value):
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        torch = None
    if torch is not None and torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "item_type": type(value[0]).__name__ if value else None,
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(value.keys()),
        }
    return {
        "type": type(value).__name__,
        "value": value,
    }


def inspect_dataloader(
    *,
    cfg: dict,
    manifest_path: str | Path,
    split: str,
) -> dict:
    loader = make_loader(str(manifest_path), cfg, shuffle=False)
    batch = next(iter(loader))
    summary = {
        "manifest": str(manifest_path),
        "split": str(split),
        "mode": str((cfg.get("data") or {}).get("mode", "mode1")),
        "stage": str((cfg.get("training") or {}).get("stage", "stage1")),
        "batch_size": len(batch["sample_id"]) if "sample_id" in batch else None,
        "keys": sorted(batch.keys()),
        "shapes": {key: _summarize_value(value) for key, value in batch.items()},
    }
    if "meta" in batch and batch["meta"]:
        first_meta = batch["meta"][0]
        summary["first_meta"] = dict(first_meta) if isinstance(first_meta, dict) else first_meta
    if "sample_id" in batch and batch["sample_id"]:
        summary["sample_ids"] = list(batch["sample_id"])
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    overrides = list(args.override or [])
    if args.mode:
        overrides.append(f"data.mode={args.mode}")
    if args.stage:
        overrides.append(f"training.stage={args.stage}")
    cfg = apply_config_overrides(cfg, overrides=overrides)
    manifest_path = resolve_manifest_path(cfg, project_root=PROJECT_ROOT, split=args.split, manifest_override=args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    summary = inspect_dataloader(cfg=cfg, manifest_path=manifest_path, split=args.split)
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
