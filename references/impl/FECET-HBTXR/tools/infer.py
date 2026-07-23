from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from _config import apply_config_overrides, load_config, resolve_training_entry

from fecet_hbtxr.io import write_json, write_jsonl
from fecet_hbtxr.runtime import run_runtime_trace
from fecet_hbtxr.trainer import build_model, load_checkpoint, make_loader, resolve_device_and_wrap


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run runtime inference with Search/Track FSM logging")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stage2_hybrid.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-jsonl", type=str, default=str(PROJECT_ROOT / "runs" / "inference" / "runtime_trace.jsonl"))
    parser.add_argument("--output-summary", type=str, default=str(PROJECT_ROOT / "runs" / "inference" / "runtime_summary.json"))
    parser.add_argument("--override", action="append", default=[])
    return parser


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    cfg = apply_config_overrides(cfg, overrides=args.override, device_override=args.device)
    entry = resolve_training_entry(cfg, config_path=args.config, project_root=PROJECT_ROOT)
    manifest = args.manifest or entry["val_manifest"] or entry["train_manifest"]
    loader = make_loader(manifest, cfg, shuffle=False)

    model = build_model(cfg)
    model, device = resolve_device_and_wrap(model, str((cfg.get("training") or {}).get("device", "cpu")))
    load_checkpoint(model=model, optimizer=None, scheduler=None, scaler=None, checkpoint_path=args.checkpoint, strict=False)
    rows = run_runtime_trace(model, loader, device=device)

    jsonl_path = Path(args.output_jsonl).resolve()
    summary_path = Path(args.output_summary).resolve()
    write_jsonl(rows, jsonl_path)
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(manifest).resolve()),
        "output_jsonl": str(jsonl_path),
        "row_count": len(rows),
    }
    write_json(summary, summary_path)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
