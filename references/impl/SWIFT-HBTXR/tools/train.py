from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from _config import apply_config_overrides, load_config, resolve_training_entry

from swift_hbtxr.trainer import train


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the simplified SWIFT-HBTXR pipeline")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stage1_search.yaml"))
    parser.add_argument("--stage", choices=["stage1", "stage2"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--val-manifest", type=str, default=None)
    parser.add_argument("--stage1-checkpoint", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    if args.stage is not None:
        args.override.append(f"training.stage={args.stage}")
    cfg = apply_config_overrides(
        cfg,
        overrides=args.override,
        device_override=args.device,
        experiment_name_override=args.experiment_name,
        output_dir_override=args.output_dir,
        train_manifest_override=args.train_manifest,
        val_manifest_override=args.val_manifest,
        checkpoint_override=args.stage1_checkpoint,
    )
    entry = resolve_training_entry(
        cfg,
        config_path=args.config,
        project_root=PROJECT_ROOT,
        train_manifest_override=args.train_manifest,
        val_manifest_override=args.val_manifest,
        output_override=args.output_dir,
        init_checkpoint_override=args.stage1_checkpoint,
    )
    result = train(
        cfg=cfg,
        train_manifest=entry["train_manifest"],
        val_manifest=entry["val_manifest"],
        output_dir=entry["output_dir"],
        stage1_checkpoint=args.stage1_checkpoint or entry["init_checkpoint"],
        resume_checkpoint=args.resume_checkpoint,
    )
    result["stage"] = entry["stage"]
    result["config"] = str(Path(args.config).resolve())
    return result


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
