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
    resolve_manifest_path,
    resolve_resume_root,
    resolve_run_contract,
    write_run_artifacts,
)
from _viz import event_to_image, frame_to_image, mask_to_image, save_panel
from hybrid.training.trainer import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HBTXR v3 dataset samples")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=16)
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
    manifest_path = resolve_manifest_path(cfg, project_root=PROJECT_ROOT, split=args.split, manifest_override=args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    output_dir = args.output or (Path(run_contract["vis_dir"]) / "dataset" / str(args.split))
    loader = make_loader(str(manifest_path), cfg, shuffle=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    for batch in loader:
        for idx, sample_id in enumerate(batch["sample_id"]):
            images = [
                frame_to_image(batch["frame"][idx].numpy()),
                event_to_image(batch["event"][idx].numpy()),
                mask_to_image(batch["mask_target"][idx].numpy()),
            ]
            save_panel(images, output_dir / f"{sample_id}.png", labels=["frame", "event", "mask"])
            produced += 1
            if produced >= args.limit:
                return


if __name__ == "__main__":
    main()
