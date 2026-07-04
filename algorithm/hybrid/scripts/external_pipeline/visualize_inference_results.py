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
    resolve_project_path,
    resolve_resume_root,
    resolve_run_contract,
    write_run_artifacts,
)
from _viz import event_to_image, frame_to_image, read_jsonl, save_panel
from src.training.trainer import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HBTXR inference rows")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--results", type=str, default=None)
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
    results_path = resolve_project_path(args.results, project_root=PROJECT_ROOT)
    if results_path is None:
        results_path = Path(run_contract["infer_dir"]) / str(args.split) / "infer_rows.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"inference results not found: {results_path}")
    output_dir = args.output or (Path(run_contract["vis_dir"]) / "inference" / str(args.split))
    loader = make_loader(str(manifest_path), cfg, shuffle=False)
    infer_rows = {row["sample_id"]: row for row in read_jsonl(results_path)}
    output_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    for batch in loader:
        for idx, sample_id in enumerate(batch["sample_id"]):
            if sample_id not in infer_rows:
                continue
            row = infer_rows[sample_id]
            frame_img = frame_to_image(batch["frame"][idx].numpy())
            event_img = event_to_image(batch["event"][idx].numpy())
            info = frame_img.copy()
            from PIL import ImageDraw

            draw = ImageDraw.Draw(info)
            draw.text((4, 4), f"sample={sample_id}", fill=(255, 255, 255))
            if row.get("track_state"):
                x, y = row["track_state"][0], row["track_state"][1]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 0, 0), width=2)
            save_panel([frame_img, event_img, info], output_dir / f"{sample_id}.png", labels=["frame", "event", "prediction"])
            produced += 1
            if produced >= args.limit:
                return


if __name__ == "__main__":
    main()
