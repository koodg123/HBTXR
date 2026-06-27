from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from _config import apply_config_overrides, load_config, resolve_training_entry

from fecet_hbtxr.trainer import build_model, epoch_loop, load_checkpoint, make_loader, resolve_device_and_wrap


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a FECET-HBTXR checkpoint")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stage2_hybrid.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
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
    stats = epoch_loop(
        model=model,
        loader=loader,
        optimizer=None,
        scaler=None,
        device=device,
        stage=str((cfg.get("training") or {}).get("stage", "stage2")),
        loss_cfg=cfg.get("loss") or {},
        amp_enabled=False,
        grad_accum_steps=1,
    )
    result = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(manifest).resolve()),
        "metrics": {key: float(value) for key, value in stats.items()},
    }
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
