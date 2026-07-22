"""Compression (pruning) entrypoint for the reimplemented HBTXR models (P5.9).

    load_config -> make_model -> load_checkpoint -> prune_model -> sparsity_report
    -> save pruned checkpoint (for a downstream fine-tune via engine.train / engine.distill).

Config block ``pruning`` (or ``compress``): ``amount`` (fraction removed),
``structured`` (bool), ``n`` / ``dim`` (Ln-structured params), ``ckpt_path``.

Run from the ``algorithm/`` root::

    python -m engine.compress.entrypoint -c configs/experiment/frame_hbtxr.yaml \
        --ckpt runs/.../final.pt -o runs/.../pruned.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from engine.compress.pruning import prune_model, sparsity_report
from engine.model_factory import make_model
from engine.tools.checkpoint import load_checkpoint, save_checkpoint
from engine.tools.load_config import load_config


def run_compress(
    config_path: str,
    *,
    ckpt: str | None = None,
    output: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Prune a checkpoint and report sparsity; optionally save the pruned model."""
    cfg = load_config(config_path)
    prune_cfg = cfg.get("pruning") or cfg.get("compress") or {}
    dev = device or str((cfg.get("training") or {}).get("device", "cpu"))

    model = make_model(cfg["model"])
    ckpt_path = ckpt or prune_cfg.get("ckpt_path")
    if ckpt_path:
        load_checkpoint(model, ckpt_path, map_location=dev)

    amount = float(prune_cfg.get("amount", 0.3))
    prune_model(
        model,
        amount,
        structured=bool(prune_cfg.get("structured", False)),
        n=int(prune_cfg.get("n", 2)),
        dim=int(prune_cfg.get("dim", 0)),
        make_permanent=True,
    )
    report = sparsity_report(model)
    print(json.dumps({k: v for k, v in report.items() if k != "per_layer"}, indent=2))

    out_path = output or prune_cfg.get("output")
    if out_path:
        save_checkpoint(
            model,
            Path(out_path),
            meta={"pruned": True, "amount": amount, "global_sparsity": report["global_sparsity"]},
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prune a reimplemented HBTXR model.")
    parser.add_argument("-c", "--config", required=True, help="experiment config path (yaml/json)")
    parser.add_argument("--ckpt", default=None, help="checkpoint to prune (overrides config)")
    parser.add_argument("-o", "--output", default=None, help="pruned checkpoint output path")
    parser.add_argument("--device", default=None, help="map_location device")
    args = parser.parse_args(argv)
    run_compress(args.config, ckpt=args.ckpt, output=args.output, device=args.device)


if __name__ == "__main__":
    main()
