"""Quantization entrypoint for the HBTXR models (PTQ / QAT).

    load_config -> make_model -> load_checkpoint (trained fp model) -> build a few
    calibration batches from the data pipeline -> PTQ (post_training_quantize) or
    QAT (prepare_qat + fine-tune via engine.train.Trainer) -> save quantized ckpt.

Config block ``quantization``: ``mode`` (ptq|qat), ``weight_bits``, ``act_bits``,
``skip`` (module-name substrings left fp), ``num_calib_batches``, ``ckpt_path``,
``output``. Uses the same experiment schema as training (see configs/experiment/*_quant.yaml).

Run from the ``algorithm/`` root::

    python -m quantization.entrypoint -c configs/experiment/frame_quant.yaml --ckpt runs/.../final.pt
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Callable

from common.optim.registry import build_optimizer
from common.schedulers import build_lr_scheduler
from engine.data.adapter import resolve_modality
from engine.data.factory import build_dataloader
from engine.model_factory import make_model
from engine.runspec.run_contract import resolve_training_entry
from engine.tools.checkpoint import load_checkpoint, save_checkpoint
from engine.tools.load_config import load_config
from engine.train.trainer import Trainer, TrainConfig

from quantization.calibrate import post_training_quantize
from quantization.qat import prepare_qat
from quantization.qlayers.linear import QuantConfig
from quantization.scheme import QuantDtype


def _make_forward_fn(modality: str) -> Callable[[Any, dict], Any]:
    if modality.startswith("hybrid"):
        return lambda model, batch: model.search_step(batch["frame"])
    return lambda model, batch: model(batch["image"])


def _quant_config(quant_cfg: dict[str, Any]) -> QuantConfig:
    return QuantConfig(
        weight_dtype=QuantDtype(int(quant_cfg.get("weight_bits", 8)), signed=True),
        act_dtype=QuantDtype(int(quant_cfg.get("act_bits", 8)), signed=True),
        skip=tuple(quant_cfg.get("skip", []) or []),
    )


def _project_root(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    root = (cfg.get("experiment") or {}).get("project_root")
    return Path(root).expanduser().resolve() if root else Path.cwd()


def run_quantize(
    config_path: str,
    *,
    ckpt: str | None = None,
    output: str | None = None,
    project_root: str | None = None,
    device: str | None = None,
) -> Any:
    """PTQ or QAT of an HBTXR model per the ``quantization`` config block."""
    cfg = load_config(config_path)
    quant_cfg = cfg.get("quantization") or {}
    mode = str(quant_cfg.get("mode", "ptq")).lower()
    modality = resolve_modality(cfg)
    root = _project_root(cfg, project_root)
    dev = device or str((cfg.get("training") or {}).get("device", "cpu"))

    model = make_model(cfg["model"])
    ckpt_path = ckpt or quant_cfg.get("ckpt_path")
    if ckpt_path:
        load_checkpoint(model, ckpt_path, map_location=dev)

    entry = resolve_training_entry(cfg, config_path=config_path, project_root=root)
    forward_fn = _make_forward_fn(modality)
    qconfig = _quant_config(quant_cfg)
    n_calib = int(quant_cfg.get("num_calib_batches", 8))

    calib_loader = build_dataloader(entry["train_manifest"], cfg, shuffle=False, modality=modality)
    calib_batches = list(itertools.islice(calib_loader, n_calib))

    if mode == "qat":
        model, _registry = prepare_qat(model, config=qconfig, calib_batches=calib_batches, forward_fn=forward_fn, device=dev)
        training_cfg = cfg.get("training") or {}
        epochs = int(training_cfg.get("epochs", training_cfg.get("max_epochs", 1)))
        optimizer, _resolved, optimizer_meta, _summary = build_optimizer(model, cfg)
        scheduler = build_lr_scheduler(optimizer, training_cfg, epochs, optimizer_meta=optimizer_meta)
        loss_weights = training_cfg.get("loss_weights") or (cfg.get("loss") or {}).get("weights")
        trainer = Trainer(
            model,
            optimizer,
            TrainConfig(modality=modality, epochs=epochs, device=dev, loss_weights=loss_weights if isinstance(loss_weights, dict) else None),
            scheduler=scheduler,
        )
        trainer.fit(build_dataloader(entry["train_manifest"], cfg, shuffle=True, modality=modality))
    else:
        model, _registry = post_training_quantize(model, calib_batches, config=qconfig, forward_fn=forward_fn, device=dev)

    out_path = output or quant_cfg.get("output")
    if out_path:
        save_checkpoint(model, Path(out_path), meta={"quantized": True, "mode": mode, "modality": modality,
                                                     "weight_bits": qconfig.weight_dtype.bits, "act_bits": qconfig.act_dtype.bits})
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantize a reimplemented HBTXR model (PTQ / QAT).")
    parser.add_argument("-c", "--config", required=True, help="experiment config path with a quantization block")
    parser.add_argument("--ckpt", default=None, help="trained fp checkpoint to quantize")
    parser.add_argument("-o", "--output", default=None, help="quantized checkpoint output path")
    parser.add_argument("--project-root", default=None, help="root for manifest resolution")
    parser.add_argument("--device", default=None, help="device")
    args = parser.parse_args(argv)
    run_quantize(args.config, ckpt=args.ckpt, output=args.output, project_root=args.project_root, device=args.device)


if __name__ == "__main__":
    main()
