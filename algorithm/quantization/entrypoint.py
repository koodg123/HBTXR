"""Quantization entrypoint for the HBTXR models (PTQ / QAT).

    load_config -> make_model -> load_checkpoint (trained fp model) -> build a few
    calibration batches from the data pipeline -> PTQ (post_training_quantize) or
    QAT (prepare_qat + fine-tune via engine.train.Trainer) -> save quantized ckpt.

Config block ``quantization``: ``mode`` (ptq|qat), ``weight``/``activation`` specs
(or legacy ``weight_bits``/``act_bits``), ``overrides``, ``skip`` (module-name
substrings left fp), ``num_calib_batches``, ``ckpt_path``, ``output``,
``export_int``. Uses the same experiment schema as training
(see configs/experiment/*_quant.yaml).

With ``export_int`` (or ``--export-int DIR``) the run continues past the quantized
checkpoint into the deployment tier: ``convert_model_to_integer`` on the same
calibration batches — which installs the I-tier kernels whose whole datapath is
integer, not the Q-tier LUTs that still reduce in float — then dumps integer
weights + scales + tables and a ``manifest.json`` for the HW / bit-exact-simulator
backend. Every op becomes integer; the graph does not (see QUANTIZATION-PLAN §9).

Run from the ``algorithm/`` root::

    python -m quantization.entrypoint -c configs/experiment/frame_quant.yaml --ckpt runs/.../final.pt
    python -m quantization.entrypoint -c configs/experiment/frame_quant.yaml \
        --ckpt runs/.../final.pt --export-int runs/frame_hbtxr/int_export
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
from quantization.convert import convert_model_to_integer
from quantization.export import export_integer_model
from quantization.qat import prepare_qat
from quantization.spec import QuantScheme


def _make_forward_fn(modality: str) -> Callable[[Any, dict], Any]:
    if modality.startswith("hybrid"):
        return lambda model, batch: model.search_step(batch["frame"])
    return lambda model, batch: model(batch["image"])


def _project_root(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    root = (cfg.get("experiment") or {}).get("project_root")
    return Path(root).expanduser().resolve() if root else Path.cwd()


def _export_integer(model: Any, export_dir: str, calib_batches: list, forward_fn: Callable) -> dict[str, Any]:
    """Q -> I deployment tier: whole-graph integer conversion, then dump artifacts.

    Uses ``convert_model_to_integer`` (not the Linear-only ``convert_to_integer``), so
    Conv2d, GeLU, LayerNorm and Softmax become I-tier modules whose *whole* datapath is
    integer — integer mean/variance/rsqrt, integer row max/exp/row-sum/reciprocal —
    rather than Q-tier LUTs that still reduce in float.

    Runs after the quantized checkpoint is saved, so the export never perturbs what
    existing callers get back on disk. Returns the manifest so a caller can inspect or
    re-verify the dump without re-reading it from disk.

    Both honesty knobs are load-bearing. ``ConversionReport.left_float`` /
    ``float_composites`` are printed because the conversion is integer *per op*, not
    end to end: tensors still move between modules as float, and the attention matmuls
    and residual adds live inside module ``forward`` bodies that no swap can reach.
    ``allow_unquantized=True`` is likewise not a way to silence the export — a real
    HBTXR model keeps a padded float conv (the mask head), which is recorded in
    ``manifest['unexported']`` and warned about instead of aborting the run. A
    *quantized* module with no export branch still aborts unconditionally; this opt-in
    does not reach it.
    """
    model, report = convert_model_to_integer(model, calib_batches, forward_fn=forward_fn)
    manifest = export_integer_model(model, Path(export_dir), allow_unquantized=True)
    print(f"[quantize] integer export -> {export_dir}: {len(report)} integer modules, "
          f"{manifest['num_modules']} exported {manifest['counts']}")
    if report.left_float or report.float_composites:
        print(f"[quantize] still float: {len(report.left_float)} leaf module(s), "
              f"{len(report.float_composites)} composite forward(s) "
              f"(residual adds / attention matmuls are written inline, not as submodules)")
    if manifest["unexported"]:
        listing = ", ".join(f"{r['name']}({r['class']})" for r in manifest["unexported"])
        print(f"[quantize] WARNING: {manifest['num_unexported']} module(s) left OUT of the "
              f"integer dump (still float): {listing}")
    return manifest


def run_quantize(
    config_path: str,
    *,
    ckpt: str | None = None,
    output: str | None = None,
    project_root: str | None = None,
    device: str | None = None,
    export_int: str | None = None,
    return_manifest: bool = False,
) -> Any:
    """PTQ or QAT of an HBTXR model per the ``quantization`` config block.

    With ``export_int`` (or ``quantization.export_int``) the quantized model is also
    converted to the integer tier and dumped as HW artifacts (see quantization.export).

    Returns the quantized model. With ``return_manifest=True`` it returns
    ``(model, manifest)`` instead, where ``manifest`` is the export manifest (``None``
    when no export ran) — the export's own description of what it wrote, which is
    otherwise only recoverable by re-reading ``manifest.json`` off disk.
    """
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
    scheme = QuantScheme.from_config(quant_cfg)
    n_calib = int(quant_cfg.get("num_calib_batches", 8))

    calib_loader = build_dataloader(entry["train_manifest"], cfg, shuffle=False, modality=modality)
    calib_batches = list(itertools.islice(calib_loader, n_calib))

    if mode == "qat":
        model, _registry = prepare_qat(model, scheme=scheme, calib_batches=calib_batches, forward_fn=forward_fn, device=dev)
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
        model, _registry = post_training_quantize(model, calib_batches, scheme=scheme, forward_fn=forward_fn, device=dev)

    out_path = output or quant_cfg.get("output")
    if out_path:
        save_checkpoint(model, Path(out_path), meta={"quantized": True, "mode": mode, "modality": modality,
                                                     "weight_bits": scheme.weight.bits, "act_bits": scheme.activation.bits})

    export_dir = export_int or quant_cfg.get("export_int")
    manifest = _export_integer(model, str(export_dir), calib_batches, forward_fn) if export_dir else None
    return (model, manifest) if return_manifest else model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantize a reimplemented HBTXR model (PTQ / QAT).")
    parser.add_argument("-c", "--config", required=True, help="experiment config path with a quantization block")
    parser.add_argument("--ckpt", default=None, help="trained fp checkpoint to quantize")
    parser.add_argument("-o", "--output", default=None, help="quantized checkpoint output path")
    parser.add_argument("--project-root", default=None, help="root for manifest resolution")
    parser.add_argument("--device", default=None, help="device")
    parser.add_argument("--export-int", default=None, metavar="DIR",
                        help="also convert to the integer tier and dump HW artifacts into DIR")
    args = parser.parse_args(argv)
    run_quantize(args.config, ckpt=args.ckpt, output=args.output, project_root=args.project_root,
                 device=args.device, export_int=args.export_int)


if __name__ == "__main__":
    main()
