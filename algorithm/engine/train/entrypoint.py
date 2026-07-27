"""Training entrypoint for the reimplemented HBTXR models (P5.6).

Replaces the old Lightning-coupled ``apps/train.py`` with a thin launcher over the
new custom stack:

    load_config -> make_model (engine.model_factory) -> build_optimizer /
    build_lr_scheduler (common.optim / common.schedulers) -> build_dataloader
    (engine.data.factory, HBTXR pipeline + model-input adapter) -> Trainer.fit
    (engine.train).

The modality (frame / event / hybrid / mask) is inferred from ``cfg['model']``.
Full experiment configs carry ``experiment`` / ``data`` / ``training`` / ``model``
blocks (composed under configs/ in P6); the modality files under
configs/modality/ are the ``model`` fragment only.

Run from the ``algorithm/`` root::

    python -m engine.train.entrypoint -c configs/experiment/frame_hbtxr.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.optim.registry import build_optimizer
from common.schedulers import build_lr_scheduler
from engine.data.adapter import resolve_modality
from engine.model_factory import make_model
from engine.runspec.run_contract import resolve_training_entry
from engine.tools.checkpoint import save_checkpoint
from engine.tools.load_config import load_config
from engine.train.trainer import Trainer, TrainConfig

if TYPE_CHECKING:  # engine.data.factory is only imported at call time (see below)
    from engine.data.factory import AdaptedLoader


def _build_dataloader(
    manifest_path: str,
    cfg: dict[str, Any],
    *,
    shuffle: bool,
    modality: str | None = None,
) -> AdaptedLoader:
    """Lazy proxy to ``engine.data.factory.build_dataloader``.

    ``engine.data.factory`` pulls the full HBTXR data pipeline (PIL / cv2 / h5py), which
    made merely *importing* this module — and therefore the ``hbtxr train`` dispatch —
    fail wherever those are absent. Kept at module scope rather than inlined into
    ``run_train`` so it stays patchable without importing the pipeline to reach it.
    """
    from engine.data.factory import build_dataloader

    return build_dataloader(manifest_path, cfg, shuffle=shuffle, modality=modality)


def _resolve_project_root(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    experiment_cfg = cfg.get("experiment") or {}
    root = experiment_cfg.get("project_root")
    return Path(root).expanduser().resolve() if root else Path.cwd()


def _resolve_epochs(training_cfg: dict[str, Any]) -> int:
    return int(training_cfg.get("epochs", training_cfg.get("max_epochs", 1)))


def _resolve_loss_weights(cfg: dict[str, Any]) -> dict[str, float] | None:
    training_cfg = cfg.get("training") or {}
    weights = training_cfg.get("loss_weights") or (cfg.get("loss") or {}).get("weights")
    return dict(weights) if isinstance(weights, dict) else None


def run_train(
    config_path: str,
    *,
    project_root: str | None = None,
    device: str | None = None,
) -> Trainer:
    """Build the model + data + optimizer from ``config_path`` and run Trainer.fit."""
    cfg = load_config(config_path)
    root = _resolve_project_root(cfg, project_root)
    entry = resolve_training_entry(cfg, config_path=config_path, project_root=root)

    modality = resolve_modality(cfg)
    training_cfg = cfg.get("training") or {}
    epochs = _resolve_epochs(training_cfg)
    dev = device or str(training_cfg.get("device", "cpu"))

    model = make_model(cfg["model"])
    optimizer, _resolved, optimizer_meta, _summary = build_optimizer(model, cfg)
    scheduler = build_lr_scheduler(optimizer, training_cfg, epochs, optimizer_meta=optimizer_meta)

    train_loader = _build_dataloader(entry["train_manifest"], cfg, shuffle=True, modality=modality)

    train_config = TrainConfig(
        modality=modality,
        epochs=epochs,
        device=dev,
        loss_weights=_resolve_loss_weights(cfg),
    )
    trainer = Trainer(model, optimizer, train_config, scheduler=scheduler)
    trainer.fit(train_loader)

    output_dir = entry.get("output_dir")
    if output_dir:
        save_checkpoint(
            model,
            Path(output_dir) / "final.pt",
            meta={"modality": modality, "epochs": epochs, "config_path": str(config_path)},
        )
    return trainer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a reimplemented HBTXR model.")
    parser.add_argument("-c", "--config", required=True, help="experiment config path (yaml/json)")
    parser.add_argument("--project-root", default=None, help="root for manifest/output resolution")
    parser.add_argument("--device", default=None, help="override training device (e.g. cpu, cuda)")
    args = parser.parse_args(argv)
    run_train(args.config, project_root=args.project_root, device=args.device)


if __name__ == "__main__":
    main()
