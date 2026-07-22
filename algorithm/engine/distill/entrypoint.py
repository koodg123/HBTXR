"""Distillation entrypoint for the reimplemented HBTXR models (P5.8).

    load_config -> build teacher (cfg['distill']['teacher'] model block + checkpoint,
    frozen) + student (cfg['model']) -> build_optimizer/scheduler on the student ->
    build_dataloader -> DistillTrainer.fit -> save student.

Run from the ``algorithm/`` root::

    python -m engine.distill.entrypoint -c configs/experiment/frame_distill.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.optim.registry import build_optimizer
from common.schedulers import build_lr_scheduler
from engine.data.adapter import resolve_modality
from engine.data.factory import build_dataloader
from engine.distill.distiller import DistillConfig, DistillTrainer
from engine.model_factory import make_model
from engine.runspec.run_contract import resolve_training_entry
from engine.tools.checkpoint import load_checkpoint, save_checkpoint
from engine.tools.load_config import load_config


def _resolve_project_root(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    root = (cfg.get("experiment") or {}).get("project_root")
    return Path(root).expanduser().resolve() if root else Path.cwd()


def _build_teacher(distill_cfg: dict[str, Any], device: str):
    teacher_cfg = distill_cfg.get("teacher") or {}
    model_block = teacher_cfg.get("model", teacher_cfg)
    if not model_block:
        raise ValueError("distill.teacher.model (or distill.teacher) config block is required")
    teacher = make_model(model_block)
    ckpt = teacher_cfg.get("ckpt_path")
    if ckpt:
        load_checkpoint(teacher, ckpt, map_location=device)
    return teacher


def run_distill(
    config_path: str,
    *,
    project_root: str | None = None,
    device: str | None = None,
) -> DistillTrainer:
    """Distill a student (cfg['model']) from a frozen teacher (cfg['distill'])."""
    cfg = load_config(config_path)
    root = _resolve_project_root(cfg, project_root)
    entry = resolve_training_entry(cfg, config_path=config_path, project_root=root)

    modality = resolve_modality(cfg)
    training_cfg = cfg.get("training") or {}
    distill_cfg = cfg.get("distill") or {}
    epochs = int(training_cfg.get("epochs", training_cfg.get("max_epochs", 1)))
    dev = device or str(training_cfg.get("device", "cpu"))

    student = make_model(cfg["model"])
    teacher = _build_teacher(distill_cfg, dev)

    optimizer, _resolved, optimizer_meta, _summary = build_optimizer(student, cfg)
    scheduler = build_lr_scheduler(optimizer, training_cfg, epochs, optimizer_meta=optimizer_meta)
    loss_weights = training_cfg.get("loss_weights") or (cfg.get("loss") or {}).get("weights")

    config = DistillConfig(
        modality=modality,
        epochs=epochs,
        device=dev,
        loss_weights=loss_weights if isinstance(loss_weights, dict) else None,
        distill_weight=float(distill_cfg.get("weight", 1.0)),
        distill_weights=distill_cfg.get("weights") if isinstance(distill_cfg.get("weights"), dict) else None,
    )
    trainer = DistillTrainer(student, teacher, optimizer, config, scheduler=scheduler)

    train_loader = build_dataloader(entry["train_manifest"], cfg, shuffle=True, modality=modality)
    trainer.fit(train_loader)

    output_dir = entry.get("output_dir")
    if output_dir:
        save_checkpoint(
            student,
            Path(output_dir) / "student_final.pt",
            meta={"modality": modality, "epochs": epochs, "distilled": True, "config_path": str(config_path)},
        )
    return trainer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Distill a reimplemented HBTXR student from a teacher.")
    parser.add_argument("-c", "--config", required=True, help="experiment config path (yaml/json)")
    parser.add_argument("--project-root", default=None, help="root for manifest/output resolution")
    parser.add_argument("--device", default=None, help="override training device")
    args = parser.parse_args(argv)
    run_distill(args.config, project_root=args.project_root, device=args.device)


if __name__ == "__main__":
    main()
