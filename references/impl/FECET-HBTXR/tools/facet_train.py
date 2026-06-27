from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning
import torch

from _bootstrap import PROJECT_ROOT
from facet_common import collect_model_checkpoint_paths, import_facet_module, load_facet_config, patch_facet_config, resolve_facet_repo_root, trainer_device_kwargs

from fecet_hbtxr.io import write_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the original FACET EPNet on the FACET reference-compatible dataset")
    parser.add_argument("--facet-root", type=str, default=str(resolve_facet_repo_root()))
    parser.add_argument("--config", type=str, default="DavisEyeEllipse_EPNet.yaml")
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "data" / "facet_reference"))
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "runs" / "comparison" / "facet"))
    parser.add_argument("--experiment-name", type=str, default="facet_epnet_gsam_compare")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=240)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> dict:
    lightning.seed_everything(int(args.seed))
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("medium")

    facet_root = resolve_facet_repo_root(args.facet_root)
    cfg = load_facet_config(args.config, facet_repo_root=facet_root)
    cfg = patch_facet_config(
        cfg,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        sensor_size_whc=(int(args.sensor_width), int(args.sensor_height), 2),
        default_resolution=(int(args.input_width), int(args.input_height)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        checkpoint_path=args.resume_checkpoint,
    )

    dataset_factory = import_facet_module(facet_root, "EvEye.dataset.dataset_factory")
    model_factory = import_facet_module(facet_root, "EvEye.model.model_factory")
    logger_factory = import_facet_module(facet_root, "EvEye.logger.logger_factory")
    callback_factory = import_facet_module(facet_root, "EvEye.callback.callback_factory")

    train_loader = dataset_factory.make_dataloader(cfg["dataloader"]["train"])
    val_loader = dataset_factory.make_dataloader(cfg["dataloader"]["val"])
    model = model_factory.make_model(dict(cfg["model"]))
    if "optimizer" in (cfg.get("train") or {}):
        model.set_optimizer_config(**cfg["train"]["optimizer"])

    trainer = lightning.Trainer(
        default_root_dir=str((Path(args.output_root) / args.experiment_name).resolve()),
        max_epochs=int((cfg.get("train") or {}).get("max_epochs", 50)),
        check_val_every_n_epoch=int((cfg.get("train") or {}).get("check_val_every_n_epoch", 1)),
        logger=logger_factory.make_logger(cfg["logger"]),
        callbacks=callback_factory.make_callbacks(cfg["callback"]),
        **trainer_device_kwargs(args.device),
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_checkpoint)

    summary = {
        "facet_root": str(facet_root),
        "config": str(args.config),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "experiment_name": str(args.experiment_name),
        "device": str(args.device),
        "checkpoint_paths": collect_model_checkpoint_paths(list(trainer.callbacks)),
    }
    summary_path = Path(args.output_root).resolve() / args.experiment_name / "facet_train_summary.json"
    write_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
