#!/usr/bin/env python3
"""Train the TENNs-Eye TennSt model on the HBTXR subject-independent split.

The upstream script depends on Hydra/OmegaConf and a 3ET directory export. This
runner keeps the original TENNs-Eye model/loss code, but reads the already-built
FACET/HBTXR 64x64 subject-independent cache directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import yaml
from timm.scheduler.step_lr import StepLRScheduler
from torch.utils.data import DataLoader


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[5]
    tenn_root = repo_root / "references" / "codebase" / "software" / "ais2024" / "eye_track_spatiotemporal"
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--tenn-root", type=Path, default=tenn_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=tenn_root / "config_hbtxr_subject_independent_img64.yaml",
    )
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--run-name", default="TENNs_Eye_subject_independent_img64")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true", default=False)
    return parser


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def make_loader(dataset_cls, root_path: Path, split: str, cfg: dict, batch_size: int, num_workers: int):
    dataset_cfg = cfg["dataset"]
    dataset = dataset_cls(
        root_path=root_path,
        split=split,
        frames_per_segment=int(dataset_cfg["frames_per_segment"]),
        stride=int(dataset_cfg["frames_per_segment"]),
        sensor_size=(346, 260, 2),
        events_interpolation=dataset_cfg.get("events_interpolation", "causal_linear_ori"),
        pupil_area=200,
        default_resolution=tuple(dataset_cfg.get("sensor_size", [64, 64])),
        temporal_transform=split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        drop_last=split == "train",
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def run_epoch(model, loader, loss_fn, metric_fn, optimizer, device: torch.device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_p10 = 0.0
    total_p10_noblinks = 0.0
    total_distance = 0.0
    num_batches = 0

    for event, center, openness in loader:
        event = event.to(device, non_blocking=True)
        center = center.to(device, non_blocking=True)
        openness = openness.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(event)
            loss = loss_fn(pred, center, openness)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        metric, metric_noblinks, distance = metric_fn(pred, center, openness)
        total_loss += float(loss.detach().cpu())
        total_p10 += float(metric.detach().cpu())
        total_p10_noblinks += float(metric_noblinks.detach().cpu())
        total_distance += float(distance.detach().cpu())
        num_batches += 1

        if num_batches == 1 and loader.dataset.split == "train" and getattr(loader.dataset, "fast_dev_run", False):
            break

    denom = max(1, num_batches)
    return {
        "loss": total_loss / denom,
        "p10": total_p10 / denom,
        "p10_noblinks": total_p10_noblinks / denom,
        "distance": total_distance / denom,
    }


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)

    sys.path.insert(0, str(args.tenn_root))
    sys.path.insert(0, str(args.repo_root / "references" / "codebase" / "software" / "FACET"))

    from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseCenterSequenceDataset import (  # noqa: PLC0415
        DavisEyeEllipseCenterSequenceDataset,
    )
    import losses  # noqa: PLC0415
    from tenn_model import TennSt  # noqa: PLC0415

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)

    trainer_cfg = cfg["trainer"]
    epochs = int(args.max_epochs or trainer_cfg["epochs"])
    batch_size = int(args.batch_size or trainer_cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else trainer_cfg.get("num_workers", 4))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(DavisEyeEllipseCenterSequenceDataset, args.root_path, "train", cfg, batch_size, num_workers)
    val_loader = make_loader(DavisEyeEllipseCenterSequenceDataset, args.root_path, "val", cfg, batch_size, num_workers)

    if args.fast_dev_run:
        train_loader.dataset.fast_dev_run = True
        val_loader.dataset.fast_dev_run = True
        epochs = 1

    model = TennSt(**cfg["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(trainer_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(trainer_cfg.get("weight_decay", 1e-5)),
    )
    scheduler = StepLRScheduler(
        optimizer,
        decay_t=10,
        decay_rate=0.7,
        warmup_lr_init=1e-5,
        warmup_t=5,
    )
    loss_fn = losses.Losses(
        cfg["model"].get("detector_head", True),
        float(trainer_cfg.get("activity_regularization", 0)),
        model,
    )
    metric_fn = partial(
        losses.p10_acc,
        detector_head=cfg["model"].get("detector_head", True),
        height=int(cfg["dataset"].get("sensor_size", [64, 64])[1]),
        width=int(cfg["dataset"].get("sensor_size", [64, 64])[0]),
        tolerance=10,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.tenn_root / "runs" / f"{args.run_name}_{timestamp}"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with (out_dir / "run_args.json").open("w") as f:
        json.dump({**vars(args), "root_path": str(args.root_path), "device": str(device)}, f, indent=2, default=str)

    best_distance = float("inf")
    print(f"[TENNs-Eye] output_dir={out_dir}", flush=True)
    print(f"[TENNs-Eye] train_segments={len(train_loader.dataset)} val_segments={len(val_loader.dataset)}", flush=True)
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, loss_fn, metric_fn, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, loss_fn, metric_fn, optimizer, device, train=False)
        scheduler.step(epoch=epoch)

        print(
            "[TENNs-Eye] "
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_p10={train_metrics['p10']:.4f} "
            f"train_distance={train_metrics['distance']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_p10={val_metrics['p10']:.4f} "
            f"val_distance={val_metrics['distance']:.4f}",
            flush=True,
        )

        torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(), "metrics": val_metrics}, ckpt_dir / "last.pth")
        if val_metrics["distance"] < best_distance:
            best_distance = val_metrics["distance"]
            torch.save(
                {"epoch": epoch + 1, "state_dict": model.state_dict(), "metrics": val_metrics},
                ckpt_dir / f"best_epoch{epoch + 1:03d}_val_distance_{best_distance:.4f}.pth",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
