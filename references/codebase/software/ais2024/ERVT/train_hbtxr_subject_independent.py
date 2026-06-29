#!/usr/bin/env python3
"""Train the ERVT/RVT model on the HBTXR subject-independent split.

The upstream ERVT script depends on MLflow, torchinfo, and a 3ET event export.
This runner keeps the original RVT model while reading the FACET/HBTXR 64x64
subject-independent cache directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from timm.scheduler.step_lr import StepLRScheduler
from torch.utils.data import DataLoader


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[5]
    ervt_root = repo_root / "references" / "codebase" / "software" / "ais2024" / "ERVT"
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--ervt-root", type=Path, default=ervt_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=ervt_root / "configs" / "hbtxr_subject_independent_img64.json",
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
    parser.add_argument("--run-name", default="ERVT_subject_independent_img64")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true", default=False)
    return parser


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def make_loader(dataset_cls, root_path: Path, split: str, cfg: dict, batch_size: int, num_workers: int):
    frames_per_segment = int(cfg["train_length"] if split == "train" else cfg["val_length"])
    stride = int(cfg["train_stride"] if split == "train" else cfg["val_stride"])
    dataset = dataset_cls(
        root_path=root_path,
        split=split,
        frames_per_segment=frames_per_segment,
        stride=stride,
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear_ori",
        pupil_area=200,
        default_resolution=(int(cfg["sensor_width"]), int(cfg["sensor_height"])),
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


class WeightedRMSE(torch.nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((inputs - targets).pow(2) * self.weights))


def pixel_metrics(target: torch.Tensor, prediction: torch.Tensor, width: int, height: int, tolerances: list[int]):
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)
    dis = target - prediction
    dis[:, 0] *= width
    dis[:, 1] *= height
    dist = torch.norm(dis, dim=-1)
    acc = {f"p{tol}": float((dist < tol).float().mean().detach().cpu()) for tol in tolerances}
    return acc, float(dist.mean().detach().cpu())


def to_ervt_batch(event: torch.Tensor, center: torch.Tensor, device: torch.device):
    inputs = event.moveaxis(1, 2).to(device, non_blocking=True)
    targets = center.moveaxis(1, 2).to(device, non_blocking=True)
    return inputs, targets


def train_epoch(model, loader, criterion, optimizer, cfg, device: torch.device):
    model.train()
    total_loss = 0.0
    total_distance = 0.0
    total_p10 = 0.0
    batches = 0
    chunks = [int(cfg["tbptt"])] * (int(cfg["train_length"]) // int(cfg["tbptt"]))

    for event, center, _openness in loader:
        inputs, targets = to_ervt_batch(event, center, device)
        split_inputs = torch.split(inputs, chunks, dim=1)
        split_targets = torch.split(targets, chunks, dim=1)

        hidden = None
        seq_loss = 0.0
        acc_outputs = []
        optimizer.zero_grad(set_to_none=True)

        for x, y in zip(split_inputs, split_targets):
            outputs, hidden = model(x, hidden)
            loss = criterion(outputs, y[:, :, :2])
            seq_loss += float(loss.detach().cpu())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            acc_outputs.append(outputs.detach())
            for i in range(len(hidden)):
                hidden[i] = (hidden[i][0].detach(), hidden[i][1].detach())

        outputs = torch.cat(acc_outputs, dim=1)
        acc, distance = pixel_metrics(
            targets.detach().cpu(),
            outputs.detach().cpu(),
            int(cfg["sensor_width"]),
            int(cfg["sensor_height"]),
            list(cfg["pixel_tolerances"]),
        )
        total_loss += seq_loss / max(1, len(chunks))
        total_distance += distance
        total_p10 += acc.get("p10", 0.0)
        batches += 1

        if getattr(loader.dataset, "fast_dev_run", False):
            break

    denom = max(1, batches)
    return {"loss": total_loss / denom, "distance": total_distance / denom, "p10": total_p10 / denom}


@torch.no_grad()
def validate_epoch(model, loader, criterion, cfg, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_distance = 0.0
    total_p10 = 0.0
    batches = 0

    for event, center, _openness in loader:
        inputs, targets = to_ervt_batch(event, center, device)
        outputs, _hidden = model(inputs)
        loss = criterion(outputs, targets[:, :, :2])
        acc, distance = pixel_metrics(
            targets.detach().cpu(),
            outputs.detach().cpu(),
            int(cfg["sensor_width"]),
            int(cfg["sensor_height"]),
            list(cfg["pixel_tolerances"]),
        )
        total_loss += float(loss.detach().cpu())
        total_distance += distance
        total_p10 += acc.get("p10", 0.0)
        batches += 1

        if getattr(loader.dataset, "fast_dev_run", False):
            break

    denom = max(1, batches)
    return {"loss": total_loss / denom, "distance": total_distance / denom, "p10": total_p10 / denom}


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_json(args.config)
    cfg["in_channels"] = 2
    cfg["device"] = args.device

    sys.path.insert(0, str(args.ervt_root))
    sys.path.insert(0, str(args.repo_root / "references" / "codebase" / "software" / "FACET"))

    from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseCenterSequenceDataset import (  # noqa: PLC0415
        DavisEyeEllipseCenterSequenceDataset,
    )
    from model.RVT import RVT  # noqa: PLC0415

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(int(cfg.get("seed", 42)))

    epochs = int(args.max_epochs or cfg["num_epochs"])
    batch_size = int(args.batch_size or cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(DavisEyeEllipseCenterSequenceDataset, args.root_path, "train", cfg, batch_size, num_workers)
    val_loader = make_loader(DavisEyeEllipseCenterSequenceDataset, args.root_path, "val", cfg, batch_size, num_workers)
    if args.fast_dev_run:
        train_loader.dataset.fast_dev_run = True
        val_loader.dataset.fast_dev_run = True
        epochs = 1

    model = RVT(SimpleNamespace(**cfg)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = StepLRScheduler(
        optimizer,
        decay_t=10,
        decay_rate=0.7,
        warmup_lr_init=1e-5,
        warmup_t=5,
    )
    criterion = WeightedRMSE(torch.tensor((1.0, 1.0), device=device))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.ervt_root / "runs" / f"{args.run_name}_{timestamp}"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    with (out_dir / "run_args.json").open("w") as f:
        json.dump({**vars(args), "root_path": str(args.root_path), "device": str(device)}, f, indent=2, default=str)

    best_distance = float("inf")
    print(f"[ERVT] output_dir={out_dir}", flush=True)
    print(f"[ERVT] train_segments={len(train_loader.dataset)} val_segments={len(val_loader.dataset)}", flush=True)
    print("[ERVT] note=in_channels set to 2 for direct HBTXR two-polarity event frames", flush=True)

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, cfg, device)
        val_metrics = validate_epoch(model, val_loader, criterion, cfg, device)
        scheduler.step(epoch=epoch)
        print(
            "[ERVT] "
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
