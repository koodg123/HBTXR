from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
FACET_ROOT = ROOT / "references/codebase/software/FACET"
if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))

from EvEye.utils.cache.MemmapCacheStructedEvents import load_event_segment


SENSOR_W = 346.0
SENSOR_H = 260.0
FRAME_SIZE = 128
EVENT_SIZE = 64
PATCH_SIZE = 16
PATCH_NUM = 8


def read_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def make_event64(events: np.ndarray) -> np.ndarray:
    volume = np.zeros((2, EVENT_SIZE, EVENT_SIZE), dtype=np.uint16)
    if len(events) == 0:
        return volume
    xs = np.clip((events["x"].astype(np.float32) * EVENT_SIZE / SENSOR_W).astype(np.int64), 0, EVENT_SIZE - 1)
    ys = np.clip((events["y"].astype(np.float32) * EVENT_SIZE / SENSOR_H).astype(np.int64), 0, EVENT_SIZE - 1)
    ps = (events["p"].astype(np.int64) > 0).astype(np.int64)
    np.add.at(volume, (ps, ys, xs), 1)
    return volume


def make_mask128(ellipse128: list[float]) -> np.ndarray:
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    x, y, a, b, angle = ellipse128
    center = (int(round(x)), int(round(y)))
    axes = (max(1, int(round(a / 2.0))), max(1, int(round(b / 2.0))))
    cv2.ellipse(mask, center, axes, float(angle), 0, 360, 1, -1)
    return mask


def sample_regions_from_ellipse64(ellipse64: list[float]) -> list[list[int]]:
    cx, cy, a, b, angle = ellipse64
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    regions = []
    for i in range(PATCH_NUM):
        theta = 2.0 * np.pi * i / PATCH_NUM
        px = cx + (a / 2.0) * np.cos(theta) * cos_a - (b / 2.0) * np.sin(theta) * sin_a
        py = cy + (a / 2.0) * np.cos(theta) * sin_a + (b / 2.0) * np.sin(theta) * cos_a
        x1 = int(np.ceil(px)) - PATCH_SIZE // 2
        y1 = int(np.ceil(py)) - PATCH_SIZE // 2
        x1 = int(np.clip(x1, 0, EVENT_SIZE - PATCH_SIZE))
        y1 = int(np.clip(y1, 0, EVENT_SIZE - PATCH_SIZE))
        regions.append([x1, y1, x1 + PATCH_SIZE, y1 + PATCH_SIZE])
    return regions


def event_patches(volume: np.ndarray, regions: list[list[int]]) -> np.ndarray:
    return np.stack([volume[:, y1:y2, x1:x2] for x1, y1, x2, y2 in regions]).astype(np.uint16)


def ellipse128_from_row(row: dict) -> list[float]:
    return [
        float(row["gt_x_128"]),
        float(row["gt_y_128"]),
        float(row["gt_a_128"]),
        float(row["gt_b_128"]),
        float(row["gt_ang"]),
    ]


def ellipse64_from_row(row: dict) -> list[float]:
    return [
        float(row["gt_x_64"]),
        float(row["gt_y_64"]),
        float(row["gt_a_64"]),
        float(row["gt_b_64"]),
        float(row["gt_ang"]),
    ]


def same_track(a: dict, b: dict) -> bool:
    return (
        a["subject"] == b["subject"]
        and a["eye"] == b["eye"]
        and a["session_code"] == b["session_code"]
        and a["session_name"] == b["session_name"]
    )


def build_tracking_pairs(rows: list[dict]) -> list[tuple[dict, dict]]:
    pairs = []
    prev = None
    for row in rows:
        if prev is not None and same_track(prev, row):
            pairs.append((row, prev))
        prev = row
    return pairs


class HybridTrainDataset(Dataset):
    def __init__(self, dataset_root: Path, source_root: Path, split: str, task: str, limit: int | None = None):
        self.dataset_root = dataset_root
        self.source_root = source_root
        self.split = split
        self.task = task
        self.rows = read_rows(dataset_root / split / "labels.csv", limit=limit)
        self.pairs = build_tracking_pairs(self.rows) if task == "ex-gaze" else []
        self.data_root = source_root / split / "cached_data"

    def __len__(self) -> int:
        return len(self.pairs) if self.task == "ex-gaze" else len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.task == "ex-gaze":
            row, pre_row = self.pairs[index]
        else:
            row = self.rows[index]
            pre_row = None
        frame = cv2.imread(row["source_frame"], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Cannot read frame: {row['source_frame']}")
        frame128 = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        ellipse128 = ellipse128_from_row(row)
        ellipse64 = ellipse64_from_row(row)
        event64 = make_event64(load_event_segment(int(row["source_global_idx"]), self.data_root))
        frame_t = torch.from_numpy(frame128.astype(np.float32) / 255.0).unsqueeze(0)
        event_t = torch.from_numpy(event64.astype(np.float32))
        if self.task == "ev-eye":
            return {
                "frame": frame_t,
                "event": event_t,
                "mask": torch.from_numpy(make_mask128(ellipse128).astype(np.int64)),
            }
        assert pre_row is not None
        pre_ellipse64 = ellipse64_from_row(pre_row)
        regions = sample_regions_from_ellipse64(pre_ellipse64)
        return {
            "frame": frame_t,
            "event": event_t,
            "patches": torch.from_numpy(event_patches(event64, regions).astype(np.float32)),
            "pre_state": torch.tensor(pre_ellipse64, dtype=torch.float32),
            "target": torch.tensor(
                [
                    ellipse128[0] / FRAME_SIZE,
                    ellipse128[1] / FRAME_SIZE,
                    ellipse128[2] / FRAME_SIZE,
                    ellipse128[3] / FRAME_SIZE,
                    ellipse128[4] / 90.0,
                ],
                dtype=torch.float32,
            ),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EVEyeHybridSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_stem = ConvBlock(1, 32)
        self.event_stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.down1 = ConvBlock(64, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 2, 1)

    def forward(self, frame: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        f = self.frame_stem(frame)
        event_up = F.interpolate(event, size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False)
        e = self.event_stem(torch.log1p(event_up))
        x0 = torch.cat([f, e], dim=1)
        x1 = self.down1(F.max_pool2d(x0, 2))
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = self.down3(F.max_pool2d(x2, 2))
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        return self.head(x)


class EXGazeHybridRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.event_encoder = nn.Sequential(
            nn.Conv3d(8, 32, kernel_size=(2, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64 + 5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 5),
        )

    def forward(self, frame: torch.Tensor, patches: torch.Tensor, pre_state: torch.Tensor) -> torch.Tensor:
        f = self.frame_encoder(frame).flatten(1)
        # B x 8 x 2 x 16 x 16, treating the 8 sampled patches as Conv3D channels.
        e = self.event_encoder(torch.log1p(patches)).flatten(1)
        p = pre_state.clone()
        p[:, 0:4] = p[:, 0:4] / EVENT_SIZE
        p[:, 4] = p[:, 4] / 90.0
        delta = self.head(torch.cat([f, e, p], dim=1))
        return p + delta


def dice_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = mask.float()
    inter = (probs * target).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1e-6
    return (1.0 - (2.0 * inter + 1e-6) / denom).mean()


def pixel_error_from_regression(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_xy = pred[:, :2] * FRAME_SIZE
    target_xy = target[:, :2] * FRAME_SIZE
    return torch.linalg.norm(pred_xy - target_xy, dim=1)


def build_model(model_name: str) -> nn.Module:
    if model_name == "ev-eye":
        return EVEyeHybridSeg()
    if model_name == "ex-gaze":
        return EXGazeHybridRegressor()
    raise ValueError(model_name)


def train_one_epoch(model, loader, optimizer, device, model_name, max_steps: int | None = None) -> dict:
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    steps = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if model_name == "ev-eye":
            frame = batch["frame"].to(device, non_blocking=True)
            event = batch["event"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(frame, event)
            loss = F.cross_entropy(logits, mask) + dice_loss(logits, mask)
            metric = (logits.argmax(dim=1) == mask).float().mean()
        else:
            frame = batch["frame"].to(device, non_blocking=True)
            patches = batch["patches"].to(device, non_blocking=True)
            pre_state = batch["pre_state"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pred = model(frame, patches, pre_state)
            loss = F.smooth_l1_loss(pred, target)
            metric = pixel_error_from_regression(pred.detach(), target).mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_metric += float(metric.item())
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {"loss": total_loss / max(steps, 1), "metric": total_metric / max(steps, 1), "steps": steps}


@torch.no_grad()
def evaluate(model, loader, device, model_name, max_steps: int | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    steps = 0
    for batch in loader:
        if model_name == "ev-eye":
            frame = batch["frame"].to(device, non_blocking=True)
            event = batch["event"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(frame, event)
            loss = F.cross_entropy(logits, mask) + dice_loss(logits, mask)
            metric = (logits.argmax(dim=1) == mask).float().mean()
        else:
            frame = batch["frame"].to(device, non_blocking=True)
            patches = batch["patches"].to(device, non_blocking=True)
            pre_state = batch["pre_state"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pred = model(frame, patches, pre_state)
            loss = F.smooth_l1_loss(pred, target)
            metric = pixel_error_from_regression(pred, target).mean()
        total_loss += float(loss.item())
        total_metric += float(metric.item())
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {"loss": total_loss / max(steps, 1), "metric": total_metric / max(steps, 1), "steps": steps}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=["ev-eye", "ex-gaze"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    train_cfg = cfg["training"]
    epochs = args.epochs or int(train_cfg["epochs"])
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg["num_workers"])
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    source_root = Path(cfg["dataset"]["source_root"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = HybridTrainDataset(dataset_root, source_root, "train", args.model, args.limit_train_samples)
    val_ds = HybridTrainDataset(dataset_root, source_root, "val", args.model, args.limit_val_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    device = torch.device(args.device)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg["scheduler"]["step_size"]),
        gamma=float(train_cfg["scheduler"]["gamma"]),
    )
    metadata = {
        "config": str(args.config),
        "model": args.model,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "dry_run": args.dry_run,
        "torch": torch.__version__,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    if args.dry_run:
        train_stats = train_one_epoch(model, train_loader, optimizer, device, args.model, max_steps=1)
        val_stats = evaluate(model, val_loader, device, args.model, max_steps=1)
        payload = {"dry_run_train": train_stats, "dry_run_val": val_stats}
        (args.output_dir / "dry_run_result.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
        return

    best_val = math.inf if args.model == "ex-gaze" else -math.inf
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_metric", "val_loss", "val_metric", "lr", "seconds"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_stats = train_one_epoch(model, train_loader, optimizer, device, args.model, args.max_train_steps)
            val_stats = evaluate(model, val_loader, device, args.model, args.max_val_steps)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_metric": train_stats["metric"],
                "val_loss": val_stats["loss"],
                "val_metric": val_stats["metric"],
                "lr": lr,
                "seconds": time.time() - started,
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps(row), flush=True)
            improved = val_stats["metric"] < best_val if args.model == "ex-gaze" else val_stats["metric"] > best_val
            if improved:
                best_val = val_stats["metric"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metric": best_val,
                        "config": metadata,
                    },
                    args.output_dir / "best.ckpt",
                )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metric": val_stats["metric"],
                    "config": metadata,
                },
                args.output_dir / "last.ckpt",
            )


if __name__ == "__main__":
    main()
