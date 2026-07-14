from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
EV_EYE_CODE = ROOT / "references/codebase/software/EV-Eye"
ANNOTATION_EV_EYE_CODE = ROOT.parent / "HBTXR-Annotation/HBTXR/references/codebase/software/EV-Eye"
if EV_EYE_CODE.exists():
    sys.path.insert(0, str(EV_EYE_CODE))
elif ANNOTATION_EV_EYE_CODE.exists():
    sys.path.insert(0, str(ANNOTATION_EV_EYE_CODE))
else:
    raise RuntimeError("Cannot find EV-Eye codebase for original UNet import")

from unet import UNet  # noqa: E402
from utils.dice_score import dice_loss  # noqa: E402


SENSOR_H = 260
SENSOR_W = 346
FRAME_SIZE = 128
EVENT_SHAPE = (2, 64, 64)
ORDERS = ("1_0_2", "2_0_1", "2_0_2")


@dataclass(frozen=True)
class SampleBlock:
    user: int
    eye: str
    session: str
    images: np.ndarray
    masks: np.ndarray


def parse_user(path: Path) -> int:
    match = re.search(r"user(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot parse user id from {path}")
    return int(match.group(1))


def resize_frames(frames: np.ndarray) -> np.ndarray:
    out = np.empty((frames.shape[0], FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        out[idx] = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    return out


def resize_masks(masks: np.ndarray) -> np.ndarray:
    out = np.empty((masks.shape[0], FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    for idx, mask in enumerate(masks):
        out[idx] = cv2.resize((mask > 0).astype(np.uint8), (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_NEAREST)
    return out


def load_eye_blocks(labelled_root: Path, eye: str, limit_users: set[int] | None = None) -> list[SampleBlock]:
    eye_dir = labelled_root / eye
    if not eye_dir.is_dir():
        raise FileNotFoundError(f"Missing EV-Eye labelled eye directory: {eye_dir}")

    blocks: list[SampleBlock] = []
    for path in sorted(eye_dir.glob("user*_session_*.h5")):
        user = parse_user(path)
        if limit_users is not None and user not in limit_users:
            continue
        session = path.stem.split("_session_", 1)[1]
        if session not in ORDERS:
            continue
        with h5py.File(path, "r") as h5:
            # Original EV-Eye stores arrays as (346, 260, N) and uses .T -> (N, 260, 346).
            images = np.transpose(h5["data"][:], (2, 1, 0))
            masks = np.transpose(h5["label"][:], (2, 1, 0))
        blocks.append(
            SampleBlock(
                user=user,
                eye=eye,
                session=session,
                images=resize_frames(images),
                masks=resize_masks(masks),
            )
        )
    if not blocks:
        raise RuntimeError(f"No EV-Eye labelled h5 blocks loaded from {eye_dir}")
    return blocks


def tensor_dataset_from_blocks(blocks: list[SampleBlock], heldout_user: int, train: bool) -> TensorDataset:
    selected = [block for block in blocks if (block.user != heldout_user if train else block.user == heldout_user)]
    if not selected:
        raise RuntimeError(f"No {'train' if train else 'test'} blocks for heldout user {heldout_user}")
    images = np.concatenate([block.images for block in selected], axis=0)
    masks = np.concatenate([block.masks for block in selected], axis=0)
    image_tensor = torch.from_numpy(images).float().div(255.0).unsqueeze(1)
    mask_tensor = torch.from_numpy(masks).long()
    return TensorDataset(image_tensor, mask_tensor)


def multiclass_iou(logits: torch.Tensor, masks: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    pred_fg = pred == 1
    target_fg = masks == 1
    inter = (pred_fg & target_fg).sum().item()
    union = (pred_fg | target_fg).sum().item()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    net.eval()
    dice_total = 0.0
    iou_total = 0.0
    loss_total = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss()
    for images, masks in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device=device, dtype=torch.long, non_blocking=True)
        logits = net(images)
        one_hot = F.one_hot(masks, net.n_classes).permute(0, 3, 1, 2).float()
        loss = criterion(logits, masks) + dice_loss(F.softmax(logits, dim=1).float(), one_hot, multiclass=True)
        pred_oh = F.one_hot(logits.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        dice = 1.0 - float(dice_loss(pred_oh[:, 1:, ...], one_hot[:, 1:, ...], multiclass=False).item())
        dice_total += dice
        iou_total += multiclass_iou(logits, masks)
        loss_total += float(loss.item())
        steps += 1
    net.train()
    denom = max(steps, 1)
    return {"loss": loss_total / denom, "dice": dice_total / denom, "iou": iou_total / denom, "steps": steps}


def shutdown_loader(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    loader._iterator = None


def train_subject(
    blocks: list[SampleBlock],
    eye: str,
    heldout_user: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float | int | str]:
    train_ds = val_ds = None
    train_loader = val_loader = None
    net = optimizer = scheduler = None
    device = torch.device(args.device)
    started = time.time()
    best_dice = -1.0
    train_samples = 0
    val_samples = 0
    try:
        train_ds = tensor_dataset_from_blocks(blocks, heldout_user, train=True)
        val_ds = tensor_dataset_from_blocks(blocks, heldout_user, train=False)
        train_samples = len(train_ds)
        val_samples = len(val_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=False,
        )

        net = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear).to(device)
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.1)
        criterion = nn.CrossEntropyLoss()

        user_dir = output_dir / eye / f"user{heldout_user}"
        user_dir.mkdir(parents=True, exist_ok=True)
        history_path = user_dir / "history.csv"
        with history_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "seconds"])
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                epoch_started = time.time()
                net.train()
                train_loss = 0.0
                train_steps = 0
                for images, masks in train_loader:
                    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                    masks = masks.to(device=device, dtype=torch.long, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = net(images)
                    loss = criterion(logits, masks) + dice_loss(
                        F.softmax(logits, dim=1).float(),
                        F.one_hot(masks, net.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True,
                    )
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.item())
                    train_steps += 1
                    if args.max_train_steps is not None and train_steps >= args.max_train_steps:
                        break
                val_stats = evaluate(net, val_loader, device)
                scheduler.step(val_stats["dice"])
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss / max(train_steps, 1),
                    "val_loss": val_stats["loss"],
                    "val_dice": val_stats["dice"],
                    "val_iou": val_stats["iou"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds": time.time() - epoch_started,
                }
                writer.writerow(row)
                f.flush()
                print(json.dumps({"eye": eye, "heldout_user": heldout_user, **row}), flush=True)
                if val_stats["dice"] > best_dice:
                    best_dice = val_stats["dice"]
                    torch.save(
                        {
                            "eye": eye,
                            "heldout_user": heldout_user,
                            "epoch": epoch,
                            "model_state_dict": net.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_stats": val_stats,
                        },
                        user_dir / "best.pth",
                    )
                torch.save(
                    {
                        "eye": eye,
                        "heldout_user": heldout_user,
                        "epoch": epoch,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_stats": val_stats,
                    },
                    user_dir / "last.pth",
                )
        return {
            "eye": eye,
            "heldout_user": heldout_user,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "best_dice": best_dice,
            "seconds": time.time() - started,
        }
    finally:
        shutdown_loader(train_loader)
        shutdown_loader(val_loader)
        del train_loader, val_loader, train_ds, val_ds, net, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train original EV-Eye UNet on manual masks with 128x128 input.")
    parser.add_argument("--labelled-root", type=Path, default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis_labelled_with_mask"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/RESULTS/EV-Eye_labelled_mask_frame128_event64_original_protocol"))
    parser.add_argument("--eyes", nargs="+", choices=["left", "right"], default=["left", "right"])
    parser.add_argument("--users", nargs="*", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bilinear", action="store_true", default=False)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_users = set(args.users) if args.users else set(range(1, 49))
    metadata = {
        "task": "EV-Eye original UNet manual-mask training with resolution-only resize",
        "labelled_root": str(args.labelled_root),
        "output_dir": str(args.output_dir),
        "frame_shape": [1, FRAME_SIZE, FRAME_SIZE],
        "event_shape": list(EVENT_SHAPE),
        "event_usage": "not used by original EV-Eye UNet training; event remains evaluation/tracking-side input",
        "original_frame_shape": [1, SENSOR_H, SENSOR_W],
        "split": "leave-one-subject-out per eye, matching original EV-Eye code",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "scheduler": "ReduceLROnPlateau(mode=max, patience=2, factor=0.1)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "eyes": args.eyes,
        "users": sorted(selected_users),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "dry_run": args.dry_run,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    summary_rows = []
    for eye in args.eyes:
        blocks = load_eye_blocks(args.labelled_root, eye, selected_users)
        users = sorted({block.user for block in blocks})
        if args.dry_run:
            users = users[:1]
        for heldout_user in users:
            row = train_subject(blocks, eye, heldout_user, args, args.output_dir)
            summary_rows.append(row)
            if args.dry_run:
                break

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = ["eye", "heldout_user", "train_samples", "val_samples", "best_dice", "seconds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"summary": str(summary_path), "runs": len(summary_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
