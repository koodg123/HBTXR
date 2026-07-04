from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import HGTXRDataset, make_synthetic_batch
from .io import ensure_dir, write_json
from .losses import compute_stage1_losses, compute_stage2_losses
from .metrics import compute_metrics
from .model import build_model


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in batch[0]:
        vals = [row[key] for row in batch]
        out[key] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
    return out


def make_loader(manifest_path: str, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    ds = HGTXRDataset(manifest_path, input_size=tuple(cfg["data"]["input_size"]))
    return DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=shuffle, num_workers=int(cfg["training"].get("num_workers", 0)), collate_fn=collate_fn)


def train(cfg: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("training", {}).get("device", "cpu"))
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    stage = str(cfg["training"].get("stage", "stage1"))
    epochs = 1 if smoke else int(cfg["training"].get("epochs", 1))
    history = []
    for epoch in range(epochs):
        model.train()
        batch = make_synthetic_batch(int(cfg["training"].get("batch_size", 2)), input_size=tuple(cfg["data"]["input_size"]))
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model.forward_train(batch)
        losses = compute_stage1_losses(batch, outputs, cfg["loss"]) if stage == "stage1" else compute_stage2_losses(batch, outputs, cfg["loss"])
        opt.zero_grad(set_to_none=True)
        losses["loss/total"].backward()
        opt.step()
        with torch.no_grad():
            metrics = compute_metrics(batch, outputs)
        row = {k: float(v.detach().cpu().item()) for k, v in {**losses, **metrics}.items()}
        row["epoch"] = epoch
        history.append(row)
    out_dir = ensure_dir(cfg["experiment"].get("output_dir") or "runs/default")
    torch.save({"model": model.state_dict(), "cfg": cfg, "history": history}, Path(out_dir) / "last.pt")
    write_json({"history": history}, Path(out_dir) / "summary.json")
    return {"output_dir": str(out_dir), "history": history}

