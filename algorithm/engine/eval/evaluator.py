"""Evaluation loop for the reimplemented HBTXR models (P5.7).

Mirrors the ``engine.train.Trainer`` modality forward paths under ``no_grad`` and
aggregates per-term losses plus contract-native metrics (center distance, mask IoU)
over a dataloader. No Lightning: a plain averaging loop, symmetric with the custom
trainer.

- direct (frame / event) and mask: one ``model(image)`` forward.
- hybrid: search_step(frame) + track_step(event, GT anchor), losses summed like
  training; the tracked state (or search state) is used for the center metric.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import nn

from engine.eval.metrics import center_distance, mask_iou
from engine.train.losses import compute_losses


def _batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    return 1


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _direct_step(model: nn.Module, batch: dict[str, Any], weights):
    outputs = model(batch["image"])
    if torch.is_tensor(outputs):
        outputs = {"mask": outputs}
    return outputs, compute_losses(outputs, batch, weights), outputs.get("state")


def _hybrid_step(model: nn.Module, batch: dict[str, Any], weights, device: str):
    merged: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=device)
    state = None
    if "frame" in batch:
        out_s = model.search_step(batch["frame"])
        ls = compute_losses(out_s, batch, weights)
        total = total + ls["total"]
        merged.update({f"search_{k}": v for k, v in ls.items() if k != "total"})
        state = out_s.get("state")
    if "event" in batch:
        anchor = batch.get("anchor_state")
        if anchor is None and "state" in batch:
            anchor = batch["state"]
        out_t = model.track_step(batch["event"], anchor)
        lt = compute_losses(out_t, batch, weights)
        total = total + lt["total"]
        merged.update({f"track_{k}": v for k, v in lt.items() if k != "total"})
        state = out_t.get("state", state)
    merged["total"] = total
    return merged, merged, state


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    *,
    modality: str,
    device: str = "cpu",
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Average per-term losses and metrics over ``dataloader``."""
    modality = str(modality).lower()
    is_hybrid = modality.startswith("hybrid")
    model = model.to(device)
    model.eval()

    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch in dataloader:
        batch = _to_device(batch, device)
        if is_hybrid:
            outputs, losses, pred_state = _hybrid_step(model, batch, loss_weights, device)
        else:
            outputs, losses, pred_state = _direct_step(model, batch, loss_weights)

        bs = _batch_size(batch)
        for name, value in losses.items():
            if torch.is_tensor(value):
                totals[f"loss/{name}"] += float(value) * bs
        if pred_state is not None and "state" in batch:
            totals["center_dist"] += center_distance(pred_state, batch["state"]) * bs
        if "mask" in outputs and "mask" in batch:
            totals["mask_iou"] += mask_iou(outputs["mask"], batch["mask"]) * bs
        count += bs

    denom = max(1, count)
    return {name: value / denom for name, value in sorted(totals.items())}


__all__ = ["evaluate"]
