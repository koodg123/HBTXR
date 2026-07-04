from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    w = weights.to(values.device).float()
    while w.ndim < values.ndim:
        w = w.unsqueeze(-1)
    return (values * w).sum() / torch.clamp(w.sum(), min=1.0)


def state_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return weighted_mean(F.smooth_l1_loss(pred, target, reduction="none"), weights)


def mask_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def compute_stage1_losses(batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor], loss_cfg: dict) -> dict[str, torch.Tensor]:
    q = batch.get("quality")
    target = batch["target_state"]
    losses = {
        "loss/search_state": state_loss(outputs["search/state"], target, q) * float(loss_cfg.get("search_xy_weight", 1.0)),
        "loss/mask": mask_loss(outputs["mask/logits"], batch["mask"]) * float(loss_cfg.get("mask_weight", 1.0)),
    }
    losses["loss/total"] = sum(losses.values())
    return losses


def compute_stage2_losses(batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor], loss_cfg: dict) -> dict[str, torch.Tensor]:
    losses = compute_stage1_losses(batch, outputs, loss_cfg)
    q = batch.get("quality")
    target_residual = batch["target_state"] - batch["prev_state"]
    losses["loss/event_residual"] = state_loss(outputs["event/residual"], target_residual, q) * float(loss_cfg.get("event_xy_weight", 1.0))
    losses["loss/track_residual"] = state_loss(outputs["track/residual"], target_residual, q) * float(loss_cfg.get("track_xy_weight", 1.0))
    losses["loss/track_state"] = state_loss(outputs["track/state"], batch["target_state"], q) * float(loss_cfg.get("track_geo_weight", 0.5))
    losses["loss/consistency"] = state_loss(outputs["track/state"], outputs["search/state"].detach(), q) * float(loss_cfg.get("consistency_weight", 0.2))
    losses["loss/total"] = sum(v for k, v in losses.items() if k != "loss/total")
    return losses

