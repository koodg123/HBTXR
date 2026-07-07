from __future__ import annotations

from typing import Any

import torch

_LR_SCHEDULERS = {"none", "cosine", "step", "plateau"}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def metric_mode(metric_name: str) -> str:
    lowered = str(metric_name).lower()
    if lowered.endswith("_pct") or lowered.endswith("_acc") or lowered.endswith("_score"):
        return "max"
    return "min"


def list_lr_scheduler_names() -> list[str]:
    return sorted(_LR_SCHEDULERS)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    total_epochs: int,
    *,
    optimizer_meta: dict[str, Any] | None = None,
):
    if optimizer_meta is not None and not bool(optimizer_meta.get("external_scheduler_allowed", True)):
        return None
    scheduler_cfg = training_cfg.get("scheduler") or {}
    scheduler_type = _normalize_name(scheduler_cfg.get("type", "none"))
    if scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_epochs)),
            eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
        )
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 10)),
            gamma=float(scheduler_cfg.get("gamma", 0.5)),
        )
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=metric_mode(str(scheduler_cfg.get("metric_name") or training_cfg.get("best_metric_name") or "")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
            threshold=float(scheduler_cfg.get("threshold", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


__all__ = [
    "build_lr_scheduler",
    "list_lr_scheduler_names",
    "metric_mode",
]
