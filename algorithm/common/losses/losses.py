from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from common.losses.metrics import compute_metrics
from common.losses.stage1 import compute_stage1_losses
from common.losses.stage2 import compute_stage2_losses

LossFn = Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]], dict[str, torch.Tensor]]


def _stage1_loss(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
    *,
    active_head: str = "all",
) -> dict[str, torch.Tensor]:
    return compute_stage1_losses(batch, outputs, loss_cfg, active_head=active_head)


def _stage2_loss(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
    *,
    active_head: str = "all",
) -> dict[str, torch.Tensor]:
    return compute_stage2_losses(batch, outputs, loss_cfg, active_head=active_head)


_LOSSES: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
    "stage1": _stage1_loss,
    "stage1_search": _stage1_loss,
    "search": _stage1_loss,
    "stage2": _stage2_loss,
    "stage2_hybrid": _stage2_loss,
    "hybrid": _stage2_loss,
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def list_loss_names() -> list[str]:
    return sorted(_LOSSES.keys())


def build_loss(name: str) -> Callable[..., dict[str, torch.Tensor]]:
    normalized = _normalize_name(name)
    try:
        return _LOSSES[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown loss: {normalized}") from exc


def compute_loss(
    name: str,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
    *,
    active_head: str = "all",
) -> dict[str, torch.Tensor]:
    return build_loss(name)(batch, outputs, loss_cfg, active_head=active_head)


__all__ = [
    "build_loss",
    "compute_loss",
    "compute_metrics",
    "list_loss_names",
]
