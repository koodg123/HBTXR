from __future__ import annotations

from typing import Any

from torch import nn

from common.optim.pool import expand_optimizer_pool_candidates, write_optimizer_pool_report
from common.optim.registry import (
    build_optimizer,
    get_optimizer_metadata,
    is_optimizer_implemented,
    list_optimizer_names,
)


def list_implemented_optimizer_names() -> list[str]:
    return [name for name in list_optimizer_names() if is_optimizer_implemented(name)]


def optimizer_pool_summary() -> list[dict[str, Any]]:
    return [get_optimizer_metadata(name) for name in list_optimizer_names()]


def build_named_optimizer(model: nn.Module, name: str, cfg: dict[str, Any]):
    merged = dict(cfg)
    training_cfg = dict(merged.get("training") or {})
    optimizer_cfg = dict(training_cfg.get("optimizer") or {})
    optimizer_cfg["name"] = name
    training_cfg["optimizer"] = optimizer_cfg
    merged["training"] = training_cfg
    return build_optimizer(model, merged)


__all__ = [
    "build_optimizer",
    "build_named_optimizer",
    "expand_optimizer_pool_candidates",
    "get_optimizer_metadata",
    "is_optimizer_implemented",
    "list_implemented_optimizer_names",
    "list_optimizer_names",
    "optimizer_pool_summary",
    "write_optimizer_pool_report",
]
