# Source: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
# Upstream: torch.optim.AdamW (PyTorch built-in)
# Diff: see docs/src/optimizers/adamw_diff.md

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_adamw_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(resolved_cfg["lr"]),
        betas=tuple(resolved_cfg["betas"]),
        eps=float(resolved_cfg["eps"]),
        weight_decay=float(resolved_cfg["weight_decay"]),
        **kwargs,
    )
