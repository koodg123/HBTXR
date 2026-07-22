# Source: https://raw.githubusercontent.com/google/automl/master/lion/lion_pytorch.py
# Upstream: google/automl lion_pytorch.py
# Diff: see docs/hbtxr/optimizers/lion_diff.md

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign_(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def build_lion_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> Lion:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    return Lion(
        model.parameters(),
        lr=float(resolved_cfg["lr"]),
        betas=tuple(resolved_cfg["betas"]),
        weight_decay=float(resolved_cfg["weight_decay"]),
        **kwargs,
    )
