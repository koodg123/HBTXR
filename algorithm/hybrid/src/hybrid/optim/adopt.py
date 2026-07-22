# Source: https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/optim/adopt.py
# Upstream: timm/optim/adopt.py adapted from iShohei220/adopt
# Diff: see docs/hbtxr/optimizers/adopt_diff.md

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class Adopt(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.9999),
        eps: float = 1.0e-6,
        clip_exp: float | None = 0.333,
        weight_decay: float = 0.0,
        decoupled: bool = False,
        corrected_weight_decay: bool = False,
        caution: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            clip_exp=clip_exp,
            weight_decay=weight_decay,
            decoupled=decoupled,
            corrected_weight_decay=corrected_weight_decay,
            caution=caution,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        max_lr = max(float(group["lr"]) for group in self.param_groups) if self.param_groups else 0.0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            clip_exp = group["clip_exp"]
            decoupled = bool(group["decoupled"])
            corrected_weight_decay = bool(group["corrected_weight_decay"])
            caution = bool(group["caution"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(grad, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = int(state["step"])

                if weight_decay != 0.0 and not decoupled:
                    grad = grad.add(p, alpha=weight_decay)

                if step == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                    continue

                if weight_decay != 0.0 and decoupled:
                    wd_scale = (lr * lr / max_lr) if corrected_weight_decay and max_lr > 0.0 else lr
                    p.add_(p, alpha=-wd_scale * weight_decay)

                denom = exp_avg_sq.sqrt().clamp_min_(eps)
                normed_grad = grad.div(denom)
                if clip_exp is not None:
                    clip_val = float((step - 1) ** float(clip_exp))
                    normed_grad.clamp_(-clip_val, clip_val)

                exp_avg.lerp_(normed_grad, 1 - beta1)
                update = exp_avg
                if caution:
                    mask = (update * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1.0e-3))
                    update = update * mask

                p.add_(update, alpha=-lr)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        return loss


def build_adopt_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> Adopt:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    return Adopt(
        model.parameters(),
        lr=float(resolved_cfg["lr"]),
        betas=tuple(resolved_cfg["betas"]),
        eps=float(resolved_cfg["eps"]),
        weight_decay=float(resolved_cfg["weight_decay"]),
        **kwargs,
    )
