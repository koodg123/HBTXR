# Source: https://raw.githubusercontent.com/apple/ml-ademamix/main/pytorch/ademamix.py
# Upstream: apple/ml-ademamix pytorch/ademamix.py
# Diff: see docs/hbtxr/optimizers/adema_mix_diff.md

from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import Optimizer


def linear_warmup_scheduler(step: int, alpha_end: float, alpha_start: float = 0.0, warmup: int = 1) -> float:
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1) -> float:
    def f(beta: float, eps: float = 1.0e-8) -> float:
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t: float) -> float:
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


class AdEMAMix(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 2.0,
        beta3_warmup: int | None = None,
        alpha_warmup: int | None = None,
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            lmbda = float(group["weight_decay"])
            eps = float(group["eps"])
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = float(group["alpha"])
            alpha_warmup = group["alpha_warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_fast"] = torch.zeros_like(p, memory_format=torch.preserve_format) if beta1 != 0.0 else None
                    state["exp_avg_slow"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg_fast = state["exp_avg_fast"]
                exp_avg_slow = state["exp_avg_slow"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                bias_correction1 = 1 - beta1**int(state["step"])
                bias_correction2 = 1 - beta2**int(state["step"])
                alpha = (
                    linear_warmup_scheduler(int(state["step"]), alpha_end=alpha_final, alpha_start=0.0, warmup=int(alpha_warmup))
                    if alpha_warmup is not None
                    else alpha_final
                )
                beta3 = (
                    linear_hl_warmup_scheduler(int(state["step"]), beta_end=beta3_final, beta_start=beta1, warmup=int(beta3_warmup))
                    if beta3_warmup is not None
                    else beta3_final
                )

                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = (exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow) / denom
                update.add_(p, alpha=lmbda)
                p.add_(update, alpha=-lr)

        return loss


def build_adema_mix_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> AdEMAMix:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    beta3 = float(kwargs.pop("beta3", 0.9999))
    alpha = float(kwargs.pop("alpha", 2.0))
    beta3_warmup = kwargs.pop("beta3_warmup", None)
    alpha_warmup = kwargs.pop("alpha_warmup", None)
    if kwargs:
        raise ValueError(f"Unsupported AdEMAMix kwargs: {sorted(kwargs.keys())}")
    return AdEMAMix(
        model.parameters(),
        lr=float(resolved_cfg["lr"]),
        betas=(float(resolved_cfg["betas"][0]), float(resolved_cfg["betas"][1]), beta3),
        alpha=alpha,
        beta3_warmup=int(beta3_warmup) if beta3_warmup is not None else None,
        alpha_warmup=int(alpha_warmup) if alpha_warmup is not None else None,
        eps=float(resolved_cfg["eps"]),
        weight_decay=float(resolved_cfg["weight_decay"]),
    )
