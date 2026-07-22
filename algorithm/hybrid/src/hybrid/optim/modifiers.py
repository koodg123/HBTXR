# Source: https://raw.githubusercontent.com/facebookresearch/schedule_free/main/schedulefree/adamw_schedulefree.py
# Upstream: facebookresearch/schedule_free
# Diff: schedule_free is integrated as an optimizer modifier within HBTXR.

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch.optim import Optimizer


class AdamWScheduleFree(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: bool | None = None,
    ) -> None:
        if foreach is None:
            foreach = hasattr(torch, "_foreach_mul_")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            k=0,
            warmup_steps=warmup_steps,
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / beta1)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    p.lerp_(end=state["z"].to(p.device), weight=1 - beta1)
            group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError("Schedule-free optimizer must be put in train mode before step().")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = float(group["eps"])
            beta1, beta2 = group["betas"]
            decay = float(group["weight_decay"])
            k = int(group["k"])
            r = float(group["r"])
            warmup_steps = int(group["warmup_steps"])
            weight_lr_power = float(group["weight_lr_power"])

            sched = (k + 1) / warmup_steps if k < warmup_steps and warmup_steps > 0 else 1.0
            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = float(group["lr"]) * sched
            group["scheduled_lr"] = lr
            lr_max = group["lr_max"] = max(lr, float(group["lr_max"]))
            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = float(group["weight_sum"]) + weight
            group["weight_sum"] = weight_sum
            ckp1 = weight / weight_sum if weight_sum != 0.0 else 0.0

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                state = self.state[p]
                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            for p in active:
                y = p
                grad = p.grad
                state = self.state[p]
                z = state["z"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                grad_normalized = grad.div(denom)
                if decay != 0.0:
                    grad_normalized.add_(y, alpha=decay)
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))
                z.sub_(grad_normalized, alpha=lr)

            group["k"] = k + 1
        return loss


class CautiousOptimizer(Optimizer):
    def __init__(self, base_optimizer: Optimizer, *, eps: float = 1.0e-3) -> None:
        self.base_optimizer = base_optimizer
        self.defaults = getattr(base_optimizer, "defaults", {})
        self.state = base_optimizer.state
        self.param_groups = base_optimizer.param_groups
        self._cautious_eps = float(eps)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        before: dict[int, torch.Tensor] = {}
        grads: dict[int, torch.Tensor] = {}
        params: list[torch.nn.Parameter] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                before[id(p)] = p.detach().clone()
                grads[id(p)] = p.grad.detach().clone()
                params.append(p)

        loss = self.base_optimizer.step(closure)

        for p in params:
            delta = before[id(p)] - p.data
            grad = grads[id(p)]
            mask = (delta * grad > 0).to(delta.dtype)
            scale = mask.mean().clamp_(min=self._cautious_eps)
            p.data.copy_(before[id(p)] - (delta * mask / scale))
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "cautious_eps": self._cautious_eps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cautious_eps = float(state_dict.get("cautious_eps", self._cautious_eps))
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.state = self.base_optimizer.state
        self.param_groups = self.base_optimizer.param_groups

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self.base_optimizer.add_param_group(param_group)
        self.param_groups = self.base_optimizer.param_groups

    def train(self) -> None:
        fn = getattr(self.base_optimizer, "train", None)
        if callable(fn):
            fn()

    def eval(self) -> None:
        fn = getattr(self.base_optimizer, "eval", None)
        if callable(fn):
            fn()


def build_schedule_free_adamw(
    params: Iterable[torch.nn.Parameter],
    resolved_cfg: dict[str, Any],
) -> AdamWScheduleFree:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    return AdamWScheduleFree(
        params,
        lr=float(resolved_cfg["lr"]),
        betas=tuple(resolved_cfg["betas"]),
        eps=float(resolved_cfg["eps"]),
        weight_decay=float(resolved_cfg["weight_decay"]),
        **kwargs,
    )
