# Source: E:/WSL/Shared/ETRI_SYNC/HBTXR/references/ultralytics-main/ultralytics/optim/muon.py
# Upstream: ultralytics/optim/muon.py
# Diff: see docs/hbtxr/optimizers/musgd_diff.md

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn, optim

from src.optim.common import named_trainable_parameters, split_params_auto_2d_plus


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in [
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
    ]:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, nesterov: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_SGD"] = torch.zeros_like(p)

                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon))

                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(state["momentum_buffer_SGD"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-lr)
        return loss


class Muon(optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss


def _musgd_param_groups(model: nn.Module, resolved_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    policy = str(kwargs.pop("musgd_group_policy", "auto_2d_plus")).strip().lower()
    base_group = {
        "lr": float(resolved_cfg["lr"]),
        "momentum": float(kwargs.get("momentum", 0.0)),
        "weight_decay": float(resolved_cfg["weight_decay"]),
        "nesterov": bool(kwargs.get("nesterov", False)),
    }
    if policy == "all_sgd":
        return [{**base_group, "params": [p for _, p in named_trainable_parameters(model)], "use_muon": False}]
    if policy == "all_muon":
        return [{**base_group, "params": [p for _, p in named_trainable_parameters(model)], "use_muon": True}]
    if policy == "explicit_from_param_groups":
        explicit = kwargs.get("param_groups")
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("musgd_group_policy=explicit_from_param_groups requires kwargs.param_groups")
        return explicit
    if policy != "auto_2d_plus":
        raise ValueError(f"Unsupported MuSGD group policy: {policy}")

    muon_params, sgd_params = split_params_auto_2d_plus(model)
    groups: list[dict[str, Any]] = []
    if muon_params:
        groups.append({**base_group, "params": muon_params, "use_muon": True})
    if sgd_params:
        groups.append({**base_group, "params": sgd_params, "use_muon": False})
    return groups


def build_musgd_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> MuSGD:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    param_groups = _musgd_param_groups(model, resolved_cfg)
    return MuSGD(
        param_groups,
        lr=float(resolved_cfg["lr"]),
        momentum=float(kwargs.get("momentum", 0.0)),
        weight_decay=float(resolved_cfg["weight_decay"]),
        nesterov=bool(kwargs.get("nesterov", False)),
        muon=float(kwargs.get("muon", 0.5)),
        sgd=float(kwargs.get("sgd", 0.5)),
    )
