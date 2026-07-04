# Source: https://raw.githubusercontent.com/AGI-Arena/MARS/main/MARS/optimizers/mars.py
# Upstream: AGI-Arena/MARS MARS/optimizers/mars.py
# Diff: see docs/src/optimizers/mars_diff.md

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


def exists(val):
    return val is not None


def update_fn(
    p,
    grad,
    exp_avg,
    exp_avg_sq,
    lr,
    wd,
    beta1,
    beta2,
    last_grad,
    eps,
    amsgrad,
    max_exp_avg_sq,
    step,
    gamma,
    mars_type,
    is_grad_2d,
    optimize_1d,
    lr_1d_factor,
    betas_1d,
    weight_decay_1d,
):
    if optimize_1d or is_grad_2d:
        c_t = (grad - last_grad).mul(gamma * (beta1 / (1.0 - beta1))).add(grad)
        c_t_norm = torch.norm(c_t)
        if c_t_norm > 1.0:
            c_t = c_t / c_t_norm
        exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)
        if (mars_type == "mars-adamw") or (mars_type == "mars-shampoo" and not is_grad_2d):
            exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            if amsgrad:
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)
            else:
                denom = exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)
            real_update_tmp = -lr * torch.mul(p.data, wd).add(exp_avg.div(denom))
        elif mars_type == "mars-lion":
            real_update_tmp = -lr * torch.mul(p.data, wd).add(exp_avg.sign())
        elif mars_type == "mars-shampoo" and is_grad_2d:
            factor = max(1, grad.size(0) / grad.size(1)) ** 0.5
            real_update_tmp = NewtonSchulz(exp_avg.mul(1.0 / (1.0 - beta1)), eps=eps).mul(factor).add(wd, p.data).mul(-lr)
        p.data.add_(real_update_tmp)
    else:
        beta1_1d, beta2_1d = betas_1d
        exp_avg.mul_(beta1_1d).add_(grad, alpha=1.0 - beta1_1d)
        exp_avg_sq.mul_(beta2_1d).addcmul_(grad, grad, value=1.0 - beta2_1d)
        bias_correction1 = 1.0 - beta1_1d**step
        bias_correction2 = 1.0 - beta2_1d**step
        if amsgrad:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)
        else:
            denom = exp_avg_sq.sqrt().mul(1 / math.sqrt(bias_correction2)).add(eps).mul(bias_correction1)
        real_update_tmp = -lr * lr_1d_factor * torch.mul(p.data, weight_decay_1d).add(exp_avg.div(denom))
        p.data.add_(real_update_tmp)
    return exp_avg, exp_avg_sq


class MARS(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3.0e-3,
        betas: tuple[float, float] = (0.95, 0.99),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        gamma: float = 0.025,
        is_approx: bool = True,
        mars_type: str = "mars-adamw",
        optimize_1d: bool = False,
        lr_1d: float = 3.0e-3,
        betas_1d: tuple[float, float] = (0.9, 0.95),
        weight_decay_1d: float = 0.1,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        assert mars_type in ["mars-adamw", "mars-lion", "mars-shampoo"], "MARS type not supported"
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            mars_type=mars_type,
            gamma=gamma,
            optimize_1d=optimize_1d,
            weight_decay_1d=weight_decay_1d,
        )
        super().__init__(params, defaults)
        self.eps = eps
        self.update_fn = update_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.step_num = 0
        self.is_approx = is_approx
        self.hbtxr_exact_multi_pass = not is_approx
        self.gamma = gamma
        self.mars_type = mars_type
        self.optimize_1d = optimize_1d
        self.lr_1d_factor = lr_1d / lr
        self.weight_decay_1d = weight_decay_1d
        self.betas_1d = betas_1d

    @torch.no_grad()
    def update_last_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    if "last_grad" not in state:
                        state["last_grad"] = torch.zeros_like(p)
                    previous_grad = state.get("previous_grad")
                    if previous_grad is None:
                        state["last_grad"].zero_()
                    else:
                        state["last_grad"].zero_().add_(previous_grad, alpha=1.0)

    @torch.no_grad()
    def update_previous_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    if "previous_grad" not in state:
                        state["previous_grad"] = torch.zeros_like(p)
                    state["previous_grad"].zero_()
                    if p.grad is not None:
                        state["previous_grad"].add_(p.grad, alpha=1.0)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                "FusedAdam has been updated. Simply initialize it identically to torch.optim.Adam, and call step() with no arguments."
            )

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        gamma = self.gamma
        step = self.step_num
        for group in self.param_groups:
            for p in filter(lambda param: exists(param.grad), group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                amsgrad = group["amsgrad"]

                state = self.state[p]
                if len(state) <= 1:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["last_grad"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                last_grad = state["last_grad"]
                lr, wd, beta1, beta2 = group["lr"], group["weight_decay"], *group["betas"]
                max_exp_avg_sq = state["max_exp_avg_sq"] if amsgrad else 0

                if "step" in state:
                    state["step"] += 1
                else:
                    state["step"] = 1
                step = state["step"]
                is_grad_2d = len(grad.shape) == 2
                exp_avg, exp_avg_sq = self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    last_grad,
                    self.eps,
                    amsgrad,
                    max_exp_avg_sq,
                    step,
                    gamma,
                    mars_type=self.mars_type,
                    is_grad_2d=is_grad_2d,
                    optimize_1d=self.optimize_1d,
                    lr_1d_factor=self.lr_1d_factor,
                    betas_1d=self.betas_1d,
                    weight_decay_1d=self.weight_decay if self.optimize_1d else self.weight_decay_1d,
                )
                if self.is_approx:
                    state["last_grad"] = grad
        self.step_num = step
        return loss


def NewtonSchulz(M, steps: int = 5, eps: float = 1.0e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M.bfloat16() / (M.norm() + eps)
    if M.size(0) > M.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if M.size(0) > M.size(1):
        X = X.T
    return X.to(M.dtype)


def build_mars_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> MARS:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    lr_1d = float(kwargs.pop("lr_1d", resolved_cfg["lr"]))
    betas_1d = tuple(kwargs.pop("betas_1d", (0.9, 0.95)))
    gamma = float(kwargs.pop("gamma", 0.025))
    amsgrad = bool(kwargs.pop("amsgrad", False))
    is_approx = bool(kwargs.pop("is_approx", True))
    mars_type = str(kwargs.pop("mars_type", "mars-adamw"))
    optimize_1d = bool(kwargs.pop("optimize_1d", False))
    weight_decay_1d = float(kwargs.pop("weight_decay_1d", 0.1))
    if kwargs:
        raise ValueError(f"Unsupported MARS kwargs: {sorted(kwargs.keys())}")
    return MARS(
        model.parameters(),
        lr=float(resolved_cfg["lr"]),
        betas=tuple(resolved_cfg["betas"]),
        eps=float(resolved_cfg["eps"]),
        weight_decay=float(resolved_cfg["weight_decay"]),
        amsgrad=amsgrad,
        gamma=gamma,
        is_approx=is_approx,
        mars_type=mars_type,
        optimize_1d=optimize_1d,
        lr_1d=lr_1d,
        betas_1d=(float(betas_1d[0]), float(betas_1d[1])),
        weight_decay_1d=weight_decay_1d,
    )
