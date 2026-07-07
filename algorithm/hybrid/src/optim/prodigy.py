# Source: https://raw.githubusercontent.com/konstmish/prodigy/main/prodigyopt/prodigy.py
# Upstream: konstmish/prodigy prodigyopt/prodigy.py
# Diff: see docs/hbtxr/optimizers/prodigy_diff.md

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn


class Prodigy(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float | None = None,
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        decouple: bool = True,
        use_bias_correction: bool = False,
        safeguard_warmup: bool = False,
        d0: float = 1.0e-6,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
        fsdp_in_use: bool = False,
        slice_p: int = 1,
    ) -> None:
        if d0 <= 0.0:
            raise ValueError(f"Invalid d0 value: {d0}")
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if int(slice_p) <= 0:
            raise ValueError(f"Invalid slice_p value: {slice_p}")
        defaults = dict(
            lr=lr,
            betas=betas,
            beta3=beta3,
            eps=eps,
            weight_decay=weight_decay,
            d=d0,
            d0=d0,
            d_max=d0,
            d_numerator=0.0,
            d_coef=d_coef,
            k=0,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            decouple=decouple,
            safeguard_warmup=safeguard_warmup,
            fsdp_in_use=fsdp_in_use,
            slice_p=int(slice_p),
        )
        self.d0 = d0
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        d_denom = 0.0
        group0 = self.param_groups[0]
        beta1, beta2 = group0["betas"]
        beta3 = group0["beta3"] if group0["beta3"] is not None else math.sqrt(beta2)
        k = group0["k"]
        d = group0["d"]
        d_max = group0["d_max"]
        d_coef = group0["d_coef"]
        base_lr = max(float(group["lr"]) for group in self.param_groups)
        if group0["use_bias_correction"]:
            bias_correction = math.sqrt(1 - beta2 ** (k + 1)) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1.0
        dlr = d * base_lr * bias_correction
        growth_rate = float(group0["growth_rate"])
        decouple = bool(group0["decouple"])
        fsdp_in_use = bool(group0["fsdp_in_use"])
        d_numerator = float(group0["d_numerator"]) * beta3
        delta_numerator = 0.0

        for group in self.param_groups:
            decay = float(group["weight_decay"])
            group_lr = float(group["lr"])
            d0 = float(group["d0"])
            eps = float(group["eps"])
            safeguard_warmup = bool(group["safeguard_warmup"])
            slice_p = int(group["slice_p"])
            if group_lr not in {base_lr, 0.0}:
                raise RuntimeError("Prodigy supports per-group lr only for values of 0 or the shared base lr")
            for p in group["params"]:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True
                grad = p.grad.data
                if decay != 0.0 and not decouple:
                    grad = grad.add(p.data, alpha=decay)

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()
                    state["p0"] = (
                        p.detach().flatten()[::slice_p].clone()
                        if bool(p.numel())
                        else torch.tensor(0, device=p.device, dtype=p.dtype)
                    )
                    if beta1 > 0:
                        state["exp_avg"] = torch.zeros_like(p.data).detach()
                    state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

                exp_avg_sq = state["exp_avg_sq"]
                s = state["s"]
                p0 = state["p0"]
                if group_lr > 0.0:
                    sliced_grad = grad.flatten()[::slice_p]
                    delta_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0.data - p.data.flatten()[::slice_p]).item()
                if beta1 > 0:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad, alpha=d * (1 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))
                if safeguard_warmup:
                    s.mul_(beta3).add_(grad.flatten()[::slice_p], alpha=(d / d0) * d)
                else:
                    s.mul_(beta3).add_(grad.flatten()[::slice_p], alpha=(d / d0) * dlr)
                d_denom += s.abs().sum().item()

        if d_denom == 0.0 and not fsdp_in_use:
            return loss

        if base_lr > 0.0:
            if fsdp_in_use and dist.is_available() and dist.is_initialized():
                dist_tensor = torch.tensor([delta_numerator, d_denom], device=self.param_groups[0]["params"][0].device)
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = d_numerator + float(dist_tensor[0].item())
                global_d_denom = float(dist_tensor[1].item())
            else:
                global_d_numerator = d_numerator + delta_numerator
                global_d_denom = d_denom
            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group0["d0"]:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)
        else:
            global_d_numerator = d_numerator
            global_d_denom = d_denom
            d_hat = d

        for group in self.param_groups:
            group["d_numerator"] = global_d_numerator
            group["d_denom"] = global_d_denom
            group["d"] = d
            group["d_max"] = d_max
            group["d_hat"] = d_hat
            decay = float(group["weight_decay"])
            eps = float(group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                denom = exp_avg_sq.sqrt().add_(d * eps)
                if decay != 0.0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)
                if beta1 > 0:
                    exp_avg = state["exp_avg"]
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr * d)
            group["k"] = k + 1
        return loss


def build_prodigy_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> Prodigy:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    return Prodigy(
        model.parameters(),
        lr=float(resolved_cfg["lr"]),
        betas=tuple(resolved_cfg["betas"]),
        eps=float(resolved_cfg["eps"]),
        weight_decay=float(resolved_cfg["weight_decay"]),
        **kwargs,
    )
