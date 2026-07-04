# Source: https://raw.githubusercontent.com/Liuhong99/Sophia/main/sophia.py
# Upstream: Liuhong99/Sophia sophia.py
# Diff: see docs/src/optimizers/sophia_g_diff.md

from __future__ import annotations

from typing import Any, Callable, List

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer


class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 1.0e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
        state_values = list(self.state.values())
        step_is_tensor = bool(state_values) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for item in state_values:
                item["step"] = torch.tensor(float(item["step"]))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None, bs: int = 5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("Hero does not support sparse gradients")
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])
                hessian.append(state["hessian"])
                if self.defaults["capturable"]:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(
                params_with_grad,
                grads,
                exp_avgs,
                hessian,
                state_steps,
                bs=bs,
                beta1=beta1,
                beta2=beta2,
                rho=group["rho"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
            )

        return loss


def sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    capturable: bool = False,
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    _single_tensor_sophiag(
        params,
        grads,
        exp_avgs,
        hessian,
        state_steps,
        bs=bs,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        capturable=capturable,
    )


def _single_tensor_sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    capturable: bool,
):
    del beta2
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        step_t += 1
        param.mul_(1 - lr * weight_decay)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
        if capturable:
            step_size_neg = lr.neg()
        else:
            step_size_neg = -lr
        param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)


def build_sophia_g_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> SophiaG:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    rho = float(kwargs.pop("rho", 0.04))
    maximize = bool(kwargs.pop("maximize", False))
    capturable = bool(kwargs.pop("capturable", False))
    hess_interval = int(kwargs.pop("hess_interval", kwargs.pop("interval", 10)))
    step_bs_scale = float(kwargs.pop("step_bs_scale", 1.0))
    if kwargs:
        raise ValueError(f"Unsupported Sophia-G kwargs: {sorted(kwargs.keys())}")
    optimizer = SophiaG(
        model.parameters(),
        lr=float(resolved_cfg["lr"]),
        betas=tuple(resolved_cfg["betas"]),
        rho=rho,
        weight_decay=float(resolved_cfg["weight_decay"]),
        maximize=maximize,
        capturable=capturable,
    )
    optimizer.hbtxr_hess_interval = max(0, hess_interval)
    optimizer.hbtxr_step_bs_scale = step_bs_scale
    return optimizer
