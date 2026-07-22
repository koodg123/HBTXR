# Source: https://raw.githubusercontent.com/zyushun/Adam-mini/main/adam_mini/adam_mini.py
# Upstream: zyushun/Adam-mini adam_mini/adam_mini.py
# Diff: see docs/hbtxr/optimizers/adam_mini_diff.md

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class AdamMini(Optimizer):
    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        *,
        model_sharding: bool | None = None,
        dim: int = 2048,
        n_heads: int = 32,
        n_kv_heads: int | None = None,
        verbose: bool = False,
    ) -> None:
        del model_sharding
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
        if int(dim) <= 0:
            raise ValueError(f"Invalid dim value: {dim}")
        if int(n_heads) <= 0:
            raise ValueError(f"Invalid n_heads value: {n_heads}")

        self.named_parameters = list(named_parameters)
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else self.n_heads
        self.verbose = bool(verbose)
        self.head_numel = (self.dim * self.dim // self.n_heads) if (self.dim * self.dim) % self.n_heads == 0 else 0

        self.embd_names = {"embed", "embd", "wte"}
        self.output_names = {"lm_head", "output", "final_layer"}
        self.wqk_names = {"k_proj", "q_proj", "wq", "wk", "query", "key"}
        self.wv_names = {"v_proj", "wv", "value"}
        self.attn_proj_names = {"o_proj", "wo", "attn.proj", "out_proj"}
        self.mlp_names = {"feed_forward", "linear", "mlp", "fc1", "fc2"}
        self.adam_block_names = {"bias"}

        optim_groups: list[dict[str, Any]] = []
        for param_name, param in self.named_parameters:
            if not param.requires_grad:
                continue
            lowered = str(param_name).lower()
            group = {
                "name": lowered,
                "params": [param],
                "weight_decay": 0.0 if ("norm" in lowered or "ln" in lowered or "bias" in lowered or param.ndim <= 1) else weight_decay,
            }
            optim_groups.append(group)

        defaults = dict(lr=lr, beta1=float(betas[0]), beta2=float(betas[1]), eps=eps)
        super().__init__(optim_groups, defaults)

    def _is_adam_group(self, name: str, param: torch.Tensor) -> bool:
        return param.ndim <= 1 or any(token in name for token in self.adam_block_names) or "norm" in name or "ln" in name

    def _is_headwise_group(self, name: str, param: torch.Tensor) -> bool:
        return self.head_numel > 0 and param.numel() % self.head_numel == 0 and any(token in name for token in self.wqk_names)

    def _is_rowwise_group(self, name: str, param: torch.Tensor) -> bool:
        rowwise_names = self.embd_names | self.output_names | self.wv_names | self.attn_proj_names | self.mlp_names
        return param.ndim >= 2 and any(token in name for token in rowwise_names)

    def _step_adam_group(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["v"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["step"] = 0
        state["step"] += 1
        if weight_decay > 0.0:
            param.mul_(1 - lr * weight_decay)
        state["m"].lerp_(grad, 1 - beta1)
        state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        bias_correction_1 = 1 - beta1**int(state["step"])
        bias_correction_2 = 1 - beta2**int(state["step"])
        denom = state["v"].sqrt().div_(math.sqrt(bias_correction_2)).add_(eps)
        param.addcdiv_(state["m"], denom, value=-(lr / bias_correction_1))

    def _step_headwise_group(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        head_numel = self.head_numel
        grad_heads = grad.reshape(-1, head_numel)
        if len(state) == 0:
            state["m"] = torch.zeros_like(grad_heads, memory_format=torch.preserve_format)
            state["vmean"] = torch.zeros_like(grad_heads[:, :1], memory_format=torch.preserve_format)
            state["step"] = 0
        state["step"] += 1
        if weight_decay > 0.0:
            param.mul_(1 - lr * weight_decay)
        state["m"].lerp_(grad_heads, 1 - beta1)
        tmp_lr = torch.mean(grad_heads * grad_heads, dim=1, keepdim=True)
        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
        bias_correction_1 = 1 - beta1**int(state["step"])
        bias_correction_2 = 1 - beta2**int(state["step"])
        h = state["vmean"].sqrt().div_(math.sqrt(bias_correction_2)).add_(eps)
        update = (state["m"] * (((1.0 / bias_correction_1) / h).view(grad_heads.shape[0], 1))).reshape_as(param)
        param.add_(update, alpha=-lr)

    def _step_rowwise_group(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        grad_rows = grad.reshape(grad.shape[0], -1)
        if len(state) == 0:
            state["m"] = torch.zeros_like(grad_rows, memory_format=torch.preserve_format)
            state["vmean"] = torch.zeros_like(grad_rows[:, :1], memory_format=torch.preserve_format)
            state["step"] = 0
        state["step"] += 1
        if weight_decay > 0.0:
            param.mul_(1 - lr * weight_decay)
        state["m"].lerp_(grad_rows, 1 - beta1)
        tmp_lr = torch.mean(grad_rows * grad_rows, dim=1, keepdim=True)
        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
        bias_correction_1 = 1 - beta1**int(state["step"])
        bias_correction_2 = 1 - beta2**int(state["step"])
        h = state["vmean"].sqrt().div_(math.sqrt(bias_correction_2)).add_(eps)
        update = (state["m"] * (((1.0 / bias_correction_1) / h).view(grad_rows.shape[0], 1))).reshape_as(param)
        param.add_(update, alpha=-lr)

    def _step_blockwise_group(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["vmean"] = torch.zeros((), device=param.device, dtype=param.dtype)
            state["step"] = 0
        state["step"] += 1
        if weight_decay > 0.0:
            param.mul_(1 - lr * weight_decay)
        state["m"].lerp_(grad, 1 - beta1)
        tmp_lr = torch.mean(grad * grad)
        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
        bias_correction_1 = 1 - beta1**int(state["step"])
        bias_correction_2 = 1 - beta2**int(state["step"])
        h = state["vmean"].sqrt().div_(math.sqrt(bias_correction_2)).add_(eps)
        update = state["m"] * ((1.0 / bias_correction_1) / h.to(state["m"].device))
        param.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            lr = float(group["lr"])
            eps = float(group["eps"])
            name = str(group["name"])
            weight_decay = float(group["weight_decay"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]

                if self._is_adam_group(name, param):
                    self._step_adam_group(param, grad, state, beta1=beta1, beta2=beta2, lr=lr, eps=eps, weight_decay=weight_decay)
                elif self._is_headwise_group(name, param):
                    self._step_headwise_group(param, grad, state, beta1=beta1, beta2=beta2, lr=lr, eps=eps, weight_decay=weight_decay)
                elif self._is_rowwise_group(name, param):
                    self._step_rowwise_group(param, grad, state, beta1=beta1, beta2=beta2, lr=lr, eps=eps, weight_decay=weight_decay)
                else:
                    self._step_blockwise_group(param, grad, state, beta1=beta1, beta2=beta2, lr=lr, eps=eps, weight_decay=weight_decay)

        return loss


def _infer_num_heads(model: nn.Module) -> int | None:
    direct = getattr(model, "num_heads", None)
    if isinstance(direct, int) and direct > 0:
        return direct
    backbone = getattr(model, "backbone", None)
    attn_stages = getattr(backbone, "attn_stages", None)
    if attn_stages and len(attn_stages) > 0:
        candidate = getattr(attn_stages[0], "num_heads", None)
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return None


def build_adam_mini_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> AdamMini:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    dim = int(kwargs.pop("dim", getattr(model, "embed_dim", 2048)))
    n_heads = int(kwargs.pop("n_heads", _infer_num_heads(model) or 32))
    n_kv_heads_raw = kwargs.pop("n_kv_heads", None)
    n_kv_heads = int(n_kv_heads_raw) if n_kv_heads_raw is not None else None
    verbose = bool(kwargs.pop("verbose", False))
    model_sharding = kwargs.pop("model_sharding", None)
    if kwargs:
        raise ValueError(f"Unsupported Adam-mini kwargs: {sorted(kwargs.keys())}")
    return AdamMini(
        model.named_parameters(),
        lr=float(resolved_cfg["lr"]),
        betas=tuple(resolved_cfg["betas"]),
        eps=float(resolved_cfg["eps"]),
        weight_decay=float(resolved_cfg["weight_decay"]),
        model_sharding=model_sharding,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        verbose=verbose,
    )
