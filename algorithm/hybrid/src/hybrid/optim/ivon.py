# Source: https://raw.githubusercontent.com/team-approx-bayes/ivon/refs/heads/main/ivon/_ivon.py
# Upstream: team-approx-bayes/ivon ivon/_ivon.py
# Diff: see docs/hbtxr/optimizers/ivon_diff.md

from __future__ import annotations

from contextlib import contextmanager
from math import pow
from typing import Any, Callable, Optional
import torch.distributed as dist

import torch
from torch import Tensor, nn


ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(torch.optim.Optimizer):
    hessian_approx_methods = ("price", "gradsq")

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = "price",
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 1 <= mc_samples:
            raise ValueError(f"Invalid number of MC samples: {mc_samples}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0.0 < hess_init:
            raise ValueError(f"Invalid Hessian initialization: {hess_init}")
        if not 0.0 < ess:
            raise ValueError(f"Invalid effective sample size: {ess}")
        if not 0.0 < clip_radius:
            raise ValueError(f"Invalid clipping radius: {clip_radius}")
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError(f"Invalid hess_approx parameter: {hess_approx}")
        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)
        self.mc_samples = int(mc_samples)
        self.hess_approx = str(hess_approx)
        self.sync = bool(sync)
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = bool(debias)
        self.rescale_lr = bool(rescale_lr)
        self._reset_samples()
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        for group in self.param_groups:
            group["numel"] = sum(p.numel() for p in group["params"] if p is not None)
            all_params += [p for p in group["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(f"Parameters are on different devices: {[str(d) for d in devices]}")
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(f"Parameters are on different dtypes: {[str(d) for d in dtypes]}")
        dtype = next(iter(dtypes))
        total = sum(group["numel"] for group in self.param_groups)
        return total, device, dtype

    def _reset_samples(self) -> None:
        self.state["count"] = 0
        self.state["avg_grad"] = None
        self.state["avg_nxg"] = None
        self.state["avg_gsq"] = None

    def _init_buffers(self) -> None:
        for group in self.param_groups:
            hess_init = float(group["hess_init"])
            numel = int(group["numel"])
            group["momentum"] = torch.zeros(numel, device=self._device, dtype=self._dtype)
            group["hess"] = torch.zeros(numel, device=self._device, dtype=self._dtype).add(torch.as_tensor(hess_init))

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(self, train: bool, param_avg: Tensor, noise: Tensor) -> None:
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue
                p_slice = slice(offset, offset + p.numel())
                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    if p.requires_grad and p.grad is not None:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(torch.zeros_like(p).flatten())
                offset += p.numel()
        assert offset == self._numel

        if train:
            grad_sample = torch.cat(param_grads, 0)
            count = int(self.state["count"]) + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(self.state["avg_grad"], grad_sample, count)
            if self.hess_approx == "price":
                self.state["avg_nxg"] = _welford_mean(self.state["avg_nxg"], noise * grad_sample, count)
            elif self.hess_approx == "gradsq":
                self.state["avg_gsq"] = _welford_mean(self.state["avg_gsq"], grad_sample.square(), count)

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self) -> None:
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)
        if self.state["avg_nxg"] is not None:
            dist.all_reduce(self.state["avg_nxg"])
            self.state["avg_nxg"].div_(world_size)
        if self.state["avg_gsq"] is not None:
            dist.all_reduce(self.state["avg_gsq"])
            self.state["avg_gsq"].div_(world_size)

    def _sample_params(self):
        noise_samples = []
        param_avgs = []
        offset = 0
        for group in self.param_groups:
            gnumel = int(group["numel"])
            noise_sample = torch.randn(gnumel, device=self._device, dtype=self._dtype) / (
                group["ess"] * (group["hess"] + group["weight_decay"])
            ).sqrt()
            noise_samples.append(noise_sample)
            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue
                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[goffset : goffset + numel]
                param_avgs.append(p_avg)
                p.data = (p_avg + p_noise).view(p.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]
        assert offset == self._numel
        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    def _update(self) -> None:
        self.current_step += 1
        offset = 0
        for group in self.param_groups:
            lr = float(group["lr"])
            b1 = float(group["beta1"])
            b2 = float(group["beta2"])
            pg_slice = slice(offset, offset + int(group["numel"]))
            param_avg = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)
            group["momentum"] = self._new_momentum(self.state["avg_grad"][pg_slice], group["momentum"], b1)
            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state["avg_gsq"],
                pg_slice,
                float(group["ess"]),
                b2,
                float(group["weight_decay"]),
            )
            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["momentum"],
                lr * (float(group["hess_init"]) + float(group["weight_decay"])) if self.rescale_lr else lr,
                float(group["weight_decay"]),
                float(group["clip_radius"]),
                1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0,
                float(group["hess_init"]),
            )
            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(p.shape)
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]
            offset += group["numel"]
        assert offset == self._numel

    @staticmethod
    def _get_nll_hess(method: str, hess: Tensor, avg_nxg: Optional[Tensor], avg_gsq: Optional[Tensor], pg_slice) -> Tensor:
        if method == "price":
            assert avg_nxg is not None
            return avg_nxg[pg_slice] * hess
        if method == "gradsq":
            assert avg_gsq is not None
            return avg_gsq[pg_slice]
        raise NotImplementedError(f"unknown hessian approx.: {method}")

    @staticmethod
    def _new_momentum(avg_grad: Tensor, momentum: Tensor, beta1: float) -> Tensor:
        return beta1 * momentum + (1.0 - beta1) * avg_grad

    @staticmethod
    def _new_hess(method: str, hess: Tensor, avg_nxg: Optional[Tensor], avg_gsq: Optional[Tensor], pg_slice, ess: float, beta2: float, wd: float) -> Tensor:
        fisher = IVON._get_nll_hess(method, hess + wd, avg_nxg, avg_gsq, pg_slice) * ess
        return beta2 * hess + (1.0 - beta2) * fisher + (0.5 * (1 - beta2) ** 2) * (hess - fisher).square() / (hess + wd)

    @staticmethod
    def _new_param_averages(
        param_avg: Tensor,
        hess: Tensor,
        momentum: Tensor,
        lr: float,
        wd: float,
        clip_radius: float,
        debias: float,
        hess_init: float,
    ) -> Tensor:
        del hess_init
        return param_avg - lr * torch.clip((momentum / debias + wd * param_avg) / (hess + wd), min=-clip_radius, max=clip_radius)


def build_ivon_optimizer(model: nn.Module, resolved_cfg: dict[str, Any]) -> IVON:
    kwargs = dict(resolved_cfg.get("kwargs") or {})
    ess = float(kwargs.pop("ess", 1.0))
    hess_init = float(kwargs.pop("hess_init", 1.0))
    mc_samples = int(kwargs.pop("mc_samples", 1))
    hess_approx = str(kwargs.pop("hess_approx", "price"))
    clip_radius = float(kwargs.pop("clip_radius", float("inf")))
    sync = bool(kwargs.pop("sync", False))
    debias = bool(kwargs.pop("debias", True))
    rescale_lr = bool(kwargs.pop("rescale_lr", True))
    if kwargs:
        raise ValueError(f"Unsupported IVON kwargs: {sorted(kwargs.keys())}")
    return IVON(
        model.parameters(),
        lr=float(resolved_cfg["lr"]),
        ess=ess,
        hess_init=hess_init,
        beta1=float(resolved_cfg["betas"][0]),
        beta2=float(resolved_cfg["betas"][1]),
        weight_decay=float(resolved_cfg["weight_decay"]),
        mc_samples=mc_samples,
        hess_approx=hess_approx,
        clip_radius=clip_radius,
        sync=sync,
        debias=debias,
        rescale_lr=rescale_lr,
    )
