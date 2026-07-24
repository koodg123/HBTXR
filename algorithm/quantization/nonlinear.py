"""HW-friendly integer LUT nonlinear modules for the HBTXR ViT.

``GeLULUT`` replaces ``nn.GELU`` with the HG-PIPE integer table kernel: quantize the
input, ``cursor = clamp((x_int + b) >> s, 0, bound)``, table lookup, dequant. A
straight-through estimator keeps it trainable under QAT.

``calibrate_gelu_luts`` observes each ``nn.GELU`` input over a few calibration
batches, builds a per-site LUT (lut_calibrate.build_gelu_lut), and swaps the GELU
for a calibrated ``GeLULUT`` in place. This is the pointwise piece of the
HW-friendly nonlinear path; Softmax / LayerNorm-rsqrt LUTs are follow-on work.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn

from quantization.lut_calibrate import build_function_lut, build_gelu_lut
from quantization.scheme import INT8, UINT8, QuantDtype

ForwardFn = Callable[[nn.Module, Any], Any]


def _table_lookup(x: torch.Tensor, b: int, s: int, bound: int, table: torch.Tensor, input_scale: float, output_scale: float) -> torch.Tensor:
    """Shared PoT-index integer table lookup -> dequantized float."""
    x_int = torch.round(x / input_scale).to(torch.int64)
    cursor = ((x_int + b) >> s).clamp(0, bound)
    return table[cursor].to(dtype=x.dtype) * output_scale


class GeLULUT(nn.Module):
    """Integer LUT GeLU (fake-quant with STE)."""

    def __init__(self, scalars: list[int], table: list[int], *, input_scale: float, output_scale: float) -> None:
        super().__init__()
        self.b, self.s, self.bound = (int(v) for v in scalars)
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.register_buffer("table", torch.as_tensor(list(table), dtype=torch.int64))

    def _lut(self, x: torch.Tensor) -> torch.Tensor:
        x_int = torch.round(x / self.input_scale).to(torch.int64)
        cursor = ((x_int + self.b) >> self.s).clamp(0, self.bound)
        return self.table[cursor].to(dtype=x.dtype) * self.output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._lut(x)
        return x + (y - x).detach()  # STE: forward = LUT GeLU, backward = identity


def _set_submodule(model: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parent = model
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def calibrate_gelu_luts(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    entries: int = 256,
    input_dtype: QuantDtype = INT8,
    output_dtype: QuantDtype = INT8,
) -> dict[str, GeLULUT]:
    """Observe each nn.GELU input, build a LUT, and replace it with a GeLULUT.

    Returns a mapping of module path -> inserted GeLULUT.
    """
    gelus = {name: module for name, module in model.named_modules() if isinstance(module, nn.GELU)}
    if not gelus:
        return {}

    samples: dict[str, list[torch.Tensor]] = {name: [] for name in gelus}

    def make_hook(name: str):
        def hook(_module, inputs):
            if inputs:
                samples[name].append(inputs[0].detach().reshape(-1).float().cpu())
        return hook

    handles = [gelus[name].register_forward_pre_hook(make_hook(name)) for name in gelus]
    run = forward_fn or (lambda m, b: m(b))
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                run(model, batch)
    finally:
        for handle in handles:
            handle.remove()

    inserted: dict[str, GeLULUT] = {}
    for name in gelus:
        collected = samples[name]
        if not collected:
            continue
        activations = torch.cat(collected).numpy()
        payload = build_gelu_lut(activations, entries=entries, input_dtype=input_dtype, output_dtype=output_dtype)
        lut = GeLULUT(payload["scalars"], payload["table"], input_scale=payload["input_scale"], output_scale=payload["output_scale"])
        _set_submodule(model, name, lut)
        inserted[name] = lut
    return inserted


class LayerNormLUT(nn.Module):
    """LayerNorm with a HW-friendly integer LUT for the rsqrt (1/sqrt(var)).

    Reductions (mean, variance) stay in higher precision; the transcendental rsqrt —
    the hardware-expensive part — is a calibrated integer table. STE forwards the LUT
    value but back-propagates the true rsqrt gradient, so weight/bias stay trainable.
    """

    def __init__(self, layernorm: nn.LayerNorm, rsqrt_payload: dict[str, object]) -> None:
        super().__init__()
        self.normalized_shape = tuple(layernorm.normalized_shape)
        self.eps = float(layernorm.eps)
        self.weight = layernorm.weight
        self.bias = layernorm.bias
        self.b, self.s, self.bound = (int(v) for v in rsqrt_payload["scalars"])
        self.input_scale = float(rsqrt_payload["input_scale"])
        self.output_scale = float(rsqrt_payload["output_scale"])
        self.register_buffer("rsqrt_table", torch.as_tensor(list(rsqrt_payload["table"]), dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        rsqrt_lut = _table_lookup(var, self.b, self.s, self.bound, self.rsqrt_table, self.input_scale, self.output_scale)
        rsqrt_true = torch.rsqrt(var + self.eps)
        rsqrt = rsqrt_true + (rsqrt_lut - rsqrt_true).detach()  # STE: forward=LUT, backward=true rsqrt
        y = (x - mean) * rsqrt
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


def calibrate_layernorm_luts(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    entries: int = 128,
) -> dict[str, LayerNormLUT]:
    """Observe each nn.LayerNorm input variance, build a rsqrt LUT, and swap it in."""
    norms = {name: m for name, m in model.named_modules() if isinstance(m, nn.LayerNorm)}
    if not norms:
        return {}
    samples: dict[str, list[torch.Tensor]] = {name: [] for name in norms}

    def make_hook(name: str):
        def hook(_module, inputs):
            if inputs:
                var = inputs[0].detach().float().var(dim=-1, unbiased=False).reshape(-1).cpu()
                samples[name].append(var)
        return hook

    handles = [norms[name].register_forward_pre_hook(make_hook(name)) for name in norms]
    run = forward_fn or (lambda m, b: m(b))
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                run(model, batch)
    finally:
        for handle in handles:
            handle.remove()

    inserted: dict[str, LayerNormLUT] = {}
    for name, norm in norms.items():
        if not samples[name]:
            continue
        var_samples = torch.cat(samples[name]).numpy()
        eps = float(norm.eps)
        payload = build_function_lut(
            var_samples,
            lambda v, e=eps: 1.0 / (np.sqrt(np.maximum(v, 0.0) + e)),
            entries=entries,
            input_dtype=UINT8,
            output_dtype=INT8,
        )
        lut = LayerNormLUT(norm, payload)
        _set_submodule(model, name, lut)
        inserted[name] = lut
    return inserted


class SoftmaxLUT(nn.Module):
    """Softmax with HW-friendly integer LUTs for exp and reciprocal.

    ``softmax(x) = exp(x - max) / sum(exp(x - max))``. The two transcendentals — exp
    and the 1/sum reciprocal — are calibrated integer tables; the max / sum reductions
    stay in higher precision. STE forwards the LUT values and back-propagates the true
    softmax gradient. Replaces ``nn.Softmax(dim=-1)``.
    """

    def __init__(self, exp_payload: dict[str, object], recip_payload: dict[str, object], *, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        self.exp_b, self.exp_s, self.exp_bound = (int(v) for v in exp_payload["scalars"])
        self.exp_in = float(exp_payload["input_scale"])
        self.exp_out = float(exp_payload["output_scale"])
        self.register_buffer("exp_table", torch.as_tensor(list(exp_payload["table"]), dtype=torch.int64))
        self.recip_b, self.recip_s, self.recip_bound = (int(v) for v in recip_payload["scalars"])
        self.recip_in = float(recip_payload["input_scale"])
        self.recip_out = float(recip_payload["output_scale"])
        self.register_buffer("recip_table", torch.as_tensor(list(recip_payload["table"]), dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maximum = x.max(dim=self.dim, keepdim=True).values
        delta = maximum - x  # >= 0
        exp_lut = _table_lookup(delta, self.exp_b, self.exp_s, self.exp_bound, self.exp_table, self.exp_in, self.exp_out)
        exp_true = torch.exp(-delta)
        numerator = exp_true + (exp_lut - exp_true).detach()  # STE: forward=LUT exp

        denom = numerator.sum(dim=self.dim, keepdim=True)
        recip_lut = _table_lookup(denom, self.recip_b, self.recip_s, self.recip_bound, self.recip_table, self.recip_in, self.recip_out)
        recip_true = 1.0 / denom.clamp_min(1e-8)
        recip = recip_true + (recip_lut - recip_true).detach()  # STE: forward=LUT recip
        return numerator * recip


def calibrate_softmax_luts(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    exp_entries: int = 128,
    recip_entries: int = 128,
) -> dict[str, SoftmaxLUT]:
    """Observe each nn.Softmax input, build exp + reciprocal LUTs, and swap it in."""
    softmaxes = {name: m for name, m in model.named_modules() if isinstance(m, nn.Softmax)}
    if not softmaxes:
        return {}
    deltas: dict[str, list[torch.Tensor]] = {name: [] for name in softmaxes}
    sums: dict[str, list[torch.Tensor]] = {name: [] for name in softmaxes}

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0].detach().float()
            delta = (x.max(dim=-1, keepdim=True).values - x)
            deltas[name].append(delta.reshape(-1).cpu())
            sums[name].append(torch.exp(-delta).sum(dim=-1).reshape(-1).cpu())
        return hook

    handles = [softmaxes[name].register_forward_pre_hook(make_hook(name)) for name in softmaxes]
    run = forward_fn or (lambda m, b: m(b))
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                run(model, batch)
    finally:
        for handle in handles:
            handle.remove()

    inserted: dict[str, SoftmaxLUT] = {}
    for name, softmax in softmaxes.items():
        if not deltas[name] or not sums[name]:
            continue
        delta_samples = torch.cat(deltas[name]).numpy()
        sum_samples = torch.cat(sums[name]).numpy()
        exp_payload = build_function_lut(delta_samples, lambda d: np.exp(-np.maximum(d, 0.0)), entries=exp_entries, input_dtype=UINT8, output_dtype=UINT8)
        recip_payload = build_function_lut(sum_samples, lambda s: 1.0 / np.maximum(s, 1e-8), entries=recip_entries, input_dtype=UINT8, output_dtype=UINT8)
        lut = SoftmaxLUT(exp_payload, recip_payload, dim=softmax.dim if softmax.dim is not None else -1)
        _set_submodule(model, name, lut)
        inserted[name] = lut
    return inserted


__all__ = [
    "GeLULUT",
    "calibrate_gelu_luts",
    "LayerNormLUT",
    "calibrate_layernorm_luts",
    "SoftmaxLUT",
    "calibrate_softmax_luts",
]
