"""Q-tier HW-friendly integer LUT nonlinear modules for the HBTXR ViT.

- ``QGeLU``: integer LUT GeLU (cursor = clamp((x_int + b) >> s, 0, bound), table lookup).
- ``QLayerNorm``: LayerNorm with a calibrated integer rsqrt LUT (reductions kept
  precise; STE forwards the LUT, back-props the true rsqrt).
- ``QSoftmax``: softmax with integer exp + reciprocal LUTs (STE toward true softmax).

Each carries pre-calibrated tables; the observe->build->swap calibration lives in
convert.py.
"""
from __future__ import annotations

import torch
from torch import nn


def _table_lookup(x: torch.Tensor, b: int, s: int, bound: int, table: torch.Tensor, input_scale: float, output_scale: float) -> torch.Tensor:
    """Shared PoT-index integer table lookup -> dequantized float."""
    x_int = torch.round(x / input_scale).to(torch.int64)
    cursor = ((x_int + b) >> s).clamp(0, bound)
    return table[cursor].to(dtype=x.dtype) * output_scale


class QGeLU(nn.Module):
    """Integer LUT GeLU (fake-quant with STE)."""

    def __init__(self, scalars: list[int], table: list[int], *, input_scale: float, output_scale: float) -> None:
        super().__init__()
        self.b, self.s, self.bound = (int(v) for v in scalars)
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.register_buffer("table", torch.as_tensor(list(table), dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _table_lookup(x, self.b, self.s, self.bound, self.table, self.input_scale, self.output_scale)
        return x + (y - x).detach()  # STE: forward = LUT GeLU, backward = identity


class QLayerNorm(nn.Module):
    """LayerNorm with a HW-friendly integer LUT for the rsqrt (1/sqrt(var))."""

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


class QSoftmax(nn.Module):
    """Softmax with HW-friendly integer LUTs for exp and reciprocal."""

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


__all__ = ["QGeLU", "QLayerNorm", "QSoftmax"]
