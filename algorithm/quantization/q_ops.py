"""Q-tier quantization primitives: fake quantizers + STE.

- ``AffineFakeQuantizer``: affine quantize->dequantize with a straight-through
  estimator (STE) on the rounding so gradients flow during QAT. Used on Linear
  weights and activations.
- ``LUTFakeQuantizer``: the HG-PIPE LUT (table) fake quantizer for HW-friendly
  integer nonlinear operators.

Scale/zero-point are registered buffers set from calibration; they default to
identity so an inserted-but-uncalibrated model still runs.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from quantization.grouping import from_slots, to_slots
from quantization.scheme import QuantDtype, INT8
from quantization.spec import TensorQuantSpec


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through estimator (identity gradient)."""
    return x + (torch.round(x) - x).detach()


class AffineFakeQuantizer(nn.Module):
    """Affine fake quantization with STE rounding (QAT-ready), configurable per ``spec``.

    ``scale`` / ``zero_point`` are per-slot buffers (shape ``[num_slots]``) driven by
    the granularity in ``spec`` (grouping.to_slots). A bare ``dtype`` construction
    keeps the original per-tensor symmetric behaviour with scalar-like ``[1]`` params.
    """

    def __init__(
        self,
        dtype: QuantDtype = INT8,
        *,
        scale: float = 1.0,
        zero_point: int = 0,
        spec: TensorQuantSpec | None = None,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.spec = spec or TensorQuantSpec(bits=dtype.bits, signed=dtype.signed)
        self.dtype = self.spec.dtype
        self.qmin = self.dtype.qmin
        self.qmax = self.dtype.qmax
        # per-slot buffers; start as [1] and get replaced (any [S]) at calibration.
        self.register_buffer("scale", torch.tensor([float(scale)]))
        self.register_buffer("zero_point", torch.tensor([float(int(zero_point))]))

    def set_qparams(self, scale, zero_point=0) -> None:
        scale_t = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if bool((scale_t <= 0).any()):
            raise ValueError("scale must be positive")
        zp_t = torch.as_tensor(zero_point, dtype=torch.float32).reshape(-1)
        if zp_t.numel() == 1 and scale_t.numel() > 1:
            zp_t = zp_t.expand_as(scale_t).clone()
        self.scale = scale_t.to(self.scale.device)          # updates the registered buffer
        self.zero_point = zp_t.to(self.zero_point.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slots = to_slots(x, self.spec)                       # [S, k]
        scale = self.scale.view(-1, 1)
        zero_point = self.zero_point.view(-1, 1)
        q = ste_round(slots / scale + zero_point)
        q = torch.clamp(q, self.qmin, self.qmax)
        deq = (q - zero_point) * scale
        return from_slots(deq, x.shape, self.spec)


class LUTFakeQuantizer(nn.Module):
    """HG-PIPE LUT fake quantizer: cursor = (round(x) + b) >> s, clamp, table[cursor]."""

    def __init__(self, *, scalars: Iterable[int], table: Iterable[int]) -> None:
        super().__init__()
        scalar_values = [int(v) for v in scalars]
        if len(scalar_values) != 3:
            raise ValueError(f"LUTFakeQuantizer expects 3 scalars (b, s, bound), got {len(scalar_values)}")
        self.b, self.s, self.bound = scalar_values
        self.register_buffer("table", torch.as_tensor(list(table), dtype=torch.int64))

    def _lookup(self, x: torch.Tensor) -> torch.Tensor:
        x_int = torch.round(x).to(torch.int64)
        cursor = ((x_int + self.b) >> self.s).clamp(0, self.bound)
        return self.table[cursor].to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._lookup(x)
        return x + (y - x).detach()  # STE: forward = table lookup, backward = identity


__all__ = ["ste_round", "AffineFakeQuantizer", "LUTFakeQuantizer"]
