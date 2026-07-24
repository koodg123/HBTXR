"""FakeQuantizer modules for HBTXR quantization (QAT + PTQ).

Adapted from the HG-PIPE reference ``fake_quant/modules.py``:

- ``AffineFakeQuantizer``: affine quantize->dequantize with a straight-through
  estimator (STE) on the rounding so gradients flow during QAT (the HG-PIPE
  original used plain ``round`` for inference-only verification). Used on Linear
  weights and activations.
- ``LUTFakeQuantizer``: the HG-PIPE LUT (table) fake quantizer — cursor =
  ``(round(x) + b) >> s``, clamp, table lookup — for the HW-friendly integer
  nonlinear operators (LayerNorm rsqrt / Softmax / GeLU). Inference behaviour;
  gradients pass straight through the table lookup (STE).

Scale/zero-point are registered buffers set from calibration (a later phase);
they default to identity so an inserted-but-uncalibrated model still runs.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from quantization.scheme import QuantDtype, INT8


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through estimator (identity gradient)."""
    return x + (torch.round(x) - x).detach()


class AffineFakeQuantizer(nn.Module):
    """Affine fake quantization with STE rounding (QAT-ready)."""

    def __init__(self, dtype: QuantDtype = INT8, *, scale: float = 1.0, zero_point: int = 0) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.dtype = dtype
        self.qmin = dtype.qmin
        self.qmax = dtype.qmax
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("zero_point", torch.tensor(float(int(zero_point))))

    def set_qparams(self, scale: float, zero_point: int = 0) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale.fill_(float(scale))
        self.zero_point.fill_(float(int(zero_point)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = ste_round(x / self.scale + self.zero_point)
        q = torch.clamp(q, self.qmin, self.qmax)
        return (q - self.zero_point) * self.scale


class LUTFakeQuantizer(nn.Module):
    """HG-PIPE LUT fake quantizer: cursor = (round(x) + b) >> s, clamp, table[cursor].

    Input is interpreted in the integer domain (rounded); output is float so it can
    stay in a fake-quant graph. The table lookup is non-differentiable, so an STE
    passes the upstream gradient through unchanged during QAT.
    """

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
