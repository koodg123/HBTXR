"""I-tier integer matmul for activation×activation products (attention).

Unlike Linear/Conv (weight × activation), the attention scores ``Q @ Kᵀ`` and the
context ``attn @ V`` multiply two *activations*, so both operands are quantized from
calibrated activation scales. ``IMatMul`` accumulates in the integer domain and
dequantizes by ``s_a · s_b``; both activations are assumed symmetric (zero-point 0),
the usual choice for attention tensors.

``forward`` returns float; ``forward_accumulator`` exposes the int32 accumulator as a
``QTensor`` (scale ``s_a · s_b``) for the integer graph.
"""
from __future__ import annotations

import torch
from torch import nn

from quantization.ilayers.int_functional import int_matmul
from quantization.ilayers.qtensor import QTensor
from quantization.scheme import INT8, QuantDtype


class IMatMul(nn.Module):
    """Integer activation×activation matmul (batched over leading dims)."""

    def __init__(self, scale_a, scale_b, *, dtype_a: QuantDtype = INT8, dtype_b: QuantDtype = INT8):
        super().__init__()
        self.scale_a = float(scale_a)
        self.scale_b = float(scale_b)
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b

    def _quantize(self, x: torch.Tensor, scale: float, dtype: QuantDtype) -> torch.Tensor:
        return torch.round(x / scale).clamp(dtype.qmin, dtype.qmax).to(torch.int64)

    def _accumulate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_int = self._quantize(a, self.scale_a, self.dtype_a)
        b_int = self._quantize(b, self.scale_b, self.dtype_b)
        return int_matmul(a_int, b_int)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        acc = self._accumulate(a, b)
        return acc.to(torch.float32) * (self.scale_a * self.scale_b)

    def forward_accumulator(self, a: torch.Tensor, b: torch.Tensor) -> QTensor:
        acc = self._accumulate(a, b)
        return QTensor(acc.to(torch.int32), scale=(self.scale_a * self.scale_b), zero_point=0.0,
                       dtype=QuantDtype(32, signed=True))


__all__ = ["IMatMul"]
