"""I-tier integer tensor ops: residual add, concat, and token pooling on QTensors.

Two int8 tensors with *different* scales cannot be added directly — they are first
requantized to a common output scale (a per-operand multiply+shift in HW; a float
rescale here), summed in the wider integer domain, and clamped back to int8. Concat
does the same alignment before joining. Token pooling is an integer mean that keeps
the input scale.

- ``IAdd``  : residual ``x + sublayer(x)`` — align (s_a, s_b) -> s_out, add, clamp.
- ``ICat``  : concat a list of QTensors -> one scale (align each -> s_out).
- ``IPool`` : mean over a token dim (Σ int / N), scale preserved.
"""
from __future__ import annotations

import torch
from torch import nn

from quantization.ilayers.qtensor import QTensor
from quantization.scheme import INT8, QuantDtype


def _scalar(v) -> float:
    return float(v.reshape(-1)[0]) if torch.is_tensor(v) else float(v)


def _rescale_to(qt: QTensor, scale_out: float) -> torch.Tensor:
    """Integer data of ``qt`` re-expressed at ``scale_out`` (rounded, not yet clamped)."""
    return torch.round(qt.int_data.to(torch.float32) * (_scalar(qt.scale) / scale_out))


class IAdd(nn.Module):
    """Residual add of two QTensors, aligned to a common output scale."""

    def __init__(self, scale_out, *, dtype: QuantDtype = INT8):
        super().__init__()
        self.scale_out = float(scale_out)
        self.dtype = dtype

    def forward(self, qa: QTensor, qb: QTensor) -> QTensor:
        acc = _rescale_to(qa, self.scale_out) + _rescale_to(qb, self.scale_out)
        out = acc.clamp(self.dtype.qmin, self.dtype.qmax).to(torch.int32)
        return QTensor(out, scale=self.scale_out, zero_point=0.0, dtype=self.dtype)


class ICat(nn.Module):
    """Concat QTensors along ``dim`` after aligning them to a common output scale."""

    def __init__(self, scale_out, *, dim: int = -1, dtype: QuantDtype = INT8):
        super().__init__()
        self.scale_out = float(scale_out)
        self.dim = dim
        self.dtype = dtype

    def forward(self, parts: list[QTensor]) -> QTensor:
        aligned = [
            _rescale_to(p, self.scale_out).clamp(self.dtype.qmin, self.dtype.qmax) for p in parts
        ]
        out = torch.cat(aligned, dim=self.dim).to(torch.int32)
        return QTensor(out, scale=self.scale_out, zero_point=0.0, dtype=self.dtype)


class IPool(nn.Module):
    """Integer mean over a token dimension (scale preserved)."""

    def __init__(self, *, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, qt: QTensor) -> QTensor:
        n = qt.int_data.shape[self.dim]
        summed = qt.int_data.to(torch.int64).sum(dim=self.dim)
        mean = torch.round(summed.to(torch.float32) / n).clamp(qt.dtype.qmin, qt.dtype.qmax).to(torch.int32)
        return QTensor(mean, scale=qt.scale, zero_point=qt.zero_point, dtype=qt.dtype)


__all__ = ["IAdd", "ICat", "IPool"]
