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

from quantization.ilayers.qtensor import QTensor, rescale_ratio
from quantization.scheme import INT8, QuantDtype


def _rescale_to(qt: QTensor, scale_out: float) -> torch.Tensor:
    """Integer data of ``qt`` re-expressed at ``scale_out`` (rounded, not yet clamped).

    ``scale_out`` is per-tensor, but the incoming scale need not be: an ``ILinear``
    accumulator carries a per-out-channel ``[C]`` vector, which aligns elementwise
    against the trailing dim (``QTensor``'s broadcast convention). Each operand of an
    ``IAdd`` / ``ICat`` is rescaled independently, so the operands may mix per-tensor
    and per-channel freely. float64 keeps the product exact for int32 accumulators,
    whose magnitudes run past float32's exact-integer limit of 2^24.
    """
    ratio = rescale_ratio(qt.scale, scale_out, qt.int_data.shape[-1])
    return torch.round(qt.int_data.to(torch.float64) * ratio)


def _check_pool_scale(scale: torch.Tensor | float, dim: int, ndim: int, channels: int) -> None:
    """Reject a per-channel scale that ``IPool`` would silently reduce across.

    ``IPool`` never truncates a scale — it forwards the whole object — so it does not
    have the element-0 bug. It has the other half of it: summing along an axis the
    scale varies over would add integers that are not on a common grid, and the
    surviving ``[C]`` vector would no longer line up with the output's trailing dim.
    Unlike softmax there is no bridge to a common grid here (``IPool`` preserves the
    input scale by definition), so this is a contract violation, not a rounding cost.
    """
    if not torch.is_tensor(scale):
        return
    flat = scale.reshape(-1)
    if flat.numel() == 1:
        return
    if flat.numel() != channels:
        raise ValueError(
            f"input scale must be per-tensor or per-channel ({channels} entries), "
            f"got {flat.numel()}")
    if dim % ndim == ndim - 1:
        raise ValueError(
            "IPool cannot pool over the per-channel axis: its elements are on different "
            f"scales (dim={dim} is the trailing dim of a {ndim}-D input)")


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
        _check_pool_scale(qt.scale, self.dim, qt.int_data.dim(), qt.int_data.shape[-1])
        n = qt.int_data.shape[self.dim]
        summed = qt.int_data.to(torch.int64).sum(dim=self.dim)
        # Integer round-half-away-from-zero, done entirely in int64. Going through
        # float32 here would lose low bits of exactly the int32-wide accumulators this
        # op is meant to accept (float32 is only exact for integers below 2^24), and a
        # sum over the token axis is the easiest place in the graph to exceed that.
        half = n // 2
        mean = torch.where(summed >= 0, (summed + half) // n, -((-summed + half) // n))
        mean = mean.clamp(qt.dtype.qmin, qt.dtype.qmax).to(torch.int32)
        return QTensor(mean, scale=qt.scale, zero_point=qt.zero_point, dtype=qt.dtype)


__all__ = ["IAdd", "ICat", "IPool"]
