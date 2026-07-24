"""Torch integer deployment kernels — bit-exact against the ``i_ops`` pure-int golden.

These are the HW-faithful primitives the I-tier layers are built from (§1 policy):

- ``int_matmul``   : int operands cast to int64 and accumulated (exact for ViT dims;
  models the int8×int8→int32 MAC array).
- ``requant``      : ``clamp((acc*multiplier + round) >> shift + zp)`` — the fixed-point
  multiply+shift a hardware requant unit performs. ``multiplier`` / ``shift`` may be
  scalars or per-channel tensors (broadcastable to ``acc``) for per-channel scales.

Right shift is arithmetic (sign-extending), matching Python ``>>`` in the golden, so
negative accumulators requantize identically.
"""
from __future__ import annotations

import torch

from quantization.scheme import INT8, QuantDtype


def int_matmul(a_int: torch.Tensor, b_int: torch.Tensor) -> torch.Tensor:
    """Integer matmul with int64 accumulation (returns int64 accumulator tensor)."""
    return a_int.to(torch.int64) @ b_int.to(torch.int64)


def requant(
    acc: torch.Tensor,
    multiplier,
    shift,
    *,
    dtype: QuantDtype = INT8,
    zero_point=0,
) -> torch.Tensor:
    """Dyadic requant of an integer accumulator to ``dtype`` (returns int32 tensor)."""
    device = acc.device
    acc64 = acc.to(torch.int64)
    mult = torch.as_tensor(multiplier, dtype=torch.int64, device=device)
    sh = torch.as_tensor(shift, dtype=torch.int64, device=device)
    if bool((sh < 0).any()):
        raise ValueError("shift must be non-negative")
    one = torch.ones_like(sh)
    rnd = torch.where(sh > 0, torch.bitwise_left_shift(one, sh - 1), torch.zeros_like(sh))
    scaled = acc64 * mult + rnd
    shifted = torch.bitwise_right_shift(scaled, sh)
    zp = torch.as_tensor(zero_point, dtype=torch.int64, device=device)
    out = (shifted + zp).clamp(dtype.qmin, dtype.qmax)
    return out.to(torch.int32)


__all__ = ["int_matmul", "requant"]
