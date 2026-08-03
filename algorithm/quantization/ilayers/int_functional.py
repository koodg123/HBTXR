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


#: Must equal ``i_ops.DYADIC_MULT_BITS``. The copy below is deliberate; the VALUE is not
#: allowed to be — a different width here means the deployed kernel and the golden it is
#: checked against are computing different functions, which is the one thing the copy must
#: never buy. ``test_dyadic_copy_matches_golden`` pins it.
_DYADIC_MULT_BITS = 18


def _dyadic_params(scale: float, *, shift_min: int = 1, shift_max: int = 31,
                   wbits: int | None = _DYADIC_MULT_BITS) -> tuple[int, int]:
    """Best ``(multiplier, shift)`` with ``multiplier / 2^shift`` closest to ``scale``.

    Deliberately a copy of ``i_ops.dyadic_params`` rather than an import: no module under
    ``ilayers`` imports ``i_ops``, because ``i_ops`` is the golden these kernels are
    checked against and a reference that shares code with the thing it validates proves
    less. Same trade the hand-copied ``_pad_pair`` in ``ilayers/conv.py`` makes.

    ``wbits`` bounds the multiplier so it fits the hardware's multiplier port; see the
    note on ``i_ops.DYADIC_MULT_BITS`` for why the bound is on the multiplier and not on
    the shift.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    best: tuple[float, int, int] | None = None
    for shift in range(shift_min, shift_max + 1):
        multiplier = max(1, round(scale * (1 << shift)))
        if wbits is not None and multiplier.bit_length() > wbits:
            continue
        error = abs(multiplier / float(1 << shift) - scale)
        if best is None or error < best[0]:
            best = (error, multiplier, shift)
    if best is None:
        raise ValueError(f"scale {scale} needs more than {wbits} multiplier bits")
    return int(best[1]), int(best[2])


def rescale_to(int_data: torch.Tensor, scale_in: float, scale_out: float, *,
               dtype: QuantDtype = INT8, extra: float = 1.0) -> torch.Tensor:
    """Move integers from grid ``scale_in`` to grid ``scale_out`` — one requant unit.

    The ratio becomes a dyadic ``multiplier / 2^shift``: an integer multiply and an
    arithmetic right shift, which is what the hardware does and what keeps the datapath
    integer. ``extra`` folds a constant factor into the same operation — the attention
    ``1/√d`` is applied this way because in hardware it is not a separate multiply.

    The identity ratio is short-circuited: a dyadic approximation of exactly 1.0 would
    still round-trip through ``multiplier/2^shift`` and could move a value by an LSB for
    no reason.
    """
    ratio = (scale_in / scale_out) * extra
    if ratio == 1.0:
        return int_data.clamp(dtype.qmin, dtype.qmax).to(torch.int64)
    multiplier, shift = _dyadic_params(ratio)
    return requant(int_data, multiplier, shift, dtype=dtype).to(torch.int64)


__all__ = ["int_matmul", "requant", "rescale_to"]
