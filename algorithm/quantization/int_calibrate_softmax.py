"""Calibration of the 14 scalars + 3 tables of the fully-integer Softmax.

``i_ops.softmax_quantize`` (and its torch twin ``ilayers.softmax.ISoftmax``) is the
HG-PIPE integer softmax: integer row max, an *inverse* exp table over ``max - x``, an
integer accumulation of the exponentials, and a **segmented** reciprocal of that
accumulator (two PoT-indexed tables, each with its own final requant ``(b3, s3)``).
This module produces those artifacts from observed pre-softmax attention logits.

The fixed-point algebra the scalars have to satisfy (why the constants are what they
are — the reference heuristic gets this wrong, see below):

    exp_value_i ~= exp(-(max - x_i) * input_scale) * EXP_SCALE
    acc         =  sum_i exp_value_i  ~= EXP_SCALE * S      (S = true softmax denom)
    recip       ~= NUMER / acc        (table entry, NUMER chosen per segment)
    out_i       =  (exp_value_i * recip + b3) >> s3
                ~= (NUMER / EXP_SCALE) * p_i >> s3          (p_i = softmax prob)

so ``out_i ~= p_i * qmax`` requires ``NUMER = qmax << s3`` with ``NUMER`` applied
*directly to acc* — i.e. ``EXP_SCALE`` cancels between numerator and denominator and
must not appear again in ``s3``. The HG-PIPE reference
(``references/hardware/hg-pipe-quantization/src/lut_calibration.py::calibrate_softmax``)
instead emits ``recip = recip_scale / acc`` with a small ``recip_scale`` (256) while
setting ``s3 = ceil(log2(exp_scale * recip_scale / qmax))``; with ``acc ~ 3e4..6e6``
that reciprocal table is all zeros and the shift is ~2^15 too large, so the composed
op returns 0 everywhere. We keep the reference's *structure* (inverse-exp PoT table,
pivot-split dual reciprocal, 14-scalar layout) and fix the numerics.

Three deviations from the reference:

1. The reciprocal segments span the **theoretical** accumulator range, not the observed
   one (``_theoretical_accumulator_range``). The observed range is a trap: an
   accumulator above it saturates ``cursor_two`` at ``bound2_two`` and the row is then
   divided by the *calibration* maximum instead of its own sum, so the row silently
   stops being normalised. Calibrating on peaky-only logits and running on flat ones
   measured row sums of 6.40 at 16 tokens and 12.80 at 64 tokens (they should be ~1.0).
   The theoretical range needs no data at all and makes that saturation unreachable —
   see ``_theoretical_accumulator_range`` for the two-line proof. The price is
   resolution: two 256-entry linear segments spread over a range of ratio
   ``max_tokens`` give a worst-case relative bin width of about
   ``2 * (sqrt(max_tokens) - 1) / (recip_entries - 1)``, so row sums drift by a couple
   of percent at large token counts instead of by a factor of 6.

2. The default segment pivot is the geometric mean of that range: with two linear
   segments over ``[lo, hi]`` the pivot ``sqrt(lo * hi)`` equalises the segments'
   relative span (both cover a ratio of ``sqrt(hi / lo)``), which is the best that can
   be argued without looking at data. That argument does *not* survive contact with the
   PoT shift, which rounds each segment's bin width up to a power of two: over 32
   fixtures x 4 table sizes (128 comparisons) the reference pivot ``lo + (hi - lo) / 8``
   won 44, this one 26, 58 tied, and the worst case over the whole sweep was identical
   (0.2235 either way). So this is a defensible default, not a measured improvement;
   ``pivot`` selects between them and ``test_pivot_strategies_are_both_bit_exact``
   covers both.

3. Reciprocal table entries are evaluated at the **geometric mean** of their PoT bin
   rather than at the bin's left edge (the reference's ``coordinates_for``). The left
   edge is the smallest accumulator in the bin, so ``NUMER / acc`` is biased high for
   every other accumulator that lands there and the whole row inflates: with a 32-entry
   reciprocal the left edge drives row sums to 1.1412 against 1.0667 for the geometric
   mean, and with 16 entries to 1.2824 against 1.1216. This is the one representative
   choice that moves accuracy, and ``test_isoftmax_accuracy_with_hardware_sized_tables``
   pins it down by rebuilding the payload both ways.
   The exp table also samples an interior point (the arithmetic bin midpoint), but that
   is emphatically *not* an accuracy claim: the bins are equally wide, so any fixed
   within-bin offset multiplies every ``exp_value`` — and therefore ``acc`` — by the
   same factor, and that factor cancels between numerator and denominator of the softmax
   ratio. Rebuilding with left-edge exp sampling moves the max abs error by rounding
   noise with no consistent sign (64/32 tables: 0.0895 midpoint vs 0.1014 left edge;
   32/32: 0.1758 vs 0.1326; and at the default ``exp_entries=256`` the exp shift is 0,
   so the two are literally the same table). The midpoint is used because it is the
   unbiased representative of its bin; ``test_exp_table_representative_is_not_a_lever``
   records that it buys nothing.
"""
from __future__ import annotations

import math

import numpy as np

from quantization.lut_calibrate import (
    PotIndexParams,
    coordinates_for,
    cursor_for,
    make_pot_index_params,
)
from quantization.scheme import INT8, QuantDtype, qrange

_S3_MAX = 48  # keep (exp_value * recip + b3) and b3 itself far inside int64


def _as_rows(values) -> np.ndarray:
    """Observed logits as a 2D ``[rows, tokens]`` float matrix (softmax over dim -1)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("softmax calibration samples must not be empty")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.reshape(-1, arr.shape[-1])


def _bin_midpoints(params: PotIndexParams) -> np.ndarray:
    """Centre of each PoT bin: bin ``i`` catches ``[coord_i, coord_i + 2^shift - 1]``."""
    return coordinates_for(params).astype(np.float64) + ((1 << params.shift) - 1) / 2.0


def _bin_geometric_means(params: PotIndexParams) -> np.ndarray:
    """``sqrt(lo * hi)`` per PoT bin — the representative that equalises ``1/x`` error."""
    lo = np.maximum(coordinates_for(params).astype(np.float64), 1.0)
    hi = np.maximum(lo + float((1 << params.shift) - 1), 1.0)
    return np.sqrt(lo * hi)


def _theoretical_accumulator_range(exp_table: np.ndarray, max_tokens: int) -> tuple[int, int]:
    """Exact ``[min, max]`` of the row accumulator over *all* possible logits.

    The exp argument is ``delta = row_max - x``, so every row contributes at least one
    ``delta == 0``, whose cursor is ``clamp((0 + b1) >> s1, ...) == 0`` (``b1 == 0``
    because the exp index is built over ``[0, delta_max]``). Hence

        acc >= exp_table[0]                     (one element at the row max, rest >= 0)
        acc <= max_tokens * max(exp_table)      (a constant row: every delta is 0)

    Both bounds are attained, so this is the tight envelope — and it depends only on the
    exp table and the token count, never on the calibration sample. Sizing the reciprocal
    segments from it is what makes ``cursor_two`` saturation unreachable.
    """
    lo = max(int(exp_table[0]), 1)
    hi = max(int(max_tokens) * int(exp_table.max()), lo + 1)
    return lo, hi


def _segment_threshold(params: PotIndexParams) -> int:
    """Smallest accumulator whose segment-one cursor exceeds ``bound`` (the real pivot).

    The golden branches on the *unclamped* segment-one cursor, so the boundary the
    hardware actually uses is set by the PoT index of segment one — not by the pivot
    that was requested. Segment two is calibrated from exactly this value so no
    accumulator ever lands on a clamped low entry of the second table.
    """
    return int(((params.bound + 1) << params.shift) - params.offset)


def _reciprocal_segment(params: PotIndexParams, *, qmax: int, recip_bits: int) -> tuple[np.ndarray, int, int]:
    """``(table, b3, s3)`` for one reciprocal segment.

    ``table[i] = round((qmax << s3) / acc_i)`` with ``s3`` the largest shift whose
    biggest entry (at the segment's smallest accumulator) still fits ``recip_bits``.
    Larger ``s3`` == finer reciprocal, so this spends the whole table width: the top
    entry always lands in ``[2^(recip_bits-1), 2^recip_bits - 1]``. The clip is not
    decoration — ``s3`` bottoms out at 1, and for a degenerate configuration such as a
    tiny ``exp_scale`` with a narrow ``recip_bits`` the unclipped top entry does overflow.
    """
    reps = _bin_geometric_means(params)
    recip_max = (1 << recip_bits) - 1
    s3 = int(math.floor(math.log2(max(recip_max * float(reps[0]) / max(qmax, 1), 2.0))))
    s3 = max(1, min(s3, _S3_MAX))
    numer = float(qmax << s3)
    table = np.clip(np.rint(numer / reps), 0, recip_max).astype(np.int64)
    return table, 1 << (s3 - 1), s3


def build_softmax_int_payload(
    logits,
    *,
    input_scale: float | None = None,
    input_dtype: QuantDtype = INT8,
    exp_entries: int = 256,
    recip_entries: int = 256,
    exp_scale: int = 32768,
    recip_bits: int = 16,
    output_bits: int = 8,
    max_tokens: int | None = None,
    pivot: str = "geometric",
) -> dict[str, object]:
    """Calibrate the integer softmax from observed pre-softmax logits.

    ``logits`` is anything shaped ``[..., tokens]``; softmax runs over the last axis.
    ``max_tokens`` is the longest row the deployed op will ever see and defaults to the
    calibration row length; it sizes the reciprocal segments (see
    ``_theoretical_accumulator_range``) and is the *only* thing standing between an
    unusually flat runtime row and an un-normalised output, so raise it if inference can
    run on longer sequences than calibration did.

    Returns the payload ``ISoftmax`` / ``i_ops.softmax_quantize`` consume:
    ``{scalars(14), exp_table, recip_table_one, recip_table_two, input_scale,
    output_scale, output_bits, metrics}``.
    """
    if exp_scale < 2:
        raise ValueError("exp_scale must be at least 2")
    if output_bits <= 0:
        raise ValueError("output_bits must be positive")
    if recip_bits <= 0:
        raise ValueError("recip_bits must be positive")

    rows = _as_rows(logits)
    if max_tokens is None:
        max_tokens = int(rows.shape[1])
    max_tokens = int(max_tokens)
    if max_tokens < rows.shape[1]:
        raise ValueError(
            f"max_tokens={max_tokens} is below the calibration row length {rows.shape[1]}"
        )

    if input_scale is None:
        input_scale = max(float(np.abs(rows).max()), 1e-8) / float(input_dtype.qmax)
    input_scale = float(input_scale)
    if input_scale <= 0.0:
        raise ValueError("input_scale must be positive")

    x_int = np.clip(np.rint(rows / input_scale), input_dtype.qmin, input_dtype.qmax).astype(np.int64)
    delta = x_int.max(axis=1, keepdims=True) - x_int  # >= 0, the inverse-exp argument

    # --- inverse exp table over (max - x) ------------------------------------
    exp_params = make_pot_index_params(0, int(delta.max()), entries=exp_entries)
    exp_table = np.clip(
        np.rint(np.exp(-np.maximum(_bin_midpoints(exp_params), 0.0) * input_scale) * exp_scale),
        0,
        exp_scale,
    ).astype(np.int64)

    # --- the segmented reciprocal spans the *theoretical* accumulator range ---
    acc_lo, acc_hi = _theoretical_accumulator_range(exp_table, max_tokens)
    if pivot == "geometric":
        split = int(round(math.sqrt(float(acc_lo) * float(acc_hi))))
    elif pivot == "reference":  # HG-PIPE calibrate_softmax
        split = acc_lo + max(1, int(math.ceil((acc_hi - acc_lo) / 8.0)))
    else:
        raise ValueError(f"unknown pivot strategy: {pivot!r}")
    split = min(max(split, acc_lo + 1), acc_hi - 1) if acc_hi - acc_lo > 1 else acc_lo + 1

    one_params = make_pot_index_params(acc_lo, split, entries=recip_entries)
    two_lo = _segment_threshold(one_params)
    two_params = make_pot_index_params(two_lo, max(acc_hi, two_lo + 1), entries=recip_entries)

    qmax = qrange(output_bits, signed=False)[1]
    recip_one, b3_one, s3_one = _reciprocal_segment(one_params, qmax=qmax, recip_bits=recip_bits)
    recip_two, b3_two, s3_two = _reciprocal_segment(two_params, qmax=qmax, recip_bits=recip_bits)

    acc = exp_table[cursor_for(delta, exp_params)].sum(axis=1)  # observed, for metrics only
    scalars = [
        *exp_params.scalars,          # b1, s1, bound1
        *one_params.scalars,          # b2_one, s2_one, bound2_one
        b3_one, s3_one,
        *two_params.scalars,          # b2_two, s2_two, bound2_two
        b3_two, s3_two,
        int(output_bits),
    ]
    return {
        "scalars": scalars,
        "exp_table": exp_table.tolist(),
        "recip_table_one": recip_one.tolist(),
        "recip_table_two": recip_two.tolist(),
        "input_scale": input_scale,
        "output_scale": 1.0 / float(qmax),
        "output_bits": int(output_bits),
        "metrics": {
            "rows": int(rows.shape[0]),
            "tokens": int(rows.shape[1]),
            "max_tokens": max_tokens,
            "delta_max": int(delta.max()),
            "acc_min": int(acc.min()),
            "acc_max": int(acc.max()),
            "acc_range_min": acc_lo,      # theoretical, what the segments actually cover
            "acc_range_max": acc_hi,
            "acc_pivot": int(split),
            "segment_threshold": int(two_lo),
            "rows_segment_two": int(np.sum(acc >= two_lo)),
            "exp_scale": int(exp_scale),
        },
    }


def build_softmax_int_payload_from_qsoftmax(qsoftmax, logits, **kwargs) -> dict[str, object]:
    """Recalibrate the integer softmax at a Q-tier ``QSoftmax``'s input scale.

    ``QSoftmax`` already fixes the integer grid of the exp argument (``exp_in``); its
    own uint8 exp/reciprocal tables are far too coarse to accumulate over a whole row,
    so the tables are rebuilt here at ``exp_scale`` while the input grid is kept so the
    Q-tier and I-tier see the same integer logits.
    """
    kwargs.setdefault("input_scale", float(qsoftmax.exp_in))
    return build_softmax_int_payload(logits, **kwargs)


__all__ = [
    "build_softmax_int_payload",
    "build_softmax_int_payload_from_qsoftmax",
]
