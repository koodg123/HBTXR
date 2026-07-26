"""Integer-scalar calibration for the fully-integer I-tier LayerNorm.

The Q-tier ``QLayerNorm`` still reduces mean/variance in float and only *looks up*
the rsqrt in an integer table. Hardware cannot do that: it has an integer datapath
and a handful of pre-computed constants. ``i_ops.layernorm_quantize`` is that
datapath — integer mean by reciprocal-multiply-and-shift, integer variance
accumulation, PoT-indexed rsqrt table, integer affine, arithmetic right shift,
clamp — and it is driven by exactly seven scalars plus three integer vectors
(``rsqrt_table``, ``lnw``, ``lnb``).

This module derives those constants from a calibrated LayerNorm (``nn.LayerNorm``
or ``QLayerNorm``) plus observed activations, so ``ilayers.layernorm.ILayerNorm``
needs no float reduction at inference time. The reciprocal comes from
``i_ops.dyadic_params`` and the requant shift ``s2`` falls out of the scale algebra
spelled out in ``_derive_shift``.

Why the rsqrt index is *fitted* and not derived from the observed envelope
-------------------------------------------------------------------------
The golden addresses the rsqrt table with one linear PoT index and nothing else::

    cursor = clamp((var_sum + b) >> s1, 0, bound)

``rsqrt`` is a log-domain function, so a *linear* index is a bad shape for it: the
relative resolution of a bin ``[v, v + 2^s1)`` is ``2^s1 / v``, which degrades
without limit as ``v -> 0``. Spanning the full observed ``[min, max]`` envelope —
the obvious choice, and what this module used to do — therefore collapses every
low-variance row into bin 0 whenever the observed variance is heteroscedastic or
heavy-tailed, which is exactly what real ViT token activations are. On the wide-spread
fixtures in ``tests/quantization/test_int_layernorm.py`` (32:1 log-uniform row scale,
~2000:1 in row variance) the envelope index makes a 14.4-16.0% RMS relative rsqrt
error where the fitted index below makes 8.9-9.4%; on a log-normal spread it reached
35% end-to-end. The toy homoscedastic case stays at ~2% either way, so the failure is
invisible unless it is calibrated for.

The golden index cannot be reshaped without changing the hardware kernel, so
``_fit_variance_index`` instead *fits* ``(b, s1)`` to the observed variance
DISTRIBUTION: it scores a grid of (lower edge, shift) candidates by the RMS
relative rsqrt error the deployed kernel would actually make on the observed rows
and keeps the best. Deliberately dropping the extreme tails (they clamp to entry 0
or ``bound``) buys a much finer grid for the bulk, which is the right trade because
the per-row output error of a LayerNorm is exactly the per-row relative rsqrt error
and every row carries about the same output energy.

PRECONDITION (unavoidable, please read before trusting the block). One linear index
of ``entries`` bins can only resolve a variance dynamic range of roughly ``entries``
before either the low end quantizes coarsely or the high end clamps. Concretely, a
bin ``[v, v + 2^s1)`` costs about ``2^s1 / (4 * v)`` of relative rsqrt error, so an
observed ``var_sum`` spread wider than ~``entries``:1 *cannot* be served to a few
percent no matter how ``(b, s1)`` are chosen. The returned ``metrics`` dict
reports what was actually achieved (``rsqrt_rel_rms``, ``rsqrt_rel_p99``,
``rows_below_range``, ``rows_above_range``, ``var_sum_dynamic_range``); raise
``entries`` (the table is ``entries`` int16 words) when ``rsqrt_rel_rms`` is too
large for the accuracy budget.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from quantization.i_ops import dyadic_params
from quantization.lut_calibrate import PotIndexParams, coordinates_for
from quantization.scheme import INT8, QuantDtype, qrange

# rsqrt table entries are int16: wide enough that a 2^-12-ish output scale keeps
# ~4 decimal digits of the reciprocal square root, narrow enough for a BRAM LUT.
RSQRT_DTYPE = QuantDtype(16, signed=True)

# Quantile levels tried as the fitted index's lower edge, spaced roughly
# logarithmically so the useful region (dropping a few per-mille of the left tail) is
# sampled densely while the coarse choices stay reachable.
_LOWER_EDGE_QUANTILES: tuple[float, ...] = (
    0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08,
    0.12, 0.18, 0.25, 0.35, 0.5,
)
# Cap on the rows the index search scores; the fit is a 2-parameter grid search and
# a few thousand rows already pin the distribution.
_SEARCH_ROWS = 8192


def _symmetric_scale(max_abs: float, dtype: QuantDtype) -> float:
    return max(float(max_abs), 1e-8) / float(dtype.qmax)


def _stack_samples(samples: Any, channels: int | None = None) -> torch.Tensor:
    """Observed activations -> a float ``[rows, C]`` matrix (rows = normalized rows)."""
    if torch.is_tensor(samples):
        flat = samples.detach().float()
    elif isinstance(samples, np.ndarray):
        flat = torch.as_tensor(samples, dtype=torch.float32)
    else:
        parts = [torch.as_tensor(s, dtype=torch.float32).reshape(-1, torch.as_tensor(s).shape[-1])
                 for s in samples]
        if not parts:
            raise ValueError("calibration samples must not be empty")
        flat = torch.cat(parts, dim=0)
    flat = flat.reshape(-1, flat.shape[-1])
    if flat.numel() == 0:
        raise ValueError("calibration samples must not be empty")
    if channels is not None and flat.shape[-1] != channels:
        raise ValueError(f"samples have {flat.shape[-1]} channels, LayerNorm expects {channels}")
    return flat


def _affine_of(module: nn.Module) -> tuple[torch.Tensor | None, torch.Tensor | None, float, int | None]:
    """(weight, bias, eps, channels) from an nn.LayerNorm / QLayerNorm-like module."""
    shape = getattr(module, "normalized_shape", None)
    channels: int | None = None
    if shape is not None:
        shape = (int(shape),) if isinstance(shape, int) else tuple(int(v) for v in shape)
        if len(shape) != 1:
            raise NotImplementedError(
                f"ILayerNorm normalizes a single trailing dim, got normalized_shape={shape}")
        channels = shape[0]
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    eps = float(getattr(module, "eps", 1e-5))
    return weight, bias, eps, channels


def integer_row_statistics(x_int: torch.Tensor, c_1_m: int, c_1_s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Golden-identical integer ``(mean, var_sum)`` per row of ``x_int`` (last dim = C).

    Shared by the calibrator and ``ILayerNorm`` so the variance distribution the rsqrt
    index is fitted to is *exactly* the distribution the deployed kernel indexes with.
    """
    if c_1_s < 1:
        raise ValueError("c_1_s must be >= 1 (the golden adds a 1 << (c_1_s - 1) rounding term)")
    acc = x_int.sum(dim=-1, keepdim=True)
    mean = torch.bitwise_right_shift(acc * c_1_m + (1 << (c_1_s - 1)), c_1_s)
    diff = x_int - mean
    var_sum = (diff * diff).sum(dim=-1, keepdim=True)
    return mean, var_sum


def rsqrt_table_values(params: PotIndexParams, unit: float, eps: float) -> np.ndarray:
    """Real-valued rsqrt of every table entry of ``params`` (``var_real = var_sum * unit``).

    Bin ``i`` catches the integer ``var_sum`` interval ``[c_i, c_i + 2^s1 - 1]``, and the
    kernel returns one value for all of them. The entry that MINIMAXES the relative error
    over that interval satisfies ``sqrt((a+eps)/r) - 1 == 1 - sqrt((b+eps)/r)``, i.e.

        rsqrt_i = 2 / (sqrt(a + eps) + sqrt(b + eps))

    with ``a``/``b`` the real variances at the bin's two ends. (The arithmetic bin
    midpoint is an equally reasonable interior point and measures within a percent of
    this; the claim here is only that an INTERIOR point is needed.) Sampling the lower
    edge instead — ``1/sqrt(a + eps)``, the naive choice for a flooring index — returns
    the largest rsqrt in the bin for every row in it and so biases the whole layer high:
    +2.9% to +3.7% mean relative rsqrt error on the wide-spread test fixtures, against
    -0.5% to +0.3% here.

    ``eps`` is load-bearing, not cosmetic: when the fitted index resolves one integer of
    variance per bin, bin 0 is exactly ``var_sum == 0`` and ``1/sqrt(0)`` is infinite.
    It also caps the table at ``1/sqrt(eps)``, the same ceiling ``F.layer_norm`` has.
    """
    lo = np.maximum(coordinates_for(params).astype(np.float64), 0.0) * float(unit)
    hi = lo + float((1 << params.shift) - 1) * float(unit)
    return 2.0 / (np.sqrt(lo + eps) + np.sqrt(hi + eps))


def _index_error(params: PotIndexParams, var_sum: np.ndarray, unit: float, eps: float) -> np.ndarray:
    """Per-row signed relative rsqrt error of the deployed kernel under ``params``."""
    table = rsqrt_table_values(params, unit, eps)
    cursor = np.clip((var_sum + params.offset) >> params.shift, 0, params.bound)
    truth = 1.0 / np.sqrt(var_sum.astype(np.float64) * float(unit) + eps)
    return table[cursor] / truth - 1.0


def _fit_variance_index(
    var_sum: np.ndarray,
    unit: float,
    eps: float,
    entries: int,
) -> tuple[PotIndexParams, dict[str, Any]]:
    """Fit ``(offset, shift)`` to the observed ``var_sum`` distribution (see module doc).

    Grid-searches the lower edge over quantiles of the observation and the shift over
    every shift that can matter, scoring each candidate by the RMS relative rsqrt error
    the golden kernel would make on the observed rows. Returns the winner plus the
    metrics that document how well the (necessarily linear) index fits.
    """
    if entries < 2:
        raise ValueError("entries must be at least 2")
    bound = entries - 1
    probe = var_sum if var_sum.size <= _SEARCH_ROWS else var_sum[:: max(1, var_sum.size // _SEARCH_ROWS)]
    span = int(var_sum.max()) - int(var_sum.min())
    max_shift = max(1, int(math.ceil(math.log2(max(span, 2)))) + 1)
    lower_edges = sorted({max(int(math.floor(np.quantile(var_sum, q))), 0) for q in _LOWER_EDGE_QUANTILES})

    best: tuple[float, PotIndexParams] | None = None
    for lower in lower_edges:
        for shift in range(0, max_shift + 1):
            candidate = PotIndexParams(offset=-lower, shift=shift, bound=bound)
            rms = float(np.sqrt(np.mean(_index_error(candidate, probe, unit, eps) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, candidate)
    assert best is not None
    params = best[1]

    err = np.abs(_index_error(params, var_sum, unit, eps))
    lo_edge = -params.offset
    hi_edge = lo_edge + ((bound + 1) << params.shift) - 1
    lo_obs, hi_obs = int(var_sum.min()), int(var_sum.max())
    metrics: dict[str, Any] = {
        "rsqrt_rel_rms": float(np.sqrt(np.mean(err ** 2))),
        "rsqrt_rel_p99": float(np.quantile(err, 0.99)),
        "rsqrt_rel_max": float(err.max()),
        "rows_below_range": int(np.count_nonzero(var_sum < lo_edge)),
        "rows_above_range": int(np.count_nonzero(var_sum > hi_edge)),
        "rows": int(var_sum.size),
        "var_sum_min": lo_obs,
        "var_sum_max": hi_obs,
        "var_sum_dynamic_range": float(max(hi_obs, 1) / max(lo_obs, 1)),
        "index_lo": int(lo_edge),
        "index_hi": int(hi_edge),
    }
    return params, metrics


def _derive_shift(unit: float, target_output_scale: float, peak_output: float, clamp_bits: int) -> tuple[int, float]:
    """The final requant shift ``s2`` and the output scale it implies.

    The integer datapath computes, per element,

        affine  = (x_int - mean_int) * rsqrt_int * lnw_int + lnb_int
        out_int = affine >> s2

    and the three integer factors carry the calibrated scales

        (x - mean)_real = (x_int - mean_int) * input_scale
        rsqrt_real      = rsqrt_int          * rsqrt_scale
        lnw_real        = lnw_int            * lnw_scale

    so their product sits at ``unit = input_scale * rsqrt_scale * lnw_scale`` while
    the requantized output represents ``out_int * output_scale``. Requiring the two
    to describe the same real number,

        affine * unit == (affine >> s2) * output_scale   =>   output_scale = unit * 2^s2
                                                        <=>  s2 = log2(output_scale / unit)

    ``s2`` must be a non-negative integer, so we round the ideal
    ``log2(target_output_scale / unit)`` and then *define* ``output_scale = unit *
    2^s2``. The algebra then closes exactly (no residual scale error); the price is
    that the realized output scale lands within a factor of 2 of the target. A
    saturation guard raises ``s2`` (coarser output LSB) until the observed peak fits
    inside the clamp, so the clamp is never hit systematically.
    """
    if unit <= 0.0:
        raise ValueError("unit scale must be positive")
    s2 = max(0, int(round(math.log2(max(target_output_scale, 1e-30) / unit))))
    output_scale = unit * float(2 ** s2)
    qmax = qrange(clamp_bits, signed=True)[1]
    while peak_output / output_scale > qmax and s2 < 62:
        s2 += 1
        output_scale *= 2.0
    return s2, output_scale


def calibrate_int_layernorm(
    module: nn.Module,
    samples: Any,
    *,
    input_scale: float | None = None,
    entries: int = 256,
    clamp_bits: int = 8,
    input_dtype: QuantDtype = INT8,
    lnw_dtype: QuantDtype = INT8,
    rsqrt_dtype: QuantDtype = RSQRT_DTYPE,
) -> dict[str, Any]:
    """Derive the 7 ``layernorm_quantize`` scalars + rsqrt table + integer affine.

    ``module`` is an ``nn.LayerNorm`` or a ``QLayerNorm`` (anything exposing
    ``weight`` / ``bias`` / ``eps`` / ``normalized_shape``); ``samples`` are observed
    float activations whose last dim is the normalized channel dim.

    Returns the payload ``ILayerNorm.from_payload`` consumes::

        scalars = [c_1_m, c_1_s, b, s1, bound, s2, clamp_bits]
        rsqrt_table, lnw, lnb, input_scale, output_scale, rsqrt_scale, lnw_scale, ...

    plus a ``metrics`` dict describing how well the (linear, PoT) variance index fits
    the observed variance distribution — see the module docstring's PRECONDITION.
    """
    weight, bias, eps, declared = _affine_of(module)
    x = _stack_samples(samples, declared)
    channels = int(x.shape[-1])
    w_real = torch.ones(channels) if weight is None else weight.detach().float().reshape(-1)
    b_real = torch.zeros(channels) if bias is None else bias.detach().float().reshape(-1)

    # --- (1) input scale + the integer input the kernel actually sees ---------
    if input_scale is None:
        input_scale = _symmetric_scale(float(x.abs().max()), input_dtype)
    input_scale = float(input_scale)
    x_int = torch.round(x / input_scale).clamp(input_dtype.qmin, input_dtype.qmax).to(torch.int64)

    # --- (2) integer mean: c_1_m / 2^c_1_s ~= 1/C (dyadic, same search as requant)
    c_1_m, c_1_s, _ = dyadic_params(1.0 / channels)
    _, var_sum_t = integer_row_statistics(x_int, c_1_m, c_1_s)
    var_sum = var_sum_t.reshape(-1).numpy().astype(np.int64)

    # --- (3) PoT index FITTED to the observed variance distribution -----------
    # var_sum is an integer in units of input_scale^2 and equals C * var_real /
    # input_scale^2, so one var_sum step is `var_unit` of real variance.
    var_unit = (input_scale * input_scale) / float(channels)
    params, index_metrics = _fit_variance_index(var_sum, var_unit, eps, entries)
    rsqrt_real = rsqrt_table_values(params, var_unit, eps)

    # --- (4) rsqrt table: power-of-two output scale, int16 entries ------------
    exponent = int(math.floor(math.log2(float(rsqrt_dtype.qmax) / float(rsqrt_real.max()))))
    rsqrt_scale = float(2.0 ** (-exponent))                # POT => HW just re-labels bits
    rsqrt_table = np.clip(np.rint(rsqrt_real / rsqrt_scale),
                          rsqrt_dtype.qmin, rsqrt_dtype.qmax).astype(np.int64)

    # --- (5) integer LayerNorm affine weight ---------------------------------
    lnw_scale = _symmetric_scale(float(w_real.abs().max()), lnw_dtype)
    lnw = torch.round(w_real / lnw_scale).clamp(lnw_dtype.qmin, lnw_dtype.qmax).to(torch.int64)

    # --- (6) output scale + requant shift (see _derive_shift for the algebra) --
    with torch.no_grad():
        y_ref = F.layer_norm(x, (channels,), w_real, b_real, eps)
    peak = max(float(y_ref.abs().max()), 1e-8)
    unit = input_scale * rsqrt_scale * lnw_scale
    s2, output_scale = _derive_shift(unit, peak / float(qrange(clamp_bits, signed=True)[1]), peak, clamp_bits)

    # --- (7) integer bias, with the rounding constant folded in ---------------
    # lnb is added *before* the shift, so it lives at `unit`: lnb_int * unit == bias.
    # The golden's ``>> s2`` is an arithmetic (floor) shift, which alone costs a
    # systematic -0.5 LSB. Folding 2^(s2-1) into the bias turns it into round-half-up
    # for free — the standard HW trick, and it keeps the golden kernel untouched.
    round_term = (1 << (s2 - 1)) if s2 > 0 else 0
    lnb = torch.round(b_real.double() / unit).to(torch.int64) + round_term

    return {
        "scalars": [int(c_1_m), int(c_1_s), int(params.offset), int(params.shift),
                    int(params.bound), int(s2), int(clamp_bits)],
        "rsqrt_table": rsqrt_table.tolist(),
        "lnw": lnw.tolist(),
        "lnb": lnb.tolist(),
        "input_scale": input_scale,
        "output_scale": float(output_scale),
        "rsqrt_scale": rsqrt_scale,
        "lnw_scale": float(lnw_scale),
        "unit_scale": float(unit),
        "channels": channels,
        "eps": eps,
        "clamp_bits": int(clamp_bits),
        "input_dtype": input_dtype,
        "var_sum_range": (index_metrics["var_sum_min"], index_metrics["var_sum_max"]),
        "metrics": index_metrics,
    }


__all__ = [
    "RSQRT_DTYPE",
    "integer_row_statistics",
    "rsqrt_table_values",
    "calibrate_int_layernorm",
]
