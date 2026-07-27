"""I-tier fully-integer LayerNorm (C3a): ILayerNorm bit-exact vs the i_ops golden.

Two rules this file follows, because the previous version broke both:

* Nothing is measured against a tolerance the calibrator itself chose. Accuracy is
  normalized by a REFERENCE LSB derived from the float LayerNorm's own output range
  and the declared output bit width, so a calibrator that picks a coarser
  ``output_scale`` scores WORSE, not better.
* Every design claim the calibrator makes in prose has a test that dies when the
  claim is removed (the folded round-half-up constant, the interior rsqrt sampling,
  the distribution-fitted variance index, the cursor's lower clamp, the input-port
  re-clamp).

Since D3 the calibrator's DEFAULT index has two segments, so this file covers both
shapes deliberately rather than by accident:

* the helpers (``_golden``, ``_deployed_rsqrt_error``) dispatch on the scalar count, so
  every test that used to run against the 7-scalar golden now runs against whichever
  golden the payload names — a payload can no longer be checked against the wrong one.
* the tests whose CLAIM is about the one-segment fit (the fitted-vs-envelope index and
  the bin-interior sampling) pass ``segments=1`` explicitly. That is not a workaround:
  the envelope baseline they compare against is a one-segment construction, so the
  comparison is only meaningful there — and it keeps ``i_ops.layernorm_quantize``, the
  golden every already-exported payload is checked against, exercised end to end.
  ``test_int_layernorm_segmented.py`` owns the two-segment claims.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn

from quantization.i_ops import dyadic_params, layernorm_quantize, layernorm_quantize_segmented
from quantization.ilayers.layernorm import ILayerNorm
from quantization.ilayers.qtensor import QTensor
from quantization.int_calibrate import calibrate_int_layernorm, integer_row_statistics
from quantization.lut_calibrate import build_function_lut, coordinates_for, make_pot_index_params
from quantization.qlayers.nonlinear import QLayerNorm
from quantization.scheme import INT8, UINT8, qrange


# --- deterministic fixtures (own generators, never the global RNG) ------------

def _layernorm(channels: int, seed: int) -> nn.LayerNorm:
    gen = torch.Generator().manual_seed(1000 + seed)
    ln = nn.LayerNorm(channels)
    with torch.no_grad():
        ln.weight.copy_(1.0 + 0.15 * torch.randn(channels, generator=gen))
        ln.bias.copy_(0.1 * torch.randn(channels, generator=gen))
    return ln


def _homoscedastic(channels: int, spread: float, seed: int, rows: int = 320) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, channels, generator=gen) * spread + 0.3


HETERO_RATIO = 32.0  # row-scale spread => a ~1000:1 spread in row variance


def _heteroscedastic(channels: int, seed: int, rows: int = 512,
                     ratio: float = HETERO_RATIO) -> torch.Tensor:
    """Rows whose scale sweeps ``ratio``:1 log-uniformly — a wide row-variance spread.

    This is the shape real ViT token activations have and the shape that breaks a
    variance index built from the observed [min, max] envelope (see LN-1). The sweep is
    deterministic (``logspace``, not a random draw) so every seed poses the *same*
    difficulty and the bounds below mean the same thing for each.
    """
    gen = torch.Generator().manual_seed(seed)
    row_scale = torch.logspace(0.0, math.log10(ratio), rows).reshape(rows, 1) * (1.5 / ratio)
    return torch.randn(rows, channels, generator=gen) * row_scale


def _envelope_index_rsqrt_error(x: torch.Tensor, payload: dict, entries: int = 256) -> np.ndarray:
    """Relative rsqrt error of the index this module USED to build, rebuilt locally.

    ``[observed min, observed max]`` spanned by ``entries`` linear PoT bins, each entry
    sampled at its bin midpoint. Recomputed here (not imported) so the comparison in
    ``test_heavy_tailed_variance_is_resolved_by_the_fitted_index`` is against a fixed
    external baseline rather than anything the calibrator still does.
    """
    var_sum = _row_var_sum(x, payload)
    unit = payload["input_scale"] ** 2 / payload["channels"]
    params = make_pot_index_params(max(int(var_sum.min()), 0), max(int(var_sum.max()), 1),
                                   entries=entries)
    coords = np.maximum(coordinates_for(params).astype(np.float64), 0.0)
    half_bin = (1 << (params.shift - 1)) if params.shift > 0 else 0
    table = 1.0 / np.sqrt((coords + half_bin) * unit + payload["eps"])
    cursor = np.clip((var_sum + params.offset) >> params.shift, 0, params.bound)
    return table[cursor] / (1.0 / np.sqrt(var_sum * unit + payload["eps"])) - 1.0


def _reference_lsb(ref: torch.Tensor, clamp_bits: int = 8) -> float:
    """The LSB an ideal ``clamp_bits`` symmetric quantizer of ``ref`` would have.

    Externally defined: it depends only on the FLOAT LayerNorm's output range and the
    declared output width, never on the scale the calibrator happened to choose.
    """
    return float(ref.abs().max()) / float(qrange(clamp_bits, signed=True)[1])


def _row_var_sum(x: torch.Tensor, payload: dict) -> np.ndarray:
    """The integer per-row ``var_sum`` the deployed kernel sees, re-derived here.

    Deliberately a local transcription of the golden's mean/variance reduction rather
    than a call into ``int_calibrate``, so the rsqrt assertions below do not lean on
    the module they are judging.
    """
    channels = payload["channels"]
    c_1_m, c_1_s = payload["scalars"][0], payload["scalars"][1]
    x_int = torch.round(x.reshape(-1, channels) / payload["input_scale"])
    x_int = x_int.clamp(INT8.qmin, INT8.qmax).to(torch.int64)
    mean = (x_int.sum(-1, keepdim=True) * c_1_m + (1 << (c_1_s - 1))) >> c_1_s
    diff = x_int - mean
    return (diff * diff).sum(-1).numpy().astype(np.int64)


def _deployed_rsqrt_error(x: torch.Tensor, payload: dict) -> np.ndarray:
    """Signed relative error of the rsqrt the kernel actually looks up, per row.

    Dispatches on the scalar count, so a segmented payload is judged by the segmented
    lookup — branch on the UNCLAMPED segment-one cursor included. Scoring a 10-scalar
    payload with the one-segment lookup would read the wrong table for exactly the rows
    the second segment exists to serve, and the error it reported would be fiction.
    """
    var_sum = _row_var_sum(x, payload)
    _, _, b, s1, bound = payload["scalars"][:5]
    table = np.asarray(payload["rsqrt_table"], dtype=np.float64)
    cursor = (var_sum + b) >> s1
    if len(payload["scalars"]) == 7:
        used = table[np.clip(cursor, 0, bound)]
    else:
        b_two, s1_two, bound_two = payload["scalars"][7:]
        table_two = np.asarray(payload["rsqrt_table_two"], dtype=np.float64)
        used = np.where(cursor > bound,
                        table_two[np.clip((var_sum + b_two) >> s1_two, 0, bound_two)],
                        table[np.clip(cursor, 0, bound)])
    used = used * payload["rsqrt_scale"]
    var_real = var_sum.astype(np.float64) * payload["input_scale"] ** 2 / payload["channels"]
    return used / (1.0 / np.sqrt(var_real + payload["eps"])) - 1.0


def _synthetic(
    channels: int,
    rows: tuple[int, ...],
    *,
    seed: int,
    b: int,
    s1: int,
    s2: int,
    clamp_bits: int,
    low: int,
    high: int,
    entries: int = 64,
) -> tuple[list[int], list[int], list[int], list[int], torch.Tensor]:
    """A random-but-valid scalar/table/affine set + random integer inputs.

    ``(b, s1, bound)`` are chosen per case so the variance cursor lands inside the
    table instead of pinning to ``bound`` for every row. ``b`` is negative in the
    cases that must exercise the cursor's LOWER clamp — real calibrated payloads have
    a negative PoT offset, so a suite that only ever passes ``b = 0`` never touches it.
    """
    gen = torch.Generator().manual_seed(seed)
    c_1_m, c_1_s, _ = dyadic_params(1.0 / channels)
    x_int = torch.randint(low, high, tuple(rows) + (channels,), generator=gen, dtype=torch.int32)
    rsqrt_table = torch.randint(64, 8192, (entries,), generator=gen).tolist()
    lnw = torch.randint(-127, 128, (channels,), generator=gen).tolist()
    lnb = torch.randint(-200000, 200000, (channels,), generator=gen).tolist()
    scalars = [c_1_m, c_1_s, b, s1, entries - 1, s2, clamp_bits]
    return scalars, rsqrt_table, lnw, lnb, x_int


def _golden(x_int: torch.Tensor, scalars: list[int], lnw: list[int], lnb: list[int],
            rsqrt_table: list[int], rsqrt_table_two: list[int] | None = None) -> torch.Tensor:
    """The golden the SCALAR COUNT names — 7 -> ``layernorm_quantize``, 10 -> segmented."""
    flat = x_int.reshape(-1).tolist()
    if len(scalars) == 7:
        assert rsqrt_table_two is None, "a 7-scalar payload has no second segment"
        out = layernorm_quantize(flat, scalars, lnw, lnb, rsqrt_table)
    else:
        out = layernorm_quantize_segmented(flat, scalars, lnw, lnb, rsqrt_table, rsqrt_table_two)
    return torch.tensor(out, dtype=torch.int32).reshape(x_int.shape)


def _golden_for(x_int: torch.Tensor, payload: dict) -> torch.Tensor:
    """``_golden`` driven straight from a calibrated payload, whichever shape it is."""
    return _golden(x_int, payload["scalars"], payload["lnw"], payload["lnb"],
                   payload["rsqrt_table"], payload.get("rsqrt_table_two"))


def _raw_cursor(x_int: torch.Tensor, scalars: list[int]) -> torch.Tensor:
    """``(var_sum + b) >> s1`` BEFORE the clamp — what the clamp has to police."""
    c_1_m, c_1_s, b, s1 = scalars[:4]
    mean, var_sum = integer_row_statistics(x_int.to(torch.int64), c_1_m, c_1_s)
    return torch.bitwise_right_shift(var_sum + b, s1).reshape(-1)


# --- (a) BIT-EXACTNESS vs i_ops.layernorm_quantize ---------------------------

# name, channels, rows, seed, b, s1, s2, clamp_bits, low, high,
# saturated fraction (lo, hi), min distinct cursors, min rows below the table
BIT_EXACT_CASES = [
    ("mixed-sign", 8, (3, 5), 101, 0, 10, 18, 8, -128, 128, (0.05, 0.25), 10, 0),
    ("all-negative-inputs", 8, (4, 3), 102, 0, 10, 18, 8, -128, 0, (0.02, 0.15), 5, 0),
    ("heavy-saturation", 8, (3, 5), 103, 0, 10, 8, 8, -128, 128, (0.95, 1.0), 10, 0),
    ("int4-output", 12, (2, 7), 104, 0, 11, 20, 4, -128, 128, (0.40, 0.70), 10, 0),
    ("wide-no-saturation", 48, (2, 4), 105, 0, 13, 20, 8, -128, 128, (0.0, 0.0), 5, 0),
    ("narrow-input-range", 6, (5,), 106, 0, 6, 14, 8, -10, 11, (0.15, 0.35), 2, 0),
    # negative PoT offset: rows whose variance sits BELOW the calibrated range make the
    # pre-clamp cursor negative, which is what a real payload does (LN-2).
    ("negative-offset-below-range", 8, (5, 4), 109, -6993, 6, 17, 8, -60, 61, (0.10, 0.35), 10, 5),
    ("negative-offset-clamped-high", 16, (4, 6), 107, -52331, 7, 18, 8, -128, 128, (0.20, 0.45), 4, 1),
]


@pytest.mark.parametrize(
    "name,channels,rows,seed,b,s1,s2,clamp_bits,low,high,sat_range,min_cursors,min_below",
    BIT_EXACT_CASES,
    ids=[c[0] for c in BIT_EXACT_CASES],
)
def test_ilayernorm_bit_exact_vs_golden(name, channels, rows, seed, b, s1, s2, clamp_bits,
                                        low, high, sat_range, min_cursors, min_below):
    scalars, rsqrt_table, lnw, lnb, x_int = _synthetic(
        channels, rows, seed=seed, b=b, s1=s1, s2=s2, clamp_bits=clamp_bits, low=low, high=high)
    iln = ILayerNorm(scalars, rsqrt_table, lnw, lnb, input_scale=1.0, output_scale=1.0)
    out = iln(QTensor(x_int, scale=1.0, zero_point=0.0, dtype=INT8))
    golden = _golden(x_int, scalars, lnw, lnb, rsqrt_table)

    assert torch.equal(out.int_data, golden), (
        f"{name}: {(out.int_data != golden).sum().item()} / {golden.numel()} elements differ")

    # the case must actually exercise the interesting paths, otherwise it proves nothing
    mean, _ = integer_row_statistics(x_int.to(torch.int64), scalars[0], scalars[1])
    assert bool((x_int.to(torch.int64) - mean < 0).any()), (
        f"{name}: no negative (value - mean), the arithmetic shift is not exercised")

    raw = _raw_cursor(x_int, scalars)
    distinct = int(torch.unique(raw.clamp(0, scalars[4])).numel())
    assert distinct >= min_cursors, (
        f"{name}: only {distinct} distinct table entries used, the LUT index is near-degenerate")
    below = int((raw < 0).sum())
    assert below >= min_below, f"{name}: {below} rows below the table, expected >= {min_below}"

    qmax = qrange(clamp_bits, signed=True)[1]
    saturated = float((golden.abs() >= qmax).float().mean())
    sat_lo, sat_hi = sat_range
    assert sat_lo <= saturated <= sat_hi, (
        f"{name}: {saturated:.3f} saturated, expected within [{sat_lo}, {sat_hi}]")


def test_cursor_lower_clamp_is_exercised_and_matches_the_golden():
    """The cursor's LOWER clamp is live on real payloads and is pinned here (LN-2).

    A calibrated payload carries a NEGATIVE PoT offset, so any row whose variance falls
    below the calibrated range drives ``(var_sum + b) >> s1`` negative. Clamping to
    anything other than 0 makes torch index the table from the far end (``table[-1]``
    is the *largest*-variance entry), so this case dies if the lower bound moves.
    """
    scalars, rsqrt_table, lnw, lnb, x_int = _synthetic(
        8, (5, 4), seed=109, b=-6993, s1=6, s2=17, clamp_bits=8, low=-60, high=61)
    raw = _raw_cursor(x_int, scalars)
    assert int((raw < 0).sum()) >= 5, "fixture no longer drives the cursor below zero"

    golden = _golden(x_int, scalars, lnw, lnb, rsqrt_table)
    iln = ILayerNorm(scalars, rsqrt_table, lnw, lnb, input_scale=1.0, output_scale=1.0)
    assert torch.equal(iln(QTensor(x_int, scale=1.0, zero_point=0.0, dtype=INT8)).int_data, golden)

    # prove the case discriminates: a kernel clamping to -1 instead of 0 would differ
    table = torch.tensor(rsqrt_table, dtype=torch.int64)
    mean, _ = integer_row_statistics(x_int.to(torch.int64), scalars[0], scalars[1])
    diff = x_int.to(torch.int64) - mean
    wrong = torch.bitwise_right_shift(
        diff * table[raw.clamp(-1, scalars[4])].reshape(mean.shape)
        * torch.tensor(lnw, dtype=torch.int64) + torch.tensor(lnb, dtype=torch.int64), scalars[5])
    qmin, qmax = qrange(scalars[6], signed=True)
    assert not torch.equal(wrong.clamp(qmin, qmax).to(torch.int32), golden), (
        "the fixture cannot tell a lower clamp of 0 from one of -1")


def test_ilayernorm_arithmetic_shift_floors_negatives():
    """Hand-derived two-channel case: the >> must FLOOR, not truncate toward zero.

    c = 2 -> dyadic 1/2 = 1 / 2^1, so for the row [0, 1]:
        mean    = (1*1 + 1) >> 1 = 1
        diff    = [-1, 0]
        var_sum = 1, s1 = 0, b = 0 -> cursor = 1, rsqrt_table[1] = 1
        affine  = [(-1)*1*1 + 0, 0*1*1 + 0] = [-1, 0]
        out     = [-1 >> 1, 0 >> 1] = [-1, 0]          (truncation would give [0, 0])
    """
    c_1_m, c_1_s, _ = dyadic_params(0.5)
    assert (c_1_m, c_1_s) == (1, 1)
    scalars = [c_1_m, c_1_s, 0, 0, 3, 1, 8]
    rsqrt_table = [1, 1, 1, 1]
    lnw, lnb = [1, 1], [0, 0]
    x_int = torch.tensor([[0, 1]], dtype=torch.int32)

    iln = ILayerNorm(scalars, rsqrt_table, lnw, lnb, input_scale=1.0, output_scale=1.0)
    out = iln(QTensor(x_int, scale=1.0, zero_point=0.0, dtype=INT8))
    assert out.int_data.tolist() == [[-1, 0]]
    assert torch.equal(out.int_data, _golden(x_int, scalars, lnw, lnb, rsqrt_table))


@pytest.mark.parametrize("segments", [1, 2])
def test_ilayernorm_bit_exact_after_calibration(segments):
    """The calibrated (not hand-made) constants also drive the golden bit-exactly.

    Both index shapes, because the DEFAULT is now the segmented one and a payload must
    be bit-exact against the golden its own scalar count names.
    """
    ln = _layernorm(48, 3)
    payload = calibrate_int_layernorm(ln, _homoscedastic(48, 1.5, 3, rows=200), segments=segments)
    iln = ILayerNorm.from_payload(payload)
    assert payload["scalars"][2] < 0, "a fitted index should carry a negative PoT offset"
    assert len(payload["scalars"]) == (7 if segments == 1 else 10)
    assert ("rsqrt_table_two" in payload) is (segments == 2)

    gen = torch.Generator().manual_seed(31)
    x_int = torch.randint(-128, 128, (3, 5, 48), generator=gen, dtype=torch.int32)
    out = iln(QTensor(x_int, scale=payload["input_scale"], zero_point=0.0, dtype=INT8))
    assert torch.equal(out.int_data, _golden_for(x_int, payload))


def test_calibration_defaults_to_the_two_segment_index():
    """The default is the segmented index — stated as a test, not only in a docstring."""
    payload = calibrate_int_layernorm(_layernorm(32, 3), _homoscedastic(32, 1.2, 3, rows=200))
    assert payload["segments"] == 2
    assert len(payload["scalars"]) == 10
    assert payload["metrics"]["segments"] == 2
    assert len(payload["rsqrt_table_two"]) == len(payload["rsqrt_table"]) == 256


def test_rescale_reclamps_to_the_input_port_width():
    """An upstream grid 4x coarser must NOT push 10-bit values into an 8-bit port (LN-3).

    ``ratio = 4`` turns an incoming 127 into 508. The block's input port is ``in_dtype``
    wide, so the bridge has to clamp exactly like ``QTensor.quantize`` does; without the
    clamp the kernel silently normalizes numbers the hardware could never have received.
    """
    ln = _layernorm(16, 6)
    payload = calibrate_int_layernorm(ln, _homoscedastic(16, 1.4, 6))
    iln = ILayerNorm.from_payload(payload)

    gen = torch.Generator().manual_seed(61)
    x_int = torch.randint(-128, 128, (6, 16), generator=gen, dtype=torch.int32)
    scaled = x_int.to(torch.int64) * 4
    assert int((scaled.abs() > INT8.qmax).sum()) > x_int.numel() // 2, "fixture does not overflow"

    out = iln(QTensor(x_int, scale=payload["input_scale"] * 4.0, zero_point=0.0, dtype=INT8))
    clamped = _golden_for(scaled.clamp(INT8.qmin, INT8.qmax), payload)
    assert torch.equal(out.int_data, clamped)
    assert not torch.equal(clamped, _golden_for(scaled, payload)), (
        "the fixture cannot tell a clamped bridge from an unclamped one")


# --- (b) ACCURACY vs the true float LayerNorm --------------------------------

ACCURACY_CASES = [(48, 1.5, 0), (16, 0.5, 2), (96, 3.0, 3), (8, 1.0, 4), (64, 1.0, 5)]


@pytest.mark.parametrize("channels,spread,seed", ACCURACY_CASES)
def test_ilayernorm_accuracy_vs_float_layernorm(channels, spread, seed):
    """Measured over these cases: rel-err 1.9-2.7%, MAE 0.41-0.64 ref-LSB, max <= 3.43 ref-LSB.

    The LSB is the EXTERNAL reference LSB (float output range / 127), not
    ``payload["output_scale"]``: normalizing by the calibrator's own choice would let a
    calibrator that picks a coarser output grid report a smaller error in LSB.
    """
    ln = _layernorm(channels, seed)
    x = _homoscedastic(channels, spread, seed)

    payload = calibrate_int_layernorm(ln, x)
    iln = ILayerNorm.from_payload(payload)
    with torch.no_grad():
        ref = ln(x)
        out = iln(x)  # float I/O drop-in path

    lsb = _reference_lsb(ref, payload["clamp_bits"])
    err = (out - ref).abs()
    assert float((out - ref).norm() / ref.norm()) < 0.035
    assert float(err.mean()) / lsb < 0.75, f"MAE {float(err.mean()) / lsb:.2f} ref-LSB"
    assert float(err.max()) / lsb < 4.0, f"max error {float(err.max()) / lsb:.2f} ref-LSB"
    # and the declared grid must not be wastefully coarser than an ideal 8-bit grid
    assert payload["output_scale"] / lsb < 2.0, f"output LSB is {payload['output_scale'] / lsb:.2f}x ideal"


HETERO_SEEDS = (0, 1, 2, 3)


@pytest.mark.parametrize("seed", HETERO_SEEDS)
def test_heavy_tailed_variance_is_resolved_by_the_fitted_index(seed):
    """The rsqrt index must survive a wide row-variance spread, not just a toy tensor (LN-1).

    Measured on these four fixtures (32:1 log-uniform row scale, var_sum dynamic range
    1.7e3-2.2e3):

        RMS relative rsqrt error   envelope index 0.1596 / 0.1514 / 0.1435 / 0.1589
                                   fitted index   0.0935 / 0.0887 / 0.0903 / 0.0941
        kernel-only relative error fitted index   0.1184 / 0.1115 / 0.1140 / 0.1191

    The [min, max]-envelope index this module used to build misses the absolute bound
    below on all four seeds; the fitted index clears it with ~1.2x to spare.

    ``segments=1`` is the point of the test, not a concession to it: the envelope
    baseline is a ONE-segment construction, so the only honest comparison is against a
    one-segment fit — and this keeps the 7-scalar ``layernorm_quantize`` golden driven by
    real calibrated constants. The two-segment index beats BOTH by another ~9x
    (``test_int_layernorm_segmented.py::test_two_segments_beat_one_on_the_same_data``).
    """
    channels = 48
    ln = _layernorm(channels, seed)
    x = _heteroscedastic(channels, seed)
    payload = calibrate_int_layernorm(ln, x, segments=1)

    # the fixture must really be wide-spread, else the assertions below are free
    assert payload["metrics"]["var_sum_dynamic_range"] > 500.0

    rms = float(np.sqrt(np.mean(_deployed_rsqrt_error(x, payload) ** 2)))
    envelope = float(np.sqrt(np.mean(_envelope_index_rsqrt_error(x, payload) ** 2)))
    assert rms < 0.115, f"rsqrt RMS relative error {rms:.4f}"
    assert rms < 0.75 * envelope, (
        f"fitted index {rms:.4f} is no better than the [min,max] envelope {envelope:.4f}")

    qt = QTensor.quantize(x, payload["input_scale"], 0.0, INT8)
    with torch.no_grad():
        ref = ln(qt.dequantize())          # float LN on the SAME quantized input
        out = ILayerNorm.from_payload(payload)(qt).dequantize()
    kernel_rel = float((out - ref).norm() / ref.norm())
    assert kernel_rel < 0.15, f"kernel-only relative error {kernel_rel:.4f}"


@pytest.mark.parametrize("seed", HETERO_SEEDS)
def test_rsqrt_table_samples_the_bin_interior_not_its_lower_edge(seed):
    """The rsqrt entries must be UNBIASED across the bin, not pinned to its left end.

    ``cursor`` floors, so bin ``i`` serves every variance in ``[c_i, c_i + 2^s1)``.
    Evaluating rsqrt at ``c_i`` (the naive choice for a flooring index) returns the
    LARGEST rsqrt in the bin for every row in it, biasing the whole layer's output high.
    Measured mean signed relative rsqrt error over these fixtures: interior sampling
    -0.005..+0.003, lower-edge sampling +0.029..+0.037.

    ``segments=1`` because the bias this detects scales with the bin width: the guard
    below demands bins coarse enough for a lower-edge bias to be visible at all, and the
    two-segment index (RMS ~0.01) is deliberately far too fine for that. The same claim
    at two segments is pinned by
    ``test_int_layernorm_segmented.py::test_segmented_tables_sample_the_bin_interior``,
    which uses 16 entries per segment to coarsen the bins on purpose.
    """
    channels = 48
    ln = _layernorm(channels, seed)
    x = _heteroscedastic(channels, seed)
    payload = calibrate_int_layernorm(ln, x, segments=1)

    err = _deployed_rsqrt_error(x, payload)
    assert float(np.sqrt(np.mean(err ** 2))) > 0.05, "bins too fine here to detect any bias"
    assert abs(float(np.mean(err))) < 0.015, f"mean signed rsqrt error {float(np.mean(err)):+.4f}"


@pytest.mark.parametrize("channels,spread,seed", ACCURACY_CASES)
def test_folded_round_half_up_removes_the_shift_bias(channels, spread, seed):
    """``lnb`` carries ``2^(s2-1)`` so the floor-shift does not cost -0.5 LSB (LN-6).

    The golden's ``>> s2`` floors, so without the folded constant every output element
    is biased down by half an output LSB. Measured mean signed error over these cases:
    with the constant -0.06..+0.01 ref-LSB, without it -0.97..-0.57 ref-LSB.
    """
    ln = _layernorm(channels, seed)
    x = _homoscedastic(channels, spread, seed)
    payload = calibrate_int_layernorm(ln, x)
    with torch.no_grad():
        ref = ln(x)
        out = ILayerNorm.from_payload(payload)(x)

    bias = float((out - ref).mean()) / _reference_lsb(ref, payload["clamp_bits"])
    assert abs(bias) < 0.20, f"systematic output bias {bias:+.3f} ref-LSB"
    # the constant must actually be there (s2 == 0 would make the claim vacuous)
    assert payload["scalars"][5] > 0


def test_ilayernorm_beats_a_mis_scaled_baseline():
    """Sanity: the calibrated scales matter — halving s2 must degrade the result."""
    ln = _layernorm(32, 11)
    x = _homoscedastic(32, 1.2, 11)
    payload = calibrate_int_layernorm(ln, x)
    good = ILayerNorm.from_payload(payload)

    broken_scalars = list(payload["scalars"])
    broken_scalars[5] -= 1  # wrong requant shift, same declared output scale
    broken = ILayerNorm(broken_scalars, payload["rsqrt_table"], payload["lnw"], payload["lnb"],
                        input_scale=payload["input_scale"], output_scale=payload["output_scale"],
                        rsqrt_table_two=payload.get("rsqrt_table_two"))
    with torch.no_grad():
        ref = ln(x)
        rel_good = float((good(x) - ref).norm() / ref.norm())
        rel_broken = float((broken(x) - ref).norm() / ref.norm())
    assert rel_good < rel_broken / 4.0, f"good={rel_good:.4f} broken={rel_broken:.4f}"


@pytest.mark.parametrize("channels,spread,seed", ACCURACY_CASES)
def test_declared_output_scale_is_the_one_the_integer_path_realizes(channels, spread, seed):
    """``output_scale`` is checked against DATA, not re-derived from its own definition.

    The old version of this test recomputed ``input_scale * rsqrt_scale * lnw_scale *
    2^s2`` and compared it to ``output_scale`` — which is literally how ``_derive_shift``
    defines it, so it could never fail. Here the scale is estimated from the integer
    codes the kernel emits and the float LayerNorm they are supposed to encode:
    ``alpha = <ref,ref> / <out_int,ref>`` is the least-squares scale that maps codes to
    the reference, and it must land on the declared value.
    """
    ln = _layernorm(channels, seed)
    x = _homoscedastic(channels, spread, seed)
    payload = calibrate_int_layernorm(ln, x)
    iln = ILayerNorm.from_payload(payload)
    with torch.no_grad():
        ref = ln(x).to(torch.float64)
        out_int = iln(QTensor.quantize(x, payload["input_scale"], 0.0, INT8)).int_data.to(torch.float64)

    alpha = float((ref * ref).sum() / (out_int * ref).sum())
    assert alpha == pytest.approx(payload["output_scale"], rel=0.01), (
        f"measured output LSB {alpha:.6g} vs declared {payload['output_scale']:.6g}")


@pytest.mark.parametrize("channels,spread,seed", ACCURACY_CASES)
def test_output_never_saturates_on_the_calibration_data(channels, spread, seed):
    """``_derive_shift``'s guard raises s2 until the observed peak fits in the clamp."""
    ln = _layernorm(channels, seed)
    x = _homoscedastic(channels, spread, seed)
    payload = calibrate_int_layernorm(ln, x)
    out = ILayerNorm.from_payload(payload)(QTensor.quantize(x, payload["input_scale"], 0.0, INT8))

    qmax = qrange(payload["clamp_bits"], signed=True)[1]
    assert int(out.int_data.abs().max()) <= qmax
    assert float((out.int_data.abs() >= qmax).float().mean()) < 0.001, "output pinned to the clamp"
    with torch.no_grad():
        assert float(ln(x).abs().max()) / payload["output_scale"] <= qmax


def test_zero_variance_rows_stay_finite():
    """A constant row has var_sum == 0; only ``eps`` keeps its rsqrt (and the table) finite."""
    channels = 12
    ln = _layernorm(channels, 7)
    x = torch.cat([_homoscedastic(channels, 1.0, 7, rows=200), torch.full((6, channels), 0.4)], 0)
    payload = calibrate_int_layernorm(ln, x)

    assert np.isfinite(np.asarray(payload["rsqrt_table"], dtype=np.float64)).all()
    assert payload["metrics"]["var_sum_min"] == 0
    iln = ILayerNorm.from_payload(payload)
    with torch.no_grad():
        ref, out = ln(x), iln(x)
    assert bool(torch.isfinite(out).all())
    # a constant row normalizes to 0, so the integer path must reproduce the bias alone
    assert float((out[-6:] - ln.bias.detach()).abs().max()) < payload["output_scale"]
    assert float((out - ref).norm() / ref.norm()) < 0.035


def test_rsqrt_table_is_capped_by_the_eps_floor():
    """``eps`` is load-bearing: a zero-width bin at var_sum == 0 is reachable.

    One huge activation drags ``input_scale`` up until most rows quantize to a handful
    of integer steps, the fitted index answers with ``(b, s1) = (0, 0)`` — one integer
    of variance per bin — and bin 0 then covers exactly ``var_sum == 0``. Without the
    ``+ eps`` inside the square roots that entry is ``1/sqrt(0)``, and the table becomes
    infinite. With it, the entry lands exactly on the ceiling ``1/sqrt(eps)`` that
    ``F.layer_norm`` itself imposes.
    """
    channels = 24
    ln = _layernorm(channels, 7)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(256, channels, generator=gen) * 0.05 + 0.2
    x[0, 0] = 25.0
    payload = calibrate_int_layernorm(ln, x)

    assert payload["scalars"][2] == 0 and payload["scalars"][3] == 0, (
        "fixture no longer forces a zero-width bin at var_sum == 0")
    assert (_row_var_sum(x, payload) == 0).sum() > 0

    table = np.asarray(payload["rsqrt_table"], dtype=np.float64) * payload["rsqrt_scale"]
    assert np.isfinite(table).all()
    ceiling = 1.0 / math.sqrt(payload["eps"])
    assert float(table.max()) == pytest.approx(ceiling, rel=1e-3), (
        f"largest rsqrt entry {table.max():.2f} vs the eps ceiling {ceiling:.2f}")


@pytest.mark.parametrize("seed", HETERO_SEEDS)
@pytest.mark.parametrize("segments", [1, 2])
def test_reported_index_metrics_match_an_independent_measurement(seed, segments):
    """``metrics`` documents the PRECONDITION, so it has to be true, not decorative.

    ``index_lo`` / ``index_hi`` keep meaning "the first and last ``var_sum`` the index
    resolves without clamping" for both shapes — for two segments that is segment ONE's
    lower edge and segment TWO's upper edge, which is what makes ``rows_below_range`` /
    ``rows_above_range`` still count rows pinned to an extreme table entry. Re-derived
    here from the scalars rather than read back from the calibrator.
    """
    channels = 48
    ln = _layernorm(channels, seed)
    x = _heteroscedastic(channels, seed)
    payload = calibrate_int_layernorm(ln, x, segments=segments)
    metrics = payload["metrics"]

    err = np.abs(_deployed_rsqrt_error(x, payload))
    assert metrics["rsqrt_rel_rms"] == pytest.approx(float(np.sqrt(np.mean(err ** 2))), rel=0.02)
    assert metrics["rsqrt_rel_max"] == pytest.approx(float(err.max()), rel=0.02)

    var_sum = _row_var_sum(x, payload)
    _, _, b, s1, bound = payload["scalars"][:5]
    lo_edge = -b
    if segments == 1:
        hi_edge = -b + ((bound + 1) << s1) - 1
        assert metrics["segment_threshold"] is None and metrics["rows_segment_two"] == 0
    else:
        b_two, s1_two, bound_two = payload["scalars"][7:]
        hi_edge = -b_two + ((bound_two + 1) << s1_two) - 1
        threshold = ((bound + 1) << s1) - b
        assert metrics["segment_threshold"] == threshold
        assert metrics["rows_segment_two"] == int(np.count_nonzero(var_sum >= threshold))
        # segment two starts exactly where segment one overflows (the softmax precedent):
        # no entry of it is wasted below, and no var_sum falls between the two.
        assert -b_two == threshold
    assert metrics["segments"] == segments
    assert metrics["index_lo"] == lo_edge and metrics["index_hi"] == hi_edge
    assert metrics["rows"] == var_sum.size
    assert metrics["rows_below_range"] == int(np.count_nonzero(var_sum < lo_edge))
    assert metrics["rows_above_range"] == int(np.count_nonzero(var_sum > hi_edge))
    assert metrics["var_sum_min"] == int(var_sum.min())
    assert metrics["var_sum_max"] == int(var_sum.max())


def test_accuracy_on_the_real_frame_model_layernorms():
    """The claim "1.2-1.9% on the FrameModel's five LayerNorms" is measured here (LN-8).

    Measured with this fixture: 1.86% / 1.93% / 1.29% / 1.37% / 1.39% relative error and
    0.435-0.597 reference LSB of MAE. The model's tokens are near-homoscedastic (var_sum
    dynamic range 1.6-2.4), which is exactly why it does NOT stress the rsqrt index —
    ``test_heavy_tailed_variance_is_resolved_by_the_fitted_index`` covers that.
    """
    from engine.model_factory import make_model

    torch.manual_seed(0)
    model = make_model({
        "target": "models.frame.FrameModel", "embed_dim": 48, "patch_size": 16,
        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1},
    }).eval()

    seen: dict[str, list] = {}
    handles = [m.register_forward_pre_hook(
        lambda mod, inp, n=name: seen.setdefault(n, []).append(inp[0].detach()))
        for name, m in model.named_modules() if isinstance(m, nn.LayerNorm)]
    with torch.no_grad():
        model(torch.rand(8, 1, 64, 64))
    for handle in handles:
        handle.remove()

    modules = dict(model.named_modules())
    assert len(seen) == 5, f"expected 5 LayerNorms, saw {sorted(seen)}"
    for name, activations in seen.items():
        ln = modules[name]
        x = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
        payload = calibrate_int_layernorm(ln, x)
        with torch.no_grad():
            ref = ln(x)
            out = ILayerNorm.from_payload(payload)(x)
        lsb = _reference_lsb(ref, payload["clamp_bits"])
        rel = float((out - ref).norm() / ref.norm())
        assert rel < 0.03, f"{name}: relative error {rel:.4f}"
        assert float((out - ref).abs().mean()) / lsb < 0.75, f"{name}: MAE too large"


# --- (c) SHAPE / DTYPE CONTRACT ----------------------------------------------

@pytest.mark.parametrize("clamp_bits", [8, 6, 4])
def test_ilayernorm_output_contract(clamp_bits):
    scalars, rsqrt_table, lnw, lnb, x_int = _synthetic(
        16, (2, 9), seed=21, b=0, s1=11, s2=14, clamp_bits=clamp_bits, low=-128, high=128)
    iln = ILayerNorm(scalars, rsqrt_table, lnw, lnb, input_scale=0.02, output_scale=0.05)
    out = iln(QTensor(x_int, scale=0.02, zero_point=0.0, dtype=INT8))

    qmin, qmax = qrange(clamp_bits, signed=True)
    assert isinstance(out, QTensor)
    assert out.int_data.shape == x_int.shape
    assert out.int_data.dtype == torch.int32
    assert out.dtype.bits == clamp_bits and out.dtype.signed
    assert int(out.int_data.min()) >= qmin and int(out.int_data.max()) <= qmax
    assert float(out.scale) == 0.05 and float(out.zero_point) == 0.0


def test_ilayernorm_float_path_matches_integer_path():
    ln = _layernorm(20, 13)
    x = _homoscedastic(20, 1.1, 13, rows=64)
    payload = calibrate_int_layernorm(ln, x)
    iln = ILayerNorm.from_payload(payload)
    with torch.no_grad():
        y_float = iln(x)
        y_int = iln(QTensor.quantize(x, payload["input_scale"], 0.0, INT8)).dequantize()
    assert torch.equal(y_float, y_int)
    assert y_float.shape == x.shape


def test_ilayernorm_rescales_a_mismatched_input_scale():
    """An upstream QTensor at a different scale is bridged, not silently misread."""
    ln = _layernorm(16, 14)
    x = _homoscedastic(16, 1.4, 14)
    payload = calibrate_int_layernorm(ln, x)
    iln = ILayerNorm.from_payload(payload)
    matched = iln(QTensor.quantize(x, payload["input_scale"], 0.0, INT8)).dequantize()
    upstream = QTensor.quantize(x, payload["input_scale"] * 2.0, 0.0, INT8)  # coarser upstream grid
    bridged = iln(upstream).dequantize()
    assert torch.allclose(bridged, matched, rtol=0.0, atol=6 * payload["output_scale"])


def test_ilayernorm_handles_a_per_channel_input_scale():
    """``ILinear.forward_accumulator`` can emit a per-channel scale; element 0 is not it."""
    channels = 16
    ln = _layernorm(channels, 8)
    x = _homoscedastic(channels, 1.0, 8)
    payload = calibrate_int_layernorm(ln, x)
    iln = ILayerNorm.from_payload(payload)

    flat = QTensor.quantize(x, payload["input_scale"], 0.0, INT8)
    uniform = QTensor(flat.int_data, scale=torch.full((channels,), payload["input_scale"]), dtype=INT8)
    assert torch.equal(iln(uniform).int_data, iln(flat).int_data)

    gen = torch.Generator().manual_seed(3)
    scales = payload["input_scale"] * torch.exp(0.3 * torch.randn(channels, generator=gen))
    per_channel = QTensor.quantize(x, scales, 0.0, INT8)
    with torch.no_grad():
        ref = ln(per_channel.dequantize())
    assert float((iln(per_channel).dequantize() - ref).norm() / ref.norm()) < 0.05
    # taking element 0 of that scale would misread 15 of the 16 channels
    wrong = QTensor(per_channel.int_data, scale=float(scales[0]), dtype=INT8)
    assert not torch.equal(iln(per_channel).int_data, iln(wrong).int_data)

    with pytest.raises(ValueError):
        iln(QTensor(torch.zeros(4, channels, dtype=torch.int32), scale=torch.ones(5), dtype=INT8))


# --- construction from the Q tier + guards -----------------------------------

def test_from_qlayernorm_reuses_the_q_tier_module():
    ln = _layernorm(32, 15)
    x = _homoscedastic(32, 1.3, 15, rows=350).reshape(50, 7, 32)
    var = x.var(dim=-1, unbiased=False).reshape(-1).numpy()
    rsqrt_payload = build_function_lut(
        var, lambda v, e=float(ln.eps): 1.0 / np.sqrt(np.maximum(v, 0.0) + e),
        entries=64, input_dtype=UINT8, output_dtype=INT8)
    qln = QLayerNorm(ln, rsqrt_payload)

    iln = ILayerNorm.from_qlayernorm(qln, x)
    assert iln.channels == 32
    with torch.no_grad():
        ref = ln(x)
        rel = float((iln(x) - ref).norm() / ref.norm())
    assert rel < 0.05, f"relative error {rel:.4f}"


def test_ilayernorm_rejects_bad_construction():
    with pytest.raises(ValueError):
        ILayerNorm([1, 1, 0, 0, 3], [1, 1, 1, 1], [1, 1], [0, 0], input_scale=1.0, output_scale=1.0)
    with pytest.raises(ValueError):  # c_1_s == 0 -> the golden's 1 << (c_1_s - 1) is undefined
        ILayerNorm([1, 0, 0, 0, 3, 2, 8], [1, 1, 1, 1], [1, 1], [0, 0], input_scale=1.0, output_scale=1.0)
    with pytest.raises(ValueError):  # table too short for the declared bound
        ILayerNorm([1, 1, 0, 0, 7, 2, 8], [1, 1], [1, 1], [0, 0], input_scale=1.0, output_scale=1.0)

    iln = ILayerNorm([1, 1, 0, 0, 3, 2, 8], [1, 1, 1, 1], [1, 1], [0, 0], input_scale=1.0, output_scale=1.0)
    with pytest.raises(ValueError):  # channel-count mismatch
        iln(QTensor(torch.zeros(2, 5, dtype=torch.int32), scale=1.0, zero_point=0.0, dtype=INT8))


def test_ilayernorm_rejects_a_half_specified_segmented_index():
    """The 10 scalars and the second table must arrive together, in both directions.

    Either half alone is worse than an error: 10 scalars without the table would index a
    table that is not there, and 7 scalars *with* one would run the one-segment kernel on
    a payload calibrated for two — reading segment one's table for every row the second
    segment exists to serve, silently.
    """
    ten = [1, 1, 0, 0, 3, 2, 8, -16, 1, 3]
    table = [1, 1, 1, 1]
    with pytest.raises(ValueError):                      # 10 scalars, no second table
        ILayerNorm(ten, table, [1, 1], [0, 0], input_scale=1.0, output_scale=1.0)
    with pytest.raises(ValueError):                      # 7 scalars, a second table
        ILayerNorm([1, 1, 0, 0, 3, 2, 8], table, [1, 1], [0, 0],
                   input_scale=1.0, output_scale=1.0, rsqrt_table_two=table)
    with pytest.raises(ValueError):                      # second table too short
        ILayerNorm(ten, table, [1, 1], [0, 0], input_scale=1.0, output_scale=1.0,
                   rsqrt_table_two=[1, 1])
    with pytest.raises(ValueError):                      # neither 7 nor 10
        ILayerNorm(ten + [0], table, [1, 1], [0, 0], input_scale=1.0, output_scale=1.0,
                   rsqrt_table_two=table)
    with pytest.raises(ValueError):                      # segments must be 1 or 2
        calibrate_int_layernorm(_layernorm(8, 1), _homoscedastic(8, 1.0, 1, rows=40), segments=3)


def test_calibrate_rejects_multi_dim_normalized_shape():
    with pytest.raises(NotImplementedError):
        calibrate_int_layernorm(nn.LayerNorm((4, 6)), torch.randn(10, 4, 6))
