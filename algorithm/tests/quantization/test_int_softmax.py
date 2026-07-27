"""I-tier fully-integer Softmax: ISoftmax bit-exact vs the i_ops golden + accuracy."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import numpy as np

import quantization.int_calibrate_softmax as ics
from quantization.i_ops import softmax_quantize as g_softmax_quantize
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.softmax import ISoftmax
from quantization.int_calibrate_softmax import (
    build_softmax_int_payload,
    build_softmax_int_payload_from_qsoftmax,
)
from quantization.lut_calibrate import PotIndexParams, coordinates_for
from quantization.scheme import INT8, UINT8


def _mixed_temperature_logits(heads: int, tokens: int, seed: int = 0) -> torch.Tensor:
    """[heads, tokens, tokens] logits whose rows alternate peaky / flat.

    Peaky rows give a small exp accumulator (reciprocal segment one), flat rows a
    large one (segment two) — so a calibration over this batch exercises both.
    """
    torch.manual_seed(seed)
    base = torch.randn(heads, tokens, tokens)
    row_id = torch.arange(heads * tokens).reshape(heads, tokens, 1).float()
    temperature = torch.where(row_id % 4 < 2, torch.tensor(6.0), torch.tensor(0.15))
    return base * temperature


def _attention_logits(heads: int, tokens: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Real pre-softmax attention scores ``Q @ K^T / sqrt(d)``."""
    torch.manual_seed(seed)
    q = torch.randn(heads, tokens, dim)
    k = torch.randn(heads, tokens, dim)
    return (q @ k.transpose(-2, -1)) / math.sqrt(dim)


def _to_int(logits: torch.Tensor, input_scale: float) -> torch.Tensor:
    return torch.round(logits / input_scale).clamp(INT8.qmin, INT8.qmax).to(torch.int32)


def _golden(x_int: torch.Tensor, payload: dict, heads: int, tokens: int) -> torch.Tensor:
    flat = g_softmax_quantize(
        x_int.reshape(-1).tolist(),
        payload["scalars"],
        payload["exp_table"],
        payload["recip_table_one"],
        payload["recip_table_two"],
        tokens=tokens,
        heads=heads,
    )
    return torch.tensor(flat, dtype=torch.int64).reshape(heads, tokens, tokens)


def _row_sums(payload: dict, logits: torch.Tensor) -> torch.Tensor:
    """Dequantized probability mass per row — the headline health metric."""
    isoftmax = ISoftmax.from_payload(payload)
    out = isoftmax.forward_int(_to_int(logits, payload["input_scale"])).to(torch.float64)
    return (out * payload["output_scale"]).sum(dim=-1)


def _max_abs_error_vs_float(payload: dict, logits: torch.Tensor) -> float:
    """Worst absolute probability error against ``torch.softmax`` of the FLOAT logits."""
    isoftmax = ISoftmax.from_payload(payload)
    out = isoftmax.forward_int(_to_int(logits, payload["input_scale"])).to(torch.float64)
    out = out * payload["output_scale"]
    return float((out - torch.softmax(logits.to(torch.float64), dim=-1)).abs().max())


def _rebuilt_with(name: str, replacement, rows, **kwargs) -> dict:
    """Rebuild a payload with one representative helper swapped out.

    Used to prove that a design decision inside the calibrator is load-bearing: the
    alternative is fed through the *whole* calibration so the comparison is apples to
    apples (tables, pivot and requant shifts all follow the swapped representative).
    """
    original = getattr(ics, name)
    setattr(ics, name, replacement)
    try:
        return build_softmax_int_payload(rows, **kwargs)
    finally:
        setattr(ics, name, original)


def _left_edge(params: PotIndexParams) -> np.ndarray:
    """The HG-PIPE reference representative: the low end of each PoT bin."""
    return np.maximum(coordinates_for(params).astype(np.float64), 1.0)


# --- (a) bit-exactness, both reciprocal segments exercised --------------------

def test_isoftmax_bit_exact_both_segments():
    heads, tokens = 2, 16
    logits = _mixed_temperature_logits(heads, tokens, seed=0)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    isoftmax = ISoftmax.from_payload(payload)
    x_int = _to_int(logits, payload["input_scale"])

    # the branch is data-dependent: prove this input really drives both tables
    mask = isoftmax.segment_mask(x_int)
    assert bool(mask.any()), "no row took reciprocal segment two"
    assert bool((~mask).any()), "no row took reciprocal segment one"

    assert torch.equal(isoftmax.forward_int(x_int), _golden(x_int, payload, heads, tokens))


@pytest.mark.parametrize("exp_entries,recip_entries", [(64, 64), (128, 256), (256, 256)])
def test_isoftmax_bit_exact_table_sizes(exp_entries, recip_entries):
    heads, tokens = 3, 8
    logits = _mixed_temperature_logits(heads, tokens, seed=1)
    payload = build_softmax_int_payload(
        logits.reshape(-1, tokens).numpy(), exp_entries=exp_entries, recip_entries=recip_entries
    )
    isoftmax = ISoftmax.from_payload(payload)
    x_int = _to_int(logits, payload["input_scale"])
    mask = isoftmax.segment_mask(x_int)
    assert bool(mask.any()) and bool((~mask).any())
    assert len(payload["exp_table"]) == exp_entries
    assert len(payload["recip_table_one"]) == recip_entries
    assert torch.equal(isoftmax.forward_int(x_int), _golden(x_int, payload, heads, tokens))


def test_isoftmax_bit_exact_on_unseen_logits():
    """Calibrated on one batch, run on another — clamping paths must stay bit-exact."""
    heads, tokens = 2, 16
    payload = build_softmax_int_payload(
        _mixed_temperature_logits(heads, tokens, seed=2).reshape(-1, tokens).numpy()
    )
    isoftmax = ISoftmax.from_payload(payload)
    unseen = _mixed_temperature_logits(heads, tokens, seed=99) * 1.7
    x_int = _to_int(unseen, payload["input_scale"])
    assert torch.equal(isoftmax.forward_int(x_int), _golden(x_int, payload, heads, tokens))


def test_isoftmax_branch_is_strictly_greater_than_bound():
    """A row whose segment-one cursor lands exactly on ``bound2_one`` stays in segment one.

    The golden branches on ``cursor_one > bound2_one`` (strict), so the top bin of the
    first reciprocal table is a segment-*one* bin. Random logits practically never land
    there, so this uses a hand-built payload: ``exp_table = [1000, 0, 0, 0]`` with
    ``s1 = 0`` makes ``acc = 1000 * (number of row elements equal to the row max)``, and
    the segment-one index is chosen so 6 and 7 hot elements sit exactly on the bound.
    """
    tokens = 8
    scalars = [
        0, 0, 3,                  # exp: b1, s1, bound1
        -1000, 11, 2,             # recip one: cursor = (acc - 1000) >> 11, bound 2
        512, 10,                  # b3_one, s3_one
        -7144, 8, 3,              # recip two starts where segment one overflows
        512, 10,                  # b3_two, s3_two
        8,                        # clamp_bits
    ]
    payload = {
        "scalars": scalars,
        "exp_table": [1000, 0, 0, 0],
        "recip_table_one": [261, 87, 43],
        "recip_table_two": [37, 36, 34, 32],
        "input_scale": 1.0,
        "output_scale": 1.0 / 255.0,
    }
    assert -scalars[8] == ((scalars[5] + 1) << scalars[4]) - scalars[3]

    # row j has j elements at the max and the rest far below -> acc = 1000 * (j + 1)
    rows = []
    for j in range(tokens):
        row = [-3] * tokens  # delta 3 -> exp_table[3] = 0
        for i in range(j + 1):
            row[i] = 0
        rows.append(row)
    x_int = torch.tensor([rows], dtype=torch.int32)  # [1, tokens, tokens]

    acc = torch.tensor([1000 * (j + 1) for j in range(tokens)], dtype=torch.int64)
    cursor_one = torch.bitwise_right_shift(acc + scalars[3], scalars[4])
    assert int((cursor_one == scalars[5]).sum()) >= 1, "no row sits exactly on bound2_one"
    assert int((cursor_one < scalars[5]).sum()) >= 1 and int((cursor_one > scalars[5]).sum()) >= 1

    isoftmax = ISoftmax.from_payload(payload)
    assert torch.equal(isoftmax.forward_int(x_int), _golden(x_int, payload, 1, tokens))
    # the boundary rows must be served by segment one, not segment two
    assert torch.equal(isoftmax.segment_mask(x_int).reshape(-1), cursor_one > scalars[5])


def test_isoftmax_detects_wrong_scalars():
    """Negative control: the golden and ISoftmax must disagree if a scalar is wrong."""
    heads, tokens = 2, 16
    logits = _mixed_temperature_logits(heads, tokens, seed=3)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    x_int = _to_int(logits, payload["input_scale"])
    broken = ISoftmax.from_payload(payload)
    broken.s3_two = broken.s3_two + 1  # only the segment-two rows change
    assert not torch.equal(broken.forward_int(x_int), _golden(x_int, payload, heads, tokens))


def test_isoftmax_honours_a_non_default_dim():
    """``dim`` really steers every reduction — it is not decoration on the last axis.

    ``ISoftmax(dim=1)`` applied to ``x`` must equal ``ISoftmax(dim=-1)`` applied to ``x``
    with the two token axes swapped. Hard-coding the reduction axis to -1 breaks this,
    because reducing the transposed tensor over its last axis is a different reduction.
    """
    heads, tokens = 2, 16
    logits = _mixed_temperature_logits(heads, tokens, seed=15)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    x_int = _to_int(logits, payload["input_scale"])

    over_last = ISoftmax.from_payload(payload).forward_int(x_int)
    over_axis_1 = ISoftmax.from_payload(payload, dim=1).forward_int(x_int.transpose(1, 2))
    assert torch.equal(over_axis_1, over_last.transpose(1, 2))
    # and the two are genuinely different tensors, so the assertion above has content
    assert not torch.equal(over_last, over_last.transpose(1, 2))


# --- (b) accuracy vs float softmax -------------------------------------------

def test_isoftmax_accuracy_vs_float_softmax():
    """Error against ``torch.softmax`` of the ORIGINAL float logits.

    Deliberately *not* against ``torch.softmax(x_int * input_scale)``: that reference is
    derived from the calibrator's own choice of ``input_scale`` and its own rounded
    integers, so it cancels the input-grid selection out of the measurement and leaves it
    untested. Measured here: mean 1.0170e-03, max 4.9463e-03 (the uint8 output step is
    1/255 = 3.92e-03, so the worst element is ~1.26 output steps out). The max bound is
    deliberately tight enough to notice a wasted input grid: doubling ``input_scale``
    takes the max to 7.3895e-03.
    """
    heads, tokens, dim = 3, 32, 16
    logits = _attention_logits(heads, tokens, dim, seed=4)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    isoftmax = ISoftmax.from_payload(payload)
    x_int = _to_int(logits, payload["input_scale"])

    out = isoftmax.forward_int(x_int).to(torch.float64) * payload["output_scale"]
    reference = torch.softmax(logits.to(torch.float64), dim=-1)
    err = (out - reference).abs()
    mean_abs, max_abs = float(err.mean()), float(err.max())
    assert mean_abs < 3e-3, f"mean abs error {mean_abs:.2e}"
    assert max_abs < 6e-3, f"max abs error {max_abs:.2e}"

    row_sums = out.sum(dim=-1)
    # measured: [0.9843, 1.0118], mean 0.9996
    assert float(row_sums.min()) > 0.95, f"min row sum {float(row_sums.min()):.4f}"
    assert float(row_sums.max()) < 1.05, f"max row sum {float(row_sums.max()):.4f}"


def test_isoftmax_accuracy_with_hardware_sized_tables():
    """A coarse 32-entry reciprocal still keeps the row probability mass near 1.

    This is the setting where the choice of table representative bites, so the test
    rebuilds the payload with the reference's left-edge representative and shows it
    misses the band this one holds. Measured: geometric mean [0.9294, 1.0667] (mean
    0.9904) against left edge [1.0000, 1.1412] — the left edge is the *smallest*
    accumulator in its bin, so every reciprocal is biased high and every row inflates.
    """
    heads, tokens = 2, 16
    logits = _mixed_temperature_logits(heads, tokens, seed=14)
    rows = logits.reshape(-1, tokens).numpy()
    payload = build_softmax_int_payload(rows, exp_entries=64, recip_entries=32)

    row_sums = _row_sums(payload, logits)
    assert float(row_sums.min()) > 0.90, f"min row sum {float(row_sums.min()):.4f}"
    assert float(row_sums.max()) < 1.09, f"max row sum {float(row_sums.max()):.4f}"
    assert abs(float(row_sums.mean()) - 1.0) < 0.03, f"mean row sum {float(row_sums.mean()):.4f}"

    left = _rebuilt_with("_bin_geometric_means", _left_edge, rows, exp_entries=64, recip_entries=32)
    left_sums = _row_sums(left, logits)
    assert float(left_sums.max()) > 1.10, (
        f"left-edge reciprocal max row sum {float(left_sums.max()):.4f} — the geometric-mean "
        "representative is supposed to be what keeps this inside the band"
    )


def test_default_input_scale_spans_the_whole_int8_grid():
    """The calibrated input grid must use int8 fully — no saturation, no wasted range.

    ``input_scale`` decides how finely the exp table's argument is resolved, and nothing
    downstream can recover resolution thrown away here. A grid that is too coarse is
    almost invisible in the output (doubling ``input_scale`` moves the max abs error only
    from 4.9e-03 to 7.4e-03), so it gets asserted directly instead.
    """
    for logits in (
        _attention_logits(3, 32, 16, seed=4),
        _mixed_temperature_logits(2, 16, seed=0),
    ):
        tokens = logits.shape[-1]
        payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
        peak = int(_to_int(logits, payload["input_scale"]).abs().max())
        assert peak == 127, f"input grid peaks at {peak}, not at the int8 limit"


def test_isoftmax_float_io_drop_in():
    """Float in / float out convenience path stays close to nn.Softmax."""
    heads, tokens, dim = 2, 24, 12
    logits = _attention_logits(heads, tokens, dim, seed=5)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    isoftmax = ISoftmax.from_payload(payload)
    with torch.no_grad():
        out = isoftmax(logits)
    assert out.shape == logits.shape
    err = float((out - torch.softmax(logits, dim=-1)).abs().mean())
    assert err < 5e-3, f"float-path mean abs error {err:.2e}"


@pytest.mark.parametrize("scale_ratio", [1.0, 2.0, 4.0])
def test_isoftmax_rescales_mismatched_qtensor_scale(scale_ratio):
    """Upstream QTensor at a coarser scale is rebased onto the exp table grid.

    Row sums alone cannot check this: skipping the rebase feeds the table a differently
    scaled logit, which still yields a perfectly normalised softmax — of the wrong
    distribution. So the assertion is against ``torch.softmax`` of the float logits.
    Measured max abs error with the rebase: 5.72e-03, 7.54e-03, 1.43e-02 at ratios 1, 2,
    4; without it, 5.72e-03, 1.78e-01, 2.53e-01.

    Only ratios >= 1 are meaningful. The calibrated scale already spans int8, so quantizing
    the same logits at a *finer* scale just clips at +-127 and measures saturation.
    """
    heads, tokens, dim = 2, 16, 8
    logits = _attention_logits(heads, tokens, dim, seed=6)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    isoftmax = ISoftmax.from_payload(payload)

    upstream_scale = payload["input_scale"] * scale_ratio
    unclamped = torch.round(logits.to(torch.float64) / upstream_scale)
    assert int(unclamped.max()) <= INT8.qmax and int(unclamped.min()) >= INT8.qmin, (
        "fixture clips at the int8 edge, so it would measure saturation, not the rebase"
    )
    qt = QTensor.quantize(logits, upstream_scale, 0.0, INT8)
    out = isoftmax(qt)
    assert isinstance(out, QTensor)

    dequantized = out.dequantize().to(torch.float64)
    row_sums = dequantized.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.05)
    err = float((dequantized - torch.softmax(logits.to(torch.float64), dim=-1)).abs().max())
    assert err < 2e-2, f"max abs error {err:.2e}: the upstream scale was not rebased"


def test_isoftmax_is_invariant_to_qtensor_zero_point():
    """``forward_qtensor`` drops ``zero_point``; prove the drop is a genuine cancellation.

    ``forward_int`` subtracts the row max first, so a uniform additive offset on the
    integer data leaves every ``delta`` untouched. Two QTensors describing the *same*
    real values — one at zero_point 0, one shifted by +7 — must therefore produce
    identical integer output. This is the guard behind the claim in the
    ``forward_qtensor`` docstring; it is exact only because both carry the same scale,
    which makes the float32 rebase an identity.
    """
    heads, tokens, dim = 2, 16, 8
    logits = _attention_logits(heads, tokens, dim, seed=16)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    isoftmax = ISoftmax.from_payload(payload)

    scale = payload["input_scale"]
    at_zero = QTensor.quantize(logits, scale, 0.0, INT8)
    shifted = QTensor(at_zero.int_data + 7, scale=scale, zero_point=7.0, dtype=INT8)
    assert not torch.equal(at_zero.int_data, shifted.int_data)
    assert torch.equal(
        isoftmax.forward_qtensor(at_zero).int_data,
        isoftmax.forward_qtensor(shifted).int_data,
    )


# --- (c) unsigned output contract --------------------------------------------

@pytest.mark.parametrize("output_bits", [4, 8])
def test_isoftmax_output_is_unsigned_and_in_range(output_bits):
    heads, tokens = 2, 16
    logits = _mixed_temperature_logits(heads, tokens, seed=7)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy(), output_bits=output_bits)
    isoftmax = ISoftmax.from_payload(payload)
    out = isoftmax.forward_int(_to_int(logits, payload["input_scale"]))
    qmax = (1 << output_bits) - 1
    assert int(out.min()) >= 0
    assert int(out.max()) <= qmax
    assert payload["scalars"][13] == output_bits
    assert payload["output_scale"] == pytest.approx(1.0 / qmax)

    qt_out = isoftmax(QTensor.quantize(logits, payload["input_scale"], 0.0, INT8))
    assert qt_out.dtype.signed is False and qt_out.dtype.bits == output_bits


def test_isoftmax_output_clamp_actually_binds():
    """The upper clamp must be load-bearing, not a decoration that never fires.

    On any payload the calibrator emits, ``exp_value * recip >> s3`` lands under ``qmax``
    by construction, so ``out.max() <= qmax`` is trivially true and deleting the clamp
    goes unnoticed. This uses a hand-built payload whose segment-one reciprocal is far
    too large — ``(1000 * 1000 + 512) >> 10 == 977`` — so the clamp is the only thing
    keeping the uint8 contract. Row 3 is left under the limit (254) so the test also
    shows the clamp is not swallowing everything.
    """
    tokens = 4
    scalars = [
        0, 0, 3,                  # exp: b1, s1, bound1
        -1000, 10, 3,             # recip one: cursor = (acc - 1000) >> 10, bound 3
        512, 10,                  # b3_one, s3_one
        -5096, 10, 3,             # recip two starts where segment one overflows
        512, 10,                  # b3_two, s3_two
        8,                        # clamp_bits -> qmax 255
    ]
    payload = {
        "scalars": scalars,
        "exp_table": [1000, 0, 0, 0],
        "recip_table_one": [1000, 400, 260, 200],
        "recip_table_two": [1, 1, 1, 1],
        "input_scale": 1.0,
        "output_scale": 1.0 / 255.0,
    }
    assert -scalars[8] == ((scalars[5] + 1) << scalars[4]) - scalars[3]

    rows = []
    for j in range(tokens):  # row j: j+1 elements at the max -> acc = 1000 * (j + 1)
        row = [-3] * tokens
        for i in range(j + 1):
            row[i] = 0
        rows.append(row)
    x_int = torch.tensor([rows], dtype=torch.int32)

    isoftmax = ISoftmax.from_payload(payload)
    out = isoftmax.forward_int(x_int)
    assert not bool(isoftmax.segment_mask(x_int).any()), "this payload must stay in segment one"

    unclamped_top = (payload["exp_table"][0] * payload["recip_table_one"][0] + scalars[6]) >> scalars[7]
    assert unclamped_top == 977 > 255, "the fixture no longer overflows uint8"
    assert int(out.max()) == 255, "the upper clamp did not bind"
    assert int(out[0, 3, 0]) == 254, "row 3 must stay under the clamp"
    assert torch.equal(out, _golden(x_int, payload, 1, tokens))


def test_isoftmax_qtensor_dtype_is_uint8_by_default():
    logits = _attention_logits(1, 8, 8, seed=8)
    payload = build_softmax_int_payload(logits.reshape(-1, 8).numpy())
    out = ISoftmax.from_payload(payload)(QTensor.quantize(logits, payload["input_scale"], 0.0, INT8))
    assert out.dtype == UINT8


# --- (d) the accumulator envelope --------------------------------------------

@pytest.mark.parametrize("flavour", ["peaky", "flat", "mixed"])
def test_accumulator_segments_cover_the_theoretical_envelope(flavour):
    """The reciprocal segments must span every accumulator that can ever occur.

    ``acc`` is bounded by ``[exp_table[0], tokens * max(exp_table)]`` for *any* logits
    (one element is always the row max, and a constant row makes every element one), so
    the calibrator sizes the segments from that envelope instead of from the observed
    accumulators. The three assertions below are exactly the two failure modes the
    observed range allows: a low accumulator clamped onto entry 0 of segment one, and a
    high one saturated at ``bound2_two``. Sizing from the observed range leaves the
    envelope uncovered for the peaky and flat fixtures.
    """
    tokens = 16
    torch.manual_seed(0)
    logits = {
        "peaky": torch.randn(2, tokens, tokens) * 6.0,
        "flat": torch.randn(2, tokens, tokens) * 0.02,
        "mixed": _mixed_temperature_logits(2, tokens, seed=0),
    }[flavour]
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    metrics, scalars, exp_table = payload["metrics"], payload["scalars"], payload["exp_table"]

    lo, hi = metrics["acc_range_min"], metrics["acc_range_max"]
    assert lo == exp_table[0] == max(exp_table)
    assert hi == metrics["max_tokens"] * max(exp_table)
    assert lo <= metrics["acc_min"] and metrics["acc_max"] <= hi

    b2_one, s2_one, bound2_one = scalars[3:6]
    b2_two, s2_two, bound2_two = scalars[8:11]
    assert (lo + b2_one) >> s2_one == 0, "the smallest possible accumulator clamps low"
    assert (hi + b2_two) >> s2_two <= bound2_two, "the largest possible accumulator saturates"
    # segment two must pick up exactly where segment one's cursor overflows its bound
    assert -b2_two == ((bound2_one + 1) << s2_one) - b2_one


@pytest.mark.parametrize("tokens", [16, 32, 64])
def test_isoftmax_stays_normalised_on_out_of_distribution_logits(tokens):
    """Rows far outside the calibration distribution must still sum to about 1.

    Sizing the reciprocal from the *observed* accumulator range fails badly here: a
    calibration set of only peaky rows never sees a large accumulator, so a flat runtime
    row saturates ``cursor_two`` at ``bound2_two`` and gets divided by the calibration
    maximum instead of its own sum. Measured with an observed-range calibrator, row sums
    reached 6.4000 at 16 tokens and 12.8000 at 64; with the theoretical envelope the
    worst row sum anywhere in this test is 1.0471 (and the worst low one 0.9529, both on
    the 32-token flat-calibrated/peaky-input case).
    """
    torch.manual_seed(0)
    peaky = torch.randn(2, tokens, tokens) * 6.0
    flat = torch.randn(2, tokens, tokens) * 0.02
    constant = torch.zeros(2, tokens, tokens)

    peaky_payload = build_softmax_int_payload(peaky.reshape(-1, tokens).numpy())
    flat_payload = build_softmax_int_payload(flat.reshape(-1, tokens).numpy())
    for name, payload, unseen in (
        ("peaky-calibrated on flat rows", peaky_payload, flat),
        ("peaky-calibrated on constant rows", peaky_payload, constant),
        ("flat-calibrated on peaky rows", flat_payload, peaky),
    ):
        row_sums = _row_sums(payload, unseen)
        assert float(row_sums.min()) > 0.90, f"{name}: min row sum {float(row_sums.min()):.4f}"
        assert float(row_sums.max()) < 1.10, f"{name}: max row sum {float(row_sums.max()):.4f}"


def test_max_tokens_extends_the_envelope_for_longer_sequences():
    """``max_tokens`` is the knob for deploying on rows longer than calibration used.

    A 16-token calibration sizes the accumulator envelope for 16 tokens; a 96-token row
    can accumulate six times more and would saturate. Measured on this fixture: row sums
    reach 2.9059 at the default, and 1.0549 once ``max_tokens=96`` widens the envelope.
    (The residual ~5% is the uint8 *output* step, not the reciprocal: 96 probabilities
    of ~1/96 = 0.0104 each round to a multiple of 1/255 = 0.0039.)
    """
    tokens, long_tokens = 16, 96
    torch.manual_seed(3)
    rows = (torch.randn(4, tokens, tokens) * 3.0).reshape(-1, tokens).numpy()
    torch.manual_seed(11)
    long_flat = torch.randn(2, long_tokens, long_tokens) * 0.3

    narrow = build_softmax_int_payload(rows)
    wide = build_softmax_int_payload(rows, max_tokens=long_tokens)
    assert wide["metrics"]["acc_range_max"] == (long_tokens // tokens) * narrow["metrics"]["acc_range_max"]

    # D2 added a runtime guard, so the narrow payload now REFUSES the long row rather
    # than silently mis-normalising it. Assert the refusal, then deliberately override
    # max_tokens to reach the kernel anyway — the damage is what justifies the guard,
    # and a guard whose justification is untested is one nobody keeps.
    with pytest.raises(ValueError, match="max_tokens"):
        _row_sums(narrow, long_flat)
    unguarded = ISoftmax.from_payload(narrow)
    # max_tokens is a buffer now (it must survive a state_dict round trip), so the
    # deliberate bypass writes the buffer rather than the read-only view.
    unguarded.max_tokens_buf.fill_(long_tokens)
    unguarded_out = unguarded.forward_int(_to_int(long_flat, narrow["input_scale"]))
    unguarded_sums = (unguarded_out.to(torch.float64) * narrow["output_scale"]).sum(dim=-1)
    assert float(unguarded_sums.max()) > 1.5, "fixture no longer overflows the envelope"
    wide_sums = _row_sums(wide, long_flat)
    assert float(wide_sums.min()) > 0.90, f"min row sum {float(wide_sums.min()):.4f}"
    assert float(wide_sums.max()) < 1.10, f"max row sum {float(wide_sums.max()):.4f}"


# --- (e) reciprocal resolution -------------------------------------------------

@pytest.mark.parametrize("recip_bits", [8, 12, 16])
@pytest.mark.parametrize("recip_entries,output_bits", [(32, 8), (256, 8), (256, 4)])
def test_reciprocal_tables_spend_the_whole_recip_bits(recip_bits, recip_entries, output_bits):
    """Both segments must pick the largest ``s3`` their width allows.

    ``s3`` is chosen so the biggest entry — the one at the segment's smallest
    accumulator — just fits ``recip_bits``; a smaller ``s3`` would throw away reciprocal
    resolution for nothing. That pins the top entry into the top binade,
    ``[2^(recip_bits-1), 2^recip_bits - 1]``. Checking *both* tables matters: the
    segment-two shift is computed independently.
    """
    logits = _mixed_temperature_logits(2, 16, seed=20)
    payload = build_softmax_int_payload(
        logits.reshape(-1, 16).numpy(),
        recip_bits=recip_bits, recip_entries=recip_entries, output_bits=output_bits,
    )
    for name in ("recip_table_one", "recip_table_two"):
        table = payload[name]
        assert min(table) >= 0, f"{name} has a negative entry"
        assert max(table) <= (1 << recip_bits) - 1, f"{name} overflows {recip_bits} bits"
        assert max(table) >= 1 << (recip_bits - 1), (
            f"{name} tops out at {max(table)}, below the {recip_bits}-bit top binade — "
            "s3 is leaving reciprocal resolution unused"
        )


def test_default_reciprocal_resolution_is_sixteen_bits():
    """The default ``recip_bits=16`` must actually reach 16 bits of reciprocal.

    Separate from the parametrized test above, which passes ``recip_bits`` explicitly and
    so cannot notice the default being lowered.
    """
    logits = _attention_logits(3, 32, 16, seed=4)
    payload = build_softmax_int_payload(logits.reshape(-1, 32).numpy())
    # measured: recip_table_one 64284, recip_table_two 57628
    for name in ("recip_table_one", "recip_table_two"):
        assert max(payload[name]) >= 1 << 15, f"{name} max {max(payload[name])} is not 16-bit"
        assert max(payload[name]) <= (1 << 16) - 1


def test_coarse_reciprocal_bits_lose_accuracy():
    """Reciprocal resolution is load-bearing, so the structural test above has a point.

    Measured max |row sum - 1| on this fixture: 0.1059 at 4 bits, 0.0353 at 6, and
    0.0196 from 8 bits up (where the uint8 output step takes over as the error floor).
    """
    logits = _mixed_temperature_logits(2, 16, seed=20)
    rows = logits.reshape(-1, 16).numpy()

    def deviation(**kwargs) -> float:
        sums = _row_sums(build_softmax_int_payload(rows, **kwargs), logits)
        return max(abs(float(sums.max()) - 1.0), abs(float(sums.min()) - 1.0))

    coarse, fine = deviation(recip_bits=4), deviation()
    assert coarse > 0.08, f"4-bit reciprocal deviation {coarse:.4f} is suspiciously good"
    assert fine < 0.03, f"default reciprocal deviation {fine:.4f}"
    assert coarse > 3.0 * fine


def test_reciprocal_table_is_clipped_when_the_shift_floors():
    """The ``recip_bits`` clip in ``_reciprocal_segment`` is reachable, not dead code.

    ``s3`` bottoms out at 1, so a degenerate configuration — a tiny ``exp_scale``, which
    drags the whole accumulator envelope down, plus a narrow reciprocal — asks for a top
    entry that does not fit. Here the unclipped top entry would be 255 against a 4-bit
    ceiling of 15.
    """
    logits = _mixed_temperature_logits(2, 16, seed=20)
    payload = build_softmax_int_payload(
        logits.reshape(-1, 16).numpy(), exp_scale=2, recip_bits=4, output_bits=8
    )
    qmax, s3_one = 255, payload["scalars"][7]
    smallest_acc = payload["exp_table"][0]
    assert s3_one == 1 and smallest_acc == 2
    unclipped = round(qmax * 2 ** s3_one / smallest_acc)
    assert unclipped == 255 > 15, "fixture no longer overflows recip_bits"
    assert max(payload["recip_table_one"]) == 15, "the recip_bits clip did not bind"
    assert max(payload["recip_table_two"]) <= 15


# --- (f) table representatives -------------------------------------------------

def test_exp_table_samples_the_bin_midpoint():
    """The exp table is evaluated at the centre of each PoT bin, not at its left edge."""
    logits = _mixed_temperature_logits(2, 16, seed=14)
    payload = build_softmax_int_payload(logits.reshape(-1, 16).numpy(), exp_entries=64)
    b1, s1, bound1 = payload["scalars"][0:3]
    assert s1 > 0, "fixture must have bins wider than 1, or midpoint == left edge"

    coords = coordinates_for(PotIndexParams(b1, s1, bound1)).astype(np.float64)
    exp_scale = payload["metrics"]["exp_scale"]

    def table_at(points: np.ndarray) -> list[int]:
        y = np.exp(-np.maximum(points, 0.0) * payload["input_scale"]) * exp_scale
        return np.clip(np.rint(y), 0, exp_scale).astype(np.int64).tolist()

    assert payload["exp_table"] == table_at(coords + ((1 << s1) - 1) / 2.0)
    assert payload["exp_table"] != table_at(coords)


@pytest.mark.parametrize("exp_entries,recip_entries", [(64, 32), (32, 32), (64, 64)])
def test_exp_table_representative_is_not_a_lever(exp_entries, recip_entries):
    """Where the exp table samples inside its bin is *not* an accuracy decision.

    The PoT bins are equally wide, so a fixed within-bin offset multiplies every
    ``exp_value``, and therefore ``acc``, by the same factor — which cancels between the
    numerator and the denominator of the softmax ratio. Rebuilding the whole payload with
    left-edge exp sampling therefore moves the error by rounding noise with no consistent
    sign; measured ratios left/midpoint on this fixture: 1.133 at 64/32, 0.755 at 32/32,
    0.851 at 64/64. Contrast ``test_isoftmax_accuracy_with_hardware_sized_tables``, where
    the same swap on the *reciprocal* table pushes row sums out of band.
    """
    logits = _mixed_temperature_logits(2, 16, seed=14)
    rows = logits.reshape(-1, 16).numpy()
    kwargs = dict(exp_entries=exp_entries, recip_entries=recip_entries)

    midpoint = build_softmax_int_payload(rows, **kwargs)
    left = _rebuilt_with("_bin_midpoints", lambda p: coordinates_for(p).astype(np.float64), rows, **kwargs)
    assert midpoint["exp_table"] != left["exp_table"], "the swap did not change anything"

    err_mid = _max_abs_error_vs_float(midpoint, logits)
    err_left = _max_abs_error_vs_float(left, logits)
    assert 0.5 < err_left / err_mid < 2.0, (
        f"left edge {err_left:.4f} vs midpoint {err_mid:.4f}: the exp representative is "
        "behaving like an accuracy lever, which contradicts the cancellation argument"
    )


# --- (g) calibration contract -----------------------------------------------

def test_payload_shape_and_scalar_layout():
    logits = _attention_logits(2, 16, 8, seed=9)
    payload = build_softmax_int_payload(logits.reshape(-1, 16).numpy(), exp_entries=128, recip_entries=64)
    scalars = payload["scalars"]
    assert len(scalars) == 14
    b1, s1, bound1 = scalars[0:3]
    assert bound1 == 127 and s1 >= 0 and b1 == 0  # delta >= 0, so the exp index starts at 0
    assert scalars[5] == 63 and scalars[10] == 63  # both reciprocal bounds = entries - 1
    assert scalars[6] == 1 << (scalars[7] - 1)  # b3_one is the round-half-up bias for s3_one
    assert scalars[11] == 1 << (scalars[12] - 1)
    # segment two must start exactly where segment one's cursor overflows its bound
    assert -scalars[8] == ((scalars[5] + 1) << scalars[4]) - scalars[3]

    assert len(payload["exp_table"]) == 128
    assert len(payload["recip_table_one"]) == 64 and len(payload["recip_table_two"]) == 64
    assert 0 <= min(payload["exp_table"]) and max(payload["exp_table"]) <= 32768
    for name in ("recip_table_one", "recip_table_two"):
        assert min(payload[name]) >= 0 and max(payload[name]) <= 65535


def test_reference_reciprocal_emission_produces_all_zeros():
    """The HG-PIPE ``calibrate_softmax`` reciprocal/requant emission is numerically dead.

    It builds ``recip = round(recip_scale / acc)`` with ``recip_scale = 256`` while the
    accumulator it divides is ``~exp_scale * S`` (>= 32768), so every table entry rounds
    to 0 — and it then folds ``exp_scale`` into ``s3`` a second time although it already
    cancels between ``exp_value`` and ``recip``. Replayed here through the *same* golden
    kernel and the same index scalars our calibrator emits, it returns zeros everywhere,
    while our emission returns a real distribution. Guards the deviation documented in
    ``int_calibrate_softmax``.
    """
    heads, tokens = 2, 16
    logits = _mixed_temperature_logits(heads, tokens, seed=10)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy())
    x_int = _to_int(logits, payload["input_scale"])
    assert int(_golden(x_int, payload, heads, tokens).sum()) > 0

    exp_scale, recip_scale, output_bits = 32768.0, 256.0, payload["output_bits"]
    qmax = (1 << output_bits) - 1
    ref_b3 = 1 << output_bits
    ref_s3 = int(math.ceil(math.log2(exp_scale * recip_scale / qmax)))

    def ref_table(offset, shift, bound):
        coords = np.maximum(coordinates_for(PotIndexParams(offset, shift, bound)).astype(np.float64), 1.0)
        return np.clip(np.rint(recip_scale / coords), 0, int(recip_scale)).astype(np.int64).tolist()

    s = list(payload["scalars"])
    ref_payload = dict(payload)
    ref_payload["scalars"] = s[0:6] + [ref_b3, ref_s3] + s[8:11] + [ref_b3, ref_s3, output_bits]
    ref_payload["recip_table_one"] = ref_table(s[3], s[4], s[5])
    ref_payload["recip_table_two"] = ref_table(s[8], s[9], s[10])
    assert max(ref_payload["recip_table_one"]) == 0

    assert int(_golden(x_int, ref_payload, heads, tokens).max()) == 0


def test_build_from_qsoftmax_keeps_input_grid():
    from quantization.lut_calibrate import build_function_lut
    from quantization.qlayers.nonlinear import QSoftmax

    heads, tokens = 2, 16
    logits = _attention_logits(heads, tokens, 8, seed=11)
    delta = (logits.max(dim=-1, keepdim=True).values - logits).reshape(-1).numpy()
    sums = torch.exp(-(logits.max(dim=-1, keepdim=True).values - logits)).sum(dim=-1).reshape(-1).numpy()
    qsoftmax = QSoftmax(
        build_function_lut(delta, lambda d: np.exp(-np.maximum(d, 0.0)), entries=128, input_dtype=UINT8, output_dtype=UINT8),
        build_function_lut(sums, lambda s: 1.0 / np.maximum(s, 1e-8), entries=128, input_dtype=UINT8, output_dtype=UINT8),
    )
    payload = build_softmax_int_payload_from_qsoftmax(qsoftmax, logits.reshape(-1, tokens).numpy())
    assert payload["input_scale"] == pytest.approx(float(qsoftmax.exp_in))

    isoftmax = ISoftmax.from_payload(payload)
    x_int = _to_int(logits, payload["input_scale"])
    assert torch.equal(isoftmax.forward_int(x_int), _golden(x_int, payload, heads, tokens))
    assert _max_abs_error_vs_float(payload, logits) < 2e-2


@pytest.mark.parametrize("pivot", ["geometric", "reference"])
def test_pivot_strategies_are_both_bit_exact(pivot):
    heads, tokens = 2, 16
    logits = _mixed_temperature_logits(heads, tokens, seed=12)
    payload = build_softmax_int_payload(logits.reshape(-1, tokens).numpy(), pivot=pivot)
    isoftmax = ISoftmax.from_payload(payload)
    x_int = _to_int(logits, payload["input_scale"])
    assert torch.equal(isoftmax.forward_int(x_int), _golden(x_int, payload, heads, tokens))
    row_sums = _row_sums(payload, logits)
    assert 0.95 < float(row_sums.min()) and float(row_sums.max()) < 1.05


def test_build_rejects_bad_arguments():
    logits = _attention_logits(1, 8, 8, seed=13).reshape(-1, 8).numpy()
    with pytest.raises(ValueError):
        build_softmax_int_payload(np.zeros((0, 8)))
    with pytest.raises(ValueError):
        build_softmax_int_payload(logits, pivot="middle")
    with pytest.raises(ValueError):
        build_softmax_int_payload(logits, output_bits=0)
    with pytest.raises(ValueError):
        build_softmax_int_payload(logits, recip_bits=0)
    with pytest.raises(ValueError):
        build_softmax_int_payload(logits, exp_scale=1)
    with pytest.raises(ValueError):  # under-covering the row length would re-open SM-1
        build_softmax_int_payload(logits, max_tokens=4)
    with pytest.raises(ValueError):
        ISoftmax([1, 2, 3], [0], [0], [0], input_scale=1.0, output_scale=1.0)


def test_percentile_clipping_of_the_accumulator_is_not_offered():
    """The old ``percentile`` knob is gone, and must not come back.

    It clipped the accumulator range to a central percentile of the *observed*
    accumulators, which is precisely the saturation that
    ``test_isoftmax_stays_normalised_on_out_of_distribution_logits`` guards against — it
    threw away coverage the calibration data had actually demonstrated. Measured on the
    seed-4 attention fixture before removal: ``percentile=50.0`` gave a max abs error of
    0.2109 and row sums spanning [0.4941, 1.6824], against 0.0030 and [0.9804, 1.0118] at
    the default 100.0. The accumulator envelope is now theoretical, so no percentile of
    the sample could inform it anyway.
    """
    logits = _attention_logits(3, 32, 16, seed=4).reshape(-1, 32).numpy()
    with pytest.raises(TypeError):
        build_softmax_int_payload(logits, percentile=50.0)
    with pytest.raises(TypeError):  # not even the old default is silently swallowed
        build_softmax_int_payload(logits, percentile=100.0)
