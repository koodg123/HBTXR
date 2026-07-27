"""The ``ISoftmax.max_tokens`` runtime guard: it fires, it must, and it round trips.

``max_tokens`` is the one calibration input the integer softmax cannot infer at runtime
and cannot survive being wrong about. The reciprocal segments are sized for
``[exp_table[0], max_tokens * max(exp_table)]``; a longer row overruns that envelope,
saturates ``cursor_two`` and gets divided by the calibration maximum instead of by its own
sum, so it comes out un-normalised with no other symptom.

The tests are in two halves. The guard's *behaviour* — fires above, silent at and below,
survives the payload round trip — and the guard's *justification*: the same over-long row
pushed through the identical kernel with the guard defeated, measured against the 1.0 a
softmax row owes its caller. Without that second half nothing records why the raise is
there, and the next person to hit it deletes it.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from quantization.i_ops import softmax_quantize as g_softmax_quantize
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.softmax import ISoftmax
from quantization.int_calibrate_softmax import build_softmax_int_payload
from quantization.scheme import INT8

CALIBRATED_TOKENS = 16
LONG_TOKENS = 96


def _calibration_rows(tokens: int = CALIBRATED_TOKENS, seed: int = 3):
    """``[rows, tokens]`` logits with enough spread to exercise both segments."""
    torch.manual_seed(seed)
    return (torch.randn(4, tokens, tokens) * 3.0).reshape(-1, tokens).numpy()


def _payload(tokens: int = CALIBRATED_TOKENS, **kwargs) -> dict:
    return build_softmax_int_payload(_calibration_rows(tokens), **kwargs)


def _to_int(logits: torch.Tensor, input_scale: float) -> torch.Tensor:
    return torch.round(logits / input_scale).clamp(INT8.qmin, INT8.qmax).to(torch.int32)


def _flat_logits(heads: int, tokens: int, seed: int = 11, spread: float = 0.3) -> torch.Tensor:
    """Near-uniform rows — the worst case for the accumulator, hence for the envelope."""
    torch.manual_seed(seed)
    return torch.randn(heads, tokens, tokens) * spread


def _unguarded(payload: dict, tokens: int, *, dim: int = -1) -> ISoftmax:
    """The same module with the guard defeated, to measure what it prevents.

    ``max_tokens`` is guard-only state: the tables, scalars and scales all come from the
    payload and none of them read it. Raising it therefore changes nothing but the
    comparison in ``forward_int``, which is exactly the pre-guard kernel.
    """
    return ISoftmax(
        payload["scalars"],
        payload["exp_table"],
        payload["recip_table_one"],
        payload["recip_table_two"],
        input_scale=payload["input_scale"],
        output_scale=payload["output_scale"],
        dim=dim,
        max_tokens=tokens,
    )


def _row_sums(module: ISoftmax, payload: dict, logits: torch.Tensor) -> torch.Tensor:
    out = module.forward_int(_to_int(logits, payload["input_scale"])).to(torch.float64)
    return (out * payload["output_scale"]).sum(dim=-1)


# --- (a) the guard fires ------------------------------------------------------

def test_guard_rejects_a_row_longer_than_calibration():
    """A 16-token calibration must refuse a 96-token row, and say both numbers."""
    payload = _payload()
    assert payload["max_tokens"] == CALIBRATED_TOKENS
    isoftmax = ISoftmax.from_payload(payload)
    x_int = _to_int(_flat_logits(2, LONG_TOKENS), payload["input_scale"])

    with pytest.raises(ValueError) as excinfo:
        isoftmax.forward_int(x_int)
    message = str(excinfo.value)
    assert str(LONG_TOKENS) in message, f"the actual row length is missing from: {message}"
    assert str(CALIBRATED_TOKENS) in message, f"the calibrated limit is missing from: {message}"
    # both numbers alone only say what went wrong; the caller also needs the way out,
    # which is a recalibration and not anything they can do at the call site
    assert "recalibrate" in message.lower(), f"no remedy in: {message}"
    assert "build_softmax_int_payload" in message, f"the remedy names no entry point: {message}"


def test_guard_fires_through_every_public_entry_point():
    """``forward_int`` is not the only door — QTensor and float callers must be stopped too."""
    payload = _payload()
    isoftmax = ISoftmax.from_payload(payload)
    logits = _flat_logits(2, LONG_TOKENS)

    with pytest.raises(ValueError):
        isoftmax.forward_qtensor(QTensor.quantize(logits, payload["input_scale"], 0.0, INT8))
    with pytest.raises(ValueError):
        isoftmax(QTensor.quantize(logits, payload["input_scale"], 0.0, INT8))
    with pytest.raises(ValueError):
        isoftmax(logits)


def test_guard_measures_the_reduction_axis_not_the_last_one():
    """With ``dim=1`` the row length is ``shape[1]``; reading ``shape[-1]`` would pass.

    The input is ``[2, 96, 16]``: over ``dim=1`` the rows are 96 long and must be refused,
    over ``dim=-1`` they are 16 long and must not be. A guard hard-coded to the last axis
    gets both of these backwards.
    """
    payload = _payload()
    x_int = _to_int(torch.randn(2, LONG_TOKENS, CALIBRATED_TOKENS) * 0.3, payload["input_scale"])

    with pytest.raises(ValueError, match=r"dim 1"):
        ISoftmax.from_payload(payload, dim=1).forward_int(x_int)
    ISoftmax.from_payload(payload).forward_int(x_int)  # same tensor over the last axis: fine


# --- (b) off-by-one in the right direction ------------------------------------

@pytest.mark.parametrize("calibrated", [4, 16, 33])
def test_guard_is_silent_at_exactly_max_tokens_and_fires_one_past_it(calibrated):
    """``max_tokens`` is inclusive: it is the longest row the envelope *covers*."""
    payload = _payload(calibrated)
    assert payload["max_tokens"] == calibrated  # defaults to the calibration row length
    isoftmax = ISoftmax.from_payload(payload)
    scale = payload["input_scale"]

    at_limit = _to_int(torch.zeros(2, 3, calibrated), scale)
    out = isoftmax.forward_int(at_limit)  # must not raise
    assert out.shape == at_limit.shape

    with pytest.raises(ValueError):
        isoftmax.forward_int(_to_int(torch.zeros(2, 3, calibrated + 1), scale))


# --- (c) why the guard exists -------------------------------------------------

@pytest.mark.parametrize(
    "spread,label,worst_floor,every_row_floor",
    # measured: flat rows sum to a flat 4.5176 everywhere (mean 4.5176); realistic logits
    # span [1.4000, 2.9059] with mean 2.3045. The floors sit just under each measurement.
    [(0.0, "flat", 4.0, 4.0), (0.3, "realistic", 2.5, 1.3)],
)
def test_the_unguarded_kernel_really_does_lose_normalisation(spread, label, worst_floor, every_row_floor):
    """Defeat the guard and measure the damage: the row stops summing to 1.

    Measured on this fixture (16-token calibration, 96-token rows): every flat row sums to
    4.5176, and realistic logits span [1.4000, 2.9059]. Those are attention weight vectors
    carrying 1.4-4.5x the probability mass they should, with nothing in the output to
    signal it — the downstream ``attn @ V`` just comes out scaled by a per-row factor
    nobody chose. This is the silent failure the raise trades for a loud one; if this test
    ever goes green with row sums near 1.0 then the guard has become unnecessary and
    should be argued about rather than quietly kept.
    """
    payload = _payload()
    logits = _flat_logits(2, LONG_TOKENS, spread=spread) if spread else torch.zeros(2, LONG_TOKENS, LONG_TOKENS)
    sums = _row_sums(_unguarded(payload, LONG_TOKENS), payload, logits)

    worst = float(sums.max())
    assert worst > worst_floor, f"{label} 96-token row sums peak at {worst:.4f}, not far enough from 1.0"
    # and it is not one outlier row: *every* row is over, so no downstream averaging saves it
    assert float(sums.min()) > every_row_floor, f"{label} min row sum {float(sums.min()):.4f}"


def test_recalibrating_at_the_larger_max_tokens_is_the_documented_cure():
    """The message tells the caller to recalibrate; that must actually work.

    Same rows, same logits, only ``max_tokens`` raised to 96: the guard goes quiet and the
    row sums come back inside the band. Measured worst 1.0549 (the residual is the uint8
    output step, not the reciprocal: 96 probabilities of ~1/96 each rounded to 1/255).
    """
    wide = _payload(CALIBRATED_TOKENS, max_tokens=LONG_TOKENS)
    assert wide["max_tokens"] == LONG_TOKENS
    logits = _flat_logits(2, LONG_TOKENS)
    sums = _row_sums(ISoftmax.from_payload(wide), wide, logits)
    assert float(sums.min()) > 0.90, f"min row sum {float(sums.min()):.4f}"
    assert float(sums.max()) < 1.10, f"max row sum {float(sums.max()):.4f}"


# --- (d) payload round trip ---------------------------------------------------

@pytest.mark.parametrize("calibrated", [8, 16, 96])
def test_max_tokens_survives_the_payload_round_trip(calibrated):
    payload = _payload(8, max_tokens=calibrated)
    assert ISoftmax.from_payload(payload).max_tokens == calibrated
    assert payload["metrics"]["max_tokens"] == calibrated  # the report still agrees


@pytest.mark.parametrize("calibrated", [8, 16, 96])
@pytest.mark.parametrize("recip_entries", [32, 64, 256])
def test_a_payload_without_the_field_falls_back_to_the_derived_bound(calibrated, recip_entries):
    """A legacy payload (or a manifest written before the field existed) still loads.

    The fallback reads the emitted segment-two index instead of the calibration, and it is
    *loose*: it must never claim a tighter bound than was calibrated, or the guard would
    start rejecting rows the tables cover. Measured derived values (identical at all three
    reciprocal widths, since the PoT shift is what rounds up): 8->10, 16->20, 96->144.
    """
    payload = _payload(8, max_tokens=calibrated, recip_entries=recip_entries)
    legacy = {k: v for k, v in payload.items() if k != "max_tokens"}
    assert "max_tokens" not in legacy

    derived = ISoftmax.from_payload(legacy).max_tokens
    assert derived >= calibrated, (
        f"derived bound {derived} is tighter than the calibrated {calibrated}: the fallback "
        "would reject rows the reciprocal segments actually resolve"
    )
    # the fallback is a bound on the tables, so it must not admit a row the segments
    # cannot reach at all — the accumulator of a `derived`-token flat row must still be
    # inside segment two's covered span.
    b2_two, s2_two, bound2_two = payload["scalars"][8:11]
    covered = ((bound2_two + 1) << s2_two) - b2_two - 1
    assert derived * max(payload["exp_table"]) <= covered


@pytest.mark.parametrize("calibrated,slack", [(8, 1.15), (16, 1.15), (96, 1.4)])
def test_the_derived_fallback_is_loose_which_is_why_it_is_second_class(calibrated, slack):
    """It over-admits by 25-50%, so it must never be preferred to a stored value.

    ``make_pot_index_params`` rounds the segment bin width up to a power of two, so the
    emitted table always reaches past the envelope it was sized for. Measured over-admission
    (derived / calibrated): 8->10 (1.25x), 16->20 (1.25x), 96->144 (1.5x). Those extra rows
    are covered but under-resolved — worst flat row sum 1.0039 at the calibrated 16 tokens
    against 1.0196 at the derived 20 — which is a gradual degradation with no cliff to
    detect, and exactly why a legacy payload gets this bound and a fresh one does not.
    """
    payload = _payload(8, max_tokens=calibrated)
    legacy = {k: v for k, v in payload.items() if k != "max_tokens"}
    derived = ISoftmax.from_payload(legacy).max_tokens
    assert derived > calibrated * slack, (
        f"derived {derived} vs calibrated {calibrated}: the fallback has become tight, so "
        "the comment calling it loose is now wrong"
    )


def test_an_explicit_max_tokens_beats_the_payload_field():
    """``__init__`` is the override point — ``from_payload`` is a convenience over it."""
    payload = _payload(CALIBRATED_TOKENS, max_tokens=CALIBRATED_TOKENS)
    assert _unguarded(payload, 4096).max_tokens == 4096
    with pytest.raises(ValueError):
        _unguarded(payload, 0)


# --- (e) nothing else moved ---------------------------------------------------

def test_a_normal_length_row_is_bit_exact_and_untouched():
    """The guard is a comparison, not a transform: in-range output must not change.

    Checked against the pure-Python golden rather than against a stored tensor, so this
    also catches the guard accidentally consuming or reshaping its input.
    """
    heads, tokens = 2, CALIBRATED_TOKENS
    torch.manual_seed(4)
    q = torch.randn(heads, tokens, 16)
    k = torch.randn(heads, tokens, 16)
    logits = (q @ k.transpose(-2, -1)) / math.sqrt(16)
    payload = _payload()
    x_int = _to_int(logits, payload["input_scale"])

    out = ISoftmax.from_payload(payload).forward_int(x_int)
    golden = torch.tensor(
        g_softmax_quantize(
            x_int.reshape(-1).tolist(),
            payload["scalars"],
            payload["exp_table"],
            payload["recip_table_one"],
            payload["recip_table_two"],
            tokens=tokens,
            heads=heads,
        ),
        dtype=torch.int64,
    ).reshape(heads, tokens, tokens)
    assert torch.equal(out, golden)
    # and the guarded module agrees element-wise with the unguarded one at this length
    assert torch.equal(out, _unguarded(payload, LONG_TOKENS).forward_int(x_int))


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 15, 16])
def test_rows_shorter_than_calibration_are_accepted_and_stay_normalised(tokens):
    """A short row is safe, not merely tolerated — it lands inside the envelope.

    Every row contributes one element at its own max, so the accumulator's lower bound
    ``exp_table[0]`` is attained at any length >= 1; a short row therefore sits strictly
    inside the covered range rather than under it. Measured row sums over these lengths
    with a 16-token calibration: [0.9922, 1.0196]. The real cost of a mismatch runs the
    other way, on the calibration side — an over-wide ``max_tokens`` spends reciprocal
    bins on accumulators that never occur (with a 32-entry reciprocal, 16-token rows at
    ``max_tokens=96`` measure 4.55e-02 max abs error against 1.81e-02 at
    ``max_tokens=16``) — which is why the guard is one-sided but the docstring is not.
    """
    payload = _payload()
    for logits in (_flat_logits(2, tokens), torch.zeros(2, tokens, tokens)):
        sums = _row_sums(ISoftmax.from_payload(payload), payload, logits)
        assert float(sums.min()) > 0.95, f"{tokens} tokens: min row sum {float(sums.min()):.4f}"
        assert float(sums.max()) < 1.05, f"{tokens} tokens: max row sum {float(sums.max()):.4f}"


# --- the guard must survive the export round trip --------------------------

def test_max_tokens_survives_export_and_replay(tmp_path):
    """A manifest that drops max_tokens silently loosens the guard on every reload."""
    import numpy as np
    from torch import nn

    from quantization.export import (
        export_integer_model, load_integer_manifest, replay_softmax, verify_export,
    )

    torch.manual_seed(0)
    tokens = 8
    rows = (torch.randn(6, tokens, tokens) * 2.0).reshape(-1, tokens).numpy()
    payload = build_softmax_int_payload(rows, exp_entries=64, recip_entries=64)
    module = ISoftmax.from_payload(payload)
    assert module.max_tokens == tokens

    holder = nn.Module()
    holder.sm = module
    out = tmp_path / "dump"
    manifest = export_integer_model(holder, out)

    entry = manifest["modules"]["sm"]
    assert entry["max_tokens"] == tokens, "the calibrated bound must reach the manifest"
    assert load_integer_manifest(out)["modules"]["sm"]["max_tokens"] == tokens

    # rebuilding from the manifest must reproduce the bound, not the looser derived one
    rebuilt = ISoftmax.from_payload({**payload, "metrics": {"max_tokens": entry["max_tokens"]}})
    assert rebuilt.max_tokens == tokens

    # the replay mirrors the module's refusal instead of returning an un-normalised row
    with pytest.raises(ValueError, match="max_tokens"):
        replay_softmax(entry, np.zeros((2, tokens * 4), dtype=np.float32))

    # and verification of a short-row softmax must not crash on the probe width
    ok, diff = verify_export(holder, out)
    assert ok, f"short-row softmax failed to verify (diff {diff})"
