"""D3: the TWO-SEGMENT rsqrt index of the integer LayerNorm.

``i_ops.layernorm_quantize``'s cursor is one unbranched line over a LOG-domain function,
so a variance dynamic range much wider than the table depth cannot be served to a few
percent however ``(b, s1)`` are fitted. ``i_ops.layernorm_quantize_segmented`` splits the
range in two and gives each half its own index; this file is the evidence that the split
is (a) implemented bit-exactly, (b) worth its silicon, and (c) actually verified end to
end rather than assumed.

Three rules, inherited from ``test_int_layernorm.py`` and one of them added here:

* nothing is measured against a tolerance the calibrator chose;
* every claim in the prose has a test that dies when the claim is removed;
* **a comparison test computes BOTH numbers it compares.** The one-segment baseline below
  is never a hard-coded constant from a scoping run — it is recalibrated from the same
  fixture in the same process, so the ratio cannot rot as the calibrator changes.

And one thing this file refuses to overstate: the ~9x is on the RMS relative *rsqrt*
error, which is what the index controls. End to end the segmented kernel is ~1.6x better
on these fixtures, because once the rsqrt error drops to ~1% the remaining error is the
int8 input grid and the 8-bit output requant — see
``test_two_segments_reach_the_accuracy_floor_of_the_datapath``, which measures that floor
instead of leaving the reader to assume the headline carries through.
"""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn

from quantization.export import (
    MANIFEST_NAME,
    _probe,
    export_integer_model,
    load_integer_manifest,
    verify_export,
    verify_export_report,
)
from quantization.i_ops import layernorm_quantize, layernorm_quantize_segmented
from quantization.ilayers.layernorm import ILayerNorm
from quantization.ilayers.qtensor import QTensor
from quantization.int_calibrate import calibrate_int_layernorm, integer_row_statistics
from quantization.scheme import INT8, qrange


# --- fixtures (own generators, never the global RNG) --------------------------

CHANNELS = 48
SEEDS = (0, 1, 2, 3)


def _layernorm(channels: int, seed: int) -> nn.LayerNorm:
    gen = torch.Generator().manual_seed(1000 + seed)
    ln = nn.LayerNorm(channels)
    with torch.no_grad():
        ln.weight.copy_(1.0 + 0.15 * torch.randn(channels, generator=gen))
        ln.bias.copy_(0.1 * torch.randn(channels, generator=gen))
    return ln


def _heteroscedastic(channels: int, seed: int, rows: int = 512, ratio: float = 32.0) -> torch.Tensor:
    """Rows whose scale sweeps ``ratio``:1 log-uniformly — the shape that breaks one index.

    Identical to the fixture in ``test_int_layernorm.py`` on purpose: the one-segment
    numbers this file compares against were measured on exactly these rows.
    """
    gen = torch.Generator().manual_seed(seed)
    row_scale = torch.logspace(0.0, math.log10(ratio), rows).reshape(rows, 1) * (1.5 / ratio)
    return torch.randn(rows, channels, generator=gen) * row_scale


def _row_var_sum(x: torch.Tensor, payload: dict) -> np.ndarray:
    """The integer per-row ``var_sum`` the deployed kernel sees, re-derived locally."""
    channels = payload["channels"]
    c_1_m, c_1_s = payload["scalars"][0], payload["scalars"][1]
    x_int = torch.round(x.reshape(-1, channels) / payload["input_scale"])
    x_int = x_int.clamp(INT8.qmin, INT8.qmax).to(torch.int64)
    mean = (x_int.sum(-1, keepdim=True) * c_1_m + (1 << (c_1_s - 1))) >> c_1_s
    diff = x_int - mean
    return (diff * diff).sum(-1).numpy().astype(np.int64)


def _rsqrt_rel_error(x: torch.Tensor, payload: dict) -> np.ndarray:
    """Signed relative error of the rsqrt the kernel looks up, for EITHER index shape.

    A local transcription of the lookup — including the branch on the UNCLAMPED
    segment-one cursor — so the accuracy claims below never lean on the calibrator's own
    ``metrics``.
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


def _rms(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(err) ** 2)))


def _golden(x_int: torch.Tensor, payload: dict) -> torch.Tensor:
    flat = x_int.reshape(-1).tolist()
    if len(payload["scalars"]) == 7:
        out = layernorm_quantize(flat, payload["scalars"], payload["lnw"], payload["lnb"],
                                 payload["rsqrt_table"])
    else:
        out = layernorm_quantize_segmented(flat, payload["scalars"], payload["lnw"],
                                           payload["lnb"], payload["rsqrt_table"],
                                           payload["rsqrt_table_two"])
    return torch.tensor(out, dtype=torch.int32).reshape(x_int.shape)


def _var_sum_of(x_int: torch.Tensor, payload: dict) -> torch.Tensor:
    _mean, var_sum = integer_row_statistics(x_int.to(torch.int64),
                                            payload["scalars"][0], payload["scalars"][1])
    return var_sum


def _both_segments_used(x_int: torch.Tensor, module: ILayerNorm, payload: dict) -> tuple[int, int]:
    """``(rows in segment one, rows in segment two)`` for this input — the fixture's proof."""
    mask = module.segment_mask(_var_sum_of(x_int, payload))
    return int((~mask).sum()), int(mask.sum())


# --- (a) BIT-EXACTNESS vs the new golden --------------------------------------

def _spanning_int_rows(payload: dict, seed: int, rows: int = 24) -> torch.Tensor:
    """Integer rows whose ``var_sum`` sweeps from below segment one to past segment two.

    Built in the VARIANCE coordinate, because that is what the index reads: a row of
    uniform random codes always has about the same variance and would address a handful of
    neighbouring entries however many rows were drawn. Each row is a normal draw rescaled
    to a target ``var_sum``, and the targets sweep the whole declared index range and
    overshoot both ends so the clamps are exercised too.
    """
    channels = payload["channels"]
    _, _, b, s1, bound = payload["scalars"][:5]
    lo = max(-b, 1)
    hi = ((bound + 1) << s1) - b - 1
    if len(payload["scalars"]) == 10:
        b_two, s1_two, bound_two = payload["scalars"][7:]
        hi = ((bound_two + 1) << s1_two) - b_two - 1
    targets = np.geomspace(max(lo / 8.0, 1.0), hi * 4.0, rows)

    gen = torch.Generator().manual_seed(seed)
    row = torch.randn(rows, channels, generator=gen).to(torch.float64)
    row = row - row.mean(dim=-1, keepdim=True)
    row = row / row.std(dim=-1, keepdim=True).clamp_min(1e-12)
    std = torch.from_numpy(np.sqrt(targets / channels)).reshape(-1, 1)
    return (row * std).round().clamp(INT8.qmin, INT8.qmax).to(torch.int32)


@pytest.mark.parametrize("seed", SEEDS)
def test_segmented_kernel_is_bit_exact_vs_the_golden(seed):
    """``ILayerNorm`` == ``layernorm_quantize_segmented``, element-wise, on both branches.

    The fixture is PROVEN to exercise both segments (and both clamps of both segments)
    rather than assumed to: if a future calibration or a future fit collapsed every row
    into one branch, the assertions below fail instead of the test quietly becoming a
    one-segment test that passes.
    """
    ln = _layernorm(CHANNELS, seed)
    payload = calibrate_int_layernorm(ln, _heteroscedastic(CHANNELS, seed))
    assert len(payload["scalars"]) == 10
    module = ILayerNorm.from_payload(payload)

    x_int = _spanning_int_rows(payload, seed=200 + seed)
    in_one, in_two = _both_segments_used(x_int, module, payload)
    assert in_one >= 3 and in_two >= 3, (
        f"fixture exercises {in_one} segment-one and {in_two} segment-two rows — "
        "this would be a one-branch test")

    var_sum = _var_sum_of(x_int, payload).reshape(-1)
    _, _, b, s1, bound = payload["scalars"][:5]
    b_two, s1_two, bound_two = payload["scalars"][7:]
    cursor_one = torch.bitwise_right_shift(var_sum + b, s1)
    cursor_two = torch.bitwise_right_shift(var_sum + b_two, s1_two)
    assert int((cursor_one < 0).sum()) >= 1, "segment one's LOW clamp is not exercised"
    assert int((cursor_one > bound).sum()) >= 3, "segment one's overflow is not exercised"
    assert int((cursor_two > bound_two).sum()) >= 1, "segment two's HIGH clamp is not exercised"

    out = module(QTensor(x_int, scale=payload["input_scale"], zero_point=0.0, dtype=INT8))
    golden = _golden(x_int, payload)
    assert torch.equal(out.int_data, golden), (
        f"{(out.int_data != golden).sum().item()} / {golden.numel()} elements differ")
    # negatives must be present, else the arithmetic (flooring) shift is untested
    assert bool((out.int_data < 0).any()) and bool((golden < 0).any())


def test_segmented_kernel_is_bit_exact_on_hand_made_scalars():
    """Bit-exactness on constants nobody calibrated, with every branch position PINNED.

    A random or geometric fixture reaches both segments but never lands *exactly* on the
    branch boundary — the cursor equal to ``bound``, where ``>`` and ``>=`` disagree — and
    a kernel with that comparison flipped survives it. Here ``var_sum`` is exact
    arithmetic, so each position can be hit deliberately: an 8-channel row
    ``[a, -a, a, -a, b, -b, b, -b]`` has integer mean 0 and ``var_sum == 4a^2 + 4b^2``.

    The eight rows are, in order: below segment one's range (low clamp); its entry 0
    exactly; its interior; ``cursor == bound`` EXACTLY (segment one's last entry — the row
    that kills ``>=``); segment two's entry 0 exactly; its interior, twice; and past its
    top (high clamp).

    The tables are ``A / sqrt(bin centre)``, i.e. physically sensible rsqrt values, so
    ``diff * rsqrt`` stays roughly constant across rows and one ``s2`` avoids saturating
    the output — a saturated output is exactly how a wrong lookup hides.
    """
    channels = 8
    c_1_m, c_1_s = 1, 3                                   # 1/8 exactly
    b, s1, bound = -100, 4, 7                             # segment one: var_sum 100..227
    b_two, s1_two, bound_two = -228, 7, 7                 # segment two: 228..1251
    s2 = 14
    scalars = [c_1_m, c_1_s, b, s1, bound, s2, 8, b_two, s1_two, bound_two]

    def _table(lo: int, width: int) -> list[int]:
        centres = [lo + width * i + (width - 1) / 2.0 for i in range(bound + 1)]
        return [int(round(30000.0 / math.sqrt(c))) for c in centres]

    table_one, table_two = _table(100, 1 << s1), _table(228, 1 << s1_two)
    assert len(set(table_one)) == len(set(table_two)) == 8, "entries must be distinguishable"
    assert min(table_one) > max(table_two), "segment two holds the smaller rsqrt values"
    gen = torch.Generator().manual_seed(5150)
    lnw = torch.randint(-127, 128, (channels,), generator=gen).tolist()
    lnb = torch.randint(-20000, 20000, (channels,), generator=gen).tolist()

    pairs = [(1, 1), (4, 3), (5, 3), (7, 2), (7, 3), (10, 5), (12, 6), (15, 10)]
    x_int = torch.tensor([[a, -a, a, -a, c, -c, c, -c] for a, c in pairs], dtype=torch.int32)
    module = ILayerNorm(scalars, table_one, lnw, lnb, input_scale=1.0, output_scale=1.0,
                        rsqrt_table_two=table_two)

    var_sum = _var_sum_of(x_int, {"scalars": scalars})
    assert var_sum.reshape(-1).tolist() == [4 * a * a + 4 * c * c for a, c in pairs]
    cursor = torch.bitwise_right_shift(var_sum + b, s1).reshape(-1)
    cursor_two = torch.bitwise_right_shift(var_sum + b_two, s1_two).reshape(-1)
    assert cursor.tolist() == [-6, 0, 2, 7, 8, 25, 38, 75]
    assert int((cursor == bound).sum()) == 1, "the exact branch boundary is not exercised"
    assert int((cursor_two[cursor > bound] == 0).sum()) == 1, "segment two's entry 0 is not read"
    assert int((cursor_two > bound_two).sum()) == 1, "segment two's high clamp is not exercised"
    assert int(module.segment_mask(var_sum).sum()) == 4, "both branches must be live here"

    out = module(QTensor(x_int, scale=1.0, zero_point=0.0, dtype=INT8))
    golden = torch.tensor(
        layernorm_quantize_segmented(x_int.reshape(-1).tolist(), scalars, lnw, lnb,
                                     table_one, table_two),
        dtype=torch.int32).reshape(x_int.shape)
    assert torch.equal(out.int_data, golden)
    qmax = qrange(8, signed=True)[1]
    assert float((golden.abs() >= qmax).float().mean()) < 0.25, (
        "most of the output saturates, so a wrong lookup could hide in the clamp")

    # and the case discriminates: the one-segment golden on the same rows differs
    one_seg = torch.tensor(
        layernorm_quantize(x_int.reshape(-1).tolist(), scalars[:7], lnw, lnb, table_one),
        dtype=torch.int32).reshape(x_int.shape)
    assert not torch.equal(one_seg, golden), (
        "the fixture cannot tell the segmented kernel from the one-segment one")


def test_golden_rejects_a_mismatched_scalar_count():
    """The two goldens are distinct kernels, not one kernel with an optional tail."""
    with pytest.raises(ValueError):
        layernorm_quantize_segmented([0, 0], [1, 1, 0, 0, 3, 2, 8], [1, 1], [0, 0], [1] * 4, [1] * 4)
    with pytest.raises(ValueError):
        layernorm_quantize([0, 0], [1, 1, 0, 0, 3, 2, 8, 0, 0, 3], [1, 1], [0, 0], [1] * 4)


# --- (b) the accuracy win, with BOTH numbers measured here --------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_two_segments_beat_one_on_the_same_data(seed):
    """Measured relative rsqrt error, 256 entries per segment, same fixture:

                    1 segment                          2 segments
        RMS         0.0935 / 0.0887 / 0.0903 / 0.0941  0.0104 / 0.0086 / 0.0089 / 0.0099
        p99         0.3956 / 0.3379 / 0.3386 / 0.3139  0.0340 / 0.0294 / 0.0286 / 0.0342
        max         0.4442 / 0.3624 / 0.4167 / 0.4286  0.0727 / 0.0471 / 0.0562 / 0.0609

    i.e. 9.0x / 10.3x / 10.1x / 9.5x on the RMS and 6.1x / 7.7x / 7.4x / 7.0x on the worst
    row. The tail is quoted as well as the bulk because a piecewise index could in
    principle buy its RMS by abandoning the extremes, and it does not.

    Both numbers are recalibrated here, so this asserts a RATIO between two live
    measurements rather than a remembered constant.
    """
    ln = _layernorm(CHANNELS, seed)
    x = _heteroscedastic(CHANNELS, seed)
    one = calibrate_int_layernorm(ln, x, segments=1, entries=256)
    two = calibrate_int_layernorm(ln, x, segments=2, entries=256)

    # the fixture must really be wide-spread, else there is nothing to resolve
    assert two["metrics"]["var_sum_dynamic_range"] > 500.0
    err_one, err_two = np.abs(_rsqrt_rel_error(x, one)), np.abs(_rsqrt_rel_error(x, two))
    rms_one, rms_two = _rms(err_one), _rms(err_two)
    assert rms_two < rms_one / 5.0, f"1 segment {rms_one:.4f} vs 2 segments {rms_two:.4f}"
    assert rms_two < 0.015, f"segmented RMS relative rsqrt error {rms_two:.4f}"
    # the tail, not only the bulk
    assert err_two.max() < err_one.max() / 4.0, (
        f"worst row: 1 segment {err_one.max():.4f} vs 2 segments {err_two.max():.4f}")
    assert float(np.quantile(err_two, 0.99)) < 0.05


@pytest.mark.parametrize("seed", SEEDS)
def test_two_segments_win_at_half_the_table_memory(seed):
    """THE HEADLINE: two 64-entry segments (128 words) beat one 256-entry index (256).

    Measured on these fixtures:

        1 x 256 entries = 256 int16 words   0.0935 / 0.0887 / 0.0903 / 0.0941
        2 x  64 entries = 128 int16 words   0.0280 / 0.0365 / 0.0354 / 0.0274
                                            -> 3.34x / 2.43x / 2.55x / 3.43x

    So the segmented index is better on accuracy AND on table cost at the same time,
    which is the whole reason it is worth a second golden kernel. The word counts are
    asserted from the emitted tables, not assumed from the arguments.
    """
    ln = _layernorm(CHANNELS, seed)
    x = _heteroscedastic(CHANNELS, seed)
    big_one = calibrate_int_layernorm(ln, x, segments=1, entries=256)
    small_two = calibrate_int_layernorm(ln, x, segments=2, entries=64)

    words_one = len(big_one["rsqrt_table"])
    words_two = len(small_two["rsqrt_table"]) + len(small_two["rsqrt_table_two"])
    assert words_one == 256 and words_two == 128
    assert words_two < words_one, "the memory claim is the point of this test"

    rms_one = _rms(_rsqrt_rel_error(x, big_one))
    rms_two = _rms(_rsqrt_rel_error(x, small_two))
    assert rms_two < rms_one / 2.0, (
        f"2x64 ({words_two} words) {rms_two:.4f} vs 1x256 ({words_one} words) {rms_one:.4f}")


@pytest.mark.parametrize("seed", SEEDS)
def test_two_segments_win_again_at_equal_table_memory(seed):
    """Same 256 words either way: 2 x 128 entries against 1 x 256.

    Measured 0.0184 / 0.0167 / 0.0166 / 0.0166 against 0.0935 / 0.0887 / 0.0903 / 0.0941,
    i.e. 5.07x / 5.32x / 5.45x / 5.67x. This is the comparison to quote when the table
    budget is fixed; the default (``entries=256`` per segment) spends 512 words to buy the
    ~9x instead.
    """
    ln = _layernorm(CHANNELS, seed)
    x = _heteroscedastic(CHANNELS, seed)
    one = calibrate_int_layernorm(ln, x, segments=1, entries=256)
    two = calibrate_int_layernorm(ln, x, segments=2, entries=128)
    assert len(one["rsqrt_table"]) == len(two["rsqrt_table"]) + len(two["rsqrt_table_two"]) == 256

    rms_one, rms_two = _rms(_rsqrt_rel_error(x, one)), _rms(_rsqrt_rel_error(x, two))
    assert rms_two < rms_one / 3.0, f"equal memory: 1x256 {rms_one:.4f} vs 2x128 {rms_two:.4f}"


@pytest.mark.parametrize("seed", SEEDS)
def test_two_segments_reach_the_accuracy_floor_of_the_datapath(seed):
    """What the ~9x on the rsqrt is worth END TO END — measured, not extrapolated.

    Against the float LayerNorm on the SAME int8-quantized input (so only the kernel is
    being judged), relative error over these fixtures:

        1 segment          0.1184 / 0.1115 / 0.1140 / 0.1191
        2 segments         0.0744 / 0.0706 / 0.0706 / 0.0763
        floor, 2 x 16384   0.0737 / 0.0700 / 0.0701 / 0.0757   (rsqrt RMS 2e-05)

    So the honest end-to-end figure is ~1.6x, not ~9x: two segments land within ~1% of the
    floor a practically-exact rsqrt table would reach, which means the index has stopped
    being the dominant error and the int8 input grid and 8-bit output requant now are.
    That is the useful claim — the rsqrt is no longer worth spending on — and it is a
    stronger statement than the ratio, so this test asserts it directly.
    """
    ln = _layernorm(CHANNELS, seed)
    x = _heteroscedastic(CHANNELS, seed)
    one = calibrate_int_layernorm(ln, x, segments=1)
    two = calibrate_int_layernorm(ln, x, segments=2)
    floor = calibrate_int_layernorm(ln, x, segments=2, entries=16384)

    qt = QTensor.quantize(x, two["input_scale"], 0.0, INT8)
    with torch.no_grad():
        ref = ln(qt.dequantize())
        rel = {name: float((ILayerNorm.from_payload(p)(qt).dequantize() - ref).norm() / ref.norm())
               for name, p in (("one", one), ("two", two), ("floor", floor))}

    assert _rms(_rsqrt_rel_error(x, floor)) < 1e-3, "the floor payload is not a floor"
    assert rel["two"] < rel["one"], f"segmented {rel['two']:.4f} vs one segment {rel['one']:.4f}"
    assert rel["two"] < 1.05 * rel["floor"], (
        f"segmented {rel['two']:.4f} is not within 5% of the datapath floor {rel['floor']:.4f}")
    assert rel["one"] > 1.3 * rel["floor"], (
        "the one-segment index is no longer the dominant error on this fixture, so this "
        "comparison has stopped meaning anything")


@pytest.mark.parametrize("seed", SEEDS)
def test_segmented_tables_sample_the_bin_interior(seed):
    """Both segments' entries must be UNBIASED across their bin, as one segment's are.

    ``cursor`` floors, so an entry evaluated at its bin's LOWER edge returns the largest
    rsqrt in the bin for every row in it and biases the whole layer high. 16 entries per
    segment on purpose: the default index is far too fine for a within-bin bias to be
    visible, and a test that cannot see the effect it names is not a test of it.
    """
    ln = _layernorm(CHANNELS, seed)
    x = _heteroscedastic(CHANNELS, seed)
    payload = calibrate_int_layernorm(ln, x, segments=2, entries=16)

    err = _rsqrt_rel_error(x, payload)
    assert _rms(err) > 0.04, "bins too fine here to detect any bias"
    assert abs(float(np.mean(err))) < 0.015, f"mean signed rsqrt error {float(np.mean(err)):+.4f}"


def test_both_segments_share_one_rsqrt_scale_and_still_fit_int16():
    """The design claim that removes the need for a second ``(b3, s3)``.

    Softmax needs a requant pair per reciprocal segment because each table carries its own
    numerator. LayerNorm does not: both rsqrt tables are quantized at ONE power-of-two
    scale, so the single ``>> s2`` after the affine is segment-independent. The risk that
    buys is int16 headroom — segment two holds the SMALLEST rsqrt values, so a shared scale
    could squeeze them to a handful of counts. Measured here: the smallest entry across
    both tables stays in the hundreds, and quantizing at the shared scale costs nothing
    against the float tables (agreement to ~1e-5 of RMS).
    """
    for seed in SEEDS:
        ln = _layernorm(CHANNELS, seed)
        x = _heteroscedastic(CHANNELS, seed)
        payload = calibrate_int_layernorm(ln, x, segments=2)

        assert math.log2(1.0 / payload["rsqrt_scale"]).is_integer(), "rsqrt_scale is not PoT"
        assert payload["metrics"]["rsqrt_min_entry"] >= 64, (
            f"seed {seed}: smallest int16 rsqrt entry is "
            f"{payload['metrics']['rsqrt_min_entry']} — the shared scale is now the "
            "bottleneck and each segment would need its own")
        qmax = qrange(16, signed=True)[1]
        for key in ("rsqrt_table", "rsqrt_table_two"):
            table = np.asarray(payload[key], dtype=np.int64)
            assert int(table.max()) <= qmax and int(table.min()) >= 1

        # the int16 tables must reproduce what unquantized float tables would do
        int_rms = _rms(_rsqrt_rel_error(x, payload))
        assert abs(int_rms - payload["metrics"]["rsqrt_rel_rms"]) < 1e-4


def test_segment_two_starts_exactly_where_segment_one_overflows():
    """The split follows the softmax precedent, and that is checkable arithmetic.

    The golden branches on segment one's UNCLAMPED cursor, so the pivot the hardware uses
    is ``((bound + 1) << s1) - b`` and nothing else. Segment two's offset must equal it: a
    lower offset wastes entries on variances segment one already resolves, a higher one
    leaves a band of ``var_sum`` that lands on segment two's clamped entry 0.
    """
    for seed in SEEDS:
        payload = calibrate_int_layernorm(_layernorm(CHANNELS, seed),
                                          _heteroscedastic(CHANNELS, seed))
        _, _, b, s1, bound = payload["scalars"][:5]
        b_two, _s1_two, _bound_two = payload["scalars"][7:]
        threshold = ((bound + 1) << s1) - b
        assert -b_two == threshold == payload["metrics"]["segment_threshold"]
        # and it must be a real split: both sides must carry rows
        rows_two = payload["metrics"]["rows_segment_two"]
        assert 0 < rows_two < payload["metrics"]["rows"], (
            f"seed {seed}: {rows_two} of {payload['metrics']['rows']} rows in segment two")


def test_segment_two_covers_the_physical_envelope_when_no_row_reaches_it():
    """A homoscedastic fixture leaves segment two with no data — it must still be useful.

    With nothing observed above the threshold there is nothing to fit, so segment two is
    sized to the largest ``var_sum`` the int8 input port can physically produce
    (``channels * (qmax - qmin)^2 / 4``) — the data-free envelope, the same move
    ``int_calibrate_softmax`` makes for its accumulator. The payoff is real: a row far
    above the calibration range gets a genuine rsqrt from segment two, where a one-segment
    index would pin it to the last entry of its only table. Measured below on such a row.
    """
    channels = 32
    ln = _layernorm(channels, 9)
    gen = torch.Generator().manual_seed(909)
    # A tight band around a large offset: ``input_scale`` is set by the offset, so every
    # row quantizes to nearly the same codes and the whole var_sum distribution fits
    # inside a 256-entry segment one at shift 0. Nothing can overflow it.
    x = 0.5 + 0.002 * torch.randn(400, channels, generator=gen)
    payload = calibrate_int_layernorm(ln, x, segments=2)
    assert payload["metrics"]["rows_segment_two"] == 0, (
        f"fixture is not homoscedastic enough: {payload['metrics']['rows_segment_two']} rows "
        "already overflow segment one")

    _, _, b, s1, bound = payload["scalars"][:5]
    b_two, s1_two, bound_two = payload["scalars"][7:]
    ceiling = channels * (INT8.qmax - INT8.qmin) ** 2 // 4
    hi_two = -b_two + ((bound_two + 1) << s1_two) - 1
    assert hi_two >= ceiling, f"segment two tops out at {hi_two}, below the envelope {ceiling}"

    # an out-of-distribution row: variance far above anything calibration saw
    ood = torch.full((1, channels), 0.0)
    ood[0, ::2], ood[0, 1::2] = 80.0 * payload["input_scale"], -80.0 * payload["input_scale"]
    module = ILayerNorm.from_payload(payload)
    qt = QTensor.quantize(ood, payload["input_scale"], 0.0, INT8)
    assert bool(module.segment_mask(_var_sum_of(qt.int_data, payload)).all()), (
        "the out-of-distribution row does not reach segment two")

    one_seg = calibrate_int_layernorm(ln, x, segments=1)
    with torch.no_grad():
        ref = ln(qt.dequantize())
        err_two = float((module(qt).dequantize() - ref).abs().max())
        err_one = float((ILayerNorm.from_payload(one_seg)(qt).dequantize() - ref).abs().max())
    assert err_two < err_one, (
        f"segment two buys nothing out of range: {err_two:.4f} vs {err_one:.4f}")


# --- (c) the one-segment payload still works ---------------------------------

def test_a_seven_scalar_payload_still_builds_runs_and_replays(tmp_path):
    """Nothing already exported is stranded: 7 scalars build, run bit-exactly, replay.

    ``segments=1`` is not deprecated plumbing — it is how the 7-scalar
    ``i_ops.layernorm_quantize`` golden, which every payload written before D3 is checked
    against, stays exercised by real calibrated constants.
    """
    ln = _layernorm(CHANNELS, 2)
    x = _heteroscedastic(CHANNELS, 2)
    payload = calibrate_int_layernorm(ln, x, segments=1)
    assert len(payload["scalars"]) == 7
    assert "rsqrt_table_two" not in payload and payload["segments"] == 1

    module = ILayerNorm.from_payload(payload)
    assert module.segments == 1
    x_int = _spanning_int_rows(payload, seed=404)
    out = module(QTensor(x_int, scale=payload["input_scale"], zero_point=0.0, dtype=INT8))
    assert torch.equal(out.int_data, _golden(x_int, payload))
    assert not bool(module.segment_mask(_var_sum_of(x_int, payload)).any()), (
        "a one-segment module must never claim a second segment")

    model = nn.Sequential(module)
    manifest = export_integer_model(model, tmp_path)
    entry = next(iter(manifest["modules"].values()))
    assert entry["segments"] == 1
    assert len(entry["scalars"]) == 7
    assert "scalars_two" not in entry and "rsqrt_table_two" not in entry
    ok, diff = verify_export(model, tmp_path)
    assert ok is True and diff == 0.0, f"one-segment replay diff {diff}"


# --- (d) end to end: convert -> export -> verify, plus a negative control ------

def _frame_model() -> nn.Module:
    from engine.model_factory import make_model

    torch.manual_seed(0)
    return make_model({
        "target": "models.frame.FrameModel", "embed_dim": 48, "patch_size": 16,
        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1},
    }).eval()


def test_segmenting_never_hurts_the_near_homoscedastic_real_model():
    """Where one segment was already fine, two must not be worse — the cost of the default.

    The FrameModel's tokens are near-homoscedastic (``var_sum`` dynamic range 1.6-2.4), so
    a single index already reaches 2.6e-04 to 9.5e-04 RMS relative rsqrt error and there is
    almost nothing left to win. That is the case a piecewise index could plausibly REGRESS
    — half the entries per half the range is a worse deal when the range was never the
    problem — so it is the case that justifies making it the default. Measured RMS
    relative rsqrt error over the five LayerNorms:

        1 x 256 entries (256 words)   0.00095 / 0.00039 / 0.00026 / 0.00030 / 0.00052
        2 x 128 entries (256 words)   0.00073 / 0.00039 / 0.00026 / 0.00030 / 0.00041
        2 x 256 entries (512 words)   0.00037 / 0.00020 / 0.00015 / 0.00015 / 0.00020

    So even at EQUAL table memory the segmented index ties or wins on all five, and the
    default (256 per segment) roughly halves the error for a second 256-word table. What it
    does not do is move the end-to-end error here at all — 0.0186 -> 0.0185 on the worst of
    the five — because on this model the rsqrt was never the binding constraint. The win is
    insurance for heteroscedastic activations, and this test says so with numbers instead
    of implying the headline carries over.
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
        one = calibrate_int_layernorm(ln, x, segments=1, entries=256)
        equal = calibrate_int_layernorm(ln, x, segments=2, entries=128)
        default = calibrate_int_layernorm(ln, x, segments=2, entries=256)

        assert one["metrics"]["var_sum_dynamic_range"] < 5.0, (
            f"{name}: this fixture is supposed to be the EASY case")
        rms_one, rms_equal, rms_default = (_rms(_rsqrt_rel_error(x, p))
                                           for p in (one, equal, default))
        assert rms_equal <= rms_one * 1.02, (
            f"{name}: segmenting at equal memory regressed {rms_one:.5f} -> {rms_equal:.5f}")
        assert rms_default < rms_one, (
            f"{name}: the default regressed {rms_one:.5f} -> {rms_default:.5f}")


@pytest.fixture(scope="module")
def converted_export(tmp_path_factory):
    """A real FrameModel converted with the DEFAULT (segmented) LayerNorms, exported."""
    from quantization.convert import convert_model_to_integer

    model = _frame_model()
    batches = [torch.rand(4, 1, 64, 64, generator=torch.Generator().manual_seed(s))
               for s in (1, 2)]
    model, _report = convert_model_to_integer(model, batches)
    out = tmp_path_factory.mktemp("segmented_export")
    manifest = export_integer_model(model, out, allow_unquantized=True)
    return model, out, manifest


def test_frame_model_converts_to_segmented_layernorms_and_verifies(converted_export):
    """The default path end to end: every LayerNorm is segmented and every replay exact."""
    model, out, manifest = converted_export
    entries = [e for e in manifest["modules"].values() if e["op"] == "layernorm_int"]
    assert len(entries) == 5, f"expected 5 LayerNorms, got {len(entries)}"
    for entry in entries:
        assert entry["segments"] == 2
        assert len(entry["scalars"]) == 7 and len(entry["scalars_two"]) == 3
        assert entry["table_entries"] == entry["table_entries_two"] == 256
        live = model.get_submodule(entry["name"])
        assert entry["scalars_two"] == [live.b_two, live.s1_two, live.bound_two]
        assert np.array_equal(entry["rsqrt_table_two"], live.rsqrt_table_two.cpu().numpy())

    ok, diff = verify_export(model, out)
    assert ok is True and diff == 0.0, f"verify_export -> ok={ok} diff={diff}"

    # the round trip through disk must preserve the second table exactly
    loaded = load_integer_manifest(out)
    for name, entry in loaded["modules"].items():
        if entry["op"] == "layernorm_int":
            live = model.get_submodule(name)
            assert np.array_equal(entry["rsqrt_table_two"], live.rsqrt_table_two.cpu().numpy())


def _verification_probes(manifest: dict, seed: int = 1234, batch: int = 4) -> dict[str, np.ndarray]:
    """The probe ``verify_export_report`` will hand each module, reproduced exactly.

    Not ``_probe(entry, default_rng(seed), batch)`` per module: the report draws every
    probe from ONE generator in module order, so a fresh generator reproduces only the
    first module's probe. Getting this wrong is not a small error — it silently corrupts
    a table entry no probe reads, and the negative control below then "passes" by failing
    to detect a corruption it never applied where it mattered. So the walk is replayed in
    the same order, for every entry that gets probed.
    """
    rng = np.random.default_rng(seed)
    probes: dict[str, np.ndarray] = {}
    for name, entry in sorted(manifest["modules"].items(), key=lambda kv: kv[1]["order"]):
        probes[name] = _probe(entry, rng, batch)
    return probes


def _segment_two_reads(entry: dict, probe: np.ndarray) -> list[int]:
    """Which entries of ``rsqrt_table_two`` this probe actually reads.

    Derived from manifest values alone, so the corruption below hits an entry the gate
    demonstrably depends on rather than an arbitrary index — and an empty result is a loud
    failure instead of a silently vacuous test.
    """
    c_1_m, c_1_s, b, s1, bound = (int(v) for v in entry["scalars"][:5])
    b_two, s1_two, bound_two = (int(v) for v in entry["scalars_two"])
    qmin, qmax = qrange(int(entry["input_bits"]), signed=bool(entry["input_signed"]))
    x_int = np.clip(np.round(probe.astype(np.float32) / np.float32(entry["input_scale"])),
                    qmin, qmax).astype(np.int64)
    mean = (x_int.sum(axis=-1, keepdims=True) * c_1_m + (1 << (c_1_s - 1))) >> c_1_s
    diff = x_int - mean
    var_sum = (diff * diff).sum(axis=-1, keepdims=True)
    cursor = (var_sum + b) >> s1
    used = np.clip((var_sum + b_two) >> s1_two, 0, bound_two)[cursor > bound]
    return sorted({int(v) for v in used.reshape(-1)})


def test_the_verification_probe_reaches_both_segments(converted_export):
    """Without this, corrupting segment two would replay clean and prove nothing.

    A LayerNorm probe of uniform noise gives every row the same variance and lands in one
    segment — that was the old probe, and under it the negative control below passed while
    detecting nothing. The probe is built in the variance coordinate precisely so both
    tables are read; this test is what stops that regressing.
    """
    _model, _out, manifest = converted_export
    probes = _verification_probes(manifest)
    seen = 0
    for name, entry in manifest["modules"].items():
        if entry["op"] != "layernorm_int":
            continue
        probe = probes[name]
        assert probe.shape == (4, entry["channels"])
        reads = _segment_two_reads(entry, probe)
        assert len(reads) >= 2, f"{name}: probe reads {reads} of segment two"
        seen += 1
    assert seen == 5


def test_corrupting_one_segment_two_entry_fails_verification(converted_export, tmp_path):
    """The negative control: one wrong entry in ``rsqrt_table_two`` and the gate must fail.

    A single entry, not the whole table — a whole-table corruption would also be caught by
    a replay that never reads segment two at all, so it could not distinguish "verified"
    from "verified segment one". The entry is chosen as one the probe demonstrably reads
    (``_segment_two_reads``), which is why this is a test of the replay and not of luck.
    """
    model, src, manifest = converted_export
    probes = _verification_probes(manifest)
    on_disk = json.loads((src / MANIFEST_NAME).read_text())
    victims = []
    for name, entry in on_disk["modules"].items():
        if entry.get("op") != "layernorm_int" or entry.get("segments") != 2:
            continue
        reads = _segment_two_reads(entry, probes[name])
        assert reads, f"{name}: probe never reads segment two"
        entry["rsqrt_table_two"][reads[0]] = 1      # a valid int16, wildly wrong rsqrt
        victims.append(name)
    assert len(victims) == 5

    dst = tmp_path / "corrupt_segment_two"
    shutil.copytree(src, dst)
    (dst / MANIFEST_NAME).write_text(json.dumps(on_disk))

    ok, diff = verify_export(model, dst)
    assert ok is False and diff > 1e-4, f"corruption undetected: ok={ok} diff={diff}"
    flagged = {r["name"] for r in verify_export_report(model, dst) if not r["ok"]}
    assert flagged == set(victims), f"flagged {sorted(flagged)}, expected {sorted(victims)}"


def test_corrupting_the_segment_two_scalars_fails_verification(converted_export, tmp_path):
    """The extension scalars are load-bearing too, not decorative metadata.

    Zeroing ``scalars_two``'s offset moves segment two's whole index; the replay must
    then disagree with the live module. If it did not, the manifest could describe an
    index the hardware does not implement and the gate would still sign it off.
    """
    model, src, _manifest = converted_export
    on_disk = json.loads((src / MANIFEST_NAME).read_text())
    victims = []
    for name, entry in on_disk["modules"].items():
        if entry.get("segments") == 2:
            entry["scalars_two"][0] = 0
            victims.append(name)
    dst = tmp_path / "corrupt_scalars_two"
    shutil.copytree(src, dst)
    (dst / MANIFEST_NAME).write_text(json.dumps(on_disk))

    ok, diff = verify_export(model, dst)
    assert ok is False and diff > 1e-4
    flagged = {r["name"] for r in verify_export_report(model, dst) if not r["ok"]}
    assert flagged == set(victims)
