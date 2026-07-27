"""Per-channel input scales must be bridged channel by channel, never truncated.

``ILinear.forward_accumulator`` hands on ``act_scale * weight_scale``, and
``weight_scale`` is per-out-channel — so a genuine ``[C]`` scale tensor reaches
``IGeLU``, ``ISoftmax``, ``IAdd``, ``ICat`` and ``IPool`` on the ordinary production
path. Every one of those used to collapse it with ``float(v.reshape(-1)[0])``, which
reads channel 0's grid onto every other channel and is silent about it.

Each test here computes the element-0 answer *explicitly* and asserts the module does
not produce it, then asserts the module matches a reference bridged one channel at a
time in python. The inequality half is what makes these tests able to fail: without it
a reintroduced truncation would still satisfy "runs and returns something".
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quantization.ilayers.linear import ILinear
from quantization.ilayers.nonlinear import IGeLU
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.softmax import ISoftmax
from quantization.ilayers.tensor_ops import IAdd, ICat, IPool
from quantization.int_calibrate_softmax import build_softmax_int_payload
from quantization.scheme import INT8, QuantDtype

# b=128, s=0, bound=255 with table[i] = i - 128 makes IGeLU the identity on the bridged
# integer (saturating at the int8 range), so the LUT never hides a bridging difference.
GELU_SCALARS = [128, 0, 255]
GELU_TABLE = [i - 128 for i in range(256)]


# --- helpers -----------------------------------------------------------------

def _per_channel(int_data: torch.Tensor, scales: list[float],
                 dtype: QuantDtype = INT8) -> QTensor:
    """A QTensor carrying a genuine ``[C]`` scale over the trailing dim."""
    assert int_data.shape[-1] == len(scales)
    return QTensor(int_data.to(torch.int32),
                   scale=torch.tensor(scales, dtype=torch.float32),
                   zero_point=0.0, dtype=dtype)


def _element_zero(qt: QTensor) -> QTensor:
    """The same tensor as the old ``_scalar`` truncation saw it: scale = element 0."""
    return QTensor(qt.int_data, scale=float(torch.as_tensor(qt.scale).reshape(-1)[0]),
                   zero_point=qt.zero_point, dtype=qt.dtype)


def _bridge_ref(int_data: torch.Tensor, scales, target_scale: float,
                dtype: QuantDtype | None = None) -> torch.Tensor:
    """Reference bridge onto ``target_scale``, one channel at a time (no broadcasting)."""
    flat = torch.as_tensor(scales).reshape(-1)
    out = torch.empty(int_data.shape, dtype=torch.float64)
    for c in range(int_data.shape[-1]):
        s = float(flat[c]) if flat.numel() > 1 else float(flat[0])
        out[..., c] = torch.round(int_data[..., c].to(torch.float64) * (s / target_scale))
    if dtype is not None:
        out = out.clamp(dtype.qmin, dtype.qmax)
    return out.to(torch.int64)


def _gelu(input_scale: float) -> IGeLU:
    return IGeLU(GELU_SCALARS, GELU_TABLE, input_scale=input_scale, output_scale=0.01)


def _gelu_ref(bridged: torch.Tensor) -> torch.Tensor:
    """``table[clamp(x + b, 0, bound)]`` spelled out from the constants above."""
    cursor = (bridged + GELU_SCALARS[0]).clamp(0, GELU_SCALARS[2])
    return torch.tensor(GELU_TABLE, dtype=torch.int64)[cursor]


def _softmax(input_scale: float, tokens: int = 6, heads: int = 2) -> ISoftmax:
    torch.manual_seed(4)
    logits = torch.randn(heads, tokens, tokens) * 2.0
    return ISoftmax.from_payload(build_softmax_int_payload(logits, input_scale=input_scale))


def _ilinear(out_features: int = 4, in_features: int = 8) -> ILinear:
    """An ILinear whose per-out-channel weight scales span ~8x."""
    torch.manual_seed(3)
    weight_int = torch.randint(-64, 65, (out_features, in_features), dtype=torch.int32)
    weight_scale = torch.tensor([0.004, 0.011, 0.020, 0.031][:out_features], dtype=torch.float32)
    return ILinear(weight_int, weight_scale, 0.05, INT8, None)


def _accumulator(shape: tuple[int, ...]) -> QTensor:
    """A real ``forward_accumulator`` QTensor — the production source of these scales."""
    lin = _ilinear()
    torch.manual_seed(5)
    x = torch.randn(*shape, 8) * 0.5
    qt = lin.forward_accumulator(x)
    assert torch.is_tensor(qt.scale) and qt.scale.numel() == 4, "expected a per-channel scale"
    return qt


def _scale_for(qt: QTensor, headroom: int = 100) -> float:
    """An input scale that lands the largest bridged accumulator value near ``headroom``."""
    peak = float(qt.scale.max()) * float(qt.int_data.abs().max())
    return peak / headroom


# --- IGeLU (elementwise: a [C] scale bridges per channel) --------------------

def test_igelu_bridges_each_channel_not_channel_zero():
    scales = [0.02, 0.05, 0.11, 0.20]
    input_scale = 0.01                       # ratios 2, 5, 11, 20
    int_data = torch.tensor([[3, 3, 3, 3], [-5, 2, -1, 4], [6, -6, 5, -2]], dtype=torch.int32)
    qt = _per_channel(int_data, scales)

    out = _gelu(input_scale)(qt)

    truncated = _gelu_ref(_bridge_ref(int_data, [scales[0]] * 4, input_scale))
    assert not torch.equal(out.int_data.to(torch.int64), truncated), \
        "IGeLU still reads channel 0's scale onto every channel"
    expected = _gelu_ref(_bridge_ref(int_data, scales, input_scale))
    assert torch.equal(out.int_data.to(torch.int64), expected)


def test_igelu_rejects_a_scale_that_is_neither_scalar_nor_per_channel():
    qt = QTensor(torch.zeros(2, 4, dtype=torch.int32),
                 scale=torch.tensor([0.02, 0.05, 0.11]), zero_point=0.0, dtype=INT8)
    with pytest.raises(ValueError, match="per-tensor or per-channel"):
        _gelu(0.01)(qt)


def test_igelu_still_treats_a_one_element_scale_tensor_as_a_scalar():
    int_data = torch.tensor([[7, -3, 5, -8]], dtype=torch.int32)
    one = QTensor(int_data, scale=torch.tensor([0.02]), zero_point=0.0, dtype=INT8)
    plain = QTensor(int_data, scale=0.02, zero_point=0.0, dtype=INT8)
    gelu = _gelu(0.01)
    assert torch.equal(gelu(one).int_data, gelu(plain).int_data)


# --- ISoftmax (the [C] axis IS the reduction axis) ---------------------------

def test_isoftmax_bridges_a_per_channel_scale_along_its_reduction_axis():
    """Pins the design decision: bridgeable, because the bridge *builds* the shared grid.

    A per-channel scale along ``dim`` means the raw row is not on one grid, which the
    integer row max and the exp table both assume. The bridge runs before any reduction
    and re-expresses every element on ``input_scale``, so the kernel still sees a uniform
    row — hence bridge rather than raise. This test fails if that ever silently becomes a
    truncation, and ``test_isoftmax_...rejects...`` below fails if it becomes a raise for
    the legitimate case.
    """
    input_scale = 0.05
    scales = [0.05, 0.08, 0.12, 0.02, 0.15, 0.04]      # ratios 1.0 .. 3.0
    torch.manual_seed(7)
    int_data = torch.randint(-40, 41, (2, 6, 6), dtype=torch.int32)
    isoftmax = _softmax(input_scale)
    qt = _per_channel(int_data, scales)

    out = isoftmax(qt)

    truncated = isoftmax.forward_int(_bridge_ref(int_data, [scales[0]] * 6, input_scale, INT8))
    assert not torch.equal(out.int_data.to(torch.int64), truncated), \
        "ISoftmax still reads channel 0's scale onto every channel"
    expected = isoftmax.forward_int(_bridge_ref(int_data, scales, input_scale, INT8))
    assert torch.equal(out.int_data.to(torch.int64), expected)


def test_isoftmax_clamps_a_bridged_value_back_into_its_input_port():
    """A bridge that overshoots int8 must saturate at the declared ``in_dtype`` port."""
    input_scale = 0.05
    scales = [0.05, 0.60, 0.12, 0.02, 0.15, 0.04]      # channel 1 ratio 12 -> overflows
    torch.manual_seed(8)
    int_data = torch.randint(-40, 41, (2, 6, 6), dtype=torch.int32)
    isoftmax = _softmax(input_scale)

    out = isoftmax(_per_channel(int_data, scales))

    unclamped = _bridge_ref(int_data, scales, input_scale)
    assert int(unclamped.abs().max()) > INT8.qmax, "test input does not exercise the clamp"
    assert not torch.equal(out.int_data.to(torch.int64), isoftmax.forward_int(unclamped)), \
        "ISoftmax did not clamp the bridged value to its input port width"
    expected = isoftmax.forward_int(unclamped.clamp(INT8.qmin, INT8.qmax))
    assert torch.equal(out.int_data.to(torch.int64), expected)


def test_isoftmax_rejects_a_scale_that_is_neither_scalar_nor_per_channel():
    qt = QTensor(torch.zeros(2, 6, 6, dtype=torch.int32),
                 scale=torch.tensor([0.05, 0.08, 0.12]), zero_point=0.0, dtype=INT8)
    with pytest.raises(ValueError, match="per-tensor or per-channel"):
        _softmax(0.05)(qt)


# --- IAdd / ICat (operands align independently) ------------------------------

def test_iadd_aligns_each_operand_channel_by_channel():
    s_out = 0.03
    sa = [0.02, 0.09, 0.15, 0.04]
    sb = 0.05
    torch.manual_seed(9)
    a_int = torch.randint(-30, 31, (3, 4), dtype=torch.int32)
    b_int = torch.randint(-30, 31, (3, 4), dtype=torch.int32)
    qa = _per_channel(a_int, sa)
    qb = QTensor(b_int, scale=sb, zero_point=0.0, dtype=INT8)

    out = IAdd(s_out)(qa, qb)

    b_ref = _bridge_ref(b_int, [sb] * 4, s_out)
    truncated = (_bridge_ref(a_int, [sa[0]] * 4, s_out) + b_ref).clamp(INT8.qmin, INT8.qmax)
    assert not torch.equal(out.int_data.to(torch.int64), truncated), \
        "IAdd still reads channel 0's scale onto every channel"
    expected = (_bridge_ref(a_int, sa, s_out) + b_ref).clamp(INT8.qmin, INT8.qmax)
    assert torch.equal(out.int_data.to(torch.int64), expected)


def test_icat_aligns_each_part_channel_by_channel():
    s_out = 0.03
    s_one = [0.02, 0.09, 0.15]
    s_two = [0.05, 0.01]
    torch.manual_seed(10)
    one_int = torch.randint(-30, 31, (2, 3), dtype=torch.int32)
    two_int = torch.randint(-30, 31, (2, 2), dtype=torch.int32)
    parts = [_per_channel(one_int, s_one), _per_channel(two_int, s_two)]

    out = ICat(s_out, dim=-1)(parts)

    def _cat(sa, sb):
        return torch.cat([_bridge_ref(one_int, sa, s_out).clamp(INT8.qmin, INT8.qmax),
                          _bridge_ref(two_int, sb, s_out).clamp(INT8.qmin, INT8.qmax)], dim=-1)

    truncated = _cat([s_one[0]] * 3, [s_two[0]] * 2)
    assert not torch.equal(out.int_data.to(torch.int64), truncated), \
        "ICat still reads channel 0's scale onto every channel"
    assert torch.equal(out.int_data.to(torch.int64), _cat(s_one, s_two))


def test_icat_rejects_a_scale_that_is_neither_scalar_nor_per_channel():
    bad = QTensor(torch.zeros(2, 3, dtype=torch.int32),
                  scale=torch.tensor([0.02, 0.09]), zero_point=0.0, dtype=INT8)
    with pytest.raises(ValueError, match="per-tensor or per-channel"):
        ICat(0.03, dim=-1)([bad])


def test_iadd_rejects_a_scale_that_is_neither_scalar_nor_per_channel():
    good = QTensor(torch.zeros(3, 4, dtype=torch.int32), scale=0.05, zero_point=0.0, dtype=INT8)
    bad = QTensor(torch.zeros(3, 4, dtype=torch.int32),
                  scale=torch.tensor([0.02, 0.09, 0.15]), zero_point=0.0, dtype=INT8)
    with pytest.raises(ValueError, match="per-tensor or per-channel"):
        IAdd(0.03)(bad, good)


# --- IPool (preserves the scale; must not reduce across it) ------------------

def test_ipool_forwards_the_whole_per_channel_scale():
    scales = [0.02, 0.09, 0.15, 0.04]
    torch.manual_seed(11)
    int_data = torch.randint(-60, 61, (2, 7, 4), dtype=torch.int32)
    qt = _per_channel(int_data, scales)

    out = IPool(dim=1)(qt)

    assert torch.is_tensor(out.scale) and out.scale.numel() == 4, "IPool truncated the scale"
    mean_int = torch.round(int_data.to(torch.int64).sum(dim=1).to(torch.float32) / 7)
    mean_int = mean_int.clamp(INT8.qmin, INT8.qmax)
    truncated = mean_int * scales[0]
    assert not torch.allclose(out.dequantize(), truncated), \
        "IPool dequantizes with channel 0's scale only"
    expected = torch.stack([mean_int[..., c] * scales[c] for c in range(4)], dim=-1)
    assert torch.allclose(out.dequantize(), expected.to(torch.float32))


def test_ipool_refuses_to_pool_over_the_per_channel_axis():
    """No bridge exists here: IPool preserves the input scale, so this must raise."""
    qt = _per_channel(torch.ones(2, 7, 4, dtype=torch.int32), [0.02, 0.09, 0.15, 0.04])
    with pytest.raises(ValueError, match="per-channel axis"):
        IPool(dim=-1)(qt)
    with pytest.raises(ValueError, match="per-channel axis"):
        IPool(dim=2)(qt)


def test_ipool_rejects_a_scale_that_is_neither_scalar_nor_per_channel():
    qt = QTensor(torch.ones(2, 7, 4, dtype=torch.int32),
                 scale=torch.tensor([0.02, 0.09, 0.15, 0.04, 0.06]), zero_point=0.0, dtype=INT8)
    with pytest.raises(ValueError, match="per-tensor or per-channel"):
        IPool(dim=1)(qt)


# --- end to end: the real ILinear accumulator into each consumer -------------

def test_ilinear_accumulator_carries_a_genuinely_per_channel_scale():
    qt = _accumulator((3,))
    flat = qt.scale.reshape(-1)
    assert flat.numel() == 4
    assert float(flat.max() / flat.min()) > 4.0, "channels must differ substantially"


def test_igelu_consumes_an_ilinear_accumulator_per_channel():
    qt = _accumulator((3,))
    input_scale = _scale_for(qt)

    out = _gelu(input_scale)(qt)

    truncated = _gelu_ref(_bridge_ref(qt.int_data, [float(qt.scale.reshape(-1)[0])] * 4, input_scale))
    assert not torch.equal(out.int_data.to(torch.int64), truncated)
    assert torch.equal(out.int_data.to(torch.int64),
                       _gelu_ref(_bridge_ref(qt.int_data, qt.scale, input_scale)))


def test_isoftmax_consumes_an_ilinear_accumulator_per_channel():
    qt = _accumulator((2, 4))                       # [B, tokens, out_features]
    input_scale = _scale_for(qt)
    isoftmax = _softmax(input_scale, tokens=4, heads=2)

    out = isoftmax(qt)

    truncated = isoftmax.forward_int(
        _bridge_ref(qt.int_data, [float(qt.scale.reshape(-1)[0])] * 4, input_scale, INT8))
    assert not torch.equal(out.int_data.to(torch.int64), truncated)
    assert torch.equal(out.int_data.to(torch.int64),
                       isoftmax.forward_int(_bridge_ref(qt.int_data, qt.scale, input_scale, INT8)))


def test_iadd_and_icat_consume_an_ilinear_accumulator_per_channel():
    qt = _accumulator((3,))
    s_out = _scale_for(qt)
    zero = _element_zero(qt)

    added = IAdd(s_out)(qt, qt)
    truncated_add = (2 * _bridge_ref(qt.int_data, zero.scale, s_out)).clamp(INT8.qmin, INT8.qmax)
    assert not torch.equal(added.int_data.to(torch.int64), truncated_add)
    expected_add = (2 * _bridge_ref(qt.int_data, qt.scale, s_out)).clamp(INT8.qmin, INT8.qmax)
    assert torch.equal(added.int_data.to(torch.int64), expected_add)

    catted = ICat(s_out, dim=-1)([qt, qt])
    aligned = _bridge_ref(qt.int_data, qt.scale, s_out).clamp(INT8.qmin, INT8.qmax)
    assert torch.equal(catted.int_data.to(torch.int64), torch.cat([aligned, aligned], dim=-1))
    truncated_cat = _bridge_ref(qt.int_data, zero.scale, s_out).clamp(INT8.qmin, INT8.qmax)
    assert not torch.equal(catted.int_data.to(torch.int64),
                           torch.cat([truncated_cat, truncated_cat], dim=-1))


def test_ipool_consumes_an_ilinear_accumulator_per_channel():
    qt = _accumulator((2, 5))                       # [B, tokens, out_features]

    out = IPool(dim=1)(qt)

    assert torch.is_tensor(out.scale) and out.scale.numel() == 4
    mean_int = torch.round(qt.int_data.to(torch.int64).sum(dim=1).to(torch.float32) / 5)
    mean_int = mean_int.clamp(qt.dtype.qmin, qt.dtype.qmax)
    scales = qt.scale.reshape(-1)
    truncated = mean_int * float(scales[0])
    assert not torch.allclose(out.dequantize(), truncated)
    expected = torch.stack([mean_int[..., c] * float(scales[c]) for c in range(4)], dim=-1)
    assert torch.allclose(out.dequantize(), expected.to(torch.float32))


# --- exactness: the integer datapath must not round-trip through float32 ------
#
# The bridges and IPool were moved off float32 because float32 represents integers
# exactly only below 2^24, and an int32 ILinear accumulator runs past that. Nothing
# pinned that, so reverting it was silent. These tests are the pin: each uses a value
# that is exactly representable in int64/float64 and NOT in float32.

_BEYOND_F32 = 2 ** 24 + 1  # first positive integer float32 cannot represent


def test_float32_really_cannot_hold_the_probe_value():
    """Guard the guard: if this ever passes, the tests below prove nothing."""
    assert float(torch.tensor([_BEYOND_F32], dtype=torch.float32)[0]) != _BEYOND_F32
    assert float(torch.tensor([_BEYOND_F32], dtype=torch.float64)[0]) == _BEYOND_F32


def test_igelu_bridge_is_exact_past_the_float32_integer_limit():
    """Drive the difference all the way through IGeLU.forward to a different LUT entry.

    A one-off difference at 2**24 vanishes into the cursor's right shift, so the probe
    sits at 2**30 where float32's spacing is 128: an offset of 32 is dropped entirely.
    ``b`` cancels the large constant so the surviving offset lands in the table.
    """
    offset, big = 32, 2 ** 30
    x = big + offset
    # premise: float32 cannot hold x, float64 can
    assert float(torch.tensor([x], dtype=torch.float32)[0]) == float(big)
    assert float(torch.tensor([x], dtype=torch.float64)[0]) == float(x)

    table = list(range(-128, 128))
    gelu = IGeLU([-big, 4, 255], table, input_scale=1.0, output_scale=1.0)
    qt = QTensor(torch.tensor([[x]], dtype=torch.int64), scale=1.0,
                 zero_point=0.0, dtype=QuantDtype(32, signed=True))
    out = int(gelu(qt).int_data.reshape(-1)[0])

    exact_cursor = (x - big) >> 4            # 2
    lossy_cursor = (big - big) >> 4          # 0, what a float32 bridge would produce
    assert exact_cursor != lossy_cursor      # the probe really discriminates
    assert out == table[exact_cursor], "IGeLU bridge lost low bits (float32 round-trip?)"
    assert out != table[lossy_cursor]


def test_iadd_alignment_is_exact_past_the_float32_integer_limit():
    from quantization.ilayers.tensor_ops import _rescale_to

    qt = QTensor(torch.tensor([[_BEYOND_F32]], dtype=torch.int64), scale=1.0,
                 zero_point=0.0, dtype=QuantDtype(32, signed=True))
    aligned = _rescale_to(qt, 1.0)  # ratio exactly 1.0
    assert int(aligned[0, 0]) == _BEYOND_F32, "alignment lost low bits (float32 round-trip?)"


def test_ipool_mean_is_exact_integer_arithmetic():
    """IPool's mean must be computed in int64, with a reference that is not its own code."""
    # 4 tokens whose sum exceeds float32's exact-integer range; the true mean is exact.
    values = [_BEYOND_F32, _BEYOND_F32 + 1, _BEYOND_F32 + 2, _BEYOND_F32 + 3]
    wide = QuantDtype(32, signed=True)
    qt = QTensor(torch.tensor(values, dtype=torch.int64).reshape(1, 4, 1),
                 scale=1.0, zero_point=0.0, dtype=wide)
    out = IPool(dim=1)(qt)
    # independent reference: python ints, round-half-away-from-zero, no torch involved
    total = sum(values)
    expected = (total + len(values) // 2) // len(values)
    assert int(out.int_data.reshape(-1)[0]) == expected


@pytest.mark.parametrize("summed,n", [(-7, 2), (-5, 2), (5, 2), (7, 2), (-1, 3), (1, 3)])
def test_ipool_rounding_is_symmetric_about_zero(summed, n):
    """Negative means must round the same distance as positive ones."""
    wide = QuantDtype(32, signed=True)
    # build n tokens summing to `summed`
    values = [summed] + [0] * (n - 1)
    qt = QTensor(torch.tensor(values, dtype=torch.int64).reshape(1, n, 1),
                 scale=1.0, zero_point=0.0, dtype=wide)
    got = int(IPool(dim=1)(qt).int_data.reshape(-1)[0])
    half = n // 2
    expected = (summed + half) // n if summed >= 0 else -((-summed + half) // n)
    assert got == expected
    # and it really is symmetric
    neg = torch.tensor([-summed] + [0] * (n - 1), dtype=torch.int64).reshape(1, n, 1)
    got_neg = int(IPool(dim=1)(QTensor(neg, scale=1.0, zero_point=0.0, dtype=wide)).int_data.reshape(-1)[0])
    assert got_neg == -got


def test_isoftmax_clamps_the_input_port_even_when_the_ratio_is_exactly_one():
    """Two spellings of the same grid must give the same answer.

    The input port is a physical width. Skipping the clamp on the identity path made
    an out-of-range integer sail through when the caller happened to pass the same
    scale, and saturate when it passed an equivalent-but-different one.
    """
    torch.manual_seed(0)
    logits = torch.randn(6, 8) * 2.0
    payload = build_softmax_int_payload(logits.numpy(), exp_entries=64, recip_entries=64)
    sm = ISoftmax.from_payload(payload, dim=-1, in_dtype=INT8)

    # integer data deliberately outside the int8 input port
    raw = torch.tensor([[200, -200, 5, 3, 1, 0, -1, -3]], dtype=torch.int64)
    same = QTensor(raw, scale=sm.input_scale, zero_point=0.0, dtype=QuantDtype(32, signed=True))
    out_same = sm.forward_qtensor(same)

    # the same grid spelled as a 1-element tensor -> ratio is still exactly 1.0
    spelled = QTensor(raw, scale=torch.tensor([sm.input_scale]), zero_point=0.0,
                      dtype=QuantDtype(32, signed=True))
    out_spelled = sm.forward_qtensor(spelled)

    assert torch.equal(out_same.int_data, out_spelled.int_data)
    # and the port really was enforced: a clamped 200 cannot behave like an unclamped one
    clamped = QTensor(raw.clamp(INT8.qmin, INT8.qmax), scale=sm.input_scale,
                      zero_point=0.0, dtype=QuantDtype(32, signed=True))
    assert torch.equal(out_same.int_data, sm.forward_qtensor(clamped).int_data)
