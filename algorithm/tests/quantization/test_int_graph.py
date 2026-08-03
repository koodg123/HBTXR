"""I-tier integer graph (Part C): QTensor + torch kernels bit-exact vs i_ops golden."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.nn import functional as F

from quantization.i_ops import (
    dyadic_params,
    int_conv2d,
    int_matmul as g_int_matmul,
    requant as g_requant,
    table_quantize as g_table_quantize,
)
from quantization.ilayers.conv import IConv2d
from quantization.ilayers.int_functional import int_matmul as t_int_matmul, requant as t_requant
from quantization.ilayers.linear import ILinear
from quantization.ilayers.matmul import IMatMul
from quantization.ilayers.nonlinear import IGeLU
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.tensor_ops import IAdd, ICat, IPool
from quantization.lut_calibrate import build_gelu_lut
from quantization.observer import build_observer
from quantization.qlayers.linear import QLinear, QuantConfig
from quantization.qlayers.nonlinear import QGeLU
from quantization.scheme import INT8, UINT8
from quantization.spec import TensorQuantSpec


def _calibrated_qlinear(weight_spec, act_spec, in_f=12, out_f=8):
    torch.manual_seed(7)
    lin = nn.Linear(in_f, out_f)
    ql = QLinear(lin, QuantConfig(weight_spec=weight_spec, act_spec=act_spec))
    ow = build_observer(ql.weight_fq.spec)
    ow.observe(lin.weight)
    ql.weight_fq.set_qparams(*ow.qparams())
    x = torch.randn(20, in_f) * 2.0 + 0.5
    oa = build_observer(ql.act_fq.spec)
    oa.observe(x)
    ql.act_fq.set_qparams(*oa.qparams())
    return ql, x


# --- QTensor -----------------------------------------------------------------

def test_qtensor_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(4, 8) * 1.5
    scale = float(x.abs().max()) / 127.0
    qt = QTensor.quantize(x, scale, 0.0, INT8)
    assert qt.int_data.dtype == torch.int32
    assert int(qt.int_data.abs().max()) <= 127
    recon = qt.dequantize()
    assert (recon - x).abs().max() < scale  # within one quantization step


def test_qtensor_passthrough_keeps_scale():
    q = QTensor(torch.arange(24, dtype=torch.int32).reshape(2, 3, 4), scale=0.05, zero_point=0.0, dtype=INT8)
    r = q.reshape(6, 4)
    t = q.transpose(0, 1)
    assert r.scale == 0.05 and t.scale == 0.05 and r.dtype is INT8
    assert torch.equal(r.int_data, q.int_data.reshape(6, 4))
    assert torch.equal(t.int_data, q.int_data.transpose(0, 1))


# --- int_matmul bit-exact ----------------------------------------------------

def test_int_matmul_matches_golden():
    torch.manual_seed(1)
    a = torch.randint(-128, 128, (6, 10), dtype=torch.int32)
    b = torch.randint(-128, 128, (10, 7), dtype=torch.int32)
    golden = torch.tensor(g_int_matmul(a.tolist(), b.tolist()), dtype=torch.int64)
    assert torch.equal(t_int_matmul(a, b), golden)


# --- requant bit-exact (scalar + per-channel) --------------------------------

def test_requant_scalar_matches_golden():
    torch.manual_seed(2)
    acc = torch.randint(-200000, 200000, (5, 9), dtype=torch.int64)
    mult, shift = 12345, 17
    golden = torch.tensor(
        g_requant(acc.reshape(-1).tolist(), mult, shift, bits=8, signed=True), dtype=torch.int32
    ).reshape(acc.shape)
    assert torch.equal(t_requant(acc, mult, shift, dtype=INT8), golden)


def test_requant_per_channel_matches_golden():
    torch.manual_seed(3)
    acc = torch.randint(-500000, 500000, (4, 6), dtype=torch.int64)  # [C, K]
    mult = torch.tensor([101, 250, 777, 4096], dtype=torch.int64)
    shift = torch.tensor([10, 12, 15, 18], dtype=torch.int64)
    golden = torch.empty_like(acc, dtype=torch.int32)
    for c in range(acc.shape[0]):
        golden[c] = torch.tensor(
            g_requant(acc[c].tolist(), int(mult[c]), int(shift[c]), bits=8, signed=True), dtype=torch.int32
        )
    out = t_requant(acc, mult.reshape(-1, 1), shift.reshape(-1, 1), dtype=INT8)
    assert torch.equal(out, golden)


def test_requant_unsigned_zero_point():
    acc = torch.tensor([[0, 1000, 50000, -3000]], dtype=torch.int64)
    golden = torch.tensor(
        g_requant(acc.reshape(-1).tolist(), 64, 8, bits=8, signed=False, zero_point=5), dtype=torch.int32
    ).reshape(acc.shape)
    out = t_requant(acc, 64, 8, dtype=UINT8, zero_point=5)
    assert torch.equal(out, golden)
    assert int(out.min()) >= 0 and int(out.max()) <= 255


# --- dyadic params -----------------------------------------------------------

@pytest.mark.parametrize("scale", [0.0131, 0.5, 1.9, 0.002, 7.25])
def test_dyadic_params_approximates(scale):
    mult, shift, eff = dyadic_params(scale)
    assert mult >= 1 and shift >= 1
    assert abs(eff - scale) / scale < 1e-3
    assert abs(mult / (1 << shift) - eff) < 1e-12


@pytest.mark.parametrize("scale", [0.0131, 0.5, 1.9, 0.002, 7.25, 1.149425, 1e-6, 3.0])
def test_dyadic_multiplier_fits_its_port(scale):
    """The multiplier is a hardware register, so it has to fit one.

    Unbounded, the search lands on shift_max every time — the error falls monotonically —
    so any ratio above 1 produces `round(scale * 2^31)`, 33 bits, and a 53-bit product
    against a 20-bit accumulator. This is the bound that stops that.
    """
    from quantization.i_ops import DYADIC_MULT_BITS

    mult, shift, _ = dyadic_params(scale)
    assert mult.bit_length() <= DYADIC_MULT_BITS, f"{scale}: {mult} needs {mult.bit_length()} bits"
    # The SHIFT is not what is bounded, and that distinction is the whole point: a small
    # ratio still gets a large shift, which is what keeps the bound lossless.
    assert 1 <= shift <= 31


def test_dyadic_copy_matches_golden():
    """`ilayers` keeps a hand copy of this kernel; the copy may not drift in VALUE.

    The copy is deliberate — a reference that shares code with the thing it validates
    proves less — but a different multiplier width there means the deployed kernel and the
    golden are computing different functions, which is the one thing the copy must not buy.
    """
    from quantization.i_ops import DYADIC_MULT_BITS
    from quantization.ilayers.int_functional import _DYADIC_MULT_BITS, _dyadic_params

    assert _DYADIC_MULT_BITS == DYADIC_MULT_BITS
    for scale in (0.0131, 0.5, 1.9, 0.002, 7.25, 1.149425, 1e-6, 1.0):
        mult, shift, _ = dyadic_params(scale)
        assert (mult, shift) == _dyadic_params(scale), scale


def test_dyadic_bound_is_on_the_multiplier_not_the_shift():
    """Capping the shift instead is the intuitive move and it is wrong.

    `max(1, round(scale * 2^n))` bottoms out at 1, so a small ratio under a small
    shift_max gets an effective ratio off by a factor — measured at 1.58e-02 max relative
    error over the HLS golden, against 3.81e-06 for the multiplier bound.
    """
    tiny = 1e-6
    mult, shift, eff = dyadic_params(tiny)
    assert shift > 17, "a small ratio must keep its large shift"
    # At a ratio this small the accuracy limit is shift_max, not wbits: the multiplier
    # here is 11 bits of the 18 available. So the bound costs nothing at all, and the
    # residual error is the same one the unbounded search had.
    assert mult.bit_length() < 18
    assert abs(eff - tiny) / tiny < 1e-3

    # What a shift cap would have done instead.
    _, _, capped = dyadic_params(tiny, shift_max=17, wbits=None)
    assert abs(capped - tiny) / tiny > 1.0, "the shift cap should be visibly worse"


# --- int_conv2d golden vs torch conv (integer-exact) -------------------------

def test_int_conv2d_matches_torch_conv():
    torch.manual_seed(4)
    cin, cout, size, patch = 2, 5, 8, 4
    inp = torch.randint(-20, 20, (cin, size, size), dtype=torch.int64)
    weight = torch.randint(-10, 10, (cout, cin, patch, patch), dtype=torch.int64)
    golden = torch.tensor(int_conv2d(inp.tolist(), weight.tolist(), stride=patch), dtype=torch.int64)
    ref = torch.nn.functional.conv2d(
        inp.unsqueeze(0).float(), weight.float(), stride=patch
    ).squeeze(0).round().to(torch.int64)
    assert torch.equal(golden, ref)


# --- ILinear reproduces QLinear (weight×act) ---------------------------------

@pytest.mark.parametrize("weight_gran", ["per-tensor", "per-channel"])
@pytest.mark.parametrize("act_symmetric", [True, False])
def test_ilinear_matches_qlinear(weight_gran, act_symmetric):
    wspec = TensorQuantSpec(granularity=weight_gran, ch_axis=0)
    aspec = TensorQuantSpec(ch_axis=-1, symmetric=act_symmetric)
    ql, x = _calibrated_qlinear(wspec, aspec)
    il = ILinear.from_qlinear(ql)
    with torch.no_grad():
        y_fq = ql(x)
        y_int = il(x)
    assert torch.allclose(y_int, y_fq, atol=1e-4), f"max diff {(y_int - y_fq).abs().max():.2e}"


def test_ilinear_accumulator_qtensor():
    ql, x = _calibrated_qlinear(TensorQuantSpec(granularity="per-channel", ch_axis=0),
                                TensorQuantSpec(ch_axis=-1))
    il = ILinear.from_qlinear(ql)
    with torch.no_grad():
        acc = il.forward_accumulator(x)
        recon = acc.dequantize()
        if il.bias is not None:
            recon = recon + il.bias
        assert torch.allclose(recon, il(x), atol=1e-4)
    assert acc.int_data.dtype == torch.int32


def test_ilinear_rejects_per_group_weight():
    ql, _ = _calibrated_qlinear(TensorQuantSpec(granularity="per-group", group_size=4, ch_axis=0),
                                TensorQuantSpec(ch_axis=-1))
    with pytest.raises(NotImplementedError):
        ILinear.from_qlinear(ql)


# --- IConv2d reproduces a fake-quant conv ------------------------------------

def test_iconv2d_matches_fakequant_conv():
    torch.manual_seed(5)
    conv = nn.Conv2d(1, 6, kernel_size=4, stride=4)
    x = torch.rand(2, 1, 16, 16)
    s_x = float(x.abs().max()) / 127.0

    # fake-quant reference: dequantized int activation × dequantized per-channel int weight
    x_fq = torch.round(x / s_x).clamp(-128, 127) * s_x
    max_abs = conv.weight.detach().abs().amax(dim=(1, 2, 3)).clamp_min(1e-8)
    s_w = max_abs / 127.0
    w_fq = torch.round(conv.weight / s_w.view(-1, 1, 1, 1)).clamp(-128, 127) * s_w.view(-1, 1, 1, 1)
    ref = F.conv2d(x_fq, w_fq, conv.bias, stride=4)

    iconv = IConv2d.from_conv(conv, act_scale=s_x)
    with torch.no_grad():
        out = iconv(x)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4), f"max diff {(out - ref).abs().max():.2e}"


# --- IMatMul (act x act, attention) ------------------------------------------

def test_imatmul_matches_fakequant_and_close():
    torch.manual_seed(6)
    # batched attention scores: Q @ K^T over [B, H, N, d]
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    a, b = q, k.transpose(-2, -1)
    s_a = float(a.abs().max()) / 127.0
    s_b = float(b.abs().max()) / 127.0

    a_fq = torch.round(a / s_a).clamp(-128, 127) * s_a
    b_fq = torch.round(b / s_b).clamp(-128, 127) * s_b
    ref = a_fq @ b_fq

    im = IMatMul(s_a, s_b)
    with torch.no_grad():
        out = im(a, b)
        acc = im.forward_accumulator(a, b)
    assert torch.allclose(out, ref, atol=1e-4), f"max diff {(out - ref).abs().max():.2e}"
    assert torch.allclose(acc.dequantize(), ref, atol=1e-4)
    # close to the true float product
    assert float((out - (a @ b)).norm() / ((a @ b).norm() + 1e-8)) < 0.1


# --- IGeLU (integer LUT, bit-exact vs table_quantize golden) -----------------

def test_igelu_matches_table_quantize_golden():
    torch.manual_seed(8)
    payload = build_gelu_lut((torch.randn(8192) * 1.5).numpy(), entries=256)
    qg = QGeLU(payload["scalars"], payload["table"],
               input_scale=payload["input_scale"], output_scale=payload["output_scale"])
    ig = IGeLU.from_qgelu(qg)
    # input QTensor already at the LUT input scale -> IGeLU is the golden lookup
    x_int = torch.randint(-40, 40, (3, 16), dtype=torch.int32)
    qt = QTensor(x_int, scale=ig.input_scale, zero_point=0.0, dtype=INT8)
    out = ig(qt)
    golden = torch.tensor(
        g_table_quantize(x_int.reshape(-1).tolist(), [ig.b, ig.s, ig.bound], ig.table.tolist()),
        dtype=torch.int32,
    ).reshape(x_int.shape)
    assert torch.equal(out.int_data, golden)
    assert out.dtype is INT8


def test_igelu_close_to_float_gelu():
    import torch.nn.functional as F

    torch.manual_seed(9)
    payload = build_gelu_lut((torch.randn(8192) * 1.5).numpy(), entries=256)
    qg = QGeLU(payload["scalars"], payload["table"],
               input_scale=payload["input_scale"], output_scale=payload["output_scale"])
    ig = IGeLU.from_qgelu(qg)
    x = torch.linspace(-3, 3, 200)
    qt = QTensor.quantize(x, ig.input_scale, 0.0, INT8)
    y = ig(qt).dequantize()
    assert float((y - F.gelu(x, approximate="tanh")).abs().mean()) < 0.08


# --- IAdd / ICat / IPool (scale alignment) -----------------------------------

def test_iadd_aligns_scales():
    torch.manual_seed(10)
    a = torch.randn(4, 6) * 2.0
    b = torch.randn(4, 6) * 0.5
    qa = QTensor.quantize(a, float(a.abs().max()) / 127.0, 0.0, INT8)
    qb = QTensor.quantize(b, float(b.abs().max()) / 127.0, 0.0, INT8)
    ref = qa.dequantize() + qb.dequantize()
    s_out = float(ref.abs().max()) / 127.0
    out = IAdd(s_out)(qa, qb)
    assert torch.allclose(out.dequantize(), ref, atol=3 * s_out)


def test_icat_aligns_and_concats():
    torch.manual_seed(11)
    parts = [torch.randn(2, 5) * s for s in (1.0, 3.0)]
    qparts = [QTensor.quantize(p, float(p.abs().max()) / 127.0, 0.0, INT8) for p in parts]
    ref = torch.cat([q.dequantize() for q in qparts], dim=-1)
    s_out = float(ref.abs().max()) / 127.0
    out = ICat(s_out, dim=-1)(qparts)
    assert out.int_data.shape == (2, 10)
    assert torch.allclose(out.dequantize(), ref, atol=3 * s_out)


def test_ipool_integer_mean():
    torch.manual_seed(12)
    x = torch.randn(2, 7, 4) * 1.5  # [B, tokens, C]
    qt = QTensor.quantize(x, float(x.abs().max()) / 127.0, 0.0, INT8)
    ref = qt.dequantize().mean(dim=1)
    out = IPool(dim=1)(qt)
    assert out.int_data.shape == (2, 4)
    assert torch.allclose(out.dequantize(), ref, atol=2 * _scalar_scale(qt))


def _scalar_scale(qt):
    return float(qt.scale)


# --- end-to-end Q -> I conversion (whole model integer compute) --------------

def _frame_model(embed=48):
    from engine.model_factory import make_model

    return make_model({"target": "models.frame.FrameModel", "embed_dim": embed, "patch_size": 16,
                       "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}})


@pytest.mark.parametrize("weight_gran", ["per-tensor", "per-channel"])
def test_convert_to_integer_matches_q_tier(weight_gran):
    from quantization.calibrate import post_training_quantize
    from quantization.convert import convert_to_integer
    from quantization.spec import QuantScheme

    torch.manual_seed(0)
    model = _frame_model()
    img = torch.rand(3, 1, 64, 64)
    scheme = QuantScheme(
        weight=TensorQuantSpec(ch_axis=0, granularity=weight_gran),
        activation=TensorQuantSpec(ch_axis=-1),
    )
    model, _ = post_training_quantize(model, [img], scheme=scheme)
    n_qlin = sum(1 for m in model.modules() if isinstance(m, QLinear))
    with torch.no_grad():
        y_q = model(img)["box"].clone()

    model, replaced = convert_to_integer(model)
    assert len(replaced) == n_qlin > 0
    assert all(isinstance(m, ILinear) for m in replaced.values())
    assert sum(1 for m in model.modules() if isinstance(m, QLinear)) == 0
    with torch.no_grad():
        y_i = model(img)["box"]
    # ILinear reproduces QLinear bit-exactly -> whole model bit-exact
    assert torch.allclose(y_i, y_q, atol=1e-3), f"max diff {(y_i - y_q).abs().max():.2e}"


def test_full_integer_graph_close_to_fp():
    """Integer linears + integer-LUT nonlinears (GeLU/LN/Softmax) end to end vs fp."""
    from quantization.calibrate import post_training_quantize
    from quantization.convert import (
        calibrate_gelu_luts,
        calibrate_layernorm_luts,
        calibrate_softmax_luts,
        convert_to_integer,
    )

    torch.manual_seed(0)
    model = _frame_model()
    img = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        out_fp = model(img)["box"].clone()

    model, _ = post_training_quantize(model, [img])
    calibrate_gelu_luts(model, [img])
    calibrate_layernorm_luts(model, [img])
    calibrate_softmax_luts(model, [img])
    model, replaced = convert_to_integer(model)
    assert len(replaced) > 0
    with torch.no_grad():
        out_int = model(img)["box"]
    assert torch.isfinite(out_int).all()
    rel = float((out_int - out_fp).norm() / (out_fp.norm() + 1e-8))
    assert rel < 0.6, f"full integer-graph rel-err {rel:.4f}"
