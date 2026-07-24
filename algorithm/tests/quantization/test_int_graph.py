"""I-tier integer graph (Part C): QTensor + torch kernels bit-exact vs i_ops golden."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quantization.i_ops import dyadic_params, int_conv2d, int_matmul as g_int_matmul, requant as g_requant
from quantization.ilayers.int_functional import int_matmul as t_int_matmul, requant as t_requant
from quantization.ilayers.qtensor import QTensor
from quantization.scheme import INT8, UINT8


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
