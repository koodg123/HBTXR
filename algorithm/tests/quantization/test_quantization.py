"""Quantization primitives + PTQ/insert/integer-equivalence tests (needs torch)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quantization import (
    AffineFakeQuantizer,
    INT8,
    QuantConfig,
    QLinear,
    insert_fake_quant,
    post_training_quantize,
    prepare_qat,
    qrange,
)
from quantization.ilayers.linear import verify_int_linear
from quantization.observer import MinMaxObserver


def _cfg(target, embed=48):
    return {"target": target, "embed_dim": embed, "patch_size": 16,
            "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}}


def _make(target):
    from engine.model_factory import make_model

    return make_model(_cfg(target))


def test_dtype_range():
    assert qrange(8, True) == (-128, 127)
    assert INT8.qmin == -128 and INT8.qmax == 127


def test_affine_fake_quant_ste_grad():
    fq = AffineFakeQuantizer(INT8, scale=0.1)
    x = torch.randn(4, 8, requires_grad=True)
    y = fq(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (y - x).abs().max().item() < 0.06


def test_insert_replaces_linears():
    model = _make("models.frame.FrameModel")
    n_lin = sum(1 for m in model.modules() if type(m).__name__ == "Linear")
    model, registry = insert_fake_quant(model)
    assert len(registry) == n_lin > 0
    assert all(isinstance(q, QLinear) for q in registry.values())
    assert model(torch.rand(2, 1, 64, 64))["box"].shape == (2, 5)


def test_ptq_close_to_fp():
    torch.manual_seed(0)
    model = _make("models.frame.FrameModel")
    x = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        out_fp = model(x)["box"].clone()
    model, _ = post_training_quantize(model, [x])
    with torch.no_grad():
        out_q = model(x)["box"]
    rel = (out_q - out_fp).norm() / (out_fp.norm() + 1e-8)
    assert torch.isfinite(out_q).all()
    assert float(rel) < 0.5


def test_integer_kernel_matches_fake_quant():
    torch.manual_seed(0)
    model = _make("models.frame.FrameModel")
    model, registry = insert_fake_quant(model)
    from quantization.calibrate import calibrate

    calibrate(model, [torch.rand(4, 1, 64, 64)])
    ql = next(iter(registry.values()))
    ok, diff = verify_int_linear(ql, torch.randn(3, ql.linear.in_features))
    assert ok, f"integer/fake-quant mismatch diff={diff}"


def test_prepare_qat_runs():
    model = _make("models.frame.FrameModel")
    model, registry = prepare_qat(model, calib_batches=[torch.rand(2, 1, 64, 64)])
    assert len(registry) > 0
    assert model(torch.rand(1, 1, 64, 64))["box"].shape == (1, 5)


def test_observer_symmetric_scale():
    obs = MinMaxObserver(INT8)
    obs.observe(torch.tensor([-2.0, 1.0, 0.5]))
    scale, zp = obs.qparams()
    assert zp == 0 and abs(scale - 2.0 / 127) < 1e-6


def test_gelu_lut_approximates_float():
    import torch.nn.functional as F

    from quantization.lut_calibrate import build_gelu_lut
    from quantization.qlayers.nonlinear import QGeLU

    torch.manual_seed(0)
    payload = build_gelu_lut((torch.randn(8192) * 1.5).numpy(), entries=256)
    lut = QGeLU(payload["scalars"], payload["table"],
                  input_scale=payload["input_scale"], output_scale=payload["output_scale"])
    x = torch.linspace(-3.0, 3.0, 400)
    err = (lut(x) - F.gelu(x, approximate="tanh")).abs()
    assert float(err.mean()) < 0.08


def test_calibrate_gelu_luts_replaces_gelu():
    from torch import nn

    from quantization.convert import calibrate_gelu_luts

    model = _make("models.frame.FrameModel")
    before = sum(1 for m in model.modules() if isinstance(m, nn.GELU))
    inserted = calibrate_gelu_luts(model, [torch.rand(2, 1, 64, 64)])
    assert before > 0 and len(inserted) == before
    assert sum(1 for m in model.modules() if isinstance(m, nn.GELU)) == 0
    assert model(torch.rand(1, 1, 64, 64))["box"].shape == (1, 5)


def test_layernorm_lut_replaces_and_close():
    from torch import nn

    from quantization.convert import calibrate_layernorm_luts

    torch.manual_seed(0)
    model = _make("models.frame.FrameModel")
    before = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    img = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        out_fp = model(img)["box"].clone()
    inserted = calibrate_layernorm_luts(model, [img])
    assert before > 0 and len(inserted) == before
    assert sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm)) == 0
    with torch.no_grad():
        out_lut = model(img)["box"]
    assert torch.isfinite(out_lut).all()
    assert float((out_lut - out_fp).norm() / (out_fp.norm() + 1e-8)) < 0.5


def test_softmax_lut_replaces_and_close():
    from torch import nn

    from quantization.convert import calibrate_softmax_luts

    torch.manual_seed(0)
    model = _make("models.frame.FrameModel")
    before = sum(1 for m in model.modules() if isinstance(m, nn.Softmax))
    img = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        out_fp = model(img)["box"].clone()
    inserted = calibrate_softmax_luts(model, [img])
    assert before > 0 and len(inserted) == before
    assert sum(1 for m in model.modules() if isinstance(m, nn.Softmax)) == 0
    with torch.no_grad():
        out_lut = model(img)["box"]
    assert torch.isfinite(out_lut).all()
    assert float((out_lut - out_fp).norm() / (out_fp.norm() + 1e-8)) < 0.5


def test_all_nonlinear_luts_together():
    from quantization.convert import calibrate_gelu_luts, calibrate_layernorm_luts, calibrate_softmax_luts

    torch.manual_seed(0)
    model = _make("models.frame.FrameModel")
    img = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        out_fp = model(img)["box"].clone()
    calibrate_gelu_luts(model, [img])
    calibrate_layernorm_luts(model, [img])
    calibrate_softmax_luts(model, [img])
    with torch.no_grad():
        out_lut = model(img)["box"]
    assert torch.isfinite(out_lut).all()
    assert float((out_lut - out_fp).norm() / (out_fp.norm() + 1e-8)) < 0.6
