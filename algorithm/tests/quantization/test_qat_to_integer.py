"""D6/B4: a QAT-fine-tuned model must survive the whole Q -> I conversion.

Nothing structural blocks it — ``prepare_qat`` is ``insert_fake_quant`` + ``calibrate``,
and the scales are then frozen, so the calibration state the converter needs is still
present after training. But "nothing blocks it" was an argument, not a measurement, and
QAT is the path a real deployment takes. These tests make it a fact.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from common.optim.registry import build_optimizer
from engine.train.trainer import Trainer, TrainConfig
from quantization.convert import convert_model_to_integer
from quantization.export import export_integer_model, verify_export
from quantization.ilayers import ILayerNorm, ILinear, ISoftmax
from quantization.qat import prepare_qat
from quantization.qlayers.linear import QLinear

_CFG = {"target": "models.frame.FrameModel", "embed_dim": 48, "patch_size": 16,
        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}}


def _batches(n: int = 3, seed: int = 5):
    gen = torch.Generator().manual_seed(seed)
    return [{"image": torch.rand(2, 1, 64, 64, generator=gen),
             "box": torch.rand(2, 5, generator=gen)} for _ in range(n)]


def _qat_model():
    """PTQ-calibrate, then actually fine-tune — the state the converter will consume."""
    from engine.model_factory import make_model

    torch.manual_seed(0)
    model = make_model(_CFG)
    batches = _batches()
    calib = [b["image"] for b in batches]
    model, registry = prepare_qat(model, calib_batches=calib)
    before = next(p.detach().clone() for n, p in model.named_parameters()
                  if "attn.qkv" in n and n.endswith("weight"))
    optimizer, _r, _meta, _s = build_optimizer(model, {"training": {"lr": 1e-2}})
    Trainer(model, optimizer, TrainConfig(modality="frame", epochs=1, device="cpu")).fit(batches)
    after = next(p.detach().clone() for n, p in model.named_parameters()
                 if "attn.qkv" in n and n.endswith("weight"))
    assert not torch.equal(before, after), "fine-tuning did not move the weights"
    return model, calib, registry


def test_qat_weights_move_but_scales_stay_frozen():
    """Fixed-scale QAT: training updates weights, never the calibrated quantizer scales.

    ``scale > 0`` would be satisfied by the *uncalibrated* identity default of 1.0, so
    the assertion is that every scale actually MOVED off that default — otherwise
    ``prepare_qat`` skipping its calibration step would pass this test, and the
    converter downstream would silently inherit identity scales.
    """
    model, calib, registry = _qat_model()
    assert registry, "no QLinear was registered"
    for name, quant in registry.items():
        assert quant.weight_fq.scale.requires_grad is False, f"{name} weight scale is trainable"
        assert quant.act_fq.scale.requires_grad is False, f"{name} act scale is trainable"
        act = float(quant.act_fq.scale.reshape(-1)[0])
        weight = float(quant.weight_fq.scale.reshape(-1)[0])
        assert act > 0.0 and act != 1.0, f"{name} activation scale is the uncalibrated default"
        assert weight > 0.0 and weight != 1.0, f"{name} weight scale is the uncalibrated default"

    # ... and the weight scale is the one the weight's own range implies, so a scale that
    # merely changed (rather than being calibrated) would not satisfy this either.
    for name, quant in registry.items():
        expected = float(quant.linear.weight.detach().abs().max()) / quant.weight_fq.dtype.qmax
        got = float(quant.weight_fq.scale.reshape(-1).max())
        assert got == pytest.approx(expected, rel=0.35), (
            f"{name}: weight scale {got:.3e} is not derived from the weight range {expected:.3e}")


def test_a_qat_model_converts_to_integer_and_verifies(tmp_path):
    """The whole point: QAT -> I tier -> export -> replay, with nothing skipped."""
    model, calib, _registry = _qat_model()
    model.eval()
    with torch.no_grad():
        before = model(calib[0])["box"].clone()

    model, report = convert_model_to_integer(model, calib)
    assert sum(1 for m in model.modules() if isinstance(m, QLinear)) == 0
    assert sum(1 for m in model.modules() if isinstance(m, ILinear)) == 14
    assert sum(1 for m in model.modules() if isinstance(m, ILayerNorm)) == 5
    assert sum(1 for m in model.modules() if isinstance(m, ISoftmax)) == 2
    assert not any(isinstance(m, (nn.Linear, nn.LayerNorm, nn.GELU, nn.Softmax))
                   for m in model.modules())

    out = tmp_path / "qat_int"
    manifest = export_integer_model(model, out)
    assert manifest["unexported"] == []
    ok, diff = verify_export(model, out)
    assert ok, f"QAT dump failed to replay (max abs diff {diff})"

    with torch.no_grad():
        after = model(calib[0])["box"]
    assert torch.isfinite(after).all()
    rel = float((after - before).norm() / (before.norm() + 1e-8))
    # The Q tier already fake-quantized every Linear, so the I tier reproduces it
    # closely; the gap is the integer nonlinears replacing float ones.
    assert rel < 0.35, f"QAT model moved {rel:.4f} on conversion"


def test_conversion_after_qat_matches_conversion_after_ptq_in_structure():
    """QAT must not produce a structurally different graph than PTQ — only different weights."""
    from engine.model_factory import make_model

    from quantization.calibrate import post_training_quantize

    qat_model, calib, _ = _qat_model()
    qat_model, qat_report = convert_model_to_integer(qat_model, calib)

    torch.manual_seed(0)
    ptq_model = make_model(_CFG)
    ptq_model, _ = post_training_quantize(ptq_model, calib)
    ptq_model, ptq_report = convert_model_to_integer(ptq_model, calib)

    assert set(qat_report.stages) == set(ptq_report.stages)
    assert {k: len(v) for k, v in qat_report.stages.items()} == \
           {k: len(v) for k, v in ptq_report.stages.items()}
    assert set(qat_report.replaced) == set(ptq_report.replaced)
    assert set(qat_report.left_float) == set(ptq_report.left_float)
