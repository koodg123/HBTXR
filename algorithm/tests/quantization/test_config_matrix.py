"""Part B config-matrix verification: granularity x sym/asym x scale_type x calibration.

Exercises the configurable quant stack end to end and asserts every combination
produces a finite, non-catastrophic PTQ result. Also unit-checks the shared slot
grouping and the scale-representation properties (pot / dyadic).
"""
from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from quantization.calibrate import calibrate
from quantization.convert import insert_fake_quant
from quantization.grouping import from_slots, num_slots, to_slots
from quantization.observer import build_observer
from quantization.spec import CalibrationSpec, QuantScheme, TensorQuantSpec, apply_scale_type

REL_TOL = 0.25  # generous whole-graph int8 PTQ bound; catches blow-ups, not tuned fp gaps


def _make(embed: int = 48):
    from engine.model_factory import make_model

    return make_model({"target": "models.frame.FrameModel", "embed_dim": embed, "patch_size": 16,
                       "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}})


def _scheme(*, weight=None, activation=None) -> QuantScheme:
    w = replace(TensorQuantSpec(ch_axis=0), **(weight or {}))
    a = replace(TensorQuantSpec(ch_axis=-1), **(activation or {}))
    return QuantScheme(weight=w, activation=a)


def _ptq_relerr(scheme: QuantScheme) -> float:
    torch.manual_seed(0)
    model = _make()
    img = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        fp = model(img)["box"].clone()
    model, _ = insert_fake_quant(model, scheme=scheme)
    calibrate(model, [img])
    with torch.no_grad():
        q = model(img)["box"]
    assert torch.isfinite(q).all()
    return float((q - fp).norm() / (fp.norm() + 1e-8))


# --- grouping / scale-type unit properties -----------------------------------

@pytest.mark.parametrize("spec", [
    TensorQuantSpec(granularity="per-tensor"),
    TensorQuantSpec(granularity="per-channel", ch_axis=0),
    TensorQuantSpec(granularity="per-group", group_size=4),
    TensorQuantSpec(granularity="per-block", block_size=(3, 4)),
])
def test_grouping_roundtrip_exact(spec):
    w = torch.randn(12, 16)
    slots = to_slots(w, spec)
    assert slots.shape[0] == num_slots(w.shape, spec)
    assert torch.equal(from_slots(slots, w.shape, spec), w)


def test_scale_type_pot_exact_powers_of_two():
    scale = torch.tensor([0.013, 0.27, 1.9, 0.5])
    pot = apply_scale_type(scale, "pot")
    assert torch.allclose(torch.log2(pot), torch.round(torch.log2(pot)))


def test_scale_type_dyadic_is_dyadic():
    scale = torch.tensor([0.013, 0.27, 1.9, 0.5])
    dyadic = apply_scale_type(scale, "dyadic")
    # each value equals multiplier / 2^shift for some shift in [1, 24]
    for d in dyadic.tolist():
        shifts = torch.arange(1, 25, dtype=torch.float64)
        prod = d * torch.pow(2.0, shifts)
        assert bool(((prod - prod.round()).abs() < 1e-6).any())
    assert (dyadic - scale).abs().max() < 1e-2  # close approximation


def test_asymmetric_observer_zero_point_and_recon():
    torch.manual_seed(0)
    x = torch.rand(64, 8) * 3.0 + 1.0  # strictly positive -> asym should shift zp
    spec = TensorQuantSpec(granularity="per-channel", ch_axis=0, symmetric=False)
    obs = build_observer(spec)
    obs.observe(x)
    scale, zp = obs.qparams()
    assert scale.shape[0] == 64 and (scale > 0).all()
    assert (zp != 0).any()  # non-zero zero-points for an offset distribution


# --- configurable PTQ matrix --------------------------------------------------

MATRIX = {
    "baseline (per-tensor sym float minmax)": _scheme(),
    "weight per-channel": _scheme(weight={"granularity": "per-channel"}),
    "weight per-group(16)": _scheme(weight={"granularity": "per-group", "group_size": 16}),
    "weight per-channel dyadic": _scheme(weight={"granularity": "per-channel", "scale_type": "dyadic"}),
    "weight per-channel pot": _scheme(weight={"granularity": "per-channel", "scale_type": "pot"}),
    "weight per-channel mse": _scheme(weight={"granularity": "per-channel", "calibration": CalibrationSpec(method="mse")}),
    "weight per-tensor kl": _scheme(weight={"calibration": CalibrationSpec(method="kl")}),
    "activation asymmetric": _scheme(activation={"symmetric": False}),
    "activation percentile": _scheme(activation={"calibration": CalibrationSpec(method="percentile")}),
    "activation per-channel": _scheme(activation={"granularity": "per-channel"}),
    "combined (per-ch+asym+dyadic+percentile)": _scheme(
        weight={"granularity": "per-channel", "scale_type": "dyadic", "calibration": CalibrationSpec(method="percentile")},
        activation={"symmetric": False, "calibration": CalibrationSpec(method="percentile")}),
}


@pytest.mark.parametrize("name", list(MATRIX))
def test_config_matrix_ptq(name):
    rel = _ptq_relerr(MATRIX[name])
    assert rel < REL_TOL, f"{name}: PTQ rel-err {rel:.4f} >= {REL_TOL}"


def test_mixed_precision_override_ptq():
    """4-bit per-group qkv + asymmetric-percentile fc1 activation via overrides."""
    scheme = QuantScheme(
        weight=TensorQuantSpec(ch_axis=0, granularity="per-channel"),
        activation=TensorQuantSpec(ch_axis=-1),
        overrides=(
            ("attn.qkv", {"weight": {"bits": 4, "granularity": "per-group", "group_size": 16}}),
            ("mlp.fc1", {"activation": {"symmetric": False, "calibration": {"method": "percentile"}}}),
        ),
    )
    torch.manual_seed(0)
    model = _make()
    model, registry = insert_fake_quant(model, scheme=scheme)
    qkv = next(q for n, q in registry.items() if n.endswith("attn.qkv"))
    assert qkv.weight_fq.dtype.bits == 4 and qkv.weight_fq.spec.granularity == "per-group"
    assert _ptq_relerr(scheme) < REL_TOL
