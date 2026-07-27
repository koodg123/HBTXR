"""Whole-graph Q -> I conversion of the ViT: is the "integer model" actually integer?

``convert_to_integer`` only ever replaced ``QLinear``, so a model that had been through
the advertised pipeline still ran LayerNorm, Softmax, GeLU and the patch conv in float
(or in the Q tier, which reduces in float and only looks the nonlinearity up in a
table). These tests hold ``convert_model_to_integer`` to the stronger claim and, just as
importantly, hold its *report* to the truth: what it says is still float has to be
exactly what a walk of the converted module tree finds.

Every accuracy bound below is a measured number, not a wish — the value that was
observed on the fixture, with the margin stated next to it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from quantization.calibrate import post_training_quantize
from quantization.convert import (
    ConversionReport,
    IAddFloatIO,
    IGeLUFloatIO,
    calibrate_gelu_luts,
    calibrate_int_layernorms,
    calibrate_int_softmaxes,
    calibrate_layernorm_luts,
    calibrate_softmax_luts,
    convert_model_to_integer,
    convert_to_integer,
    float_modules,
)
from quantization.ilayers import IAdd, IConv2d, IGeLU, ILayerNorm, ILinear, IMatMul, ISoftmax
from quantization.qlayers.linear import QLinear
from quantization.qlayers.nonlinear import QGeLU, QLayerNorm, QSoftmax

FLOAT_OPS = (nn.Linear, nn.GELU, nn.LayerNorm, nn.Softmax)
Q_TIER_OPS = (QLinear, QGeLU, QLayerNorm, QSoftmax)
# Modules whose arithmetic is integer. Kept here, written out by hand, so the test does
# not inherit the production ``INTEGER_MODULE_TYPES`` it is supposed to be checking.
INTEGER_OPS = (IConv2d, IGeLU, IGeLUFloatIO, ILayerNorm, ILinear, ISoftmax,
               IMatMul, IAdd, IAddFloatIO)


def _frame_model(seed: int = 0):
    from engine.model_factory import make_model

    torch.manual_seed(seed)
    return make_model({"target": "models.frame.FrameModel", "embed_dim": 48, "patch_size": 16,
                       "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}})


def _batches(seed: int = 100, n: int = 4):
    gen = torch.Generator().manual_seed(seed)
    return [torch.rand(n, 1, 64, 64, generator=gen)]


def _converted(seed: int = 0, *, qtier_first: bool = False, **kwargs):
    """float model -> PTQ -> (optional Q-tier LUTs) -> full integer conversion."""
    calib = _batches()
    model = _frame_model(seed)
    model, _ = post_training_quantize(model, calib)
    if qtier_first:
        calibrate_gelu_luts(model, calib)
        calibrate_layernorm_luts(model, calib)
        calibrate_softmax_luts(model, calib)
    model, report = convert_model_to_integer(model, calib, **kwargs)
    return model.eval(), report


def _count(model: nn.Module, types) -> int:
    return sum(1 for m in model.modules() if isinstance(m, types))


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float((actual - reference).norm() / (reference.norm() + 1e-8))


# --- (a) the converted graph really is integer, by type inspection -----------

@pytest.mark.parametrize("qtier_first", [False, True])
def test_converted_graph_has_no_float_or_q_tier_ops(qtier_first):
    """No nn.Linear / GELU / LayerNorm / Softmax and no Q-tier module survives."""
    reference = _frame_model(0)
    expected = {t.__name__: _count(reference, t) for t in FLOAT_OPS}
    assert expected == {"Linear": 14, "GELU": 6, "LayerNorm": 5, "Softmax": 2}

    model, report = _converted(qtier_first=qtier_first)

    survivors = {name: type(m).__name__ for name, m in model.named_modules()
                 if isinstance(m, FLOAT_OPS + Q_TIER_OPS)}
    assert survivors == {}, f"still float / Q tier: {survivors}"
    # ... and the integer replacements are there one-for-one.
    assert _count(model, ILinear) == expected["Linear"]
    assert _count(model, IGeLU) == expected["GELU"]
    assert _count(model, ILayerNorm) == expected["LayerNorm"]
    assert _count(model, ISoftmax) == expected["Softmax"]
    assert set(report.stages) == {"layernorm", "softmax", "gelu", "conv",
                                  "matmul", "add", "linear"}
    assert len(report) == (14 + 6 + 5 + 2 + len(report.stages["conv"])
                           + len(report.stages["matmul"]) + len(report.stages["add"]))
    # the attention matmuls and both residual joins are integer now, not just reported
    assert _count(model, IMatMul) == 4 and _count(model, IAddFloatIO) == 4


def test_integer_modules_hold_integer_tables_and_weights():
    """Type inspection is not enough: the installed kernels must carry integer state.

    The table depths are asserted at their DEFAULT value too, because that default is a
    hardware cost figure (256 int16 words per rsqrt table, 256 per exp / reciprocal
    segment, 256 per GeLU table) and this model is too well conditioned for an accuracy
    test to notice it shrinking.
    """
    model, _ = _converted()
    for module in model.modules():
        if isinstance(module, ILayerNorm):
            assert module.rsqrt_table.dtype == torch.int64 and module.lnw.dtype == torch.int64
            assert module.rsqrt_table.numel() == 256
        elif isinstance(module, ISoftmax):
            assert module.exp_table.dtype == torch.int64
            assert module.recip_table_one.dtype == torch.int64
            assert module.exp_table.numel() == module.recip_table_one.numel() == 256
        elif isinstance(module, IGeLU):
            assert module.table.dtype == torch.int64
            assert module.table.numel() == 256
        elif isinstance(module, (ILinear, IConv2d)):
            assert module.weight_int.dtype == torch.int32
            assert torch.equal(module.weight_int, module.weight_int.round())


@pytest.mark.parametrize("entries", [64, 256])
def test_table_size_knobs_reach_the_installed_tables(entries):
    """The ``*_entries`` arguments are the LUT depth a BRAM has to hold.

    Accuracy cannot police this on the FrameModel — its LayerNorm variance spans less
    than 2:1, so even a 4-entry rsqrt table measures the same end-to-end error — so the
    knob is pinned structurally instead: the number a caller asks for is the number of
    words that end up in the module.
    """
    model, _ = _converted(layernorm_entries=entries, softmax_entries=entries,
                          gelu_entries=entries)
    for module in model.modules():
        if isinstance(module, ILayerNorm):
            assert module.rsqrt_table.numel() == entries and module.bound == entries - 1
        elif isinstance(module, ISoftmax):
            assert module.exp_table.numel() == entries
            assert module.recip_table_one.numel() == entries
            assert module.recip_table_two.numel() == entries
        elif isinstance(module, IGeLU):
            assert module.table.numel() == entries


def test_converted_layernorm_and_softmax_run_on_the_integer_datapath():
    """The installed ILayerNorm/ISoftmax emit integer-valued data, not float reductions."""
    from quantization.ilayers import QTensor

    model, _ = _converted()
    norm = next(m for m in model.modules() if isinstance(m, ILayerNorm))
    x = torch.randn(3, norm.channels, generator=torch.Generator().manual_seed(1))
    out = norm.forward_int(QTensor.quantize(x, norm.input_scale, 0.0, norm.in_dtype))
    assert out.int_data.dtype == torch.int32
    assert int(out.int_data.min()) >= norm.out_dtype.qmin
    assert int(out.int_data.max()) <= norm.out_dtype.qmax

    softmax = next(m for m in model.modules() if isinstance(m, ISoftmax))
    logits = torch.randint(-60, 60, (4, 16), dtype=torch.int64)
    probs = softmax.forward_int(logits)
    assert probs.dtype == torch.int64 and int(probs.min()) >= 0
    assert int(probs.max()) <= (1 << softmax.clamp_bits) - 1


# --- (b) end-to-end accuracy against the ORIGINAL float model ----------------

def test_converted_graph_tracks_the_float_model():
    """MEASURED on this fixture (model seed 0, eval seed 7): rel-err 0.0557, max abs
    0.0196 against the untouched float model. The bounds asserted are 0.08 and 0.03,
    ~1.4x and ~1.5x the measured values; the fixture is fully seeded, so the margin is
    for a torch version change, not for flakiness.

    Honest context, because the headline number alone would flatter this pass: over
    model seeds 0..7 the same pipeline measured 0.0557 0.0242 0.0094 0.0103 0.0150
    0.0253 0.0237 0.1037. The 0.1037 outlier at seed 7 is NOT the integer nonlinears —
    converting only the Linears already costs 0.1016 there, i.e. it is one-batch PTQ
    calibration on a randomly-initialized model, and the whole I tier adds 2% on top of
    it. Seed 0 is asserted here because it is the fixture the rest of the file uses;
    ``test_each_conversion_stage_stays_within_budget`` is what pins the I tier's own
    share of the error.
    """
    evaluation = torch.rand(3, 1, 64, 64, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        expected = _frame_model(0).eval()(evaluation)["box"].clone()

    model, _ = _converted()
    with torch.no_grad():
        actual = model(evaluation)["box"]

    assert torch.isfinite(actual).all()
    rel = _relative_error(actual, expected)
    assert rel < 0.08, f"integer graph rel-err {rel:.4f}"
    assert float((actual - expected).abs().max()) < 0.03
    # the reference is not degenerate (a constant output would make rel-err meaningless)
    assert float(expected.std()) > 0.05
    # and the conversion is not a no-op: an integer graph does not match float exactly
    assert not torch.equal(actual, expected)


def test_each_conversion_stage_stays_within_budget():
    """How much of the error the *nonlinear* I tier is responsible for.

    MEASURED (model seed 0, eval seed 7): Linear only 0.0459, +conv 0.0517, +the three
    integer nonlinears 0.0557 — so installing ILayerNorm/ISoftmax/IGeLU multiplies the
    error by 1.21. Across model seeds 0..7 that multiplier measured 1.21 1.15 1.54 1.36
    1.52 0.92 1.71 1.02, worst 1.71. The bound below is 1.6x on the seed-0 fixture
    (1.3x margin over its measurement); it is the claim that matters — the integer
    nonlinears are a correction on top of linear quantization, not the dominant term.
    """
    evaluation = torch.rand(3, 1, 64, 64, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        expected = _frame_model(0).eval()(evaluation)["box"].clone()

    linear_only, report = _converted(include_nonlinear=False, include_conv=False)
    with torch.no_grad():
        rel_linear = _relative_error(linear_only(evaluation)["box"], expected)
    assert set(report.stages) == {"matmul", "add", "linear"}
    assert rel_linear < 0.07, f"linear-only rel-err {rel_linear:.4f}"

    full, _ = _converted()
    with torch.no_grad():
        rel_full = _relative_error(full(evaluation)["box"], expected)
    assert rel_full < 0.08, f"full-graph rel-err {rel_full:.4f}"
    assert rel_full < 1.6 * rel_linear, f"nonlinear tier multiplied the error by {rel_full / rel_linear:.2f}"


# --- (c) determinism ---------------------------------------------------------

def test_conversion_is_deterministic():
    evaluation = torch.rand(2, 1, 64, 64, generator=torch.Generator().manual_seed(11))
    first, report_a = _converted(seed=3)
    second, report_b = _converted(seed=3)
    with torch.no_grad():
        a = first(evaluation)["box"]
        b = second(evaluation)["box"]
    assert torch.equal(a, b)
    assert sorted(report_a) == sorted(report_b)
    assert sorted(report_a.left_float) == sorted(report_b.left_float)
    norm_a = next(m for m in first.modules() if isinstance(m, ILayerNorm))
    norm_b = next(m for m in second.modules() if isinstance(m, ILayerNorm))
    assert torch.equal(norm_a.rsqrt_table, norm_b.rsqrt_table)
    assert norm_a.input_scale == norm_b.input_scale


# --- (d) the report is honest ------------------------------------------------

def _walk_float_leaves(model: nn.Module) -> dict[str, str]:
    """Independent walk: every childless module that is not an integer kernel."""
    found: dict[str, str] = {}
    for name, module in model.named_modules():
        if next(module.children(), None) is not None:
            continue
        if isinstance(module, INTEGER_OPS):
            continue
        found[name] = type(module).__name__
    return found


@pytest.mark.parametrize("include_nonlinear", [True, False])
def test_left_float_is_exactly_what_is_still_float(include_nonlinear):
    model, report = _converted(include_nonlinear=include_nonlinear)
    walked = _walk_float_leaves(model)
    assert {name: type(m).__name__ for name, m in report.left_float.items()} == walked
    # every reported entry is the live object in the tree, not a stale copy
    live = dict(model.named_modules())
    for name, module in report.left_float.items():
        assert live[name] is module
    # nothing is claimed both converted and left float
    assert not set(report.replaced) & set(report.left_float)


def test_every_conv_including_the_padded_mask_conv_converts():
    """D2 gave ``IConv2d`` padding, so the 3x3/pad-1 mask conv is no longer refused.

    This test used to assert the opposite — that ``mask_head.proj`` stayed float and was
    honestly reported. That was true and worth pinning while the I tier could not express
    padding; now it can, and the thing worth pinning is that NO conv is left behind.
    """
    model, report = _converted()
    convs = {name: m for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, IConv2d))}
    still_float = {name: type(m).__name__ for name, m in convs.items() if not isinstance(m, IConv2d)}
    assert not still_float, f"conv left in float after conversion: {still_float}"
    # the padded one specifically, since that is the geometry D2 unlocked
    assert isinstance(model.mask_head.proj, IConv2d)
    assert model.mask_head.proj.padding == (1, 1), "padding must survive the conversion"
    assert isinstance(model.mask_head.to_logits, IConv2d)
    assert not any(isinstance(m, nn.Conv2d) for m in report.left_float.values())


def test_skipping_the_nonlinear_stage_is_reported_not_hidden():
    model, report = _converted(include_nonlinear=False, include_conv=False)
    left = {name: type(m).__name__ for name, m in report.left_float.items()}
    assert sorted(n for n, t in left.items() if t == "LayerNorm") == [
        "backbone.blocks.0.norm1", "backbone.blocks.0.norm2",
        "backbone.blocks.1.norm1", "backbone.blocks.1.norm2", "backbone.norm",
    ]
    assert sum(1 for t in left.values() if t == "GELU") == 6
    assert sum(1 for t in left.values() if t == "Softmax") == 2
    assert set(report.stages) == {"matmul", "add", "linear"}


def test_report_names_the_float_composites_the_swap_cannot_reach():
    """The attention matmuls / residual adds live in a forward body, not a submodule."""
    model, report = _converted()
    kinds = {type(m).__name__ for m in report.float_composites.values()}
    assert "MultiHeadAttention" in kinds     # Q·Kᵀ, attn·V and the 1/sqrt(d) scaling
    assert "TransformerBlock" in kinds       # the two residual adds
    assert "" in report.float_composites     # the root model's own forward
    for name, module in report.float_composites.items():
        assert next(module.children(), None) is not None
        assert type(module).forward is not nn.Module.forward
    assert not set(report.float_composites) & set(report.left_float)


def test_float_modules_is_a_pure_inspection():
    before, _ = _converted()
    leaves_a, composites_a = float_modules(before)
    leaves_b, composites_b = float_modules(before)
    assert list(leaves_a) == list(leaves_b) and list(composites_a) == list(composites_b)
    assert _count(before, FLOAT_OPS + Q_TIER_OPS) == 0


def test_conversion_report_is_the_replaced_mapping():
    _, report = _converted()
    assert isinstance(report, ConversionReport)
    assert len(report) == len(report.replaced)
    name = next(iter(report))
    assert report[name] is report.replaced[name]
    assert dict(report.items()) == report.replaced
    assert "integer modules:" in report.summary()


# --- backward compatibility: convert_to_integer is still Linear-only ---------

def test_convert_to_integer_touches_linear_only():
    model = _frame_model(0)
    model, _ = post_training_quantize(model, _batches())
    model, replaced = convert_to_integer(model)
    assert len(replaced) == 14 and all(isinstance(m, ILinear) for m in replaced.values())
    assert _count(model, ILinear) == 14
    # everything else is untouched — this is the honesty gap the docstring now admits
    assert _count(model, nn.LayerNorm) == 5
    assert _count(model, nn.Softmax) == 2
    assert _count(model, nn.GELU) == 6


# --- the Q tier is a legal starting point, not an error ----------------------

def test_int_layernorms_convert_a_q_tier_layernorm():
    calib = _batches()
    model = _frame_model(0)
    calibrate_layernorm_luts(model, calib)
    assert _count(model, QLayerNorm) == 5

    inserted = calibrate_int_layernorms(model, calib)
    assert len(inserted) == 5
    assert _count(model, QLayerNorm) == 0 and _count(model, ILayerNorm) == 5
    # the affine parameters came through the Q tier intact: the integer lnw is the
    # original LayerNorm weight on a symmetric grid, so it matches sign for sign
    weight = _frame_model(0).backbone.norm.weight.detach()
    norm = model.backbone.norm
    assert norm.channels == weight.numel()
    assert torch.equal(torch.sign(norm.lnw.float()), torch.sign(weight))


def test_int_softmaxes_keep_a_q_tier_softmax_input_grid():
    calib = _batches()
    model = _frame_model(0)
    calibrate_softmax_luts(model, calib)
    qsoftmax = model.backbone.blocks[0].attn.attn_softmax
    assert isinstance(qsoftmax, QSoftmax)
    grid = float(qsoftmax.exp_in)

    inserted = calibrate_int_softmaxes(model, calib)
    assert len(inserted) == 2 and _count(model, QSoftmax) == 0
    isoftmax = model.backbone.blocks[0].attn.attn_softmax
    assert isinstance(isoftmax, ISoftmax)
    assert isoftmax.input_scale == pytest.approx(grid)
    # and that is a real constraint, not a coincidence: calibrating the same softmax
    # from the float model (no Q tier to inherit a grid from) picks a different one
    fresh = calibrate_int_softmaxes(_frame_model(0), calib)
    assert float(fresh["backbone.blocks.0.attn.attn_softmax"].input_scale) != pytest.approx(grid)


def test_q_tier_gelu_table_is_reused_verbatim():
    calib = _batches()
    model = _frame_model(0)
    inserted_q = calibrate_gelu_luts(model, calib)
    tables = {name: m.table.clone() for name, m in inserted_q.items()}

    from quantization.convert import calibrate_int_gelus

    inserted_i = calibrate_int_gelus(model, calib)
    assert set(inserted_i) == set(tables)
    for name, module in inserted_i.items():
        assert torch.equal(module.kernel.table, tables[name])


# --- softmax over a non-default axis -----------------------------------------

class _AxisSoftmax(nn.Module):
    """Softmax over dim=1 of a [B, K, T] tensor — the axis is NOT the trailing one."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(x)


def test_int_softmax_calibrates_over_the_declared_axis():
    """Rows must be read along ``dim``, not along the trailing axis.

    The fixture is deliberately anisotropic: along dim=1 the logits span ~12, along
    dim=-1 only ~0.02, and the two axes have different lengths (8 vs 6). The row LENGTH
    is what makes this observable — it sets ``max_tokens``, which sizes the reciprocal
    segments — so a flat 8-long row is the probe: reading rows off the 6-long axis
    calibrates the accumulator envelope 8/6 too small, the segment-two cursor saturates,
    and the row stops being normalized. Measured: |rowsum-1| = 0.0039 reading the
    declared axis, 0.1294 reading the trailing one.
    """
    gen = torch.Generator().manual_seed(5)
    base = torch.linspace(-6.0, 6.0, 8).reshape(1, 8, 1)
    x = base + torch.rand(4, 8, 6, generator=gen) * 0.02
    model = _AxisSoftmax(dim=1)

    inserted = calibrate_int_softmaxes(model, [x])
    assert len(inserted) == 1
    isoftmax = model.softmax
    assert isoftmax.dim == 1
    with torch.no_grad():
        out = isoftmax(x)
        flat = isoftmax(torch.full((2, 8, 6), 0.37))
    expected = torch.softmax(x, dim=1)
    assert float((out - expected).abs().max()) < 0.02, f"max {float((out - expected).abs().max()):.4f}"
    assert float((out.sum(dim=1) - 1.0).abs().max()) < 0.05
    drift = float((flat.sum(dim=1) - 1.0).abs().max())
    assert drift < 0.05, f"a flat row of the declared length drifts by {drift:.4f}"


# --- observation is bounded --------------------------------------------------

def test_layernorm_calibration_rows_are_capped(monkeypatch):
    import quantization.convert as convert_module

    seen: list[int] = []
    original = convert_module.calibrate_int_layernorm

    def spy(module, samples, **kwargs):
        seen.append(int(torch.as_tensor(samples).shape[0]))
        return original(module, samples, **kwargs)

    monkeypatch.setattr(convert_module, "calibrate_int_layernorm", spy)
    model = _frame_model(0)
    # 4 images x 16 tokens = 64 rows available per LayerNorm
    calibrate_int_layernorms(model, _batches(), max_rows=10)
    assert seen and set(seen) == {10}


def test_modules_never_reached_by_the_calibration_batches_stay_float():
    """No batch, no observation — the op must stay float and say so, not fake a table."""
    model = _frame_model(0)
    inserted = calibrate_int_layernorms(model, [])
    assert inserted == {}
    assert _count(model, nn.LayerNorm) == 5

    model, report = convert_model_to_integer(model, [])
    assert report.stages["layernorm"] == ()
    assert sum(1 for m in report.left_float.values() if isinstance(m, nn.LayerNorm)) == 5
