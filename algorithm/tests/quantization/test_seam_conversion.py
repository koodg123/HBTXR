"""D4: the attention matmuls and residual joins are integer, not merely reported.

Before this, ``IMatMul`` / ``IAdd`` were built, tested, and used by nothing. A
conversion pass swaps children, and ``Q @ Kᵀ``, ``attn @ V`` and both residual ``+``
were written inline in ``MultiHeadAttention.forward`` / ``TransformerBlock.forward``,
so no swap could reach them — they stayed float forever and the kernels were dead code.

Promoting them to parameter-free seam modules is what makes them reachable. These tests
pin BOTH halves of that: the refactor must be free (same float output, same state_dict,
so every checkpoint still loads), and the conversion must actually replace them with
integer kernels rather than just naming them in a report.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from models.blocks.seams import Add, MatMul, Scale
from models.blocks.transformer_block import TransformerBlock
from quantization.calibrate import post_training_quantize
from quantization.convert import IAddFloatIO, convert_model_to_integer
from quantization.ilayers import IMatMul


def _frame_model(seed: int = 0):
    from engine.model_factory import make_model

    torch.manual_seed(seed)
    return make_model({"target": "models.frame.FrameModel", "embed_dim": 48, "patch_size": 16,
                       "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}})


def _converted(seed: int = 0, **kwargs):
    calib = [torch.rand(4, 1, 64, 64, generator=torch.Generator().manual_seed(100))]
    model = _frame_model(seed)
    model, _ = post_training_quantize(model, calib)
    model, report = convert_model_to_integer(model, calib, **kwargs)
    return model.eval(), report


# --- the refactor must cost nothing ------------------------------------------

def test_seam_modules_carry_no_state():
    """A seam that owned a parameter would change every checkpoint."""
    block = TransformerBlock(48, 2, mlp_ratio=2.0)
    seams = [(n, m) for n, m in block.named_modules() if isinstance(m, (MatMul, Add, Scale))]
    assert len(seams) == 5, [n for n, _ in seams]     # qk, av, attn_scale, 2 residuals
    for name, module in seams:
        assert list(module.parameters()) == [], f"{name} owns parameters"
        assert list(module.buffers()) == [], f"{name} owns buffers"
        assert module.state_dict() == {}, f"{name} contributes to the state_dict"


def test_seams_compute_exactly_what_they_replaced():
    """Each seam is the identity of its operator, to the bit."""
    torch.manual_seed(0)
    a, b = torch.randn(2, 3, 5, 7), torch.randn(2, 3, 7, 5)
    assert torch.equal(MatMul()(a, b), a @ b)
    x, y = torch.randn(4, 6), torch.randn(4, 6)
    assert torch.equal(Add()(x, y), x + y)
    assert torch.equal(Scale(0.125)(x), x * 0.125)


def test_attention_scale_is_a_power_of_two_for_the_shipped_head_dim():
    """head_dim=64 -> 1/sqrt(64) = 0.125 = 2^-3, a pure shift in hardware."""
    block = TransformerBlock(192, 3)               # head_dim 64, the shipped config
    factor = block.attn.attn_scale.factor
    assert factor == 0.125
    assert torch.log2(torch.tensor(factor)) == torch.round(torch.log2(torch.tensor(factor)))


# --- the conversion must reach them ------------------------------------------

def test_conversion_replaces_every_matmul_and_residual_add():
    model, report = _converted()
    assert sum(1 for m in model.modules() if isinstance(m, MatMul)) == 0
    assert sum(1 for m in model.modules() if isinstance(m, Add)) == 0
    assert sum(1 for m in model.modules() if isinstance(m, IMatMul)) == 4
    assert sum(1 for m in model.modules() if isinstance(m, IAddFloatIO)) == 4
    assert len(report.stages["matmul"]) == 4 and len(report.stages["add"]) == 4


def test_the_swapped_matmul_really_quantizes():
    """Type inspection is not proof: the output must land on the integer grid."""
    model, _ = _converted()
    mm = next(m for m in model.modules() if isinstance(m, IMatMul))
    torch.manual_seed(3)
    a, b = torch.randn(2, 4, 6) * 0.5, torch.randn(2, 6, 4) * 0.5
    out = mm(a, b)
    grid = mm.scale_a * mm.scale_b
    assert torch.allclose(out / grid, torch.round(out / grid), atol=1e-2), "not on the s_a*s_b grid"
    assert not torch.allclose(out, a @ b, atol=1e-6), "quantization had no effect at all"


def test_the_swapped_add_aligns_onto_one_output_scale():
    model, _ = _converted()
    add = next(m for m in model.modules() if isinstance(m, IAddFloatIO))
    torch.manual_seed(4)
    x, y = torch.randn(3, 8) * 0.2, torch.randn(3, 8) * 0.2
    out = add(x, y)
    assert torch.allclose(out / add.scale_out, torch.round(out / add.scale_out), atol=1e-4)
    # the two operands genuinely sit on different grids — that is why IAdd exists
    assert add.scale_a != add.scale_b


def test_include_seams_false_leaves_them_float_and_says_so():
    """The knob must be honest in both positions."""
    model, report = _converted(include_seams=False)
    assert "matmul" not in report.stages and "add" not in report.stages
    left = {type(m).__name__ for m in report.left_float.values()}
    assert {"MatMul", "Add"} <= left, f"seams vanished from the report: {left}"


def test_scale_stays_float_and_that_is_deliberate():
    """``Scale`` is an exact constant multiply — quantizing it would only add error.

    It is left in ``left_float`` rather than filtered out of the report, because the
    report is an observation of what is still float and hiding an inert entry is how a
    report starts lying. In a fully-integer datapath the factor folds into the
    requantization that follows it.
    """
    model, report = _converted()
    scales = {n: m for n, m in model.named_modules() if isinstance(m, Scale)}
    assert len(scales) == 2
    assert all(name in report.left_float for name in scales)
    x = torch.randn(2, 3, 4, 4)
    for module in scales.values():
        assert torch.equal(module(x), x * module.factor)     # exact, nothing to quantize
