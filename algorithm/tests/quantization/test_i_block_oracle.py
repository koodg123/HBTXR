"""A1 step 1: the pure-Python whole-block integer oracle.

``i_block.replay_block_int`` composes an entire transformer block from ``i_ops``
primitives only. It exists because the graph tier has no bit-exact oracle and this
repo's whole verification idiom assumes one: a QTensor-threaded block is deliberately a
different function from the per-op-integer model, so the only other option is
``allclose(tolerance)`` — and a tolerance wide enough to pass hides a wrong lowering.

It earned its keep immediately. The first draft requantized the qkv accumulator onto
Q's grid before splitting, so K and V were clamped through a grid never sized for them:
3.2% error at the qkv output, invisible to any end-to-end bound anyone would have
written by hand. ``test_each_composition_rule_is_load_bearing`` pins that rule and two more.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from quantization.calibrate import post_training_quantize
from quantization.convert import convert_model_to_integer
from quantization.i_block import (AddSpec, BlockSpec, LayerNormSpec, LinearSpec,
                                  MatMulSpec, SoftmaxSpec, TableSpec, rescale,
                                  replay_block_int)
from quantization.scheme import INT8

TOKENS, CHANNELS = 16, 48


def spec_from_block(blk) -> BlockSpec:
    """Read a converted ``TransformerBlock`` into the oracle's spec.

    Lives in the test, not in ``i_block``: that module must not import ``ilayers``, or
    the golden would share code with the thing it validates.
    """
    a = blk.attn

    def ln(m):
        base = [m.c_1_m, m.c_1_s, m.b, m.s1, m.bound, m.s2, m.clamp_bits]
        two = [m.b_two, m.s1_two, m.bound_two] if m.segments == 2 else []
        return LayerNormSpec([int(v) for v in base + two], m.lnw.tolist(), m.lnb.tolist(),
                             m.rsqrt_table.tolist(),
                             m.rsqrt_table_two.tolist() if m.segments == 2 else None,
                             float(m.input_scale), float(m.output_scale))

    def lin(m):
        return LinearSpec(m.weight_int.tolist(),
                          [float(v) for v in m.weight_scale.reshape(-1)],
                          float(m.act_scale),
                          None if m.bias is None else m.bias.tolist())

    sm, g = a.attn_softmax, getattr(blk.mlp.act, "kernel", blk.mlp.act)
    return BlockSpec(
        a.num_heads, a.head_dim, ln(blk.norm1), lin(a.qkv),
        MatMulSpec(a.qk_matmul.scale_a, a.qk_matmul.scale_b), float(a.attn_scale.factor),
        SoftmaxSpec([int(v) for v in (sm.b1, sm.s1, sm.bound1, sm.b2_one, sm.s2_one,
                                      sm.bound2_one, sm.b3_one, sm.s3_one, sm.b2_two,
                                      sm.s2_two, sm.bound2_two, sm.b3_two, sm.s3_two,
                                      sm.clamp_bits)],
                    sm.exp_table.tolist(), sm.recip_table_one.tolist(),
                    sm.recip_table_two.tolist(), float(sm.input_scale), float(sm.output_scale)),
        MatMulSpec(a.av_matmul.scale_a, a.av_matmul.scale_b), lin(a.proj),
        AddSpec(blk.attn_residual.scale_a, blk.attn_residual.scale_b, blk.attn_residual.scale_out),
        ln(blk.norm2), lin(blk.mlp.fc1),
        TableSpec([int(g.b), int(g.s), int(g.bound)], g.table.tolist(),
                  float(g.input_scale), float(g.output_scale)),
        lin(blk.mlp.fc2),
        AddSpec(blk.mlp_residual.scale_a, blk.mlp_residual.scale_b, blk.mlp_residual.scale_out))


@pytest.fixture(scope="module")
def block_and_spec():
    from engine.model_factory import make_model

    torch.manual_seed(0)
    model = make_model({"target": "models.frame.FrameModel", "embed_dim": CHANNELS,
                        "patch_size": 16,
                        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0,
                                     "cut_point": 1}})
    calib = [torch.rand(2, 1, 64, 64, generator=torch.Generator().manual_seed(7))]
    model, _ = post_training_quantize(model, calib)
    model, _ = convert_model_to_integer(model, calib)
    blk = model.backbone.blocks[0].eval()
    return blk, spec_from_block(blk)


def _input_int(spec, seed: int):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(TOKENS, CHANNELS, generator=gen) * 0.05
    return torch.round(x / spec.attn_residual.scale_a).clamp(-128, 127).to(torch.int64)


# On this input the oracle reproduces the live block EXACTLY, which makes it the right
# probe for a composition error: any deviation at all is the perturbation, not noise.
EXACT_SEED = 2


def _differing_elements(blk, spec, x_int) -> int:
    """How many output integers the oracle and the live block disagree on."""
    out = replay_block_int(spec, x_int.reshape(-1).tolist(), tokens=TOKENS, channels=CHANNELS)
    with torch.no_grad():
        ref = blk((x_int.to(torch.float32) * spec.attn_residual.scale_a).unsqueeze(0)).squeeze(0)
    ref_int = torch.round(ref.to(torch.float64) / spec.mlp_residual.scale_out)
    got_int = torch.tensor(out, dtype=torch.float64).reshape(TOKENS, CHANNELS)
    return int((got_int != ref_int).sum())


def _deviation_lsb(blk, spec, x_int) -> int:
    """Worst |difference| in OUTPUT-GRID INTEGERS between the oracle and the live block."""
    out = replay_block_int(spec, x_int.reshape(-1).tolist(), tokens=TOKENS, channels=CHANNELS)
    with torch.no_grad():
        ref = blk((x_int.to(torch.float32) * spec.attn_residual.scale_a).unsqueeze(0)).squeeze(0)
    ref_int = torch.round(ref.to(torch.float64) / spec.mlp_residual.scale_out)
    got_int = torch.tensor(out, dtype=torch.float64).reshape(TOKENS, CHANNELS)
    return int((got_int - ref_int).abs().max())


# --- the oracle is built from the goldens, and nothing else -------------------

def test_the_oracle_does_not_import_the_thing_it_validates():
    """A golden that shares code with its subject proves less. Enforce it statically."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "quantization" / "i_block.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any(name.startswith("quantization.ilayers") for name in imported), imported
    assert not any(name == "torch" or name.startswith("torch.") for name in imported), imported
    assert "quantization.i_ops" in imported, "the oracle must be built from the goldens"


# --- what the oracle proves about the composition -----------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_the_integer_graph_tracks_the_per_op_block_within_two_lsb(block_and_spec, seed):
    """No float anywhere on the datapath, and the result stays within 2 LSB.

    2 is a MEASURED bound, not a comfortable one: across these seeds the deviation is 0
    on three and at most 2 on the rest, matching the ~1 LSB per-channel dyadic requant
    cost the Part D plan predicted. A bound loose enough to feel safe is a bound that
    hides the next composition bug — which is the entire reason this file exists.
    """
    blk, spec = block_and_spec
    assert _deviation_lsb(blk, spec, _input_int(spec, seed)) <= 2


def test_the_oracle_is_exact_on_the_probe_input(block_and_spec):
    """The baseline the perturbation tests below stand on."""
    blk, spec = block_and_spec
    assert _differing_elements(blk, spec, _input_int(spec, EXACT_SEED)) == 0


@pytest.mark.parametrize("name,break_it", [
    ("K and V routed through Q's grid",
     lambda s: (setattr(s.qk, "scale_b", s.qk.scale_a), setattr(s.av, "scale_b", s.qk.scale_a))),
    ("the 1/sqrt(d) fold dropped",
     lambda s: setattr(s, "attn_scale", 1.0)),
    ("bias not added on the accumulator",
     lambda s: [setattr(l, "bias", None) for l in (s.qkv, s.proj, s.fc1, s.fc2)]),
])
def test_each_composition_rule_is_load_bearing(block_and_spec, name, break_it):
    """Break one rule of the scale contract; exactness must be lost.

    Measured at the BLOCK OUTPUT these perturbations move the worst element by only
    1-2 LSB — the residual join re-anchors the stream on the next calibrated grid, so an
    attention-path error is heavily attenuated by the time it leaves the block. That is
    a real property (and the reason a max-error bound is a poor probe here), but it is
    also why this asserts on EXACTNESS rather than on a magnitude: against an exact
    baseline the perturbation is unmissable — 300+ of 768 elements move.
    """
    blk, spec = block_and_spec
    x_int = _input_int(spec, EXACT_SEED)
    broken = copy.deepcopy(spec)
    break_it(broken)
    differing = _differing_elements(blk, broken, x_int)
    assert differing > 100, f"{name}: only {differing}/768 elements moved"


def test_rescale_is_an_integer_operation_with_an_identity_shortcut():
    """The per-edge requant must be what a requant unit does, not a float divide."""
    values = list(range(-100, 101))
    same = rescale(values, 0.02, 0.02, dtype=INT8)
    assert same == [max(-128, min(127, v)) for v in values], "identity must not round-trip"

    halved = rescale(values, 0.01, 0.02, dtype=INT8)
    assert halved[values.index(100)] == 50, "a 1:2 ratio must halve"
    assert all(-128 <= v <= 127 for v in halved)

    doubled = rescale(values, 0.02, 0.01, dtype=INT8)
    assert doubled[0] == -128 and doubled[-1] == 127, "a 2:1 ratio must saturate the ends"
    assert all(isinstance(v, int) for v in doubled), "the datapath must stay integer"
