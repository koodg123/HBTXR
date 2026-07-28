"""A1 step 2: ``ilayers/vit.py`` — the block forwards with no float on the datapath.

The per-op I tier gives every module a float I/O port, so a converted model runs through
the unmodified ``forward`` — but the tensor *between* two ops is float32, and a float
edge is a float bus. ``ITransformerBlock`` threads ``QTensor`` instead.

It is checked against ``i_block.replay_block_int``, the pure-Python oracle, ELEMENT-WISE.
That is only possible because the oracle exists: the graph is deliberately a different
function from the per-op model, so without an independent integer reference the only
available comparison is a tolerance — and the composition bugs this catches (a shared
intermediate grid for Q/K/V, a bias added after the requant) are invisible at any
tolerance a person would write by hand.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from quantization.calibrate import post_training_quantize
from quantization.convert import convert_model_to_integer
from quantization.i_block import replay_block_int
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.vit import IMlp, IMultiHeadAttention, ITransformerBlock
from quantization.scheme import INT8

from tests.quantization.test_i_block_oracle import (CHANNELS, TOKENS, _input_int,
                                                    spec_from_block)


@pytest.fixture(scope="module")
def converted():
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
    return blk, spec_from_block(blk), ITransformerBlock(blk).eval()


def _run(graph, x_int):
    with torch.no_grad():
        return graph(QTensor(x_int.to(torch.int32), scale=graph.input_scale,
                             zero_point=0.0, dtype=INT8))


# --- the headline: element-wise agreement with the independent oracle ---------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_the_torch_graph_equals_the_pure_python_oracle_exactly(converted, seed):
    """Not ``allclose``: every one of the 768 output integers must be identical."""
    blk, spec, graph = converted
    x_int = _input_int(spec, seed)
    gold = replay_block_int(spec, x_int.reshape(-1).tolist(), tokens=TOKENS, channels=CHANNELS)
    got = _run(graph, x_int).int_data.reshape(-1).tolist()
    assert got == gold, (
        f"{sum(1 for a, b in zip(got, gold) if a != b)}/{len(gold)} integers differ, "
        f"worst {max(abs(a - b) for a, b in zip(got, gold))}")


def test_the_graph_output_is_on_the_declared_grid(converted):
    blk, spec, graph = converted
    out = _run(graph, _input_int(spec, 0))
    assert out.scale == graph.output_scale == spec.mlp_residual.scale_out
    assert out.int_data.dtype == torch.int32
    assert int(out.int_data.min()) >= INT8.qmin and int(out.int_data.max()) <= INT8.qmax


# --- the actual claim: nothing on the datapath is a float tensor --------------

def test_no_float_tensor_crosses_the_datapath(converted):
    """Instrument every tensor op and assert none of them produced a float result.

    This is the property the module exists for, so it is checked mechanically rather
    than by reading: a ``__torch_function__`` mode sees every operation the forward
    performs. Scale arithmetic is python floats at *setup* — a dyadic multiplier and a
    shift — and must never become a float TENSOR on the wire.
    """
    from torch.overrides import TorchFunctionMode

    blk, spec, graph = converted
    offenders: list[str] = []

    class Watch(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                offenders.append(f"{getattr(func, '__name__', func)} -> {out.dtype}")
            return out

    x_int = _input_int(spec, 0)
    with torch.no_grad(), Watch():
        graph(QTensor(x_int.to(torch.int32), scale=graph.input_scale, zero_point=0.0,
                      dtype=INT8))
    assert not offenders, f"float tensors on the integer datapath: {offenders[:8]}"


def test_the_per_op_block_does_produce_float_tensors(converted):
    """The contrast that makes the test above mean something.

    If the per-op block were also float-free, the check above would pass for the wrong
    reason and prove nothing about the rewrite.
    """
    from torch.overrides import TorchFunctionMode

    blk, spec, graph = converted
    seen: list[str] = []

    class Watch(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                seen.append(str(out.dtype))
            return out

    x_int = _input_int(spec, 0)
    with torch.no_grad(), Watch():
        blk((x_int.to(torch.float32) * spec.attn_residual.scale_a).unsqueeze(0))
    assert seen, "the per-op block was expected to move float tensors between ops"


# --- structure mirrors models/blocks ------------------------------------------

def test_the_assembly_mirrors_the_float_block(converted):
    """One integer class per float class, reusing the already-verified kernels."""
    blk, spec, graph = converted
    assert isinstance(graph.attn, IMultiHeadAttention) and isinstance(graph.mlp, IMlp)
    # the kernels are the SAME objects, not copies: only the composition is new
    assert graph.norm1 is blk.norm1 and graph.norm2 is blk.norm2
    assert graph.attn.softmax is blk.attn.attn_softmax
    assert graph.attn.query.linear is blk.attn.qkv
    assert graph.attn.key.linear is blk.attn.qkv and graph.attn.value.linear is blk.attn.qkv
    assert graph.attn.proj.linear is blk.attn.proj
    assert graph.mlp.fc1.linear is blk.mlp.fc1 and graph.mlp.fc2.linear is blk.mlp.fc2


def test_qkv_gets_three_independent_requants(converted):
    """Q, K and V leave ONE accumulator by three different dyadic multipliers."""
    blk, spec, graph = converted
    a = graph.attn
    mults = {int(a.query.multiplier[0]), int(a.key.multiplier[0]), int(a.value.multiplier[0])}
    shifts = {int(a.query.shift[0]), int(a.key.shift[0]), int(a.value.shift[0])}
    assert len(mults) + len(shifts) > 2, (
        f"Q/K/V share requant constants ({mults}, {shifts}) - they were collapsed onto one grid")
    assert a.query.columns.start == 0
    assert a.key.columns.start == a.query.columns.stop
    assert a.value.columns.start == a.key.columns.stop


def test_the_input_port_refuses_a_wrong_grid(converted):
    """A silent bridge here would be a per-call float ratio - the thing being removed."""
    blk, spec, graph = converted
    x_int = _input_int(spec, 0)
    with pytest.raises(ValueError, match="requantize at the producer"):
        graph(QTensor(x_int.to(torch.int32), scale=graph.input_scale * 2.0,
                      zero_point=0.0, dtype=INT8))


def test_the_graph_tracks_the_per_op_block_within_two_lsb(converted):
    """Removing the float transport costs at most 2 LSB — a measured number.

    Notable in itself: the float edges were never buying precision, because the next
    consumer immediately re-quantized to int8 anyway. The value of this rewrite is
    deployability, not accuracy, and saying so requires having measured it.
    """
    blk, spec, graph = converted
    worst = 0
    for seed in range(6):
        x_int = _input_int(spec, seed)
        got = _run(graph, x_int).int_data.to(torch.float64)
        with torch.no_grad():
            ref = blk((x_int.to(torch.float32) * spec.attn_residual.scale_a).unsqueeze(0))
        ref_int = torch.round(ref.squeeze(0).to(torch.float64) / graph.output_scale)
        worst = max(worst, int((got - ref_int).abs().max()))
    assert worst <= 2, f"worst deviation {worst} LSB"


def test_a_per_channel_weight_scale_is_applied_channel_by_channel(converted):
    """Per-channel constants must be BUILT and USED per channel.

    The shipped fixture has a per-TENSOR weight scale, so every multiplier is identical
    and indexing element 0 is an equivalent mutant there — which is exactly how this
    class of bug survives. The check therefore builds a genuinely per-channel linear and
    asserts the requant output differs from what channel 0's constants alone would give.
    """
    from quantization.ilayers.int_functional import requant
    from quantization.ilayers.vit import _IntLinear

    blk, spec, graph = converted
    linear = blk.attn.qkv
    original = linear.weight_scale.clone()
    try:
        out_features = linear.weight_int.shape[0]
        linear.weight_scale = torch.linspace(0.001, 0.05, out_features)
        built = _IntLinear(linear, 0.01)
        assert built.multiplier.numel() == out_features
        assert len(set(built.multiplier.tolist())) > 1, "channels share one multiplier"
        assert built.bias_int is not None and built.bias_int.numel() == out_features

        acc = torch.full((3, out_features), 1000, dtype=torch.int64)
        per_channel = built.requant(acc)
        channel_zero_only = requant(acc, built.multiplier[0], built.shift[0], dtype=INT8)
        assert not torch.equal(per_channel, channel_zero_only.to(torch.int64)), (
            "the requant collapsed every channel onto channel 0's constants")
    finally:
        linear.weight_scale = original
