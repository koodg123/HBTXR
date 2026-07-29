"""A1 remainder: the whole detector as one integer graph, checked against its oracle.

``ITransformerBlock`` made a block float-free, but ``DirectPupilDetector`` composes the
stem, the backbone norm and the heads outside the block and those edges were still
float32. ``IDirectPupilDetector`` closes them, and ``i_block.replay_model_int`` — pure
Python over ``i_ops``, arbitrary-precision, importing nothing from ``ilayers`` — is what
makes "closes them correctly" a checkable statement rather than a tolerance.

The parts the block replay never had to deal with, and which therefore get their own
tests here: the stem conv's zero-point correction and per-out-channel bias, the
block-to-block bridge, the integer token mean, and the head's final accumulator.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quantization.calibrate import post_training_quantize
from quantization.convert import convert_model_to_integer
from quantization.i_block import ConvSpec, HeadSpec, ModelSpec, replay_model_int
from quantization.ilayers.model import IDirectPupilDetector
from quantization.ilayers.qtensor import QTensor
from quantization.scheme import INT8

from tests.quantization.test_i_block_oracle import spec_from_block

EMBED, PATCH, SIDE = 48, 16, 64
GRID = SIDE // PATCH                       # 4x4 tokens


def _model(head: str = "bbox", *, in_chans: int = 1, modality: str = "frame"):
    from engine.model_factory import make_model

    torch.manual_seed(0)
    model = make_model({"target": "models.frame.FrameModel", "embed_dim": EMBED,
                        "patch_size": PATCH, "modality": modality, "head": head,
                        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0,
                                     "cut_point": 1}})
    calib = [torch.rand(2, in_chans, SIDE, SIDE,
                        generator=torch.Generator().manual_seed(7))]
    model, _ = post_training_quantize(model, calib)
    model, _ = convert_model_to_integer(model, calib)
    return model.eval()


def _linear_spec(m):
    from quantization.i_block import LinearSpec

    return LinearSpec(m.weight_int.tolist(),
                      [float(v) for v in m.weight_scale.reshape(-1)],
                      float(m.act_scale),
                      None if m.bias is None else m.bias.tolist())


def _model_spec(model) -> ModelSpec:
    """Read the converted model into the oracle's spec (in the test, not in i_block)."""
    from quantization.i_block import TableSpec

    conv = model.patch_embed.proj
    stem = ConvSpec(conv.weight_int.tolist(),
                    [float(v) for v in conv.weight_scale.reshape(-1)],
                    float(conv.act_scale), int(conv.act_zero_point),
                    conv.stride, conv.padding,
                    None if conv.bias is None else conv.bias.tolist())
    blocks = [spec_from_block(b) for b in model.backbone.blocks]
    norm = spec_from_block(model.backbone.blocks[0]).norm1
    ln = model.backbone.norm
    base = [ln.c_1_m, ln.c_1_s, ln.b, ln.s1, ln.bound, ln.s2, ln.clamp_bits]
    two = [ln.b_two, ln.s1_two, ln.bound_two] if ln.segments == 2 else []
    norm = type(norm)([int(v) for v in base + two], ln.lnw.tolist(), ln.lnb.tolist(),
                      ln.rsqrt_table.tolist(),
                      ln.rsqrt_table_two.tolist() if ln.segments == 2 else None,
                      float(ln.input_scale), float(ln.output_scale))
    reg = model.head.regressor
    gelu = getattr(reg[1], "kernel", reg[1])
    head = HeadSpec(_linear_spec(reg[0]),
                    TableSpec([int(gelu.b), int(gelu.s), int(gelu.bound)],
                              gelu.table.tolist(), float(gelu.input_scale),
                              float(gelu.output_scale)),
                    _linear_spec(reg[2]))
    return ModelSpec(stem, blocks, norm, head)


@pytest.fixture(scope="module")
def converted():
    model = _model()
    return model, _model_spec(model), IDirectPupilDetector(model).eval()


def _image_int(graph, seed: int) -> QTensor:
    gen = torch.Generator().manual_seed(seed)
    return graph.quantize_input(torch.rand(1, 1, SIDE, SIDE, generator=gen))


# --- the headline: element-wise agreement with the independent oracle ---------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_whole_model_equals_the_pure_python_oracle_exactly(converted, seed):
    """Every output integer identical — stem, both blocks, final norm, pool and head."""
    model, spec, graph = converted
    image = _image_int(graph, seed)
    gold = replay_model_int(spec, image.int_data.reshape(-1).tolist(),
                            channels=1, height=SIDE, width=SIDE)
    got = graph(image)["head"].int_data.reshape(-1).tolist()
    assert got == gold, f"{sum(1 for a, b in zip(got, gold) if a != b)}/{len(gold)} differ"


def test_no_float_tensor_crosses_the_model_datapath(converted):
    """The claim, checked mechanically over every op the forward performs.

    ``forward`` is the accelerator; quantizing the image and dequantizing the answer are
    the host's two multiplies and sit outside it, which is why this instruments the graph
    call and not ``forward_real``.
    """
    from torch.overrides import TorchFunctionMode

    model, spec, graph = converted
    image = _image_int(graph, 0)
    offenders: list[str] = []

    class Watch(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                offenders.append(f"{getattr(func, '__name__', func)} -> {out.dtype}")
            return out

    with torch.no_grad(), Watch():
        graph(image)
    assert not offenders, f"float tensors on the integer datapath: {offenders[:8]}"


def test_the_float_io_model_does_produce_float_tensors(converted):
    """The control: without it the check above could pass for the wrong reason."""
    from torch.overrides import TorchFunctionMode

    model, spec, graph = converted
    seen: list[str] = []

    class Watch(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                seen.append(str(out.dtype))
            return out

    with torch.no_grad(), Watch():
        model(torch.rand(1, 1, SIDE, SIDE, generator=torch.Generator().manual_seed(0)))
    assert seen, "the float-I/O model was expected to move float tensors between ops"


# --- the edges the block replay never had ------------------------------------

def test_the_conversion_pipeline_gives_convs_a_symmetric_input_grid(converted):
    """A finding, pinned so it is a decision on record rather than an assumption.

    ``calibrate_int_convs`` observes ``abs().amax()`` and leaves the zero-point at 0, so
    on the current pipeline no converted conv has an asymmetric input grid — which means
    ``IConv2d``'s zero-point correction and its zero-point padding, both of which D2 built
    and measured, are never exercised by a converted model. They are not wrong and not
    unused-by-mistake; they are what makes an asymmetric grid legal if one is ever
    calibrated. The next test drives that path directly, since conversion will not.
    """
    model, spec, graph = converted
    assert model.patch_embed.proj.act_zero_point == 0


def test_the_stem_carries_the_zero_point_through_the_weights():
    """An asymmetric activation grid must be corrected as ``- zp·Σw``, per out-channel.

    Checked against the arbitrary-precision golden, with the golden told to pad with the
    zero-point: on an asymmetric grid the integer standing for a real zero is ``zp``, not
    0, so padding with 0 feeds ``-zp·s_x`` into every border tap. Dropping either half is
    invisible by inspection — the result stays a plausible feature map.
    """
    from quantization.i_ops import int_conv2d
    from quantization.ilayers.conv import IConv2d

    gen = torch.Generator().manual_seed(2)
    w = torch.randint(-128, 128, (4, 3, 3, 3), generator=gen, dtype=torch.int64)
    conv = IConv2d(w.to(torch.int32), torch.ones(4), 0.02, INT8, None, stride=1,
                   padding=1, act_zero_point=17)
    x_int = torch.randint(-128, 128, (1, 3, 6, 6), generator=gen, dtype=torch.int64)

    got = conv.integer_accumulator(x_int)
    raw = int_conv2d(x_int[0].tolist(), w.tolist(), stride=1, padding=1, pad_value=17)
    wsum = [sum(v for plane in w[oc].tolist() for row in plane for v in row) for oc in range(4)]
    gold = [[[v - 17 * wsum[oc] for v in row] for row in plane]
            for oc, plane in enumerate(raw)]
    assert got[0].tolist() == gold

    conv.wsum = torch.zeros_like(conv.wsum)          # the correction, removed
    assert conv.integer_accumulator(x_int)[0].tolist() != gold


def test_the_block_to_block_edge_is_bridged_by_a_derived_ratio(converted):
    """One bridge per edge, and its ratio comes from the two blocks, not from a guess.

    On this model the ratio is exactly 1 and the bridge is an identity — necessarily so,
    because block k's output tensor and block k+1's input tensor are the SAME activation,
    observed once, so calibration cannot give them different scales. That is worth pinning
    rather than deleting the bridge over: the identity is derived, and the moment the two
    grids differ (a re-calibrated block, a modality split at the cut point) the edge has to
    requantize. The second half of this test forces that case.
    """
    from quantization.ilayers.vit import _Requant

    model, spec, graph = converted
    blocks = graph.backbone.blocks
    assert len(graph.backbone.bridges) == len(blocks) - 1
    for prev, nxt, bridge in zip(blocks[:-1], blocks[1:], graph.backbone.bridges):
        assert prev.output_scale == nxt.input_scale
        assert bridge.identity, "an equal-scale edge must not round-trip through a dyadic"

    real = _Requant(0.02, 0.01)
    assert not real.identity
    assert torch.equal(real(torch.tensor([[10, -10]], dtype=torch.int64)),
                       torch.tensor([[20, -20]], dtype=torch.int64))


@pytest.mark.parametrize("edge", ["block", "final_norm"])
def test_every_bridge_is_actually_wired_into_the_forward(converted, edge):
    """A bridge that does nothing measurable is indistinguishable from one never called.

    Neither of this graph's bridges is observable by deleting it — the block-to-block one
    because its ratio is exactly 1, the final-norm one for the reason the next test
    documents. So "is the requant correct" cannot be tested by removing it. What can be
    tested is that the forward routes through it: swap in a bridge with a ratio big enough
    to be visible, and the answer must move.
    """
    from quantization.ilayers.vit import _Requant

    model, spec, graph = converted
    image = _image_int(graph, 0)
    with torch.no_grad():
        baseline = graph(image)["head"].int_data.clone()
    holder, name = ((graph.backbone.bridges, "0") if edge == "block"
                    else (graph.backbone, "to_norm"))
    saved = getattr(holder, name)
    try:
        setattr(holder, name, _Requant(1.25, 1.0))
        with torch.no_grad():
            assert not torch.equal(graph(image)["head"].int_data, baseline), (
                f"the {edge} bridge is not on the datapath at all")
    finally:
        setattr(holder, name, saved)


def test_the_final_norm_bridge_is_real_but_below_the_norms_resolution(converted):
    """Why deleting THAT bridge changes nothing, stated rather than left as a puzzle.

    Two facts meet here. Its ratio is 1.0042, not 1: the last block's output and the
    norm's input are the same tensor, but ``calibrate_int_adds`` and
    ``calibrate_int_layernorms`` measure it separately and land a few tenths of a percent
    apart. And a LayerNorm divides its input by that input's own standard deviation, so a
    uniform rescale is normalized away — it survives only through the PoT LUT cursor,
    which a 0.4% shift does not move.

    That makes the requant invisible today and load-bearing tomorrow: measured on this
    fixture a 1.05x ratio already moves the output by 865 LSB. The bound is the reason the
    edge keeps its requant instead of being simplified away as dead weight.
    """
    from quantization.ilayers.vit import _Requant

    model, spec, graph = converted
    bridge = graph.backbone.to_norm
    ratio = graph.backbone.blocks[-1].output_scale / float(graph.backbone.norm.input_scale)
    assert not bridge.identity and 1.0 < ratio < 1.01

    image = _image_int(graph, 0)
    saved = graph.backbone.to_norm
    try:
        with torch.no_grad():
            baseline = graph(image)["head"].int_data.clone()
            graph.backbone.to_norm = _Requant(1.0, 1.0)          # the bridge, removed
            assert torch.equal(graph(image)["head"].int_data, baseline)
            graph.backbone.to_norm = _Requant(1.05, 1.0)         # just past the resolution
            assert int((graph(image)["head"].int_data - baseline).abs().max()) > 100
    finally:
        graph.backbone.to_norm = saved


def test_a_block_refuses_an_unbridged_input(converted):
    """The consumer rejecting a wrong grid is what forces the bridge to exist at all."""
    model, spec, graph = converted
    second = graph.backbone.blocks[1]
    wrong = QTensor(torch.zeros(1, GRID * GRID, EMBED, dtype=torch.int32),
                    scale=second.input_scale * 2, zero_point=0.0, dtype=INT8)
    with pytest.raises(ValueError, match="requantize at the producer"):
        second(wrong)


def test_the_input_port_refuses_a_wrong_grid_or_offset(converted):
    """Both halves of the port: an offset silently dropped is a pure bias error."""
    model, spec, graph = converted
    image = _image_int(graph, 0)
    with pytest.raises(ValueError, match="requantize at the producer"):
        graph(QTensor(image.int_data, scale=graph.input_scale * 2,
                      zero_point=graph.input_zero_point, dtype=INT8))
    with pytest.raises(ValueError, match="zero-point"):
        graph(QTensor(image.int_data, scale=graph.input_scale,
                      zero_point=graph.input_zero_point + 3, dtype=INT8))


def test_the_head_ends_on_its_accumulator_not_on_an_invented_grid(converted):
    """int32 out, one scale per output channel — what a regression head reads out."""
    model, spec, graph = converted
    out = graph(_image_int(graph, 0))["head"]
    assert out.int_data.dtype == torch.int32
    assert int(out.int_data.abs().max()) > INT8.qmax, (
        "the accumulator was requantized into int8 range; precision thrown away for nothing")
    fc2 = graph.head.mlp.fc2
    assert fc2.out_scale is None and fc2.acc_scale.numel() == out.int_data.shape[-1]
    with pytest.raises(ValueError, match="ends the graph"):
        fc2.requant(torch.zeros(1, 1, dtype=torch.int64))


def test_the_token_mean_is_exact_integer_arithmetic(converted):
    """IPool must round half away from zero in int64, matching the oracle."""
    from quantization.ilayers.tensor_ops import IPool

    values = torch.tensor([[[3], [4], [0], [0]]], dtype=torch.int32)       # sum 7, n 4
    pooled = IPool(dim=1)(QTensor(values, scale=0.1, zero_point=0.0, dtype=INT8))
    assert int(pooled.int_data.reshape(-1)[0]) == 2                        # (7+2)//4
    negative = IPool(dim=1)(QTensor(-values, scale=0.1, zero_point=0.0, dtype=INT8))
    assert int(negative.int_data.reshape(-1)[0]) == -2                     # away from zero


# --- structure ---------------------------------------------------------------

def test_the_assembly_mirrors_the_float_model(converted):
    """One integer class per float class, reusing the already-verified kernels."""
    model, spec, graph = converted
    assert graph.patch_embed.proj.conv is model.patch_embed.proj
    assert graph.backbone.norm is model.backbone.norm
    assert graph.head.mlp.gelu is getattr(model.head.regressor[1], "kernel",
                                          model.head.regressor[1])
    assert graph.head.mlp.fc2.linear is model.head.regressor[2]
    assert graph.roi_head is not None and graph.reliability_head is not None


def test_the_omitted_head_is_named_rather_than_silently_missing(converted):
    """A gap stated in the API beats a gap the reader has to infer from an absence.

    And the reason has to be the real one. This head is omitted because it does not run at
    inference — stage-1 auxiliary supervision, absent from ``HybridModel`` entirely — not
    because its bilinear upsample is hard to quantize. The wrong reason would have put a
    piece of work on the backlog that nobody owes.
    """
    model, spec, graph = converted
    assert "mask_head" in graph.float_io_heads
    assert "inference path" in graph.float_io_heads["mask_head"]
    assert model.mask_head is not None, "the float model does have the head this omits"


def test_the_host_boundary_produces_real_values(converted):
    """forward_real is the host's view: quantize once in, dequantize once out."""
    model, spec, graph = converted
    image = torch.rand(1, 1, SIDE, SIDE, generator=torch.Generator().manual_seed(3))
    out = graph.forward_real(image)
    assert out["head"].shape == (1, 5) and out["head"].dtype == torch.float32
    assert torch.all((out["reliability"] >= 0) & (out["reliability"] <= 1))


# --- the ellipse head, which is the one with a second input -------------------

def test_the_ellipse_head_joins_two_grids_through_icat():
    """Pooled features and a host-supplied anchor state are not on a common scale."""
    model = _model(head="ellipse")
    graph = IDirectPupilDetector(model).eval()
    head = graph.head
    assert head.cat is not None
    fc1_scale = float(model.head.regressor[0].act_scale)
    assert head.cat.scale_out == fc1_scale
    # ICat is the producer that lands on fc1's grid, so the mlp must not bridge again
    assert head.mlp.to_fc1.identity, "the pooled features would be rescaled twice"

    image = graph.quantize_input(torch.rand(1, 1, SIDE, SIDE,
                                            generator=torch.Generator().manual_seed(1)))
    state = QTensor.quantize(torch.tensor([[0.4, -0.2, 0.1, 0.0, 0.3]]),
                             head.mlp.input_scale, 0.0, INT8)
    with torch.no_grad():
        conditioned = graph(image, anchor_state=state)["head"].int_data
        unconditioned = graph(image)["head"].int_data
    assert not torch.equal(conditioned, unconditioned), "the anchor state did not reach fc1"
