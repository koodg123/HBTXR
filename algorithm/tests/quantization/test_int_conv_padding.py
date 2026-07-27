"""Padded (and overlapping) integer convolution — D2a.

``IConv2d`` used to implement only the non-overlapping patch-embed conv, so every 3x3
pad-1 projection in the repo (mask head, heatmap heads, the UNet mask model) was
refused by ``_conv_is_convertible`` and left float. The arithmetic that unlocks them is
one line, and it is the line the obvious implementation gets wrong:

    a padded integer activation must be filled with the ACTIVATION ZERO-POINT, not 0.

``IConv2d`` folds the zero-point as a single ``- zp·Σw`` per output channel, which
assumes every window holds ``Cin·kh·kw`` real taps. Zero-filling the integer grid
breaks that assumption at the border only — per output pixel, not uniformly — so the
error does not cancel and is not small. ``test_zero_filled_padding_is_wrong_by_a_lot``
constructs that exact mistake and measures how far off it is.

Every reference here is built independently of the code under test: the golden is
checked against ``F.conv2d`` on the zero-point-shifted operand, ``IConv2d`` against a
hand-rolled fake-quant conv, and the export replay against the live module.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.nn import functional as F

from quantization.convert import _conv_is_convertible, convert_model_to_integer
from quantization.export import (
    MANIFEST_NAME,
    export_integer_model,
    load_integer_manifest,
    replay_conv,
    verify_export,
    verify_export_report,
)
from quantization.i_ops import int_conv2d
from quantization.ilayers.conv import IConv2d, conv_padding_pair
from quantization.scheme import INT8

# (kh, kw, stride, (ph, pw), pad_value) — square and rectangular kernels, symmetric and
# asymmetric padding, unit and overlapping strides, zero and non-zero fill.
GEOMETRIES = [
    (1, 1, 1, (0, 0), 0),
    (3, 3, 1, (0, 0), 0),          # overlapping, unpadded (the pre-existing capability)
    (3, 3, 2, (0, 0), 0),
    (4, 4, 4, (0, 0), 0),          # patch embed
    (3, 3, 1, (1, 1), 0),          # the mask/heatmap head geometry
    (3, 3, 1, (1, 1), 17),         # ... with a non-zero fill
    (3, 3, 2, (1, 1), -5),
    (5, 5, 1, (2, 2), 3),
    (3, 5, 1, (1, 2), 7),          # rectangular kernel + asymmetric padding
    (3, 3, 1, (2, 0), -9),         # padding on one axis only
]


def _golden_reference(x_int: torch.Tensor, w_int: torch.Tensor, *, stride: int,
                      padding: tuple[int, int], pad_value: int) -> torch.Tensor:
    """``conv(pad(x, v))`` built from torch alone — independent of ``i_ops.int_conv2d``.

    Padding with a constant ``v`` is exactly zero-padding the shifted operand and adding
    the constant's contribution back, since every tap of a full window sees ``+v``::

        conv(pad(x, v), w) == conv(pad(x - v, 0), w) + v·Σw

    That identity is what lets ``F.conv2d`` — which can only zero-pad — act as the
    reference for a zero-point-filled convolution.
    """
    ref = F.conv2d((x_int - pad_value).float(), w_int.float(), stride=stride, padding=padding)
    return (ref + float(pad_value) * w_int.to(torch.float64).sum(dim=(1, 2, 3)).float().view(1, -1, 1, 1)
            ).round().to(torch.int64)


# --- (a) the pure-python golden ------------------------------------------------

@pytest.mark.parametrize("kh,kw,stride,padding,pad_value", GEOMETRIES)
def test_int_conv2d_padded_is_bit_exact_against_torch(kh, kw, stride, padding, pad_value):
    torch.manual_seed(kh * 100 + kw * 10 + stride)
    cin, cout, size = 3, 4, 9
    inp = torch.randint(-40, 40, (cin, size, size), dtype=torch.int64)
    weight = torch.randint(-12, 12, (cout, cin, kh, kw), dtype=torch.int64)

    golden = torch.tensor(
        int_conv2d(inp.tolist(), weight.tolist(), stride=stride, padding=padding,
                   pad_value=pad_value),
        dtype=torch.int64,
    )
    ref = _golden_reference(inp.unsqueeze(0), weight, stride=stride, padding=padding,
                            pad_value=pad_value).squeeze(0)
    assert golden.shape == ref.shape
    assert torch.equal(golden, ref), f"max diff {(golden - ref).abs().max().item()}"


def test_int_conv2d_padding_int_and_pair_agree():
    """``padding=2`` is ``padding=(2, 2)`` — and neither is ``padding=(2, 0)``."""
    torch.manual_seed(21)
    inp = torch.randint(-9, 9, (2, 6, 6), dtype=torch.int64)
    weight = torch.randint(-6, 6, (3, 2, 3, 3), dtype=torch.int64)
    scalar = int_conv2d(inp.tolist(), weight.tolist(), padding=2, pad_value=4)
    pair = int_conv2d(inp.tolist(), weight.tolist(), padding=(2, 2), pad_value=4)
    lopsided = int_conv2d(inp.tolist(), weight.tolist(), padding=(2, 0), pad_value=4)
    assert scalar == pair
    assert scalar != lopsided


def test_int_conv2d_pad_value_actually_changes_the_result():
    """``pad_value`` is load-bearing: a golden that ignored it would pass everything above."""
    torch.manual_seed(22)
    inp = torch.randint(-9, 9, (2, 6, 6), dtype=torch.int64)
    weight = torch.randint(1, 6, (3, 2, 3, 3), dtype=torch.int64)   # Sigma w != 0 by construction
    zero_fill = int_conv2d(inp.tolist(), weight.tolist(), padding=1, pad_value=0)
    zp_fill = int_conv2d(inp.tolist(), weight.tolist(), padding=1, pad_value=17)
    assert zero_fill != zp_fill


def test_int_conv2d_unpadded_default_is_unchanged():
    """The default call is byte-for-byte the pre-existing non-padded conv."""
    torch.manual_seed(23)
    inp = torch.randint(-30, 30, (2, 8, 8), dtype=torch.int64)
    weight = torch.randint(-10, 10, (5, 2, 4, 4), dtype=torch.int64)
    default = int_conv2d(inp.tolist(), weight.tolist(), stride=4)
    explicit = int_conv2d(inp.tolist(), weight.tolist(), stride=4, padding=0, pad_value=0)
    ref = F.conv2d(inp.unsqueeze(0).float(), weight.float(), stride=4).squeeze(0).round().to(torch.int64)
    assert default == explicit
    assert torch.equal(torch.tensor(default, dtype=torch.int64), ref)


@pytest.mark.parametrize("bad", [-1, (1, -1), (1, 2, 3), (1,)])
def test_int_conv2d_rejects_malformed_padding(bad):
    inp = [[[1, 2], [3, 4]]]
    weight = [[[[1]]]]
    with pytest.raises(ValueError):
        int_conv2d(inp, weight, padding=bad)


# --- (b) IConv2d with a padded geometry and a non-zero zero-point ---------------

def _fake_quant_reference(conv: nn.Conv2d, x: torch.Tensor, s_x: float, zp: int):
    """(reference, x_int, w_int, s_w) for the fake-quant conv ``IConv2d`` must reproduce.

    Built by hand from the quantization definition — dequantized int activation
    convolved with dequantized per-channel int weight — so it shares no code with
    ``IConv2d`` beyond ``torch``. ``F.conv2d`` pads the REAL-valued operand with 0.0,
    which is the semantics the integer kernel has to match.
    """
    x_int = torch.round(x / s_x + zp).clamp(INT8.qmin, INT8.qmax)
    max_abs = conv.weight.detach().abs().amax(dim=(1, 2, 3)).clamp_min(1e-8)
    s_w = max_abs / float(INT8.qmax)
    w_int = torch.round(conv.weight / s_w.view(-1, 1, 1, 1)).clamp(INT8.qmin, INT8.qmax)
    x_fq = (x_int - zp) * s_x
    w_fq = w_int * s_w.view(-1, 1, 1, 1)
    padding = conv.padding if not isinstance(conv.padding, str) else conv_padding_pair(conv)
    ref = F.conv2d(x_fq, w_fq.detach(), conv.bias.detach(), stride=conv.stride, padding=padding)
    return ref, x_int, w_int.detach(), s_w.detach()


@pytest.mark.parametrize("zp", [0, 17, -23])
def test_iconv2d_padded_matches_fakequant_conv(zp):
    """3x3 pad-1 — the geometry every refused conv in the repo has."""
    torch.manual_seed(31)
    conv = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1)
    x = torch.rand(2, 3, 7, 7) * 2.0 - 0.6
    s_x = float(x.abs().max()) / 100.0
    ref, _x_int, _w_int, _s_w = _fake_quant_reference(conv, x, s_x, zp)

    iconv = IConv2d.from_conv(conv, act_scale=s_x, act_zero_point=zp)
    assert iconv.padding == (1, 1)
    with torch.no_grad():
        out = iconv(x)
    assert out.shape == ref.shape
    diff = float((out - ref).abs().max())
    assert diff < 1e-5, f"padded IConv2d drifted from fake-quant by {diff:.3e}"


def test_zero_filled_padding_is_wrong_by_a_lot():
    """The mistake this whole change is about, constructed and measured.

    Handing ``padding=`` straight to ``F.conv2d`` zero-fills the INTEGER grid while the
    uniform ``- zp·Σw`` correction keeps assuming a full window. The result is not
    slightly off — it is off by orders of magnitude more than the correct formulation,
    and the gap is what proves ``IConv2d`` is not accidentally doing the naive thing.
    """
    torch.manual_seed(32)
    zp = 17
    conv = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1)
    x = torch.rand(2, 3, 9, 9) * 2.0 - 0.6
    s_x = float(x.abs().max()) / 100.0
    ref, x_int, w_int, s_w = _fake_quant_reference(conv, x, s_x, zp)

    wsum = w_int.to(torch.int64).sum(dim=(1, 2, 3))
    naive_acc = F.conv2d(x_int.float(), w_int.float(), stride=1, padding=1).round().to(torch.int64)
    naive_acc = naive_acc - zp * wsum.view(1, -1, 1, 1)
    naive = naive_acc.float() * (s_x * s_w.view(1, -1, 1, 1)) + conv.bias.detach().view(1, -1, 1, 1)

    with torch.no_grad():
        good = IConv2d.from_conv(conv, act_scale=s_x, act_zero_point=zp)(x)

    naive_err = float((naive - ref).abs().max())
    good_err = float((good - ref).abs().max())
    assert good_err < 1e-5, f"correct formulation drifted by {good_err:.3e}"
    assert naive_err > 1e-2, (
        "zero-filled padding reproduced the reference, so this test proves nothing "
        f"(err {naive_err:.3e}); the fixture must have Sigma w ~= 0 or zp == 0")
    assert naive_err > 1000 * good_err

    # Interior pixels are correct in BOTH: the damage is exactly the padded ring, which
    # is why the error cannot be waved away as a uniform bias.
    interior = slice(1, -1)
    assert float((naive - ref)[:, :, interior, interior].abs().max()) < 1e-5


@pytest.mark.parametrize("kh,kw,stride,padding,pad_value", GEOMETRIES)
def test_iconv2d_accumulator_matches_the_pure_int_golden(kh, kw, stride, padding, pad_value):
    """``IConv2d``'s float32 accumulation == ``i_ops.int_conv2d`` in arbitrary-precision int.

    ``pad_value`` doubles as the activation zero-point here: that IS the contract — the
    integer that represents a real zero on the activation grid.
    """
    torch.manual_seed(41)
    conv = nn.Conv2d(2, 4, kernel_size=(kh, kw), stride=stride, padding=padding)
    x = torch.rand(1, 2, 11, 11) * 3.0 - 1.0
    s_x = float(x.abs().max()) / 100.0
    iconv = IConv2d.from_conv(conv, act_scale=s_x, act_zero_point=pad_value)

    x_int = torch.round(x / s_x + iconv.act_zero_point).clamp(INT8.qmin, INT8.qmax).to(torch.int64)
    golden = torch.tensor(
        int_conv2d(x_int[0].tolist(), iconv.weight_int.to(torch.int64).tolist(),
                   stride=stride, padding=padding, pad_value=iconv.act_zero_point),
        dtype=torch.int64,
    ) - iconv.act_zero_point * iconv.wsum.view(-1, 1, 1)

    with torch.no_grad():
        y = iconv(x)
    recon = (y - (conv.bias.detach().view(1, -1, 1, 1) if conv.bias is not None else 0.0))
    acc = torch.round(recon / (s_x * iconv.weight_scale.view(1, -1, 1, 1))).to(torch.int64)
    assert torch.equal(acc.squeeze(0), golden)


# --- (c) the unpadded path is untouched ----------------------------------------

@pytest.mark.parametrize("zp", [0, 11])
@pytest.mark.parametrize("kernel,stride", [(4, 4), (3, 1), (3, 2), (1, 1)])
def test_unpadded_path_is_bit_identical_to_the_previous_implementation(kernel, stride, zp):
    """The pre-D2a formula, written out, must still be what an unpadded IConv2d computes.

    Not ``allclose`` — ``torch.equal``. The unpadded conv is a shipped, exported and
    replayed op; a change of even one ULP there would be a silent behaviour change for
    every already-converted patch embedding.
    """
    torch.manual_seed(51)
    conv = nn.Conv2d(3, 6, kernel_size=kernel, stride=stride)
    x = torch.rand(2, 3, 12, 12) * 2.0 - 0.5
    s_x = float(x.abs().max()) / 100.0
    iconv = IConv2d.from_conv(conv, act_scale=s_x, act_zero_point=zp)
    assert iconv.padding == (0, 0)

    # verbatim pre-D2a IConv2d.forward
    x_int = torch.round(x / s_x + zp).clamp(INT8.qmin, INT8.qmax)
    legacy = F.conv2d(x_int.float(), iconv.weight_int.float(), stride=stride).round().to(torch.int64)
    if zp != 0:
        legacy = legacy - zp * iconv.wsum.view(1, -1, 1, 1)
    legacy_y = legacy.to(torch.float32) * (s_x * iconv.weight_scale.view(1, -1, 1, 1))
    legacy_y = legacy_y + conv.bias.detach().view(1, -1, 1, 1)

    with torch.no_grad():
        out = iconv(x)
    assert torch.equal(out, legacy_y)


# --- from_conv normalisation + the convertibility gate -------------------------

@pytest.mark.parametrize("spelling,expected", [
    (0, (0, 0)),
    (1, (1, 1)),
    ((1, 2), (1, 2)),
    ("valid", (0, 0)),
    ("same", (1, 1)),                       # odd kernel -> symmetric
])
def test_from_conv_normalises_every_padding_spelling(spelling, expected):
    kwargs = {"stride": 1} if spelling == "same" else {}
    conv = nn.Conv2d(2, 3, kernel_size=3, padding=spelling, **kwargs)
    assert conv_padding_pair(conv) == expected
    assert IConv2d.from_conv(conv, act_scale=0.02).padding == expected


def test_same_padding_on_an_even_kernel_is_refused_not_rounded():
    """torch splits an odd total as ``left = total // 2``; IConv2d cannot express that."""
    conv = nn.Conv2d(2, 3, kernel_size=4, padding="same")
    with pytest.raises(ValueError, match="asymmetric"):
        conv_padding_pair(conv)
    with pytest.raises(ValueError, match="asymmetric"):
        IConv2d.from_conv(conv, act_scale=0.02)
    assert _conv_is_convertible(conv) is False


def test_from_conv_refuses_mismatched_strides():
    conv = nn.Conv2d(2, 3, kernel_size=3, stride=(1, 2))
    with pytest.raises(ValueError, match="one stride"):
        IConv2d.from_conv(conv, act_scale=0.02)
    assert _conv_is_convertible(conv) is False


@pytest.mark.parametrize("conv_kwargs", [
    {"kernel_size": 3, "padding": 1},
    {"kernel_size": 3, "padding": (1, 2)},
    {"kernel_size": 3, "padding": "same"},
    {"kernel_size": 3, "padding": "valid"},
    {"kernel_size": 3, "stride": 2, "padding": 1},
    {"kernel_size": 16, "stride": 16},
    {"kernel_size": 1},
])
def test_conv_is_convertible_accepts_padded_and_overlapping(conv_kwargs):
    assert _conv_is_convertible(nn.Conv2d(4, 4, **conv_kwargs)) is True


@pytest.mark.parametrize("conv_kwargs", [
    {"kernel_size": 3, "padding": 2, "dilation": 2},
    {"kernel_size": 3, "padding": 1, "groups": 2},
    {"kernel_size": 4, "padding": "same"},
    {"kernel_size": 3, "stride": (1, 2)},
])
def test_conv_is_convertible_still_refuses_what_iconv2d_cannot_express(conv_kwargs):
    assert _conv_is_convertible(nn.Conv2d(4, 4, **conv_kwargs)) is False


# --- (d) end to end: FrameModel -> convert -> export -> verify ------------------

def _frame_model(embed: int = 48):
    from engine.model_factory import make_model

    return make_model({"target": "models.frame.FrameModel", "embed_dim": embed, "patch_size": 16,
                       "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}})


@pytest.fixture(scope="module")
def converted_frame():
    """(model, report) for a fully converted FrameModel — the real D2a acceptance case.

    PTQ first (so the Linears are ``QLinear`` and become ``ILinear``), then the whole
    Q -> I graph conversion. Anything less would leave float Linears behind and
    ``export_integer_model`` would refuse the dump for a reason unrelated to convs.
    """
    from quantization.calibrate import post_training_quantize

    torch.manual_seed(0)
    model = _frame_model()
    batches = [torch.rand(2, 1, 64, 64) for _ in range(2)]
    model, _ = post_training_quantize(model, batches)
    return convert_model_to_integer(model, batches)


def test_frame_mask_conv_is_no_longer_left_float(converted_frame):
    model, report = converted_frame
    assert isinstance(model.mask_head.proj, IConv2d)
    assert model.mask_head.proj.padding == (1, 1)
    assert "mask_head.proj" not in report.left_float
    assert "mask_head.proj" in report.replaced
    # the whole point: the 3x3 pad-1 head projection joins the two convs that already
    # converted, rather than being the documented hole in the dump.
    conv_names = {n for n, m in model.named_modules() if isinstance(m, IConv2d)}
    assert conv_names == {"patch_embed.proj", "mask_head.proj", "mask_head.to_logits"}


def test_frame_export_has_no_unexported_modules_and_verifies(converted_frame, tmp_path):
    model, _report = converted_frame
    manifest = export_integer_model(model, tmp_path)          # no allow_unquantized escape hatch
    assert manifest["unexported"] == []
    assert manifest["num_unexported"] == 0

    on_disk = json.loads((tmp_path / MANIFEST_NAME).read_text())
    entry = on_disk["modules"]["mask_head.proj"]
    assert entry["op"] == "conv2d_int"
    assert entry["padding"] == [1, 1]
    assert entry["stride"] == 1
    assert entry["weight_shape"][2:] == [3, 3]

    ok, max_diff = verify_export(model, tmp_path)
    assert ok, f"export did not verify (worst diff {max_diff:.3e})"

    report = {r["name"]: r for r in verify_export_report(model, tmp_path)}
    for name in ("patch_embed.proj", "mask_head.proj", "mask_head.to_logits"):
        record = report[name]
        assert record["status"] == "checked" and record["ok"]
        assert record["max_abs_diff"] == 0.0, (
            f"{name} replay is not bit-exact: {record['max_abs_diff']:.3e}")


def test_replay_conv_needs_the_padding_field(converted_frame, tmp_path):
    """Strip ``padding`` from the entry and the replay stops matching the live module.

    Without this, ``test_frame_export_...`` would pass just as happily if ``_conv_entry``
    forgot to emit the field and ``replay_conv`` defaulted to (0, 0) — the probe would
    simply be replayed unpadded on both sides. This pins the field as load-bearing.
    """
    model, _report = converted_frame
    export_integer_model(model, tmp_path)
    manifest = load_integer_manifest(tmp_path)
    entry = manifest["modules"]["mask_head.proj"]
    assert entry["padding"] == [1, 1]

    rng = np.random.default_rng(7)
    span = float(entry["act_scale"]) * 127
    probe = (rng.uniform(-1.0, 1.0, size=(2, entry["weight_shape"][1], 5, 5)) * span).astype(np.float32)
    with torch.no_grad():
        live = model.mask_head.proj(torch.from_numpy(probe)).numpy()

    padded = replay_conv(entry, probe)
    assert padded.shape == live.shape
    assert float(np.max(np.abs(padded - live))) == 0.0

    stripped = dict(entry)
    stripped.pop("padding")
    unpadded = replay_conv(stripped, probe)
    assert unpadded.shape != live.shape, "dropping padding did not even change the output shape"


class _Holder(nn.Module):
    """Minimal parent so a bare ``IConv2d`` can be exported by path."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.op = module


@pytest.mark.parametrize("zp", [0, 17, -23])
@pytest.mark.parametrize("padding", [(1, 1), (1, 2), (2, 0)])
def test_padded_conv_replays_bit_exactly_with_a_nonzero_zero_point(tmp_path, zp, padding):
    """The export replay must fill padding with the zero-point too — and nothing else tests it.

    ``calibrate_int_convs`` builds every ``IConv2d`` with a symmetric activation grid, so
    a converted FrameModel has ``act_zero_point == 0`` on all three of its convs. That
    makes the end-to-end export test blind to ``replay_conv``'s fill value: 0 and ``zp``
    coincide. A ``replay_conv`` that zero-filled would sail through it. This constructs
    the asymmetric case directly, which is the only place the two differ.
    """
    torch.manual_seed(61)
    conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=padding)
    holder = _Holder(IConv2d.from_conv(conv, act_scale=0.02, act_zero_point=zp))
    assert holder.op.padding == padding and holder.op.act_zero_point == zp

    manifest = export_integer_model(_Holder(holder.op), tmp_path)
    entry = manifest["modules"]["op"]
    assert entry["padding"] == list(padding)
    assert entry["act_zero_point"] == zp

    ok, max_diff = verify_export(holder, tmp_path)
    assert ok and max_diff == 0.0, f"padded conv replay drifted by {max_diff:.3e}"


def test_probe_exercises_the_padded_border(converted_frame, tmp_path):
    """The conv probe must be small enough that padded taps reach every output pixel.

    A 3x3 pad-1 conv on a 64x64 probe is 96% interior: a replay that mishandled the
    border would still agree to within float noise on almost every pixel and could pass
    a max-abs gate by luck. The exporter sizes the probe minimally on purpose, and this
    asserts it kept doing so.
    """
    from quantization.export import _CONV_PROBE_OUT, _probe

    model, _report = converted_frame
    export_integer_model(model, tmp_path)
    entry = load_integer_manifest(tmp_path)["modules"]["mask_head.proj"]
    probe = _probe(entry, np.random.default_rng(1), 2)
    _n, _c, height, width = probe.shape
    ph, pw = entry["padding"]
    kh, kw = entry["weight_shape"][2:]
    stride = entry["stride"]

    assert (height + 2 * ph - kh) // stride + 1 == _CONV_PROBE_OUT
    assert (width + 2 * pw - kw) // stride + 1 == _CONV_PROBE_OUT
    # every input row/column is within one kernel radius of the padded ring
    assert height <= kh and width <= kw


# --- (e) negative control: the replay has teeth --------------------------------

def _corrupt_preserving_range(array: np.ndarray) -> np.ndarray:
    """Change one element without moving ``min``/``max``.

    ``load_integer_manifest`` validates the recorded ``[int_min, int_max]``, so a blunt
    corruption is rejected before any replay runs — which would make this a test of the
    loader, not of the replay. Keeping the extremes intact forces the replay to fire.
    """
    flat = array.reshape(-1).copy()
    lo, hi = int(flat.min()), int(flat.max())
    assert lo < hi, "fixture weight is constant; cannot corrupt inside its own range"
    protected = {int(np.argmin(flat)), int(np.argmax(flat))}
    for idx in range(flat.size):
        if idx in protected:
            continue
        replacement = hi if int(flat[idx]) != hi else lo
        if replacement != int(flat[idx]):
            flat[idx] = replacement
            break
    else:                                                    # pragma: no cover - 2-element weight
        pytest.fail("no corruptible element in the weight")
    out = flat.reshape(array.shape)
    assert int(out.min()) == lo and int(out.max()) == hi
    assert not np.array_equal(out, array)
    return out


def test_corrupted_padded_conv_weight_fails_verification(converted_frame, tmp_path):
    model, _report = converted_frame
    manifest = export_integer_model(model, tmp_path)
    ok, _ = verify_export(model, tmp_path)
    assert ok, "fixture export must verify before corruption, or the negative is meaningless"

    weight_file = tmp_path / manifest["modules"]["mask_head.proj"]["weight_file"]
    np.save(weight_file, _corrupt_preserving_range(np.load(weight_file)))

    ok, max_diff = verify_export(model, tmp_path)
    assert ok is False, "a corrupted padded conv weight replayed clean — the branch has no teeth"
    assert max_diff > 0.0

    record = {r["name"]: r for r in verify_export_report(model, tmp_path)}["mask_head.proj"]
    assert record["status"] == "checked" and record["ok"] is False


# --- padding_mode: converting a non-zeros conv would compute a different function ---
#
# Accepting padded convs is what made this reachable. IConv2d constant-pads with the
# activation zero-point (the integer spelling of a real zero); reflect / replicate /
# circular fill the border from the image instead. Converting one would be silent.

@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_non_zeros_padding_mode_is_refused(mode):
    from quantization.convert import _conv_is_convertible

    conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, padding_mode=mode)
    assert not _conv_is_convertible(conv), f"padding_mode={mode!r} must not convert"
    # and the difference is real, not theoretical: torch's own output differs from the
    # zero-padded one this kernel would compute
    zeros = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, padding_mode="zeros")
    zeros.load_state_dict(conv.state_dict())
    x = torch.randn(1, 3, 6, 6)
    with torch.no_grad():
        assert not torch.allclose(conv(x), zeros(x)), (
            f"fixture is degenerate: {mode} and zeros padding agree here")


def test_plain_zeros_padded_conv_still_converts():
    """The guard must refuse only the modes it means to (no blanket rejection)."""
    from quantization.convert import _conv_is_convertible

    assert _conv_is_convertible(nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1))
    assert _conv_is_convertible(nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1,
                                          padding_mode="zeros"))


def test_rectangular_same_padding_is_not_transposed():
    """(ph, pw) must follow (kh, kw); transposing them survived the whole suite."""
    from quantization.ilayers.conv import conv_padding_pair

    conv = nn.Conv2d(2, 3, kernel_size=(7, 3), padding="same")
    assert conv_padding_pair(conv) == (3, 1)          # (7-1)//2, (3-1)//2
    conv_t = nn.Conv2d(2, 3, kernel_size=(3, 7), padding="same")
    assert conv_padding_pair(conv_t) == (1, 3)
