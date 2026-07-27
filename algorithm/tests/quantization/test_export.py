"""Integer-artifact export (C6): manifest fidelity + replay-from-artifacts equivalence.

The export is only worth something if a backend that has *nothing but the dumped
directory* can reproduce the model. So these tests assert the manifest is complete
and self-consistent, that the loader returns exactly the live buffers, and that
``verify_export`` — which rebuilds each op from the artifacts alone — both passes on
a good dump and fails on a corrupted one.

Two fixtures, because the export has to survive both deployment tiers:

``q_tier``   PTQ + LUT calibration + ``convert_to_integer``: ``ILinear`` plus the
             Q-tier ``QGeLU`` / ``QLayerNorm`` / ``QSoftmax`` (float reductions,
             integer LUTs). This is what the shipped pipeline produces today.
``int_tier`` the same graph with every nonlinear swapped for its fully-integer
             I-tier module (``IGeLU`` / ``ILayerNorm`` / ``ISoftmax``). Those are
             built here rather than by ``convert_to_integer`` so the export's
             I-tier branches are covered no matter when the conversion path adopts
             them — which is exactly the hole where they used to vanish silently.

A recurring theme below: an assertion that a written value equals the constant that
wrote it proves nothing. Expected literals are hardcoded, and every field is checked
against the *live module*, never against the exporter's own output.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn

from quantization.calibrate import post_training_quantize
from quantization.convert import (
    calibrate_gelu_luts,
    calibrate_layernorm_luts,
    calibrate_softmax_luts,
    convert_to_integer,
)
from quantization.export import (
    MANIFEST_NAME,
    export_integer_model,
    load_integer_manifest,
    verify_export,
    verify_export_report,
)
from quantization.ilayers.conv import IConv2d
from quantization.ilayers.layernorm import ILayerNorm
from quantization.ilayers.linear import ILinear
from quantization.ilayers.nonlinear import IGeLU
from quantization.ilayers.softmax import ISoftmax
from quantization.int_calibrate_softmax import build_softmax_int_payload_from_qsoftmax
from quantization.observer import build_observer
from quantization.q_ops import AffineFakeQuantizer
from quantization.qlayers.linear import QLinear, QuantConfig
from quantization.qlayers.nonlinear import QGeLU, QLayerNorm, QSoftmax
from quantization.spec import QuantScheme, TensorQuantSpec

# The on-disk contract, spelled out independently of quantization.export. Importing
# FORMAT / FORMAT_VERSION and comparing them to themselves is a tautology: a reviewer
# mutated both and the suite stayed green. These literals are the actual promise made
# to every backend that reads a dump, so changing them must break this test.
EXPECTED_FORMAT = "hbtxr-int-v1"
# v4 (D2): conv entries gained `padding`, softmax entries gained `max_tokens`.
# v5 (D3): layernorm_int entries gained `segments` and the optional `scalars_two` /
#          `rsqrt_table_two` of the two-segment rsqrt index. Additive, but a v4 reader
#          would run the one-segment kernel on a segmented entry, so it is breaking.
EXPECTED_FORMAT_VERSION = 5

# Every module class the export knows how to write, and the replay kernel each maps to.
EXPECTED_OPS = {
    "ILinear": "linear_int",
    "IConv2d": "conv2d_int",
    "QGeLU": "gelu_lut",
    "IGeLU": "gelu_lut",
    "QLayerNorm": "layernorm_lut",
    "ILayerNorm": "layernorm_int",
    "QSoftmax": "softmax_lut",
    "ISoftmax": "softmax_int",
}


class _Holder(nn.Module):
    """Minimal parent so a bare I-tier op gets the module path ``op``."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.op = inner


class _Colliding(nn.Module):
    """Two ILinears whose module paths (``blk.0`` and ``blk_0``) sanitize identically."""

    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        super().__init__()
        self.blk = nn.ModuleList([first])
        self.blk_0 = second


def _set_submodule(model: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parent = model
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _frame_model(embed: int = 48):
    from engine.model_factory import make_model

    return make_model({"target": "models.frame.FrameModel", "embed_dim": embed, "patch_size": 16,
                       "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1}})


def _converted_model(weight_gran: str = "per-channel", act_symmetric: bool = True):
    """PTQ + nonlinear LUT calibration + Q->I conversion — the real export input."""
    torch.manual_seed(0)
    model = _frame_model()
    img = torch.rand(2, 1, 64, 64)
    scheme = QuantScheme(
        weight=TensorQuantSpec(ch_axis=0, granularity=weight_gran),
        activation=TensorQuantSpec(ch_axis=-1, symmetric=act_symmetric),
    )
    model, _ = post_training_quantize(model, [img], scheme=scheme)
    calibrate_gelu_luts(model, [img])
    calibrate_layernorm_luts(model, [img])
    calibrate_softmax_luts(model, [img])
    model, _ = convert_to_integer(model)
    return model


def _int_tier_model():
    """``_converted_model`` with every nonlinear promoted to its fully-integer I-tier form.

    The LN/Softmax payloads need observed activations, so the Q-tier model is run once
    with pre-hooks to collect them. ``IGeLU`` is QTensor-in/QTensor-out and therefore
    not a float drop-in — the resulting model is not meant to be run end to end, only
    exported and replayed op by op, which is precisely what ``verify_export`` does.
    """
    model = _converted_model()
    img = torch.rand(2, 1, 64, 64)
    targets = {n: m for n, m in model.named_modules() if isinstance(m, (QLayerNorm, QSoftmax))}
    seen: dict[str, list] = {n: [] for n in targets}

    def make_hook(name: str):
        def hook(_module, inputs):
            if inputs:
                seen[name].append(inputs[0].detach().float())
        return hook

    handles = [targets[n].register_forward_pre_hook(make_hook(n)) for n in targets]
    with torch.no_grad():
        model(img)
    for handle in handles:
        handle.remove()

    for name, module in targets.items():
        assert seen[name], f"no observations captured for {name}"
        if isinstance(module, QLayerNorm):
            _set_submodule(model, name, ILayerNorm.from_qlayernorm(module, seen[name]))
        else:
            logits = torch.cat([s.reshape(-1, s.shape[-1]) for s in seen[name]]).numpy()
            payload = build_softmax_int_payload_from_qsoftmax(module, logits)
            _set_submodule(model, name, ISoftmax.from_payload(payload, dim=int(module.dim)))
    for name, module in [(n, m) for n, m in model.named_modules() if isinstance(m, QGeLU)]:
        _set_submodule(model, name, IGeLU.from_qgelu(module))
    return model


def _bare_ilinear(in_f: int = 12, out_f: int = 8, symmetric: bool = True, seed: int = 7) -> ILinear:
    torch.manual_seed(seed)
    lin = nn.Linear(in_f, out_f)
    ql = QLinear(lin, QuantConfig(weight_spec=TensorQuantSpec(granularity="per-channel", ch_axis=0),
                                  act_spec=TensorQuantSpec(ch_axis=-1, symmetric=symmetric)))
    ow = build_observer(ql.weight_fq.spec)
    ow.observe(lin.weight)
    ql.weight_fq.set_qparams(*ow.qparams())
    oa = build_observer(ql.act_fq.spec)
    oa.observe(torch.randn(20, in_f) * 2.0 + 0.5)
    ql.act_fq.set_qparams(*oa.qparams())
    return ILinear.from_qlinear(ql)


def _bare_iconv(act_zero_point: float = 0.0) -> IConv2d:
    """A non-overlapping patch-embed-shaped conv, the only conv shape the I tier targets."""
    torch.manual_seed(3)
    return IConv2d.from_conv(nn.Conv2d(2, 3, 4, stride=4), act_scale=0.02,
                             act_zero_point=act_zero_point)


def _corrupt_one_value_in_range(array: np.ndarray) -> np.ndarray:
    """Change exactly one element while preserving the array's ``[min, max]``.

    The loader validates the recorded ``weight_int_min`` / ``weight_int_max``, so a
    blunt corruption is caught before any replay runs. Keeping the extremes intact is
    what forces the *replay* — not the cheap range check — to be the thing that fires.
    """
    flat = array.reshape(-1)
    lo, hi = int(flat.min()), int(flat.max())
    assert lo < hi, "fixture weight is constant; cannot corrupt inside its own range"
    protected = {int(np.argmin(flat)), int(np.argmax(flat))}
    victim = next(i for i in range(flat.size) if i not in protected)
    flat[victim] = lo if int(flat[victim]) == hi else hi
    assert int(flat.min()) == lo and int(flat.max()) == hi
    return array


@pytest.fixture(scope="module")
def q_tier(tmp_path_factory):
    """(model, out_dir, manifest) for a Q-tier converted FrameModel — read-only for tests."""
    model = _converted_model()
    out = tmp_path_factory.mktemp("int_export_q")
    manifest = export_integer_model(model, out, allow_unquantized=True)
    return model, out, manifest


@pytest.fixture(scope="module")
def int_tier(tmp_path_factory):
    """(model, out_dir, manifest) for the fully-integer tier — read-only for tests."""
    model = _int_tier_model()
    out = tmp_path_factory.mktemp("int_export_i")
    manifest = export_integer_model(model, out, allow_unquantized=True)
    return model, out, manifest


@pytest.fixture(params=["q_tier", "int_tier"])
def either_tier(request):
    """Both deployment tiers, so no assertion below is accidentally tier-specific."""
    return request.getfixturevalue(request.param)


# --- manifest structure + metadata -------------------------------------------

def test_manifest_exists_parses_and_carries_metadata(q_tier):
    model, out, manifest = q_tier
    on_disk = json.loads((out / MANIFEST_NAME).read_text())
    assert on_disk == json.loads(json.dumps(manifest))          # returned == written
    assert on_disk["format"] == EXPECTED_FORMAT
    assert on_disk["format_version"] == EXPECTED_FORMAT_VERSION
    assert on_disk["model_class"] == type(model).__name__
    assert on_disk["num_modules"] == len(on_disk["modules"]) > 0
    assert on_disk["total_params"] == sum(p.numel() for p in model.parameters())
    assert on_disk["total_int_weights"] == sum(
        int(np.prod(e["weight_shape"])) for e in on_disk["modules"].values() if "weight_shape" in e)


def test_manifest_type_field_names_the_live_module_class(either_tier):
    """``type`` is an integrity field: it must be what the module IS, not a guess.

    Every LayerNorm/Softmax entry used to be stamped ``ILayerNorm``/``ISoftmax`` while
    the only dispatch that reached them matched the Q-tier classes, so the whole dump
    mislabelled its own ops — and once real I-tier entries existed the two tiers were
    indistinguishable in the manifest.
    """
    model, _out, manifest = either_tier
    assert manifest["modules"]
    for name, entry in manifest["modules"].items():
        live = model.get_submodule(name)
        assert entry["type"] == type(live).__name__, f"{name}: manifest lies about its op type"
        assert entry["op"] == EXPECTED_OPS[entry["type"]]


def test_the_two_tiers_really_do_export_different_op_types(q_tier, int_tier):
    """Guards the fixtures themselves: without this the tier parametrization is a no-op."""
    _qm, _qo, q_manifest = q_tier
    _im, _io, i_manifest = int_tier
    assert set(q_manifest["counts"]) == {"ILinear", "QGeLU", "QLayerNorm", "QSoftmax"}
    assert set(i_manifest["counts"]) == {"ILinear", "IGeLU", "ILayerNorm", "ISoftmax"}
    assert q_manifest["counts"]["QLayerNorm"] == i_manifest["counts"]["ILayerNorm"] == 5
    assert q_manifest["counts"]["QSoftmax"] == i_manifest["counts"]["ISoftmax"] == 2


def test_manifest_counts_match_the_live_module_types(either_tier):
    model, _out, manifest = either_tier
    live: dict[str, int] = {}
    for _name, module in model.named_modules():
        cls = type(module).__name__
        if cls in EXPECTED_OPS:
            live[cls] = live.get(cls, 0) + 1
    assert live.get("ILinear", 0) == 14, "conversion produced no integer linears — fixture is broken"
    assert manifest["counts"] == live
    assert manifest["num_modules"] == sum(live.values()) == 27


def test_module_names_round_trip_to_module_paths(either_tier):
    model, _out, manifest = either_tier
    orders = [entry["order"] for entry in manifest["modules"].values()]
    assert orders == sorted(orders) and len(set(orders)) == len(orders)
    for name, entry in manifest["modules"].items():
        assert entry["name"] == name
        model.get_submodule(name)                               # raises if not resolvable


# --- entry contents: VALUES, not just key presence ----------------------------

def test_linear_entries_record_everything_needed_to_rebuild(q_tier):
    model, _out, manifest = q_tier
    linears = [e for e in manifest["modules"].values() if e["type"] == "ILinear"]
    assert linears
    for entry in linears:
        for key in ("weight_file", "weight_scale", "act_scale", "act_zero_point",
                    "act_bits", "act_signed", "weight_bits", "weight_signed", "bias"):
            assert key in entry, f"{entry['name']} missing {key}"
        assert entry["weight_shape"] == [entry["out_features"], entry["in_features"]]
        assert len(entry["weight_scale"]) in (1, entry["out_features"])

        live = model.get_submodule(entry["name"])
        assert entry["weight_signed"] is True                   # ILinear enforces symmetric
        assert entry["weight_zero_point"] == 0
        assert entry["act_zero_point"] == live.act_zero_point
        assert entry["act_bits"] == live.act_dtype.bits
        assert entry["act_signed"] == live.act_dtype.signed
        assert entry["act_scale"] == pytest.approx(live.act_scale)
        assert np.allclose(entry["bias"], live.bias.cpu().numpy())


def test_q_layernorm_entry_values_match_the_live_module(q_tier):
    model, _out, manifest = q_tier
    entries = [e for e in manifest["modules"].values() if e["type"] == "QLayerNorm"]
    assert entries
    for entry in entries:
        live = model.get_submodule(entry["name"])
        assert entry["normalized_shape"] == [int(d) for d in live.normalized_shape]
        assert entry["eps"] == pytest.approx(live.eps)
        assert entry["scalars"] == [live.b, live.s, live.bound]
        assert entry["table_entries"] == live.rsqrt_table.numel()
        assert np.array_equal(entry["rsqrt_table"], live.rsqrt_table.cpu().numpy())
        assert np.allclose(entry["weight"], live.weight.detach().cpu().numpy())
        assert np.allclose(entry["bias"], live.bias.detach().cpu().numpy())


def test_q_softmax_entry_values_match_the_live_module(q_tier):
    model, _out, manifest = q_tier
    entries = [e for e in manifest["modules"].values() if e["type"] == "QSoftmax"]
    assert entries
    for entry in entries:
        live = model.get_submodule(entry["name"])
        assert entry["dim"] == live.dim
        assert entry["exp"]["scalars"] == [live.exp_b, live.exp_s, live.exp_bound]
        assert entry["recip"]["scalars"] == [live.recip_b, live.recip_s, live.recip_bound]
        assert np.array_equal(entry["exp"]["table"], live.exp_table.cpu().numpy())
        assert np.array_equal(entry["recip"]["table"], live.recip_table.cpu().numpy())


def test_int_layernorm_entry_values_match_the_live_module(int_tier):
    model, _out, manifest = int_tier
    entries = [e for e in manifest["modules"].values() if e["type"] == "ILayerNorm"]
    assert entries
    for entry in entries:
        live = model.get_submodule(entry["name"])
        assert entry["scalars"] == [live.c_1_m, live.c_1_s, live.b, live.s1,
                                    live.bound, live.s2, live.clamp_bits]
        assert entry["channels"] == live.channels
        assert entry["input_bits"] == live.in_dtype.bits
        assert entry["input_signed"] == live.in_dtype.signed
        assert entry["output_bits"] == live.out_dtype.bits
        assert np.array_equal(entry["lnw"], live.lnw.cpu().numpy())
        assert np.array_equal(entry["lnb"], live.lnb.cpu().numpy())
        assert np.array_equal(entry["rsqrt_table"], live.rsqrt_table.cpu().numpy())


def test_int_softmax_entry_values_match_the_live_module(int_tier):
    model, _out, manifest = int_tier
    entries = [e for e in manifest["modules"].values() if e["type"] == "ISoftmax"]
    assert entries
    for entry in entries:
        live = model.get_submodule(entry["name"])
        assert entry["dim"] == live.dim
        assert entry["scalars"] == [
            live.b1, live.s1, live.bound1,
            live.b2_one, live.s2_one, live.bound2_one, live.b3_one, live.s3_one,
            live.b2_two, live.s2_two, live.bound2_two, live.b3_two, live.s3_two,
            live.clamp_bits,
        ]
        assert entry["input_bits"] == live.in_dtype.bits
        assert np.array_equal(entry["exp_table"], live.exp_table.cpu().numpy())
        assert np.array_equal(entry["recip_table_one"], live.recip_table_one.cpu().numpy())
        assert np.array_equal(entry["recip_table_two"], live.recip_table_two.cpu().numpy())


def test_lut_entries_record_scalars_tables_and_scales(either_tier):
    _model, _out, manifest = either_tier
    for entry in manifest["modules"].values():
        if entry["op"] == "gelu_lut":
            assert len(entry["scalars"]) == 3 and len(entry["table"]) == entry["table_entries"] > 0
            assert entry["input_scale"] > 0 and entry["output_scale"] > 0
        elif entry["op"] == "layernorm_lut":
            assert len(entry["scalars"]) == 3 and len(entry["rsqrt_table"]) == entry["table_entries"] > 0
            assert entry["normalized_shape"] and entry["eps"] > 0
        elif entry["op"] == "layernorm_int":
            assert len(entry["scalars"]) == 7
            assert len(entry["lnw"]) == len(entry["lnb"]) == entry["channels"] > 0
        elif entry["op"] == "softmax_lut":
            for part in (entry["exp"], entry["recip"]):
                assert len(part["scalars"]) == 3 and len(part["table"]) == part["table_entries"] > 0
        elif entry["op"] == "softmax_int":
            assert len(entry["scalars"]) == 14
            assert len(entry["recip_table_one"]) == len(entry["recip_table_two"]) > 0


# --- artifacts on disk --------------------------------------------------------

def test_every_referenced_npy_exists_and_matches_the_manifest(either_tier):
    _model, out, manifest = either_tier
    referenced = 0
    for name, entry in manifest["modules"].items():
        file = entry.get("weight_file")
        if file is None:
            continue
        referenced += 1
        path = out / file
        assert path.is_file(), f"{name} references missing {file}"
        array = np.load(path)
        assert str(array.dtype) == entry["weight_dtype"]
        assert list(array.shape) == entry["weight_shape"]
        assert int(array.min()) == entry["weight_int_min"]
        assert int(array.max()) == entry["weight_int_max"]
    assert referenced == manifest["counts"]["ILinear"]
    assert len({e["weight_file"] for e in manifest["modules"].values() if e.get("weight_file")}) == referenced


def test_loader_returns_exactly_the_live_buffers(either_tier):
    model, out, _manifest = either_tier
    loaded = load_integer_manifest(out)
    checked = 0
    for name, entry in loaded["modules"].items():
        live = model.get_submodule(name)
        if entry["op"] == "linear_int":
            assert np.array_equal(entry["weight_int"].astype(np.int64),
                                  live.weight_int.cpu().numpy().astype(np.int64))
            assert np.array_equal(entry["weight_scale"], live.weight_scale.cpu().numpy())
            assert np.array_equal(entry["bias"], live.bias.cpu().numpy())
        elif entry["op"] == "gelu_lut":
            assert np.array_equal(entry["table"], live.table.cpu().numpy())
        elif entry["op"] == "layernorm_lut":
            assert np.array_equal(entry["rsqrt_table"], live.rsqrt_table.cpu().numpy())
            assert np.array_equal(entry["weight"], live.weight.detach().cpu().numpy())
        elif entry["op"] == "layernorm_int":
            assert np.array_equal(entry["rsqrt_table"], live.rsqrt_table.cpu().numpy())
            assert np.array_equal(entry["lnw"], live.lnw.cpu().numpy())
            assert np.array_equal(entry["lnb"], live.lnb.cpu().numpy())
        elif entry["op"] == "softmax_lut":
            assert np.array_equal(entry["exp"]["table"], live.exp_table.cpu().numpy())
            assert np.array_equal(entry["recip"]["table"], live.recip_table.cpu().numpy())
        elif entry["op"] == "softmax_int":
            assert np.array_equal(entry["exp_table"], live.exp_table.cpu().numpy())
            assert np.array_equal(entry["recip_table_one"], live.recip_table_one.cpu().numpy())
            assert np.array_equal(entry["recip_table_two"], live.recip_table_two.cpu().numpy())
        checked += 1
    assert checked == loaded["num_modules"] > 0


def test_loader_rejects_a_missing_artifact(q_tier, tmp_path):
    _model, src, manifest = q_tier
    dst = tmp_path / "broken"
    shutil.copytree(src, dst)
    victim = next(e["weight_file"] for e in manifest["modules"].values() if e.get("weight_file"))
    (dst / victim).unlink()
    with pytest.raises(FileNotFoundError):
        load_integer_manifest(dst)


@pytest.mark.parametrize("mangle", ["dtype", "shape"])
def test_loader_rejects_an_artifact_whose_dtype_or_shape_disagrees(q_tier, tmp_path, mangle):
    """The loader is a backend's only schema check; it has to be able to fail."""
    _model, src, manifest = q_tier
    dst = tmp_path / f"mangled_{mangle}"
    shutil.copytree(src, dst)
    entry = next(e for e in manifest["modules"].values() if e.get("weight_file"))
    path = dst / entry["weight_file"]
    array = np.load(path)
    np.save(path, array.astype(np.int32) if mangle == "dtype" else array[:-1])

    with pytest.raises(ValueError, match="manifest says"):
        load_integer_manifest(dst)


def test_loader_rejects_a_value_corrupted_artifact_that_keeps_dtype_and_shape(q_tier, tmp_path):
    """The recorded int range is an integrity bit — either enforce it or stop writing it.

    A zeroed weight file has the declared dtype and shape, so dtype/shape validation
    alone waves it straight through while the manifest still advertises ``[-127, 127]``.
    """
    _model, src, manifest = q_tier
    dst = tmp_path / "zeroed"
    shutil.copytree(src, dst)
    entry = next(e for e in manifest["modules"].values() if e.get("weight_file"))
    assert entry["weight_int_min"] < 0 < entry["weight_int_max"]
    path = dst / entry["weight_file"]
    np.save(path, np.zeros_like(np.load(path)))

    with pytest.raises(ValueError, match="spans"):
        load_integer_manifest(dst)


def test_module_paths_that_sanitize_alike_get_distinct_artifacts(tmp_path):
    """``blk.0`` and ``blk_0`` both sanitize to ``blk_0``; only the order prefix separates them.

    Without it the second export overwrites the first ``.npy`` and both modules load
    the *same* weights — a corruption no other fixture in this file can produce,
    because no other fixture has two paths that collide.
    """
    first, second = _bare_ilinear(seed=7), _bare_ilinear(seed=99)
    assert not torch.equal(first.weight_int, second.weight_int), "fixture linears are identical"
    manifest = export_integer_model(_Colliding(first, second), tmp_path)

    files = {name: entry["weight_file"] for name, entry in manifest["modules"].items()}
    assert set(files) == {"blk.0", "blk_0"}
    assert files["blk.0"] != files["blk_0"], "colliding paths shared one artifact file"

    loaded = load_integer_manifest(tmp_path)
    assert np.array_equal(loaded["modules"]["blk.0"]["weight_int"].astype(np.int64),
                          first.weight_int.cpu().numpy().astype(np.int64))
    assert np.array_equal(loaded["modules"]["blk_0"]["weight_int"].astype(np.int64),
                          second.weight_int.cpu().numpy().astype(np.int64))


# --- the dispatch must not drop modules ---------------------------------------

def test_export_refuses_to_silently_drop_an_unexportable_stateful_module(tmp_path):
    """A FrameModel still holds float convs; strict export names them instead of dropping them."""
    model = _converted_model()
    with pytest.raises(TypeError, match="not exportable") as excinfo:
        export_integer_model(model, tmp_path)
    message = str(excinfo.value)
    for name in ("patch_embed.proj", "mask_head.proj", "mask_head.to_logits"):
        assert name in message


def test_allow_unquantized_records_the_hole_instead_of_hiding_it(tmp_path):
    model = _converted_model()
    manifest = export_integer_model(model, tmp_path, allow_unquantized=True)
    assert manifest["num_unexported"] == len(manifest["unexported"]) == 3
    assert {r["name"] for r in manifest["unexported"]} == {
        "patch_embed.proj", "mask_head.proj", "mask_head.to_logits"}
    assert all(r["class"] == "Conv2d" for r in manifest["unexported"])
    assert all(set(r["parameters"]) == {"weight", "bias"} for r in manifest["unexported"])
    assert set(manifest["unexported"][0]) >= {"name", "class", "defined_in", "parameters", "buffers"}


def test_a_quantized_module_with_no_export_branch_is_always_fatal(tmp_path):
    """``allow_unquantized`` must not reach a quantization-package module.

    That is the exact shape of the bug: ``ILayerNorm``/``ISoftmax`` existed in the tree,
    no dispatch branch matched them, and the walk's bare ``continue`` dropped them while
    ``num_modules`` and ``counts`` stayed plausible.
    """
    holder = _Holder(AffineFakeQuantizer())
    for allow in (False, True):
        with pytest.raises(TypeError, match="no export branch"):
            export_integer_model(holder, tmp_path, allow_unquantized=allow)


def test_stateless_containers_are_skipped_without_complaint(tmp_path):
    model = nn.Sequential(nn.ReLU(), nn.Dropout(), nn.Identity())
    manifest = export_integer_model(model, tmp_path)
    assert manifest["num_modules"] == 0 and manifest["num_unexported"] == 0


def test_save_int_refuses_a_non_integral_weight_buffer(tmp_path):
    """The only genuinely lossy cast ``_save_int`` can face: a buffer that was never rounded."""
    linear = _bare_ilinear()
    linear.weight_int = linear.weight_int.float() + 0.5
    with pytest.raises(ValueError, match="non-integral"):
        export_integer_model(_Holder(linear), tmp_path)


# --- replay-from-artifacts equivalence ----------------------------------------

def test_verify_export_reproduces_the_live_model(either_tier):
    model, out, _manifest = either_tier
    ok, max_diff = verify_export(model, out)
    assert ok, f"replay from artifacts disagreed, max abs diff {max_diff:.3e}"
    assert max_diff < 1e-4


def test_every_exported_module_is_actually_replayed(either_tier):
    """No op may be signed off without being compared.

    A previous revision replayed only ILinear and QGeLU and reported ``skipped, ok=True,
    max_abs_diff=0.0`` for the rest — 7 of 27 modules, 26% of the dump, with a
    fabricated zero standing in for a measurement that never ran.
    """
    model, out, manifest = either_tier
    report = verify_export_report(model, out)
    assert len(report) == manifest["num_modules"] == 27
    unchecked = [r for r in report if r["status"] != "checked"]
    assert unchecked == [], f"{len(unchecked)} exported module(s) were never replayed"
    assert all(isinstance(r["max_abs_diff"], float) for r in report)
    assert {r["type"] for r in report} == set(manifest["counts"])


def test_int_tier_replay_is_bit_exact(int_tier):
    """The I tier is pure integer, so 'close enough' is not the bar — it must be identical."""
    model, out, _manifest = int_tier
    report = verify_export_report(model, out)
    worst = max(report, key=lambda r: r["max_abs_diff"])
    assert worst["max_abs_diff"] == 0.0, f"{worst['name']} ({worst['type']}) drifted"


def test_gelu_replay_clamps_at_bound_not_at_the_table_length(tmp_path):
    """``bound`` and the table length are independent, and the cursor clamps at ``bound``.

    Nothing forbids a manifest whose table is longer than its index range — the entries
    above ``bound`` are simply unreachable. A replay that clamps to ``len(table) - 1``
    reads them anyway and disagrees with the hardware for every saturating input, yet is
    indistinguishable on any calibrator-built table (where ``bound == entries - 1``).
    The probe deliberately overshoots the table's input span so the clamp is exercised
    rather than assumed.
    """
    entries, bound = 64, 31
    gelu = IGeLU([0, 0, bound], list(range(entries)), input_scale=0.05, output_scale=0.01)
    manifest = export_integer_model(_Holder(gelu), tmp_path)

    entry = manifest["modules"]["op"]
    assert entry["scalars"][2] == bound
    assert entry["table_entries"] == entries > bound + 1, "fixture must have unreachable entries"

    ok, max_diff = verify_export(_Holder(gelu), tmp_path)
    assert ok and max_diff == 0.0, f"saturating GeLU inputs replayed differently ({max_diff:.3e})"


def test_linear_replay_is_bit_exact(q_tier):
    """Every ILinear must replay *identically*, not merely within atol."""
    model, out, _manifest = q_tier
    report = verify_export_report(model, out)
    linears = [r for r in report if r["type"] == "ILinear"]
    assert linears
    assert all(r["status"] == "checked" for r in linears)
    assert max(r["max_abs_diff"] for r in linears) == 0.0


@pytest.mark.parametrize("weight_gran", ["per-tensor", "per-channel"])
@pytest.mark.parametrize("act_symmetric", [True, False])
def test_verify_export_across_quant_configs(tmp_path, weight_gran, act_symmetric):
    model = _converted_model(weight_gran=weight_gran, act_symmetric=act_symmetric)
    manifest = export_integer_model(model, tmp_path, allow_unquantized=True)
    entry = next(e for e in manifest["modules"].values() if e["type"] == "ILinear")
    expected_slots = 1 if weight_gran == "per-tensor" else entry["out_features"]
    assert len(entry["weight_scale"]) == expected_slots
    ok, max_diff = verify_export(model, tmp_path)
    assert ok, f"{weight_gran}/sym={act_symmetric} replay diff {max_diff:.3e}"


# --- IConv2d: exercised, not shipped as "verified" while untested --------------

@pytest.mark.parametrize("act_zero_point", [0.0, 5.0])
def test_conv_entry_exports_and_replays_bit_exactly(tmp_path, act_zero_point):
    """``convert_to_integer`` never emits an IConv2d, so nothing else reaches this branch."""
    conv = _bare_iconv(act_zero_point=act_zero_point)
    manifest = export_integer_model(_Holder(conv), tmp_path)

    entry = manifest["modules"]["op"]
    assert entry["type"] == "IConv2d" and entry["op"] == "conv2d_int"
    assert entry["weight_shape"] == list(conv.weight_int.shape) == [3, 2, 4, 4]
    assert entry["stride"] == conv.stride == 4
    assert entry["act_zero_point"] == conv.act_zero_point == int(act_zero_point)
    assert len(entry["weight_scale"]) == 3

    ok, max_diff = verify_export(_Holder(conv), tmp_path)
    assert ok and max_diff == 0.0, f"conv replay drifted by {max_diff:.3e}"


def test_conv_replay_detects_a_corrupted_weight_file(tmp_path):
    conv = _bare_iconv()
    manifest = export_integer_model(_Holder(conv), tmp_path)
    path = tmp_path / manifest["modules"]["op"]["weight_file"]
    np.save(path, _corrupt_one_value_in_range(np.load(path)))

    ok, max_diff = verify_export(_Holder(conv), tmp_path)
    assert ok is False, "a corrupted conv weight replayed clean — the branch has no teeth"
    assert max_diff > 1e-4


# --- negative controls (the checks must be able to fail) ----------------------

def test_verify_export_detects_a_corrupted_weight(q_tier, tmp_path):
    model, src, manifest = q_tier
    dst = tmp_path / "corrupt"
    shutil.copytree(src, dst)
    entry = next(e for e in manifest["modules"].values() if e["type"] == "ILinear")
    path = dst / entry["weight_file"]
    np.save(path, _corrupt_one_value_in_range(np.load(path)))

    ok, max_diff = verify_export(model, dst)
    assert ok is False, "verify_export passed a corrupted weight — the check has no teeth"
    assert max_diff > 1e-4
    bad = [r for r in verify_export_report(model, dst) if not r["ok"]]
    assert [r["name"] for r in bad] == [entry["name"]]


def _zero_tables_in_manifest(model, src, dst, manifest, keys):
    """Zero every occurrence of ``keys`` in a copied dump; returns (ok, max_diff, victims)."""
    shutil.copytree(src, dst)
    on_disk = json.loads((dst / MANIFEST_NAME).read_text())
    victims = sorted({e["name"] for e in manifest["modules"].values()
                      for key in keys if key in e})
    assert victims, f"no exported module carries any of {keys}"
    for name in victims:
        for key in keys:
            if key in on_disk["modules"][name]:
                on_disk["modules"][name][key] = [0] * len(on_disk["modules"][name][key])
    (dst / MANIFEST_NAME).write_text(json.dumps(on_disk))
    ok, max_diff = verify_export(model, dst)
    return ok, max_diff, victims


@pytest.mark.parametrize("table_key", ["table", "rsqrt_table", "exp_table"])
def test_verify_export_detects_a_corrupted_lut_in_the_manifest(int_tier, tmp_path, table_key):
    """The GeLU, LayerNorm-rsqrt and softmax-exp tables are all load-bearing and all checked."""
    model, src, manifest = int_tier
    ok, max_diff, victims = _zero_tables_in_manifest(
        model, src, tmp_path / f"corrupt_{table_key}", manifest, (table_key,))
    assert ok is False and max_diff > 1e-4
    flagged = {r["name"] for r in verify_export_report(model, tmp_path / f"corrupt_{table_key}")
               if not r["ok"]}
    assert flagged == set(victims)


def test_verify_export_detects_a_corrupted_reciprocal_table(int_tier, tmp_path):
    """The segmented reciprocal is checked — with a caveat this test refuses to hide.

    ``ISoftmax`` picks one of two reciprocal tables from the row accumulator, so a
    given payload only ever exercises one of them. With the shipped calibration no exp
    entry falls below ~40% of the table maximum, which puts every 16-token row above
    the segment threshold: ``recip_table_one`` is unreachable for this fixture and
    zeroing it is *provably* a no-op, not a gap in the replay.

    So the fixture-independent claim is that corrupting both tables always fails, and
    the per-table outcome is asserted afterwards: if a future calibration flips which
    segment is live, ``any(...)`` still holds and this test cannot silently become
    vacuous the way a hardcoded single-table corruption would.
    """
    model, src, manifest = int_tier
    both_ok, both_diff, victims = _zero_tables_in_manifest(
        model, src, tmp_path / "corrupt_recip_both", manifest,
        ("recip_table_one", "recip_table_two"))
    assert both_ok is False and both_diff > 1e-4
    flagged = {r["name"] for r in verify_export_report(model, tmp_path / "corrupt_recip_both")
               if not r["ok"]}
    assert flagged == set(victims)

    detected = {}
    for key in ("recip_table_one", "recip_table_two"):
        single_ok, _diff, _victims = _zero_tables_in_manifest(
            model, src, tmp_path / f"corrupt_{key}", manifest, (key,))
        detected[key] = not single_ok
    assert any(detected.values()), (
        "neither reciprocal segment is exercised by the verification probe — the "
        f"segmented reciprocal is not being checked at all (detected={detected})")


def test_verify_export_detects_zeroed_live_luts(tmp_path):
    """The reproduction that exposed the whole defect, kept as a regression test.

    Zeroing every live rsqrt / exp / reciprocal / GeLU table makes the model and the
    manifest disagree completely. The old gate still returned ``(True, 5.96e-08)``,
    because the 13 modules that carry those tables were all in the skipped set.
    """
    model = _int_tier_model()
    export_integer_model(model, tmp_path, allow_unquantized=True)
    assert verify_export(model, tmp_path)[0] is True, "fixture must start out verifiable"

    zeroed = 0
    for _name, module in model.named_modules():
        for buffer_name in ("table", "rsqrt_table", "exp_table",
                            "recip_table_one", "recip_table_two"):
            buffer = getattr(module, buffer_name, None)
            if torch.is_tensor(buffer):
                buffer.zero_()
                zeroed += 1
    assert zeroed >= 13, f"only {zeroed} LUT buffers found — fixture is not the intended one"

    ok, max_diff = verify_export(model, tmp_path)
    assert ok is False, "the gate signed off on a dump that no longer matches the model"
    assert max_diff > 1e-4


def test_verify_export_flags_a_manifest_type_that_does_not_match_the_live_module(int_tier, tmp_path):
    model, src, manifest = int_tier
    dst = tmp_path / "mistyped"
    shutil.copytree(src, dst)
    on_disk = json.loads((dst / MANIFEST_NAME).read_text())
    victim = next(e["name"] for e in manifest["modules"].values() if e["type"] == "ILayerNorm")
    on_disk["modules"][victim]["type"] = "QLayerNorm"          # the old hardcoded-type bug, inverted
    (dst / MANIFEST_NAME).write_text(json.dumps(on_disk))

    report = {r["name"]: r for r in verify_export_report(model, dst)}
    assert report[victim]["status"] == "type_mismatch"
    assert report[victim]["ok"] is False and report[victim]["max_abs_diff"] is None
    assert verify_export(model, dst)[0] is False
    # a graph disagreement is never excusable, not even under the lenient opt-in
    assert verify_export(model, dst, allow_unverified=True)[0] is False


def test_unverified_entries_report_no_measurement_and_fail_the_gate(int_tier, tmp_path):
    """An op the replayer cannot rebuild must never contribute a fabricated ``0.0``."""
    model, src, manifest = int_tier
    dst = tmp_path / "unknown_op"
    shutil.copytree(src, dst)
    on_disk = json.loads((dst / MANIFEST_NAME).read_text())
    victim = next(e["name"] for e in manifest["modules"].values() if e["type"] == "ISoftmax")
    on_disk["modules"][victim]["op"] = "some_future_op"
    (dst / MANIFEST_NAME).write_text(json.dumps(on_disk))

    report = {r["name"]: r for r in verify_export_report(model, dst)}
    assert report[victim]["status"] == "unverified"
    assert report[victim]["ok"] is False
    assert report[victim]["max_abs_diff"] is None
    assert verify_export(model, dst)[0] is False, "an unverified op must not pass the gate"


def test_allow_unverified_is_an_explicit_opt_in(int_tier, tmp_path):
    """Leniency has to be asked for by name; the default is the strict gate."""
    model, src, manifest = int_tier
    dst = tmp_path / "lenient"
    shutil.copytree(src, dst)
    on_disk = json.loads((dst / MANIFEST_NAME).read_text())
    victim = next(e["name"] for e in manifest["modules"].values() if e["type"] == "ISoftmax")
    on_disk["modules"][victim]["op"] = "some_future_op"
    (dst / MANIFEST_NAME).write_text(json.dumps(on_disk))

    assert verify_export(model, dst)[0] is False
    ok, max_diff = verify_export(model, dst, allow_unverified=True)
    assert ok is True and max_diff < 1e-4


def test_verify_export_flags_a_name_that_no_longer_resolves(tmp_path):
    export_integer_model(_Holder(_bare_ilinear()), tmp_path)
    ok, _max_diff = verify_export(nn.Module(), tmp_path)
    assert ok is False
    report = verify_export_report(nn.Module(), tmp_path)
    assert [r["status"] for r in report] == ["missing"]
    assert report[0]["max_abs_diff"] is None and report[0]["ok"] is False


def test_verify_export_does_not_vacuously_pass_on_an_empty_export(tmp_path):
    model = nn.Sequential(nn.ReLU())
    manifest = export_integer_model(model, tmp_path)
    assert manifest["num_modules"] == 0
    ok, _max_diff = verify_export(model, tmp_path)
    assert ok is False, "an export with nothing to check must not report success"
    assert verify_export(model, tmp_path, allow_unverified=True)[0] is False


# --- hardening: container width + JSON coercion -------------------------------

def test_wide_integer_weights_use_a_wider_container_losslessly(tmp_path):
    linear = _bare_ilinear()
    wide = linear.weight_int.clone()
    wide[0, 0], wide[0, 1] = 5000, -6000          # beyond int8: must not silently wrap
    linear.weight_int.copy_(wide)
    manifest = export_integer_model(_Holder(linear), tmp_path)

    entry = manifest["modules"]["op"]
    assert entry["weight_dtype"] == "int16"
    assert entry["weight_bits"] == 14 and entry["weight_bits_inferred"] is True
    assert (entry["weight_int_min"], entry["weight_int_max"]) == (-6000, 5000)
    loaded = load_integer_manifest(tmp_path)
    assert np.array_equal(loaded["modules"]["op"]["weight_int"].astype(np.int64),
                          wide.cpu().numpy().astype(np.int64))


def test_manifest_survives_numpy_and_torch_scalar_attributes(tmp_path):
    linear = _bare_ilinear()
    linear.act_scale = np.float32(linear.act_scale)               # numpy scalar
    linear.act_zero_point = torch.tensor(3)                       # 0-d torch tensor
    manifest = export_integer_model(_Holder(linear), tmp_path)

    entry = json.loads((tmp_path / MANIFEST_NAME).read_text())["modules"]["op"]
    assert isinstance(entry["act_scale"], float) and entry["act_scale"] > 0
    assert entry["act_zero_point"] == 3
    assert manifest["modules"]["op"]["act_scale"] == entry["act_scale"]
