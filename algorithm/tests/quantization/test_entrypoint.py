"""The shipped CLI path (D1): ``quantization.entrypoint`` end to end.

Everything here exists because the entrypoint was the one piece of the quantization
stack with no test at all, and it was broken in two independent ways:

1. It imported ``engine.data.factory`` at module scope, which drags the whole
   dataset pipeline (PIL / cv2 / h5py / tonic) in at import time — so
   ``import quantization.entrypoint`` raised ``ModuleNotFoundError`` in any
   environment that has torch but no image stack, and ``hbtxr quantize`` was a dead
   command. :func:`test_import_does_not_pull_in_the_data_pipeline` is the guard, and
   it runs in a *subprocess* so it stays honest even where PIL happens to be
   installed: it fails on the leaked module, not on the crash.
2. The hybrid ``forward_fn`` drove ``search_step`` only, leaving the four track-path
   QLinear layers with no observations and killing PTQ inside
   ``AffineObserver.qparams()``. See the two hybrid tests at the bottom.

``build_dataloader`` is the only thing faked: it is the sole step that needs a
dataset on disk. Model construction, calibration, QAT fine-tuning, checkpointing,
Q->I conversion and the integer export all run for real.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types
from typing import Any, Iterator

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from torch import nn

from engine.model_factory import make_model
from quantization import entrypoint
from quantization.export import verify_export
from quantization.qlayers.linear import QLinear

# algorithm/ — the import root the entrypoint is run from.
_ROOT = Path(__file__).resolve().parents[2]

# Top-level packages that must NOT be reachable from a bare entrypoint import. The
# dataset pipeline and its third-party deps are exactly what made the module
# unimportable; ``engine.data.factory`` is the submodule that pulls them in (and the
# one the PEP-562 guard in ``engine.data.__init__`` cannot protect, since importing a
# submodule directly bypasses the package ``__getattr__``).
_DATA_PIPELINE_ROOTS = ("PIL", "cv2", "h5py", "tonic", "dataset")
_DATA_PIPELINE_MODULES = ("engine.data.factory",)

_FRAME_MODEL = {
    "target": "models.frame.FrameModel",
    "embed_dim": 48,
    "patch_size": 16,
    "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1},
}
_HYBRID_MODEL = {**_FRAME_MODEL, "target": "models.hybrid.HybridModel"}

# I-tier module classes an integer export may contain. Spelled out here rather than
# imported from quantization.export so that renaming a class in the exporter breaks
# this test instead of silently redefining what "integer dump" means.
_INT_TIER_TYPES = {"ILinear", "IConv2d", "IGeLU", "ILayerNorm", "ISoftmax"}


# --- helpers -----------------------------------------------------------------

class _CountingLoader:
    """Re-iterable batch source that records how much of it was actually consumed.

    Re-iterable rather than a generator because ``Trainer.fit`` loops the loader once
    per epoch; a generator would silently be empty from the second epoch on.
    """

    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self.batches = batches
        self.batches_seen = 0
        self.epochs_seen = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.epochs_seen += 1
        for batch in self.batches:
            self.batches_seen += 1
            yield batch


class _FakeDataloaderFactory:
    """Stand-in for ``engine.data.factory.build_dataloader``.

    The only faked step in these tests: it is the one call that needs an HBTXR
    dataset on disk. Every call is recorded so a test can assert *how* the entrypoint
    asked for its data (shuffle / modality) and how much of it was consumed.
    """

    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self._batches = batches
        self.calls: list[dict[str, Any]] = []
        self.loaders: list[_CountingLoader] = []

    def __call__(self, manifest: str, cfg: dict[str, Any], *, shuffle: bool,
                 modality: str | None = None) -> _CountingLoader:
        loader = _CountingLoader(self._batches)
        self.calls.append({"manifest": manifest, "shuffle": bool(shuffle), "modality": modality})
        self.loaders.append(loader)
        return loader


def _write_config(tmp_path: Path, model_cfg: dict[str, Any], quant_cfg: dict[str, Any],
                  *, training: dict[str, Any] | None = None, name: str = "experiment") -> str:
    cfg = {
        "experiment": {"name": "quant_entrypoint_test", "project_root": str(tmp_path)},
        "model": model_cfg,
        "training": {"device": "cpu", **(training or {})},
        "quantization": quant_cfg,
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(path)


def _frame_batches(count: int = 2, batch: int = 2) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    return [{"image": torch.rand(batch, 1, 64, 64), "box": torch.rand(batch, 5)}
            for _ in range(count)]


def _hybrid_batches(count: int = 2, batch: int = 2) -> list[dict[str, Any]]:
    """A hybrid batch as ``engine.data.adapter.adapt_batch`` emits it (frame + event)."""
    torch.manual_seed(1)
    return [{
        "frame": torch.rand(batch, 1, 64, 64),
        "event": torch.rand(batch, 2, 64, 64),
        "anchor_state": torch.rand(batch, 5),
        "state": torch.rand(batch, 5),
        "box": torch.rand(batch, 5),
        "residual": torch.rand(batch, 5) * 0.1,
    } for _ in range(count)]


def _install_fake_loader(monkeypatch: pytest.MonkeyPatch,
                         batches: list[dict[str, Any]]) -> _FakeDataloaderFactory:
    factory = _FakeDataloaderFactory(batches)
    monkeypatch.setattr(entrypoint, "_build_dataloader", factory)
    return factory


def _unwrapped_linears(model: nn.Module) -> list[str]:
    """Names of ``nn.Linear`` modules that are NOT the payload of a ``QLinear``."""
    wrapped = {f"{name}.linear" for name, m in model.named_modules() if isinstance(m, QLinear)}
    return [name for name, m in model.named_modules()
            if isinstance(m, nn.Linear) and name not in wrapped]


def _count_linears(model_cfg: dict[str, Any]) -> int:
    """How many ``nn.Linear`` a freshly built model of this config has.

    The expectation for the quantized graph comes from the *model factory*, not from
    the quantizer, so a conversion that quietly skipped layers cannot move the target.
    """
    return sum(1 for m in make_model(model_cfg).modules() if isinstance(m, nn.Linear))


def _observation_counts(registry: dict[str, QLinear], model: nn.Module,
                        forward_fn, batches: list[dict[str, Any]]) -> dict[str, int]:
    """Times each registered ``QLinear`` saw an input while ``forward_fn`` drove the model."""
    seen = {name: 0 for name in registry}

    def make_hook(name: str):
        def hook(_module, _inputs) -> None:
            seen[name] += 1
        return hook

    handles = [q.register_forward_pre_hook(make_hook(n)) for n, q in registry.items()]
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                forward_fn(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return seen


# --- (a) the import guard ------------------------------------------------------

def test_import_does_not_pull_in_the_data_pipeline() -> None:
    """A bare ``import quantization.entrypoint`` must reach no dataset dependency.

    Run in a fresh interpreter, because in-process ``sys.modules`` is already polluted
    by whatever the rest of the suite imported. Asserting on the *leaked modules*
    rather than on the import succeeding is what keeps this a regression guard rather
    than an environment probe: re-add ``from engine.data.factory import ...`` at module
    scope and this fails even on a machine where PIL, cv2 and tonic are all installed.
    """
    code = textwrap.dedent(f"""
        import sys
        import quantization.entrypoint  # noqa: F401
        roots = {_DATA_PIPELINE_ROOTS!r}
        modules = {_DATA_PIPELINE_MODULES!r}
        leaked = sorted(m for m in sys.modules
                        if m.split(".")[0] in roots or m in modules)
        print(";".join(leaked))
    """)
    env = {**os.environ, "PYTHONPATH": str(_ROOT), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"importing quantization.entrypoint failed:\n{proc.stderr}"
    assert proc.stdout.strip() == "", (
        f"entrypoint import reached the data pipeline: {proc.stdout.strip()}")


def test_build_dataloader_proxy_forwards_to_the_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy proxy defers the import to call time and passes the call through verbatim.

    Substituting the module in ``sys.modules`` is only possible because the import
    lives inside the function body; it also proves the proxy is a pass-through rather
    than a stub, which the run_quantize tests below take for granted when they patch it.
    """
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fake = types.ModuleType("engine.data.factory")

    def build_dataloader(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "loader-sentinel"

    fake.build_dataloader = build_dataloader
    monkeypatch.setitem(sys.modules, "engine.data.factory", fake)

    result = entrypoint._build_dataloader("train.jsonl", {"data": {}}, shuffle=True, modality="frame")

    assert result == "loader-sentinel"
    assert calls == [(("train.jsonl", {"data": {}}), {"shuffle": True, "modality": "frame"})]


# --- (b) PTQ end to end --------------------------------------------------------

def test_run_quantize_ptq_quantizes_every_linear(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """PTQ through the real entrypoint leaves a graph whose every Linear is a QLinear."""
    factory = _install_fake_loader(monkeypatch, _frame_batches())
    config = _write_config(tmp_path, _FRAME_MODEL,
                           {"mode": "ptq", "num_calib_batches": 2,
                            "weight": {"bits": 8, "granularity": "per-channel"},
                            "activation": {"bits": 8}})

    model = entrypoint.run_quantize(config)

    expected = _count_linears(_FRAME_MODEL)
    assert expected > 0
    assert sum(1 for m in model.modules() if isinstance(m, QLinear)) == expected
    assert _unwrapped_linears(model) == []
    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert (call["shuffle"], call["modality"]) == (False, "frame")
    # the manifest the entrypoint asked for is the run contract's train split under
    # the configured project_root — not some path invented by run_quantize.
    manifest = Path(call["manifest"])
    assert manifest.name == "train_manifest.jsonl" and tmp_path in manifest.parents
    assert factory.loaders[0].batches_seen == 2, "calibration never consumed the batches"


def test_run_quantize_ptq_respects_num_calib_batches(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """``num_calib_batches`` caps what calibration pulls off the loader."""
    factory = _install_fake_loader(monkeypatch, _frame_batches(count=5))
    config = _write_config(tmp_path, _FRAME_MODEL, {"mode": "ptq", "num_calib_batches": 2})

    entrypoint.run_quantize(config)

    assert factory.loaders[0].batches_seen == 2


def test_run_quantize_writes_the_configured_checkpoint(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """``quantization.output`` gets a loadable checkpoint carrying the quantization meta."""
    _install_fake_loader(monkeypatch, _frame_batches())
    out = tmp_path / "nested" / "quantized.pt"
    config = _write_config(tmp_path, _FRAME_MODEL,
                           {"mode": "ptq", "num_calib_batches": 2, "output": str(out),
                            "weight": {"bits": 4}, "activation": {"bits": 8}})

    model = entrypoint.run_quantize(config)

    assert out.is_file()
    payload = torch.load(out, map_location="cpu")
    assert payload["meta"] == {"quantized": True, "mode": "ptq", "modality": "frame",
                               "weight_bits": 4, "act_bits": 8}
    assert set(payload["state_dict"]) == set(model.state_dict())


# --- (c) QAT end to end --------------------------------------------------------

def test_run_quantize_qat_fine_tunes_the_quantized_model(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """The QAT branch calibrates, then really runs ``Trainer.fit`` over the batches.

    Two independent witnesses that training happened rather than being skipped: the
    shuffled fit loader was iterated once per epoch, and parameters carry non-zero
    gradients afterwards (``training_step`` zeroes grads *before* the backward pass,
    so a populated ``.grad`` cannot be left over from anything else).
    """
    factory = _install_fake_loader(monkeypatch, _frame_batches(count=3))
    config = _write_config(tmp_path, _FRAME_MODEL,
                           {"mode": "qat", "num_calib_batches": 2},
                           training={"epochs": 2, "lr": 1e-3})

    model = entrypoint.run_quantize(config)

    assert sum(1 for m in model.modules() if isinstance(m, QLinear)) == _count_linears(_FRAME_MODEL)
    assert [c["shuffle"] for c in factory.calls] == [False, True], (
        "QAT must build a calibration loader and then a shuffled training loader")
    fit_loader = factory.loaders[1]
    assert fit_loader.epochs_seen == 2
    assert fit_loader.batches_seen == 6
    assert any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in model.parameters()), (
        "no parameter carries a gradient — Trainer.fit did not back-propagate")


# --- (d) integer export --------------------------------------------------------

def test_run_quantize_export_int_writes_a_verifiable_integer_dump(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--export-int`` produces an on-disk manifest of I-tier ops that replays exactly.

    This is the first end-to-end proof of the shipped deployment path: config ->
    PTQ -> ``convert_model_to_integer`` -> artifact dump. Everything is asserted
    against the manifest *read back from disk* and against the live model, never
    against the manifest object the exporter handed back.
    """
    _install_fake_loader(monkeypatch, _frame_batches())
    export_dir = tmp_path / "int_export"
    config = _write_config(tmp_path, _FRAME_MODEL, {"mode": "ptq", "num_calib_batches": 2})

    model, returned = entrypoint.run_quantize(config, export_int=str(export_dir),
                                              return_manifest=True)

    manifest_path = export_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert returned is not None
    assert manifest["num_modules"] > 0

    # 1. Nothing float is left that the integer tier can express. Without this the
    #    dump can be "all integer" simply by having converted less of the graph —
    #    a stage that never ran leaves plain nn.LayerNorm / nn.GELU behind, and those
    #    just drop out of the counts. The padded mask conv is the one documented
    #    exception and is checked below.
    leftover = [name for name, m in model.named_modules()
                if isinstance(m, (nn.Linear, nn.LayerNorm, nn.GELU, nn.Softmax))]
    assert leftover == [], f"float ops the export should have replaced: {leftover}"

    # 2. The manifest accounts for every integer module the live model has.
    live_counts = Counter(type(m).__name__ for _, m in model.named_modules()
                          if type(m).__name__ in _INT_TIER_TYPES)
    assert set(manifest["counts"]) <= _INT_TIER_TYPES, (
        f"non-integer entries in the dump: {sorted(set(manifest['counts']) - _INT_TIER_TYPES)}")
    assert manifest["counts"] == dict(live_counts)
    assert manifest["counts"].get("ILinear") == _count_linears(_FRAME_MODEL)

    # 3. Every entry names a live module of the class it claims, and the holes are
    #    named rather than silently omitted.
    live = dict(model.named_modules())
    for name, entry in manifest["modules"].items():
        assert type(live[name]).__name__ == entry["type"]
    for record in manifest["unexported"]:
        assert record["name"] in live
        assert not record["defined_in"].startswith("quantization.")
    # D2 taught IConv2d padding, so the padded 3x3 mask conv — previously the single
    # documented hole in a FrameModel dump — now converts and the list is empty.
    # Pinning it EXACTLY (rather than asserting "small") is what makes a conversion
    # stage that quietly did not run visible: everything it skipped shows up here.
    assert [r["name"] for r in manifest["unexported"]] == []
    assert any(entry["weight_file"] for entry in manifest["modules"].values()
               if "weight_file" in entry)

    ok, max_diff = verify_export(model, export_dir)
    assert ok, f"replaying the dump disagreed with the live model (max abs diff {max_diff})"


def test_run_quantize_without_export_int_writes_nothing(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``export_int`` -> no dump, and ``return_manifest`` says so instead of guessing."""
    _install_fake_loader(monkeypatch, _frame_batches())
    config = _write_config(tmp_path, _FRAME_MODEL, {"mode": "ptq", "num_calib_batches": 2})

    model, manifest = entrypoint.run_quantize(config, return_manifest=True)

    assert manifest is None
    assert sum(1 for m in model.modules() if isinstance(m, QLinear)) > 0
    assert not list(tmp_path.glob("**/manifest.json"))


# --- (e) the hybrid regression -------------------------------------------------

def test_hybrid_forward_fn_observes_every_quantized_layer() -> None:
    """THE hybrid regression guard: the calibration drive must reach both branches.

    The shipped ``lambda model, batch: model.search_step(batch["frame"])`` never
    touched ``event_stem``, ``backbone.forward_track``, ``track_head`` or
    ``track_reliability``, so those quantizers saw no data at all. Fails loudly with
    the offending layer names, which is the information the old failure
    (``AffineObserver.qparams() called before any observe()``) did not carry.
    """
    from quantization.convert import insert_fake_quant

    model = make_model(_HYBRID_MODEL)
    model, registry = insert_fake_quant(model)
    assert len(registry) == _count_linears(_HYBRID_MODEL)

    seen = _observation_counts(registry, model, entrypoint._make_forward_fn("hybrid"),
                               _hybrid_batches())

    unobserved = sorted(name for name, count in seen.items() if count == 0)
    assert unobserved == [], f"{len(unobserved)} QLinear never observed: {unobserved}"


def test_hybrid_forward_fn_drives_track_without_a_batch_anchor() -> None:
    """With no anchor in the batch the drive chains from search, as ``run_step`` does."""
    from quantization.convert import insert_fake_quant

    model = make_model(_HYBRID_MODEL)
    model, registry = insert_fake_quant(model)
    bare = [{k: v for k, v in batch.items() if k in ("frame", "event")}
            for batch in _hybrid_batches()]

    seen = _observation_counts(registry, model, entrypoint._make_forward_fn("hybrid"), bare)

    assert sorted(name for name, count in seen.items() if count == 0) == []


def test_run_quantize_ptq_hybrid_completes_and_both_branches_still_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hybrid PTQ through the entrypoint: it finishes, and the result still forwards.

    Completing at all is itself the calibration proof — ``calibrate_activations`` calls
    ``qparams()`` on *every* registered quantizer, which raises for any that was never
    observed. That is the exact exception the search-only drive used to hit.
    """
    _install_fake_loader(monkeypatch, _hybrid_batches())
    config = _write_config(tmp_path, _HYBRID_MODEL, {"mode": "ptq", "num_calib_batches": 2})

    model = entrypoint.run_quantize(config)

    assert sum(1 for m in model.modules() if isinstance(m, QLinear)) == _count_linears(_HYBRID_MODEL)
    for quant in (m for m in model.modules() if isinstance(m, QLinear)):
        assert bool(torch.all(quant.act_fq.scale > 0))
        assert bool(torch.all(torch.isfinite(quant.weight_fq.scale)))
    with torch.no_grad():
        search = model.search_step(torch.rand(2, 1, 64, 64))
        track = model.track_step(torch.rand(2, 2, 64, 64), search["state"])
    assert bool(torch.isfinite(search["box"]).all())
    assert bool(torch.isfinite(track["residual"]).all())


# --- (f) CLI argument parsing --------------------------------------------------

def test_main_forwards_export_int_to_run_quantize(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--export-int DIR`` reaches ``run_quantize`` as ``export_int``.

    Argparse wiring only: ``run_quantize`` is replaced, so this proves the CLI surface,
    not that an export happens (that is
    test_run_quantize_export_int_writes_a_verifiable_integer_dump).
    """
    recorded: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(entrypoint, "run_quantize",
                        lambda *args, **kwargs: recorded.append((args, kwargs)))

    entrypoint.main(["-c", "cfg.yaml", "--ckpt", "fp.pt", "-o", "q.pt",
                     "--project-root", "/root", "--device", "cpu",
                     "--export-int", "dump_dir"])

    assert recorded == [(("cfg.yaml",), {"ckpt": "fp.pt", "output": "q.pt",
                                         "project_root": "/root", "device": "cpu",
                                         "export_int": "dump_dir"})]


def test_main_defaults_export_int_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the flag the export is off — the optional argument is not sticky."""
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(entrypoint, "run_quantize",
                        lambda *args, **kwargs: recorded.append(kwargs))

    entrypoint.main(["-c", "cfg.yaml"])

    assert recorded == [{"ckpt": None, "output": None, "project_root": None,
                         "device": None, "export_int": None}]


def test_main_requires_a_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entrypoint, "run_quantize", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit):
        entrypoint.main([])


# --- the anchor-selection policy is a decision, so pin it -------------------
#
# _hybrid_forward argues at length for anchor_state > state > search["state"], but an
# argument in a docstring is not a guard: both reorderings survived the rest of this
# file, and each changes the activation distribution calibration observes.

def _capture_track_anchor(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """Run the entrypoint's hybrid drive and return the anchor track_step received."""
    seen: dict[str, torch.Tensor] = {}
    original = model.track_step

    def spy(event, anchor):
        seen["anchor"] = anchor
        return original(event, anchor)

    model.track_step = spy  # type: ignore[method-assign]
    try:
        entrypoint._make_forward_fn("hybrid")(model, batch)
    finally:
        model.track_step = original  # type: ignore[method-assign]
    return seen["anchor"]


def test_hybrid_anchor_prefers_the_batch_anchor_state() -> None:
    model = make_model(_HYBRID_MODEL)
    batch = _hybrid_batches(count=1)[0]
    anchor = _capture_track_anchor(model, batch)
    assert torch.equal(anchor, batch["anchor_state"]), "anchor_state must win when present"
    # and the alternatives really are distinguishable, so the assertion has content
    assert not torch.equal(batch["anchor_state"], batch["state"])


def test_hybrid_anchor_falls_back_to_state_then_to_the_search_output() -> None:
    model = make_model(_HYBRID_MODEL)
    batch = _hybrid_batches(count=1)[0]

    without_anchor = {k: v for k, v in batch.items() if k != "anchor_state"}
    anchor = _capture_track_anchor(model, without_anchor)
    assert torch.equal(anchor, batch["state"]), "state is the documented second choice"

    bare = {"frame": batch["frame"], "event": batch["event"]}
    with torch.no_grad():
        expected = model.search_step(bare["frame"])["state"]
    anchor = _capture_track_anchor(model, bare)
    assert torch.allclose(anchor, expected), "a bare batch must chain from the search state"
    assert not torch.allclose(anchor, batch["state"])


# --- QAT and PTQ must be observably different -------------------------------

def test_qat_updates_weights_and_ptq_does_not(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The branch is only real if fine-tuning actually moves the weights.

    Swapping prepare_qat for post_training_quantize was an undetectable mutation: both
    insert and calibrate. What separates them is that QAT then trains. Compare the same
    weight after each path against its pre-quantization value.
    """
    def _probe_weight(model: nn.Module) -> torch.Tensor:
        # QLinear wraps the nn.Linear, so the path gains a ".linear" segment once
        # quantized; match on the attention qkv weight either way.
        for name, param in model.named_parameters():
            if "attn.qkv" in name and name.endswith("weight"):
                return param.detach().clone()
        raise AssertionError("no qkv weight found to probe")

    torch.manual_seed(0)
    before = _probe_weight(make_model(_FRAME_MODEL))

    _install_fake_loader(monkeypatch, _frame_batches(count=3))
    ptq_cfg = _write_config(tmp_path, _FRAME_MODEL,
                            {"mode": "ptq", "num_calib_batches": 2}, name="ptq")
    torch.manual_seed(0)
    ptq_model = entrypoint.run_quantize(ptq_cfg, project_root=str(tmp_path))
    assert torch.equal(_probe_weight(ptq_model), before), "PTQ must not train the weights"

    _install_fake_loader(monkeypatch, _frame_batches(count=3))
    qat_cfg = _write_config(tmp_path, _FRAME_MODEL,
                            {"mode": "qat", "num_calib_batches": 2},
                            training={"epochs": 1, "lr": 1e-2},
                            name="qat")
    torch.manual_seed(0)
    qat_model = entrypoint.run_quantize(qat_cfg, project_root=str(tmp_path))
    assert not torch.equal(_probe_weight(qat_model), before), "QAT must fine-tune the weights"


def test_qat_branch_calls_prepare_qat_not_the_ptq_helper(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the branch wiring, because behaviour cannot currently distinguish it.

    ``prepare_qat`` and ``post_training_quantize`` both insert and calibrate, and the
    QAT branch trains afterwards either way — so swapping them is an *equivalent*
    mutant today and no behavioural assertion can catch it. ``qat.py``'s docstring says
    the two will diverge (learnable-scale QAT is the stated next step), and on that day
    a silent swap would train the wrong thing. Asserting the call is the only guard
    available, so it is used deliberately rather than as a substitute for behaviour:
    ``test_qat_updates_weights_and_ptq_does_not`` covers what the branch must *do*.
    """
    calls: list[str] = []
    real_prepare = entrypoint.prepare_qat
    real_ptq = entrypoint.post_training_quantize

    def spy_prepare(*args: Any, **kwargs: Any):
        calls.append("prepare_qat")
        return real_prepare(*args, **kwargs)

    def spy_ptq(*args: Any, **kwargs: Any):
        calls.append("post_training_quantize")
        return real_ptq(*args, **kwargs)

    monkeypatch.setattr(entrypoint, "prepare_qat", spy_prepare)
    monkeypatch.setattr(entrypoint, "post_training_quantize", spy_ptq)
    _install_fake_loader(monkeypatch, _frame_batches(count=3))

    config = _write_config(tmp_path, _FRAME_MODEL, {"mode": "qat", "num_calib_batches": 2},
                           training={"epochs": 1, "lr": 1e-3}, name="qat_wiring")
    entrypoint.run_quantize(config, project_root=str(tmp_path))
    assert calls == ["prepare_qat"], f"QAT branch must use prepare_qat, saw {calls}"
