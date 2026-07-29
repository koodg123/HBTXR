"""B2: what one set of scalars for two modalities costs in the shared early blocks.

With ``cut_point < depth`` the first ``c`` blocks are traversed by BOTH paths — frames on
the search path, event voxels on the track path — and one activation observer sees both
distributions. Conversion was exercised but no number had ever been put on that.

The result, measured below and in ``docs/QUANTIZATION-PARTCD-REPORT.md``: the two
modalities' ranges really do differ (up to 2.21x on ``blocks.0.attn.proj``, where the
event range is under half the frame range), and a max-abs observer resolves the conflict
by taking the wider of the two. That costs the narrower modality up to 1.14 bits of
resolution — and buys it a range margin, because the wider grid clips less of whatever
the calibration set did not contain. Which of those two effects wins did not settle to a
consistent sign across calibration budgets; see the report for the numbers.

**These tests pin the mechanism, not an accuracy figure.** The model is randomly
initialised, and a random ViT's two input branches produce far more similar activation
distributions than a trained one's would, so the *magnitude* here bounds nothing about a
trained network. What is structural — that the shared scale is the max of the two, that
the two differ, that the wider grid cannot clip more — holds either way.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from quantization.calibrate import calibrate_weights
from quantization.convert import convert_model_to_integer, insert_fake_quant
from quantization.entrypoint import _hybrid_forward
from quantization.observer import build_observer

DEPTH, CUT, SIDE = 4, 2, 64
CFG = {"target": "models.hybrid.HybridModel", "embed_dim": 48, "patch_size": 16,
       "backbone": {"depth": DEPTH, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": CUT}}


def _batches(n=2, seed=7):
    gen = torch.Generator().manual_seed(seed)
    return [{"frame": torch.rand(2, 1, SIDE, SIDE, generator=gen),
             "event": torch.randn(2, 2, SIDE, SIDE, generator=gen) * 0.4} for _ in range(n)]


def _search(model, batch):
    return model.search_step(batch["frame"])


def _track(model, batch):
    return model.track_step(batch["event"], torch.zeros(batch["event"].shape[0], 5))


def _observe(model, registry, batches, drive):
    """Per-module activation observers under ONE drive; None where the drive never ran it."""
    observers = {n: build_observer(q.act_fq.spec) for n, q in registry.items()}
    seen: set[str] = set()

    def hook(name):
        def fn(_module, inputs):
            if inputs:
                observers[name].observe(inputs[0])
                seen.add(name)
        return fn

    handles = [q.register_forward_pre_hook(hook(n)) for n, q in registry.items()]
    try:
        with torch.no_grad():
            for batch in batches:
                drive(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return {n: (observers[n] if n in seen else None) for n in registry}


def _peak(observer) -> float:
    lo, hi = observer._range()
    return max(abs(float(lo)), abs(float(hi)))


@pytest.fixture(scope="module")
def observed():
    from engine.model_factory import make_model

    torch.manual_seed(0)
    base = make_model(CFG).eval()
    calib = _batches()
    model, registry = insert_fake_quant(copy.deepcopy(base))
    calibrate_weights(model, registry)
    shared_names = [n for n in registry
                    if n.startswith("backbone.blocks.") and int(n.split(".")[2]) < CUT]
    return (base, calib, model, registry, sorted(shared_names),
            _observe(model, registry, calib, _search),
            _observe(model, registry, calib, _track),
            _observe(model, registry, calib, _hybrid_forward))


# --- the question is not vacuous ---------------------------------------------

def test_the_two_modalities_really_do_want_different_grids(observed):
    """If the ranges matched, sharing would cost nothing and there would be no question."""
    _, _, _, _, names, frame, event, _ = observed
    assert names, f"no shared blocks below cut_point={CUT}"
    ratios = [max(_peak(frame[n]), _peak(event[n])) / min(_peak(frame[n]), _peak(event[n]))
              for n in names]
    assert max(ratios) > 1.5, f"worst modality mismatch only {max(ratios):.2f}x"


def test_both_drives_reach_every_shared_block(observed):
    """The premise of the whole comparison: these blocks are on both paths."""
    _, _, _, _, names, frame, event, _ = observed
    assert all(frame[n] is not None and event[n] is not None for n in names)


def test_only_the_early_blocks_are_shared(observed):
    """Past the cut point the track path is gone, so those blocks see frames only."""
    _, _, _, registry, names, frame, event, _ = observed
    late = [n for n in registry
            if n.startswith("backbone.blocks.") and int(n.split(".")[2]) >= CUT]
    assert late, "fixture must have blocks past the cut point"
    assert all(event[n] is None for n in late)
    assert all(frame[n] is not None for n in late)


# --- how the conflict is resolved --------------------------------------------

def test_the_shared_grid_is_the_wider_of_the_two(observed):
    """A max-abs observer takes the union of the ranges — exactly, not approximately."""
    _, _, _, _, names, frame, event, both = observed
    for name in names:
        assert _peak(both[name]) == pytest.approx(
            max(_peak(frame[name]), _peak(event[name])), rel=1e-6), name


def test_sharing_costs_resolution_and_the_amount_is_bounded(observed):
    """The cost, in bits, stated as a number rather than as a worry."""
    _, _, _, _, names, frame, event, both = observed
    lost = [_peak(both[n]) / min(_peak(frame[n]), _peak(event[n])) for n in names]
    bits = max(float(torch.log2(torch.tensor(v))) for v in lost)
    assert 0.0 < bits < 2.0, f"the narrower modality gives up {bits:.2f} bits"


def test_the_wider_grid_cannot_clip_more_than_either_narrow_one(observed):
    """The other half of the trade, and the reason the net sign is not obvious.

    A per-modality grid is fitted to the calibration set; anything the probe data does
    beyond it is clipped. The shared grid is the union, so it clips a subset of what
    either narrow grid clips — never more. That is why narrowing to the 'right' modality
    is not automatically an improvement.
    """
    _, calib, model, registry, names, frame, event, both = observed
    probes = _batches(3, seed=101)
    counts: dict[str, list[int]] = {}

    def rate(drive, scales) -> float:
        counts.clear()
        over = [0, 0]

        def hook(name):
            def fn(_module, inputs):
                if inputs:
                    x = inputs[0].detach()
                    over[0] += int((x.abs() > scales[name]).sum())
                    over[1] += x.numel()
            return fn

        handles = [registry[n].register_forward_pre_hook(hook(n)) for n in names]
        try:
            with torch.no_grad():
                for probe in probes:
                    drive(model, probe)
        finally:
            for handle in handles:
                handle.remove()
        return over[0] / max(1, over[1])

    shared_scales = {n: _peak(both[n]) for n in names}
    assert rate(_search, shared_scales) <= rate(_search, {n: _peak(frame[n]) for n in names})
    assert rate(_track, shared_scales) <= rate(_track, {n: _peak(event[n]) for n in names})


# --- and the converted hybrid actually runs both branches ---------------------

def test_both_branches_of_the_converted_hybrid_produce_a_bounded_answer(observed):
    """An accuracy number for the hybrid path, which previously had none at all.

    The bound is generous on purpose: the model is randomly initialised, so this asserts
    the two-branch conversion is sane end to end, not that the network is accurate. The
    figure that matters — shared versus per-modality scalars — is in the report, because
    its sign did not settle and a test asserting a sign would be asserting noise.
    """
    base, calib, _, _, _, _, _, _ = observed
    model = copy.deepcopy(base)
    from quantization.calibrate import post_training_quantize

    model, _ = post_training_quantize(model, calib, forward_fn=_hybrid_forward)
    model, report = convert_model_to_integer(model, calib, forward_fn=_hybrid_forward)
    model.eval()

    probe = _batches(1, seed=101)[0]
    zeros = torch.zeros(2, 5)
    with torch.no_grad():
        float_search = base.search_step(probe["frame"])["box"]
        float_track = base.track_step(probe["event"], zeros)["residual"]
        got_search = model.search_step(probe["frame"])["box"]
        got_track = model.track_step(probe["event"], zeros)["residual"]

    def rms(a, b):
        return float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt().clamp_min(1e-12))

    assert torch.isfinite(got_search).all() and torch.isfinite(got_track).all()
    assert rms(got_search, float_search) < 0.25
    assert rms(got_track, float_track) < 0.25
