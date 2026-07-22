from __future__ import annotations

import dataclasses
import inspect

from hybrid.models.hybrid_tracker import HBTXRTracker, HBTXRTrackerConfig


def _structure(model):
    return {name: tuple(p.shape) for name, p in model.state_dict().items()}


def test_config_mirrors_the_constructor_one_to_one():
    ctor = [n for n in inspect.signature(HBTXRTracker.__init__).parameters if n != "self"]
    cfg = [f.name for f in dataclasses.fields(HBTXRTrackerConfig)]
    assert ctor == cfg


def test_from_config_default_matches_the_kwarg_constructor():
    assert _structure(HBTXRTracker()) == _structure(HBTXRTracker.from_config(HBTXRTrackerConfig()))


def test_a_variant_threaded_through_config_matches_kwargs():
    cfg = dataclasses.replace(HBTXRTrackerConfig(), depth=4, num_heads=1, enable_mask_head=False)
    via_config = _structure(HBTXRTracker.from_config(cfg))
    via_kwargs = _structure(HBTXRTracker(depth=4, num_heads=1, enable_mask_head=False))
    assert via_config == via_kwargs
    assert via_config != _structure(HBTXRTracker())
