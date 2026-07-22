"""Model factory + forward smoke for the reimplemented models (needs torch)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.model_factory import list_model_names, make_model


def _cfg(target, **extra):
    cfg = {
        "target": target,
        "embed_dim": 48,
        "patch_size": 16,
        "backbone": {"depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "cut_point": 1},
    }
    cfg.update(extra)
    return cfg


def test_list_model_names_covers_modalities():
    assert {"frame", "event", "hybrid", "mask"} <= set(list_model_names())


def test_frame_forward_smoke():
    model = make_model(_cfg("models.frame.FrameModel"))
    out = model(torch.rand(2, 1, 64, 64))
    assert out["box"].shape == (2, 5)
    assert out["state"].shape == (2, 5)


def test_event_forward_smoke():
    model = make_model(_cfg("models.event.EventModel"))
    out = model(torch.rand(2, 2, 64, 64))
    assert out["box"].shape == (2, 5)


def test_hybrid_search_track_smoke():
    model = make_model(_cfg("models.hybrid.HybridModel"))
    search = model.search_step(torch.rand(2, 1, 64, 64))
    assert search["box"].shape == (2, 5)
    track = model.track_step(torch.rand(2, 2, 64, 64), search["state"])
    assert track["residual"].shape == (2, 5)
    assert track["state"].shape == (2, 5)


def test_mask_model_builds():
    assert make_model({"target": "models.mask.MaskModel"}) is not None


def test_embed_dim_is_authoritative_without_backbone_width():
    # regression: setting embed_dim (but not backbone.embed_dim) must not mismatch
    model = make_model(_cfg("models.frame.FrameModel", embed_dim=64))
    out = model(torch.rand(1, 1, 64, 64))
    assert out["box"].shape == (1, 5)
