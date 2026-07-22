"""Data adapter: HBTXR sample batch -> reimplemented-model I/O contract per modality."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.data.adapter import adapt_batch, resolve_modality


def _fake_batch(batch=2, height=64, width=64):
    return {
        "frame": torch.rand(batch, 1, height, width),
        "event": torch.rand(batch, 2, height, width),
        "mask_target": torch.rand(batch, 1, height, width),
        "cur_state": torch.rand(batch, 6),
        "prev_state": torch.rand(batch, 6),
        "pupil_search_target": torch.rand(batch, 7),
        "annotation_quality": torch.rand(batch),
        "eye_target": torch.rand(batch, 5),
        "sample_id": ["a", "b"],
        "meta": [{}, {}],
    }


def test_frame_contract_keys_and_shapes():
    out = adapt_batch(_fake_batch(), "frame")
    assert {"image", "box", "state", "mask", "reliability", "eye_box"} <= set(out)
    assert out["image"].shape[1] == 1        # frame is the image
    assert out["box"].shape[-1] == 5
    assert out["state"].shape[-1] == 5
    assert out["reliability"].shape[-1] == 2
    assert out["eye_box"].shape[-1] == 4


def test_event_uses_event_tensor_as_image():
    out = adapt_batch(_fake_batch(), "event")
    assert out["image"].shape[1] == 2        # 2-channel event tensor


def test_hybrid_contract_keys():
    out = adapt_batch(_fake_batch(), "hybrid")
    assert {"frame", "event", "box", "anchor_state", "residual", "reliability"} <= set(out)
    assert out["anchor_state"].shape[-1] == 5
    assert out["residual"].shape[-1] == 5


def test_mask_contract_is_minimal():
    out = adapt_batch(_fake_batch(), "mask")
    assert set(out) >= {"image", "mask"}
    assert "box" not in out


def test_reliability_target_in_unit_range():
    out = adapt_batch(_fake_batch(), "frame")
    assert float(out["reliability"].min()) >= 0.0
    assert float(out["reliability"].max()) <= 1.0


def test_resolve_modality_from_target():
    assert resolve_modality({"model": {"target": "models.hybrid.HybridModel"}}) == "hybrid"
    assert resolve_modality({"model": {"target": "models.mask.MaskModel"}}) == "mask"
