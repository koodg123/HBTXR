from __future__ import annotations

import pytest

from src.loss.losses import build_loss, compute_loss, list_loss_names

EXPECTED_ALIASES = ["hybrid", "search", "stage1", "stage1_search", "stage2", "stage2_hybrid"]


def test_registry_exposes_six_sorted_aliases() -> None:
    assert list_loss_names() == EXPECTED_ALIASES
    assert len(EXPECTED_ALIASES) == 6


def test_aliases_collapse_to_two_stage_implementations() -> None:
    stage1 = {build_loss(name) for name in ("stage1", "stage1_search", "search")}
    stage2 = {build_loss(name) for name in ("stage2", "stage2_hybrid", "hybrid")}
    assert len(stage1) == 1
    assert len(stage2) == 1
    assert stage1.isdisjoint(stage2)


def test_name_normalization_strips_case_and_maps_dash_to_underscore() -> None:
    assert build_loss("  STAGE1  ") is build_loss("stage1")
    assert build_loss("STAGE2-HYBRID") is build_loss("stage2_hybrid")


def test_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        build_loss("does_not_exist")


def test_compute_loss_delegates_with_active_head(monkeypatch) -> None:
    from src.loss import losses as losses_module

    captured = {}

    def _spy(batch, outputs, loss_cfg, *, active_head="all"):
        captured["args"] = (batch, outputs, loss_cfg, active_head)
        return {"total": "sentinel"}

    monkeypatch.setitem(losses_module._LOSSES, "stage1", _spy)
    result = compute_loss("stage1", {"b": 1}, {"o": 2}, {"c": 3}, active_head="center")
    assert result == {"total": "sentinel"}
    assert captured["args"] == ({"b": 1}, {"o": 2}, {"c": 3}, "center")
