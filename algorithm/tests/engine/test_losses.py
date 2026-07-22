"""Training losses: term selection by present keys + weighted total."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.train.losses import DEFAULT_WEIGHTS, compute_losses


def test_only_present_terms_are_scored():
    outputs = {"box": torch.zeros(2, 5), "mask": torch.zeros(2, 1, 4, 4)}
    targets = {"box": torch.ones(2, 5), "mask": torch.ones(2, 1, 4, 4)}
    losses = compute_losses(outputs, targets)
    assert {"box", "mask", "total"} <= set(losses)
    assert "reliability" not in losses


def test_total_is_weighted_sum():
    outputs = {"box": torch.zeros(2, 5)}
    targets = {"box": torch.ones(2, 5)}
    losses = compute_losses(outputs, targets, {"box": 2.0})
    assert torch.allclose(losses["total"], 2.0 * losses["box"])


def test_residual_maps_to_ellipse_term():
    outputs = {"residual": torch.zeros(2, 5)}
    targets = {"residual": torch.ones(2, 5)}
    losses = compute_losses(outputs, targets)
    assert "ellipse" in losses


def test_eye_box_term_wired():
    assert "eye_box" in DEFAULT_WEIGHTS
    losses = compute_losses({"eye_box": torch.zeros(2, 4)}, {"eye_box": torch.ones(2, 4)})
    assert "eye_box" in losses
