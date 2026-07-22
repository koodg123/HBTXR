"""Geometry codec: box<->state round-trips + canonicalization (paper Eq 1-4, 7)."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from models.geometry import STATE_DIM, apply_residual, box_to_state, canonicalize, state_to_box


def test_state_dim_is_five():
    assert STATE_DIM == 5


def test_state_to_box_roundtrip():
    state = torch.tensor([[0.5, 0.4, 0.3, 0.1, 0.7]])   # canonical: a=0.3 >= b=0.1
    box = state_to_box(state)
    assert torch.allclose(box, torch.tensor([[0.5, 0.4, 0.6, 0.2, 0.7]]), atol=1e-5)
    assert torch.allclose(box_to_state(box), canonicalize(state), atol=1e-5)


def test_canonicalize_enforces_a_ge_b_and_theta_range():
    state = torch.tensor([[0.0, 0.0, 0.1, 0.3, 0.2]])   # a < b -> must swap
    c = canonicalize(state)
    assert float(c[0, 2]) >= float(c[0, 3])
    assert 0.0 <= float(c[0, 4]) < math.pi


def test_apply_residual_recovers_target():
    anchor = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.4]])
    target = torch.tensor([[0.6, 0.55, 0.32, 0.18, 0.5]])
    out = apply_residual(anchor, target - anchor)
    assert torch.allclose(out, canonicalize(target), atol=1e-5)
