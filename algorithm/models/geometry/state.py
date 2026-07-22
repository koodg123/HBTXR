"""Pupil runtime state and canonicalization (paper Eq 1).

The canonical runtime state is ``s = (x, y, a, b, theta)``: center (x, y),
ellipse semi-axes with the convention ``a >= b``, and orientation
``theta in [0, pi)``. STATE_DIM = 5.

``canonicalize`` implements the re-canonicalizer ``Pi(.)``: positive semi-axes,
``a >= b`` (swap the axes and rotate by pi/2 if needed), and theta wrapped to
``[0, pi)``. Applied after every additive residual update so the state stays
well-formed.
"""
from __future__ import annotations

import math

import torch

STATE_DIM = 5
PI = math.pi


def wrap_pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap an angle to ``[0, pi)``."""
    return torch.remainder(theta, PI)


def canonicalize(state: torch.Tensor) -> torch.Tensor:
    """``Pi(.)``: enforce positive semi-axes, ``a >= b``, ``theta in [0, pi)``.

    ``state`` has shape ``[..., 5]`` = (x, y, a, b, theta).
    """
    x = state[..., 0]
    y = state[..., 1]
    a = state[..., 2].abs()
    b = state[..., 3].abs()
    theta = state[..., 4]
    swap = b > a
    a_new = torch.where(swap, b, a)
    b_new = torch.where(swap, a, b)
    theta = torch.where(swap, theta + PI / 2, theta)
    return torch.stack([x, y, a_new, b_new, wrap_pi(theta)], dim=-1)
