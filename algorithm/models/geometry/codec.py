"""Box-to-state decoder g() and residual state update (paper Eq 3-4, 7).

- ``box_to_state`` = g(): decode an oriented pupil box (x, y, w, h, r) from the
  search head into an ellipse state (x, y, a, b, theta), with
  ``a = max(w, h)/2``, ``b = min(w, h)/2`` and a pi/2 orientation correction when
  ``w < h`` (the long axis is then vertical). Output is canonicalized.
- ``state_to_box`` = g^{-1}(): the inverse used to build a box regression target
  from a canonical state — (x, y, a, b, theta) -> (x, y, 2a, 2b, theta). Because
  the state is canonical (a >= b), the long axis is horizontal so w >= h and
  ``box_to_state(state_to_box(s)) == canonicalize(s)``.
- ``apply_residual``: the track path update ``s_cand = Pi(z_t + d_s_hat)`` — add
  the anchor-relative residual to the decoded anchor state and re-canonicalize.
"""
from __future__ import annotations

import math

import torch

from models.geometry.state import canonicalize, wrap_pi

PI = math.pi


def box_to_state(box: torch.Tensor) -> torch.Tensor:
    """g(): box ``[..., 5]=(x,y,w,h,r)`` -> ellipse state ``[..., 5]=(x,y,a,b,theta)``."""
    x = box[..., 0]
    y = box[..., 1]
    w = box[..., 2].abs()
    h = box[..., 3].abs()
    r = box[..., 4]
    a = torch.maximum(w, h) / 2
    b = torch.minimum(w, h) / 2
    theta = torch.where(w >= h, r, r + PI / 2)
    return canonicalize(torch.stack([x, y, a, b, wrap_pi(theta)], dim=-1))


def state_to_box(state: torch.Tensor) -> torch.Tensor:
    """g^{-1}(): canonical state ``[..., 5]=(x,y,a,b,theta)`` -> box ``[..., 5]=(x,y,w,h,r)``.

    The state is canonicalized first (``a >= b``, ``theta in [0, pi)``), so the long
    axis is horizontal and ``w = 2a >= h = 2b``, ``r = theta``. This is the exact
    inverse of ``box_to_state`` on canonical states and is used to turn a ground-truth
    ellipse state into a box regression target for the search / direct box head.
    """
    s = canonicalize(state)
    return torch.stack([s[..., 0], s[..., 1], 2.0 * s[..., 2], 2.0 * s[..., 3], s[..., 4]], dim=-1)


def apply_residual(anchor_state: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """``s_cand = Pi(z + d_s)``: additive anchor-relative update, then canonicalize."""
    return canonicalize(anchor_state + residual)
