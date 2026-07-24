"""Granularity slotting: reshape a tensor into per-scale "slots" and back.

A quantizer with a given granularity owns one ``(scale, zero_point)`` pair per
*slot*. ``to_slots`` reshapes a tensor ``x`` to ``[num_slots, k]`` so that row ``i``
holds exactly the elements sharing slot ``i``'s scale; ``from_slots`` is the exact
inverse. Both the observer (which reduces each row to a scale) and the fake
quantizer (which divides each row by its scale) use these so their slot layout can
never diverge.

Granularities (see ``spec.TensorQuantSpec``):

- ``per-tensor``   : 1 slot (whole tensor).
- ``per-channel``  : 1 slot per index along ``ch_axis`` (weights: 0 = out-features;
  activations: -1 = feature). All other axes are reduced.
- ``per-group``    : 2D weight ``[out, in]``; the ``in`` axis is split into groups of
  ``group_size`` → ``out * (in // group_size)`` slots of ``group_size`` elements.
- ``per-block``    : 2D weight ``[out, in]``; tiled into ``block_size = (bo, bi)``
  blocks → ``(out // bo) * (in // bi)`` slots of ``bo * bi`` elements.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # avoid an import cycle at runtime (spec imports nothing from here)
    from quantization.spec import TensorQuantSpec


def _norm_axis(axis: int, ndim: int) -> int:
    return axis % ndim


def num_slots(x_shape: tuple[int, ...], spec: "TensorQuantSpec") -> int:
    """Number of independent scale slots for ``x_shape`` under ``spec``."""
    g = spec.granularity
    if g == "per-tensor":
        return 1
    if g == "per-channel":
        return int(x_shape[_norm_axis(spec.ch_axis, len(x_shape))])
    if g == "per-group":
        out, inf = int(x_shape[-2]), int(x_shape[-1])
        group = int(spec.group_size or inf)
        if group <= 0 or inf % group != 0:
            raise ValueError(f"group_size={spec.group_size} must divide in-features {inf}")
        return out * (inf // group)
    if g == "per-block":
        out, inf = int(x_shape[-2]), int(x_shape[-1])
        bo, bi = spec.block_size or (out, inf)
        if bo <= 0 or bi <= 0 or out % bo != 0 or inf % bi != 0:
            raise ValueError(f"block_size={spec.block_size} must tile [{out}, {inf}]")
        return (out // bo) * (inf // bi)
    raise ValueError(f"unknown granularity {g!r}")


def to_slots(x: torch.Tensor, spec: "TensorQuantSpec") -> torch.Tensor:
    """Reshape ``x`` to ``[num_slots, k]`` (row i = elements sharing slot i's scale)."""
    g = spec.granularity
    if g == "per-tensor":
        return x.reshape(1, -1)
    if g == "per-channel":
        ax = _norm_axis(spec.ch_axis, x.ndim)
        return x.movedim(ax, 0).reshape(x.shape[ax], -1)
    if g == "per-group":
        out, inf = x.shape[-2], x.shape[-1]
        group = int(spec.group_size or inf)
        return x.reshape(out * (inf // group), group)
    if g == "per-block":
        out, inf = x.shape[-2], x.shape[-1]
        bo, bi = spec.block_size or (out, inf)
        tiled = x.reshape(out // bo, bo, inf // bi, bi).permute(0, 2, 1, 3)
        return tiled.reshape((out // bo) * (inf // bi), bo * bi)
    raise ValueError(f"unknown granularity {g!r}")


def from_slots(slots: torch.Tensor, x_shape: tuple[int, ...], spec: "TensorQuantSpec") -> torch.Tensor:
    """Inverse of :func:`to_slots` — rebuild the original ``x_shape`` layout."""
    g = spec.granularity
    if g in ("per-tensor", "per-group"):
        return slots.reshape(x_shape)
    if g == "per-channel":
        ax = _norm_axis(spec.ch_axis, len(x_shape))
        moved = [x_shape[ax]] + [x_shape[i] for i in range(len(x_shape)) if i != ax]
        return slots.reshape(moved).movedim(0, ax)
    if g == "per-block":
        out, inf = int(x_shape[-2]), int(x_shape[-1])
        bo, bi = spec.block_size or (out, inf)
        untiled = slots.reshape(out // bo, inf // bi, bo, bi).permute(0, 2, 1, 3)
        return untiled.reshape(x_shape)
    raise ValueError(f"unknown granularity {g!r}")


__all__ = ["num_slots", "to_slots", "from_slots"]
