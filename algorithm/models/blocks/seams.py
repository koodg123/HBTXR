"""Seam modules: the arithmetic a conversion pass has to be able to reach.

A quantization pass converts a model by walking ``named_modules()`` and swapping
children. That reaches ``nn.Linear``, ``nn.LayerNorm``, ``nn.GELU`` — anything that is
an attribute. It cannot reach an operator written inline in a ``forward`` body, because
there is no object to replace. In this ViT that left the attention matmuls, the
``1/√d`` scaling and both residual adds permanently in float, and with them the
``IMatMul`` / ``IAdd`` kernels that exist to run them.

These three modules are the seam. Each is an exact identity of the operator it stands
in for and owns **no parameters and no buffers**, so promoting an inline operator to one
leaves the float forward bit-identical and the ``state_dict`` unchanged — an existing
checkpoint still loads with ``strict=True``. What it buys is a named node the converter
can find, and a loud failure (a missing attribute) instead of a silent one (an operator
nobody noticed was still float) if the seam is ever removed.

The precedent is already in this package: ``MultiHeadAttention.attn_softmax`` was
``F.softmax`` until it had to become quant-swappable.
"""
from __future__ import annotations

import torch
from torch import nn


class MatMul(nn.Module):
    """``a @ b`` — an activation×activation product (attention scores and context)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b


class Add(nn.Module):
    """``a + b`` — a residual join, where two differently-scaled tensors meet."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class Scale(nn.Module):
    """``x * factor`` for a constant factor, held as a plain float, never a buffer.

    ``factor`` stays out of the ``state_dict`` on purpose: it is a property of the
    architecture (``head_dim ** -0.5``), not something training learns, and adding a
    buffer would change every checkpoint. In the integer graph the multiply folds into
    the requantization that follows — for the shipped ``head_dim=64`` the factor is
    exactly ``0.125 = 2⁻³``, i.e. a pure arithmetic right shift.
    """

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = float(factor)

    def extra_repr(self) -> str:
        return f"factor={self.factor}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


__all__ = ["MatMul", "Add", "Scale"]
