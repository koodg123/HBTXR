"""Range observers for affine quantization calibration.

A ``MinMaxObserver`` accumulates the observed dynamic range of a tensor (weights,
statically; activations, over calibration batches) and derives a symmetric affine
scale ``scale = max_abs / qmax`` with ``zero_point = 0``. Symmetric max-abs is the
standard PTQ baseline; percentile/KL refinements can be layered on later (the
HG-PIPE reference dyadic-KL calibrator lives in references/).
"""
from __future__ import annotations

import torch

from quantization.scheme import QuantDtype


class MinMaxObserver:
    """Track running max-abs of observed tensors; emit a symmetric affine scale."""

    def __init__(self, dtype: QuantDtype) -> None:
        self.dtype = dtype
        self.max_abs = 0.0

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        value = float(x.detach().abs().max())
        if value > self.max_abs:
            self.max_abs = value

    def qparams(self) -> tuple[float, int]:
        qmax = max(abs(self.dtype.qmin), self.dtype.qmax)
        scale = max(self.max_abs, 1e-8) / float(qmax)
        return scale, 0


__all__ = ["MinMaxObserver"]
