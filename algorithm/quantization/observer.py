"""Range observers for affine quantization calibration.

Two layers:

- ``MinMaxObserver`` — the original per-tensor **symmetric** observer returning a
  scalar ``(scale, zero_point)`` (``scale = max_abs / qmax``, ``zp = 0``). Kept as a
  simple building block and for the direct unit test.
- ``AffineObserver`` — the configurable observer driven by a ``TensorQuantSpec``:
  per-tensor/channel/group/block slots (grouping.py), symmetric or asymmetric
  zero-points, and a calibration ``method`` of minmax / percentile / mse / kl. It
  emits **tensor** ``(scale, zero_point)`` of shape ``[num_slots]`` and applies the
  requested ``scale_type`` (float / dyadic / pot) as the final step.

The KL calibrator ports the HG-PIPE ``_kl_divergence`` histogram search
(references/); MSE is a per-slot clip-ratio grid search.
"""
from __future__ import annotations

import numpy as np
import torch

from quantization.grouping import to_slots
from quantization.scheme import QuantDtype
from quantization.spec import TensorQuantSpec, apply_scale_type


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
        # symmetric: map max_abs onto the positive qmax (HG-PIPE convention), so the
        # peak maps to +qmax with the -qmin slot as headroom (no clamp at the peak).
        scale = max(self.max_abs, 1e-8) / float(self.dtype.qmax)
        return scale, 0


class AffineObserver:
    """Configurable per-slot affine observer (granularity × sym/asym × method × scale_type)."""

    def __init__(self, spec: TensorQuantSpec) -> None:
        self.spec = spec
        self.dtype = spec.dtype
        self._need_samples = spec.calibration.method in ("percentile", "mse", "kl")
        self._samples: list[torch.Tensor] = []
        self._min: torch.Tensor | None = None
        self._max: torch.Tensor | None = None

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        slots = to_slots(x.detach().float(), self.spec)  # [S, k]
        slot_min = slots.amin(dim=1)
        slot_max = slots.amax(dim=1)
        if self._min is None:
            self._min, self._max = slot_min.clone(), slot_max.clone()
        else:
            self._min = torch.minimum(self._min, slot_min)
            self._max = torch.maximum(self._max, slot_max)
        if self._need_samples:
            self._samples.append(slots.cpu())

    # --- per-method clip / range ---------------------------------------------

    def _range(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-slot real ``(lo, hi)`` used to build the affine scale/zero-point."""
        if self._min is None:
            raise ValueError("AffineObserver.qparams() called before any observe()")
        method = self.spec.calibration.method
        if method == "minmax":
            return self._min, self._max
        samples = torch.cat(self._samples, dim=1)  # [S, total_k]
        if method == "percentile":
            p = self.spec.calibration.percentile / 100.0
            hi = torch.quantile(samples, p, dim=1)
            lo = torch.quantile(samples, 1.0 - p, dim=1)
            return lo, hi
        if method == "mse":
            clip = self._mse_clip(samples)
            return -clip, clip
        if method == "kl":
            clip = self._kl_clip(samples)
            return -clip, clip
        raise ValueError(f"unknown calibration method {method!r}")

    def _mse_clip(self, samples: torch.Tensor) -> torch.Tensor:
        """Per-slot symmetric clip minimizing quantization MSE (grid search)."""
        d = self.dtype
        amax = samples.abs().amax(dim=1).clamp_min(1e-12)      # [S]
        best_clip = amax.clone()
        best_err = torch.full_like(amax, float("inf"))
        for r in torch.linspace(0.4, 1.0, self.spec.calibration.mse_grid):
            clip = (amax * float(r)).clamp_min(1e-12)
            scale = (clip / d.qmax).unsqueeze(1)               # [S,1]
            q = torch.clamp(torch.round(samples / scale), d.qmin, d.qmax)
            err = ((q * scale) - samples).pow(2).mean(dim=1)   # [S]
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_clip = torch.where(mask, clip, best_clip)
        return best_clip

    def _kl_clip(self, samples: torch.Tensor) -> torch.Tensor:
        """Per-slot symmetric clip via KL-divergence histogram search (per slot)."""
        clips = [self._kl_clip_1d(samples[s].numpy()) for s in range(samples.shape[0])]
        return torch.tensor(clips, dtype=samples.dtype)

    def _kl_clip_1d(self, arr: np.ndarray) -> float:
        cal = self.spec.calibration
        d = self.dtype
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        max_q = max(abs(d.qmin), abs(d.qmax)) if self.spec.signed else d.qmax
        percentiles = (cal.percentile, 99.5, 99.9, 100.0)
        best = None
        for pct in percentiles:
            clip = float(np.percentile(np.abs(arr), pct))
            clip = max(clip, 1e-12)
            scale = clip / float(max(max_q, 1))
            clipped = np.clip(arr, -clip, clip)
            q = np.clip(np.rint(clipped / scale), d.qmin, d.qmax)
            dequant = q * scale
            ref_hist, _ = np.histogram(clipped, bins=cal.num_bins, range=(-clip, clip))
            deq_hist, _ = np.histogram(np.clip(dequant, -clip, clip), bins=cal.num_bins, range=(-clip, clip))
            kl = _kl_divergence(ref_hist, deq_hist)
            if best is None or kl < best[0]:
                best = (kl, clip)
        assert best is not None
        return best[1]

    # --- scale / zero-point ---------------------------------------------------

    def qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-slot ``(scale [S], zero_point [S])`` tensors, in the requested scale_type."""
        lo, hi = self._range()
        d = self.dtype
        if self.spec.symmetric:
            max_abs = torch.maximum(lo.abs(), hi.abs()).clamp_min(1e-8)
            scale = max_abs / float(d.qmax)
            zero_point = torch.zeros_like(scale)
        else:
            lo = torch.minimum(lo, torch.zeros_like(lo))
            hi = torch.maximum(hi, torch.zeros_like(hi))
            scale = ((hi - lo) / float(d.qmax - d.qmin)).clamp_min(1e-8)
            zero_point = torch.round(d.qmin - lo / scale).clamp(d.qmin, d.qmax)
        scale = apply_scale_type(scale, self.spec.scale_type)
        return scale, zero_point


def _kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> float:
    eps = 1e-12
    ref = reference.astype(np.float64) + eps
    cand = candidate.astype(np.float64) + eps
    ref /= np.sum(ref)
    cand /= np.sum(cand)
    return float(np.sum(ref * np.log(ref / cand)))


def build_observer(spec: TensorQuantSpec) -> AffineObserver:
    """Construct the configurable observer for ``spec``."""
    return AffineObserver(spec)


__all__ = ["MinMaxObserver", "AffineObserver", "build_observer"]
