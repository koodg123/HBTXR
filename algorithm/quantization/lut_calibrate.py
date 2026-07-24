"""LUT calibration for HW-friendly integer nonlinear operators (HG-PIPE style).

Builds power-of-two (PoT) index lookup tables for the pointwise nonlinear ops of
the HBTXR ViT, adapted from the HG-PIPE reference ``lut_calibration.py`` (now under
references/hardware/hg-pipe-quantization). A table is addressed by
``cursor = clamp((x_int + offset) >> shift, 0, bound)`` — the same kernel as
``LUTFakeQuantizer`` / ``int_ops.table_quantize`` — so a calibrated table plugs
straight into the fake-quant and integer paths.

This module provides the GeLU table builder (pointwise, the cleanest HW-friendly
nonlinear). The row-wise Softmax and LayerNorm-rsqrt builders are heavier (they need
per-row reductions) and are follow-on work; the reference builders remain available
under references/ for porting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantization.scheme import INT8, QuantDtype


@dataclass(frozen=True)
class PotIndexParams:
    """Power-of-two table index: cursor = clamp((x + offset) >> shift, 0, bound)."""

    offset: int
    shift: int
    bound: int

    @property
    def scalars(self) -> list[int]:
        return [self.offset, self.shift, self.bound]


def make_pot_index_params(input_min: int, input_max: int, *, entries: int) -> PotIndexParams:
    """Choose (offset, shift, bound) covering [input_min, input_max] over ``entries``."""
    if entries < 2:
        raise ValueError("entries must be at least 2")
    if input_max <= input_min:
        input_max = input_min + 1
    bound = entries - 1
    shift = max(0, int(math.ceil(math.log2(max((input_max - input_min) / float(bound), 1.0)))))
    return PotIndexParams(offset=-int(input_min), shift=shift, bound=bound)


def coordinates_for(params: PotIndexParams) -> np.ndarray:
    """The representative integer input value at each table entry."""
    indices = np.arange(params.bound + 1, dtype=np.int64)
    return (indices << params.shift) - params.offset


def cursor_for(x_int: np.ndarray, params: PotIndexParams) -> np.ndarray:
    x = np.asarray(x_int, dtype=np.int64)
    return np.clip((x + params.offset) >> params.shift, 0, params.bound).astype(np.int64)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """The tanh GeLU approximation (nn.GELU default matches erf; tanh is HW-friendly)."""
    xf = np.asarray(x, dtype=np.float64)
    return 0.5 * xf * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (xf + 0.044715 * np.power(xf, 3))))


def _symmetric_scale(max_abs: float, dtype: QuantDtype) -> float:
    return max(float(max_abs), 1e-8) / float(dtype.qmax)


def build_gelu_lut(
    activations,
    *,
    entries: int = 256,
    input_dtype: QuantDtype = INT8,
    output_dtype: QuantDtype = INT8,
) -> dict[str, object]:
    """Calibrate a pointwise GeLU LUT from observed GeLU-input activations.

    Returns ``{scalars, table, input_scale, output_scale}`` where integer GeLU is
    ``y = table[clamp((round(x/input_scale) + offset) >> shift, 0, bound)] * output_scale``.
    """
    x = np.asarray(activations, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("calibration activations must not be empty")

    input_scale = _symmetric_scale(float(np.abs(x).max()), input_dtype)
    x_int = np.rint(x / input_scale).astype(np.int64)
    params = make_pot_index_params(int(x_int.min()), int(x_int.max()), entries=entries)

    coords_real = coordinates_for(params).astype(np.float64) * input_scale
    gelu_real = gelu_tanh(coords_real)
    output_scale = _symmetric_scale(float(np.abs(gelu_real).max()), output_dtype)
    table = np.clip(np.rint(gelu_real / output_scale), output_dtype.qmin, output_dtype.qmax).astype(np.int64)

    return {
        "scalars": params.scalars,
        "table": table.tolist(),
        "input_scale": float(input_scale),
        "output_scale": float(output_scale),
    }


__all__ = [
    "PotIndexParams",
    "make_pot_index_params",
    "coordinates_for",
    "cursor_for",
    "gelu_tanh",
    "build_gelu_lut",
]
