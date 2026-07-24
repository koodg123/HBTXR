"""Integer inference kernels for the HBTXR quantized ViT (HW-equivalence check).

The affine fake-quant path used in training dequantizes every quantized value to
float. The deployment path instead keeps values in the integer domain: quantize
activation and weight to integers, do an integer matmul (int64 accumulator), and
requant once by ``scale_x * scale_w``. For a single ``QuantLinear`` the two paths
are mathematically identical, so ``verify_int_linear`` checks the integer kernel is
bit-exact (up to float accumulation order) with the fake-quant module — the same
kind of golden check the HG-PIPE reference runs on its artifact graph.

The full pure-integer ViT graph (chained requant between layers + integer LayerNorm
/ Softmax / GeLU via int_ops LUTs) is the larger HW-friendly deliverable and needs
the nonlinear LUT tables calibrated on HBTXR activations; this module provides the
integer linear kernel and the per-layer equivalence check it builds on.
"""
from __future__ import annotations

import torch

from quantization.insert import QuantLinear
from quantization.scheme import QuantDtype


def quantize_to_int(x: torch.Tensor, scale: float, dtype: QuantDtype) -> torch.Tensor:
    """x -> clamp(round(x / scale), qmin, qmax) as an integer tensor (int64)."""
    q = torch.round(x / scale).clamp(dtype.qmin, dtype.qmax)
    return q.to(torch.int64)


def int_linear_forward(quant_linear: QuantLinear, x: torch.Tensor) -> torch.Tensor:
    """Pure-integer ``QuantLinear`` forward, dequantized to float at the output.

    x_int @ w_intᵀ in int64, then requant by ``scale_x * scale_w`` (+ float bias).
    Matches the fake-quant path for the same calibrated scales.
    """
    act_fq = quant_linear.act_fq
    weight_fq = quant_linear.weight_fq
    scale_x = float(act_fq.scale)
    scale_w = float(weight_fq.scale)

    x_int = quantize_to_int(x, scale_x, act_fq.dtype)
    w_int = quantize_to_int(quant_linear.linear.weight, scale_w, weight_fq.dtype)
    acc = x_int @ w_int.transpose(-2, -1)  # int64 accumulate
    y = acc.to(x.dtype) * (scale_x * scale_w)
    if quant_linear.linear.bias is not None:
        y = y + quant_linear.linear.bias
    return y


def verify_int_linear(quant_linear: QuantLinear, x: torch.Tensor, *, atol: float = 1e-4) -> tuple[bool, float]:
    """Return (matches, max_abs_diff) between the integer and fake-quant paths."""
    with torch.no_grad():
        y_int = int_linear_forward(quant_linear, x)
        y_fq = quant_linear(x)
        max_diff = float((y_int - y_fq).abs().max())
        matches = bool(torch.allclose(y_int, y_fq, atol=atol))
    return matches, max_diff


__all__ = ["quantize_to_int", "int_linear_forward", "verify_int_linear"]
