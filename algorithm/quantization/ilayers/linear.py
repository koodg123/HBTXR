"""I-tier integer linear kernel + fake-quant equivalence check.

Deployment-domain forward of a ``QLinear``: quantize activation and weight to
integers, integer matmul (int64 accumulate for now; int32 accumulation lands with
the full graph in Part C), requant by ``scale_x * scale_w``. ``verify_int_linear``
checks it is bit-exact with the fake-quant path (the HG-PIPE-style golden check).
"""
from __future__ import annotations

import torch

from quantization.qlayers.linear import QLinear
from quantization.scheme import QuantDtype


def quantize_to_int(x: torch.Tensor, scale: float, dtype: QuantDtype) -> torch.Tensor:
    """x -> clamp(round(x / scale), qmin, qmax) as an integer tensor (int64)."""
    q = torch.round(x / scale).clamp(dtype.qmin, dtype.qmax)
    return q.to(torch.int64)


def int_linear_forward(quant_linear: QLinear, x: torch.Tensor) -> torch.Tensor:
    """Pure-integer ``QLinear`` forward, dequantized to float at the output."""
    act_fq = quant_linear.act_fq
    weight_fq = quant_linear.weight_fq
    scale_x = float(act_fq.scale)
    scale_w = float(weight_fq.scale)

    x_int = quantize_to_int(x, scale_x, act_fq.dtype)
    w_int = quantize_to_int(quant_linear.linear.weight, scale_w, weight_fq.dtype)
    acc = x_int @ w_int.transpose(-2, -1)  # integer accumulate
    y = acc.to(x.dtype) * (scale_x * scale_w)
    if quant_linear.linear.bias is not None:
        y = y + quant_linear.linear.bias
    return y


def verify_int_linear(quant_linear: QLinear, x: torch.Tensor, *, atol: float = 1e-4) -> tuple[bool, float]:
    """Return (matches, max_abs_diff) between the integer and fake-quant paths."""
    with torch.no_grad():
        y_int = int_linear_forward(quant_linear, x)
        y_fq = quant_linear(x)
        max_diff = float((y_int - y_fq).abs().max())
        matches = bool(torch.allclose(y_int, y_fq, atol=atol))
    return matches, max_diff


__all__ = ["quantize_to_int", "int_linear_forward", "verify_int_linear"]
