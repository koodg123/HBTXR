"""I-tier integer linear kernel + fake-quant equivalence check.

Deployment-domain forward of a ``QLinear``: quantize activation and weight to
integers, integer matmul (int64 accumulate for now; int32 accumulation lands with
the full graph in Part C), requant by ``scale_x * scale_w``. ``verify_int_linear``
checks it is bit-exact with the fake-quant path (the HG-PIPE-style golden check).

``ILinear`` is the deployable module form: it holds int8 weights + per-out-channel
weight scales and reproduces ``QLinear`` bit-exactly, with activation zero-point
folded as a ``- zp * Σw`` correction so asymmetric activations stay pure-integer.
"""
from __future__ import annotations

import torch
from torch import nn

from quantization.grouping import from_slots, to_slots
from quantization.ilayers.int_functional import int_matmul
from quantization.ilayers.qtensor import QTensor
from quantization.qlayers.linear import QLinear
from quantization.scheme import INT8, QuantDtype


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


def _quantize_weight(quant_linear: QLinear):
    """Integer weight + per-out-channel scale from a calibrated ``QLinear``.

    Supports per-tensor and per-channel(out-axis) symmetric weight scales — the HW
    deployable granularities. per-group would need per-group partial-sum requant and
    is intentionally rejected here (still available at the Q tier for analysis).
    """
    fq = quant_linear.weight_fq
    spec = fq.spec
    if spec.granularity not in ("per-tensor", "per-channel") or (
        spec.granularity == "per-channel" and spec.ch_axis not in (0, -2)
    ):
        raise NotImplementedError(
            f"ILinear supports per-tensor / per-channel(out) weight, got {spec.granularity} axis={spec.ch_axis}")
    if bool((fq.zero_point != 0).any()):
        raise NotImplementedError("ILinear assumes symmetric weights (zero_point=0)")
    weight = quant_linear.linear.weight
    slots = to_slots(weight, spec)
    scale = fq.scale.view(-1, 1)
    q = torch.round(slots / scale + fq.zero_point.view(-1, 1)).clamp(fq.qmin, fq.qmax)
    weight_int = from_slots(q, weight.shape, spec).to(torch.int32)
    # per-out-channel scale [out] (per-tensor collapses to a single value broadcast)
    weight_scale = fq.scale.reshape(-1)
    return weight_int, weight_scale


class ILinear(nn.Module):
    """Deployable integer Linear: int8 weight × int8 activation, dequant at the output.

    Reproduces the source ``QLinear`` bit-exactly (activation zero-point folded as a
    ``- zp·Σw`` correction). ``forward`` returns float; ``forward_accumulator`` exposes
    the raw int32 accumulator as a ``QTensor`` (scale ``s_x·s_w``) for the integer graph.
    """

    def __init__(self, weight_int, weight_scale, act_scale, act_dtype, bias, *, act_zero_point=0.0):
        super().__init__()
        self.register_buffer("weight_int", weight_int.to(torch.int32))            # [out, in]
        self.register_buffer("weight_scale", torch.as_tensor(weight_scale, dtype=torch.float32).reshape(-1))
        self.register_buffer("wsum", weight_int.to(torch.int64).sum(dim=1))       # [out]
        self.act_scale = float(act_scale)
        self.act_zero_point = int(round(float(act_zero_point)))
        self.act_dtype = act_dtype
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().float())
        else:
            self.bias = None

    @classmethod
    def from_qlinear(cls, quant_linear: QLinear) -> "ILinear":
        act_fq = quant_linear.act_fq
        if act_fq.spec.granularity != "per-tensor":
            raise NotImplementedError("ILinear supports per-tensor activation scale (static PTQ)")
        weight_int, weight_scale = _quantize_weight(quant_linear)
        return cls(weight_int, weight_scale, float(act_fq.scale.reshape(-1)[0]),
                   act_fq.dtype, quant_linear.linear.bias, act_zero_point=float(act_fq.zero_point.reshape(-1)[0]))

    def _accumulate(self, x: torch.Tensor) -> torch.Tensor:
        x_int = torch.round(x / self.act_scale + self.act_zero_point).clamp(
            self.act_dtype.qmin, self.act_dtype.qmax).to(torch.int64)
        acc = int_matmul(x_int, self.weight_int.transpose(0, 1))                  # [.., out]
        if self.act_zero_point != 0:
            acc = acc - self.act_zero_point * self.wsum
        return acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = self._accumulate(x)
        y = acc.to(torch.float32) * (self.act_scale * self.weight_scale)
        if self.bias is not None:
            y = y + self.bias
        return y

    def forward_accumulator(self, x: torch.Tensor) -> QTensor:
        """Raw int32 accumulator (bias-free) with scale ``s_x·s_w`` for the integer graph."""
        acc = self._accumulate(x)
        return QTensor(acc.to(torch.int32), scale=(self.act_scale * self.weight_scale), zero_point=0.0,
                       dtype=QuantDtype(32, signed=True))


__all__ = ["quantize_to_int", "int_linear_forward", "verify_int_linear", "ILinear"]
