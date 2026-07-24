"""QTensor — an integer tensor carrying its affine quantization metadata.

The I-tier integer graph passes ``QTensor`` between ops so scales propagate across
add / concat / passthrough (reshape, transpose). ``int_data`` holds integer-valued
data at its real bit width (int8 / uint8; int32 for un-requantized accumulators);
the real value it represents is ``(int_data - zero_point) * scale``.

- ``quantize(x, scale, zero_point, dtype)``  float  -> QTensor (clamp to dtype range)
- ``dequantize()``                           QTensor -> float
- ``reshape`` / ``transpose`` / ``permute`` / ``flatten`` — passthrough: move
  ``int_data``, keep the (scale, zero_point, dtype) metadata.

``scale`` / ``zero_point`` may be scalars or per-channel tensors (broadcastable to
``int_data``); the integer kernels in ``int_functional`` interpret them the same way
the fake-quant path does.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from quantization.scheme import INT8, QuantDtype


def _as_tensor(v: Any, ref: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=torch.float32)
    return t.to(ref.device)


@dataclass
class QTensor:
    int_data: torch.Tensor          # integer-valued (int8/uint8/int32)
    scale: torch.Tensor | float     # per-tensor float or per-channel tensor
    zero_point: torch.Tensor | float = 0.0
    dtype: QuantDtype = INT8

    # --- conversions ---------------------------------------------------------

    @classmethod
    def quantize(cls, x: torch.Tensor, scale, zero_point=0.0, dtype: QuantDtype = INT8) -> "QTensor":
        s = _as_tensor(scale, x)
        zp = _as_tensor(zero_point, x)
        q = torch.round(x / s + zp).clamp(dtype.qmin, dtype.qmax)
        return cls(q.to(torch.int32), scale, zero_point, dtype)

    def dequantize(self) -> torch.Tensor:
        s = _as_tensor(self.scale, self.int_data)
        zp = _as_tensor(self.zero_point, self.int_data)
        return (self.int_data.to(torch.float32) - zp) * s

    # --- passthrough (data movement keeps the same scale) --------------------

    def _with(self, int_data: torch.Tensor) -> "QTensor":
        return replace(self, int_data=int_data)

    def reshape(self, *shape) -> "QTensor":
        return self._with(self.int_data.reshape(*shape))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "QTensor":
        return self._with(self.int_data.flatten(start_dim, end_dim))

    def transpose(self, dim0: int, dim1: int) -> "QTensor":
        return self._with(self.int_data.transpose(dim0, dim1))

    def permute(self, *dims) -> "QTensor":
        return self._with(self.int_data.permute(*dims))

    @property
    def shape(self) -> torch.Size:
        return self.int_data.shape

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"QTensor(shape={tuple(self.int_data.shape)}, dtype={self.dtype}, scale={self.scale})"


__all__ = ["QTensor"]
