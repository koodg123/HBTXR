"""PyTorch FakeQuantizer modules for HG-PIPE quantization contracts."""

from __future__ import annotations

from typing import Iterable


def _torch():
    import torch

    return torch


class AffineFakeQuantizer(_torch().nn.Module):
    """Affine fake quantization with dequantized floating output."""

    def __init__(self, *, scale: float, zero_point: int, qmin: int, qmax: int):
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)
        self.zero_point = int(zero_point)
        self.qmin = int(qmin)
        self.qmax = int(qmax)

    def forward(self, x):
        torch = _torch()
        q = torch.round(x / self.scale + self.zero_point).clamp(self.qmin, self.qmax)
        return (q - self.zero_point) * self.scale


class HGTableFakeQuantizer(_torch().nn.Module):
    """HG-PIPE LUT fake quantizer.

    The input is interpreted in the integer domain after rounding.  The output
    is returned as floating point so it can remain in a PyTorch fake-quant graph.
    """

    def __init__(self, *, scalars: Iterable[int], table: Iterable[int], output_dtype=None):
        super().__init__()
        torch = _torch()
        scalar_values = [int(value) for value in scalars]
        if len(scalar_values) != 3:
            raise ValueError(f"HGTableFakeQuantizer expects 3 scalars, got {len(scalar_values)}")
        self.b, self.s, self.bound = scalar_values
        table_tensor = torch.as_tensor(list(table), dtype=torch.int64)
        self.register_buffer("table", table_tensor)
        self.output_dtype = output_dtype

    def forward(self, x):
        torch = _torch()
        x_int = torch.round(x).to(torch.int64)
        cursor = ((x_int + self.b) >> self.s).clamp(0, self.bound)
        return self.table[cursor].to(dtype=x.dtype)

