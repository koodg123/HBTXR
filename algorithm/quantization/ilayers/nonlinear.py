"""I-tier integer LUT nonlinears (deployment domain, QTensor in/out).

``IGeLU`` is the integer GeLU: it reuses a calibrated ``QGeLU``'s ``(b, s, bound)`` +
table + input/output scales and does the exact ``i_ops.table_quantize`` lookup
(``table[clamp((x + b) >> s, 0, bound)]``) in the integer domain, so it is bit-exact
against that golden when the input QTensor is already at the LUT input scale (an
explicit rescale bridges any scale mismatch coming from upstream).

``ILayerNorm`` / ``ISoftmax`` (fully-integer mean/var/rsqrt and exp/reciprocal, using
the ``i_ops.layernorm_quantize`` / ``softmax_quantize`` goldens with HG-PIPE
calibrate_rsqrt/softmax scalars) land in the next phase; they implement a different,
fully-integer algorithm than the Q-tier float-reduction LUTs.
"""
from __future__ import annotations

import torch
from torch import nn

from quantization.ilayers.qtensor import QTensor
from quantization.scheme import INT8, QuantDtype


def _scalar(v) -> float:
    return float(v.reshape(-1)[0]) if torch.is_tensor(v) else float(v)


class IGeLU(nn.Module):
    """Integer LUT GeLU: QTensor -> QTensor via the table_quantize golden lookup."""

    def __init__(self, scalars, table, *, input_scale, output_scale, out_dtype: QuantDtype = INT8):
        super().__init__()
        self.b, self.s, self.bound = (int(v) for v in scalars)
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.out_dtype = out_dtype
        self.register_buffer("table", torch.as_tensor(list(table), dtype=torch.int64))

    @classmethod
    def from_qgelu(cls, qgelu) -> "IGeLU":
        return cls([qgelu.b, qgelu.s, qgelu.bound], qgelu.table.tolist(),
                   input_scale=qgelu.input_scale, output_scale=qgelu.output_scale)

    def forward(self, qt: QTensor) -> QTensor:
        # bring the input integer into the LUT's input scale (identity when equal)
        ratio = _scalar(qt.scale) / self.input_scale
        x_lut = torch.round(qt.int_data.to(torch.float32) * ratio).to(torch.int64)
        cursor = torch.bitwise_right_shift(x_lut + self.b, self.s).clamp(0, self.bound)
        out_int = self.table[cursor]
        return QTensor(out_int.to(torch.int32), scale=self.output_scale, zero_point=0.0, dtype=self.out_dtype)


__all__ = ["IGeLU"]
