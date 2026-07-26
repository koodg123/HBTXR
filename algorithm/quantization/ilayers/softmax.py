"""I-tier fully-integer Softmax (HG-PIPE segmented-reciprocal form).

The Q-tier ``QSoftmax`` still does the row max and the row sum in float and only looks
exp / reciprocal up from tables — hardware cannot. ``ISoftmax`` is the deployment op:
*every* step is integer. Integer row max, inverse-exp PoT table lookup, integer
accumulation, then a **segmented** reciprocal — the accumulator's own PoT cursor picks
one of two reciprocal tables, each with its own final requant ``(b3, s3)`` — an integer
multiply, an arithmetic right shift and an unsigned clamp.

It is the vectorized twin of ``i_ops.softmax_quantize`` and is bit-exact against it:
the per-row segment branch of the golden becomes a ``torch.where`` over rows (each row
independently selects its table entry, ``b3`` and ``s3``), so there is no Python loop
over heads or tokens. Everything runs in ``torch.int64``: ``exp_value * recip`` reaches
``exp_scale * (2^recip_bits - 1)`` (~2^31), well past int32.

Calibration of the 14 scalars and the three tables lives in
``quantization.int_calibrate_softmax``.
"""
from __future__ import annotations

import torch
from torch import nn

from quantization.ilayers.qtensor import QTensor
from quantization.scheme import INT8, QuantDtype, qrange


def _scalar(v) -> float:
    return float(v.reshape(-1)[0]) if torch.is_tensor(v) else float(v)


class ISoftmax(nn.Module):
    """Fully-integer softmax: QTensor -> QTensor, bit-exact vs ``softmax_quantize``.

    ``dim`` is a real reduction axis, not decoration: every reduction in the kernel goes
    through it, so ``ISoftmax(dim=1)`` on ``x`` equals ``ISoftmax(dim=-1)`` on ``x``
    transposed (``test_isoftmax_honours_a_non_default_dim``). Only the last axis has an
    ``i_ops.softmax_quantize`` counterpart to be bit-exact against, because the golden
    walks rows of a flat attention buffer.
    """

    def __init__(
        self,
        scalars,
        exp_table,
        recip_table_one,
        recip_table_two,
        *,
        input_scale,
        output_scale,
        dim: int = -1,
        in_dtype: QuantDtype = INT8,
        out_dtype: QuantDtype | None = None,
    ) -> None:
        super().__init__()
        scalars = [int(v) for v in scalars]
        if len(scalars) != 14:
            raise ValueError(f"softmax expects 14 scalars, got {len(scalars)}")
        (
            self.b1, self.s1, self.bound1,
            self.b2_one, self.s2_one, self.bound2_one, self.b3_one, self.s3_one,
            self.b2_two, self.s2_two, self.bound2_two, self.b3_two, self.s3_two,
            self.clamp_bits,
        ) = scalars
        self.dim = int(dim)
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype or QuantDtype(self.clamp_bits, signed=False)
        self.register_buffer("exp_table", torch.as_tensor(list(exp_table), dtype=torch.int64))
        self.register_buffer("recip_table_one", torch.as_tensor(list(recip_table_one), dtype=torch.int64))
        self.register_buffer("recip_table_two", torch.as_tensor(list(recip_table_two), dtype=torch.int64))

    @classmethod
    def from_payload(cls, payload: dict, *, dim: int = -1, in_dtype: QuantDtype = INT8) -> "ISoftmax":
        """Build from a ``int_calibrate_softmax.build_softmax_int_payload`` result."""
        return cls(
            payload["scalars"],
            payload["exp_table"],
            payload["recip_table_one"],
            payload["recip_table_two"],
            input_scale=payload["input_scale"],
            output_scale=payload["output_scale"],
            dim=dim,
            in_dtype=in_dtype,
        )

    # --- integer kernel ------------------------------------------------------

    def forward_int(self, x_int: torch.Tensor) -> torch.Tensor:
        """The golden ``softmax_quantize`` over ``dim``, vectorized (int64 in/out)."""
        x = x_int.to(torch.int64)
        delta = x.amax(dim=self.dim, keepdim=True) - x  # >= 0, inverse-exp argument
        cursor1 = torch.bitwise_right_shift(delta + self.b1, self.s1).clamp(0, self.bound1)
        exp_values = self.exp_table[cursor1]
        acc = exp_values.sum(dim=self.dim, keepdim=True)

        # segment select: the golden compares the *unclamped* segment-one cursor
        cursor_one = torch.bitwise_right_shift(acc + self.b2_one, self.s2_one)
        use_two = cursor_one > self.bound2_one
        cursor_two = torch.bitwise_right_shift(acc + self.b2_two, self.s2_two).clamp(0, self.bound2_two)
        recip = torch.where(
            use_two,
            self.recip_table_two[cursor_two],
            self.recip_table_one[cursor_one.clamp(0, self.bound2_one)],
        )
        b3 = torch.where(use_two, torch.full_like(acc, self.b3_two), torch.full_like(acc, self.b3_one))
        s3 = torch.where(use_two, torch.full_like(acc, self.s3_two), torch.full_like(acc, self.s3_one))

        rel = torch.bitwise_right_shift(exp_values * recip + b3, s3)
        qmin, qmax = qrange(self.clamp_bits, signed=False)
        return rel.clamp(qmin, qmax)

    def segment_mask(self, x_int: torch.Tensor) -> torch.Tensor:
        """Per-row ``True`` where the second reciprocal segment is taken (test aid)."""
        x = x_int.to(torch.int64)
        delta = x.amax(dim=self.dim, keepdim=True) - x
        cursor1 = torch.bitwise_right_shift(delta + self.b1, self.s1).clamp(0, self.bound1)
        acc = self.exp_table[cursor1].sum(dim=self.dim, keepdim=True)
        return torch.bitwise_right_shift(acc + self.b2_one, self.s2_one) > self.bound2_one

    # --- module interfaces ---------------------------------------------------

    def forward_qtensor(self, qt: QTensor) -> QTensor:
        """Rebase an upstream QTensor onto the exp table's integer grid, then run.

        Two things this path does *not* do, both deliberate and both bounded:

        * the rebase goes through float32, so it is exact only while
          ``|int_data| * ratio`` stays under 2^24. That covers every activation dtype in
          this codebase (int8/uint8, and int32 accumulators well below 2^24 after
          requant) and it matches the precedent set by ``IGeLU``; it is *not* an exact
          integer requant, and a genuinely int32-wide input would round. When the scales
          already match, ``ratio`` is 1.0 and the conversion is exact for any input.
        * ``qt.zero_point`` is not applied. It is cancelled, not dropped: ``forward_int``
          subtracts the row max before anything else, so a uniform additive offset on the
          integer data leaves every ``delta`` — and therefore the whole output —
          unchanged. ``test_isoftmax_is_invariant_to_qtensor_zero_point`` pins that down.
          A *per-element* zero point would not cancel, but ``QTensor`` carries a scalar or
          per-channel one and softmax reduces along the last axis.
        """
        ratio = _scalar(qt.scale) / self.input_scale
        x_lut = torch.round(qt.int_data.to(torch.float32) * ratio).to(torch.int64)
        out_int = self.forward_int(x_lut)
        return QTensor(out_int.to(torch.int32), scale=self.output_scale, zero_point=0.0, dtype=self.out_dtype)

    def forward(self, x):
        """QTensor -> QTensor; a plain float tensor -> float (nn.Softmax drop-in)."""
        if isinstance(x, QTensor):
            return self.forward_qtensor(x)
        qt = QTensor.quantize(x, self.input_scale, 0.0, self.in_dtype)
        return self.forward_qtensor(qt).dequantize()


__all__ = ["ISoftmax"]
