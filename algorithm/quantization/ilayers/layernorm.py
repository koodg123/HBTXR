"""I-tier fully-integer LayerNorm (QTensor in / QTensor out).

Unlike the Q-tier ``QLayerNorm`` — which still reduces mean/variance in float and
only looks the rsqrt up in a table — ``ILayerNorm`` runs the whole normalization on
the integer datapath a HW block owns:

    mean    = (sum(row) * c_1_m + 2^(c_1_s-1)) >> c_1_s      reciprocal multiply+shift
    var_sum = sum((x - mean)^2)                              integer accumulation
    rsqrt   = rsqrt_table[clamp((var_sum + b) >> s1, 0, bound)]   PoT-indexed LUT
    out     = clamp(((x - mean) * rsqrt * lnw + lnb) >> s2)   integer affine + requant

That is exactly ``i_ops.layernorm_quantize``, and this module is bit-exact against
it: every step stays in ``torch.int64`` (the ``(x-mean)*rsqrt*lnw`` product is ~2^30
wide) and every shift is ``torch.bitwise_right_shift``, which sign-extends like
Python's ``>>`` so negative deviations floor identically. Neither ``//`` nor float
division appears in that kernel. The reductions are vectorized over the last dim —
no python loop over tokens.

With a TEN-scalar payload the rsqrt lookup instead runs the two-segment index of
``i_ops.layernorm_quantize_segmented``: the *unclamped* segment-one cursor selects
between two tables, each with its own ``(b, s1, bound)``. That per-row branch becomes a
``torch.where`` here, exactly as ``ISoftmax`` does for its dual reciprocal, so it stays
one vectorized pass. Both payload shapes are supported and both are bit-exact against
their own golden; ``segments`` says which is loaded.

Segment one's table keeps the plain name ``rsqrt_table`` rather than becoming
``rsqrt_table_one``. That is not sloppiness about symmetry: a one-segment payload has
exactly this table and calls it that, and the segmented index extends the one-segment
index rather than replacing it (``scalars[:7]`` is the seven-scalar tuple verbatim), so
one name for one thing is the accurate description.

One thing here is NOT part of the hardware block: when the incoming ``QTensor``
carries a scale different from the calibrated ``input_scale``, ``forward_int``
bridges the two grids with a float multiply-and-round before entering the kernel
(and re-clamps to ``in_dtype``, because the block's input port is that many bits
wide). A real deployment folds that ratio into the *producer's* requant as a dyadic
multiply+shift and the bridge disappears; it exists here so an I-tier graph that has
not had its scales unified is merely inexact rather than silently wrong.

The scalars and the integer vectors come from
``quantization.int_calibrate.calibrate_int_layernorm``.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from quantization.ilayers.qtensor import QTensor, rescale_ratio
from quantization.scheme import INT8, QuantDtype


class ILayerNorm(nn.Module):
    """Fully-integer LayerNorm over the last dim; bit-exact vs ``i_ops.layernorm_quantize``.

    ``forward`` accepts either a ``QTensor`` (integer graph: integer in, integer out)
    or a plain float tensor (drop-in replacement for ``nn.LayerNorm`` inside an
    otherwise-float forward, quantize -> integer kernel -> dequantize).
    """

    def __init__(
        self,
        scalars: Sequence[int],
        rsqrt_table: Sequence[int],
        lnw: Sequence[int],
        lnb: Sequence[int],
        *,
        input_scale: float,
        output_scale: float,
        in_dtype: QuantDtype = INT8,
        rsqrt_table_two: Sequence[int] | None = None,
        metrics: dict | None = None,
    ) -> None:
        super().__init__()
        # How well the (necessarily linear) PoT index fits the variance distribution it
        # was fitted to. Diagnostic, not arithmetic — a plain attribute, not a buffer, and
        # the forward never reads it. It is carried because the fit quality is not
        # recoverable from the tables afterwards: ``rows_above_range`` in particular is
        # documented as "a warning about the fit", and a warning nobody can read after
        # calibration is not a warning.
        self.metrics = dict(metrics or {})
        values = [int(v) for v in scalars]
        if len(values) not in (7, 10):
            raise ValueError(f"ILayerNorm expects 7 or 10 scalars, got {len(values)}")
        (self.c_1_m, self.c_1_s, self.b, self.s1, self.bound, self.s2, self.clamp_bits) = values[:7]
        self.segments = 2 if len(values) == 10 else 1
        # A second table with no second index (or the reverse) would silently run the
        # one-segment kernel on a payload calibrated for two, i.e. use the wrong table
        # for every high-variance row. Both halves must arrive together.
        if (self.segments == 2) != (rsqrt_table_two is not None):
            raise ValueError(
                "the segmented index needs both 10 scalars and rsqrt_table_two "
                f"(got {len(values)} scalars, rsqrt_table_two="
                f"{'present' if rsqrt_table_two is not None else 'absent'})")
        self.b_two, self.s1_two, self.bound_two = values[7:] if self.segments == 2 else (0, 0, 0)
        if self.c_1_s < 1:
            raise ValueError("c_1_s must be >= 1 (the golden adds a 1 << (c_1_s - 1) rounding term)")
        if self.s1 < 0 or self.s2 < 0 or self.s1_two < 0:
            raise ValueError("s1 / s1_two / s2 must be non-negative (arithmetic right shifts)")
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.in_dtype = in_dtype
        self.out_dtype = QuantDtype(self.clamp_bits, signed=True)
        self.register_buffer("rsqrt_table", torch.as_tensor(list(rsqrt_table), dtype=torch.int64))
        self.register_buffer("rsqrt_table_two",
                             torch.as_tensor(list(rsqrt_table_two or []), dtype=torch.int64))
        self.register_buffer("lnw", torch.as_tensor(list(lnw), dtype=torch.int64))
        self.register_buffer("lnb", torch.as_tensor(list(lnb), dtype=torch.int64))
        if self.lnw.numel() != self.lnb.numel():
            raise ValueError("lnw and lnb must have the same length (the channel count)")
        if self.rsqrt_table.numel() <= self.bound:
            raise ValueError(f"rsqrt_table needs at least bound+1={self.bound + 1} entries")
        if self.segments == 2 and self.rsqrt_table_two.numel() <= self.bound_two:
            raise ValueError(
                f"rsqrt_table_two needs at least bound_two+1={self.bound_two + 1} entries")

    # --- construction --------------------------------------------------------

    @property
    def channels(self) -> int:
        return int(self.lnw.numel())

    @classmethod
    def from_payload(cls, payload: dict) -> "ILayerNorm":
        """Build from a ``int_calibrate.calibrate_int_layernorm`` payload.

        ``rsqrt_table_two`` is read with ``.get``: a one-segment payload — including
        every payload written before the segmented index existed — simply does not have
        it, and the 7-scalar count already says so.
        """
        return cls(
            payload["scalars"],
            payload["rsqrt_table"],
            payload["lnw"],
            payload["lnb"],
            input_scale=payload["input_scale"],
            output_scale=payload["output_scale"],
            in_dtype=payload.get("input_dtype", INT8),
            rsqrt_table_two=payload.get("rsqrt_table_two"),
            metrics=payload.get("metrics"),
        )

    @classmethod
    def from_qlayernorm(cls, qln: nn.Module, samples: Any, **kwargs: Any) -> "ILayerNorm":
        """Calibrate from a Q-tier ``QLayerNorm`` (or ``nn.LayerNorm``) + observed inputs."""
        from quantization.int_calibrate import calibrate_int_layernorm

        return cls.from_payload(calibrate_int_layernorm(qln, samples, **kwargs))

    # --- integer kernel ------------------------------------------------------

    def segment_mask(self, var_sum: torch.Tensor) -> torch.Tensor:
        """Per-row ``True`` where the second rsqrt segment is taken (test aid).

        The condition is the golden's: the *unclamped* segment-one cursor above
        ``bound``. Always ``False`` for a one-segment module, which is the truth — it has
        no second segment — rather than an error, so a caller can ask either kind.
        """
        if self.segments == 1:
            return torch.zeros_like(var_sum, dtype=torch.bool)
        return torch.bitwise_right_shift(var_sum + self.b, self.s1) > self.bound

    def _rsqrt(self, var_sum: torch.Tensor) -> torch.Tensor:
        """The rsqrt each row looks up — one table, or the two-segment branch."""
        cursor = torch.bitwise_right_shift(var_sum + self.b, self.s1)
        if self.segments == 1:
            return self.rsqrt_table[cursor.clamp(0, self.bound)]
        # the golden branches on the UNCLAMPED segment-one cursor; both sides are
        # evaluated and selected per row, which is the vectorized form of that branch
        # (ISoftmax does the same for its dual reciprocal).
        cursor_two = torch.bitwise_right_shift(var_sum + self.b_two, self.s1_two)
        return torch.where(
            cursor > self.bound,
            self.rsqrt_table_two[cursor_two.clamp(0, self.bound_two)],
            self.rsqrt_table[cursor.clamp(0, self.bound)],
        )

    def forward_int(self, qt: QTensor) -> QTensor:
        """The golden datapath, vectorized over the trailing channel dim."""
        if qt.int_data.shape[-1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {qt.int_data.shape[-1]}")
        x = qt.int_data.to(torch.int64)
        ratio = rescale_ratio(qt.scale, self.input_scale, self.channels)
        if torch.is_tensor(ratio) or ratio != 1.0:
            # bridge an upstream scale mismatch; the result re-enters an in_dtype-wide
            # input port, so it must be clamped exactly like QTensor.quantize would.
            x = torch.round(x.to(torch.float64) * ratio)
            x = x.clamp(self.in_dtype.qmin, self.in_dtype.qmax).to(torch.int64)

        acc = x.sum(dim=-1, keepdim=True)
        mean = torch.bitwise_right_shift(acc * self.c_1_m + (1 << (self.c_1_s - 1)), self.c_1_s)
        diff = x - mean
        var_sum = (diff * diff).sum(dim=-1, keepdim=True)
        rsqrt = self._rsqrt(var_sum)                           # [..., 1], broadcast over C
        affine = diff * rsqrt * self.lnw + self.lnb
        shifted = torch.bitwise_right_shift(affine, self.s2)   # arithmetic: floors negatives
        out = shifted.clamp(self.out_dtype.qmin, self.out_dtype.qmax)
        return QTensor(out.to(torch.int32), scale=self.output_scale, zero_point=0.0, dtype=self.out_dtype)

    def forward(self, x: QTensor | torch.Tensor) -> QTensor | torch.Tensor:
        if isinstance(x, QTensor):
            return self.forward_int(x)
        qt = QTensor.quantize(x, self.input_scale, 0.0, self.in_dtype)
        return self.forward_int(qt).dequantize().to(x.dtype)


__all__ = ["ILayerNorm"]
