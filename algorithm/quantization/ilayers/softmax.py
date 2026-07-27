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

from quantization.ilayers.qtensor import QTensor, rescale_ratio
from quantization.scheme import INT8, QuantDtype, qrange


def _derived_max_tokens(b2_two: int, s2_two: int, bound2_two: int, exp_table: torch.Tensor) -> int:
    """Longest row the *emitted scalars* can still normalise — a legacy-payload fallback.

    ``cursor_two = clamp((acc + b2_two) >> s2_two, 0, bound2_two)`` stops tracking ``acc``
    once it saturates, so the last accumulator segment two actually resolves is
    ``covered = ((bound2_two + 1) << s2_two) - b2_two - 1``. A row of ``n`` tokens tops
    out at ``n * max(exp_table)`` (every element at the row max), hence
    ``n <= covered // max(exp_table)``.

    This is **second-class on purpose**: it is a bound on the *emitted table*, not the
    calibrated intent, and it is loose. ``make_pot_index_params`` rounds the segment bin
    width up to a power of two, so the tables habitually reach past the envelope they were
    sized for — measured 10 for an 8-token calibration, 20 for 16, 144 for 96, i.e. 25-50%
    over. Rows in that slack are not rejected and not broken either (the extra coverage is
    real), only under-resolved, and the degradation is gradual with no cliff at the
    calibrated boundary to detect: worst flat row sum 1.0039 at the calibrated 16 tokens
    against 1.0196 at the derived 20, and on realistic logits 1.0627 at 96 against 1.0667
    at 144. A bound that admits rows nobody calibrated for, by a margin nothing measures,
    is a fallback and not a contract — use the calibrated value whenever the payload
    carries one.
    """
    covered = ((int(bound2_two) + 1) << int(s2_two)) - int(b2_two) - 1
    return max(1, covered // max(int(exp_table.max()), 1))


class ISoftmax(nn.Module):
    """Fully-integer softmax: QTensor -> QTensor, bit-exact vs ``softmax_quantize``.

    ``dim`` is a real reduction axis, not decoration: every reduction in the kernel goes
    through it, so ``ISoftmax(dim=1)`` on ``x`` equals ``ISoftmax(dim=-1)`` on ``x``
    transposed (``test_isoftmax_honours_a_non_default_dim``). Only the last axis has an
    ``i_ops.softmax_quantize`` counterpart to be bit-exact against, because the golden
    walks rows of a flat attention buffer.

    ``max_tokens`` is the single load-bearing *assumption* of the calibration: the
    reciprocal segments span ``[exp_table[0], max_tokens * max(exp_table)]``, which is
    what makes ``cursor_two`` saturation unreachable — but only for rows no longer than
    ``max_tokens``. ``forward_int`` therefore rejects a longer row instead of returning an
    un-normalised one (see the guard for the measured damage). A *shorter* row is safe,
    not merely tolerated: the envelope's lower bound ``exp_table[0]`` is attained by any
    row of length >= 1, so a short row lands strictly inside the covered range. It costs
    resolution rather than correctness, and the cost is on the calibration side — with a
    32-entry reciprocal, running 16-token rows through a ``max_tokens=96`` payload moves
    the max abs error from 1.81e-02 to 4.55e-02. So size ``max_tokens`` to the longest row
    that will really occur, not to a comfortable round number.
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
        max_tokens: int | None = None,
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
        # Optional, not required: a manifest written before the field existed — and any
        # hand-built payload — still has to rebuild. Falling back keeps those working at
        # a looser bound rather than refusing to load them; see _derived_max_tokens.
        if max_tokens is None:
            max_tokens = _derived_max_tokens(self.b2_two, self.s2_two, self.bound2_two, self.exp_table)
        # A BUFFER: this is a SAFETY bound, and a state_dict round-trip that dropped it
        # would silently fall back to the looser derived value — weakening a guard is a
        # worse failure than losing a shape, because nothing downstream would complain.
        self.register_buffer("max_tokens_buf", torch.tensor(int(max_tokens), dtype=torch.int64))
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}")

    @property
    def max_tokens(self) -> int:
        """The calibrated row-length bound; stored as a buffer so it survives a reload."""
        return int(self.max_tokens_buf)

    @classmethod
    def from_payload(cls, payload: dict, *, dim: int = -1, in_dtype: QuantDtype = INT8) -> "ISoftmax":
        """Build from a ``int_calibrate_softmax.build_softmax_int_payload`` result.

        ``max_tokens`` is read from the payload's top level, not from ``payload["metrics"]``
        — metrics are a report, and nothing downstream (``export.py``) carries them.
        """
        return cls(
            payload["scalars"],
            payload["exp_table"],
            payload["recip_table_one"],
            payload["recip_table_two"],
            input_scale=payload["input_scale"],
            output_scale=payload["output_scale"],
            dim=dim,
            in_dtype=in_dtype,
            max_tokens=payload.get("max_tokens"),
        )

    # --- integer kernel ------------------------------------------------------

    def forward_int(self, x_int: torch.Tensor) -> torch.Tensor:
        """The golden ``softmax_quantize`` over ``dim``, vectorized (int64 in/out).

        Rejects a row longer than the calibration covered. Clamping or truncating is not
        an alternative: the op's contract is a normalised row, and dropping tokens
        computes a different function. Returning the row anyway is worse still — it is
        silently un-normalised, because the accumulator saturates ``cursor_two`` and the
        row is then divided by the calibration maximum instead of by its own sum. Measured
        on a 16-token calibration replayed at 96 tokens: row sum 4.5176 on a flat row,
        2.9059 on realistic logits, against the 1.0 the caller is entitled to.

        The cost is one metadata read: ``dim`` is a real reduction axis, so the row length
        is ``shape[dim]`` — O(1), no data-dependent scan. In an RTL port the token count is
        a compile-time constant and the comparison folds away entirely.
        """
        tokens = int(x_int.shape[self.dim])
        if tokens > self.max_tokens:
            raise ValueError(
                f"softmax row has {tokens} tokens along dim {self.dim} but this ISoftmax "
                f"was calibrated for at most max_tokens={self.max_tokens}; the reciprocal "
                f"segments do not cover the accumulator a longer row reaches, so the "
                f"output row would be silently un-normalised. Recalibrate with "
                f"build_softmax_int_payload(..., max_tokens={tokens})."
            )
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

        Three things this path does, each deliberate and each bounded:

        * the rebase goes through float64, so ``|int_data| * ratio`` is exact out to
          2^53 — every activation dtype in this codebase including a full-width int32
          accumulator. It is still *not* an exact integer requant (the rounding is a
          real rounding), but it never loses magnitude. When the scales already match,
          ``ratio`` is 1.0, the bridge is skipped entirely and the input passes through
          untouched.
        * a per-channel scale **along the reduction axis is bridged, not rejected**. The
          worry is real: ``forward_int`` takes an integer row max and indexes one exp
          table, and both assume the whole row sits on a single grid — which a raw
          per-channel row does not. But the bridge is precisely what establishes that
          grid. It runs before any reduction and re-expresses every element on
          ``input_scale``, so ``forward_int`` still sees a uniform row; raising instead
          would ban the one production path that generates these scales
          (``ILinear.forward_accumulator`` feeding an attention softmax) over a
          condition the bridge has already removed. The price is that the rounding is
          now per element rather than uniform (still <= 0.5 LSB of ``input_scale``
          each), and that a channel whose scale far exceeds ``input_scale`` saturates
          the ``in_dtype`` port — visible clamping, not a silent misread.
        * the bridged value re-enters an ``in_dtype``-wide input port, so it is clamped
          there exactly as ``QTensor.quantize`` and ``ILayerNorm.forward_int`` do.
          Without that a bridge could hand ``forward_int`` a value the block's input
          port cannot physically carry.

        One thing it does *not* do: ``qt.zero_point`` is not applied. It is cancelled,
        not dropped — ``forward_int`` subtracts the row max before anything else, so a
        uniform additive offset on the integer data leaves every ``delta`` — and
        therefore the whole output — unchanged.
        ``test_isoftmax_is_invariant_to_qtensor_zero_point`` pins that down. A
        *per-element* zero point would not cancel, but ``QTensor`` carries a scalar or
        per-channel one and softmax reduces along the last axis.
        """
        ratio = rescale_ratio(qt.scale, self.input_scale, qt.int_data.shape[-1])
        x_lut = qt.int_data.to(torch.int64)
        if torch.is_tensor(ratio) or ratio != 1.0:
            x_lut = torch.round(x_lut.to(torch.float64) * ratio).to(torch.int64)
        # Clamp unconditionally, not only when a bridge ran: the input port is a
        # physical width, so whether a value fits it cannot depend on how the caller
        # happened to spell an equivalent scale. Skipping this when ratio == 1.0 made
        # two spellings of the same grid produce different outputs.
        x_lut = x_lut.clamp(self.in_dtype.qmin, self.in_dtype.qmax)
        out_int = self.forward_int(x_lut)
        return QTensor(out_int.to(torch.int32), scale=self.output_scale, zero_point=0.0, dtype=self.out_dtype)

    def forward(self, x):
        """QTensor -> QTensor; a plain float tensor -> float (nn.Softmax drop-in)."""
        if isinstance(x, QTensor):
            return self.forward_qtensor(x)
        qt = QTensor.quantize(x, self.input_scale, 0.0, self.in_dtype)
        return self.forward_qtensor(qt).dequantize()


__all__ = ["ISoftmax"]
