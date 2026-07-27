"""Pure-Python integer replay of a whole transformer block — the graph-tier golden.

Every torch integer *kernel* already has an oracle: ``i_ops.py`` re-implements it in
pure Python with arbitrary-precision integers, and the two are compared element-wise
``==``. That idiom is what makes the I tier trustworthy, and it is exactly what the
**graph** tier lacks.

The problem it solves. A QTensor-threaded block is *by design* a different function
from today's per-op-integer model: today every module dequantizes at its output port
and the next one requantizes, whereas the graph requantizes once per edge and clamps
back to int8. So ``graph == per_op_model`` is false on purpose, and the only remaining
comparison is ``allclose(tolerance)`` — where the tolerance gets widened until it
passes and a genuinely wrong lowering (a swapped operand scale, a missed requant, a
per-channel vector truncated to element 0 — all three have happened in this subsystem)
hides inside it.

So this module composes an entire block from ``i_ops`` primitives ONLY. The lowered
torch block is then compared against it element-wise, and the per-op-vs-graph delta
becomes a *reported measurement* rather than the correctness gate.

Independence, and its limits — stated because it is the whole point. This is not a
second author writing the same thing twice; the independence that matters here is
implementational:

- it imports ``i_ops`` and nothing from ``ilayers`` (the repo already makes this call
  explicitly: ``ilayers/conv.py`` keeps a hand copy of ``_pad_pair`` rather than import
  the golden, "because a reference that shares code with the thing it validates proves
  less");
- it is plain loops over python lists with unbounded ints, so a fixed-width overflow or
  a vectorization mistake in the torch graph cannot be mirrored here;
- each primitive it calls is *already* independently validated against the torch kernel,
  so what this adds — and all it claims to prove — is the **composition**: which scale
  each edge carries, and where a requant happens.

THE SCALE CONTRACT this encodes, which is the thing worth reviewing:

    every consumer declares the scale it wants at its input port, so the graph never
    invents a scale. An edge needs a requant exactly when the producer's scale differs
    from the consumer's declared ``input_scale``.

``ILayerNorm`` / ``ISoftmax`` / ``IGeLU`` declare both an input and an output scale.
``ILinear`` / ``IMatMul`` declare only their input(s): their output is the accumulator
scale ``s_x·s_w`` (or ``s_a·s_b``), which is derived, so an edge leaving them ALWAYS
requantizes. The ``1/√d`` factor folds into the requant that follows the score matmul —
for the shipped ``head_dim=64`` it is exactly ``2⁻³``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from quantization.i_ops import (
    dyadic_params,
    int_matmul,
    layernorm_quantize,
    layernorm_quantize_segmented,
    requant,
    softmax_quantize,
    table_quantize,
)
from quantization.scheme import INT8, QuantDtype


# --- the per-edge requant -----------------------------------------------------

def rescale(values: Sequence[int], scale_in: float, scale_out: float, *,
            dtype: QuantDtype = INT8, extra: float = 1.0) -> list[int]:
    """Move integers from grid ``scale_in`` to grid ``scale_out``, as hardware would.

    The ratio becomes a dyadic ``multiplier / 2^shift`` — an integer multiply and an
    arithmetic right shift, which is what a requant unit is — and the result is clamped
    into ``dtype``. ``extra`` folds a constant factor into the same operation; the ``1/√d``
    attention scaling is applied this way rather than as a separate multiply, because in
    hardware it is not a separate multiply.

    The identity case is short-circuited: with ``ratio == 1`` a dyadic approximation
    would still round-trip through ``multiplier/2^shift`` and could move a value by an
    LSB for no reason.
    """
    ratio = (scale_in / scale_out) * extra
    if ratio == 1.0:
        return [max(dtype.qmin, min(dtype.qmax, int(v))) for v in values]
    multiplier, shift, _effective = dyadic_params(ratio)
    return requant(list(values), multiplier, shift, bits=dtype.bits, signed=dtype.signed)


# --- the payloads one block needs --------------------------------------------

@dataclass
class LayerNormSpec:
    """Everything ``i_ops.layernorm_quantize[_segmented]`` needs, plus its two scales."""

    scalars: list[int]
    lnw: list[int]
    lnb: list[int]
    rsqrt_table: list[int]
    rsqrt_table_two: list[int] | None
    input_scale: float
    output_scale: float

    def apply(self, values: Sequence[int]) -> list[int]:
        if self.rsqrt_table_two is None:
            return layernorm_quantize(list(values), self.scalars, self.lnw, self.lnb,
                                      self.rsqrt_table)
        return layernorm_quantize_segmented(list(values), self.scalars, self.lnw, self.lnb,
                                            self.rsqrt_table, self.rsqrt_table_two)


@dataclass
class SoftmaxSpec:
    scalars: list[int]
    exp_table: list[int]
    recip_table_one: list[int]
    recip_table_two: list[int]
    input_scale: float
    output_scale: float

    def apply(self, values: Sequence[int], *, tokens: int, heads: int) -> list[int]:
        return softmax_quantize(list(values), self.scalars, self.exp_table,
                                self.recip_table_one, self.recip_table_two,
                                tokens=tokens, heads=heads)


@dataclass
class TableSpec:
    """A pointwise PoT-indexed table (GeLU)."""

    scalars: list[int]
    table: list[int]
    input_scale: float
    output_scale: float

    def apply(self, values: Sequence[int]) -> list[int]:
        return table_quantize(list(values), self.scalars, self.table)


@dataclass
class LinearSpec:
    """int weight ``[out, in]`` + the two scales that make its output grid."""

    weight: list[list[int]]
    weight_scale: list[float]      # per out-channel; length 1 means per-tensor
    input_scale: float
    bias: list[float] | None = None

    def out_scale(self, out_channel: int) -> float:
        scales = self.weight_scale
        return self.input_scale * float(scales[out_channel if len(scales) > 1 else 0])


@dataclass
class MatMulSpec:
    scale_a: float
    scale_b: float

    @property
    def out_scale(self) -> float:
        return self.scale_a * self.scale_b


@dataclass
class AddSpec:
    scale_a: float
    scale_b: float
    scale_out: float


@dataclass
class BlockSpec:
    """One transformer block, in the order its forward runs."""

    num_heads: int
    head_dim: int
    norm1: LayerNormSpec
    qkv: LinearSpec
    qk: MatMulSpec
    attn_scale: float
    softmax: SoftmaxSpec
    av: MatMulSpec
    proj: LinearSpec
    attn_residual: AddSpec
    norm2: LayerNormSpec
    fc1: LinearSpec
    gelu: TableSpec
    fc2: LinearSpec
    mlp_residual: AddSpec
    dtype: QuantDtype = field(default=INT8)


# --- helpers over flat row-major lists ---------------------------------------

def _rows(flat: Sequence[int], width: int) -> list[list[int]]:
    return [list(flat[i:i + width]) for i in range(0, len(flat), width)]


def _flat(rows: Sequence[Sequence[int]]) -> list[int]:
    return [int(v) for row in rows for v in row]


def _linear_acc(spec: LinearSpec, tokens_rows: list[list[int]]) -> list[list[int]]:
    """``x @ Wᵀ + b`` left ON THE ACCUMULATOR, un-requantized.

    Exposed separately because requantizing too early is a real and easy mistake: the
    qkv projection feeds three consumers (Q, K and V) whose calibrated grids differ, so
    squeezing its accumulator into one of them first and re-scaling afterwards clamps
    the other two through a grid that was never sized for them. Measured on the shipped
    fixture, doing that costs 3.2% at the qkv output alone. Each consumer must requant
    from the accumulator directly.
    """
    weight_t = [list(col) for col in zip(*spec.weight)]          # [in, out]
    acc = int_matmul(tokens_rows, weight_t)                      # [tokens, out]
    if spec.bias is None:
        return acc
    bias_int = [round(float(spec.bias[oc]) / spec.out_scale(oc)) for oc in range(len(spec.weight))]
    return [[v + bias_int[oc] for oc, v in enumerate(row)] for row in acc]


def _requant_columns(spec: LinearSpec, acc: list[list[int]], out_scale: float,
                     dtype: QuantDtype, *, columns: range | None = None) -> list[list[int]]:
    """Requant an accumulator (or a column slice of one) onto ``out_scale``, per channel."""
    span = columns if columns is not None else range(len(spec.weight))
    moved = [rescale([row[oc] for row in acc], spec.out_scale(oc), out_scale, dtype=dtype)
             for oc in span]
    return [[moved[i][t] for i in range(len(moved))] for t in range(len(acc))]


def _linear(spec: LinearSpec, tokens_rows: list[list[int]], out_scale: float,
            dtype: QuantDtype) -> list[list[int]]:
    """``x @ Wᵀ + b`` in integers, then one requant per OUT-CHANNEL to ``out_scale``.

    Two things here are the reason this oracle exists.

    The bias is quantized onto the ACCUMULATOR grid ``s_x·s_w`` and added *before* the
    requant, not after. That is what hardware does — the accumulator is where a bias can
    be added for free — and it is also the only place it is exact: adding it after the
    requant would round it onto the coarser output grid.

    The requant runs per OUT-CHANNEL. A per-out-channel weight scale puts every column of
    the accumulator on its own grid, so collapsing them to one ratio is precisely the
    element-0 truncation bug this subsystem has already shipped once.
    """
    weight_t = [list(col) for col in zip(*spec.weight)]          # [in, out]
    acc = int_matmul(tokens_rows, weight_t)                      # [tokens, out]
    out_features = len(spec.weight)
    columns: list[list[int]] = []
    for oc in range(out_features):
        acc_scale = spec.out_scale(oc)
        column = [row[oc] for row in acc]
        if spec.bias is not None:
            bias_int = round(float(spec.bias[oc]) / acc_scale)
            column = [v + bias_int for v in column]
        columns.append(rescale(column, acc_scale, out_scale, dtype=dtype))
    return [[columns[oc][t] for oc in range(out_features)] for t in range(len(acc))]


def _add(spec: AddSpec, a: Sequence[int], b: Sequence[int], dtype: QuantDtype) -> list[int]:
    """Align two grids onto ``scale_out`` and add — the residual join."""
    a_out = rescale(a, spec.scale_a, spec.scale_out, dtype=dtype)
    b_out = rescale(b, spec.scale_b, spec.scale_out, dtype=dtype)
    return [max(dtype.qmin, min(dtype.qmax, x + y)) for x, y in zip(a_out, b_out)]


# --- the block ----------------------------------------------------------------

def replay_block_int(spec: BlockSpec, x_int: Sequence[int], *, tokens: int,
                     channels: int) -> list[int]:
    """One transformer block, entirely in integers, on a flat row-major ``[tokens, channels]``.

    ``x_int`` is on ``spec.attn_residual.scale_a`` — the residual stream's own grid,
    which is what the block's input port declares. The return is on
    ``spec.mlp_residual.scale_out``, ready to be the next block's input.
    """
    dtype = spec.dtype
    if len(x_int) != tokens * channels:
        raise ValueError(f"expected {tokens * channels} values, got {len(x_int)}")
    residual = [int(v) for v in x_int]

    # --- attention sublayer ---------------------------------------------------
    normed = spec.norm1.apply(rescale(residual, spec.attn_residual.scale_a,
                                      spec.norm1.input_scale, dtype=dtype))
    qkv_acc = _linear_acc(spec.qkv, _rows(rescale(normed, spec.norm1.output_scale,
                                                  spec.qkv.input_scale, dtype=dtype), channels))

    # Split into per-head Q / K / V. The rows are [3 * heads * head_dim], laid out the
    # way MultiHeadAttention reshapes them: (3, heads, head_dim). Each of the three goes
    # from the ACCUMULATOR straight to its own calibrated grid — see _linear_acc for why
    # a shared intermediate grid is wrong.
    heads, head_dim = spec.num_heads, spec.head_dim

    def part(index: int, target_scale: float) -> list[list[list[int]]]:
        base = index * heads * head_dim
        return [_requant_columns(spec.qkv, qkv_acc, target_scale, dtype,
                                 columns=range(base + h * head_dim,
                                               base + (h + 1) * head_dim))
                for h in range(heads)]

    query = part(0, spec.qk.scale_a)
    key = part(1, spec.qk.scale_b)
    value = part(2, spec.av.scale_b)

    # scores, with the 1/sqrt(d) factor folded into the requant that follows
    scores: list[int] = []
    for h in range(heads):
        acc = int_matmul(query[h], [list(col) for col in zip(*key[h])])   # [tokens, tokens]
        scores.extend(rescale(_flat(acc), spec.qk.out_scale, spec.softmax.input_scale,
                              dtype=dtype, extra=spec.attn_scale))

    probs = spec.softmax.apply(scores, tokens=tokens, heads=heads)

    context_heads: list[list[list[int]]] = []
    for h in range(heads):
        block = probs[h * tokens * tokens:(h + 1) * tokens * tokens]
        attn_rows = _rows(rescale(block, spec.softmax.output_scale, spec.av.scale_a,
                                  dtype=dtype), tokens)
        acc = int_matmul(attn_rows, value[h])                              # [tokens, head_dim]
        context_heads.append(_rows(rescale(_flat(acc), spec.av.out_scale,
                                           spec.proj.input_scale, dtype=dtype), head_dim))

    # concatenate heads back to [tokens, channels] — heads are already on one grid
    context = [[context_heads[h][t][c] for h in range(heads) for c in range(head_dim)]
               for t in range(tokens)]
    attn_out = _flat(_linear(spec.proj, context, spec.attn_residual.scale_b, dtype))
    residual = _add(spec.attn_residual, residual, attn_out, dtype)

    # --- MLP sublayer ---------------------------------------------------------
    normed = spec.norm2.apply(rescale(residual, spec.mlp_residual.scale_a,
                                      spec.norm2.input_scale, dtype=dtype))
    hidden = _linear(spec.fc1, _rows(rescale(normed, spec.norm2.output_scale,
                                             spec.fc1.input_scale, dtype=dtype), channels),
                     spec.gelu.input_scale, dtype)
    activated = spec.gelu.apply(_flat(hidden))
    hidden_rows = _rows(rescale(activated, spec.gelu.output_scale, spec.fc2.input_scale,
                                dtype=dtype), len(spec.fc1.weight))
    mlp_out = _flat(_linear(spec.fc2, hidden_rows, spec.mlp_residual.scale_b, dtype))
    return _add(spec.mlp_residual, residual, mlp_out, dtype)


__all__ = [
    "rescale",
    "LayerNormSpec",
    "SoftmaxSpec",
    "TableSpec",
    "LinearSpec",
    "MatMulSpec",
    "AddSpec",
    "BlockSpec",
    "replay_block_int",
]
