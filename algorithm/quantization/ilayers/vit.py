"""Integer ViT assembly — the block forwards rewritten to thread ``QTensor``.

This mirrors ``models/blocks/`` one class at a time (``IMultiHeadAttention`` for
``MultiHeadAttention``, ``IMlp`` for ``Mlp``, ``ITransformerBlock`` for
``TransformerBlock``), and the only real difference is the transport: nothing here
dequantizes between ops.

Why that is a separate module rather than a fix to the existing I-tier layers.
``ILinear`` / ``IConv2d`` / ``ILayerNorm`` / ``ISoftmax`` have a *float I/O port* — each
quantizes at its input and dequantizes at its output — which is exactly what lets a
converted model run through the unmodified model ``forward``. Every op is integer, but
the tensor between two ops is float32. That is deployable as a simulation and not as
hardware: a float edge is a float bus.

So these classes reuse the same verified kernels and replace only the plumbing. The
kernels stay untouched precisely because they are already bit-exact against ``i_ops``;
what is new is the composition, and the composition has its own oracle
(``quantization.i_block.replay_block_int``) which these are compared against
element-wise.

THE SCALE CONTRACT, which is the whole design:

    every consumer declares the scale it wants at its input port, so the graph never
    invents one. An edge requantizes exactly when the producer's scale differs from the
    consumer's declared ``input_scale``, and the requant is a dyadic multiply + shift.

``ILayerNorm`` / ``ISoftmax`` / ``IGeLU`` declare both ends. ``ILinear`` declares only
its input: its output rides the accumulator grid ``s_x·s_w``, which is derived, so an
edge leaving one ALWAYS requantizes — and the qkv projection requantizes *three times*,
once per consumer, straight from the accumulator. Routing Q, K and V through a shared
intermediate grid instead measured 3.2% error at the qkv output, because the two that
did not own that grid were clamped through a range never calibrated for them.

``1/√d`` is folded into the requant that follows the score matmul rather than applied as
its own multiply; for the shipped ``head_dim=64`` it is exactly ``2⁻³``.
"""
from __future__ import annotations

import torch
from torch import nn

from quantization.ilayers.int_functional import _dyadic_params, int_matmul, requant
from quantization.ilayers.layernorm import ILayerNorm
from quantization.ilayers.linear import ILinear
from quantization.ilayers.nonlinear import IGeLU
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.softmax import ISoftmax
from quantization.scheme import INT8, QuantDtype


def _weight_scales(linear: ILinear) -> list[float]:
    """Per-out-channel weight scale as python floats, read ONCE at build time."""
    flat = linear.weight_scale.reshape(-1).tolist()
    return [float(v) for v in flat] * (linear.weight_int.shape[0] if len(flat) == 1 else 1)


class _IntLinear(nn.Module):
    """An ``ILinear`` with every float constant resolved at BUILD time.

    Nothing in ``forward`` touches a float. The integer bias and the per-output-channel
    dyadic ``(multiplier, shift)`` are what a compiler would emit once and a requant unit
    would then apply forever; computing them per call would be float arithmetic on the
    datapath, which is precisely what this module exists to remove — and a mechanical
    check (``test_no_float_tensor_crosses_the_datapath``) enforces it.

    One instance per (linear, destination scale) pair, because the multiplier depends on
    where the accumulator is going. The qkv projection therefore gets three.
    """

    def __init__(self, linear: ILinear, out_scale: float, *, dtype: QuantDtype = INT8,
                 columns: slice | None = None) -> None:
        super().__init__()
        self.linear = linear
        self.dtype = dtype
        self.columns = columns
        self.out_scale = float(out_scale)
        self.act_scale = float(linear.act_scale)

        scales = _weight_scales(linear)
        span = range(*columns.indices(len(scales))) if columns is not None else range(len(scales))
        acc_scales = [self.act_scale * scales[oc] for oc in span]

        mults, shifts = [], []
        for acc_scale in acc_scales:
            multiplier, shift = _dyadic_params(acc_scale / self.out_scale)
            mults.append(multiplier)
            shifts.append(shift)
        self.register_buffer("multiplier", torch.tensor(mults, dtype=torch.int64))
        self.register_buffer("shift", torch.tensor(shifts, dtype=torch.int64))

        if linear.bias is None:
            self.bias_int = None
        else:
            bias = linear.bias.reshape(-1).to(torch.float64)
            full = [round(float(bias[oc]) / (self.act_scale * scales[oc]))
                    for oc in range(len(scales))]
            self.register_buffer("bias_int", torch.tensor(full, dtype=torch.int64))

    def accumulate(self, x_int: torch.Tensor) -> torch.Tensor:
        """``x @ Wᵀ + b`` on the accumulator — the bias is added here, exactly."""
        acc = int_matmul(x_int, self.linear.weight_int.transpose(0, 1))
        return acc if self.bias_int is None else acc + self.bias_int

    def requant(self, acc: torch.Tensor) -> torch.Tensor:
        """Accumulator -> ``out_scale``, one dyadic multiply+shift PER out-channel."""
        sliced = acc if self.columns is None else acc[..., self.columns]
        return requant(sliced, self.multiplier, self.shift, dtype=self.dtype).to(torch.int64)

    def forward(self, x_int: torch.Tensor) -> torch.Tensor:
        return self.requant(self.accumulate(x_int))


class _Requant(nn.Module):
    """A single edge's requant, with its dyadic constants resolved at build time."""

    def __init__(self, scale_in: float, scale_out: float, *, dtype: QuantDtype = INT8,
                 extra: float = 1.0) -> None:
        super().__init__()
        self.dtype = dtype
        ratio = (float(scale_in) / float(scale_out)) * float(extra)
        self.identity = ratio == 1.0
        multiplier, shift = (1, 0) if self.identity else _dyadic_params(ratio)
        self.register_buffer("multiplier", torch.tensor(int(multiplier), dtype=torch.int64))
        self.register_buffer("shift", torch.tensor(int(shift), dtype=torch.int64))

    def forward(self, int_data: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return int_data.clamp(self.dtype.qmin, self.dtype.qmax).to(torch.int64)
        return requant(int_data, self.multiplier, self.shift, dtype=self.dtype).to(torch.int64)


class IMultiHeadAttention(nn.Module):
    """``MultiHeadAttention`` with a QTensor datapath and no float hop."""

    def __init__(self, attn: nn.Module, *, input_scale: float,
                 out_scale: float, dtype: QuantDtype = INT8) -> None:
        super().__init__()
        heads, head_dim = int(attn.num_heads), int(attn.head_dim)
        self.num_heads, self.head_dim = heads, head_dim
        self.softmax: ISoftmax = attn.attn_softmax
        self.dtype = dtype

        width = heads * head_dim
        qkv, proj = attn.qkv, attn.proj
        self.to_qkv_port = _Requant(input_scale, qkv.act_scale, dtype=dtype)
        # three requants out of ONE accumulator, one per consumer's calibrated grid
        self.query = _IntLinear(qkv, attn.qk_matmul.scale_a, dtype=dtype,
                                columns=slice(0, width))
        self.key = _IntLinear(qkv, attn.qk_matmul.scale_b, dtype=dtype,
                              columns=slice(width, 2 * width))
        self.value = _IntLinear(qkv, attn.av_matmul.scale_b, dtype=dtype,
                                columns=slice(2 * width, 3 * width))
        # the 1/sqrt(d) factor rides this requant rather than being its own multiply
        self.to_softmax = _Requant(attn.qk_matmul.scale_a * attn.qk_matmul.scale_b,
                                   self.softmax.input_scale, dtype=dtype,
                                   extra=float(attn.attn_scale.factor))
        self.to_context = _Requant(self.softmax.output_scale, attn.av_matmul.scale_a,
                                   dtype=dtype)
        self.to_proj_port = _Requant(attn.av_matmul.scale_a * attn.av_matmul.scale_b,
                                     proj.act_scale, dtype=dtype)
        self.proj = _IntLinear(proj, out_scale, dtype=dtype)

    def forward(self, x_int: torch.Tensor) -> torch.Tensor:
        heads, head_dim = self.num_heads, self.head_dim
        acc = self.query.accumulate(self.to_qkv_port(x_int))

        def head_split(branch: _IntLinear) -> torch.Tensor:
            moved = branch.requant(acc)
            return moved.reshape(moved.shape[:-1] + (heads, head_dim)).transpose(-3, -2)

        query, key, value = head_split(self.query), head_split(self.key), head_split(self.value)
        probs = self.softmax.forward_int(
            self.to_softmax(int_matmul(query, key.transpose(-2, -1))))
        context = self.to_proj_port(int_matmul(self.to_context(probs), value))
        merged = context.transpose(-3, -2).reshape(
            context.shape[:-3] + (context.shape[-2], heads * head_dim))
        return self.proj(merged)


class IMlp(nn.Module):
    """``Mlp`` with a QTensor datapath: fc1 -> integer GeLU table -> fc2."""

    def __init__(self, mlp: nn.Module, *, input_scale: float, out_scale: float,
                 dtype: QuantDtype = INT8) -> None:
        super().__init__()
        self.gelu: IGeLU = getattr(mlp.act, "kernel", mlp.act)
        self.dtype = dtype
        self.to_fc1_port = _Requant(input_scale, mlp.fc1.act_scale, dtype=dtype)
        self.fc1 = _IntLinear(mlp.fc1, self.gelu.input_scale, dtype=dtype)
        self.to_fc2_port = _Requant(self.gelu.output_scale, mlp.fc2.act_scale, dtype=dtype)
        self.fc2 = _IntLinear(mlp.fc2, out_scale, dtype=dtype)

    def forward(self, x_int: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(self.to_fc1_port(x_int))
        activated = self.gelu(QTensor(hidden.to(torch.int32), scale=self.gelu.input_scale,
                                      zero_point=0.0, dtype=self.dtype))
        return self.fc2(self.to_fc2_port(activated.int_data))


class ITransformerBlock(nn.Module):
    """``TransformerBlock`` end to end in integers: no tensor here is ever float.

    Built from an already-converted block, so every kernel it runs is one that is already
    bit-exact against ``i_ops``. What this class adds — and what
    ``i_block.replay_block_int`` exists to check — is where each requant happens.
    """

    def __init__(self, block: nn.Module, *, dtype: QuantDtype = INT8) -> None:
        super().__init__()
        self.norm1: ILayerNorm = block.norm1
        self.norm2: ILayerNorm = block.norm2
        self.dtype = dtype
        res1, res2 = block.attn_residual, block.mlp_residual
        self.attn_scale_a = float(res1.scale_a)
        self.mlp_scale_a = float(res2.scale_a)
        self.output_scale = float(res2.scale_out)

        self.to_norm1 = _Requant(res1.scale_a, self.norm1.input_scale, dtype=dtype)
        self.attn = IMultiHeadAttention(block.attn, input_scale=self.norm1.output_scale,
                                        out_scale=res1.scale_b, dtype=dtype)
        self.stream_to_attn_out = _Requant(res1.scale_a, res1.scale_out, dtype=dtype)
        self.branch_to_attn_out = _Requant(res1.scale_b, res1.scale_out, dtype=dtype)

        self.to_norm2 = _Requant(res2.scale_a, self.norm2.input_scale, dtype=dtype)
        self.mlp = IMlp(block.mlp, input_scale=self.norm2.output_scale,
                        out_scale=res2.scale_b, dtype=dtype)
        self.stream_to_out = _Requant(res2.scale_a, res2.scale_out, dtype=dtype)
        self.branch_to_out = _Requant(res2.scale_b, res2.scale_out, dtype=dtype)

    @property
    def input_scale(self) -> float:
        """The grid the block expects its residual stream on."""
        return self.attn_scale_a

    def _norm(self, norm: ILayerNorm, port: _Requant, stream: torch.Tensor) -> torch.Tensor:
        normed = norm.forward_int(QTensor(port(stream).to(torch.int32),
                                          scale=norm.input_scale, zero_point=0.0,
                                          dtype=norm.in_dtype))
        return normed.int_data.to(torch.int64)

    def forward(self, qt: QTensor) -> QTensor:
        stream = _as_int(qt, self.attn_scale_a, self.dtype)

        branch = self.attn(self._norm(self.norm1, self.to_norm1, stream))
        stream = (self.stream_to_attn_out(stream) + self.branch_to_attn_out(branch)
                  ).clamp(self.dtype.qmin, self.dtype.qmax)

        branch = self.mlp(self._norm(self.norm2, self.to_norm2, stream))
        stream = (self.stream_to_out(stream) + self.branch_to_out(branch)
                  ).clamp(self.dtype.qmin, self.dtype.qmax)
        return QTensor(stream.to(torch.int32), scale=self.output_scale,
                       zero_point=0.0, dtype=self.dtype)


def _as_int(qt: QTensor, expected_scale: float, dtype: QuantDtype) -> torch.Tensor:
    """The block's input port. The caller must already be on the declared grid.

    A rescale here would be a float ratio computed per call — the one thing this module
    removes — so a mismatch is rejected instead of silently bridged. The producer knows
    its own destination and requantizes on its way out.
    """
    scale = float(qt.scale.reshape(-1)[0]) if torch.is_tensor(qt.scale) else float(qt.scale)
    if scale != expected_scale:
        raise ValueError(
            f"block input is on scale {scale!r} but this block declares {expected_scale!r}; "
            "requantize at the producer, not here")
    return qt.int_data.to(torch.int64)


__all__ = ["IMultiHeadAttention", "IMlp", "ITransformerBlock"]
