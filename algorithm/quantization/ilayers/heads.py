"""Integer regression heads — ``models/heads/`` with a QTensor datapath.

Mirrors ``models/heads/common.mlp_head`` (Linear -> GeLU -> Linear over pooled tokens),
which is the shape every non-dense head in this model shares: the box head, the ROI
guidance head, the reliability head and the ellipse residual head differ only in their
output width and in what the host does with the result.

Two decisions here are worth stating, because both are places where the obvious thing is
wrong.

**The last linear is not requantized.** Every other edge in the graph has a consumer that
declares the scale it wants; the last one's consumer is the host. There is no calibrated
output grid, so requantizing would round the answer onto a grid nobody measured, for
nothing. What leaves the accelerator is the int32 accumulator and its per-out-channel
scale ``s_x·s_w`` — exactly what a regression head reads out in hardware.

**The pooling is an integer mean, and the concat is a requant.** ``IPool`` sums along the
token axis in int64 and rounds half away from zero (a token-axis sum is the easiest place
in this graph to run past float32's exact-integer limit). ``IEllipseHead``'s concat joins
two tensors that are NOT on a common grid — pooled features and a host-supplied anchor
state — so ``ICat`` aligns both onto the scale the following linear declares, which is
the same scale contract every other edge follows.

NOT covered here: the dense mask/heatmap heads. Their conv stack would thread the same
way, but they end in ``F.interpolate(mode="bilinear")``, and an integer bilinear resample
is a real design choice (nearest, or fixed-point weights at some fractional width) that
nobody in this repo has made. Guessing one silently would be worse than leaving them on
the float-I/O path and saying so.
"""
from __future__ import annotations

import torch
from torch import nn

from models.heads.ellipse import STATE_DIM
from quantization.ilayers.nonlinear import IGeLU
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.tensor_ops import ICat, IPool
from quantization.ilayers.vit import _IntLinear, _Requant
from quantization.scheme import INT8, QuantDtype


def _kernel(act: nn.Module) -> IGeLU:
    """The integer GeLU inside a converted activation (which may be a float-I/O wrapper)."""
    return getattr(act, "kernel", act)


class _IntMlp(nn.Module):
    """Linear -> GeLU -> Linear on a pooled feature vector, ending on the accumulator."""

    def __init__(self, regressor: nn.Sequential, *, input_scale: float,
                 dtype: QuantDtype = INT8) -> None:
        super().__init__()
        fc1, act, fc2 = regressor[0], regressor[1], regressor[2]
        self.gelu = _kernel(act)
        self.dtype = dtype
        self.to_fc1 = _Requant(input_scale, fc1.act_scale, dtype=dtype)
        self.fc1 = _IntLinear(fc1, self.gelu.input_scale, dtype=dtype)
        self.to_fc2 = _Requant(self.gelu.output_scale, fc2.act_scale, dtype=dtype)
        self.fc2 = _IntLinear(fc2, None, dtype=dtype)      # None: the consumer is the host

    @property
    def input_scale(self) -> float:
        """The grid this head wants its pooled features on, before its own bridge."""
        return float(self.fc1.act_scale)

    def forward(self, x_int: torch.Tensor) -> QTensor:
        hidden = self.fc1(self.to_fc1(x_int))
        activated = self.gelu(QTensor(hidden.to(torch.int32), scale=self.gelu.input_scale,
                                      zero_point=0.0, dtype=self.dtype))
        acc = self.fc2(self.to_fc2(activated.int_data))
        return QTensor(acc.to(torch.int32), scale=self.fc2.acc_scale, zero_point=0.0,
                       dtype=self.fc2.linear.act_dtype)


class IPooledMlpHead(nn.Module):
    """``PupilBoxHead`` / ``EyeRegionHead`` / ``ReliabilityHead``: pool then regress.

    ``ReliabilityHead`` also applies a sigmoid. That stays on the host side of the
    boundary: it is a monotone squash of the final accumulator, so running it after the
    single dequantization changes nothing about what the accelerator computes, and putting
    a second LUT in the graph to avoid one host-side sigmoid would be inventing work.
    """

    def __init__(self, head: nn.Module, *, input_scale: float,
                 dtype: QuantDtype = INT8) -> None:
        super().__init__()
        regressor = getattr(head, "regressor", None) or head.estimator
        self.pool = IPool(dim=1)
        self.mlp = _IntMlp(regressor, input_scale=input_scale, dtype=dtype)
        self.input_scale = float(input_scale)

    def forward(self, tokens: QTensor) -> QTensor:
        return self.mlp(self.pool(tokens).int_data.to(torch.int64))


class IEllipseHead(nn.Module):
    """``PupilEllipseHead``: pooled features concatenated with the anchor state.

    The anchor state is the one input to this graph that does not come from the previous
    op — the host holds ``z_t`` and supplies it — so it arrives as its own QTensor on
    whatever grid the host chose. ``ICat`` is what makes that legal: it aligns both
    operands onto the scale ``fc1`` declares before joining them. Concatenating two
    tensors on different grids and calling the result one tensor would silently rescale
    five of the columns.
    """

    def __init__(self, head: nn.Module, *, input_scale: float,
                 dtype: QuantDtype = INT8) -> None:
        super().__init__()
        self.condition_on_state = bool(head.condition_on_state)
        fc1_scale = float(head.regressor[0].act_scale)
        # Whoever lands on fc1's grid owns the requant, and when conditioning that is
        # ICat, not the mlp: giving the mlp its own bridge as well would rescale a
        # tensor that is already exactly where it belongs.
        self.mlp = _IntMlp(head.regressor, dtype=dtype,
                           input_scale=fc1_scale if self.condition_on_state else input_scale)
        self.cat = ICat(fc1_scale, dim=-1, dtype=dtype) if self.condition_on_state else None
        self.pool = IPool(dim=1)
        self.input_scale = float(input_scale)
        self.dtype = dtype

    def forward(self, tokens: QTensor, anchor_state: QTensor | None = None) -> QTensor:
        pooled = self.pool(tokens)
        if self.cat is None:
            return self.mlp(pooled.int_data.to(torch.int64))
        if anchor_state is None:
            zeros = pooled.int_data.new_zeros(pooled.int_data.shape[0], STATE_DIM)
            anchor_state = QTensor(zeros, scale=self.mlp.input_scale, zero_point=0.0,
                                   dtype=self.dtype)
        joined = self.cat([pooled, anchor_state])
        return self.mlp(joined.int_data.to(torch.int64))


__all__ = ["IEllipseHead", "IPooledMlpHead"]
