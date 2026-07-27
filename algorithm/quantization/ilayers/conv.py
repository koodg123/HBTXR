"""I-tier integer Conv2d — int8 weight × int8 activation, strided and/or padded.

Covers every conv the HBTXR graph contains: the non-overlapping patch embedding
(kernel == stride == patch), the 1x1 projections, and the 3x3/pad-1/stride-1 convs of
the mask and heatmap heads. ``IConv2d`` quantizes weight per output channel
(symmetric max-abs) and the input activation per-tensor, accumulates in the integer
domain, folds the activation zero-point as a ``- zp·Σw`` correction, and dequantizes
by ``s_x · s_w``.

Integer accumulation is done through ``F.conv2d`` on the integer operands cast to
float: for ViT patch dims the products stay exact within float32's 2^24 integer
range, and the result is checked bit-exact against the ``i_ops.int_conv2d`` golden.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from quantization.scheme import INT8, QuantDtype


def _quantize_conv_weight(weight: torch.Tensor, dtype: QuantDtype):
    """Per-output-channel symmetric int weight + scale [Cout]."""
    max_abs = weight.detach().abs().amax(dim=(1, 2, 3)).clamp_min(1e-8)     # [Cout]
    scale = max_abs / float(dtype.qmax)
    w_int = torch.round(weight / scale.view(-1, 1, 1, 1)).clamp(dtype.qmin, dtype.qmax)
    return w_int.to(torch.int32), scale


def _pad_pair(padding: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """``padding`` as ``(ph, pw)``; an int means the same amount on both axes.

    Deliberately a copy of ``i_ops._pad_pair`` rather than an import: no module under
    ``ilayers`` imports ``i_ops``, because ``i_ops`` is the golden these kernels are
    checked against and a reference that shares code with the thing it validates
    proves less. Eleven lines of duplication is the cheaper side of that trade.
    """
    if isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError(f"padding must be an int or a (ph, pw) pair, got {padding!r}")
        ph, pw = int(padding[0]), int(padding[1])
    else:
        ph = pw = int(padding)
    if ph < 0 or pw < 0:
        raise ValueError(f"padding must be non-negative, got {padding!r}")
    return ph, pw


def conv_padding_pair(conv: nn.Conv2d) -> tuple[int, int]:
    """A ``nn.Conv2d``'s padding as the symmetric ``(ph, pw)`` pair ``IConv2d`` stores.

    Normalises every spelling torch accepts: ``int``, ``(ph, pw)``, ``"valid"`` and
    ``"same"``. ``"same"`` is *not* always symmetric — torch splits a total padding of
    ``dilation·(k-1)`` as ``left = total // 2``, so an even kernel gets one more column
    on the right than on the left. ``IConv2d`` has no way to express that, so it is
    rejected here rather than silently rounded into a conv computing a shifted output.
    """
    padding = conv.padding
    if not isinstance(padding, str):
        return _pad_pair(padding)
    if padding == "valid":
        return 0, 0
    if padding != "same":
        raise ValueError(f"unknown conv padding mode {padding!r}")
    dilation = conv.dilation if isinstance(conv.dilation, (tuple, list)) else (conv.dilation,) * 2
    pair: list[int] = []
    for k, d in zip(conv.kernel_size, dilation):
        total = int(d) * (int(k) - 1)
        if total % 2:
            raise ValueError(
                f"padding='same' with kernel_size={tuple(conv.kernel_size)} pads asymmetrically "
                f"(total {total} splits {total // 2}/{total - total // 2}); IConv2d only stores "
                f"symmetric padding")
        pair.append(total // 2)
    return pair[0], pair[1]


class IConv2d(nn.Module):
    """Deployable integer Conv2d; reproduces a fake-quant conv bit-exactly."""

    def __init__(self, weight_int, weight_scale, act_scale, act_dtype, bias, *, stride,
                 padding: int | tuple[int, int] = 0, act_zero_point=0.0):
        super().__init__()
        self.register_buffer("weight_int", weight_int.to(torch.int32))          # [Cout, Cin, kh, kw]
        self.register_buffer("weight_scale", torch.as_tensor(weight_scale, dtype=torch.float32).reshape(-1))
        self.register_buffer("wsum", weight_int.to(torch.int64).sum(dim=(1, 2, 3)))  # [Cout]
        self.act_scale = float(act_scale)
        self.act_zero_point = int(round(float(act_zero_point)))
        self.act_dtype = act_dtype
        # A BUFFER, not a plain attribute: padding changes the output shape and the
        # function computed, so a state_dict round-trip that dropped it would rebuild a
        # conv that quietly computes something else. ``stride`` is a plain int for the
        # same reason it always was — it is not restorable either, and is fixed next.
        self.register_buffer("padding_hw", torch.tensor(_pad_pair(padding), dtype=torch.int64))
        self.register_buffer("stride_hw", torch.tensor(int(stride), dtype=torch.int64))
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().float())
        else:
            self.bias = None

    @property
    def padding(self) -> tuple[int, int]:
        """``(ph, pw)`` as python ints — the buffer is the storage, this is the reader."""
        return tuple(int(v) for v in self.padding_hw)

    @property
    def stride(self) -> int:
        return int(self.stride_hw)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, *, act_scale, act_dtype: QuantDtype = INT8,
                  weight_dtype: QuantDtype = INT8, act_zero_point=0.0) -> "IConv2d":
        stride_pair = conv.stride if isinstance(conv.stride, (tuple, list)) else (conv.stride,)
        strides = {int(s) for s in stride_pair}
        if len(strides) != 1:
            raise ValueError(f"IConv2d stores one stride; conv has stride={tuple(stride_pair)}")
        weight_int, weight_scale = _quantize_conv_weight(conv.weight, weight_dtype)
        return cls(weight_int, weight_scale, act_scale, act_dtype, conv.bias,
                   stride=strides.pop(), padding=conv_padding_pair(conv),
                   act_zero_point=act_zero_point)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int = torch.round(x / self.act_scale + self.act_zero_point).clamp(
            self.act_dtype.qmin, self.act_dtype.qmax)
        ph, pw = (int(v) for v in self.padding_hw)
        if ph or pw:
            # Pad the INTEGER activation with the zero-point, not with 0, and only then
            # convolve with padding=0. Handing ``padding=`` to F.conv2d instead would
            # zero-pad the integer grid, and 0 is not a real zero on an asymmetric
            # activation grid — it is the real value -zp*s_x. The uniform
            # ``- zp·Σw`` correction below assumes every window holds Cin·kh·kw real
            # taps, so border windows would be corrected for taps that were never
            # there: measured 0.21 max abs error on a 3x3/pad-1 conv with zp=17, vs
            # 1.8e-07 here. Padding with zp keeps the correction exactly right, since
            # a padded tap contributes (zp - zp)·w = 0.
            x_int = F.pad(x_int, (pw, pw, ph, ph), value=float(self.act_zero_point))
        acc = F.conv2d(x_int.float(), self.weight_int.float(), stride=int(self.stride_hw)).round().to(torch.int64)
        if self.act_zero_point != 0:
            acc = acc - self.act_zero_point * self.wsum.view(1, -1, 1, 1)
        y = acc.to(torch.float32) * (self.act_scale * self.weight_scale.view(1, -1, 1, 1))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


__all__ = ["IConv2d", "conv_padding_pair"]
