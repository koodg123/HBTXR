"""I-tier integer Conv2d (strided PatchEmbed) — int8 weight × int8 activation.

The HBTXR patch embedding is a non-overlapping ``nn.Conv2d`` (kernel == stride ==
patch). ``IConv2d`` quantizes weight per output channel (symmetric max-abs) and the
input activation per-tensor, accumulates in the integer domain, folds the activation
zero-point as a ``- zp·Σw`` correction, and dequantizes by ``s_x · s_w``.

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


class IConv2d(nn.Module):
    """Deployable integer Conv2d; reproduces a fake-quant conv bit-exactly."""

    def __init__(self, weight_int, weight_scale, act_scale, act_dtype, bias, *, stride, act_zero_point=0.0):
        super().__init__()
        self.register_buffer("weight_int", weight_int.to(torch.int32))          # [Cout, Cin, kh, kw]
        self.register_buffer("weight_scale", torch.as_tensor(weight_scale, dtype=torch.float32).reshape(-1))
        self.register_buffer("wsum", weight_int.to(torch.int64).sum(dim=(1, 2, 3)))  # [Cout]
        self.act_scale = float(act_scale)
        self.act_zero_point = int(round(float(act_zero_point)))
        self.act_dtype = act_dtype
        self.stride = int(stride)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().float())
        else:
            self.bias = None

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, *, act_scale, act_dtype: QuantDtype = INT8,
                  weight_dtype: QuantDtype = INT8, act_zero_point=0.0) -> "IConv2d":
        stride = conv.stride[0] if isinstance(conv.stride, (tuple, list)) else conv.stride
        weight_int, weight_scale = _quantize_conv_weight(conv.weight, weight_dtype)
        return cls(weight_int, weight_scale, act_scale, act_dtype, conv.bias, stride=stride,
                   act_zero_point=act_zero_point)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int = torch.round(x / self.act_scale + self.act_zero_point).clamp(
            self.act_dtype.qmin, self.act_dtype.qmax)
        acc = F.conv2d(x_int.float(), self.weight_int.float(), stride=self.stride).round().to(torch.int64)
        if self.act_zero_point != 0:
            acc = acc - self.act_zero_point * self.wsum.view(1, -1, 1, 1)
        y = acc.to(torch.float32) * (self.act_scale * self.weight_scale.view(1, -1, 1, 1))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


__all__ = ["IConv2d"]
