from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DCNv2(nn.Module):
    """DCNv2-compatible module implemented with torchvision deform_conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        expected_offset = 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        expected_mask = self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        if offset.shape[1] != expected_offset:
            raise ValueError(f"offset channels must be {expected_offset}, got {offset.shape[1]}")
        if mask.shape[1] != expected_mask:
            raise ValueError(f"mask channels must be {expected_mask}, got {mask.shape[1]}")

        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class DCN(DCNv2):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups: int = 1,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            deformable_groups=deformable_groups,
        )

        channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return super().forward(input, offset, mask)
