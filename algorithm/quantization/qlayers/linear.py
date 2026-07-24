"""Q-tier Linear: nn.Linear with fake-quantized weight + input activation (QAT-ready).

``QuantConfig`` (weight/act dtype, skip filter) also drives the fake-quant insertion
(convert.insert_fake_quant); it lives here with ``QLinear`` to avoid an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from quantization.q_ops import AffineFakeQuantizer
from quantization.scheme import INT8, QuantDtype


@dataclass
class QuantConfig:
    weight_dtype: QuantDtype = INT8
    act_dtype: QuantDtype = INT8
    skip: tuple[str, ...] = field(default_factory=tuple)  # name substrings to leave in fp


class QLinear(nn.Module):
    """nn.Linear with fake-quantized weight and input activation."""

    def __init__(self, linear: nn.Linear, config: QuantConfig) -> None:
        super().__init__()
        self.linear = linear
        self.weight_fq = AffineFakeQuantizer(config.weight_dtype)
        self.act_fq = AffineFakeQuantizer(config.act_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fq(x)
        weight = self.weight_fq(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


__all__ = ["QuantConfig", "QLinear"]
