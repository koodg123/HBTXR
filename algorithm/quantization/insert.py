"""Insert fake quantizers into the HBTXR ViT models (QAT / PTQ graph).

Walks a built HBTXR model (models.frame / event / hybrid) and wraps every
``nn.Linear`` (attention qkv/proj, MLP fc1/fc2, regression heads) with a
``QuantLinear`` that fake-quantizes the weight and the input activation. This is
the bulk of the ViT compute (the RMU/SMU workload) and the core of QAT.

The HW-friendly integer nonlinear operators (LayerNorm rsqrt / Softmax / GeLU LUT)
need calibrated tables and are substituted in a later phase (calibration + PTQ);
here they are left in floating point, with activation quantizers on the Linear
boundaries around them.

``insert_fake_quant`` returns the (in-place modified) model and a name->QuantLinear
registry so a calibration pass can observe ranges and set the quantizer scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from quantization.fake_quant import AffineFakeQuantizer
from quantization.scheme import INT8, QuantDtype


@dataclass
class QuantConfig:
    weight_dtype: QuantDtype = INT8
    act_dtype: QuantDtype = INT8
    skip: tuple[str, ...] = field(default_factory=tuple)  # name substrings to leave in fp


class QuantLinear(nn.Module):
    """nn.Linear with fake-quantized weight and input activation (QAT-ready)."""

    def __init__(self, linear: nn.Linear, config: QuantConfig) -> None:
        super().__init__()
        self.linear = linear
        self.weight_fq = AffineFakeQuantizer(config.weight_dtype)
        self.act_fq = AffineFakeQuantizer(config.act_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fq(x)
        weight = self.weight_fq(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


def insert_fake_quant(model: nn.Module, config: QuantConfig | None = None) -> tuple[nn.Module, dict[str, QuantLinear]]:
    """Replace every eligible ``nn.Linear`` in ``model`` with a ``QuantLinear``.

    Returns ``(model, registry)`` where registry maps the module path to the
    inserted ``QuantLinear`` (used by calibration to set scales).
    """
    cfg = config or QuantConfig()
    registry: dict[str, QuantLinear] = {}
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
                if any(token in full for token in cfg.skip):
                    continue
                quant = QuantLinear(child, cfg)
                setattr(module, child_name, quant)
                registry[full] = quant
    return model, registry


def collect_quantizers(model: nn.Module) -> dict[str, AffineFakeQuantizer]:
    """All AffineFakeQuantizer modules keyed by path (weight and activation)."""
    return {name: m for name, m in model.named_modules() if isinstance(m, AffineFakeQuantizer)}


__all__ = ["QuantConfig", "QuantLinear", "insert_fake_quant", "collect_quantizers"]
