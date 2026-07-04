"""Simple PyTorch fake-quant utilities for ImageNet accuracy experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class FakeQuantConfig:
    weight_bits: int
    activation_bits: int
    quantize_weights: bool = True
    quantize_activations: bool = True
    symmetric: bool = True


def fake_quant_tensor(tensor: torch.Tensor, bits: int, *, symmetric: bool = True) -> torch.Tensor:
    if bits >= 32:
        return tensor
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        scale = tensor.detach().abs().max().clamp(min=1e-12) / qmax
        return torch.clamp(torch.round(tensor / scale), -qmax - 1, qmax) * scale
    qmin, qmax = 0, (1 << bits) - 1
    min_val = tensor.detach().min()
    max_val = tensor.detach().max()
    scale = (max_val - min_val).clamp(min=1e-12) / float(qmax - qmin)
    zero = torch.round(qmin - min_val / scale)
    return (torch.clamp(torch.round(tensor / scale + zero), qmin, qmax) - zero) * scale


def quantize_model_weights(model: nn.Module, config: FakeQuantConfig) -> None:
    if not config.quantize_weights or config.weight_bits >= 32:
        return
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.copy_(fake_quant_tensor(module.weight, config.weight_bits, symmetric=config.symmetric))
                if module.bias is not None:
                    module.bias.copy_(fake_quant_tensor(module.bias, config.weight_bits, symmetric=config.symmetric))


@contextmanager
def activation_fake_quant_hooks(model: nn.Module, config: FakeQuantConfig) -> Iterator[None]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if config.quantize_activations and config.activation_bits < 32:
        target_types = (nn.Linear, nn.Conv2d, nn.GELU, nn.LayerNorm, nn.Softmax)

        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            if torch.is_tensor(output):
                return fake_quant_tensor(output, config.activation_bits, symmetric=config.symmetric)
            return output

        for module in model.modules():
            if isinstance(module, target_types):
                handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def precision_to_config(precision: str) -> FakeQuantConfig | None:
    normalized = precision.lower()
    if normalized == "fp32":
        return None
    if normalized in {"int8", "8bit", "8-bit", "a8w8"}:
        return FakeQuantConfig(weight_bits=8, activation_bits=8)
    if normalized in {"int4", "4bit", "4-bit", "a4w4", "w4a4"}:
        return FakeQuantConfig(weight_bits=4, activation_bits=4)
    if normalized in {"w4a8", "a8w4", "weight4activation8", "weight-4-activation-8"}:
        return FakeQuantConfig(weight_bits=4, activation_bits=8)
    raise ValueError(f"Unsupported precision: {precision}")
