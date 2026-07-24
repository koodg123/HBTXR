"""Calibration + post-training quantization (PTQ) for the HBTXR quantized ViT.

After ``insert_fake_quant`` wraps the Linear layers, calibration sets each
quantizer's scale from observed ranges:

- weights: static max-abs of each ``QuantLinear.linear.weight``.
- activations: max-abs of each ``QuantLinear`` input over a few calibration
  batches, captured with forward-pre-hooks.

``post_training_quantize`` is the end-to-end PTQ entry: insert -> calibrate on a
handful of batches -> return the quantized model. A ``forward_fn`` adapts the drive
per modality (direct ``model(image)`` by default; hybrid uses search/track steps).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn

from quantization.insert import QuantConfig, QuantLinear, insert_fake_quant
from quantization.observer import MinMaxObserver

ForwardFn = Callable[[nn.Module, Any], Any]


def _registry(model: nn.Module) -> dict[str, QuantLinear]:
    return {name: m for name, m in model.named_modules() if isinstance(m, QuantLinear)}


def calibrate_weights(model: nn.Module, registry: dict[str, QuantLinear] | None = None) -> None:
    """Set each weight quantizer scale from the static weight max-abs."""
    registry = registry or _registry(model)
    for quant in registry.values():
        observer = MinMaxObserver(quant.weight_fq.dtype)
        observer.observe(quant.linear.weight)
        quant.weight_fq.set_qparams(*observer.qparams())


def calibrate_activations(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    registry: dict[str, QuantLinear] | None = None,
    device: str = "cpu",
) -> None:
    """Set each activation quantizer scale from observed inputs over ``batches``."""
    registry = registry or _registry(model)
    observers = {name: MinMaxObserver(quant.act_fq.dtype) for name, quant in registry.items()}

    def make_hook(observer: MinMaxObserver):
        def hook(_module, inputs):
            if inputs:
                observer.observe(inputs[0])
        return hook

    handles = [quant.register_forward_pre_hook(make_hook(observers[name])) for name, quant in registry.items()]
    run = forward_fn or (lambda m, b: m(b))
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                run(model, batch)
    finally:
        for handle in handles:
            handle.remove()

    for name, quant in registry.items():
        quant.act_fq.set_qparams(*observers[name].qparams())


def calibrate(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Calibrate both weight and activation quantizers in ``model``."""
    registry = _registry(model)
    if not registry:
        raise ValueError("no QuantLinear modules found; call insert_fake_quant first")
    calibrate_weights(model, registry)
    calibrate_activations(model, batches, forward_fn=forward_fn, registry=registry, device=device)
    return model


def post_training_quantize(
    model: nn.Module,
    calib_batches: Iterable[Any],
    *,
    config: QuantConfig | None = None,
    forward_fn: ForwardFn | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, QuantLinear]]:
    """Insert fake quantizers and calibrate them (PTQ). Returns (model, registry)."""
    model, registry = insert_fake_quant(model, config)
    calibrate(model, calib_batches, forward_fn=forward_fn, device=device)
    return model, registry


__all__ = [
    "calibrate_weights",
    "calibrate_activations",
    "calibrate",
    "post_training_quantize",
]
