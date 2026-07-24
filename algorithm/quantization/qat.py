"""Quantization-aware training (QAT) preparation for the HBTXR quantized ViT.

QAT reuses the existing training stack: after inserting fake quantizers and
initializing their scales from calibration, the model is an ordinary ``nn.Module``
whose ``QuantLinear`` weights train normally (gradients flow through the weight/act
fake-quant STE, verified in Q2). ``prepare_qat`` returns that QAT-ready model; the
caller then fine-tunes it with ``engine.train.Trainer`` (or ``engine.distill``) and
saves the result like any other checkpoint.

Scales are fixed from calibration during QAT (standard fixed-scale QAT); only the
underlying weights are updated. Learnable-scale (LSQ) is a later refinement.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from torch import nn

from quantization.calibrate import calibrate, calibrate_weights
from quantization.insert import QuantConfig, QuantLinear, insert_fake_quant

ForwardFn = Callable[[nn.Module, Any], Any]


def prepare_qat(
    model: nn.Module,
    *,
    config: QuantConfig | None = None,
    calib_batches: Iterable[Any] | None = None,
    forward_fn: ForwardFn | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, QuantLinear]]:
    """Insert fake quantizers and initialize their scales for QAT fine-tuning.

    If ``calib_batches`` is given, both weight and activation scales are calibrated;
    otherwise only the (static) weight scales are set. Returns ``(model, registry)``.
    Fine-tune the returned model with ``engine.train.Trainer``.
    """
    model, registry = insert_fake_quant(model, config)
    if calib_batches is not None:
        calibrate(model, calib_batches, forward_fn=forward_fn, device=device)
    else:
        calibrate_weights(model, registry)
    return model, registry


__all__ = ["prepare_qat"]
