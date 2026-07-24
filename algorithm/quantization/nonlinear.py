"""HW-friendly integer LUT nonlinear modules for the HBTXR ViT.

``GeLULUT`` replaces ``nn.GELU`` with the HG-PIPE integer table kernel: quantize the
input, ``cursor = clamp((x_int + b) >> s, 0, bound)``, table lookup, dequant. A
straight-through estimator keeps it trainable under QAT.

``calibrate_gelu_luts`` observes each ``nn.GELU`` input over a few calibration
batches, builds a per-site LUT (lut_calibrate.build_gelu_lut), and swaps the GELU
for a calibrated ``GeLULUT`` in place. This is the pointwise piece of the
HW-friendly nonlinear path; Softmax / LayerNorm-rsqrt LUTs are follow-on work.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn

from quantization.lut_calibrate import build_gelu_lut
from quantization.scheme import INT8, QuantDtype

ForwardFn = Callable[[nn.Module, Any], Any]


class GeLULUT(nn.Module):
    """Integer LUT GeLU (fake-quant with STE)."""

    def __init__(self, scalars: list[int], table: list[int], *, input_scale: float, output_scale: float) -> None:
        super().__init__()
        self.b, self.s, self.bound = (int(v) for v in scalars)
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.register_buffer("table", torch.as_tensor(list(table), dtype=torch.int64))

    def _lut(self, x: torch.Tensor) -> torch.Tensor:
        x_int = torch.round(x / self.input_scale).to(torch.int64)
        cursor = ((x_int + self.b) >> self.s).clamp(0, self.bound)
        return self.table[cursor].to(dtype=x.dtype) * self.output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._lut(x)
        return x + (y - x).detach()  # STE: forward = LUT GeLU, backward = identity


def _set_submodule(model: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parent = model
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def calibrate_gelu_luts(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    entries: int = 256,
    input_dtype: QuantDtype = INT8,
    output_dtype: QuantDtype = INT8,
) -> dict[str, GeLULUT]:
    """Observe each nn.GELU input, build a LUT, and replace it with a GeLULUT.

    Returns a mapping of module path -> inserted GeLULUT.
    """
    gelus = {name: module for name, module in model.named_modules() if isinstance(module, nn.GELU)}
    if not gelus:
        return {}

    samples: dict[str, list[torch.Tensor]] = {name: [] for name in gelus}

    def make_hook(name: str):
        def hook(_module, inputs):
            if inputs:
                samples[name].append(inputs[0].detach().reshape(-1).float().cpu())
        return hook

    handles = [gelus[name].register_forward_pre_hook(make_hook(name)) for name in gelus]
    run = forward_fn or (lambda m, b: m(b))
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                run(model, batch)
    finally:
        for handle in handles:
            handle.remove()

    inserted: dict[str, GeLULUT] = {}
    for name in gelus:
        collected = samples[name]
        if not collected:
            continue
        activations = torch.cat(collected).numpy()
        payload = build_gelu_lut(activations, entries=entries, input_dtype=input_dtype, output_dtype=output_dtype)
        lut = GeLULUT(payload["scalars"], payload["table"], input_scale=payload["input_scale"], output_scale=payload["output_scale"])
        _set_submodule(model, name, lut)
        inserted[name] = lut
    return inserted


__all__ = ["GeLULUT", "calibrate_gelu_luts"]
