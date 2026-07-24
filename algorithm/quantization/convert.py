"""Model conversion: float -> Q (fake-quant insertion) and observe -> replace.

- ``insert_fake_quant``: replace every eligible ``nn.Linear`` with a ``QLinear``.
- ``collect_quantizers``: gather the inserted AffineFakeQuantizer modules.
- ``calibrate_{gelu,layernorm,softmax}_luts``: observe each nonlinear op's input over
  a few batches, build a calibrated LUT, and swap the float op for its Q module.
- ``convert_to_integer``: Q -> I deployment swap — replace each calibrated ``QLinear``
  with an ``ILinear`` (true integer matmul, float I/O drop-in). The nonlinear LUT
  modules (QGeLU/QLayerNorm/QSoftmax) already perform integer table lookups, so a
  converted model computes every matmul/conv in the integer domain.

(Float->Q for linear; the nonlinear calibrators fold observe+build+swap.)
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn

from quantization.lut_calibrate import build_function_lut, build_gelu_lut
from quantization.q_ops import AffineFakeQuantizer
from quantization.qlayers.linear import QLinear, QuantConfig
from quantization.qlayers.nonlinear import QGeLU, QLayerNorm, QSoftmax
from quantization.scheme import INT8, UINT8, QuantDtype
from quantization.spec import QuantScheme

ForwardFn = Callable[[nn.Module, Any], Any]


def insert_fake_quant(
    model: nn.Module,
    config: QuantConfig | None = None,
    *,
    scheme: QuantScheme | None = None,
) -> tuple[nn.Module, dict[str, QLinear]]:
    """Replace every eligible ``nn.Linear`` in ``model`` with a ``QLinear``.

    With a ``QuantScheme`` (Part B), each Linear's ``(weight, activation)`` spec is
    resolved *per module path* so ``overrides`` give mixed-precision / mixed-config
    per layer; otherwise a single ``QuantConfig`` is applied uniformly.
    """
    if scheme is not None:
        skip = scheme.skip

        def config_for(full: str) -> QuantConfig:
            weight_spec, act_spec = scheme.resolve(full)
            return QuantConfig(weight_spec=weight_spec, act_spec=act_spec, skip=skip)
    else:
        cfg = config or QuantConfig()
        skip = cfg.skip

        def config_for(full: str) -> QuantConfig:
            return cfg

    registry: dict[str, QLinear] = {}
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and not isinstance(child, QLinear):
                if any(token in full for token in skip):
                    continue
                quant = QLinear(child, config_for(full))
                setattr(module, child_name, quant)
                registry[full] = quant
    return model, registry


def collect_quantizers(model: nn.Module) -> dict[str, AffineFakeQuantizer]:
    """All AffineFakeQuantizer modules keyed by path (weight and activation)."""
    return {name: m for name, m in model.named_modules() if isinstance(m, AffineFakeQuantizer)}


def convert_to_integer(model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    """Q -> I: replace each calibrated ``QLinear`` with an ``ILinear`` (in place).

    ``ILinear`` has a float I/O drop-in interface but computes the matmul in the
    integer domain (int8 weight × int8 activation, int32 accumulate, dequant), so the
    converted model runs through the ordinary forward while every Linear is integer —
    and reproduces the fake-quant model bit-exactly. Call after PTQ/QAT calibration.
    """
    from quantization.ilayers.linear import ILinear

    replaced: dict[str, Any] = {}
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, QLinear):
                integer = ILinear.from_qlinear(child)
                setattr(module, child_name, integer)
                replaced[full] = integer
    return model, replaced


def _set_submodule(model: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parent = model
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _observe(model: nn.Module, modules: dict[str, nn.Module], batches: Iterable[Any], forward_fn: ForwardFn | None, capture) -> dict[str, list[torch.Tensor]]:
    samples: dict[str, list[torch.Tensor]] = {name: [] for name in modules}

    def make_hook(name: str):
        def hook(_module, inputs):
            if inputs:
                capture(name, inputs[0], samples)
        return hook

    handles = [modules[name].register_forward_pre_hook(make_hook(name)) for name in modules]
    run = forward_fn or (lambda m, b: m(b))
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                run(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return samples


def calibrate_gelu_luts(model, batches, *, forward_fn=None, entries=256, input_dtype: QuantDtype = INT8, output_dtype: QuantDtype = INT8) -> dict[str, QGeLU]:
    """Observe each nn.GELU input, build a LUT, and replace it with a QGeLU."""
    gelus = {name: m for name, m in model.named_modules() if isinstance(m, nn.GELU)}
    if not gelus:
        return {}
    samples = _observe(model, gelus, batches, forward_fn, lambda n, x, s: s[n].append(x.detach().reshape(-1).float().cpu()))
    inserted: dict[str, QGeLU] = {}
    for name in gelus:
        if not samples[name]:
            continue
        payload = build_gelu_lut(torch.cat(samples[name]).numpy(), entries=entries, input_dtype=input_dtype, output_dtype=output_dtype)
        module = QGeLU(payload["scalars"], payload["table"], input_scale=payload["input_scale"], output_scale=payload["output_scale"])
        _set_submodule(model, name, module)
        inserted[name] = module
    return inserted


def calibrate_layernorm_luts(model, batches, *, forward_fn=None, entries=128) -> dict[str, QLayerNorm]:
    """Observe each nn.LayerNorm input variance, build a rsqrt LUT, and swap it in."""
    norms = {name: m for name, m in model.named_modules() if isinstance(m, nn.LayerNorm)}
    if not norms:
        return {}
    samples = _observe(model, norms, batches, forward_fn, lambda n, x, s: s[n].append(x.detach().float().var(dim=-1, unbiased=False).reshape(-1).cpu()))
    inserted: dict[str, QLayerNorm] = {}
    for name, norm in norms.items():
        if not samples[name]:
            continue
        eps = float(norm.eps)
        payload = build_function_lut(
            torch.cat(samples[name]).numpy(),
            lambda v, e=eps: 1.0 / (np.sqrt(np.maximum(v, 0.0) + e)),
            entries=entries,
            input_dtype=UINT8,
            output_dtype=INT8,
        )
        module = QLayerNorm(norm, payload)
        _set_submodule(model, name, module)
        inserted[name] = module
    return inserted


def calibrate_softmax_luts(model, batches, *, forward_fn=None, exp_entries=128, recip_entries=128) -> dict[str, QSoftmax]:
    """Observe each nn.Softmax input, build exp + reciprocal LUTs, and swap it in."""
    softmaxes = {name: m for name, m in model.named_modules() if isinstance(m, nn.Softmax)}
    if not softmaxes:
        return {}
    deltas: dict[str, list[torch.Tensor]] = {name: [] for name in softmaxes}
    sums: dict[str, list[torch.Tensor]] = {name: [] for name in softmaxes}

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0].detach().float()
            delta = x.max(dim=-1, keepdim=True).values - x
            deltas[name].append(delta.reshape(-1).cpu())
            sums[name].append(torch.exp(-delta).sum(dim=-1).reshape(-1).cpu())
        return hook

    handles = [softmaxes[name].register_forward_pre_hook(make_hook(name)) for name in softmaxes]
    run = forward_fn or (lambda m, b: m(b))
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                run(model, batch)
    finally:
        for handle in handles:
            handle.remove()

    inserted: dict[str, QSoftmax] = {}
    for name, softmax in softmaxes.items():
        if not deltas[name] or not sums[name]:
            continue
        exp_payload = build_function_lut(torch.cat(deltas[name]).numpy(), lambda d: np.exp(-np.maximum(d, 0.0)), entries=exp_entries, input_dtype=UINT8, output_dtype=UINT8)
        recip_payload = build_function_lut(torch.cat(sums[name]).numpy(), lambda s: 1.0 / np.maximum(s, 1e-8), entries=recip_entries, input_dtype=UINT8, output_dtype=UINT8)
        module = QSoftmax(exp_payload, recip_payload, dim=softmax.dim if softmax.dim is not None else -1)
        _set_submodule(model, name, module)
        inserted[name] = module
    return inserted


__all__ = [
    "insert_fake_quant",
    "collect_quantizers",
    "convert_to_integer",
    "calibrate_gelu_luts",
    "calibrate_layernorm_luts",
    "calibrate_softmax_luts",
]
