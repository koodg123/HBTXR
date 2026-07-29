"""Model conversion: float -> Q (fake-quant insertion) and Q -> I (integer deployment).

Float -> Q
    - ``insert_fake_quant``: replace every eligible ``nn.Linear`` with a ``QLinear``.
    - ``collect_quantizers``: gather the inserted AffineFakeQuantizer modules.
    - ``calibrate_{gelu,layernorm,softmax}_luts``: observe each nonlinear op's input over
      a few batches, build a calibrated LUT, and swap the float op for its Q module.
      These stay at the Q tier — QLayerNorm/QSoftmax still reduce mean/var/row-sum in
      FLOAT and only look the rsqrt / exp / reciprocal up in a table.

Q -> I (deployment)
    - ``calibrate_int_{layernorms,softmaxes,gelus,convs}``: the I-tier twins of the
      calibrators above. Same observe -> build -> swap shape, but they install
      ``ILayerNorm`` / ``ISoftmax`` / ``IGeLU`` / ``IConv2d``, whose *whole* datapath is
      integer (integer mean, integer variance accumulation, integer row max and row sum).
    - ``convert_to_integer``: LINEAR ONLY (``QLinear`` -> ``ILinear``); kept unchanged
      for backward compatibility.
    - ``convert_model_to_integer``: the honest whole-graph entry point. Runs every stage
      above and returns a ``ConversionReport`` that names what became integer AND what
      is still float, so a partial conversion cannot be mistaken for a total one.

Nothing here silently skips a module: anything the I tier cannot express (a dilated or
grouped conv, a LayerNorm the calibration batches never reached, a Linear that was never
fake-quantized) stays float and is reported in ``ConversionReport.left_float``.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn

from quantization.ilayers.conv import IConv2d, conv_padding_pair
from quantization.ilayers.layernorm import ILayerNorm
from quantization.ilayers.linear import ILinear
from quantization.ilayers.matmul import IMatMul
from quantization.ilayers.nonlinear import IGeLU
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.softmax import ISoftmax
from quantization.ilayers.tensor_ops import IAdd, ICat, IPool
from models.blocks.seams import Add, MatMul
from quantization.int_calibrate import calibrate_int_layernorm
from quantization.int_calibrate_softmax import (
    build_softmax_int_payload,
    build_softmax_int_payload_from_qsoftmax,
)
from quantization.lut_calibrate import build_function_lut, build_gelu_lut
from quantization.q_ops import AffineFakeQuantizer
from quantization.qlayers.linear import QLinear, QuantConfig
from quantization.qlayers.nonlinear import QGeLU, QLayerNorm, QSoftmax
from quantization.scheme import INT8, UINT8, QuantDtype
from quantization.spec import QuantScheme

ForwardFn = Callable[[nn.Module, Any], Any]
CaptureFn = Callable[[str, torch.Tensor, dict[str, list[torch.Tensor]]], None]

# Default cap on the calibration rows kept per module. The fitted rsqrt index and the
# softmax envelope are both distribution statistics — a few thousand rows already pin
# them — while a real HBTXR run has millions of tokens, so an uncapped observer is a
# memory hazard for no accuracy.
DEFAULT_MAX_ROWS = 16384


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


class IAddFloatIO(nn.Module):
    """``IAdd`` wearing the float input/output port the swapped-in graph runs on.

    ``IAdd`` aligns two ``QTensor`` operands onto a common output scale — that scale
    alignment is the whole point of it, and it is what a residual join needs. But the
    converted model still moves float tensors between modules, so a drop-in replacement
    for ``models.blocks.seams.Add`` has to quantize at its two input ports and dequantize
    at its output port, exactly as ``IGeLUFloatIO`` does for ``IGeLU``.

    The two operands genuinely have different scales — one is the residual stream, the
    other a sublayer output — which is why each carries its own observed scale rather
    than sharing one.
    """

    def __init__(self, kernel: IAdd, scale_a: float, scale_b: float, *,
                 in_dtype: QuantDtype = INT8) -> None:
        super().__init__()
        self.kernel = kernel
        self.scale_a = float(scale_a)
        self.scale_b = float(scale_b)
        self.in_dtype = in_dtype

    @property
    def scale_out(self) -> float:
        return float(self.kernel.scale_out)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        qa = QTensor.quantize(a, self.scale_a, 0.0, self.in_dtype)
        qb = QTensor.quantize(b, self.scale_b, 0.0, self.in_dtype)
        return self.kernel(qa, qb).dequantize()


def convert_to_integer(model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    """Q -> I for the **Linear layers only**: each ``QLinear`` becomes an ``ILinear``.

    This converts NOTHING else. GeLU, LayerNorm, Softmax, the attention matmuls, the
    residual adds and the patch-embed conv are left exactly as they were — if they are
    Q-tier LUT modules they still reduce in float, and if they are plain ``nn.GELU`` /
    ``nn.LayerNorm`` / ``nn.Softmax`` they are not quantized at all. Use
    ``convert_model_to_integer`` for the whole-graph conversion and an explicit report
    of what is still float.

    ``ILinear`` has a float I/O drop-in interface but computes the matmul in the integer
    domain (int8 weight × int8 activation, int32 accumulate, dequant), so the converted
    model runs through the ordinary forward and reproduces the fake-quant model
    bit-exactly. Call after PTQ/QAT calibration.
    """
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


def _observe_rows(
    model: nn.Module,
    modules: dict[str, nn.Module],
    batches: Iterable[Any],
    forward_fn: ForwardFn | None,
    *,
    reduce_dim: Callable[[nn.Module], int] | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, list[torch.Tensor]]:
    """Observe each module's input as ``[rows, width]``, capped at ``max_rows`` rows.

    ``reduce_dim`` names the axis the op reduces over (softmax's ``dim``); it is moved
    to the trailing position so the calibrators — which all normalize/reduce over the
    last axis — see the same rows the deployed kernel will. Rows are taken in arrival
    order, never sampled at random, so a conversion is reproducible from a seed.
    """
    counts: dict[str, int] = {name: 0 for name in modules}

    def capture(name: str, x: torch.Tensor, store: dict[str, list[torch.Tensor]]) -> None:
        budget = max_rows - counts[name]
        if budget <= 0:
            return
        t = x.detach().float()
        axis = -1 if reduce_dim is None else int(reduce_dim(modules[name]))
        if axis not in (-1, t.dim() - 1):
            t = t.transpose(axis, -1)
        t = t.reshape(-1, t.shape[-1])
        keep = min(int(t.shape[0]), budget)
        store[name].append(t[:keep].cpu())
        counts[name] += keep

    return _observe(model, modules, batches, forward_fn, capture)


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


# --- I tier: fully-integer nonlinear conversion -------------------------------

class IGeLUFloatIO(nn.Module):
    """``IGeLU`` wearing the float input/output port every other I-tier module has.

    ``IGeLU`` is deliberately a pure ``QTensor -> QTensor`` kernel, so on its own it
    cannot be dropped into a forward that moves float tensors between modules.
    ``ILinear``, ``IConv2d``, ``ILayerNorm`` and ``ISoftmax`` all solve that internally
    by quantizing at their input port and dequantizing at their output port; this
    adapter gives ``IGeLU`` the same port. The arithmetic is untouched — the table
    lookup still runs on integers — only the transport between modules is float, which
    is the same boundary condition the rest of the converted graph runs under.
    """

    def __init__(self, kernel: IGeLU, *, in_dtype: QuantDtype = INT8) -> None:
        super().__init__()
        self.kernel = kernel
        self.in_dtype = in_dtype

    @property
    def input_scale(self) -> float:
        return float(self.kernel.input_scale)

    @property
    def output_scale(self) -> float:
        return float(self.kernel.output_scale)

    def forward(self, x: torch.Tensor | QTensor) -> torch.Tensor | QTensor:
        if isinstance(x, QTensor):
            return self.kernel(x)
        qt = QTensor.quantize(x, self.kernel.input_scale, 0.0, self.in_dtype)
        return self.kernel(qt).dequantize().to(x.dtype)


def calibrate_int_layernorms(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    entries: int = 256,
    segments: int = 2,
    clamp_bits: int = 8,
    input_dtype: QuantDtype = INT8,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, ILayerNorm]:
    """Observe every LayerNorm input and swap the op for a fully-integer ``ILayerNorm``.

    Accepts **both** tiers as input, because both are legitimate starting points:

    * ``nn.LayerNorm`` — the float model straight out of PTQ (Linear-only fake quant).
    * ``QLayerNorm`` — a model that already went through ``calibrate_layernorm_luts``.
      It is *converted*, not refused: ``QLayerNorm`` keeps the original module's
      ``weight`` / ``bias`` / ``eps`` / ``normalized_shape``, which is everything
      ``calibrate_int_layernorm`` needs, and its Q-tier rsqrt LUT is discarded on
      purpose — it is indexed by a float variance, so it cannot drive the integer
      kernel. The I-tier table is rebuilt from the observed variance distribution.

    Modules the calibration batches never reached keep their float implementation and
    are simply absent from the returned mapping (``convert_model_to_integer`` then
    reports them under ``left_float``).

    ``entries`` is the depth of EACH rsqrt table and ``segments`` (1 or 2, default 2)
    how many there are, so the default costs ``2 * entries`` int16 words per LayerNorm —
    the same convention ``recip_entries`` already uses for the softmax reciprocal. See
    ``int_calibrate`` for why two segments is the default and what it measures.
    """
    norms = {name: m for name, m in model.named_modules() if isinstance(m, (nn.LayerNorm, QLayerNorm))}
    if not norms:
        return {}
    samples = _observe_rows(model, norms, batches, forward_fn, max_rows=max_rows)
    inserted: dict[str, ILayerNorm] = {}
    for name, norm in norms.items():
        if not samples[name]:
            continue
        payload = calibrate_int_layernorm(
            norm,
            torch.cat(samples[name]),
            entries=entries,
            segments=segments,
            clamp_bits=clamp_bits,
            input_dtype=input_dtype,
        )
        module = ILayerNorm.from_payload(payload)
        _set_submodule(model, name, module)
        inserted[name] = module
    return inserted


def _softmax_dim(module: nn.Module) -> int:
    """The reduction axis of an ``nn.Softmax`` / ``QSoftmax`` (``dim=None`` means last)."""
    dim = getattr(module, "dim", None)
    return -1 if dim is None else int(dim)


def calibrate_int_softmaxes(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    exp_entries: int = 256,
    recip_entries: int = 256,
    output_bits: int = 8,
    max_tokens: int | None = None,
    input_dtype: QuantDtype = INT8,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, ISoftmax]:
    """Observe every softmax's logits and swap the op for a fully-integer ``ISoftmax``.

    Accepts ``nn.Softmax`` and ``QSoftmax`` for the same reason
    ``calibrate_int_layernorms`` accepts both LayerNorm tiers. A ``QSoftmax`` is
    converted through ``build_softmax_int_payload_from_qsoftmax``, which keeps that
    module's already-fixed input grid (``exp_in``) so the Q tier and the I tier see the
    same integer logits; its own uint8 exp/reciprocal tables are rebuilt, since they are
    far too coarse to accumulate over a whole row.

    ``max_tokens`` is the longest row the deployed op will ever see. It defaults to the
    calibration row length and sizes the reciprocal segments; raise it if inference can
    run on longer sequences than calibration did, or a flat long row will come out
    un-normalized (see ``int_calibrate_softmax``).
    """
    softmaxes = {name: m for name, m in model.named_modules() if isinstance(m, (nn.Softmax, QSoftmax))}
    if not softmaxes:
        return {}
    samples = _observe_rows(model, softmaxes, batches, forward_fn,
                            reduce_dim=_softmax_dim, max_rows=max_rows)
    inserted: dict[str, ISoftmax] = {}
    for name, softmax in softmaxes.items():
        if not samples[name]:
            continue
        logits = torch.cat(samples[name]).numpy()
        options: dict[str, Any] = {
            "input_dtype": input_dtype,
            "exp_entries": exp_entries,
            "recip_entries": recip_entries,
            "output_bits": output_bits,
            "max_tokens": max_tokens,
        }
        if isinstance(softmax, QSoftmax):
            payload = build_softmax_int_payload_from_qsoftmax(softmax, logits, **options)
        else:
            payload = build_softmax_int_payload(logits, **options)
        module = ISoftmax.from_payload(payload, dim=_softmax_dim(softmax), in_dtype=input_dtype)
        _set_submodule(model, name, module)
        inserted[name] = module
    return inserted


def calibrate_int_gelus(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    entries: int = 256,
    input_dtype: QuantDtype = INT8,
    output_dtype: QuantDtype = INT8,
) -> dict[str, IGeLUFloatIO]:
    """Swap every GeLU for the integer ``IGeLU`` lookup (behind a float I/O port).

    A ``QGeLU`` already carries a calibrated integer table and integer index scalars —
    the Q tier's only float step is the dequantize at the end — so it is reused verbatim
    via ``IGeLU.from_qgelu`` and needs no observation. A plain ``nn.GELU`` has no table,
    so its input is observed and a fresh LUT is built.
    """
    gelus = {name: m for name, m in model.named_modules() if isinstance(m, (nn.GELU, QGeLU))}
    if not gelus:
        return {}
    uncalibrated = {name: m for name, m in gelus.items() if not isinstance(m, QGeLU)}
    samples: dict[str, list[torch.Tensor]] = {}
    if uncalibrated:
        samples = _observe(model, uncalibrated, batches, forward_fn,
                           lambda n, x, s: s[n].append(x.detach().reshape(-1).float().cpu()))
    inserted: dict[str, IGeLUFloatIO] = {}
    for name, gelu in gelus.items():
        if isinstance(gelu, QGeLU):
            kernel = IGeLU.from_qgelu(gelu)
        else:
            if not samples.get(name):
                continue
            payload = build_gelu_lut(torch.cat(samples[name]).numpy(), entries=entries,
                                     input_dtype=input_dtype, output_dtype=output_dtype)
            kernel = IGeLU(payload["scalars"], payload["table"], input_scale=payload["input_scale"],
                           output_scale=payload["output_scale"], out_dtype=output_dtype)
        module = IGeLUFloatIO(kernel, in_dtype=input_dtype)
        _set_submodule(model, name, module)
        inserted[name] = module
    return inserted


def _observe_io(
    model: nn.Module,
    modules: dict[str, nn.Module],
    batches: Iterable[Any],
    forward_fn: ForwardFn | None,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Per-module max-abs of EACH positional input, and of the output.

    ``_observe`` only captures ``inputs[0]``, which is enough for a one-argument op. The
    seam modules take two operands with genuinely different ranges — a residual stream
    and a sublayer output, or Q and Kᵀ — so each needs its own scale, and ``IAdd`` also
    needs a scale for the sum it produces. Returns ``(per-input max-abs, output max-abs)``.
    """
    inputs_max: dict[str, list[float]] = {name: [] for name in modules}
    output_max: dict[str, float] = {name: 0.0 for name in modules}

    def make_hook(name: str):
        def hook(_module, args, output):
            observed = [float(a.detach().abs().max()) for a in args if torch.is_tensor(a)]
            if not inputs_max[name]:
                inputs_max[name] = observed
            else:
                inputs_max[name] = [max(o, n) for o, n in zip(inputs_max[name], observed)]
            if torch.is_tensor(output):
                output_max[name] = max(output_max[name], float(output.detach().abs().max()))
        return hook

    handles = [modules[name].register_forward_hook(make_hook(name)) for name in modules]
    run = forward_fn or (lambda m, b: m(b))
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                run(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return inputs_max, output_max


def _symmetric(max_abs: float, dtype: QuantDtype) -> float:
    return max(float(max_abs), 1e-8) / float(dtype.qmax)


def calibrate_int_matmuls(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    dtype: QuantDtype = INT8,
) -> dict[str, IMatMul]:
    """Swap every ``seams.MatMul`` for an ``IMatMul`` calibrated on its two operands.

    These are the activation×activation products of attention — ``Q @ Kᵀ`` and
    ``attn @ V`` — which have no weights, so neither operand's scale can come from a
    parameter: both must be observed. They were unreachable until the operators became
    modules, which is why ``IMatMul`` existed with no production caller.
    """
    targets = {n: m for n, m in model.named_modules() if isinstance(m, MatMul)}
    if not targets:
        return {}
    inputs_max, _out = _observe_io(model, targets, batches, forward_fn)
    inserted: dict[str, IMatMul] = {}
    for name in targets:
        observed = inputs_max[name]
        if len(observed) != 2:
            continue  # never driven by the calibration batches; stays float and is reported
        integer = IMatMul(_symmetric(observed[0], dtype), _symmetric(observed[1], dtype),
                          dtype_a=dtype, dtype_b=dtype)
        _set_submodule(model, name, integer)
        inserted[name] = integer
    return inserted


def calibrate_int_adds(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    dtype: QuantDtype = INT8,
) -> dict[str, IAddFloatIO]:
    """Swap every ``seams.Add`` for an ``IAdd`` behind a float port.

    A residual join is where two tensors on DIFFERENT scales meet, so it needs three
    observed scales: one per operand, and one for the sum — the output range is not the
    max of the inputs (they can reinforce), so it is observed rather than inferred.
    """
    targets = {n: m for n, m in model.named_modules() if isinstance(m, Add)}
    if not targets:
        return {}
    inputs_max, output_max = _observe_io(model, targets, batches, forward_fn)
    inserted: dict[str, IAddFloatIO] = {}
    for name in targets:
        observed = inputs_max[name]
        if len(observed) != 2 or output_max[name] <= 0.0:
            continue
        kernel = IAdd(_symmetric(output_max[name], dtype), dtype=dtype)
        wrapper = IAddFloatIO(kernel, _symmetric(observed[0], dtype),
                              _symmetric(observed[1], dtype), in_dtype=dtype)
        _set_submodule(model, name, wrapper)
        inserted[name] = wrapper
    return inserted


def _conv_is_convertible(conv: nn.Conv2d) -> bool:
    """Whether ``IConv2d`` can express this conv exactly.

    ``IConv2d`` is a single-group, dilation-1 conv with one stride and a symmetric
    ``(ph, pw)`` padding, so *padded* and *overlapping* convs are converted too — that
    covers the 3x3/pad-1 projections of the mask and heatmap heads, not just the
    patch-embed stem. What is still refused is what the kernel genuinely cannot
    compute: a dilated conv, a grouped/depthwise conv, a conv whose two axes stride
    differently, a non-``zeros`` ``padding_mode``, and the asymmetric ``padding="same"``
    of an even kernel (see ``conv_padding_pair``). Those stay float and are reported
    under ``left_float`` rather than converted into something that computes a
    different function.

    ``padding_mode`` is the subtle one and it is checked FIRST, because accepting a
    padded conv is what made it reachable: ``IConv2d`` constant-pads with the
    activation zero-point, which is the integer spelling of a real zero. A conv
    declaring ``reflect`` / ``replicate`` / ``circular`` fills its border from the
    image instead, so converting it would silently compute a different function —
    the exact failure this predicate exists to prevent.
    """
    if getattr(conv, "padding_mode", "zeros") != "zeros":
        return False
    stride = conv.stride if isinstance(conv.stride, (tuple, list)) else (conv.stride,)
    dilation = conv.dilation if isinstance(conv.dilation, (tuple, list)) else (conv.dilation,)
    if len({int(s) for s in stride}) != 1 or int(conv.groups) != 1:
        return False
    if not all(int(d) == 1 for d in dilation):
        return False
    try:
        conv_padding_pair(conv)
    except ValueError:
        return False
    return True


def calibrate_int_convs(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    act_dtype: QuantDtype = INT8,
    weight_dtype: QuantDtype = INT8,
) -> dict[str, IConv2d]:
    """Observe each convertible conv's input range and swap it for an ``IConv2d``.

    Only the convs ``IConv2d`` actually implements are converted — any single-group,
    dilation-1, uniformly-strided conv, padded or not: the patch-embed stem, the 1x1
    projections, and the 3x3/pad-1 head projections. A dilated / grouped conv is left
    float on purpose rather than converted into something that computes a different
    function; ``convert_model_to_integer`` then reports it under ``left_float``.
    """
    convs = {name: m for name, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and _conv_is_convertible(m)}
    if not convs:
        return {}
    samples = _observe(model, convs, batches, forward_fn,
                       lambda n, x, s: s[n].append(x.detach().abs().amax().reshape(1).float().cpu()))
    inserted: dict[str, IConv2d] = {}
    for name, conv in convs.items():
        if not samples[name]:
            continue
        peak = float(torch.cat(samples[name]).max())
        module = IConv2d.from_conv(conv, act_scale=max(peak, 1e-8) / float(act_dtype.qmax),
                                   act_dtype=act_dtype, weight_dtype=weight_dtype)
        _set_submodule(model, name, module)
        inserted[name] = module
    return inserted


# --- whole-graph conversion + honest report -----------------------------------

# Everything whose arithmetic runs on integers. IGeLUFloatIO is included because it is
# only a port around IGeLU: no float math of its own.
INTEGER_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    IAdd, ICat, IConv2d, IGeLU, IGeLUFloatIO, ILayerNorm, ILinear, IMatMul, IPool, ISoftmax,
)


def float_modules(model: nn.Module) -> tuple[dict[str, nn.Module], dict[str, nn.Module]]:
    """``(leaves, composites)`` — every module in ``model`` that still computes in float.

    * *leaves* are modules with no children that are not I-tier integer kernels: the
      ops that were not converted. Inert ones (``nn.Dropout``, ``nn.Identity``) are
      listed too — they are float-typed data movement, and hiding them would make the
      report a judgement call instead of a fact.
    * *composites* are modules with children whose own ``forward`` body does arithmetic
      the module swap cannot reach: the residual adds of a ``TransformerBlock``, the
      ``Q·Kᵀ`` / ``attn·V`` matmuls and the ``1/√d`` scaling inside
      ``MultiHeadAttention``. Those are ``torch`` function calls, not submodules, so no
      converter can replace them — ``ilayers.IMatMul`` / ``IAdd`` exist but wiring them
      in needs a rewritten attention forward, not a conversion pass. The root model
      appears under the empty name.

    This is a pure inspection of the module tree: nothing is mutated, and the answer
    depends only on what is actually installed.
    """
    leaves: dict[str, nn.Module] = {}
    composites: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, INTEGER_MODULE_TYPES):
            continue
        if next(module.children(), None) is not None:
            if type(module).forward is not nn.Module.forward:
                composites[name] = module
        else:
            leaves[name] = module
    return leaves, composites


@dataclass
class ConversionReport(Mapping):
    """What ``convert_model_to_integer`` did, and — just as loudly — what it did not.

    The report *is* the ``name -> new module`` mapping of everything that became
    integer (it implements ``Mapping``, so ``len(report)``, ``report[name]``,
    ``report.items()`` all read the ``replaced`` dict), and it additionally carries:

    * ``left_float``  — leaf modules still computing in float after the conversion.
    * ``float_composites`` — parent modules whose own forward body still composes their
      children in float (residual adds, attention matmuls, the attention scaling).
    * ``stages`` — what each conversion stage replaced, so a partial run is traceable.
    * ``auxiliary`` — of the replaced modules, those under a head the model declares as
      training-only. Conversion is a general pass and converts them like anything else;
      counting them in with the rest is what would turn conversion coverage into an
      overstatement of *deployment* coverage. On a FrameModel that is 3 of 38 modules —
      two convs and a GeLU belonging to the auxiliary mask head, which no accelerator
      ever runs.

    ``left_float`` and ``float_composites`` are recomputed from the converted module
    tree by ``float_modules``; they are an observation of the result, not a log of the
    intent, so they cannot drift away from what the model actually contains.
    """

    replaced: dict[str, nn.Module] = field(default_factory=dict)
    left_float: dict[str, nn.Module] = field(default_factory=dict)
    float_composites: dict[str, nn.Module] = field(default_factory=dict)
    stages: dict[str, tuple[str, ...]] = field(default_factory=dict)
    auxiliary: tuple[str, ...] = ()

    def __getitem__(self, name: str) -> nn.Module:
        return self.replaced[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.replaced)

    def __len__(self) -> int:
        return len(self.replaced)

    @property
    def deployed(self) -> dict[str, nn.Module]:
        """The replaced modules that are actually on the inference path."""
        aux = set(self.auxiliary)
        return {name: module for name, module in self.replaced.items() if name not in aux}

    def summary(self) -> str:
        """One line per stage plus the float remainder — for logs and PTQ reports."""
        lines = [f"{stage}: {len(names)}" for stage, names in self.stages.items()]
        lines.append(f"integer modules: {len(self.replaced)}")
        if self.auxiliary:
            lines.append(f"  on the inference path: {len(self.deployed)}")
            lines.append(f"  training-only (aux heads): {len(self.auxiliary)}")
        lines.append(f"left float (leaf): {len(self.left_float)}")
        lines.append(f"float composites: {len(self.float_composites)}")
        return "\n".join(lines)


def convert_model_to_integer(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_fn: ForwardFn | None = None,
    include_nonlinear: bool = True,
    include_conv: bool = True,
    include_seams: bool = True,
    layernorm_entries: int = 256,
    layernorm_segments: int = 2,
    softmax_entries: int = 256,
    gelu_entries: int = 256,
    max_tokens: int | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[nn.Module, ConversionReport]:
    """Full Q -> I deployment conversion, in place, with an honest report.

    Stages, in the order they run — each observes the model as the previous stage left
    it, so every calibration sees the inputs the deployed graph will actually produce:

        1. LayerNorm -> ``ILayerNorm``   (integer mean / variance / rsqrt)
        2. Softmax   -> ``ISoftmax``     (integer row max / exp / row sum / reciprocal)
        3. GeLU      -> ``IGeLU``        (integer table lookup)
        4. Conv2d    -> ``IConv2d``      (integer conv, strided and/or padded; ``include_conv``)
        5. QLinear   -> ``ILinear``      (integer matmul)

    ``batches`` is consumed several times (once per observing stage), so it is
    materialized to a list first; ``forward_fn(model, batch)`` adapts the drive per
    modality exactly as in ``calibrate``.

    WHAT THIS DOES NOT DO — read before quoting the model as "fully integer". Tensors
    move between modules as float: each I-tier module quantizes at its input port and
    dequantizes at its output port, so every *op* is integer while the *transport* is
    not. And the arithmetic written directly in a parent's ``forward`` — the two
    attention matmuls, the ``1/√d`` scaling, the residual adds — is not a submodule and
    therefore cannot be swapped; it stays float. Both facts are reported:
    ``report.left_float`` lists the unconverted leaves and ``report.float_composites``
    the parents whose bodies still compose in float.

    Returns ``(model, report)``; the report doubles as the ``name -> module`` mapping of
    what became integer.
    """
    batches = list(batches)
    replaced: dict[str, nn.Module] = {}
    stages: dict[str, tuple[str, ...]] = {}

    def run_stage(stage: str, inserted: dict[str, Any]) -> None:
        stages[stage] = tuple(inserted)
        replaced.update(inserted)

    if include_nonlinear:
        run_stage("layernorm", calibrate_int_layernorms(
            model, batches, forward_fn=forward_fn, entries=layernorm_entries,
            segments=layernorm_segments, max_rows=max_rows))
        run_stage("softmax", calibrate_int_softmaxes(
            model, batches, forward_fn=forward_fn, exp_entries=softmax_entries,
            recip_entries=softmax_entries, max_tokens=max_tokens, max_rows=max_rows))
        run_stage("gelu", calibrate_int_gelus(
            model, batches, forward_fn=forward_fn, entries=gelu_entries))
    if include_conv:
        run_stage("conv", calibrate_int_convs(model, batches, forward_fn=forward_fn))
    if include_seams:
        run_stage("matmul", calibrate_int_matmuls(model, batches, forward_fn=forward_fn))
        run_stage("add", calibrate_int_adds(model, batches, forward_fn=forward_fn))
    _, linears = convert_to_integer(model)
    run_stage("linear", linears)

    leaves, composites = float_modules(model)
    # The model, not the quantizer, decides which of its heads are training-only: that is
    # a modelling fact (the mask head is Sec III-D.1 auxiliary supervision), and a name
    # hardcoded here would be a second copy of it, free to drift.
    declared = getattr(model, "auxiliary_module_names", None)
    aux = set(declared()) if callable(declared) else set()
    return model, ConversionReport(replaced=replaced, left_float=leaves,
                                   float_composites=composites, stages=stages,
                                   auxiliary=tuple(n for n in replaced if n in aux))


__all__ = [
    "ConversionReport",
    "DEFAULT_MAX_ROWS",
    "IGeLUFloatIO",
    "INTEGER_MODULE_TYPES",
    "calibrate_gelu_luts",
    "calibrate_int_convs",
    "calibrate_int_gelus",
    "calibrate_int_layernorms",
    "calibrate_int_softmaxes",
    "calibrate_layernorm_luts",
    "calibrate_softmax_luts",
    "collect_quantizers",
    "convert_model_to_integer",
    "convert_to_integer",
    "float_modules",
    "insert_fake_quant",
]
