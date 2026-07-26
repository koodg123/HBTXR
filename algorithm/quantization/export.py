"""Export a converted integer model to HW-deployable artifacts — and replay them.

Walks a model that has been through ``convert_to_integer`` (+ nonlinear LUT
calibration) and writes, per module, the integer weights, scales, zero-points and
LUT tables/scalars a hardware / bit-exact-simulator backend needs, plus a
``manifest.json`` describing the graph. Integer weights go to ``.npy`` in the
narrowest *lossless* integer container (int8 / int16 / int32 — never a silent
wrap); everything else is inlined in the manifest as plain JSON.

An export nobody can replay is worthless, so this module also owns the way back:

- :func:`load_integer_manifest` re-reads the directory into a dict with the
  ``.npy`` payloads materialized as numpy arrays, and rejects an artifact whose
  dtype, shape *or value range* disagrees with what the manifest declared.
- :func:`verify_export` rebuilds each op's integer forward *from the exported
  artifacts alone* (numpy, no live module state) and checks it reproduces the live
  module. That is the property the whole format stands on.

Three rules keep the gate from signing off on garbage — every one of them was a
real defect here, and each is worth stating because the obvious implementation
gets it wrong:

1. **Every** exported op is replayed. There is no "this op is awkward, call it a
   pass" branch. A previous revision skipped LayerNorm and Softmax — 26% of a
   FrameModel dump — while still returning ``True``, so zeroing every live LUT
   left the gate reporting byte-identical success.
2. An op the replayer does not understand is reported ``ok=False`` with
   ``max_abs_diff=None``. Never a fabricated ``0.0``: a number that was never
   measured reads downstream as a measurement. :func:`verify_export` only tolerates
   such entries under the explicit opt-in ``allow_unverified=True``.
3. :func:`export_integer_model` refuses to drop a module on the floor. A module
   that owns parameters or buffers and has no export branch either aborts the
   export (always, for anything from the ``quantization`` package — that is the
   exact shape of the bug where a newly installed ``ILayerNorm`` silently vanished)
   or is listed in ``manifest["unexported"]`` under ``allow_unquantized=True``.

    from quantization.convert import convert_to_integer
    from quantization.export import export_integer_model, verify_export
    model, _ = convert_to_integer(model)   # after PTQ/QAT + LUT calibration
    export_integer_model(model, "runs/frame_hbtxr/int_export")
    ok, max_diff = verify_export(model, "runs/frame_hbtxr/int_export")
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

from quantization.ilayers.conv import IConv2d
from quantization.ilayers.layernorm import ILayerNorm
from quantization.ilayers.linear import ILinear
from quantization.ilayers.nonlinear import IGeLU
from quantization.ilayers.qtensor import QTensor
from quantization.ilayers.softmax import ISoftmax
from quantization.qlayers.nonlinear import QGeLU, QLayerNorm, QSoftmax
from quantization.scheme import qrange

FORMAT = "hbtxr-int-v1"
FORMAT_VERSION = 3
MANIFEST_NAME = "manifest.json"

# candidate .npy containers, narrowest first — the first one that holds the real
# value range losslessly wins (a wider weight dtype must never wrap into int8).
_CONTAINERS = (np.int8, np.int16, np.int32)

# probe geometry: a softmax probe needs a token axis and a conv probe needs a
# spatial extent, neither of which the manifest records (they are runtime shapes).
_SOFTMAX_PROBE_TOKENS = 16
_CONV_PROBE_OUT = 2
# fraction of a LUT's input span probed beyond each end, so the cursor clamp is
# exercised rather than assumed (see _probe).
_LUT_PROBE_OVERSHOOT = 0.25


# --- serialization helpers ---------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Last-resort JSON coercion for numpy / torch scalars and tensors."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__} into the export manifest")


def _scalar_float(value: Any) -> float:
    """Python float from a python / numpy / 0-d or 1-element torch scalar."""
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0])
    return int(value)


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
    return [float(v) for v in np.asarray(value).reshape(-1).tolist()]


def _int_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in np.asarray(value).reshape(-1).tolist()]


def _dtype_fields(dtype: Any, prefix: str) -> dict[str, Any]:
    """``{prefix}_bits`` / ``{prefix}_signed`` for a ``QuantDtype`` (None-safe)."""
    if dtype is None:
        return {f"{prefix}_bits": None, f"{prefix}_signed": None}
    return {f"{prefix}_bits": int(dtype.bits), f"{prefix}_signed": bool(dtype.signed)}


def _safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "root"


def _stem(order: int, name: str) -> str:
    """Filename stem that is unique even if two module paths sanitize alike.

    ``blk.0`` and ``blk_0`` both sanitize to ``blk_0``; without the ``order`` prefix
    the second export silently overwrites the first ``.npy`` and both modules then
    load the same weights.
    """
    return f"{order:03d}_{_safe(name)}"


def _signed_bits(lo: int, hi: int) -> int:
    """Narrowest signed bit-width whose range covers ``[lo, hi]``."""
    magnitude = max(-int(lo), int(hi) + 1, 1)
    return max(2, (magnitude - 1).bit_length() + 1)


def _save_int(out: Path, stem: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Save an integer tensor losslessly; returns the manifest fields describing it."""
    arr = tensor.detach().cpu().numpy()
    if arr.size == 0:
        raise ValueError(f"refusing to export empty integer tensor {stem!r}")
    if arr.dtype.kind == "f":
        # The one genuinely lossy input: a module whose "integer" buffer is a float
        # tensor that never got rounded. Truncating it would corrupt the dump quietly.
        if not bool(np.all(np.rint(arr) == arr)):
            raise ValueError(f"integer tensor {stem!r} holds non-integral values; refusing to truncate")
        arr = np.rint(arr)
    elif arr.dtype.kind not in "iub":
        raise ValueError(f"integer tensor {stem!r} has non-numeric dtype {arr.dtype}")
    lo, hi = int(arr.min()), int(arr.max())
    container = next(
        (c for c in _CONTAINERS if lo >= np.iinfo(c).min and hi <= np.iinfo(c).max), None
    )
    if container is None:
        raise ValueError(f"integer tensor {stem!r} range [{lo}, {hi}] does not fit int32")
    # The container was picked FROM [lo, hi], so this cast cannot lose information —
    # the teeth are the integral-value guard above and the int32 rejection here, not
    # a post-cast round-trip assert (which by construction could never fire).
    stored = arr.astype(container)
    fname = f"{stem}.npy"
    np.save(out / fname, stored)
    return {
        "file": fname,
        "dtype": str(np.dtype(container)),
        "shape": [int(d) for d in arr.shape],
        "int_min": lo,
        "int_max": hi,
    }


def _weight_bits(module: nn.Module, info: dict[str, Any]) -> tuple[int, bool]:
    """(bits, inferred). I-tier modules keep no weight dtype, so fall back to range."""
    dtype = getattr(module, "weight_dtype", None)
    if dtype is not None and getattr(dtype, "bits", None):
        return int(dtype.bits), False
    return _signed_bits(info["int_min"], info["int_max"]), True


# --- per-module manifest entries ---------------------------------------------
#
# ``type`` states what the live module ACTUALLY is (``type(m).__name__``) — it is an
# integrity field, checked against the live class by verify_export_report, so it must
# never be a hardcoded guess. ``op`` names the replay kernel and is what dispatch
# keys on, which keeps a legitimate subclass replayable without lying about its class.

def _linear_entry(out: Path, order: int, name: str, m: ILinear) -> dict[str, Any]:
    info = _save_int(out, f"{_stem(order, name)}.weight", m.weight_int)
    bits, inferred = _weight_bits(m, info)
    return {
        "type": type(m).__name__,
        "op": "linear_int",
        "weight_file": info["file"],
        "weight_dtype": info["dtype"],
        "weight_shape": info["shape"],
        "weight_int_min": info["int_min"],
        "weight_int_max": info["int_max"],
        "weight_bits": bits,
        "weight_bits_inferred": inferred,
        "weight_signed": True,
        "weight_zero_point": 0,          # ILinear enforces symmetric weights
        "weight_scale": _float_list(m.weight_scale),
        "act_scale": _scalar_float(m.act_scale),
        "act_zero_point": _scalar_int(m.act_zero_point),
        "act_bits": int(m.act_dtype.bits),
        "act_signed": bool(m.act_dtype.signed),
        "in_features": info["shape"][1],
        "out_features": info["shape"][0],
        "bias": _float_list(m.bias),
    }


def _conv_entry(out: Path, order: int, name: str, m: IConv2d) -> dict[str, Any]:
    info = _save_int(out, f"{_stem(order, name)}.weight", m.weight_int)
    bits, inferred = _weight_bits(m, info)
    return {
        "type": type(m).__name__,
        "op": "conv2d_int",
        "weight_file": info["file"],
        "weight_dtype": info["dtype"],
        "weight_shape": info["shape"],                 # [Cout, Cin, kh, kw]
        "weight_int_min": info["int_min"],
        "weight_int_max": info["int_max"],
        "weight_bits": bits,
        "weight_bits_inferred": inferred,
        "weight_signed": True,
        "weight_zero_point": 0,
        "weight_scale": _float_list(m.weight_scale),
        "act_scale": _scalar_float(m.act_scale),
        "act_zero_point": _scalar_int(m.act_zero_point),
        "act_bits": int(m.act_dtype.bits),
        "act_signed": bool(m.act_dtype.signed),
        "stride": int(m.stride),
        "bias": _float_list(m.bias),
    }


def _gelu_entry(m: QGeLU | IGeLU) -> dict[str, Any]:
    """QGeLU and IGeLU share one PoT-indexed table, so they share one replay kernel."""
    entry = {
        "type": type(m).__name__,
        "op": "gelu_lut",
        "scalars": [_scalar_int(m.b), _scalar_int(m.s), _scalar_int(m.bound)],
        "input_scale": _scalar_float(m.input_scale),
        "output_scale": _scalar_float(m.output_scale),
        "table": _int_list(m.table),
        "table_entries": int(m.table.numel()),
    }
    entry.update(_dtype_fields(getattr(m, "out_dtype", None), "output"))
    return entry


def _int_layernorm_entry(m: ILayerNorm) -> dict[str, Any]:
    entry = {
        "type": type(m).__name__,
        "op": "layernorm_int",
        "scalars": [int(m.c_1_m), int(m.c_1_s), int(m.b), int(m.s1),
                    int(m.bound), int(m.s2), int(m.clamp_bits)],
        "channels": int(m.channels),
        "input_scale": _scalar_float(m.input_scale),
        "output_scale": _scalar_float(m.output_scale),
        "rsqrt_table": _int_list(m.rsqrt_table),
        "table_entries": int(m.rsqrt_table.numel()),
        "lnw": _int_list(m.lnw),
        "lnb": _int_list(m.lnb),
    }
    entry.update(_dtype_fields(m.in_dtype, "input"))
    entry.update(_dtype_fields(m.out_dtype, "output"))
    return entry


def _q_layernorm_entry(m: QLayerNorm) -> dict[str, Any]:
    """Q-tier LayerNorm: float mean/var, integer rsqrt LUT (``input_scale`` is a *variance* scale)."""
    return {
        "type": type(m).__name__,
        "op": "layernorm_lut",
        "normalized_shape": [int(d) for d in m.normalized_shape],
        "eps": float(m.eps),
        "scalars": [_scalar_int(m.b), _scalar_int(m.s), _scalar_int(m.bound)],
        "input_scale": _scalar_float(m.input_scale),
        "output_scale": _scalar_float(m.output_scale),
        "rsqrt_table": _int_list(m.rsqrt_table),
        "table_entries": int(m.rsqrt_table.numel()),
        "weight": _float_list(m.weight),
        "bias": _float_list(m.bias),
    }


def _int_softmax_entry(m: ISoftmax) -> dict[str, Any]:
    entry = {
        "type": type(m).__name__,
        "op": "softmax_int",
        "dim": int(m.dim),
        "scalars": [
            int(m.b1), int(m.s1), int(m.bound1),
            int(m.b2_one), int(m.s2_one), int(m.bound2_one), int(m.b3_one), int(m.s3_one),
            int(m.b2_two), int(m.s2_two), int(m.bound2_two), int(m.b3_two), int(m.s3_two),
            int(m.clamp_bits),
        ],
        "input_scale": _scalar_float(m.input_scale),
        "output_scale": _scalar_float(m.output_scale),
        "exp_table": _int_list(m.exp_table),
        "exp_table_entries": int(m.exp_table.numel()),
        "recip_table_one": _int_list(m.recip_table_one),
        "recip_table_two": _int_list(m.recip_table_two),
        "recip_table_entries": int(m.recip_table_one.numel()),
    }
    entry.update(_dtype_fields(m.in_dtype, "input"))
    entry.update(_dtype_fields(m.out_dtype, "output"))
    return entry


def _q_softmax_entry(m: QSoftmax) -> dict[str, Any]:
    """Q-tier softmax: float max/sum, integer exp + reciprocal LUTs (one segment each)."""
    return {
        "type": type(m).__name__,
        "op": "softmax_lut",
        "dim": int(m.dim),
        "exp": {
            "scalars": [_scalar_int(m.exp_b), _scalar_int(m.exp_s), _scalar_int(m.exp_bound)],
            "input_scale": _scalar_float(m.exp_in),
            "output_scale": _scalar_float(m.exp_out),
            "table": _int_list(m.exp_table),
            "table_entries": int(m.exp_table.numel()),
        },
        "recip": {
            "scalars": [_scalar_int(m.recip_b), _scalar_int(m.recip_s), _scalar_int(m.recip_bound)],
            "input_scale": _scalar_float(m.recip_in),
            "output_scale": _scalar_float(m.recip_out),
            "table": _int_list(m.recip_table),
            "table_entries": int(m.recip_table.numel()),
        },
    }


# --- the walk ------------------------------------------------------------------

def _own_state(m: nn.Module) -> tuple[list[str], list[str]]:
    """(parameter names, buffer names) the module owns itself (not via children)."""
    return ([n for n, _ in m.named_parameters(recurse=False)],
            [n for n, _ in m.named_buffers(recurse=False)])


def _is_quant_tier(m: nn.Module) -> bool:
    """True for a module defined in the quantization package.

    Class *namespace* rather than a base class, because the Q- and I-tier modules
    deliberately share no ancestor. Anything from here that carries state and has no
    export branch is a quantized op that would vanish from the dump — the exact
    failure mode this guard exists for — so it is fatal with no opt-out.
    """
    return type(m).__module__.split(".")[0] == "quantization"


def export_integer_model(
    model: nn.Module,
    out_dir,
    *,
    allow_unquantized: bool = False,
) -> dict[str, Any]:
    """Write integer weights + scales + LUTs and a manifest; returns the manifest.

    ``named_modules`` order is stable for a given model, so ``order`` doubles as the
    artifact filename prefix and as a deterministic execution-order hint; each entry
    also carries its dotted ``name``, which is exactly the path ``get_submodule``
    takes to get back to the live module.

    A module with own parameters/buffers and no export branch is never dropped
    silently. From the ``quantization`` package it always raises. From anywhere else
    (a float ``nn.Conv2d`` the PTQ pipeline never converts, say) it raises unless
    ``allow_unquantized=True``, which instead lists it in ``manifest["unexported"]``
    so the hole stays visible to whoever reads the dump.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modules: dict[str, Any] = {}
    counts: dict[str, int] = {}
    unexported: list[dict[str, Any]] = []
    total_int_weights = 0
    for name, m in model.named_modules():
        order = len(modules)
        if isinstance(m, ILinear):
            entry = _linear_entry(out, order, name, m)
        elif isinstance(m, IConv2d):
            entry = _conv_entry(out, order, name, m)
        elif isinstance(m, (IGeLU, QGeLU)):
            entry = _gelu_entry(m)
        elif isinstance(m, ILayerNorm):
            entry = _int_layernorm_entry(m)
        elif isinstance(m, QLayerNorm):
            entry = _q_layernorm_entry(m)
        elif isinstance(m, ISoftmax):
            entry = _int_softmax_entry(m)
        elif isinstance(m, QSoftmax):
            entry = _q_softmax_entry(m)
        else:
            params, buffers = _own_state(m)
            if not params and not buffers:
                continue                                  # container / stateless op
            record = {"name": name, "class": type(m).__name__,
                      "defined_in": type(m).__module__,
                      "parameters": params, "buffers": buffers}
            if _is_quant_tier(m):
                raise TypeError(
                    f"module {name!r} is a {type(m).__name__} from {type(m).__module__} with "
                    f"state {params + buffers} and no export branch — it would vanish from the "
                    f"manifest. Add a branch to export_integer_model.")
            unexported.append(record)
            continue
        entry["order"] = order
        entry["name"] = name
        modules[name] = entry
        counts[entry["type"]] = counts.get(entry["type"], 0) + 1
        if "weight_shape" in entry:
            total_int_weights += int(np.prod(entry["weight_shape"]))

    if unexported and not allow_unquantized:
        listing = ", ".join(f"{r['name']}({r['class']})" for r in unexported)
        raise TypeError(
            f"{len(unexported)} module(s) carry parameters/buffers but are not exportable: "
            f"{listing}. Convert them, or pass allow_unquantized=True to record them in "
            f"manifest['unexported'] instead.")

    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model_class": type(model).__name__,
        "num_modules": len(modules),
        "total_int_weights": total_int_weights,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "counts": counts,
        "num_unexported": len(unexported),
        "unexported": unexported,
        "modules": modules,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=_json_default))
    return manifest


# --- loading back -------------------------------------------------------------

_INT_ARRAY_KEYS = ("table", "rsqrt_table", "lnw", "lnb",
                   "exp_table", "recip_table_one", "recip_table_two")
_FLOAT_ARRAY_KEYS = ("weight_scale", "bias", "weight")


def load_integer_manifest(out_dir) -> dict[str, Any]:
    """Read an exported directory back; ``.npy`` payloads become numpy arrays.

    Numeric payloads are materialized in place (``weight_int`` as the loaded array,
    tables as int64 arrays, scales/bias as float32 arrays) so a backend can consume
    the manifest directly. The result is therefore no longer plain JSON.

    Every integrity field the export recorded is checked, not just dtype and shape:
    an all-zero weight file whose manifest declares ``[-127, 127]`` is a corrupted
    artifact, and the recorded range costs two comparisons to catch it.
    """
    out = Path(out_dir)
    manifest_path = out / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {out}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"unexpected export format {manifest.get('format')!r}, expected {FORMAT!r}")

    for name, entry in manifest["modules"].items():
        file = entry.get("weight_file")
        if file is not None:
            path = out / file
            if not path.is_file():
                raise FileNotFoundError(f"module {name!r} references missing artifact {file}")
            array = np.load(path)
            if str(array.dtype) != entry["weight_dtype"] or list(array.shape) != entry["weight_shape"]:
                raise ValueError(
                    f"module {name!r} artifact {file} is {array.dtype}{list(array.shape)}, "
                    f"manifest says {entry['weight_dtype']}{entry['weight_shape']}")
            lo, hi = int(array.min()), int(array.max())
            if lo != entry["weight_int_min"] or hi != entry["weight_int_max"]:
                raise ValueError(
                    f"module {name!r} artifact {file} spans [{lo}, {hi}], manifest declares "
                    f"[{entry['weight_int_min']}, {entry['weight_int_max']}]")
            entry["weight_int"] = array
        for key in _FLOAT_ARRAY_KEYS:
            if entry.get(key) is not None:
                entry[key] = np.asarray(entry[key], dtype=np.float32)
        for key in _INT_ARRAY_KEYS:
            if entry.get(key) is not None:
                entry[key] = np.asarray(entry[key], dtype=np.int64)
        for key in ("exp", "recip"):
            if entry.get(key) is not None:
                entry[key]["table"] = np.asarray(entry[key]["table"], dtype=np.int64)
    return manifest


# --- replay from artifacts + equivalence proof --------------------------------
#
# Every replay below reads ONLY ``entry`` (manifest values) and the probe. None of
# them may touch the live module — that is what makes the comparison a proof that the
# dump is self-sufficient rather than a re-run of the model.

def replay_linear(entry: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """``ILinear`` forward rebuilt from manifest values only (float32 like torch).

    Mirrors ``ILinear.forward`` exactly: quantize the activation, integer matmul,
    fold the activation zero-point as ``- zp·Σw``, dequantize by ``s_x·s_w``, add bias.
    """
    s_x = np.float32(entry["act_scale"])
    zero_point = int(entry["act_zero_point"])
    qmin, qmax = qrange(int(entry["act_bits"]), signed=bool(entry["act_signed"]))
    x_int = np.clip(np.round(x.astype(np.float32) / s_x + zero_point), qmin, qmax).astype(np.int64)
    weight = np.asarray(entry["weight_int"], dtype=np.int64)               # [out, in]
    acc = x_int @ weight.T
    if zero_point != 0:
        acc = acc - zero_point * weight.sum(axis=1)
    s_w = np.asarray(entry["weight_scale"], dtype=np.float32)
    y = acc.astype(np.float32) * (s_x * s_w)
    bias = entry.get("bias")
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)
    return y


def replay_conv(entry: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """``IConv2d`` forward rebuilt from manifest values only (non-padded, strided).

    The live module accumulates through ``F.conv2d`` on float32 operands because the
    products stay inside float32's exact integer range for ViT patch dims; the replay
    accumulates in int64 via strided slices, so agreement is also a check that the
    float accumulation has not silently left that range.
    """
    s_x = np.float32(entry["act_scale"])
    zero_point = int(entry["act_zero_point"])
    qmin, qmax = qrange(int(entry["act_bits"]), signed=bool(entry["act_signed"]))
    x_int = np.clip(np.round(x.astype(np.float32) / s_x + zero_point), qmin, qmax).astype(np.int64)
    weight = np.asarray(entry["weight_int"], dtype=np.int64)               # [Cout, Cin, kh, kw]
    stride = int(entry["stride"])
    _cout, _cin, kh, kw = weight.shape
    n, _c, height, width = x_int.shape
    ho = (height - kh) // stride + 1
    wo = (width - kw) // stride + 1
    if ho <= 0 or wo <= 0:
        raise ValueError(f"conv probe {x_int.shape} is smaller than the {kh}x{kw} kernel")
    acc = np.zeros((n, weight.shape[0], ho, wo), dtype=np.int64)
    for dy in range(kh):
        for dx in range(kw):
            patch = x_int[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
            acc += np.einsum("nchw,oc->nohw", patch, weight[:, :, dy, dx])
    if zero_point != 0:
        acc = acc - zero_point * weight.sum(axis=(1, 2, 3)).reshape(1, -1, 1, 1)
    s_w = np.asarray(entry["weight_scale"], dtype=np.float32).reshape(1, -1, 1, 1)
    y = acc.astype(np.float32) * (s_x * s_w)
    bias = entry.get("bias")
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32).reshape(1, -1, 1, 1)
    return y


def replay_gelu(entry: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """PoT-index table lookup — the shared kernel of ``QGeLU`` and ``IGeLU``.

    The two modules differ only in their interface (float in/out vs QTensor in/out);
    both compute ``table[clamp((round(x/s_in) + b) >> s, 0, bound)] * s_out``.
    """
    b, shift, bound = (int(v) for v in entry["scalars"])
    table = np.asarray(entry["table"], dtype=np.int64)
    x_int = np.round(x.astype(np.float32) / np.float32(entry["input_scale"])).astype(np.int64)
    cursor = np.clip((x_int + b) >> shift, 0, bound)
    return table[cursor].astype(np.float32) * np.float32(entry["output_scale"])


def replay_layernorm(entry: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """``ILayerNorm`` rebuilt from manifest values only — fully integer, no float reduction.

    This is ``i_ops.layernorm_quantize`` in numpy: integer mean by reciprocal
    multiply-and-shift, integer variance accumulation, PoT-indexed rsqrt table,
    integer affine, arithmetic right shift, clamp. numpy's ``>>`` sign-extends on
    signed integers exactly like Python's and ``torch.bitwise_right_shift``, so
    negative deviations floor identically and the replay is bit-exact, not close.
    """
    c_1_m, c_1_s, b, s1, bound, s2, _clamp_bits = (int(v) for v in entry["scalars"])
    in_qmin, in_qmax = qrange(int(entry["input_bits"]), signed=bool(entry["input_signed"]))
    x_int = np.clip(np.round(x.astype(np.float32) / np.float32(entry["input_scale"])),
                    in_qmin, in_qmax).astype(np.int64)
    acc = x_int.sum(axis=-1, keepdims=True)
    mean = (acc * c_1_m + (1 << (c_1_s - 1))) >> c_1_s
    diff = x_int - mean
    var_sum = (diff * diff).sum(axis=-1, keepdims=True)
    cursor = np.clip((var_sum + b) >> s1, 0, bound)
    rsqrt = np.asarray(entry["rsqrt_table"], dtype=np.int64)[cursor]
    lnw = np.asarray(entry["lnw"], dtype=np.int64)
    lnb = np.asarray(entry["lnb"], dtype=np.int64)
    shifted = (diff * rsqrt * lnw + lnb) >> s2
    out_qmin, out_qmax = qrange(int(entry["output_bits"]), signed=bool(entry["output_signed"]))
    out = np.clip(shifted, out_qmin, out_qmax)
    return out.astype(np.float32) * np.float32(entry["output_scale"])


def replay_softmax(entry: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """``ISoftmax`` rebuilt from manifest values only — fully integer, segmented reciprocal.

    ``i_ops.softmax_quantize`` in numpy: integer row max, inverse-exp PoT table,
    integer accumulation, then the segment branch on the *unclamped* segment-one
    cursor (each row picks its own table, ``b3`` and ``s3``).
    """
    (b1, s1, bound1,
     b2_one, s2_one, bound2_one, b3_one, s3_one,
     b2_two, s2_two, bound2_two, b3_two, s3_two,
     clamp_bits) = (int(v) for v in entry["scalars"])
    dim = int(entry["dim"])
    in_qmin, in_qmax = qrange(int(entry["input_bits"]), signed=bool(entry["input_signed"]))
    x_int = np.clip(np.round(x.astype(np.float32) / np.float32(entry["input_scale"])),
                    in_qmin, in_qmax).astype(np.int64)
    delta = x_int.max(axis=dim, keepdims=True) - x_int
    cursor1 = np.clip((delta + b1) >> s1, 0, bound1)
    exp_values = np.asarray(entry["exp_table"], dtype=np.int64)[cursor1]
    acc = exp_values.sum(axis=dim, keepdims=True)

    cursor_one = (acc + b2_one) >> s2_one
    use_two = cursor_one > bound2_one
    cursor_two = np.clip((acc + b2_two) >> s2_two, 0, bound2_two)
    recip = np.where(use_two,
                     np.asarray(entry["recip_table_two"], dtype=np.int64)[cursor_two],
                     np.asarray(entry["recip_table_one"], dtype=np.int64)[np.clip(cursor_one, 0, bound2_one)])
    b3 = np.where(use_two, b3_two, b3_one)
    s3 = np.where(use_two, s3_two, s3_one)
    rel = (exp_values * recip + b3) >> s3
    qmin, qmax = qrange(clamp_bits, signed=False)
    return np.clip(rel, qmin, qmax).astype(np.float32) * np.float32(entry["output_scale"])


def replay_q_layernorm(entry: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """``QLayerNorm`` rebuilt from manifest values only (float mean/var + rsqrt LUT).

    The Q tier reduces in float, so this replay is *not* bit-exact by construction:
    a mean/variance that differs from torch's reduction order by a float32 ulp could
    in principle select a neighbouring LUT cursor. Measured over 1000 random probes
    across the FrameModel's five LayerNorms the worst disagreement was 2.4e-07 —
    ~400x under the 1e-4 gate — and no cursor ever flipped. The failure mode is
    fail-safe anyway: a flip shows up as a verification failure, never as a pass.

    The STE fold in the live module (``r_true + (r_lut - r_true)``) is value-identical
    to ``r_lut`` up to that same float32 round-off, which is where the 2.4e-07 comes from.
    """
    b, shift, bound = (int(v) for v in entry["scalars"])
    xf = x.astype(np.float32)
    mean = xf.mean(axis=-1, keepdims=True, dtype=np.float32)
    var = xf.var(axis=-1, keepdims=True, dtype=np.float32)
    var_int = np.round(var / np.float32(entry["input_scale"])).astype(np.int64)
    cursor = np.clip((var_int + b) >> shift, 0, bound)
    rsqrt = (np.asarray(entry["rsqrt_table"], dtype=np.int64)[cursor].astype(np.float32)
             * np.float32(entry["output_scale"]))
    y = (xf - mean) * rsqrt
    weight, bias = entry.get("weight"), entry.get("bias")
    if weight is not None:
        y = y * np.asarray(weight, dtype=np.float32)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)
    return y


def replay_q_softmax(entry: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """``QSoftmax`` rebuilt from manifest values only (float max/sum + exp & recip LUTs)."""
    dim = int(entry["dim"])
    xf = x.astype(np.float32)
    exp, recip = entry["exp"], entry["recip"]
    e_b, e_s, e_bound = (int(v) for v in exp["scalars"])
    delta = xf.max(axis=dim, keepdims=True) - xf
    delta_int = np.round(delta / np.float32(exp["input_scale"])).astype(np.int64)
    cursor = np.clip((delta_int + e_b) >> e_s, 0, e_bound)
    numerator = (np.asarray(exp["table"], dtype=np.int64)[cursor].astype(np.float32)
                 * np.float32(exp["output_scale"]))
    denom = numerator.sum(axis=dim, keepdims=True, dtype=np.float32)

    r_b, r_s, r_bound = (int(v) for v in recip["scalars"])
    denom_int = np.round(denom / np.float32(recip["input_scale"])).astype(np.int64)
    cursor2 = np.clip((denom_int + r_b) >> r_s, 0, r_bound)
    reciprocal = (np.asarray(recip["table"], dtype=np.int64)[cursor2].astype(np.float32)
                  * np.float32(recip["output_scale"]))
    return numerator * reciprocal


_REPLAY: dict[str, Callable[[dict[str, Any], np.ndarray], np.ndarray]] = {
    "linear_int": replay_linear,
    "conv2d_int": replay_conv,
    "gelu_lut": replay_gelu,
    "layernorm_int": replay_layernorm,
    "layernorm_lut": replay_q_layernorm,
    "softmax_int": replay_softmax,
    "softmax_lut": replay_q_softmax,
}


# --- probing + the live-module reference --------------------------------------

def _pot_input_range(scalars: Sequence[int], scale: float) -> tuple[float, float]:
    """Real-valued input interval a PoT table index ``(b, s, bound)`` resolves.

    ``cursor = clamp((x_int + b) >> s, 0, bound)`` saturates outside
    ``x_int ∈ [-b, ((bound+1) << s) - 1 - b]``; probing that interval exercises the
    whole table instead of a handful of entries around zero.
    """
    b, shift, bound = (int(v) for v in scalars)
    return float(-b) * scale, float(((bound + 1) << shift) - 1 - b) * scale


def _act_span(entry: dict[str, Any], scale_key: str, bits_key: str, signed_key: str) -> float:
    return float(entry[scale_key]) * qrange(int(entry[bits_key]), signed=bool(entry[signed_key]))[1]


def _probe(entry: dict[str, Any], rng: np.random.Generator, batch: int) -> np.ndarray:
    """Random input that covers the op's real input range (per op family)."""
    op = entry["op"]
    if op == "linear_int":
        span = _act_span(entry, "act_scale", "act_bits", "act_signed")
        return (rng.uniform(-1.0, 1.0, size=(batch, int(entry["in_features"]))) * span).astype(np.float32)
    if op == "conv2d_int":
        _cout, cin, kh, kw = (int(d) for d in entry["weight_shape"])
        stride = int(entry["stride"])
        height = kh + stride * (_CONV_PROBE_OUT - 1)
        width = kw + stride * (_CONV_PROBE_OUT - 1)
        span = _act_span(entry, "act_scale", "act_bits", "act_signed")
        return (rng.uniform(-1.0, 1.0, size=(batch, cin, height, width)) * span).astype(np.float32)
    if op == "gelu_lut":
        lo, hi = _pot_input_range(entry["scalars"], float(entry["input_scale"]))
        # Overshoot both ends: the cursor clamp is part of the op, and a probe derived
        # from ``bound`` alone can never saturate it. Without this margin a replay that
        # clamps to ``len(table) - 1`` instead of ``bound`` is indistinguishable, which
        # is wrong for any manifest whose table is longer than its index range.
        margin = _LUT_PROBE_OVERSHOOT * (hi - lo)
        return rng.uniform(lo - margin, hi + margin, size=(batch, 32)).astype(np.float32)
    if op == "layernorm_int":
        span = _act_span(entry, "input_scale", "input_bits", "input_signed")
        return (rng.uniform(-1.0, 1.0, size=(batch, int(entry["channels"]))) * span).astype(np.float32)
    if op == "layernorm_lut":
        # the Q-tier table is indexed by the *variance*, so the probe is sized by the
        # standard deviation that lands the row variance inside the table's range.
        _lo, var_hi = _pot_input_range(entry["scalars"], float(entry["input_scale"]))
        channels = int(entry["normalized_shape"][-1])
        std = np.sqrt(max(var_hi, 1e-12) * rng.uniform(0.02, 0.9, size=(batch, 1)))
        return (rng.standard_normal((batch, channels)) * std).astype(np.float32)
    if op == "softmax_int":
        span = _act_span(entry, "input_scale", "input_bits", "input_signed")
        return (rng.uniform(-1.0, 1.0, size=(batch, _SOFTMAX_PROBE_TOKENS)) * span).astype(np.float32)
    if op == "softmax_lut":
        _lo, delta_hi = _pot_input_range(entry["exp"]["scalars"], float(entry["exp"]["input_scale"]))
        return (rng.uniform(-1.0, 1.0, size=(batch, _SOFTMAX_PROBE_TOKENS))
                * (delta_hi / 2.0)).astype(np.float32)
    raise KeyError(f"no probe for op {op!r}")


def _reference(module: nn.Module, entry: dict[str, Any], probe: np.ndarray) -> np.ndarray:
    """Run the live module on the probe and return float output.

    Only ``IGeLU`` needs special handling: it is a QTensor-in / QTensor-out op with no
    float drop-in, so the probe is quantized on its own input grid and the result
    dequantized — which is exactly what ``replay_gelu`` reproduces.
    """
    x = torch.from_numpy(probe)
    with torch.no_grad():
        if isinstance(module, IGeLU):
            scale = float(entry["input_scale"])
            qt = QTensor(torch.round(x / scale).to(torch.int32), scale=scale, zero_point=0.0)
            return module(qt).dequantize().cpu().numpy()
        out = module(x)
    if isinstance(out, QTensor):
        return out.dequantize().cpu().numpy()
    return out.cpu().numpy()


def _dim_is_probeable(entry: dict[str, Any]) -> bool:
    """A 2-D probe can only address a reduction dim in ``{-2, -1, 0, 1}``."""
    if entry["op"] != "softmax_int":
        return True
    return int(entry["dim"]) in (-2, -1, 0, 1)


def verify_export_report(
    model: nn.Module,
    out_dir,
    *,
    atol: float = 1e-4,
    seed: int = 1234,
    batch: int = 4,
) -> list[dict[str, Any]]:
    """Per-module replay results: rebuild from artifacts, compare against the live module.

    Each record is ``{name, type, status, ok, max_abs_diff}``. ``status`` is one of

    ``checked``        replayed and compared; ``max_abs_diff`` is a real measurement.
    ``missing``        the manifest path no longer resolves to a live module.
    ``type_mismatch``  the manifest's ``type`` is not the live module's class — the
                       dump describes a different graph than the one in memory.
    ``unverified``     no replay kernel for this entry (a foreign or hand-edited
                       manifest). ``ok`` is False and ``max_abs_diff`` is **None**.

    ``max_abs_diff`` is ``None`` for everything that was not compared. It is never
    filled in with ``0.0``: a fabricated zero is indistinguishable downstream from a
    perfect match, which is how an unverified 26% of a dump once read as verified.
    """
    manifest = load_integer_manifest(out_dir)
    live = dict(model.named_modules())
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    for name, entry in sorted(manifest["modules"].items(), key=lambda kv: kv[1]["order"]):
        record: dict[str, Any] = {"name": name, "type": entry["type"],
                                  "status": "unverified", "ok": False, "max_abs_diff": None}
        module = live.get(name)
        if module is None:
            record["status"] = "missing"
            record["reason"] = "manifest path does not resolve to a live module"
            results.append(record)
            continue
        replay = _REPLAY.get(entry.get("op"))
        if replay is None or not _dim_is_probeable(entry):
            record["reason"] = f"no replay kernel for op {entry.get('op')!r}"
            results.append(record)
            continue
        if type(module).__name__ != entry["type"]:
            record["status"] = "type_mismatch"
            record["reason"] = (f"manifest says {entry['type']}, live module is "
                                f"{type(module).__name__}")
            results.append(record)
            continue
        probe = _probe(entry, rng, batch)
        replayed = replay(entry, probe)
        reference = _reference(module, entry, probe)
        diff = float(np.max(np.abs(replayed.astype(np.float64) - reference.astype(np.float64))))
        record.update(status="checked", ok=bool(diff <= atol), max_abs_diff=diff)
        results.append(record)
    return results


def verify_export(
    model: nn.Module,
    out_dir,
    *,
    atol: float = 1e-4,
    seed: int = 1234,
    batch: int = 4,
    allow_unverified: bool = False,
) -> tuple[bool, float]:
    """(ok, max_abs_diff) for replaying every exported op from its artifacts alone.

    ``ok`` is False if any replay disagrees with the live module, if a manifest entry
    no longer resolves to a module path, if an entry's declared type is not the live
    module's class, if nothing at all could be checked, **or if any exported module
    went unverified**. That last clause is the point of the gate: a sign-off that
    returns True while a quarter of the dump was never replayed is worse than no gate.

    ``allow_unverified=True`` is the explicit — and deliberately non-default — opt-in
    for a manifest that carries ops this replayer does not implement. It never excuses
    ``missing`` or ``type_mismatch``, which mean the dump and the model disagree about
    the graph itself.

    ``max_abs_diff`` summarizes only what was actually measured, and is ``inf`` when
    nothing was.
    """
    results = verify_export_report(model, out_dir, atol=atol, seed=seed, batch=batch)
    checked = [r for r in results if r["status"] == "checked"]
    if not checked:
        return False, float("inf")
    worst = max(r["max_abs_diff"] for r in checked)
    fatal = any(r["status"] in ("missing", "type_mismatch") for r in results)
    unverified = any(r["status"] == "unverified" for r in results)
    ok = all(r["ok"] for r in checked) and not fatal and not (unverified and not allow_unverified)
    return bool(ok), float(worst)


__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "export_integer_model",
    "load_integer_manifest",
    "replay_conv",
    "replay_gelu",
    "replay_layernorm",
    "replay_linear",
    "replay_q_layernorm",
    "replay_q_softmax",
    "replay_softmax",
    "verify_export",
    "verify_export_report",
]
