"""LUT generation and calibration helpers for HG-PIPE style tables.

The public HG-PIPE release contains deployed scalar/table artifacts but not
the original training-time generator.  This module implements a reproducible
builder that follows the paper's published contract: power-of-two index
approximation, optional joint range calibration, fused GeLU-ReQuant sampling,
and segmented reciprocal tables for Softmax.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class QuantizedRange:
    bits: int
    signed: bool

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1


@dataclass(frozen=True)
class PotIndexParams:
    offset: int
    shift: int
    bound: int
    input_min: int
    input_max: int
    rounding: bool = False

    @property
    def scalars(self) -> list[int]:
        return [self.offset, self.shift, self.bound]


def _as_1d_int(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError("calibration values must not be empty")
    return np.rint(arr.reshape(-1).astype(np.float64)).astype(np.int64)


def _clip_percentile(values: np.ndarray, percentile: float) -> tuple[int, int]:
    if percentile <= 0.0 or percentile > 100.0:
        raise ValueError("percentile must be in (0, 100]")
    if percentile == 100.0:
        lo = float(np.min(values))
        hi = float(np.max(values))
    else:
        tail = (100.0 - percentile) / 2.0
        lo, hi = np.percentile(values, [tail, 100.0 - tail])
    lo_i = int(math.floor(lo))
    hi_i = int(math.ceil(hi))
    if lo_i == hi_i:
        hi_i = lo_i + 1
    return lo_i, hi_i


def _ceil_log2(value: float) -> int:
    return max(0, int(math.ceil(math.log2(max(float(value), 1.0)))))


def make_pot_index_params(
    input_min: int,
    input_max: int,
    *,
    entries: int,
    rounding: bool = False,
) -> PotIndexParams:
    if entries < 2:
        raise ValueError("entries must be at least 2")
    if input_max <= input_min:
        input_max = input_min + 1
    bound = entries - 1
    shift = _ceil_log2((input_max - input_min) / float(bound))
    rounding_bias = (1 << (shift - 1)) if rounding and shift > 0 else 0
    return PotIndexParams(
        offset=-int(input_min) + rounding_bias,
        shift=shift,
        bound=bound,
        input_min=int(input_min),
        input_max=int(input_max),
        rounding=rounding,
    )


def coordinates_for(params: PotIndexParams) -> np.ndarray:
    indices = np.arange(params.bound + 1, dtype=np.int64)
    return ((indices << params.shift) - params.offset).astype(np.int64)


def cursor_for(values, params: PotIndexParams) -> np.ndarray:
    x = np.asarray(values, dtype=np.int64)
    return np.clip((x + params.offset) >> params.shift, 0, params.bound).astype(np.int64)


def apply_table(values, params: PotIndexParams, table) -> np.ndarray:
    lut = np.asarray(table, dtype=np.int64)
    return lut[cursor_for(values, params)]


def quantize_clamp(values, qrange: QuantizedRange) -> np.ndarray:
    return np.clip(np.rint(values), qrange.qmin, qrange.qmax).astype(np.int64)


def gelu_tanh(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def _edge_active_span(table: np.ndarray) -> tuple[int, int]:
    if table.size == 0:
        return 0, 0
    left = 0
    while left + 1 < table.size and table[left + 1] == table[left]:
        left += 1
    right = table.size - 1
    while right - 1 >= 0 and table[right - 1] == table[right]:
        right -= 1
    return max(0, left), min(table.size - 1, right)


def _build_table_with_joint_range(
    samples: np.ndarray,
    *,
    entries: int,
    target: Callable[[np.ndarray], np.ndarray],
    percentile: float,
    max_iterations: int,
    rounding: bool,
) -> tuple[PotIndexParams, np.ndarray, list[dict[str, int]]]:
    lo, hi = _clip_percentile(samples, percentile)
    history: list[dict[str, int]] = []
    params = make_pot_index_params(lo, hi, entries=entries, rounding=rounding)
    table = target(coordinates_for(params))
    for _ in range(max(1, max_iterations)):
        coords = coordinates_for(params)
        table = target(coords)
        lsi, msi = _edge_active_span(table)
        new_lo = int(coords[lsi])
        new_hi = int(coords[msi])
        history.append(
            {
                "input_min": params.input_min,
                "input_max": params.input_max,
                "shift": params.shift,
                "offset": params.offset,
                "lsi": int(lsi),
                "msi": int(msi),
            }
        )
        if new_lo == params.input_min and new_hi == params.input_max:
            break
        if new_hi <= new_lo:
            break
        next_params = make_pot_index_params(new_lo, new_hi, entries=entries, rounding=rounding)
        if next_params == params:
            break
        params = next_params
    return params, np.asarray(table, dtype=np.int64), history


def _metrics(samples: np.ndarray, params: PotIndexParams, table: np.ndarray, target: Callable[[np.ndarray], np.ndarray]) -> dict[str, float | int]:
    approx = apply_table(samples, params, table)
    expected = target(samples)
    error = approx.astype(np.float64) - expected.astype(np.float64)
    return {
        "samples": int(samples.size),
        "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(error))) if error.size else 0.0,
        "mse": float(np.mean(error * error)) if error.size else 0.0,
        "clamped_low": int(np.sum(cursor_for(samples, params) == 0)),
        "clamped_high": int(np.sum(cursor_for(samples, params) == params.bound)),
    }


def calibrate_requant(
    values,
    *,
    entries: int = 64,
    bits: int = 3,
    signed: bool = True,
    scale: float = 1.0,
    zero_point: int = 0,
    percentile: float = 100.0,
    max_iterations: int = 8,
    rounding: bool = False,
) -> dict[str, object]:
    samples = _as_1d_int(values)
    qrange = QuantizedRange(bits=bits, signed=signed)

    def target(x):
        return quantize_clamp((np.asarray(x, dtype=np.float64) - zero_point) * float(scale), qrange)

    params, table, history = _build_table_with_joint_range(
        samples,
        entries=entries,
        target=target,
        percentile=percentile,
        max_iterations=max_iterations,
        rounding=rounding,
    )
    return _payload(
        kind="requant_table",
        params=params,
        tables={"table": table},
        metrics=_metrics(samples, params, table, target),
        qrange=qrange,
        method="pot_index_joint_range_calibration",
        history=history,
        extra={"scale": float(scale), "zero_point": int(zero_point), "percentile": float(percentile)},
    )


def calibrate_gelu_requant(
    values,
    *,
    entries: int = 64,
    bits: int = 3,
    signed: bool = False,
    input_scale: float = 1.0,
    output_scale: float = 1.0,
    percentile: float = 100.0,
    max_iterations: int = 8,
    rounding: bool = False,
) -> dict[str, object]:
    samples = _as_1d_int(values)
    qrange = QuantizedRange(bits=bits, signed=signed)

    def target(x):
        real = np.asarray(x, dtype=np.float64) * float(input_scale)
        return quantize_clamp(gelu_tanh(real) / float(output_scale), qrange)

    params, table, history = _build_table_with_joint_range(
        samples,
        entries=entries,
        target=target,
        percentile=percentile,
        max_iterations=max_iterations,
        rounding=rounding,
    )
    return _payload(
        kind="gelu_requant_table",
        params=params,
        tables={"table": table},
        metrics=_metrics(samples, params, table, target),
        qrange=qrange,
        method="fused_gelu_requant_pot_index_joint_range_calibration",
        history=history,
        extra={"input_scale": float(input_scale), "output_scale": float(output_scale), "percentile": float(percentile)},
    )


def calibrate_rsqrt(
    values,
    *,
    entries: int = 128,
    bits: int = 12,
    signed: bool = False,
    output_scale: float = 4096.0,
    epsilon: float = 1.0,
    percentile: float = 100.0,
    max_iterations: int = 1,
    rounding: bool = False,
) -> dict[str, object]:
    samples = np.maximum(_as_1d_int(values), 0)
    qrange = QuantizedRange(bits=bits, signed=signed)

    def target(x):
        x_float = np.maximum(np.asarray(x, dtype=np.float64), float(epsilon))
        return quantize_clamp(float(output_scale) / np.sqrt(x_float), qrange)

    params, table, history = _build_table_with_joint_range(
        samples,
        entries=entries,
        target=target,
        percentile=percentile,
        max_iterations=max_iterations,
        rounding=rounding,
    )
    return _payload(
        kind="layernorm_rsqrt_table",
        params=params,
        tables={"rsqrt_table": table},
        metrics=_metrics(samples, params, table, target),
        qrange=qrange,
        method="rsqrt_pot_index_calibration",
        history=history,
        extra={"output_scale": float(output_scale), "epsilon": float(epsilon), "percentile": float(percentile)},
    )


def calibrate_softmax(
    values,
    *,
    exp_entries: int = 32,
    recip_entries: int = 64,
    output_bits: int = 3,
    exp_scale: float = 32768.0,
    recip_scale: float = 256.0,
    input_scale: float = 1.0,
    percentile: float = 100.0,
) -> dict[str, object]:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        raise ValueError("softmax calibration expects a 1D vector or 2D row matrix")
    minus = np.rint(np.max(rows, axis=1, keepdims=True) - rows).astype(np.int64)
    minus_samples = minus.reshape(-1)

    def exp_target(x):
        return np.clip(
            np.rint(np.exp(-np.maximum(np.asarray(x, dtype=np.float64), 0.0) * float(input_scale)) * float(exp_scale)),
            0,
            int(exp_scale),
        ).astype(np.int64)

    exp_params, exp_table, exp_history = _build_table_with_joint_range(
        minus_samples,
        entries=exp_entries,
        target=exp_target,
        percentile=percentile,
        max_iterations=1,
        rounding=True,
    )

    exp_values = apply_table(minus_samples, exp_params, exp_table).reshape(minus.shape)
    acc_samples = np.sum(exp_values, axis=1).astype(np.int64)
    acc_min, acc_max = _clip_percentile(acc_samples, percentile)
    pivot = acc_min + max(1, int(math.ceil((acc_max - acc_min) / 8.0)))
    one_params = make_pot_index_params(acc_min, pivot, entries=recip_entries, rounding=True)
    two_params = make_pot_index_params(pivot + 1, acc_max, entries=recip_entries, rounding=True)

    def recip_values(params: PotIndexParams) -> np.ndarray:
        coords = np.maximum(coordinates_for(params).astype(np.float64), 1.0)
        return np.clip(np.rint(float(recip_scale) / coords), 0, int(recip_scale)).astype(np.int64)

    recip_one = recip_values(one_params)
    recip_two = recip_values(two_params)
    output_qrange = QuantizedRange(bits=output_bits, signed=False)
    # HG-PIPE Softmax stores reciprocal lookup and final requant scalars in one tuple.
    scalars = [
        exp_params.offset,
        exp_params.shift,
        exp_params.bound,
        one_params.offset,
        one_params.shift,
        one_params.bound,
        1 << max(output_bits, 0),
        max(0, int(math.ceil(math.log2(max(exp_scale * recip_scale / max(output_qrange.qmax, 1), 1.0))))),
        two_params.offset,
        two_params.shift,
        two_params.bound,
        1 << max(output_bits, 0),
        max(0, int(math.ceil(math.log2(max(exp_scale * recip_scale / max(output_qrange.qmax, 1), 1.0))))),
        output_bits,
    ]
    return {
        "schema": "hgpipe_lut_calibration_v1",
        "kind": "softmax_segmented_table",
        "method": "inversed_exp_and_segmented_recip_pot_index_calibration",
        "paper_equivalent": False,
        "source_basis": _source_basis(),
        "scalars": scalars,
        "exp_index": _params_json(exp_params),
        "recip_index_one": _params_json(one_params),
        "recip_index_two": _params_json(two_params),
        "output_dtype": {"signed": False, "bits": output_bits, "qmin": output_qrange.qmin, "qmax": output_qrange.qmax},
        "tables": {
            "exp_table": exp_table.astype(int).tolist(),
            "recip_table_one": recip_one.astype(int).tolist(),
            "recip_table_two": recip_two.astype(int).tolist(),
        },
        "metrics": {
            "rows": int(rows.shape[0]),
            "row_width": int(rows.shape[1]),
            "minus_samples": int(minus_samples.size),
            "acc_min": int(np.min(acc_samples)),
            "acc_max": int(np.max(acc_samples)),
            "recip_pivot": int(pivot),
            "exp_mse": _metrics(minus_samples, exp_params, exp_table, exp_target)["mse"],
        },
        "calibration_history": {"exp": exp_history},
        "parameters": {
            "input_scale": float(input_scale),
            "exp_scale": float(exp_scale),
            "recip_scale": float(recip_scale),
            "percentile": float(percentile),
        },
    }


def _params_json(params: PotIndexParams) -> dict[str, int | bool]:
    return {
        "offset": params.offset,
        "shift_scale": params.shift,
        "effective_divisor": 1 << params.shift,
        "bound": params.bound,
        "input_min": params.input_min,
        "input_max": params.input_max,
        "rounding": params.rounding,
    }


def _source_basis() -> list[str]:
    return [
        "ICCAD24-HG-PIPE src/quant.h and src/gelu.h use cursor=(x+b)>>s, clamp, table lookup.",
        "ICCAD24-HG-PIPE src/layernorm.h uses the same PoT cursor for rsqrt table lookup.",
        "ICCAD24-HG-PIPE src/softmax.h uses inverse-exp and two segmented reciprocal tables.",
        "Paper Sec. 4.4 publishes PoT index approximation, GeLU-ReQuant fusion, joint table range calibration, segmented Recip, and inversed Exp.",
    ]


def _payload(
    *,
    kind: str,
    params: PotIndexParams,
    tables: dict[str, np.ndarray],
    metrics: dict[str, float | int],
    qrange: QuantizedRange,
    method: str,
    history: list[dict[str, int]],
    extra: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "hgpipe_lut_calibration_v1",
        "kind": kind,
        "method": method,
        "paper_equivalent": False,
        "source_basis": _source_basis(),
        "scalars": params.scalars,
        "index": _params_json(params),
        "output_dtype": {"signed": qrange.signed, "bits": qrange.bits, "qmin": qrange.qmin, "qmax": qrange.qmax},
        "tables": {name: table.astype(int).tolist() for name, table in tables.items()},
        "metrics": metrics,
        "calibration_history": history,
        "parameters": extra,
    }


def calibrate_lut_from_array(kind: str, values, **kwargs) -> dict[str, object]:
    if kind == "requant":
        return calibrate_requant(values, **kwargs)
    if kind == "gelu-requant":
        return calibrate_gelu_requant(values, **kwargs)
    if kind == "rsqrt":
        return calibrate_rsqrt(values, **kwargs)
    if kind == "softmax":
        return calibrate_softmax(values, **kwargs)
    raise ValueError(f"unsupported calibration kind: {kind}")


def write_lut_payload_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_hgpipe_txt_artifacts(payload: dict[str, object], directory: Path, *, stem: str = "calibrated") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    scalars = payload.get("scalars")
    if isinstance(scalars, list):
        path = directory / f"{stem}_scalars.txt"
        path.write_text(",".join(str(int(value)) for value in scalars) + ",")
        written.append(path)
    tables = payload.get("tables", {})
    if isinstance(tables, dict):
        for name, table in tables.items():
            suffix = "table_m" if name == "table" else name
            path = directory / f"{stem}_{suffix}.txt"
            path.write_text(",".join(str(int(value)) for value in table) + ",")
            written.append(path)
    return written
