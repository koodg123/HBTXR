"""Structured HG-PIPE quantization parameter loading.

HG-PIPE uses both ordinary tensor dtype/range metadata and LUT-driven
quantization contracts.  Most deployed operators are not affine
scale/zero-point quantizers; their behavior is encoded by scalar tuples and
lookup tables from ``case/refs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import HgPipeSource, load_statistics, read_ints, stem_to_stat_key


@dataclass(frozen=True)
class TensorDTypeSpec:
    signed: bool
    bits: int

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1


@dataclass(frozen=True)
class TensorRangeSpec:
    minimum: float
    maximum: float


@dataclass(frozen=True)
class AffineQuantParams:
    scale: float
    zero_point: int
    dtype: TensorDTypeSpec


@dataclass(frozen=True)
class LutQuantParams:
    name: str
    scalars: tuple[int, ...]
    tables: tuple[tuple[int, ...], ...]
    input_dtype: TensorDTypeSpec | None = None
    output_dtype: TensorDTypeSpec | None = None
    observed_range: TensorRangeSpec | None = None
    zero_point: None = None

    @property
    def offset(self) -> int | None:
        return int(self.scalars[0]) if len(self.scalars) >= 1 else None

    @property
    def shift_scale(self) -> int | None:
        return int(self.scalars[1]) if len(self.scalars) >= 2 else None

    @property
    def effective_divisor(self) -> int | None:
        return 1 << self.shift_scale if self.shift_scale is not None else None

    @property
    def bound(self) -> int | None:
        return int(self.scalars[2]) if len(self.scalars) >= 3 else None


@dataclass(frozen=True)
class OpQuantContract:
    name: str
    kind: str
    stat_key: str
    input_dtype: TensorDTypeSpec | None
    output_dtype: TensorDTypeSpec | None
    observed_range: dict[str, TensorRangeSpec]
    params: LutQuantParams | None


def _dtype_from_stats(value: Any) -> TensorDTypeSpec | None:
    if not value:
        return None
    sign, bits = value
    if sign not in {"signed", "unsigned"}:
        raise ValueError(f"unknown sign type in statistics: {value!r}")
    return TensorDTypeSpec(signed=sign == "signed", bits=int(bits))


def _range_from_stats(value: Any) -> TensorRangeSpec | None:
    if value is None or isinstance(value, dict):
        return None
    if len(value) != 2:
        return None
    minimum, maximum = value
    return TensorRangeSpec(float(minimum), float(maximum))


class QuantParamStore:
    """Load scalar, LUT, dtype, and range metadata from an HG-PIPE source."""

    def __init__(self, source: HgPipeSource | str | Path):
        if not isinstance(source, HgPipeSource):
            source = HgPipeSource.from_path(source)
        self.source = source
        self.refs = source.refs
        self.stats = load_statistics(source)

    def tensor_dtype(self, key: str) -> TensorDTypeSpec | None:
        return _dtype_from_stats(self.stats.get("type", {}).get(key))

    def observed_range(self, key: str) -> TensorRangeSpec | None:
        return _range_from_stats(self.stats.get("range", {}).get(key))

    def observed_range_group(self, key: str) -> dict[str, TensorRangeSpec]:
        group = self.stats.get("range", {}).get(key, {})
        if not isinstance(group, dict):
            return {}
        return {name: spec for name, raw in group.items() if (spec := _range_from_stats(raw)) is not None}

    def table_params(self, name: str, scalars_file: str, *table_files: str) -> LutQuantParams:
        stat_key = stem_to_stat_key(name)
        return LutQuantParams(
            name=name,
            scalars=tuple(read_ints(self.refs / scalars_file)),
            tables=tuple(tuple(read_ints(self.refs / table_file)) for table_file in table_files),
            input_dtype=self.tensor_dtype(f"{stat_key}.input"),
            output_dtype=self.tensor_dtype(f"{stat_key}.output"),
            observed_range=self.observed_range(f"{stat_key}.output") or self.observed_range(stat_key),
        )

    def contract_for_table_case(self, name: str, kind: str, scalars_file: str, *table_files: str) -> OpQuantContract:
        stat_key = stem_to_stat_key(name)
        return OpQuantContract(
            name=name,
            kind=kind,
            stat_key=stat_key,
            input_dtype=self.tensor_dtype(f"{stat_key}.input"),
            output_dtype=self.tensor_dtype(f"{stat_key}.output"),
            observed_range=self.observed_range_group(stat_key),
            params=self.table_params(name, scalars_file, *table_files),
        )

