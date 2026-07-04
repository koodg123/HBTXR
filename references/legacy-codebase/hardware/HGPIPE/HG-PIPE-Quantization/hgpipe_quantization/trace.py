"""Shared tensor trace schema for runtime comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TensorTrace:
    name: str
    runner: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    minimum: float
    maximum: float
    mean: float
    sample: tuple[float, ...]
    values: tuple[float, ...] | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "runner": self.runner,
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "numel": self.numel,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "sample": list(self.sample),
        }
        if self.values is not None:
            payload["values"] = list(self.values)
        return payload


def make_trace(
    *,
    name: str,
    runner: str,
    kind: str,
    values: Iterable[int | float],
    shape: tuple[int, ...] | None = None,
    include_values: bool = True,
    sample_size: int = 8,
) -> TensorTrace:
    arr = np.asarray(list(values)).reshape(-1)
    if shape is None:
        shape = (int(arr.size),)
    numeric = arr.astype(np.float64, copy=False) if arr.size else np.asarray([], dtype=np.float64)
    return TensorTrace(
        name=name,
        runner=runner,
        kind=kind,
        dtype=str(arr.dtype),
        shape=tuple(int(dim) for dim in shape),
        numel=int(arr.size),
        minimum=float(np.min(numeric)) if numeric.size else 0.0,
        maximum=float(np.max(numeric)) if numeric.size else 0.0,
        mean=float(np.mean(numeric)) if numeric.size else 0.0,
        sample=tuple(float(value) for value in numeric[:sample_size]),
        values=tuple(float(value) for value in numeric) if include_values else None,
    )

