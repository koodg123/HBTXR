"""Inference result summaries for artifact-backed runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TopKEntry:
    index: int
    value: int | float

    def to_json(self) -> dict[str, int | float]:
        return {"index": self.index, "value": self.value}


@dataclass(frozen=True)
class InferenceResult:
    runner: str
    output_name: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    minimum: float
    maximum: float
    mean: float
    topk: tuple[TopKEntry, ...]
    values: tuple[int | float, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "runner": self.runner,
            "output_name": self.output_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "numel": self.numel,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "topk": [entry.to_json() for entry in self.topk],
            "values": list(self.values),
        }


@dataclass(frozen=True)
class InferenceComparison:
    left_runner: str
    right_runner: str
    elements: int
    passed: bool
    mismatches: int
    max_abs_error: float
    mean_abs_error: float
    top1_equal: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "left_runner": self.left_runner,
            "right_runner": self.right_runner,
            "elements": self.elements,
            "passed": self.passed,
            "mismatches": self.mismatches,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "top1_equal": self.top1_equal,
        }


def make_inference_result(
    *,
    runner: str,
    output_name: str,
    values: Iterable[int | float],
    shape: tuple[int, ...] | None = None,
    topk: int = 5,
) -> InferenceResult:
    arr = np.asarray(list(values)).reshape(-1)
    if shape is None:
        shape = (int(arr.size),)
    numeric = arr.astype(np.float64, copy=False) if arr.size else np.asarray([], dtype=np.float64)
    count = min(topk, int(arr.size))
    top_indices = np.argsort(numeric)[::-1][:count] if count else np.asarray([], dtype=np.int64)
    top_entries = tuple(TopKEntry(index=int(index), value=arr[index].item()) for index in top_indices)
    return InferenceResult(
        runner=runner,
        output_name=output_name,
        dtype=str(arr.dtype),
        shape=tuple(int(dim) for dim in shape),
        numel=int(arr.size),
        minimum=float(np.min(numeric)) if numeric.size else 0.0,
        maximum=float(np.max(numeric)) if numeric.size else 0.0,
        mean=float(np.mean(numeric)) if numeric.size else 0.0,
        topk=top_entries,
        values=tuple(value.item() if hasattr(value, "item") else value for value in arr),
    )


def compare_inference_results(left: dict[str, Any], right: dict[str, Any]) -> InferenceComparison:
    left_values = np.asarray(left["values"], dtype=np.float64)
    right_values = np.asarray(right["values"], dtype=np.float64)
    shared = min(left_values.size, right_values.size)
    mismatches = abs(left_values.size - right_values.size)
    max_abs_error = 0.0
    mean_abs_error = 0.0
    if shared:
        diff = left_values[:shared] - right_values[:shared]
        abs_diff = np.abs(diff)
        mismatches += int(np.count_nonzero(diff))
        max_abs_error = float(np.max(abs_diff))
        mean_abs_error = float(np.mean(abs_diff))
    left_top1 = left.get("topk", [{}])[0].get("index") if left.get("topk") else None
    right_top1 = right.get("topk", [{}])[0].get("index") if right.get("topk") else None
    return InferenceComparison(
        left_runner=str(left.get("runner", "left")),
        right_runner=str(right.get("runner", "right")),
        elements=int(max(left_values.size, right_values.size)),
        passed=mismatches == 0,
        mismatches=mismatches,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        top1_equal=left_top1 == right_top1,
    )



def _format_topk(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "(none)"
    parts = []
    for rank, entry in enumerate(entries):
        parts.append("#{}: index={} value={}".format(rank + 1, entry.get("index"), entry.get("value")))
    return ", ".join(parts)


def write_runner_pair_markdown(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    torch_int = payload["torch_int"]
    fakequant = payload["fakequant_graph"]
    comparison = payload["comparison"]
    status = "PASS" if comparison["passed"] else "FAIL"
    lines = [
        "# HG-PIPE Runner Comparison",
        "",
        "## Summary",
        "",
        "- Status: {}".format(status),
        "- Elements: {}".format(comparison["elements"]),
        "- Mismatches: {}".format(comparison["mismatches"]),
        "- Max abs error: {}".format(comparison["max_abs_error"]),
        "- Mean abs error: {}".format(comparison["mean_abs_error"]),
        "- Top-1 equal: {}".format(comparison["top1_equal"]),
        "",
        "## Outputs",
        "",
        "| Runner | Shape | DType | Min | Max | Mean | Top-k |",
        "|---|---:|---|---:|---:|---:|---|",
        "| {} | {} | {} | {} | {} | {} | {} |".format(torch_int["runner"], torch_int["shape"], torch_int["dtype"], torch_int["min"], torch_int["max"], torch_int["mean"], _format_topk(torch_int["topk"])),
        "| {} | {} | {} | {} | {} | {} | {} |".format(fakequant["runner"], fakequant["shape"], fakequant["dtype"], fakequant["min"], fakequant["max"], fakequant["mean"], _format_topk(fakequant["topk"])),
        "",
        "## Interpretation",
        "",
        "The FakeQuantizer-inserted artifact graph and the torch.int artifact graph are expected to match exactly for the recovered HG-PIPE reference input. Any nonzero mismatch indicates a numerical or graph-reconstruction regression.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
