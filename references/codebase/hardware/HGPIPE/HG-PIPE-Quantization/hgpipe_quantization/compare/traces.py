"""Compare FakeQuant and torch integer tensor traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TraceComparisonResult:
    name: str
    left_runner: str
    right_runner: str
    elements: int
    passed: bool
    mismatches: int
    max_abs_error: float
    mean_abs_error: float

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "left_runner": self.left_runner,
            "right_runner": self.right_runner,
            "elements": self.elements,
            "passed": self.passed,
            "mismatches": self.mismatches,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
        }


def _trace_map(payload: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in payload}


def compare_trace_payloads(left_payload: list[dict[str, Any]], right_payload: list[dict[str, Any]]) -> list[TraceComparisonResult]:
    left = _trace_map(left_payload)
    right = _trace_map(right_payload)
    common = sorted(set(left) & set(right))
    results: list[TraceComparisonResult] = []
    for name in common:
        left_values = left[name].get("values")
        right_values = right[name].get("values")
        if left_values is None or right_values is None:
            raise ValueError(f"trace {name!r} is missing values; cannot compare exactly")
        l_arr = np.asarray(left_values, dtype=np.float64)
        r_arr = np.asarray(right_values, dtype=np.float64)
        shared = min(l_arr.size, r_arr.size)
        mismatches = abs(l_arr.size - r_arr.size)
        max_abs_error = 0.0
        mean_abs_error = 0.0
        if shared:
            diff = l_arr[:shared] - r_arr[:shared]
            abs_diff = np.abs(diff)
            mismatches += int(np.count_nonzero(diff))
            max_abs_error = float(np.max(abs_diff))
            mean_abs_error = float(np.mean(abs_diff))
        results.append(
            TraceComparisonResult(
                name=name,
                left_runner=str(left[name].get("runner", "left")),
                right_runner=str(right[name].get("runner", "right")),
                elements=int(max(l_arr.size, r_arr.size)),
                passed=mismatches == 0,
                mismatches=mismatches,
                max_abs_error=max_abs_error,
                mean_abs_error=mean_abs_error,
            )
        )
    return results


def write_comparison_markdown(results: list[TraceComparisonResult], path: Path) -> None:
    passed = sum(1 for result in results if result.passed)
    mismatches = sum(result.mismatches for result in results)
    lines = [
        "# Trace Comparison Report",
        "",
        f"- Cases: {passed}/{len(results)} passed",
        f"- Total mismatches: {mismatches}",
        "",
        "| Name | Left | Right | Elements | Mismatches | Max abs error | Mean abs error |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.left_runner} | {result.right_runner} | "
            f"{result.elements} | {result.mismatches} | {result.max_abs_error:.6g} | {result.mean_abs_error:.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

