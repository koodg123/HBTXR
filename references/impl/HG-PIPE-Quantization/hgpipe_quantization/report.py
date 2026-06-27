"""Report writers for quantization verification results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import VerificationResult


def result_to_dict(result: VerificationResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "kind": result.kind,
        "passed": result.passed,
        "elements": result.elements,
        "mismatches": result.mismatches,
        "metadata": result.metadata,
    }


def write_json(results: list[VerificationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([result_to_dict(result) for result in results], indent=2, sort_keys=True))


def write_markdown(results: list[VerificationResult], path: Path, source_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for result in results if result.passed)
    total_elements = sum(result.elements for result in results)
    total_mismatches = sum(result.mismatches for result in results)
    by_kind: dict[str, list[VerificationResult]] = {}
    for result in results:
        by_kind.setdefault(result.kind, []).append(result)

    lines = [
        "# HG-PIPE Quantization Verification Report",
        "",
        f"- Source: `{source_root}`",
        f"- Cases: {passed}/{len(results)} passed",
        f"- Elements checked: {total_elements}",
        f"- Total mismatches: {total_mismatches}",
        "",
        "## Case Summary",
        "",
        "| Kind | Cases | Passed | Elements | Mismatches |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, items in sorted(by_kind.items()):
        lines.append(
            f"| {kind} | {len(items)} | {sum(1 for item in items if item.passed)} | "
            f"{sum(item.elements for item in items)} | {sum(item.mismatches for item in items)} |"
        )

    lines.extend(["", "## Cases", "", "| Name | Kind | Elements | Mismatches | IO statistics key |", "|---|---|---:|---:|---|"])
    for result in sorted(results, key=lambda item: (item.kind, item.name)):
        lines.append(
            f"| {result.name} | {result.kind} | {result.elements} | {result.mismatches} | "
            f"{result.metadata.get('stat_key')} |"
        )

    path.write_text("\n".join(lines) + "\n")
