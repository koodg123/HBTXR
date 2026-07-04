from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.optim.common import resolve_optimizer_cfg, resolve_optimizer_modifiers
from src.optim.registry import get_optimizer_metadata


def optimizer_candidate_label(name: str, modifiers: dict[str, bool]) -> str:
    suffixes = [flag for flag, enabled in sorted(modifiers.items()) if enabled]
    return "__".join([name, *suffixes]) if suffixes else name


def expand_optimizer_pool_candidates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    training_cfg = cfg.get("training") or {}
    pool_cfg = dict(training_cfg.get("optimizer_pool") or {})
    candidates = list(pool_cfg.get("candidates") or [])
    base_resolved = resolve_optimizer_cfg(training_cfg)
    base_modifiers = resolve_optimizer_modifiers(training_cfg)
    implemented_only = bool(pool_cfg.get("implemented_only", True))

    expanded: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        resolved = deepcopy(base_resolved)
        modifiers = dict(base_modifiers)
        if isinstance(candidate, str):
            resolved["name"] = candidate.strip().lower()
        else:
            candidate = dict(candidate or {})
            if "name" in candidate:
                resolved["name"] = str(candidate["name"]).strip().lower()
            for key in ("lr", "weight_decay", "eps"):
                if key in candidate:
                    resolved[key] = float(candidate[key])
            if "betas" in candidate:
                betas = candidate["betas"]
                resolved["betas"] = (float(betas[0]), float(betas[1]))
            if "kwargs" in candidate:
                resolved["kwargs"] = {**resolved["kwargs"], **dict(candidate["kwargs"] or {})}
            if "modifiers" in candidate:
                modifiers.update({k: bool(v) for k, v in dict(candidate["modifiers"] or {}).items()})

        metadata = get_optimizer_metadata(resolved["name"])
        if implemented_only and not metadata["implemented"]:
            continue
        expanded.append(
            {
                "index": index,
                "label": optimizer_candidate_label(resolved["name"], modifiers),
                "resolved": resolved,
                "modifiers": modifiers,
                "metadata": metadata,
            }
        )
    return expanded


def write_optimizer_pool_report(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "optimizer_pool_report.json"
    summary_path = target_dir / "optimizer_pool_summary.csv"
    rankings_path = target_dir / "optimizer_pool_rankings.json"

    report_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    reverse = not metric_name.lower().startswith("loss")
    rankings = sorted(rows, key=lambda row: float(row.get(metric_name, float("-inf") if reverse else float("inf"))), reverse=reverse)
    rankings_path.write_text(json.dumps(rankings, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "rankings": str(rankings_path),
    }
