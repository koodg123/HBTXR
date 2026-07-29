#!/usr/bin/env python3
"""Extract resource/timing/latency hints from implementation reports.

This is a deterministic first-pass parser for `impl_repos`. It intentionally
keeps provenance for every row so image-derived and report-derived values can be
compared later.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path


REPORT_SUFFIXES = {".rpt", ".log", ".syr", ".xml", ".txt"}

PATTERNS = [
    ("resource", "LUT", "#", re.compile(r"\b(?:LUTs?|CLB LUTs?)\b[^0-9]*(\d[\d,]*)", re.I)),
    ("resource", "FF", "#", re.compile(r"\b(?:FFs?|Flip-Flops?|CLB Registers?)\b[^0-9]*(\d[\d,]*)", re.I)),
    ("resource", "DSP", "#", re.compile(r"\b(?:DSPs?|DSP48E2|DSP48)\b[^0-9]*(\d[\d,]*)", re.I)),
    ("resource", "BRAM_18K", "#", re.compile(r"\b(?:BRAM_18K|Block RAM Tile|BRAM)\b[^0-9]*(\d[\d,]*)", re.I)),
    ("resource", "URAM", "#", re.compile(r"\b(?:URAM|UltraRAM)\b[^0-9]*(\d[\d,]*)", re.I)),
    ("timing", "WNS_ns", "ns", re.compile(r"\bWNS\b[^-0-9]*(-?\d+(?:\.\d+)?)", re.I)),
    ("timing", "TNS_ns", "ns", re.compile(r"\bTNS\b[^-0-9]*(-?\d+(?:\.\d+)?)", re.I)),
    ("timing", "Clk_Period_ns", "ns", re.compile(r"\b(?:period|clock period)\b[^0-9]*(\d+(?:\.\d+)?)\s*ns", re.I)),
    ("timing", "Achieved_Clk_MHz", "MHz", re.compile(r"\b(?:Fmax|Frequency)\b[^0-9]*(\d+(?:\.\d+)?)\s*MHz", re.I)),
    ("latency", "Latency_cycles", "cycles", re.compile(r"\bLatency\b[^0-9]*(\d[\d,]*)\s*(?:cycles?|clk)", re.I)),
    ("latency", "Latency_ms", "ms", re.compile(r"\bLatency\b[^0-9]*(\d+(?:\.\d+)?)\s*ms", re.I)),
]


def numeric(token: str) -> str:
    return token.replace(",", "")


def iter_report_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in REPORT_SUFFIXES:
            yield path


def extract_file(path: Path, run_id: str, target: str, fit_goal: str):
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError as exc:
        return [
            {
                "run_id": run_id,
                "experiment_name": "impl_repos",
                "fpga_target": target,
                "fit_goal": fit_goal,
                "source_type": "report_parsed",
                "source_ref": str(path),
                "source_confidence": "0",
                "metric_domain": "resource",
                "metric_name": "READ_ERROR",
                "metric_value": "0",
                "metric_unit": "#",
                "metric_context": "",
                "raw_token": str(exc),
                "normalization": "",
                "status": "needs_review",
                "qa_note": "file read failed",
                "recorded_utc": timestamp,
            }
        ]

    rows = []
    for lineno, line in enumerate(lines, start=1):
        for domain, name, unit, pattern in PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "experiment_name": "impl_repos",
                    "fpga_target": target,
                    "fit_goal": fit_goal,
                    "source_type": "report_parsed",
                    "source_ref": f"{path}:{lineno}",
                    "source_confidence": "0.7",
                    "metric_domain": domain,
                    "metric_name": name,
                    "metric_value": numeric(match.group(1)),
                    "metric_unit": unit,
                    "metric_context": "",
                    "raw_token": line.strip(),
                    "normalization": "regex_first_pass",
                    "status": "ok",
                    "qa_note": "",
                    "recorded_utc": timestamp,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl-root", default="../impl_repos")
    parser.add_argument("--out", default="docs/resources/deit_tiny_csyn_resource_timing_metrics.csv")
    parser.add_argument("--run-id", default="impl_repos_scan")
    parser.add_argument("--target", default="ZCU104")
    parser.add_argument("--fit-goal", default="zcu104_fit")
    args = parser.parse_args()

    impl_root = Path(args.impl_root).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "run_id",
        "experiment_name",
        "fpga_target",
        "fit_goal",
        "source_type",
        "source_ref",
        "source_confidence",
        "metric_domain",
        "metric_name",
        "metric_value",
        "metric_unit",
        "metric_context",
        "raw_token",
        "normalization",
        "status",
        "qa_note",
        "recorded_utc",
    ]

    rows = []
    if impl_root.exists():
        for report in iter_report_files(impl_root):
            rows.extend(extract_file(report, args.run_id, args.target, args.fit_goal))
    else:
        rows.append(
            {
                "run_id": args.run_id,
                "experiment_name": "impl_repos",
                "fpga_target": args.target,
                "fit_goal": args.fit_goal,
                "source_type": "report_parsed",
                "source_ref": str(impl_root),
                "source_confidence": "0",
                "metric_domain": "resource",
                "metric_name": "MISSING_IMPL_ROOT",
                "metric_value": "0",
                "metric_unit": "#",
                "metric_context": "",
                "raw_token": "",
                "normalization": "",
                "status": "missing",
                "qa_note": "impl root not found",
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

