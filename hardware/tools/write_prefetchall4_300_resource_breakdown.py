#!/usr/bin/env python3
"""Write resource breakdown evidence for the active 300 MHz prefetch-all4 build."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16"
OVERLAY = f"hgtxr_e2e_axis_dma_{PROFILE}_overlay"
WRAPPER = "hgtxr_e2e_axis_dma_system_wrapper"
RUNS_ROOT = HARDWARE_ROOT / "generated" / "build" / "vivado" / OVERLAY / f"{OVERLAY}.runs"
IMPL_ROOT = RUNS_ROOT / "impl_1"

TOP_PLACED_REPORT = IMPL_ROOT / f"{WRAPPER}_utilization_placed.rpt"
HIERARCHICAL_IMPLEMENTED_REPORT = IMPL_ROOT / (
    "hgtxr_e2e_axis_dma_"
    f"{PROFILE}_utilization_hierarchical_implemented.rpt"
)
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_resource_breakdown_2026_06_29.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_resource_breakdown_2026_06_29.md"

MAJOR_BLOCK_REPORTS = {
    "pl_hgtxr_e2e_axis_top": RUNS_ROOT
    / "hgtxr_e2e_axis_dma_system_hgtxr_e2e_axis_top_0_0_synth_1"
    / "hgtxr_e2e_axis_dma_system_hgtxr_e2e_axis_top_0_0_utilization_synth.rpt",
    "axi_dma_in": RUNS_ROOT
    / "hgtxr_e2e_axis_dma_system_axi_dma_in_0_synth_1"
    / "hgtxr_e2e_axis_dma_system_axi_dma_in_0_utilization_synth.rpt",
    "axi_dma_out": RUNS_ROOT
    / "hgtxr_e2e_axis_dma_system_axi_dma_out_0_synth_1"
    / "hgtxr_e2e_axis_dma_system_axi_dma_out_0_utilization_synth.rpt",
    "axi_mem_interconnect": RUNS_ROOT
    / "hgtxr_e2e_axis_dma_system_axi_mem_0_synth_1"
    / "hgtxr_e2e_axis_dma_system_axi_mem_0_utilization_synth.rpt",
    "axi_control_interconnect": RUNS_ROOT
    / "hgtxr_e2e_axis_dma_system_axi_ctrl_0_synth_1"
    / "hgtxr_e2e_axis_dma_system_axi_ctrl_0_utilization_synth.rpt",
    "psu": RUNS_ROOT
    / "hgtxr_e2e_axis_dma_system_psu_0_synth_1"
    / "hgtxr_e2e_axis_dma_system_psu_0_utilization_synth.rpt",
}

MAJOR_BLOCK_INSTANCE_PATTERNS = {
    "pl_hgtxr_e2e_axis_top": ["hgtxr_e2e_axis_top_0"],
    "axi_dma_in": ["axi_dma_in"],
    "axi_dma_out": ["axi_dma_out"],
    "axi_mem_interconnect": ["axi_mem"],
    "axi_control_interconnect": ["axi_ctrl"],
    "psu": ["psu"],
}

RESOURCE_ROWS = {
    "clb_luts": "CLB LUTs",
    "lut_as_logic": "LUT as Logic",
    "lut_as_memory": "LUT as Memory",
    "clb_registers": "CLB Registers",
    "block_ram_tile": "Block RAM Tile",
    "ramb36_fifo": "RAMB36/FIFO",
    "ramb18": "RAMB18",
    "uram": "URAM",
    "dsp": "DSPs",
}

PRIMARY_RESOURCES = ["clb_luts", "clb_registers", "block_ram_tile", "uram", "dsp"]


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def clean_label(label: str) -> str:
    return label.replace("*", "").strip()


def parse_resource_table(text: str) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        label = clean_label(cells[0])
        for key, row_label in RESOURCE_ROWS.items():
            if label != row_label:
                continue
            try:
                used = parse_number(cells[1])
            except ValueError:
                continue
            try:
                available = parse_number(cells[4])
            except ValueError:
                available = None
            try:
                util_pct = parse_number(cells[5])
            except ValueError:
                util_pct = None
            rows[key] = {
                "used": used,
                "available": available,
                "device_util_pct": util_pct,
            }
    return rows


def split_table_line(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def normalize_header(value: str) -> str:
    value = clean_label(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def find_header_index(headers: list[str], candidates: Sequence[str]) -> int | None:
    normalized = [normalize_header(header) for header in headers]
    for candidate in candidates:
        candidate_norm = normalize_header(candidate)
        for index, header in enumerate(normalized):
            if header == candidate_norm or candidate_norm in header:
                return index
    return None


def parse_hierarchical_table(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = split_table_line(line)
        normalized = [normalize_header(cell) for cell in cells]
        if (
            "instance" in normalized
            and any("lut" in cell for cell in normalized)
            and any(cell in {"ffs", "ff", "clb_registers"} or "register" in cell for cell in normalized)
        ):
            header = cells
            continue
        if header is None or len(cells) < 4:
            continue
        if clean_label(cells[0]).lower() in {"instance", "site type"}:
            continue
        instance_idx = find_header_index(header, ["Instance"])
        if instance_idx is None or instance_idx >= len(cells):
            continue
        instance = cells[instance_idx]
        if not instance or set(instance) <= {"-", " "}:
            continue
        rows.append({"instance": instance, "cells": cells, "header": header})
    return rows


def cell_float(cells: list[str], headers: list[str], candidates: Sequence[str]) -> float | None:
    index = find_header_index(headers, candidates)
    if index is None or index >= len(cells):
        return None
    try:
        return parse_number(cells[index])
    except ValueError:
        return None


def hierarchical_row_resources(
    row: Mapping[str, Any],
    top_rows: Mapping[str, Mapping[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    cells = list(row.get("cells") or [])
    headers = list(row.get("header") or [])
    ramb36 = cell_float(cells, headers, ["RAMB36", "RAMB36/FIFO"])
    ramb18 = cell_float(cells, headers, ["RAMB18"])
    bram_tile = cell_float(cells, headers, ["Block RAM Tile", "BRAM Tile", "BRAM"])
    if bram_tile is None:
        bram_tile = (ramb36 or 0.0) + 0.5 * (ramb18 or 0.0) if ramb36 is not None or ramb18 is not None else None
    rows = {
        "clb_luts": {"used": cell_float(cells, headers, ["Total LUTs", "CLB LUTs", "LUTs"])},
        "clb_registers": {"used": cell_float(cells, headers, ["FFs", "CLB Registers", "Registers"])},
        "block_ram_tile": {"used": bram_tile},
        "uram": {"used": cell_float(cells, headers, ["URAM"])},
        "dsp": {"used": cell_float(cells, headers, ["DSP Blocks", "DSPs", "DSP"])},
    }
    for name, value in rows.items():
        available = (top_rows.get(name) or {}).get("available")
        used = value.get("used")
        value["available"] = available
        value["device_util_pct"] = (
            100.0 * float(used) / float(available)
            if isinstance(used, (int, float)) and isinstance(available, (int, float)) and available > 0
            else None
        )
    return add_share_of_top(rows, top_rows)


def find_hierarchical_block_row(rows: Sequence[Mapping[str, Any]], patterns: Sequence[str]) -> Mapping[str, Any] | None:
    lowered_patterns = [pattern.lower() for pattern in patterns]
    for row in rows:
        instance = str(row.get("instance") or "").lower()
        if any(pattern in instance for pattern in lowered_patterns):
            return row
    return None


def add_share_of_top(
    rows: dict[str, dict[str, float | None]],
    top_rows: Mapping[str, Mapping[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    enriched: dict[str, dict[str, float | None]] = {}
    for key, row in rows.items():
        used = row.get("used")
        top_used = (top_rows.get(key) or {}).get("used")
        share = None
        if isinstance(used, (int, float)) and isinstance(top_used, (int, float)) and top_used > 0:
            share = 100.0 * float(used) / float(top_used)
        enriched[key] = {**row, "share_of_placed_top_pct": share}
    return enriched


def normalize_rows(rows: Mapping[str, Mapping[str, float | None]]) -> dict[str, dict[str, float | None]]:
    return {key: dict(rows.get(key) or {}) for key in PRIMARY_RESOURCES}


def block_entry(name: str, report: Path, top_rows: Mapping[str, Mapping[str, float | None]]) -> dict[str, Any]:
    rows = parse_resource_table(read_text(report))
    return {
        "name": name,
        "report": str(report),
        "exists": report.exists(),
        "evidence_level": "ooc_synth_utilization",
        "resources": normalize_rows(add_share_of_top(rows, top_rows)),
    }


def hierarchical_block_entry(
    name: str,
    row: Mapping[str, Any],
    report: Path,
    top_rows: Mapping[str, Mapping[str, float | None]],
) -> dict[str, Any]:
    return {
        "name": name,
        "report": str(report),
        "exists": report.exists(),
        "evidence_level": "implemented_hierarchical_utilization",
        "instance": row.get("instance"),
        "resources": normalize_rows(hierarchical_row_resources(row, top_rows)),
    }


def build_breakdown(
    top_report: Path = TOP_PLACED_REPORT,
    major_reports: Mapping[str, Path] = MAJOR_BLOCK_REPORTS,
    hierarchical_report: Path = HIERARCHICAL_IMPLEMENTED_REPORT,
) -> dict[str, Any]:
    top_rows = parse_resource_table(read_text(top_report))
    top_resources = normalize_rows(add_share_of_top(top_rows, top_rows))
    hierarchical_rows = parse_hierarchical_table(read_text(hierarchical_report)) if hierarchical_report.exists() else []
    blocks: dict[str, dict[str, Any]] = {}
    for name, report in major_reports.items():
        row = find_hierarchical_block_row(hierarchical_rows, MAJOR_BLOCK_INSTANCE_PATTERNS.get(name, [name]))
        if row is not None:
            blocks[name] = hierarchical_block_entry(name, row, hierarchical_report, top_rows)
        else:
            blocks[name] = block_entry(name, report, top_rows)
    missing_primary = [
        resource
        for resource in PRIMARY_RESOURCES
        if not top_resources.get(resource) or top_resources[resource].get("used") is None
    ]
    missing_block_reports = [name for name, block in blocks.items() if not block["exists"]]
    hgtxr_ok = bool(blocks.get("pl_hgtxr_e2e_axis_top", {}).get("exists"))
    status = "pass" if not missing_primary and not missing_block_reports and hgtxr_ok else "partial"
    return {
        "status": status,
        "profile": PROFILE,
        "methodology": (
            "Top resources are parsed from Vivado placed utilization. Major-block resources prefer "
            "the implemented hierarchical utilization report when present; otherwise they fall back "
            "to Vivado OOC synth utilization reports for each BD/IP block."
        ),
        "hierarchical_report": {
            "report": str(hierarchical_report),
            "exists": hierarchical_report.exists(),
            "rows_parsed": len(hierarchical_rows),
        },
        "top": {
            "name": "hgtxr_e2e_axis_dma_system_wrapper",
            "report": str(top_report),
            "exists": top_report.exists(),
            "evidence_level": "placed_top_utilization",
            "resources": top_resources,
        },
        "major_blocks": blocks,
        "summary": {
            "missing_primary_top_resources": missing_primary,
            "missing_block_reports": missing_block_reports,
            "block_count": len(blocks),
            "highest_pressure": highest_pressure(top_resources),
            "implemented_hierarchical_blocks": [
                name for name, block in blocks.items() if block.get("evidence_level") == "implemented_hierarchical_utilization"
            ],
        },
    }


def highest_pressure(resources: Mapping[str, Mapping[str, float | None]]) -> dict[str, Any]:
    candidates = []
    for name, row in resources.items():
        value = row.get("device_util_pct")
        if isinstance(value, (int, float)):
            candidates.append((float(value), name))
    if not candidates:
        return {"resource": None, "device_util_pct": None}
    pct, name = max(candidates)
    return {"resource": name, "device_util_pct": pct}


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(breakdown: dict[str, Any]) -> str:
    lines = [
        "# Prefetch-All4 300 MHz Resource Breakdown",
        "",
        f"- status: `{breakdown['status']}`",
        f"- profile: `{breakdown['profile']}`",
        f"- methodology: {breakdown['methodology']}",
        "",
        "## Top Placed Utilization",
        "",
        "| Resource | Used | Available | Device Util % |",
        "|---|---:|---:|---:|",
    ]
    for name, row in breakdown["top"]["resources"].items():
        lines.append(
            f"| {name} | {fmt(row.get('used'))} | {fmt(row.get('available'))} | {fmt(row.get('device_util_pct'))} |"
        )
    lines.extend(
        [
            "",
            "## Major Block Breakdown",
            "",
            f"- hierarchical report: `{breakdown['hierarchical_report']['report']}`",
            f"- hierarchical report exists: `{breakdown['hierarchical_report']['exists']}`",
            f"- hierarchical rows parsed: `{breakdown['hierarchical_report']['rows_parsed']}`",
            "",
            "| Block | Evidence | LUT | FF | BRAM Tile | URAM | DSP | LUT Share Of Placed Top % | DSP Share Of Placed Top % |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, block in breakdown["major_blocks"].items():
        res = block["resources"]
        lines.append(
            f"| {name} | {block['evidence_level']} | {fmt(res['clb_luts'].get('used'))} | {fmt(res['clb_registers'].get('used'))} | "
            f"{fmt(res['block_ram_tile'].get('used'))} | {fmt(res['uram'].get('used'))} | "
            f"{fmt(res['dsp'].get('used'))} | {fmt(res['clb_luts'].get('share_of_placed_top_pct'))} | "
            f"{fmt(res['dsp'].get('share_of_placed_top_pct'))} |"
        )
    lines.extend(["", "## Sources", ""])
    lines.append(f"- top: `{breakdown['top']['report']}`")
    for name, block in breakdown["major_blocks"].items():
        lines.append(f"- {name}: `{block['report']}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write prefetch-all4 300 MHz resource breakdown evidence.")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    breakdown = build_breakdown()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(breakdown, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(breakdown))
    print(json.dumps(breakdown, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
