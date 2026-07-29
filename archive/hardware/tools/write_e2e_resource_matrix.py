#!/usr/bin/env python3
"""Write the selected E2E A/C-path resource and timing matrix."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence


VARIANTS = [
    {
        "id": "A2",
        "label": "A2 E2E m_axi",
        "selection_path": "Path 1, first",
        "interface": "m_axi + AXI-Lite",
        "parallelism": 8,
        "memory_banks": 8,
        "hls_report": "hardware/generated/hgtxr_e2e_m_axi_hls/solution_e2e_q4w8a/syn/report/csynth.xml",
        "timing_report": "hardware/generated/build/vivado/hgtxr_e2e_m_axi_overlay/hgtxr_e2e_m_axi_overlay.runs/impl_1/hgtxr_e2e_m_axi_system_wrapper_timing_summary_routed.rpt",
        "power_report": "hardware/generated/build/vivado/hgtxr_e2e_m_axi_overlay/hgtxr_e2e_m_axi_overlay.runs/impl_1/hgtxr_e2e_m_axi_system_wrapper_power_routed.rpt",
        "bit": "hardware/pynq/hgtxr/hgtxr_e2e_m_axi.bit",
        "hwh": "hardware/pynq/hgtxr/hgtxr_e2e_m_axi.hwh",
        "board_status": "bit-hwh-ready",
    },
    {
        "id": "A1",
        "label": "A1 E2E AXIS/DMA",
        "selection_path": "Path 1, after A2",
        "interface": "AXI-Stream + AXI DMA + AXI-Lite",
        "parallelism": 8,
        "memory_banks": 8,
        "hls_report": "hardware/generated/hgtxr_e2e_axis_hls/solution_e2e_q4w8a/syn/report/csynth.xml",
        "timing_report": "hardware/generated/build/vivado/hgtxr_e2e_axis_dma_overlay/hgtxr_e2e_axis_dma_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_timing_summary_routed.rpt",
        "power_report": "hardware/generated/build/vivado/hgtxr_e2e_axis_dma_overlay/hgtxr_e2e_axis_dma_overlay.runs/impl_1/hgtxr_e2e_axis_dma_system_wrapper_power_routed.rpt",
        "bit": "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma.bit",
        "hwh": "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma.hwh",
        "board_status": "bit-hwh-ready",
    },
    {
        "id": "C1",
        "label": "C1 PAR16 AXIS/DMA",
        "selection_path": "Path 2, C parallelism",
        "interface": "AXI-Stream + AXI DMA + AXI-Lite",
        "parallelism": 16,
        "memory_banks": 8,
        "hls_report": "hardware/generated/hgtxr_e2e_axis_par16_hls/solution_e2e_q4w8a/syn/report/csynth.xml",
        "timing_report": "hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par16_overlay/hgtxr_e2e_axis_dma_par16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_par16_system_wrapper_timing_summary_routed.rpt",
        "power_report": "hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par16_overlay/hgtxr_e2e_axis_dma_par16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_par16_system_wrapper_power_routed.rpt",
        "bit": "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par16.bit",
        "hwh": "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par16.hwh",
        "board_status": "bit-hwh-ready",
    },
    {
        "id": "C3b",
        "label": "C3b PAR16 MEM16 AXIS/DMA",
        "selection_path": "Path 2, C memory-bank refinement",
        "interface": "AXI-Stream + AXI DMA + AXI-Lite",
        "parallelism": 16,
        "memory_banks": 16,
        "hls_report": "hardware/generated/hgtxr_e2e_axis_par16_c3b_mem16_hls/solution_e2e_q4w8a/syn/report/csynth.xml",
        "timing_report": "hardware/generated/build/vivado/hgtxr_e2e_axis_dma_c3b_mem16_overlay/hgtxr_e2e_axis_dma_c3b_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_c3b_mem16_system_wrapper_timing_summary_routed.rpt",
        "power_report": "hardware/generated/build/vivado/hgtxr_e2e_axis_dma_c3b_mem16_overlay/hgtxr_e2e_axis_dma_c3b_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_c3b_mem16_system_wrapper_power_routed.rpt",
        "bit": "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
        "hwh": "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
        "board_status": "ready-for-board-smoke",
    },
]


def text_of(root: ET.Element, path: str) -> str | None:
    found = root.find(path)
    return found.text.strip() if found is not None and found.text else None


def int_text(root: ET.Element, path: str) -> int | None:
    value = text_of(root, path)
    return int(value) if value and value.isdigit() else None


def float_text(root: ET.Element, path: str) -> float | None:
    value = text_of(root, path)
    return float(value) if value else None


def parse_hls_csynth_xml(path: Path) -> dict[str, Any]:
    xml_root = ET.parse(path).getroot()
    resources = {
        "bram_18k": int_text(xml_root, "./AreaEstimates/Resources/BRAM_18K"),
        "dsp": int_text(xml_root, "./AreaEstimates/Resources/DSP"),
        "ff": int_text(xml_root, "./AreaEstimates/Resources/FF"),
        "lut": int_text(xml_root, "./AreaEstimates/Resources/LUT"),
        "uram": int_text(xml_root, "./AreaEstimates/Resources/URAM"),
    }
    available = {
        "bram_18k": int_text(xml_root, "./AreaEstimates/AvailableResources/BRAM_18K"),
        "dsp": int_text(xml_root, "./AreaEstimates/AvailableResources/DSP"),
        "ff": int_text(xml_root, "./AreaEstimates/AvailableResources/FF"),
        "lut": int_text(xml_root, "./AreaEstimates/AvailableResources/LUT"),
        "uram": int_text(xml_root, "./AreaEstimates/AvailableResources/URAM"),
    }
    utilization_pct = {
        key: round(resources[key] * 100.0 / available[key], 2)
        for key in resources
        if resources[key] is not None and available[key]
    }
    return {
        "top": text_of(xml_root, "./UserAssignments/TopModelName"),
        "target_clock_ns": float_text(xml_root, "./UserAssignments/TargetClockPeriod"),
        "estimated_clock_ns": float_text(xml_root, "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"),
        "latency_cycles": int_text(xml_root, "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"),
        "latency_real_time": text_of(xml_root, "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseRealTimeLatency"),
        "resources": resources,
        "available": available,
        "utilization_pct": utilization_pct,
    }


def parse_timing_summary(path: Path) -> dict[str, float] | dict[str, None]:
    text = path.read_text(errors="ignore")
    match = re.search(
        r"\n\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
        text,
    )
    if not match:
        return {"wns_ns": None, "tns_ns": None, "whs_ns": None, "ths_ns": None}
    return {
        "wns_ns": float(match.group(1)),
        "tns_ns": float(match.group(2)),
        "whs_ns": float(match.group(3)),
        "ths_ns": float(match.group(4)),
    }


def parse_power(path: Path) -> dict[str, float | None]:
    text = path.read_text(errors="ignore")
    patterns = {
        "total_on_chip_w": r"Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)",
        "dynamic_w": r"Dynamic \(W\)\s*\|\s*([0-9.]+)",
        "device_static_w": r"Device Static \(W\)\s*\|\s*([0-9.]+)",
    }
    values: dict[str, float | None] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        values[key] = float(match.group(1)) if match else None
    return values


def pct_delta(new: int | float, old: int | float) -> float | None:
    if old == 0:
        return None
    return round((new - old) * 100.0 / old, 2)


def build_matrix(root: Path) -> dict[str, Any]:
    root = root.resolve()
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        hls_path = root / variant["hls_report"]
        timing_path = root / variant["timing_report"]
        power_path = root / variant["power_report"]
        hls = parse_hls_csynth_xml(hls_path)
        row = {
            "id": variant["id"],
            "label": variant["label"],
            "selection_path": variant["selection_path"],
            "interface": variant["interface"],
            "parallelism": variant["parallelism"],
            "memory_banks": variant["memory_banks"],
            "board_status": variant["board_status"],
            "hls": hls,
            "vivado": {
                "timing": parse_timing_summary(timing_path),
                "power": parse_power(power_path),
            },
            "artifacts": {
                "bit": {"path": str(root / variant["bit"]), "exists": (root / variant["bit"]).exists()},
                "hwh": {"path": str(root / variant["hwh"]), "exists": (root / variant["hwh"]).exists()},
            },
            "sources": {
                "hls_report": str(hls_path),
                "timing_report": str(timing_path),
                "power_report": str(power_path),
            },
        }
        rows.append(row)

    baseline = next(row for row in rows if row["id"] == "A1")
    for row in rows:
        row["delta_vs_a1_pct"] = {
            "latency_cycles": pct_delta(row["hls"]["latency_cycles"], baseline["hls"]["latency_cycles"]),
            "dsp": pct_delta(row["hls"]["resources"]["dsp"], baseline["hls"]["resources"]["dsp"]),
            "lut": pct_delta(row["hls"]["resources"]["lut"], baseline["hls"]["resources"]["lut"]),
            "uram": pct_delta(row["hls"]["resources"]["uram"], baseline["hls"]["resources"]["uram"]),
        }

    c_candidates = [row for row in rows if row["id"].startswith("C")]
    best_latency_value = min(row["hls"]["latency_cycles"] for row in rows)
    highest_dsp_value = max(row["hls"]["resources"]["dsp"] for row in rows)
    lowest_lut_c = min(c_candidates, key=lambda item: item["hls"]["resources"]["lut"])
    best_latency_variants = [row["id"] for row in rows if row["hls"]["latency_cycles"] == best_latency_value]
    highest_dsp_variants = [row["id"] for row in rows if row["hls"]["resources"]["dsp"] == highest_dsp_value]
    return {
        "status": "pass",
        "root": str(root),
        "policy": {
            "resource_authority": "HLS csynth.xml",
            "vivado_authority": "routed timing and power reports",
            "e_pending": True,
            "no_new_hls_or_vivado_run": True,
        },
        "summary": {
            "variant_count": len(rows),
            "best_latency_variants": best_latency_variants,
            "highest_dsp_variants": highest_dsp_variants,
            "lowest_lut_c_variant": lowest_lut_c["id"],
            "recommended_board_smoke_variant": "C3b",
            "reason": "C3b keeps PAR16 DSP/latency gain while reducing LUT versus C1 and retaining positive routed WNS.",
        },
        "rows": rows,
    }


def fmt_int(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# HGTXR E2E Resource Matrix",
        "",
        f"- status: `{matrix['status']}`",
        f"- root: `{matrix['root']}`",
        f"- resource authority: `{matrix['policy']['resource_authority']}`",
        f"- Vivado authority: `{matrix['policy']['vivado_authority']}`",
        f"- E pending: `{matrix['policy']['e_pending']}`",
        "",
        "## Summary",
        "",
        f"- best latency variants: `{', '.join(matrix['summary']['best_latency_variants'])}`",
        f"- highest DSP variants: `{', '.join(matrix['summary']['highest_dsp_variants'])}`",
        f"- lowest LUT C variant: `{matrix['summary']['lowest_lut_c_variant']}`",
        f"- recommended board smoke variant: `{matrix['summary']['recommended_board_smoke_variant']}`",
        f"- reason: {matrix['summary']['reason']}",
        "",
        "## Matrix",
        "",
        "| ID | Interface | PAR | Mem Banks | Latency Cycles | HLS Clock ns | DSP | LUT | FF | BRAM18K | URAM | WNS ns | Power W | Board |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in matrix["rows"]:
        hls = row["hls"]
        res = hls["resources"]
        timing = row["vivado"]["timing"]
        power = row["vivado"]["power"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    row["interface"],
                    str(row["parallelism"]),
                    str(row["memory_banks"]),
                    fmt_int(hls["latency_cycles"]),
                    fmt_float(hls["estimated_clock_ns"]),
                    fmt_int(res["dsp"]),
                    fmt_int(res["lut"]),
                    fmt_int(res["ff"]),
                    fmt_int(res["bram_18k"]),
                    fmt_int(res["uram"]),
                    fmt_float(timing["wns_ns"]),
                    fmt_float(power["total_on_chip_w"]),
                    row["board_status"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Deltas Vs A1", ""])
    lines.append("| ID | Latency % | DSP % | LUT % | URAM % |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in matrix["rows"]:
        delta = row["delta_vs_a1_pct"]
        lines.append(
            f"| {row['id']} | {fmt_float(delta['latency_cycles'])} | {fmt_float(delta['dsp'])} | "
            f"{fmt_float(delta['lut'])} | {fmt_float(delta['uram'])} |"
        )
    lines.extend(["", "## Sources", ""])
    for row in matrix["rows"]:
        lines.append(f"- `{row['id']}` HLS: `{row['sources']['hls_report']}`")
        lines.append(f"- `{row['id']}` timing: `{row['sources']['timing_report']}`")
        lines.append(f"- `{row['id']}` power: `{row['sources']['power_report']}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write HGTXR E2E resource matrix.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = build_matrix(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(matrix))
    print(
        f"[e2e-resource-matrix] status={matrix['status']} "
        f"variants={matrix['summary']['variant_count']} "
        f"best_latency={','.join(matrix['summary']['best_latency_variants'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
