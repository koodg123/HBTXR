#!/usr/bin/env python3
"""Write the current patch32+dtok4 300 MHz goal status report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16"
PROJECT = f"hgtxr_e2e_axis_{PROFILE}_no_board"
SEARCH_PROJECT = f"hgtxr_e2e_axis_{PROFILE}_search_only_no_board"
TRACK_PROJECT = f"hgtxr_e2e_axis_{PROFILE}_track_only_no_board"
OVERLAY = f"hgtxr_e2e_axis_dma_{PROFILE}_overlay"
WRAPPER = "hgtxr_e2e_axis_dma_system_wrapper"

SOLUTION = "solution_e2e_q4w8a"
SIGNOFF_ROOT = HARDWARE_ROOT / "generated" / "signoff"
HLS_ROOT = HARDWARE_ROOT / "generated"
VIVADO_IMPL = (
    HARDWARE_ROOT
    / "generated"
    / "build"
    / "vivado"
    / OVERLAY
    / f"{OVERLAY}.runs"
    / "impl_1"
)

COMBINED_HLS_REPORT = HLS_ROOT / PROJECT / SOLUTION / "syn" / "report" / "hgtxr_e2e_axis_top_csynth.rpt"
SEARCH_HLS_REPORT = HLS_ROOT / SEARCH_PROJECT / SOLUTION / "syn" / "report" / "hgtxr_e2e_axis_top_csynth.rpt"
TRACK_HLS_REPORT = HLS_ROOT / TRACK_PROJECT / SOLUTION / "syn" / "report" / "hgtxr_e2e_axis_top_csynth.rpt"
TIMING_REPORT = VIVADO_IMPL / f"{WRAPPER}_timing_summary_postroute_physopted.rpt"
ARTIFACT = f"hgtxr_e2e_axis_dma_{PROFILE}"
UTILIZATION_REPORT = VIVADO_IMPL / f"{ARTIFACT}_utilization_implemented.rpt"
POWER_REPORT = VIVADO_IMPL / f"{ARTIFACT}_power_implemented.rpt"
ROUTE_REPORT = VIVADO_IMPL / f"{ARTIFACT}_route_status_implemented.rpt"
CONTRACT_AUDIT_JSON = SIGNOFF_ROOT / "patch32_dtok4_300_contract_audit_2026_06_30.json"
BOARD_LATENCY_RUN_JSON = SIGNOFF_ROOT / "par32_patch32_dtok4_300_board_latency_run_2026_06_30.json"
BOARD_LATENCY_GATE_JSON = SIGNOFF_ROOT / "par32_patch32_dtok4_300_board_latency_gate_2026_06_30.json"
BUNDLE_VALIDATION_JSONS = {
    "search": SIGNOFF_ROOT / "par32_patch32_dtok4_300_search_bundle_validation_2026_06_30.json",
    "track": SIGNOFF_ROOT / "par32_patch32_dtok4_300_track_bundle_validation_2026_06_30.json",
    "hybrid_10_90": SIGNOFF_ROOT / "par32_patch32_dtok4_300_hybrid_10_90_bundle_validation_2026_06_30.json",
}
DEFAULT_JSON = SIGNOFF_ROOT / "patch32_dtok4_300_goal_status_2026_06_30.json"
DEFAULT_MARKDOWN = SIGNOFF_ROOT / "patch32_dtok4_300_goal_status_2026_06_30.md"

PL_CLOCK_MHZ = 300.0
SEARCH_RATIO = 0.10
TRACK_RATIO = 0.90
SEARCH_TARGET_MS = 4.0
TRACK_TARGET_MS = 1.0
EXPERIMENTAL_WNS_FLOOR_NS = -0.5
OFFICIAL_WNS_FLOOR_NS = 0.0


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="ignore"))
    except json.JSONDecodeError:
        return {}


def number(value: str) -> float:
    return float(value.replace(",", ""))


def cycles_to_ms(cycles: int, clock_mhz: float = PL_CLOCK_MHZ) -> float:
    return cycles / (clock_mhz * 1000.0)


def parse_hls_mode(text: str) -> dict[str, Any]:
    match = re.search(
        r"\|\s*([0-9,]+)\s*\|\s*([0-9,]+)\s*\|\s*([0-9.]+)\s+(\w+)\s*\|\s*"
        r"([0-9.]+)\s+(\w+)\s*\|\s*([0-9,]+)\s*\|\s*([0-9,]+)\s*\|",
        text,
    )
    if not match:
        return {"status": "missing"}
    min_cycles = int(match.group(1).replace(",", ""))
    max_cycles = int(match.group(2).replace(",", ""))
    min_interval = int(match.group(7).replace(",", ""))
    max_interval = int(match.group(8).replace(",", ""))
    return {
        "status": "available",
        "latency_min_cycles": min_cycles,
        "latency_max_cycles": max_cycles,
        "interval_min_cycles": min_interval,
        "interval_max_cycles": max_interval,
        "latency_min_report": f"{match.group(3)} {match.group(4)}",
        "latency_max_report": f"{match.group(5)} {match.group(6)}",
        "latency_max_ms_at_300mhz": cycles_to_ms(max_cycles),
        "interval_max_ms_at_300mhz": cycles_to_ms(max_interval),
    }


def parse_instances(text: str) -> dict[str, dict[str, Any]]:
    instances: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"\|[^|]*\|(?P<module>hgtxr_[a-zA-Z0-9_]+)\s*\|\s*"
        r"(?P<min>[0-9,]+)\s*\|\s*(?P<max>[0-9,]+)\s*\|[^|]*\|[^|]*\|\s*"
        r"(?P<int_min>[0-9,]+)\s*\|\s*(?P<int_max>[0-9,]+)\s*\|"
    )
    for match in pattern.finditer(text):
        module = match.group("module")
        instances[module] = {
            "latency_min_cycles": int(match.group("min").replace(",", "")),
            "latency_max_cycles": int(match.group("max").replace(",", "")),
            "interval_min_cycles": int(match.group("int_min").replace(",", "")),
            "interval_max_cycles": int(match.group("int_max").replace(",", "")),
        }
    return instances


def parse_timing(text: str) -> dict[str, Any]:
    summary = re.search(
        r"\n\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s+([0-9]+)\s+[0-9]+\s+"
        r"(-?[0-9.]+)\s+(-?[0-9.]+)",
        text,
    )
    clock = re.search(r"clk_pl_0\s+\{[^}]+\}\s+([0-9.]+)\s+([0-9.]+)", text)
    wns = number(summary.group(1)) if summary else None
    return {
        "status": "available" if summary else "missing",
        "wns_ns": wns,
        "tns_ns": number(summary.group(2)) if summary else None,
        "setup_failing_endpoints": int(summary.group(3)) if summary else None,
        "whs_ns": number(summary.group(4)) if summary else None,
        "ths_ns": number(summary.group(5)) if summary else None,
        "period_ns": number(clock.group(1)) if clock else None,
        "frequency_mhz": number(clock.group(2)) if clock else None,
        "experimental_status": "pass"
        if wns is not None and wns >= EXPERIMENTAL_WNS_FLOOR_NS
        else "missing"
        if wns is None
        else "fail",
        "official_status": "pass"
        if wns is not None and wns >= OFFICIAL_WNS_FLOOR_NS
        else "missing"
        if wns is None
        else "fail",
    }


def parse_utilization(text: str) -> dict[str, Any]:
    def row(label: str) -> dict[str, Any] | None:
        match = re.search(
            rf"\|\s*{re.escape(label)}\s*\|\s*([0-9.]+)\s*\|\s*[0-9.]+\s*\|\s*"
            rf"[0-9.]+\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)",
            text,
        )
        if not match:
            return None
        return {
            "used": number(match.group(1)),
            "available": number(match.group(2)),
            "util_pct": number(match.group(3)),
        }

    resources = {
        "clb_luts": row("CLB LUTs"),
        "lut_as_logic": row("LUT as Logic"),
        "lut_as_memory": row("LUT as Memory"),
        "clb_registers": row("CLB Registers"),
        "block_ram_tile": row("Block RAM Tile"),
        "uram": row("URAM"),
        "dsp": row("DSPs"),
    }
    available = [value for value in resources.values() if value is not None]
    fit = bool(available) and all(value["used"] <= value["available"] for value in available)
    return {"status": "pass" if fit else "missing", "resources": resources}


def parse_power(text: str) -> dict[str, Any]:
    def value(label: str) -> float | None:
        match = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([0-9.]+)", text)
        return number(match.group(1)) if match else None

    return {
        "status": "available" if "Total On-Chip Power" in text else "missing",
        "total_on_chip_w": value("Total On-Chip Power (W)"),
        "dynamic_w": value("Dynamic (W)"),
        "device_static_w": value("Device Static (W)"),
        "clock_w": value("Clocks"),
        "clb_logic_w": value("CLB Logic"),
        "signals_w": value("Signals"),
        "block_ram_w": value("Block RAM"),
        "uram_w": value("URAM"),
        "dsp_w": value("DSPs"),
        "ps8_w": value("PS8"),
        "ps_static_w": value("PS Static"),
        "pl_static_w": value("PL Static"),
        "methodology": "Vivado implemented vectorless estimate; not mode-specific, not board rail power, and no sensor I/O rail included.",
    }


def parse_route(text: str) -> dict[str, Any]:
    match = re.search(r"(?:Fully Routed Nets|# of fully routed nets)[^:]*:\s*([0-9,]+)", text, re.IGNORECASE)
    errors = re.search(r"(?:routing errors|# of nets with routing errors)[^:]*:\s*([0-9,]+)", text, re.IGNORECASE)
    return {
        "status": "available" if text else "missing",
        "fully_routed_nets": int(match.group(1).replace(",", "")) if match else None,
        "routing_errors": int(errors.group(1).replace(",", "")) if errors else None,
    }


def mode_status(mode: dict[str, Any], target_ms: float) -> str:
    latency = mode.get("interval_max_ms_at_300mhz")
    if latency is None:
        return "missing"
    return "pass" if latency <= target_ms else "fail"


def build_status() -> dict[str, Any]:
    search = parse_hls_mode(read_text(SEARCH_HLS_REPORT))
    track = parse_hls_mode(read_text(TRACK_HLS_REPORT))
    search_instances = parse_instances(read_text(SEARCH_HLS_REPORT))
    track_instances = parse_instances(read_text(TRACK_HLS_REPORT))
    timing = parse_timing(read_text(TIMING_REPORT))
    utilization = parse_utilization(read_text(UTILIZATION_REPORT))
    power = parse_power(read_text(POWER_REPORT))
    route = parse_route(read_text(ROUTE_REPORT))
    contract = read_json(CONTRACT_AUDIT_JSON)
    board_run = read_json(BOARD_LATENCY_RUN_JSON)
    board_gate = read_json(BOARD_LATENCY_GATE_JSON)
    bundle_validations = {name: read_json(path) for name, path in BUNDLE_VALIDATION_JSONS.items()}

    search_ii = search.get("interval_max_cycles")
    track_ii = track.get("interval_max_cycles")
    hybrid_cycles = None
    hybrid_ms = None
    if isinstance(search_ii, int) and isinstance(track_ii, int):
        hybrid_cycles = SEARCH_RATIO * search_ii + TRACK_RATIO * track_ii
        hybrid_ms = cycles_to_ms(hybrid_cycles)

    search_target_cycles = int(SEARCH_TARGET_MS * PL_CLOCK_MHZ * 1000)
    track_target_cycles = int(TRACK_TARGET_MS * PL_CLOCK_MHZ * 1000)

    status = {
        "profile": PROFILE,
        "status": "partial-misses-latency-targets",
        "rules": {
            "pl_clock_mhz": PL_CLOCK_MHZ,
            "search_ratio": SEARCH_RATIO,
            "track_ratio": TRACK_RATIO,
            "search_target_ms": SEARCH_TARGET_MS,
            "track_target_ms": TRACK_TARGET_MS,
            "search_target_cycles": search_target_cycles,
            "track_target_cycles": track_target_cycles,
            "experimental_wns_floor_ns": EXPERIMENTAL_WNS_FLOOR_NS,
            "official_wns_floor_ns": OFFICIAL_WNS_FLOOR_NS,
        },
        "latency": {
            "search": search | {
                "target_status": mode_status(search, SEARCH_TARGET_MS),
                "target_gap_cycles": search_ii - search_target_cycles if isinstance(search_ii, int) else None,
                "target_gap_ms": cycles_to_ms(search_ii - search_target_cycles)
                if isinstance(search_ii, int)
                else None,
                "instances": search_instances,
            },
            "track": track | {
                "target_status": mode_status(track, TRACK_TARGET_MS),
                "target_gap_cycles": track_ii - track_target_cycles if isinstance(track_ii, int) else None,
                "target_gap_ms": cycles_to_ms(track_ii - track_target_cycles)
                if isinstance(track_ii, int)
                else None,
                "instances": track_instances,
            },
            "hybrid_10_90": {
                "status": "hls-force-mode-derived" if hybrid_cycles is not None else "missing",
                "interval_cycles_expected": hybrid_cycles,
                "interval_ms_expected_at_300mhz": hybrid_ms,
                "throughput_invocations_per_s": 1000.0 / hybrid_ms if hybrid_ms else None,
                "worst_case_mode": "search",
                "worst_case_interval_cycles": search_ii,
                "worst_case_interval_ms_at_300mhz": search.get("interval_max_ms_at_300mhz"),
            },
        },
        "timing": timing,
        "route": route,
        "utilization": utilization,
        "power": power,
        "contract_audit": {
            "status": contract.get("status", "missing"),
            "checks_pass": (contract.get("summary") or {}).get("checks_pass"),
            "checks_total": (contract.get("summary") or {}).get("checks_total"),
            "interpretation": contract.get("interpretation"),
        },
        "board_plumbing": {
            "status": "pass"
            if all((payload or {}).get("status") == "pass" for payload in bundle_validations.values())
            and board_run.get("status") == "dry-run"
            and board_gate.get("status") == "missing"
            else "partial",
            "bundle_validations": {
                name: {
                    "status": payload.get("status", "missing"),
                    "variant": payload.get("variant"),
                    "errors": payload.get("errors", []),
                }
                for name, payload in bundle_validations.items()
            },
            "remote_dry_run_status": board_run.get("status", "missing"),
            "remote_profiles": sorted((board_run.get("profiles") or {}).keys()),
            "board_latency_gate_status": board_gate.get("status", "missing"),
            "board_latency_gate_summary": board_gate.get("summary"),
        },
        "measurement_gaps": [
            "mode-specific Search/Track power needs SAIF or board rail measurement",
            "board Search/Track/hybrid 10:90 execution is plumbed, but board result JSON is still missing for p95/p99 and DMA bandwidth",
            "official timing-clean signoff is still missing because WNS is negative",
            "ROM constants are synthetic/patterned unless a learned-weight export-to-ROM flow is later attached",
        ],
        "sources": {
            "combined_hls_report": str(COMBINED_HLS_REPORT),
            "search_hls_report": str(SEARCH_HLS_REPORT),
            "track_hls_report": str(TRACK_HLS_REPORT),
            "timing_report": str(TIMING_REPORT),
            "utilization_report": str(UTILIZATION_REPORT),
            "power_report": str(POWER_REPORT),
            "route_report": str(ROUTE_REPORT),
            "contract_audit_json": str(CONTRACT_AUDIT_JSON),
            "board_latency_run_json": str(BOARD_LATENCY_RUN_JSON),
            "board_latency_gate_json": str(BOARD_LATENCY_GATE_JSON),
            "search_bundle_validation_json": str(BUNDLE_VALIDATION_JSONS["search"]),
            "track_bundle_validation_json": str(BUNDLE_VALIDATION_JSONS["track"]),
            "hybrid_bundle_validation_json": str(BUNDLE_VALIDATION_JSONS["hybrid_10_90"]),
        },
    }
    if (
        status["contract_audit"]["status"] == "pass"
        and timing.get("experimental_status") == "pass"
        and utilization.get("status") == "pass"
        and route.get("routing_errors") == 0
    ):
        status["status"] = "physical-structural-pass-latency-target-fail"
    return status


def fmt(value: Any, digits: int | None = None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(status: dict[str, Any]) -> str:
    resources = status["utilization"]["resources"]
    power = status["power"]
    latency = status["latency"]
    lines = [
        "# Patch32 DenseToken4 300 MHz Goal Status",
        "",
        f"- status: `{status['status']}`",
        f"- profile: `{status['profile']}`",
        f"- structural contract: `{status['contract_audit']['status']}` "
        f"({fmt(status['contract_audit']['checks_pass'])}/{fmt(status['contract_audit']['checks_total'])})",
        f"- timing: experiment `{status['timing']['experimental_status']}`, official `{status['timing']['official_status']}`",
        f"- route errors: `{fmt(status['route']['routing_errors'])}`",
        "",
        "## Latency",
        "",
        "| Mode | II cycles | Time @300MHz ms | Target ms | Gap cycles | Gap ms | Status |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Search | {latency['search'].get('interval_max_cycles')} | "
        f"{fmt(latency['search'].get('interval_max_ms_at_300mhz'), 6)} | {SEARCH_TARGET_MS:.3f} | "
        f"{latency['search'].get('target_gap_cycles')} | {fmt(latency['search'].get('target_gap_ms'), 6)} | "
        f"{latency['search'].get('target_status')} |",
        f"| Track | {latency['track'].get('interval_max_cycles')} | "
        f"{fmt(latency['track'].get('interval_max_ms_at_300mhz'), 6)} | {TRACK_TARGET_MS:.3f} | "
        f"{latency['track'].get('target_gap_cycles')} | {fmt(latency['track'].get('target_gap_ms'), 6)} | "
        f"{latency['track'].get('target_status')} |",
        f"| Hybrid 10/90 | {fmt(latency['hybrid_10_90'].get('interval_cycles_expected'), 1)} | "
        f"{fmt(latency['hybrid_10_90'].get('interval_ms_expected_at_300mhz'), 6)} | n/a | n/a | n/a | derived |",
        "",
        "## Path Breakdown",
        "",
        "| Mode | axis_read | conv/event_conv | global_load | controller | head |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, conv_name in [("search", "hgtxr_conv_patch_embedding"), ("track", "hgtxr_event_conv_patch_embedding")]:
        inst = latency[mode]["instances"]
        lines.append(
            f"| {mode} | {fmt((inst.get('hgtxr_axis_read_frame') or {}).get('interval_max_cycles'))} | "
            f"{fmt((inst.get(conv_name) or {}).get('interval_max_cycles'))} | "
            f"{fmt((inst.get('hgtxr_global_buffer_load') or {}).get('interval_max_cycles'))} | "
            f"{fmt((inst.get('hgtxr_e2e_controller_run') or {}).get('interval_max_cycles'))} | "
            f"{fmt((inst.get('hgtxr_e2e_mlp_head') or {}).get('interval_max_cycles'))} |"
        )
    lines.extend(
        [
            "",
            "## Vivado Timing And Utilization",
            "",
            f"- WNS/TNS/WHS/THS: `{fmt(status['timing']['wns_ns'])}` / `{fmt(status['timing']['tns_ns'])}` / "
            f"`{fmt(status['timing']['whs_ns'])}` / `{fmt(status['timing']['ths_ns'])}` ns",
            f"- clock: `{fmt(status['timing']['frequency_mhz'])} MHz`, period `{fmt(status['timing']['period_ns'])} ns`",
            "",
            "| Resource | Used | Available | Util % |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ["clb_luts", "lut_as_logic", "lut_as_memory", "clb_registers", "block_ram_tile", "uram", "dsp"]:
        row = resources.get(name) or {}
        lines.append(
            f"| {name} | {fmt(row.get('used'))} | {fmt(row.get('available'))} | {fmt(row.get('util_pct'))} |"
        )
    lines.extend(
        [
            "",
            "## Power",
            "",
            f"- total on-chip: `{fmt(power.get('total_on_chip_w'))} W`",
            f"- dynamic/static: `{fmt(power.get('dynamic_w'))} W` / `{fmt(power.get('device_static_w'))} W`",
            f"- clocks/CLB/signals: `{fmt(power.get('clock_w'))}` / `{fmt(power.get('clb_logic_w'))}` / `{fmt(power.get('signals_w'))}` W",
            f"- BRAM/URAM/DSP/PS8: `{fmt(power.get('block_ram_w'))}` / `{fmt(power.get('uram_w'))}` / "
            f"`{fmt(power.get('dsp_w'))}` / `{fmt(power.get('ps8_w'))}` W",
            f"- methodology: {power.get('methodology')}",
            "",
            "## Board Plumbing",
            "",
            f"- status: `{status['board_plumbing']['status']}`",
            f"- remote dry-run: `{status['board_plumbing']['remote_dry_run_status']}`",
            f"- board latency gate: `{status['board_plumbing']['board_latency_gate_status']}`",
            f"- remote profiles: `{', '.join(status['board_plumbing']['remote_profiles']) or 'n/a'}`",
            "",
            "| Bundle | Status | Variant |",
            "|---|---:|---|",
        ]
    )
    for name, payload in status["board_plumbing"]["bundle_validations"].items():
        lines.append(f"| {name} | `{payload['status']}` | `{payload.get('variant')}` |")
    lines.extend(
        [
            "",
            "## Remaining Gaps",
            "",
        ]
    )
    for gap in status["measurement_gaps"]:
        lines.append(f"- {gap}")
    lines.extend(["", "## Sources", ""])
    for name, path in sorted(status["sources"].items()):
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write patch32_dtok4 300 MHz goal status.")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    status = build_status()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(status))
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
