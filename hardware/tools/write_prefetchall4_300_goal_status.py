#!/usr/bin/env python3
"""Write the current 300 MHz prefetch-all4 goal status report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
PROFILE = "par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16"
OVERLAY = f"hgtxr_e2e_axis_dma_{PROFILE}_overlay"
WRAPPER = "hgtxr_e2e_axis_dma_system_wrapper"
HLS_PROJECT = f"hgtxr_e2e_axis_{PROFILE}_no_board"

VIVADO_IMPL = HARDWARE_ROOT / "generated" / "build" / "vivado" / OVERLAY / f"{OVERLAY}.runs" / "impl_1"
HLS_REPORT = (
    HARDWARE_ROOT
    / "generated"
    / HLS_PROJECT
    / "solution_e2e_q4w8a"
    / "syn"
    / "report"
    / "hgtxr_e2e_axis_top_csynth.rpt"
)
TIMING_REPORT = VIVADO_IMPL / f"{WRAPPER}_timing_summary_postroute_physopted.rpt"
UTILIZATION_REPORT = VIVADO_IMPL / f"{WRAPPER}_utilization_placed.rpt"
POWER_REPORT = VIVADO_IMPL / f"{WRAPPER}_power_routed.rpt"
BOARD_GATE_JSON = HARDWARE_ROOT / "generated" / "signoff" / "par32_prefetchall4_300_board_latency_gate_2026_06_29.json"
BOARD_RUN_JSON = HARDWARE_ROOT / "generated" / "signoff" / "par32_prefetchall4_300_board_latency_run_2026_06_29.json"
BOARD_SEARCH_JSON = HARDWARE_ROOT / "pynq" / "hgtxr" / "e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.json"
BOARD_TRACK_JSON = HARDWARE_ROOT / "pynq" / "hgtxr" / "e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.json"
BOARD_HYBRID_JSON = HARDWARE_ROOT / "pynq" / "hgtxr" / "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json"
COSIM_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json"
XSIM_SMOKE_JSON = HARDWARE_ROOT / "generated" / "signoff" / "xsim_snapshot_smoke_2026_06_29.json"
DIRECT_XSIMK_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_hls_xsimk_direct_probe_2026_06_29.json"
PROGRESS_XSIMK_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_hls_xsimk_progress_probe_2026_06_29.json"
CONTRACT_AUDIT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_contract_audit_2026_06_29.json"
RESOURCE_BREAKDOWN_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_resource_breakdown_2026_06_29.json"
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_goal_status_2026_06_29.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_goal_status_2026_06_29.md"

EXPERIMENTAL_WNS_FLOOR_NS = -0.5
OFFICIAL_WNS_FLOOR_NS = 0.0
PL_CLOCK_MHZ = 300.0
SEARCH_TARGET_MS = 4.0
TRACK_TARGET_MS = 1.0
SEARCH_INVOCATION_RATIO = 0.10
TRACK_INVOCATION_RATIO = 0.90


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="ignore"))
    except json.JSONDecodeError:
        return {}


def parse_float(value: str) -> float:
    return float(value.replace(",", ""))


def parse_timing(text: str) -> dict[str, Any]:
    clock = None
    period_ns = None
    wns = None
    tns = None
    failing = None
    whs = None
    match = re.search(r"clk_pl_0\s+\{[^}]+\}\s+([0-9.]+)\s+([0-9.]+)", text)
    if match:
        period_ns = parse_float(match.group(1))
        clock = parse_float(match.group(2))
    match = re.search(
        r"clk_pl_0\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+([0-9]+)\s+[0-9]+\s+(-?[0-9.]+)",
        text,
    )
    if match:
        wns = parse_float(match.group(1))
        tns = parse_float(match.group(2))
        failing = int(match.group(3))
        whs = parse_float(match.group(4))
    return {
        "clock_mhz": clock,
        "period_ns": period_ns,
        "wns_ns": wns,
        "tns_ns": tns,
        "setup_failing_endpoints": failing,
        "whs_ns": whs,
        "experimental_timing_status": "pass"
        if wns is not None and wns >= EXPERIMENTAL_WNS_FLOOR_NS
        else "missing"
        if wns is None
        else "fail",
        "official_timing_status": "pass"
        if wns is not None and wns >= OFFICIAL_WNS_FLOOR_NS
        else "missing"
        if wns is None
        else "fail",
    }


def parse_utilization(text: str) -> dict[str, Any]:
    patterns = {
        "clb_luts": r"\| CLB LUTs\s+\|\s+([0-9]+)\s+\|\s+[0-9]+\s+\|\s+[0-9]+\s+\|\s+([0-9]+)\s+\|\s+([0-9.]+)",
        "clb_registers": r"\| CLB Registers\s+\|\s+([0-9]+)\s+\|\s+[0-9]+\s+\|\s+[0-9]+\s+\|\s+([0-9]+)\s+\|\s+([0-9.]+)",
        "block_ram_tile": r"\| Block RAM Tile\s+\|\s+([0-9]+)\s+\|\s+[0-9]+\s+\|\s+[0-9]+\s+\|\s+([0-9]+)\s+\|\s+([0-9.]+)",
        "uram": r"\| URAM\s+\|\s+([0-9]+)\s+\|\s+[0-9]+\s+\|\s+[0-9]+\s+\|\s+([0-9]+)\s+\|\s+([0-9.]+)",
        "dsp": r"\| DSPs\s+\|\s+([0-9]+)\s+\|\s+[0-9]+\s+\|\s+[0-9]+\s+\|\s+([0-9]+)\s+\|\s+([0-9.]+)",
    }
    resources: dict[str, Any] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        resources[name] = (
            {
                "used": int(match.group(1)),
                "available": int(match.group(2)),
                "util_pct": parse_float(match.group(3)),
            }
            if match
            else None
        )
    fit = all(value and value["used"] <= value["available"] for value in resources.values())
    return {"status": "pass" if fit else "missing" if any(v is None for v in resources.values()) else "fail", "resources": resources}


def parse_power(text: str) -> dict[str, Any]:
    def summary_value(label: str) -> float | None:
        match = re.search(rf"\|\s+{re.escape(label)}\s+\|\s+([0-9.]+)", text)
        return parse_float(match.group(1)) if match else None

    hierarchy = {}
    for name in ["axi_ctrl", "axi_dma_in", "axi_dma_out", "axi_mem", "hgtxr_e2e_axis_top_0", "psu"]:
        match = re.search(rf"\|\s+{re.escape(name)}\s+\|\s+([0-9.]+)", text)
        hierarchy[name] = parse_float(match.group(1)) if match else None

    return {
        "total_on_chip_w": summary_value("Total On-Chip Power (W)"),
        "dynamic_w": summary_value("Dynamic (W)"),
        "device_static_w": summary_value("Device Static (W)"),
        "ps_static_w": summary_value("PS Static"),
        "pl_static_w": summary_value("PL Static"),
        "hierarchy_dynamic_w": hierarchy,
        "methodology": "Vivado routed vectorless estimate; not board rail measurement and not mode-specific.",
    }


def parse_hls_latency(text: str) -> dict[str, Any]:
    match = re.search(
        r"\|\s*([0-9,]+)\|\s*([0-9,]+)\|\s*([0-9.]+\s+\w+)\|\s*([0-9.]+\s+\w+)\|\s*([0-9,]+)\|\s*([0-9,]+)\|",
        text,
    )
    if not match:
        return {"status": "missing"}
    return {
        "status": "available",
        "min_cycles": int(match.group(1).replace(",", "")),
        "max_cycles": int(match.group(2).replace(",", "")),
        "min_absolute": match.group(3).strip(),
        "max_absolute": match.group(4).strip(),
        "min_interval": int(match.group(5).replace(",", "")),
        "max_interval": int(match.group(6).replace(",", "")),
        "boundary": "HLS top latency envelope, not board p95/p99 and not a mode-specific measured latency distribution.",
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": percentile(ordered, 50.0),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": percentile(ordered, 95.0),
        "p99": percentile(ordered, 99.0),
    }


def numeric_samples(payload: dict[str, Any], key: str) -> list[float]:
    values = payload.get(key)
    if not isinstance(values, list):
        return []
    return [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]


def summary_mean(payload: dict[str, Any], key: str) -> float | None:
    summary = payload.get(key)
    if not isinstance(summary, dict):
        return None
    value = summary.get("mean")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def weighted_value(search_value: float | None, track_value: float | None) -> float | None:
    if search_value is None or track_value is None:
        return None
    return SEARCH_INVOCATION_RATIO * search_value + TRACK_INVOCATION_RATIO * track_value


def dict_mean(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("mean")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def dict_max(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("max")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def synthetic_mix_samples(search_samples: list[float], track_samples: list[float]) -> list[float]:
    if not search_samples or not track_samples:
        return []
    # 10:90 distribution represented as one Search draw and nine Track draws per round.
    rounds = max(len(search_samples), len(track_samples))
    mixed: list[float] = []
    for index in range(rounds):
        mixed.append(search_samples[index % len(search_samples)])
        for offset in range(9):
            mixed.append(track_samples[(index * 9 + offset) % len(track_samples)])
    return mixed


def mode_from_hybrid(hybrid_payload: dict[str, Any], mode: str) -> dict[str, Any]:
    summaries = hybrid_payload.get("mode_summaries")
    item = summaries.get(mode) if isinstance(summaries, dict) else None
    return item if isinstance(item, dict) else {}


def build_board_measurements(
    search_payload: dict[str, Any],
    track_payload: dict[str, Any],
    hybrid_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hybrid_payload = hybrid_payload or {}
    hybrid_search = mode_from_hybrid(hybrid_payload, "search")
    hybrid_track = mode_from_hybrid(hybrid_payload, "track")
    search_exists = bool(search_payload)
    track_exists = bool(track_payload)
    measured_hybrid_exists = bool(hybrid_payload)
    search_accel = numeric_samples(search_payload, "accelerator_latency_ms_samples")
    track_accel = numeric_samples(track_payload, "accelerator_latency_ms_samples")
    search_board = numeric_samples(search_payload, "latency_ms_samples")
    track_board = numeric_samples(track_payload, "latency_ms_samples")
    search_dma = numeric_samples(search_payload, "dma_bandwidth_Bps_samples")
    track_dma = numeric_samples(track_payload, "dma_bandwidth_Bps_samples")
    search_aggregate_dma = numeric_samples(search_payload, "aggregate_dma_effective_bandwidth_Bps_samples")
    track_aggregate_dma = numeric_samples(track_payload, "aggregate_dma_effective_bandwidth_Bps_samples")

    hybrid_accel_samples = synthetic_mix_samples(search_accel, track_accel)
    hybrid_board_samples = synthetic_mix_samples(search_board, track_board)

    search_accel_mean = summary_mean(search_payload, "accelerator_latency_ms_summary")
    track_accel_mean = summary_mean(track_payload, "accelerator_latency_ms_summary")
    search_board_mean = summary_mean(search_payload, "latency_ms_summary")
    track_board_mean = summary_mean(track_payload, "latency_ms_summary")
    search_dma_mean = summary_mean(search_payload, "dma_bandwidth_Bps_summary")
    track_dma_mean = summary_mean(track_payload, "dma_bandwidth_Bps_summary")
    search_aggregate_dma_mean = summary_mean(search_payload, "aggregate_dma_effective_bandwidth_Bps_summary")
    track_aggregate_dma_mean = summary_mean(track_payload, "aggregate_dma_effective_bandwidth_Bps_summary")

    hybrid_accel_mean = weighted_value(search_accel_mean, track_accel_mean)
    hybrid_board_mean = weighted_value(search_board_mean, track_board_mean)
    hybrid_dma_mean = weighted_value(search_dma_mean, track_dma_mean)
    hybrid_aggregate_dma_mean = weighted_value(search_aggregate_dma_mean, track_aggregate_dma_mean)
    measured_hybrid_accel_summary = hybrid_payload.get("accelerator_latency_ms_summary")
    measured_hybrid_board_summary = hybrid_payload.get("latency_ms_summary")
    measured_hybrid_dma_summary = hybrid_payload.get("dma_bandwidth_Bps_summary")
    measured_hybrid_aggregate_dma_summary = hybrid_payload.get("aggregate_dma_effective_bandwidth_Bps_summary")
    measured_distribution = hybrid_payload.get("search_track_invocation_distribution")
    if isinstance(measured_distribution, dict):
        search_count = measured_distribution.get("search")
        track_count = measured_distribution.get("track")
        total = (
            search_count + track_count
            if isinstance(search_count, int) and isinstance(track_count, int) and search_count + track_count > 0
            else None
        )
    else:
        total = None

    return {
        "status": "available" if (search_exists and track_exists) or measured_hybrid_exists else "missing",
        "methodology": (
            "Search/Track metrics are read from canonical per-mode board JSONs when available. "
            "If the canonical interleaved hybrid JSON is available, hybrid 10:90 metrics are measured from that run. "
            "Otherwise hybrid metrics are synthetic from per-mode samples and means."
        ),
        "search": {
            "exists": search_exists or bool(hybrid_search),
            "source": "per-mode-json" if search_exists else "hybrid-json" if hybrid_search else "missing",
            "runtime_state": search_payload.get("runtime_state", hybrid_search.get("expected_runtime_state")),
            "repeat": search_payload.get("repeat", hybrid_search.get("count")),
            "accelerator_latency_ms_summary": search_payload.get(
                "accelerator_latency_ms_summary", hybrid_search.get("accelerator_latency_ms_summary")
            ),
            "latency_ms_summary": search_payload.get("latency_ms_summary", hybrid_search.get("latency_ms_summary")),
            "dma_bandwidth_Bps_summary": search_payload.get(
                "dma_bandwidth_Bps_summary", hybrid_search.get("dma_bandwidth_Bps_summary")
            ),
            "aggregate_dma_effective_bandwidth_Bps_summary": search_payload.get(
                "aggregate_dma_effective_bandwidth_Bps_summary",
                hybrid_search.get("aggregate_dma_effective_bandwidth_Bps_summary"),
            ),
        },
        "track": {
            "exists": track_exists or bool(hybrid_track),
            "source": "per-mode-json" if track_exists else "hybrid-json" if hybrid_track else "missing",
            "runtime_state": track_payload.get("runtime_state", hybrid_track.get("expected_runtime_state")),
            "repeat": track_payload.get("repeat", hybrid_track.get("count")),
            "accelerator_latency_ms_summary": track_payload.get(
                "accelerator_latency_ms_summary", hybrid_track.get("accelerator_latency_ms_summary")
            ),
            "latency_ms_summary": track_payload.get("latency_ms_summary", hybrid_track.get("latency_ms_summary")),
            "dma_bandwidth_Bps_summary": track_payload.get(
                "dma_bandwidth_Bps_summary", hybrid_track.get("dma_bandwidth_Bps_summary")
            ),
            "aggregate_dma_effective_bandwidth_Bps_summary": track_payload.get(
                "aggregate_dma_effective_bandwidth_Bps_summary",
                hybrid_track.get("aggregate_dma_effective_bandwidth_Bps_summary"),
            ),
        },
        "hybrid_10_90": {
            "status": "measured-interleaved-board-run"
            if measured_hybrid_exists and hybrid_payload.get("status") == "pass"
            else "measured-interleaved-board-run-fail"
            if measured_hybrid_exists
            else "synthetic-from-per-mode-samples"
            if hybrid_accel_samples or hybrid_board_samples
            else "missing",
            "search_ratio": (search_count / total) if total else SEARCH_INVOCATION_RATIO,
            "track_ratio": (track_count / total) if total else TRACK_INVOCATION_RATIO,
            "accelerator_latency_ms_summary": measured_hybrid_accel_summary
            if measured_hybrid_exists
            else summarize(hybrid_accel_samples),
            "accelerator_latency_ms_weighted_mean": dict_mean(measured_hybrid_accel_summary)
            if measured_hybrid_exists
            else hybrid_accel_mean,
            "board_latency_ms_summary": measured_hybrid_board_summary if measured_hybrid_exists else summarize(hybrid_board_samples),
            "board_latency_ms_weighted_mean": dict_mean(measured_hybrid_board_summary)
            if measured_hybrid_exists
            else hybrid_board_mean,
            "dma_bandwidth_Bps_weighted_mean": dict_mean(measured_hybrid_dma_summary)
            if measured_hybrid_exists
            else hybrid_dma_mean,
            "aggregate_dma_effective_bandwidth_Bps_weighted_mean": dict_mean(measured_hybrid_aggregate_dma_summary)
            if measured_hybrid_exists
            else hybrid_aggregate_dma_mean,
            "throughput_fps_from_accelerator_weighted_mean": 1000.0 / dict_mean(measured_hybrid_accel_summary)
            if measured_hybrid_exists and dict_mean(measured_hybrid_accel_summary)
            else 1000.0 / hybrid_accel_mean
            if hybrid_accel_mean and hybrid_accel_mean > 0
            else None,
            "throughput_fps_from_board_weighted_mean": 1000.0 / dict_mean(measured_hybrid_board_summary)
            if measured_hybrid_exists and dict_mean(measured_hybrid_board_summary)
            else 1000.0 / hybrid_board_mean
            if hybrid_board_mean and hybrid_board_mean > 0
            else None,
            "worst_case_accelerator_latency_ms": dict_max(measured_hybrid_accel_summary)
            if measured_hybrid_exists
            else max(search_accel + track_accel)
            if search_accel or track_accel
            else None,
            "worst_case_board_latency_ms": dict_max(measured_hybrid_board_summary)
            if measured_hybrid_exists
            else max(search_board + track_board)
            if search_board or track_board
            else None,
            "measured_distribution": measured_distribution if measured_hybrid_exists else None,
        },
    }


def build_status() -> dict[str, Any]:
    timing = parse_timing(read_text(TIMING_REPORT))
    utilization = parse_utilization(read_text(UTILIZATION_REPORT))
    power = parse_power(read_text(POWER_REPORT))
    hls_latency = parse_hls_latency(read_text(HLS_REPORT))
    board_gate = read_json(BOARD_GATE_JSON)
    board_run = read_json(BOARD_RUN_JSON)
    board_measurements = build_board_measurements(
        read_json(BOARD_SEARCH_JSON),
        read_json(BOARD_TRACK_JSON),
        read_json(BOARD_HYBRID_JSON),
    )
    cosim = read_json(COSIM_JSON)
    xsim_smoke = read_json(XSIM_SMOKE_JSON)
    direct_xsimk = read_json(DIRECT_XSIMK_JSON)
    progress_xsimk = read_json(PROGRESS_XSIMK_JSON)
    contract_audit = read_json(CONTRACT_AUDIT_JSON)
    resource_breakdown = read_json(RESOURCE_BREAKDOWN_JSON)

    board_status = board_gate.get("status") or "missing"
    board_measurement_status = board_measurements.get("status")
    hybrid_status = (board_measurements.get("hybrid_10_90") or {}).get("status")
    cosim_status = cosim.get("status") or "missing"
    xsim_status = xsim_smoke.get("status") or "missing"
    direct_xsimk_status = direct_xsimk.get("status") or (cosim.get("direct_xsimk") or {}).get("status") or "missing"
    progress_xsimk_status = progress_xsimk.get("status") or (cosim.get("progress_xsimk") or {}).get("status") or "missing"
    contract_status = contract_audit.get("status") or "missing"
    resource_breakdown_status = resource_breakdown.get("status") or "missing"
    resource_breakdown_summary = resource_breakdown.get("summary") or {}
    implemented_hierarchical_blocks = resource_breakdown_summary.get("implemented_hierarchical_blocks") or []
    resource_breakdown_has_hierarchy = bool(implemented_hierarchical_blocks)
    experiment_timing_ok = timing.get("experimental_timing_status") == "pass"
    fit_ok = utilization.get("status") == "pass"

    questions = {
        "q1_hybrid_resource_latency_power": {
            "status": "partial" if board_measurement_status == "available" else "blocked-board-measurement",
            "current_answer": (
                f"Routed resources and vectorless total power are available. Hybrid 10/90 board metric status is {hybrid_status}."
                if board_measurement_status == "available"
                else "Routed resources and vectorless total power are available. Hybrid 10/90 latency and mode-specific power need board Search/Track result JSONs."
            ),
        },
        "q2_search_latency_power": {
            "status": "partial" if board_measurement_status == "available" else "blocked-board-measurement",
            "current_answer": (
                "Search board latency distribution is available from canonical JSON; Search-mode power is still Vivado vectorless/global unless rail measurement is added."
                if board_measurement_status == "available"
                else "Search expected output is known from CSim, but board Search latency/p95/p99 and Search-mode power are not captured."
            ),
        },
        "q3_search_active_path": {
            "status": "partial" if contract_status == "pass" else "blocked-contract-audit",
            "current_answer": (
                "Search uses Frame Conv plus shared ATTN/MLP/Head path with dispatcher prefetch traces clean in C testbench; "
                f"structural contract audit passes; standard RTL cosim remains blocked by XSIM wrapper launch, "
                f"direct xsimk probe status is {direct_xsimk_status}, and instrumented progress probe status is {progress_xsimk_status}."
            ),
        },
        "q4_track_latency_power": {
            "status": "partial" if board_measurement_status == "available" else "blocked-board-measurement",
            "current_answer": (
                "Track board latency distribution is available from canonical JSON; Track-mode power is still Vivado vectorless/global unless rail measurement is added."
                if board_measurement_status == "available"
                else "Track expected output is known from CSim, but board Track latency/p95/p99 and Track-mode power are not captured."
            ),
        },
        "q5_track_active_path": {
            "status": "partial" if contract_status == "pass" else "blocked-contract-audit",
            "current_answer": (
                "Track uses Event Conv plus shared ATTN/MLP/Head path with dispatcher trace clean in C testbench; "
                f"structural contract audit passes; standard RTL cosim remains blocked by XSIM wrapper launch, "
                f"direct xsimk probe status is {direct_xsimk_status}, and instrumented progress probe status is {progress_xsimk_status}."
            ),
        },
        "q6_only_vs_hybrid_throughput_ii_worst_case": {
            "status": "partial" if board_measurement_status == "available" else "blocked-board-measurement",
            "current_answer": (
                "Search-only, Track-only, and synthetic 10:90 hybrid throughput/worst-case metrics are computed from canonical board JSONs."
                if board_measurement_status == "available"
                else "HLS top interval envelope is available, but Search-only/Track-only board throughput and maximum event/frame rate need board runs."
            ),
        },
        "q7_measurement_methodology_distribution": {
            "status": "partial",
            "current_answer": (
                "Board latency/DMA summaries are computed from canonical Search/Track JSONs; 10:90 distribution is synthetic from per-mode samples unless an interleaved run is added. Board rail power and sensor I/O power remain missing."
                if board_measurement_status == "available"
                else "Vivado vectorless power and dry-run board plan exist. PS/PL/DRAM/sensor I/O inclusion, DMA bandwidth, batch size, p95/p99, and real invocation distribution need board measurement."
            ),
        },
        "q8_resource_absolute_percent_by_block": {
            "status": "available"
            if resource_breakdown_status == "pass" and resource_breakdown_has_hierarchy
            else "partial"
            if resource_breakdown_status in {"pass", "partial"}
            else "blocked-resource-breakdown",
            "current_answer": (
                "Top placed absolute/percent utilization and implemented hierarchical major-block resource breakdown are available."
                if resource_breakdown_status == "pass" and resource_breakdown_has_hierarchy
                else "Top placed absolute/percent utilization and major-block OOC synth breakdown are available. "
                "Use this as current source-utilization evidence; exact post-place hierarchical attribution still requires an implementation-time hierarchical utilization report."
                if resource_breakdown_status in {"pass", "partial"}
                else "Top routed absolute/percent utilization is available, but major-block breakdown artifact is missing."
            ),
        },
    }

    overall_status = (
        "structural-contract-fail"
        if contract_status not in {"pass", "missing"}
        else
        "experiment-pass-missing-board-rtlcosim"
        if experiment_timing_ok and fit_ok and board_status != "pass" and cosim_status != "pass"
        else "partial"
    )

    return {
        "status": overall_status,
        "profile": PROFILE,
        "rules": {
            "pl_clock_mhz": PL_CLOCK_MHZ,
            "hls_clock_period_ns": 3.333,
            "experimental_routed_wns_floor_ns": EXPERIMENTAL_WNS_FLOOR_NS,
            "official_clean_wns_floor_ns": OFFICIAL_WNS_FLOOR_NS,
            "search_target_ms": SEARCH_TARGET_MS,
            "track_target_ms": TRACK_TARGET_MS,
            "search_invocation_ratio": SEARCH_INVOCATION_RATIO,
            "track_invocation_ratio": TRACK_INVOCATION_RATIO,
        },
        "timing": timing,
        "utilization": utilization,
        "power": power,
        "hls_latency": hls_latency,
        "rtl_cosim": {
            "status": cosim_status,
            "verilog_status": (cosim.get("cosim") or {}).get("verilog_status"),
            "xsim_environment_status": xsim_status,
            "direct_xsimk_status": direct_xsimk_status,
            "direct_xsimk_checks": direct_xsimk.get("checks") or (cosim.get("direct_xsimk") or {}).get("checks"),
            "progress_xsimk_status": progress_xsimk_status,
            "progress_xsimk_checks": progress_xsimk.get("checks") or (cosim.get("progress_xsimk") or {}).get("checks"),
            "interpretation": cosim.get("interpretation"),
        },
        "contract_audit": {
            "status": contract_status,
            "checks_total": (contract_audit.get("summary") or {}).get("checks_total"),
            "checks_pass": (contract_audit.get("summary") or {}).get("checks_pass"),
            "checks_fail": (contract_audit.get("summary") or {}).get("checks_fail"),
            "interpretation": contract_audit.get("interpretation"),
        },
        "resource_breakdown": {
            "status": resource_breakdown_status,
            "methodology": resource_breakdown.get("methodology"),
            "top": resource_breakdown.get("top"),
            "major_blocks": resource_breakdown.get("major_blocks"),
            "summary": resource_breakdown.get("summary"),
        },
        "board": {
            "gate_status": board_status,
            "run_status": board_run.get("status") or "missing",
            "host_preflight": (board_run.get("host_preflight") or {}).get("status"),
            "search_result_status": ((board_gate.get("cases") or {}).get("search") or {}).get("status"),
            "track_result_status": ((board_gate.get("cases") or {}).get("track") or {}).get("status"),
            "measurements": board_measurements,
        },
        "questions": questions,
        "sources": {
            "hls_report": str(HLS_REPORT),
            "timing_report": str(TIMING_REPORT),
            "utilization_report": str(UTILIZATION_REPORT),
            "power_report": str(POWER_REPORT),
            "board_gate_json": str(BOARD_GATE_JSON),
            "board_run_json": str(BOARD_RUN_JSON),
            "board_search_json": str(BOARD_SEARCH_JSON),
            "board_track_json": str(BOARD_TRACK_JSON),
            "board_hybrid_json": str(BOARD_HYBRID_JSON),
            "cosim_json": str(COSIM_JSON),
            "xsim_smoke_json": str(XSIM_SMOKE_JSON),
            "direct_xsimk_json": str(DIRECT_XSIMK_JSON),
            "progress_xsimk_json": str(PROGRESS_XSIMK_JSON),
            "contract_audit_json": str(CONTRACT_AUDIT_JSON),
            "resource_breakdown_json": str(RESOURCE_BREAKDOWN_JSON),
        },
    }


def fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def render_markdown(status: dict[str, Any]) -> str:
    timing = status["timing"]
    power = status["power"]
    resources = status["utilization"]["resources"]
    resource_breakdown = status.get("resource_breakdown") or {}
    major_blocks = resource_breakdown.get("major_blocks") or {}
    measurements = status["board"]["measurements"]
    hybrid = measurements["hybrid_10_90"]
    lines = [
        "# Prefetch-All4 300 MHz Goal Status",
        "",
        f"- status: `{status['status']}`",
        f"- profile: `{status['profile']}`",
        f"- PL clock rule: `{status['rules']['pl_clock_mhz']} MHz`",
        f"- experimental WNS floor: `{status['rules']['experimental_routed_wns_floor_ns']} ns`",
        f"- official clean WNS floor: `{status['rules']['official_clean_wns_floor_ns']} ns`",
        "",
        "## Timing",
        "",
        f"- routed clock: `{fmt(timing.get('clock_mhz'))} MHz` / `{fmt(timing.get('period_ns'))} ns`",
        f"- WNS/TNS/WHS: `{fmt(timing.get('wns_ns'))}` / `{fmt(timing.get('tns_ns'))}` / `{fmt(timing.get('whs_ns'))}` ns",
        f"- setup failing endpoints: `{fmt(timing.get('setup_failing_endpoints'))}`",
        f"- experiment timing status: `{timing.get('experimental_timing_status')}`",
        f"- official timing status: `{timing.get('official_timing_status')}`",
        "",
        "## Routed Utilization",
        "",
        "| Resource | Used | Available | Util % |",
        "|---|---:|---:|---:|",
    ]
    for name in ["clb_luts", "clb_registers", "block_ram_tile", "uram", "dsp"]:
        row = resources.get(name)
        if row:
            lines.append(f"| {name} | {row['used']} | {row['available']} | {row['util_pct']} |")
        else:
            lines.append(f"| {name} | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Major Block Resource Breakdown",
            "",
            f"- status: `{resource_breakdown.get('status', 'missing')}`",
            f"- methodology: {resource_breakdown.get('methodology', 'n/a')}",
            "",
            "| Block | Evidence | LUT | FF | BRAM Tile | URAM | DSP | LUT Share Of Placed Top % | DSP Share Of Placed Top % |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for block_name, block in major_blocks.items():
        block_resources = block.get("resources") or {}
        lut = block_resources.get("clb_luts") or {}
        ff = block_resources.get("clb_registers") or {}
        bram = block_resources.get("block_ram_tile") or {}
        uram = block_resources.get("uram") or {}
        dsp = block_resources.get("dsp") or {}
        lines.append(
            f"| {block_name} | {fmt(block.get('evidence_level'))} | {fmt(lut.get('used'))} | {fmt(ff.get('used'))} | "
            f"{fmt(bram.get('used'))} | {fmt(uram.get('used'))} | {fmt(dsp.get('used'))} | "
            f"{fmt(lut.get('share_of_placed_top_pct'))} | {fmt(dsp.get('share_of_placed_top_pct'))} |"
        )
    lines.extend(
        [
            "",
            "## Power",
            "",
            f"- total on-chip: `{fmt(power.get('total_on_chip_w'))} W`",
            f"- dynamic/static: `{fmt(power.get('dynamic_w'))} W` / `{fmt(power.get('device_static_w'))} W`",
            f"- PS static / PL static: `{fmt(power.get('ps_static_w'))} W` / `{fmt(power.get('pl_static_w'))} W`",
            f"- IP dynamic bucket: `{fmt(power['hierarchy_dynamic_w'].get('hgtxr_e2e_axis_top_0'))} W`",
            f"- PS dynamic bucket: `{fmt(power['hierarchy_dynamic_w'].get('psu'))} W`",
            f"- AXI mem / DMA in / DMA out: `{fmt(power['hierarchy_dynamic_w'].get('axi_mem'))} W` / `{fmt(power['hierarchy_dynamic_w'].get('axi_dma_in'))} W` / `{fmt(power['hierarchy_dynamic_w'].get('axi_dma_out'))} W`",
            f"- methodology: {power['methodology']}",
            "",
            "## Latency And Gates",
            "",
            f"- HLS latency envelope: `{status['hls_latency'].get('min_absolute', 'n/a')}` to `{status['hls_latency'].get('max_absolute', 'n/a')}`",
            f"- HLS interval envelope: `{status['hls_latency'].get('min_interval', 'n/a')}` to `{status['hls_latency'].get('max_interval', 'n/a')}` cycles",
            f"- board gate: `{status['board']['gate_status']}`",
            f"- board run: `{status['board']['run_status']}`, host preflight `{status['board']['host_preflight']}`",
            f"- RTL cosim: `{status['rtl_cosim']['status']}`, XSIM env `{status['rtl_cosim']['xsim_environment_status']}`",
            f"- direct xsimk probe: `{status['rtl_cosim'].get('direct_xsimk_status')}`",
            f"- instrumented xsimk progress probe: `{status['rtl_cosim'].get('progress_xsimk_status')}`",
            f"- structural contract audit: `{status['contract_audit']['status']}` "
            f"({fmt(status['contract_audit']['checks_pass'])}/{fmt(status['contract_audit']['checks_total'])} checks)",
            "",
            "## Board Measurement Aggregation",
            "",
            f"- measurement status: `{measurements['status']}`",
            f"- hybrid 10:90 status: `{hybrid['status']}`",
            f"- hybrid accelerator weighted mean: `{fmt(hybrid['accelerator_latency_ms_weighted_mean'])} ms`",
            f"- hybrid board weighted mean: `{fmt(hybrid['board_latency_ms_weighted_mean'])} ms`",
            f"- hybrid accelerator throughput: `{fmt(hybrid['throughput_fps_from_accelerator_weighted_mean'])} fps`",
            f"- hybrid board throughput: `{fmt(hybrid['throughput_fps_from_board_weighted_mean'])} fps`",
            f"- hybrid DMA weighted mean: `{fmt(hybrid['dma_bandwidth_Bps_weighted_mean'])} B/s`",
            f"- hybrid aggregate DMA weighted mean: `{fmt(hybrid['aggregate_dma_effective_bandwidth_Bps_weighted_mean'])} B/s`",
            f"- worst-case accelerator latency: `{fmt(hybrid['worst_case_accelerator_latency_ms'])} ms`",
            f"- worst-case board latency: `{fmt(hybrid['worst_case_board_latency_ms'])} ms`",
            f"- methodology: {measurements['methodology']}",
            "",
            "| Mode | Runtime State | Repeat | Accelerator max ms | Accelerator p95 ms | Board max ms | Board p95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in ["search", "track"]:
        item = measurements[mode]
        accel = item.get("accelerator_latency_ms_summary") or {}
        board = item.get("latency_ms_summary") or {}
        lines.append(
            f"| {mode} | {fmt(item.get('runtime_state'))} | {fmt(item.get('repeat'))} | "
            f"{fmt(accel.get('max'))} | {fmt(accel.get('p95'))} | {fmt(board.get('max'))} | {fmt(board.get('p95'))} |"
        )
    lines.extend(
        [
            "",
            "## Eight-Question Current Status",
            "",
            "| ID | Status | Current Answer |",
            "|---|---:|---|",
        ]
    )
    for qid, item in status["questions"].items():
        lines.append(f"| `{qid}` | `{item['status']}` | {item['current_answer']} |")
    lines.extend(["", "## Sources", ""])
    for name, path in sorted(status["sources"].items()):
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the prefetch-all4 300 MHz goal status report.")
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
