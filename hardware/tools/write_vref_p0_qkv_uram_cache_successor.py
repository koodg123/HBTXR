#!/usr/bin/env python3
"""Write VREF-P0-02 QKV weight-cache URAM successor evidence."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"
HGTXR_ROOT = Path(__file__).resolve().parents[2]

QKV_PROJECT = "hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram"
BASE_PROJECT = "hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream"
DEFAULT_PROJECT = "hgtxr_e2e_axis_vref_p0_softmax_input_x2"
QKV_IP_PROJECT = f"{QKV_PROJECT}_ip"
QKV_OVERLAY_PROJECT = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay"
QKV_BD_NAME = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_system"
QKV_ARTIFACT = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram"
QKV_VARIANT = "vref-p0-softmax-input-x2-qkv-uram"
QKV_PRESET = "axis-vref-p0-softmax-input-x2-qkv-uram"
QKV_SMOKE_JSON = "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
QKV_BUNDLE_DIR = "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle"
QKV_CUSTOM_FLAGS = (
    "-DHGTXR_E2E_BLOCKS=2 "
    "-DHGTXR_E2E_ACTIVE_TOKENS=16 "
    "-DHGTXR_E2E_PATCH_GRID_H=1 "
    "-DHGTXR_E2E_PATCH_GRID_W=16 "
    "-DHGTXR_E2E_FF_DIM=32 "
    "-DHGTXR_E2E_STRICT_GOLDEN=1 "
    "-DHGTXR_E2E_USE_VREF_P0_HGPIPE_LNQ_ACTIVE16_SOFTMAX_INPUT_X2_GOLDEN=1 "
    "-DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1 "
    "-DHGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE=16 "
    "-DHGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE=4 "
    "-DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1 "
    "-DHGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE=32 "
    "-DHGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE=4 "
    "-DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1 "
    "-DHGTXR_E2E_USE_HGPIPE_LNQ_GAMMA_RAW=1 "
    "-DHGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE=16 "
    "-DHGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE=4 "
    "-DHGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT=33 "
    "-DHGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1"
)

THRESHOLDS = {
    "latency_cycles": 37508072,
    "estimated_clock_ns": 5.0,
    "routed_wns_ns": 4.415,
    "dsp": 604,
    "lut": 126506,
    "uram": 64,
}

EXPECTED_CSIM_MARKERS = [
    "E2E AXIS vector comparison passed",
    "runtime_state=2 count=6 last=1 failures=0",
    "CSim done with 0 errors",
]


def text_at(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def int_at(root: ET.Element, path: str) -> int | None:
    value = text_at(root, path)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def float_at(root: ET.Element, path: str) -> float | None:
    value = text_at(root, path)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def parse_csynth(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    root = ET.parse(path).getroot()
    resources = {
        "bram_18k": int_at(root, ".//AreaEstimates/Resources/BRAM_18K"),
        "dsp": int_at(root, ".//AreaEstimates/Resources/DSP"),
        "ff": int_at(root, ".//AreaEstimates/Resources/FF"),
        "lut": int_at(root, ".//AreaEstimates/Resources/LUT"),
        "uram": int_at(root, ".//AreaEstimates/Resources/URAM"),
    }
    latency = int_at(root, ".//SummaryOfOverallLatency/Average-caseLatency")
    if latency is None:
        latency = int_at(root, ".//SummaryOfOverallLatency/Worst-caseLatency")
    missing = [
        key
        for key, value in {
            "estimated_clock_ns": float_at(root, ".//EstimatedClockPeriod"),
            "latency_cycles": latency,
            **resources,
        }.items()
        if value is None
    ]
    return {
        "status": "pass" if not missing else "partial",
        "path": str(path),
        "estimated_clock_ns": float_at(root, ".//EstimatedClockPeriod"),
        "latency_cycles": latency,
        "resources": resources,
        "missing_fields": missing,
    }


def read_csim(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    text = path.read_text(errors="ignore")
    missing = [marker for marker in EXPECTED_CSIM_MARKERS if marker not in text]
    return {
        "status": "pass" if not missing else "fail",
        "path": str(path),
        "expected_markers": EXPECTED_CSIM_MARKERS,
        "missing_markers": missing,
    }


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any, threshold: Any = None) -> None:
    check: dict[str, Any] = {"name": name, "status": "pass" if ok else "fail", "detail": detail}
    if threshold is not None:
        check["threshold"] = threshold
    checks.append(check)


def parse_timing_summary(text: str) -> dict[str, Any]:
    in_summary = False
    for line in text.splitlines():
        if "Design Timing Summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        fields = line.split()
        if len(fields) >= 6 and re.fullmatch(r"[-+]?\d+\.\d+", fields[0]):
            return {
                "wns_ns": float(fields[0]),
                "tns_ns": float(fields[1]),
                "tns_failing_endpoints": int(fields[2]),
                "whs_ns": float(fields[4]),
                "ths_ns": float(fields[5]),
                "ths_failing_endpoints": int(fields[6]) if len(fields) > 6 and fields[6].isdigit() else None,
            }
    return {}


def parse_route_status(text: str) -> dict[str, Any]:
    def find_int(pattern: str) -> int | None:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    return {
        "routable_nets": find_int(r"# of routable nets.*?:\s*(\d+)"),
        "fully_routed_nets": find_int(r"# of fully routed nets.*?:\s*(\d+)"),
        "routing_error_nets": find_int(r"# of nets with routing errors.*?:\s*(\d+)"),
    }


def qkv_package_command(hardware: Path) -> str:
    return (
        "LD_LIBRARY_PATH=/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH "
        f"HGTXR_E2E_PROJECT_NAME={QKV_IP_PROJECT} "
        "HGTXR_E2E_SCALE=custom "
        "HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream "
        f"HGTXR_E2E_CUSTOM_SCALE_FLAGS='{QKV_CUSTOM_FLAGS}' "
        f"/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f {hardware / 'vivado/scripts/package_e2e_axis_ip.tcl'}"
    )


def qkv_overlay_command(hardware: Path) -> str:
    hls_ip_repo = hardware / "generated" / QKV_IP_PROJECT / "solution_e2e_q4w8a" / "impl" / "ip"
    return (
        "LD_LIBRARY_PATH=/tools/Xilinx/Vivado/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH "
        "/tools/Xilinx/Vivado/2023.2/bin/vivado -mode batch "
        f"-source {hardware / 'vivado/scripts/build_e2e_axis_dma_bitstream.tcl'} "
        "-tclargs "
        f"-project_name {QKV_OVERLAY_PROJECT} "
        f"-bd_name {QKV_BD_NAME} "
        f"-artifact_name {QKV_ARTIFACT} "
        f"-hls_ip_repo {hls_ip_repo}"
    )


def read_ip_package_evidence(hardware: Path) -> dict[str, Any]:
    ip_dir = hardware / "generated" / QKV_IP_PROJECT / "solution_e2e_q4w8a" / "impl" / "ip"
    component_xml = ip_dir / "component.xml"
    export_zip = hardware / "generated" / QKV_IP_PROJECT / "solution_e2e_q4w8a" / "impl" / "export.zip"
    checks = [
        {"name": "component_xml_exists", "status": "pass" if component_xml.exists() else "missing", "path": str(component_xml)},
        {"name": "export_zip_exists", "status": "pass" if export_zip.exists() else "missing", "path": str(export_zip)},
    ]
    missing = [check["name"] for check in checks if check["status"] != "pass"]
    status = "pass" if not missing else "missing" if len(missing) == len(checks) else "partial"
    return {
        "status": status,
        "project_name": QKV_IP_PROJECT,
        "ip_dir": str(ip_dir),
        "component_xml": str(component_xml),
        "export_zip": str(export_zip),
        "command": qkv_package_command(hardware),
        "checks": checks,
        "missing": missing,
    }


def read_overlay_evidence(hardware: Path) -> dict[str, Any]:
    project_dir = hardware / "generated" / "build" / "vivado" / QKV_OVERLAY_PROJECT
    impl_dir = project_dir / f"{QKV_OVERLAY_PROJECT}.runs" / "impl_1"
    overlay_dir = hardware / "generated" / "build" / "vivado" / "overlay" / QKV_OVERLAY_PROJECT
    pynq_dir = hardware / "pynq" / "hgtxr"
    wrapper = f"{QKV_BD_NAME}_wrapper"
    bit = overlay_dir / f"{QKV_ARTIFACT}.bit"
    hwh = overlay_dir / f"{QKV_ARTIFACT}.hwh"
    pynq_bit = pynq_dir / f"{QKV_ARTIFACT}.bit"
    pynq_hwh = pynq_dir / f"{QKV_ARTIFACT}.hwh"
    timing_rpt = impl_dir / f"{wrapper}_timing_summary_routed.rpt"
    route_status_rpt = impl_dir / f"{wrapper}_route_status.rpt"
    runme_log = impl_dir / "runme.log"
    timing = parse_timing_summary(timing_rpt.read_text(errors="ignore")) if timing_rpt.exists() else {}
    route_status = parse_route_status(route_status_rpt.read_text(errors="ignore")) if route_status_rpt.exists() else {}
    checks = [
        {"name": "overlay_bit_exists", "status": "pass" if bit.exists() else "missing", "path": str(bit)},
        {"name": "overlay_hwh_exists", "status": "pass" if hwh.exists() else "missing", "path": str(hwh)},
        {"name": "pynq_bit_exists", "status": "pass" if pynq_bit.exists() else "missing", "path": str(pynq_bit)},
        {"name": "pynq_hwh_exists", "status": "pass" if pynq_hwh.exists() else "missing", "path": str(pynq_hwh)},
        {
            "name": "timing_wns_gte_c3b",
            "status": "pass"
            if timing.get("wns_ns") is not None and timing["wns_ns"] >= THRESHOLDS["routed_wns_ns"]
            else "missing" if timing.get("wns_ns") is None else "fail",
            "detail": timing.get("wns_ns"),
            "threshold": THRESHOLDS["routed_wns_ns"],
        },
        {
            "name": "route_errors_zero",
            "status": "pass"
            if route_status.get("routing_error_nets") == 0
            else "missing" if route_status.get("routing_error_nets") is None else "fail",
            "detail": route_status.get("routing_error_nets"),
        },
    ]
    nonpass = [check["name"] for check in checks if check["status"] != "pass"]
    fail = [check["name"] for check in checks if check["status"] == "fail"]
    status = "pass" if not nonpass else "fail" if fail else "missing" if len(nonpass) == len(checks) else "partial"
    return {
        "status": status,
        "project_name": QKV_OVERLAY_PROJECT,
        "bd_name": QKV_BD_NAME,
        "artifact_name": QKV_ARTIFACT,
        "project_dir": str(project_dir),
        "impl_dir": str(impl_dir),
        "overlay_bit": str(bit),
        "overlay_hwh": str(hwh),
        "pynq_bit": str(pynq_bit),
        "pynq_hwh": str(pynq_hwh),
        "timing_report": str(timing_rpt),
        "route_status_report": str(route_status_rpt),
        "runme_log": str(runme_log),
        "timing": timing,
        "route_status": route_status,
        "command": qkv_overlay_command(hardware),
        "checks": checks,
        "nonpass": nonpass,
    }


def read_physical_smoke_evidence(hardware: Path) -> dict[str, Any]:
    candidate_paths = [
        hardware / "pynq" / "hgtxr" / QKV_SMOKE_JSON,
        hardware / "generated" / "pynq" / QKV_SMOKE_JSON,
    ]
    existing = [path for path in candidate_paths if path.exists()]
    return {
        "status": "present" if existing else "missing",
        "result_json": str(existing[0]) if existing else "",
        "checked_paths": [str(path) for path in candidate_paths],
    }


def qkv_pynq_plumbing(hardware: Path, root: Path) -> dict[str, Any]:
    bundle_dir = hardware / "generated" / "pynq" / QKV_BUNDLE_DIR
    tar_path = bundle_dir.with_suffix(".tar.gz")
    return {
        "status": "ready-for-artifacts",
        "variant": QKV_VARIANT,
        "preset": QKV_PRESET,
        "artifact_prefix": QKV_ARTIFACT,
        "bundle_dir": str(bundle_dir),
        "tar": str(tar_path),
        "expected_result_json": str(hardware / "pynq" / "hgtxr" / QKV_SMOKE_JSON),
        "run_script": f"run_e2e_axis_dma_{QKV_VARIANT.replace('-', '_')}_file_smoke.sh",
        "validate_script": f"validate_e2e_axis_dma_{QKV_VARIANT.replace('-', '_')}_file_smoke.sh",
        "package_command": (
            "python3 tools/package_e2e_axis_dma_pynq_bundle.py "
            f"--variant {QKV_VARIANT} "
            f"--out-dir {bundle_dir} "
            f"--tar {tar_path}"
        ),
        "session_command": (
            "python3 tools/prepare_zcu104_smoke_session.py "
            f"--variant {QKV_VARIANT} "
            f"--preset {QKV_PRESET} "
            f"--bundle-dir {bundle_dir} "
            f"--tar {tar_path} "
            f"--root {root}"
        ),
        "remote_dry_run_command": (
            "python3 tools/run_zcu104_c3b_smoke_remote.py "
            f"--profile {QKV_VARIANT} --host <zcu104-ip-or-host> --user xilinx"
        ),
        "dry_run_import_command": (
            "python3 tools/import_pynq_smoke_result.py "
            f"/path/to/{QKV_SMOKE_JSON} --preset {QKV_PRESET} --dry-run"
        ),
    }


def build_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    hardware = root / "hardware"
    qkv_csim_log = (
        hardware
        / "generated"
        / f"{QKV_PROJECT}_csim"
        / "solution_e2e_q4w8a"
        / "csim"
        / "report"
        / "hgtxr_e2e_axis_top_csim.log"
    )
    qkv_csynth_xml = (
        hardware
        / "generated"
        / f"{QKV_PROJECT}_csynth"
        / "solution_e2e_q4w8a"
        / "syn"
        / "report"
        / "csynth.xml"
    )
    base_csynth_xml = (
        hardware
        / "generated"
        / f"{BASE_PROJECT}_csynth"
        / "solution_e2e_q4w8a"
        / "syn"
        / "report"
        / "csynth.xml"
    )
    default_csynth_xml = (
        hardware
        / "generated"
        / f"{DEFAULT_PROJECT}_csynth"
        / "solution_e2e_q4w8a"
        / "syn"
        / "report"
        / "csynth.xml"
    )
    source_path = hardware / "hls" / "include" / "hgtxr_e2e_vit.hpp"

    csim = read_csim(qkv_csim_log)
    qkv = parse_csynth(qkv_csynth_xml)
    base = parse_csynth(base_csynth_xml)
    default = parse_csynth(default_csynth_xml)
    ip_package = read_ip_package_evidence(hardware)
    overlay = read_overlay_evidence(hardware)
    physical_smoke = read_physical_smoke_evidence(hardware)
    pynq_plumbing = qkv_pynq_plumbing(hardware, root)

    qkv_res = qkv.get("resources", {}) if isinstance(qkv.get("resources"), dict) else {}
    base_res = base.get("resources", {}) if isinstance(base.get("resources"), dict) else {}
    default_res = default.get("resources", {}) if isinstance(default.get("resources"), dict) else {}
    checks: list[dict[str, Any]] = []
    add_check(checks, "csim_pass", csim.get("status") == "pass", csim.get("missing_markers", []))
    add_check(checks, "csynth_pass", qkv.get("status") == "pass", qkv.get("missing_fields", []))
    add_check(
        checks,
        "latency_lte_c3b",
        isinstance(qkv.get("latency_cycles"), int) and qkv["latency_cycles"] <= THRESHOLDS["latency_cycles"],
        qkv.get("latency_cycles"),
        THRESHOLDS["latency_cycles"],
    )
    add_check(
        checks,
        "estimated_clock_lte_target",
        isinstance(qkv.get("estimated_clock_ns"), float)
        and qkv["estimated_clock_ns"] <= THRESHOLDS["estimated_clock_ns"],
        qkv.get("estimated_clock_ns"),
        THRESHOLDS["estimated_clock_ns"],
    )
    for key in ["dsp", "lut", "uram"]:
        add_check(
            checks,
            f"{key}_lte_c3b",
            isinstance(qkv_res.get(key), int) and qkv_res[key] <= THRESHOLDS[key],
            qkv_res.get(key),
            THRESHOLDS[key],
        )
    add_check(
        checks,
        "bram_reduced_vs_dsp_mixed_stream",
        isinstance(qkv_res.get("bram_18k"), int)
        and isinstance(base_res.get("bram_18k"), int)
        and qkv_res["bram_18k"] < base_res["bram_18k"],
        {"qkv_uram": qkv_res.get("bram_18k"), "dsp_mixed_stream": base_res.get("bram_18k")},
    )
    add_check(
        checks,
        "uram_increased_vs_dsp_mixed_stream",
        isinstance(qkv_res.get("uram"), int)
        and isinstance(base_res.get("uram"), int)
        and qkv_res["uram"] > base_res["uram"],
        {"qkv_uram": qkv_res.get("uram"), "dsp_mixed_stream": base_res.get("uram")},
    )
    add_check(
        checks,
        "uram_reduced_vs_default",
        isinstance(qkv_res.get("uram"), int)
        and isinstance(default_res.get("uram"), int)
        and qkv_res["uram"] < default_res["uram"],
        {"qkv_uram": qkv_res.get("uram"), "default": default_res.get("uram")},
    )

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    promotion_remaining: list[str] = []
    if ip_package["status"] != "pass":
        promotion_remaining.append("ip_package")
    if overlay["status"] != "pass":
        promotion_remaining.append("routed_overlay_timing")
    if physical_smoke["status"] != "present":
        promotion_remaining.append("physical_smoke_json")
    promotion_status = (
        "promotion-evidence-ready"
        if fail_count == 0 and not promotion_remaining
        else "hls-resource-pass-routed-pending"
        if fail_count == 0
        else "blocked-by-hls-resource-check"
    )
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "date_tag": DATE_TAG,
        "root": str(root),
        "candidate": "VREF-P0-02-qkv-weight-cache-uram",
        "base_candidate": "VREF-P0-01-softmax_input_x2-dsp_mixed_stream",
        "macro": "HGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1",
        "resource_policy": "dsp_mixed_stream",
        "source": {
            "hgtxr_e2e_vit": str(source_path),
            "csim_log": str(qkv_csim_log),
            "qkv_csynth_xml": str(qkv_csynth_xml),
            "base_csynth_xml": str(base_csynth_xml),
            "default_csynth_xml": str(default_csynth_xml),
        },
        "csim": csim,
        "csynth": qkv,
        "comparison": {
            "default": default,
            "dsp_mixed_stream": base,
            "delta_vs_dsp_mixed_stream": {
                "bram_18k": qkv_res.get("bram_18k") - base_res.get("bram_18k")
                if isinstance(qkv_res.get("bram_18k"), int) and isinstance(base_res.get("bram_18k"), int)
                else None,
                "uram": qkv_res.get("uram") - base_res.get("uram")
                if isinstance(qkv_res.get("uram"), int) and isinstance(base_res.get("uram"), int)
                else None,
                "dsp": qkv_res.get("dsp") - base_res.get("dsp")
                if isinstance(qkv_res.get("dsp"), int) and isinstance(base_res.get("dsp"), int)
                else None,
                "lut": qkv_res.get("lut") - base_res.get("lut")
                if isinstance(qkv_res.get("lut"), int) and isinstance(base_res.get("lut"), int)
                else None,
            },
        },
        "ip_package": ip_package,
        "overlay": overlay,
        "physical_smoke": physical_smoke,
        "pynq_plumbing": pynq_plumbing,
        "thresholds": THRESHOLDS,
        "checks": checks,
        "check_count": len(checks),
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "promotion": {
            "status": promotion_status,
            "remaining": promotion_remaining,
            "note": "Successor-only evidence; C3b remains protected baseline.",
        },
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "writes_hls_source": False,
            "overwrites_c3b_artifacts": False,
            "creates_board_result": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    res = audit["csynth"].get("resources", {})
    delta = audit["comparison"]["delta_vs_dsp_mixed_stream"]
    overlay = audit["overlay"]
    ip_package = audit["ip_package"]
    physical_smoke = audit["physical_smoke"]
    pynq = audit["pynq_plumbing"]
    lines = [
        "# VREF-P0-02 QKV Weight Cache URAM Successor",
        "",
        f"- status: `{audit['status']}`",
        f"- candidate: `{audit['candidate']}`",
        f"- macro: `{audit['macro']}`",
        f"- resource_policy: `{audit['resource_policy']}`",
        f"- CSim: `{audit['csim']['status']}`",
        f"- CSynth: `{audit['csynth']['status']}`",
        f"- latency_cycles: `{audit['csynth'].get('latency_cycles')}`",
        f"- estimated_clock_ns: `{audit['csynth'].get('estimated_clock_ns')}`",
        f"- resources: BRAM_18K `{res.get('bram_18k')}`, DSP `{res.get('dsp')}`, FF `{res.get('ff')}`, LUT `{res.get('lut')}`, URAM `{res.get('uram')}`",
        f"- delta_vs_dsp_mixed_stream: BRAM_18K `{delta.get('bram_18k')}`, URAM `{delta.get('uram')}`, DSP `{delta.get('dsp')}`, LUT `{delta.get('lut')}`",
        f"- ip_package: `{ip_package['status']}`",
        f"- overlay: `{overlay['status']}`",
        f"- physical_smoke: `{physical_smoke['status']}`",
        f"- pynq_plumbing: `{pynq['status']}`",
        f"- promotion: `{audit['promotion']['status']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail | Threshold |",
        "|---|---|---|---|",
    ]
    for check in audit["checks"]:
        detail = str(check.get("detail")).replace("|", "\\|")
        threshold = str(check.get("threshold", "")).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} | {threshold} |")
    lines.extend(
        [
            "",
            "## Optional Promotion Evidence",
            "",
            f"- ip_component_xml: `{ip_package['component_xml']}`",
            f"- ip_export_zip: `{ip_package['export_zip']}`",
            f"- ip_command: `{ip_package['command']}`",
            f"- overlay_bit: `{overlay['overlay_bit']}`",
            f"- overlay_hwh: `{overlay['overlay_hwh']}`",
            f"- pynq_bit: `{overlay['pynq_bit']}`",
            f"- pynq_hwh: `{overlay['pynq_hwh']}`",
            f"- timing_report: `{overlay['timing_report']}`",
            f"- route_status_report: `{overlay['route_status_report']}`",
            f"- overlay_command: `{overlay['command']}`",
            f"- physical_smoke_checked_paths: `{', '.join(physical_smoke['checked_paths'])}`",
            f"- pynq_variant: `{pynq['variant']}`",
            f"- pynq_preset: `{pynq['preset']}`",
            f"- pynq_package_command: `{pynq['package_command']}`",
            f"- pynq_session_command: `{pynq['session_command']}`",
            f"- pynq_remote_dry_run_command: `{pynq['remote_dry_run_command']}`",
            f"- pynq_dry_run_import_command: `{pynq['dry_run_import_command']}`",
        ]
    )
    lines.extend(["", "## Remaining", ""])
    if audit["promotion"]["remaining"]:
        for item in audit["promotion"]["remaining"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Safety", ""])
    for key, value in audit["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write VREF-P0-02 QKV cache URAM successor evidence.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--json-out", type=Path, default=Path(f"generated/signoff/vref_p0_qkv_uram_cache_successor_{DATE_TAG}.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path(f"generated/signoff/vref_p0_qkv_uram_cache_successor_{DATE_TAG}.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audit))
    print(
        "[vref-p0-qkv-uram-cache-successor] "
        f"status={audit['status']} checks={audit['pass_count']}/{audit['check_count']}"
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
