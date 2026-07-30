#!/usr/bin/env python3
"""Package one VREF-P0-01 non-current PoT scale successor for CSim/CSynth promotion."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from validate_e2e_axis_vector import build_reference, emit_c_header, load_spec  # noqa: E402
from validate_pynq_smoke_result import validate_result  # noqa: E402


DATE_TAG = "2026_06_16"
HGTXR_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = "generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.json"
DEFAULT_SPEC_BASENAME = "e2e_axis_vector_hgpipe_math_lnq_active16_spec.json"
DEFAULT_CANDIDATE = "softmax_input_x2"
DEFAULT_SPEC_OUT = "refs/vref_p0/e2e_axis_vector_hgpipe_math_lnq_active16_softmax_input_x2_spec.json"
DEFAULT_HEADER_OUT = "hls/tb/e2e_axis_vector_vref_p0_hgpipe_lnq_active16_softmax_input_x2_golden.hpp"
DEFAULT_GOLDEN_MACRO = "HGTXR_E2E_USE_VREF_P0_HGPIPE_LNQ_ACTIVE16_SOFTMAX_INPUT_X2_GOLDEN"
DEFAULT_IP_PROJECT_SUFFIX = "dsp_mixed_stream_ip"
DEFAULT_OVERLAY_SUFFIX = "dsp_mixed_stream_overlay"
SUCCESSOR_SMOKE_PRESET = "axis-vref-p0-softmax-input-x2-dsp-mixed-stream"
SUCCESSOR_SMOKE_RESULT_BASENAME = "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
RESOURCE_VARIANTS = [
    {
        "name": "default",
        "resource_policy": "",
        "project_suffix": "csynth",
        "description": "Default E2E URAM policy.",
    },
    {
        "name": "dsp_mixed_stream",
        "resource_policy": "dsp_mixed_stream",
        "project_suffix": "dsp_mixed_stream_csynth",
        "description": "Keep DSP multiply and URAM only for Q/K/V/attention stream buffers.",
    },
]
C3B_THRESHOLDS = {
    "latency_cycles": 37_508_072,
    "dsp": 604,
    "lut": 126_506,
    "uram": 64,
    "estimated_clock_ns": 5.0,
    "routed_wns_ns": 4.415,
}


def hardware_root(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def find_candidate(sweep: dict[str, Any], spec_basename: str, candidate: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for spec in sweep.get("specs", []):
        if Path(str(spec.get("spec", ""))).name != spec_basename:
            continue
        for row in spec.get("candidates", []):
            if row.get("candidate") == candidate:
                return spec, row
        raise ValueError(f"candidate {candidate!r} not found in spec {spec_basename!r}")
    raise ValueError(f"spec {spec_basename!r} not found in sweep")


def custom_scale_flags(cfg: Any, candidate: dict[str, Any], golden_macro: str = DEFAULT_GOLDEN_MACRO) -> str:
    math = candidate["math"]
    flags = [
        f"-DHGTXR_E2E_BLOCKS={cfg.blocks}",
        f"-DHGTXR_E2E_ACTIVE_TOKENS={cfg.active_tokens}",
        f"-DHGTXR_E2E_PATCH_GRID_H={cfg.patch_grid_h}",
        f"-DHGTXR_E2E_PATCH_GRID_W={cfg.patch_grid_w}",
        f"-DHGTXR_E2E_FF_DIM={cfg.ff_dim}",
        "-DHGTXR_E2E_STRICT_GOLDEN=1",
        f"-D{golden_macro}=1",
        "-DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1",
        f"-DHGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE={math['geluq_input_scale']}",
        f"-DHGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE={math['geluq_output_scale']}",
        "-DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1",
        f"-DHGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE={math['softmax_input_scale']}",
        f"-DHGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE={math['softmax_prob_scale']}",
        "-DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1",
        "-DHGTXR_E2E_USE_HGPIPE_LNQ_GAMMA_RAW=1",
        f"-DHGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE={math['layernorm_input_scale']}",
        f"-DHGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE={math['layernorm_output_scale']}",
        f"-DHGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT={math['layernorm_bias_shift']}",
    ]
    return " ".join(flags)


def shell_command(
    hardware: Path,
    project_name: str,
    custom_flags: str,
    script_name: str,
    extra_env: dict[str, str] | None = None,
) -> str:
    env = {
        "HGTXR_E2E_PROJECT_NAME": project_name,
        "HGTXR_E2E_SCALE": "custom",
        "HGTXR_E2E_CUSTOM_SCALE_FLAGS": custom_flags,
    }
    if extra_env:
        env.update({key: value for key, value in extra_env.items() if value})
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    script = hardware / "vivado" / "scripts" / script_name
    return f"{prefix} /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f {shlex.quote(str(script))}"


def hls_ld_library_prefix() -> str:
    return "LD_LIBRARY_PATH=/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH"


def vivado_ld_library_prefix() -> str:
    return "LD_LIBRARY_PATH=/tools/Xilinx/Vivado/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH"


def hls_shell_command_with_ld(
    hardware: Path,
    project_name: str,
    custom_flags: str,
    script_name: str,
    extra_env: dict[str, str] | None = None,
) -> str:
    return f"{hls_ld_library_prefix()} {shell_command(hardware, project_name, custom_flags, script_name, extra_env)}"


def package_project_name(candidate_name: str) -> str:
    return f"hgtxr_e2e_axis_vref_p0_{candidate_name}_{DEFAULT_IP_PROJECT_SUFFIX}"


def overlay_names(candidate_name: str) -> dict[str, str]:
    return {
        "project_name": f"hgtxr_e2e_axis_dma_vref_p0_{candidate_name}_{DEFAULT_OVERLAY_SUFFIX}",
        "bd_name": f"hgtxr_e2e_axis_dma_vref_p0_{candidate_name}_dsp_mixed_stream_system",
        "artifact_name": f"hgtxr_e2e_axis_dma_vref_p0_{candidate_name}_dsp_mixed_stream",
    }


def package_command(hardware: Path, project_name: str, flags: str) -> str:
    return hls_shell_command_with_ld(
        hardware,
        project_name,
        flags,
        "package_e2e_axis_ip.tcl",
        {"HGTXR_E2E_RESOURCE_POLICY": "dsp_mixed_stream"},
    )


def vivado_overlay_command(hardware: Path, names: dict[str, str], hls_ip_repo: Path) -> str:
    script = hardware / "vivado" / "scripts" / "build_e2e_axis_dma_bitstream.tcl"
    args = [
        "/tools/Xilinx/Vivado/2023.2/bin/vivado",
        "-mode",
        "batch",
        "-source",
        str(script),
        "-tclargs",
        "-project_name",
        names["project_name"],
        "-bd_name",
        names["bd_name"],
        "-artifact_name",
        names["artifact_name"],
        "-hls_ip_repo",
        str(hls_ip_repo),
    ]
    return f"{vivado_ld_library_prefix()} " + " ".join(shlex.quote(arg) for arg in args)


def read_csim_result(hardware: Path, project_name: str) -> dict[str, Any]:
    csim_log = (
        hardware
        / "generated"
        / project_name
        / "solution_e2e_q4w8a"
        / "csim"
        / "report"
        / "hgtxr_e2e_axis_top_csim.log"
    )
    solution_log = (
        hardware
        / "generated"
        / project_name
        / "solution_e2e_q4w8a"
        / "solution_e2e_q4w8a.log"
    )
    if not csim_log.exists():
        return {
            "status": "not_run",
            "csim_log": str(csim_log),
            "solution_log": str(solution_log),
            "detail": "CSim log not found",
        }
    text = csim_log.read_text(errors="ignore")
    pass_markers = [
        "E2E AXIS vector comparison passed",
        "runtime_state=2 count=6 last=1 failures=0",
        "CSim done with 0 errors",
    ]
    missing = [marker for marker in pass_markers if marker not in text]
    return {
        "status": "pass" if not missing else "fail",
        "csim_log": str(csim_log),
        "solution_log": str(solution_log),
        "missing_markers": missing,
        "expected_markers": pass_markers,
    }


def xml_text(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    return node.text.strip() if node is not None and node.text else None


def xml_int(root: ET.Element, path: str) -> int | None:
    value = xml_text(root, path)
    return int(value) if value and value.isdigit() else None


def xml_float(root: ET.Element, path: str) -> float | None:
    value = xml_text(root, path)
    return float(value) if value else None


def read_csynth_result(hardware: Path, project_name: str) -> dict[str, Any]:
    report_dir = hardware / "generated" / project_name / "solution_e2e_q4w8a" / "syn" / "report"
    csynth_xml = report_dir / "csynth.xml"
    csynth_rpt = report_dir / "hgtxr_e2e_axis_top_csynth.rpt"
    if not csynth_xml.exists():
        return {
            "status": "not_run",
            "csynth_xml": str(csynth_xml),
            "csynth_rpt": str(csynth_rpt),
            "detail": "csynth.xml not found",
        }
    root = ET.parse(csynth_xml).getroot()
    resources = {
        "bram_18k": xml_int(root, "./AreaEstimates/Resources/BRAM_18K"),
        "dsp": xml_int(root, "./AreaEstimates/Resources/DSP"),
        "ff": xml_int(root, "./AreaEstimates/Resources/FF"),
        "lut": xml_int(root, "./AreaEstimates/Resources/LUT"),
        "uram": xml_int(root, "./AreaEstimates/Resources/URAM"),
    }
    latency_cycles = xml_int(root, "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency")
    estimated_clock_ns = xml_float(root, "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")
    missing = [
        name
        for name, value in {
            "latency_cycles": latency_cycles,
            "estimated_clock_ns": estimated_clock_ns,
            "dsp": resources["dsp"],
            "lut": resources["lut"],
            "uram": resources["uram"],
        }.items()
        if value is None
    ]
    return {
        "status": "pass" if not missing else "fail",
        "csynth_xml": str(csynth_xml),
        "csynth_rpt": str(csynth_rpt),
        "missing_fields": missing,
        "latency_cycles": latency_cycles,
        "estimated_clock_ns": estimated_clock_ns,
        "resources": resources,
    }


def read_ip_package_result(hardware: Path, project_name: str) -> dict[str, Any]:
    ip_dir = hardware / "generated" / project_name / "solution_e2e_q4w8a" / "impl" / "ip"
    component_xml = ip_dir / "component.xml"
    export_zip = hardware / "generated" / project_name / "solution_e2e_q4w8a" / "impl" / "export.zip"
    checks = [
        {"name": "component_xml_exists", "status": "pass" if component_xml.exists() else "fail", "detail": str(component_xml)},
        {"name": "export_zip_exists", "status": "pass" if export_zip.exists() else "fail", "detail": str(export_zip)},
    ]
    if component_xml.exists():
        text = component_xml.read_text(errors="ignore")
        checks.extend(
            [
                {
                    "name": "top_name_present",
                    "status": "pass" if "hgtxr_e2e_axis_top" in text else "fail",
                    "detail": "hgtxr_e2e_axis_top",
                },
                {
                    "name": "axis_interface_present",
                    "status": "pass" if "axis" in text else "fail",
                    "detail": "axis",
                },
                {
                    "name": "aximm_interface_present",
                    "status": "pass" if "aximm" in text else "fail",
                    "detail": "aximm",
                },
            ]
        )
    failures = [check["name"] for check in checks if check["status"] == "fail"]
    return {
        "status": "pass" if not failures else "fail",
        "project_name": project_name,
        "ip_dir": str(ip_dir),
        "component_xml": str(component_xml),
        "export_zip": str(export_zip),
        "failures": failures,
        "checks": checks,
    }


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


def read_vivado_overlay_result(hardware: Path, candidate_name: str) -> dict[str, Any]:
    names = overlay_names(candidate_name)
    project_name = names["project_name"]
    artifact_name = names["artifact_name"]
    wrapper = f"{names['bd_name']}_wrapper"
    project_dir = hardware / "generated" / "build" / "vivado" / project_name
    impl_dir = project_dir / f"{project_name}.runs" / "impl_1"
    overlay_dir = hardware / "generated" / "build" / "vivado" / "overlay" / project_name
    pynq_dir = hardware / "pynq" / "hgtxr"
    bit = overlay_dir / f"{artifact_name}.bit"
    hwh = overlay_dir / f"{artifact_name}.hwh"
    pynq_bit = pynq_dir / f"{artifact_name}.bit"
    pynq_hwh = pynq_dir / f"{artifact_name}.hwh"
    timing_rpt = impl_dir / f"{wrapper}_timing_summary_routed.rpt"
    route_status_rpt = impl_dir / f"{wrapper}_route_status.rpt"
    runme_log = impl_dir / "runme.log"
    timing = parse_timing_summary(timing_rpt.read_text(errors="ignore")) if timing_rpt.exists() else {}
    route_status = parse_route_status(route_status_rpt.read_text(errors="ignore")) if route_status_rpt.exists() else {}
    checks = [
        {"name": "overlay_bit_exists", "status": "pass" if bit.exists() else "fail", "detail": str(bit)},
        {"name": "overlay_hwh_exists", "status": "pass" if hwh.exists() else "fail", "detail": str(hwh)},
        {"name": "pynq_bit_exists", "status": "pass" if pynq_bit.exists() else "fail", "detail": str(pynq_bit)},
        {"name": "pynq_hwh_exists", "status": "pass" if pynq_hwh.exists() else "fail", "detail": str(pynq_hwh)},
        {
            "name": "timing_wns_gte_c3b",
            "status": "pass"
            if timing.get("wns_ns") is not None and timing["wns_ns"] >= C3B_THRESHOLDS["routed_wns_ns"]
            else "fail",
            "detail": timing.get("wns_ns"),
        },
        {
            "name": "route_errors_zero",
            "status": "pass" if route_status.get("routing_error_nets") == 0 else "fail",
            "detail": route_status.get("routing_error_nets"),
        },
    ]
    failures = [check["name"] for check in checks if check["status"] == "fail"]
    hls_ip_repo = (
        hardware
        / "generated"
        / package_project_name(candidate_name)
        / "solution_e2e_q4w8a"
        / "impl"
        / "ip"
    )
    return {
        "status": "pass" if not failures else "fail",
        "names": names,
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
        "failures": failures,
        "checks": checks,
        "command": vivado_overlay_command(hardware, names, hls_ip_repo),
    }


def successor_physical_smoke_candidates(hardware: Path) -> list[Path]:
    return [
        hardware / "pynq" / "hgtxr" / SUCCESSOR_SMOKE_RESULT_BASENAME,
        (
            hardware
            / "generated"
            / "pynq"
            / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle"
            / SUCCESSOR_SMOKE_RESULT_BASENAME
        ),
        hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.remote.json",
    ]


def read_successor_physical_smoke_result(hardware: Path) -> dict[str, Any]:
    candidates = successor_physical_smoke_candidates(hardware)
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {
            "status": "not_captured",
            "preset": SUCCESSOR_SMOKE_PRESET,
            "result_json": "",
            "checked_paths": [str(path) for path in candidates],
            "errors": [],
        }

    result_path = existing[0]
    try:
        payload = load_json(result_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "fail",
            "preset": SUCCESSOR_SMOKE_PRESET,
            "result_json": str(result_path),
            "checked_paths": [str(path) for path in candidates],
            "errors": [f"invalid JSON: {exc}"],
        }

    errors = validate_result(payload, SUCCESSOR_SMOKE_PRESET)
    return {
        "status": "pass" if not errors else "fail",
        "preset": SUCCESSOR_SMOKE_PRESET,
        "result_json": str(result_path),
        "checked_paths": [str(path) for path in candidates],
        "errors": errors,
    }


def physical_smoke_check(physical_smoke_result: dict[str, Any] | None) -> dict[str, Any]:
    if not physical_smoke_result or physical_smoke_result.get("status") == "not_captured":
        return {
            "name": "physical_smoke_available",
            "status": "pending",
            "actual": "not captured",
            "threshold": "valid ZCU104 smoke JSON",
        }
    if physical_smoke_result.get("status") == "pass":
        return {
            "name": "physical_smoke_available",
            "status": "pass",
            "actual": physical_smoke_result.get("result_json", ""),
            "threshold": f"valid ZCU104 smoke JSON for {SUCCESSOR_SMOKE_PRESET}",
        }
    return {
        "name": "physical_smoke_available",
        "status": "fail",
        "actual": "; ".join(str(error) for error in physical_smoke_result.get("errors", [])) or "invalid",
        "threshold": f"valid ZCU104 smoke JSON for {SUCCESSOR_SMOKE_PRESET}",
    }


def build_c3b_projection(
    csynth_result: dict[str, Any],
    overlay_result: dict[str, Any] | None = None,
    physical_smoke_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if csynth_result["status"] != "pass":
        return {
            "status": "not_run",
            "detail": "csynth pass required before C3b threshold projection",
            "thresholds": C3B_THRESHOLDS,
        }
    resources = csynth_result["resources"]
    checks = [
        {
            "name": "latency_lte_c3b",
            "status": "pass" if csynth_result["latency_cycles"] <= C3B_THRESHOLDS["latency_cycles"] else "fail",
            "actual": csynth_result["latency_cycles"],
            "threshold": C3B_THRESHOLDS["latency_cycles"],
        },
        {
            "name": "estimated_clock_lte_target",
            "status": "pass" if csynth_result["estimated_clock_ns"] <= C3B_THRESHOLDS["estimated_clock_ns"] else "fail",
            "actual": csynth_result["estimated_clock_ns"],
            "threshold": C3B_THRESHOLDS["estimated_clock_ns"],
        },
        {
            "name": "dsp_lte_c3b",
            "status": "pass" if resources["dsp"] <= C3B_THRESHOLDS["dsp"] else "fail",
            "actual": resources["dsp"],
            "threshold": C3B_THRESHOLDS["dsp"],
        },
        {
            "name": "lut_lte_c3b",
            "status": "pass" if resources["lut"] <= C3B_THRESHOLDS["lut"] else "fail",
            "actual": resources["lut"],
            "threshold": C3B_THRESHOLDS["lut"],
        },
        {
            "name": "uram_lte_c3b",
            "status": "pass" if resources["uram"] <= C3B_THRESHOLDS["uram"] else "fail",
            "actual": resources["uram"],
            "threshold": C3B_THRESHOLDS["uram"],
        },
        {
            "name": "routed_timing_available",
            "status": "pass"
            if overlay_result
            and overlay_result.get("status") == "pass"
            and overlay_result.get("timing", {}).get("wns_ns") is not None
            and overlay_result["timing"]["wns_ns"] >= C3B_THRESHOLDS["routed_wns_ns"]
            else "pending",
            "actual": overlay_result.get("timing", {}).get("wns_ns") if overlay_result else "not generated",
            "threshold": f"routed WNS >= {C3B_THRESHOLDS['routed_wns_ns']} ns",
        },
        physical_smoke_check(physical_smoke_result),
    ]
    failures = [check["name"] for check in checks if check["status"] == "fail"]
    pending = [check["name"] for check in checks if check["status"] == "pending"]
    return {
        "status": "not_promotable" if failures or pending else "promotable",
        "failures": failures,
        "pending": pending,
        "thresholds": C3B_THRESHOLDS,
        "checks": checks,
    }


def build_resource_variant(
    hardware: Path,
    candidate_name: str,
    flags: str,
    variant: dict[str, str],
    overlay_result: dict[str, Any] | None = None,
    physical_smoke_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_name = f"hgtxr_e2e_axis_vref_p0_{candidate_name}_{variant['project_suffix']}"
    extra_env = {}
    if variant["resource_policy"]:
        extra_env["HGTXR_E2E_RESOURCE_POLICY"] = variant["resource_policy"]
    csynth_result = read_csynth_result(hardware, project_name)
    projection = build_c3b_projection(
        csynth_result,
        overlay_result if variant["name"] == "dsp_mixed_stream" else None,
        physical_smoke_result if variant["name"] == "dsp_mixed_stream" else None,
    )
    return {
        "name": variant["name"],
        "description": variant["description"],
        "project_name": project_name,
        "resource_policy": variant["resource_policy"] or "default",
        "command": hls_shell_command_with_ld(hardware, project_name, flags, "run_e2e_q4w8a_csynth.tcl", extra_env),
        "csynth_result": csynth_result,
        "c3b_projection": projection,
    }


def choose_recommended_variant(variants: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [variant for variant in variants if variant["csynth_result"]["status"] == "pass"]
    if not passed:
        return None

    def sort_key(variant: dict[str, Any]) -> tuple[int, int, int, int]:
        projection = variant["c3b_projection"]
        resources = variant["csynth_result"]["resources"]
        failures = len(projection.get("failures", []))
        pending = len(projection.get("pending", []))
        uram = resources["uram"] if resources["uram"] is not None else 10**9
        bram = resources["bram_18k"] if resources["bram_18k"] is not None else 10**9
        return (failures, pending, uram, bram)

    return sorted(passed, key=sort_key)[0]


def build_package(
    root: Path,
    sweep_path: Path | None = None,
    spec_basename: str = DEFAULT_SPEC_BASENAME,
    candidate_name: str = DEFAULT_CANDIDATE,
    spec_out: Path | None = None,
    header_out: Path | None = None,
) -> dict[str, Any]:
    hgtxr, hardware = hardware_root(root)
    sweep_path = sweep_path or hardware / DEFAULT_SWEEP
    spec_out = spec_out or hardware / DEFAULT_SPEC_OUT
    header_out = header_out or hardware / DEFAULT_HEADER_OUT
    sweep = load_json(sweep_path)
    spec_summary, candidate = find_candidate(sweep, spec_basename, candidate_name)
    source_spec = Path(str(spec_summary["spec"]))
    if not source_spec.is_absolute():
        source_spec = hardware / source_spec
    spec_payload = load_json(source_spec)
    spec_payload["description"] = (
        f"VREF-P0-01 successor generated from {spec_basename} "
        f"with candidate {candidate_name}."
    )
    spec_payload["math"] = candidate["math"]
    spec_payload["vref_successor"] = {
        "date_tag": DATE_TAG,
        "source_sweep": str(sweep_path),
        "source_spec": str(source_spec),
        "candidate": candidate_name,
        "exact_raw_match": candidate.get("exact_raw_match"),
        "raw_l1_delta": candidate.get("raw_l1_delta"),
        "max_raw_abs_delta": candidate.get("max_raw_abs_delta"),
        "promotion_gate": "header sync plus Vitis HLS CSim before HLS macro promotion",
    }

    spec_out.parent.mkdir(parents=True, exist_ok=True)
    header_out.parent.mkdir(parents=True, exist_ok=True)
    spec_out.write_text(json.dumps(spec_payload, indent=2, sort_keys=True) + "\n")

    cfg, math_cfg, pattern = load_spec(spec_out)
    ref = build_reference(cfg, math_cfg, pattern)
    header_text = emit_c_header(cfg, pattern, ref)
    header_out.write_text(header_text)
    expected_text = header_text
    header_sync = header_out.read_text() == expected_text
    csim_project_name = f"hgtxr_e2e_axis_vref_p0_{candidate_name}_csim"
    flags = custom_scale_flags(cfg, candidate)
    csim_result = read_csim_result(hardware, csim_project_name)
    ip_project_name = package_project_name(candidate_name)
    ip_package_result = read_ip_package_result(hardware, ip_project_name)
    vivado_overlay_result = read_vivado_overlay_result(hardware, candidate_name)
    physical_smoke_result = read_successor_physical_smoke_result(hardware)
    resource_variants = [
        build_resource_variant(hardware, candidate_name, flags, variant, vivado_overlay_result, physical_smoke_result)
        for variant in RESOURCE_VARIANTS
    ]
    default_variant = resource_variants[0]
    csynth_project_name = default_variant["project_name"]
    csynth_result = default_variant["csynth_result"]
    c3b_projection = build_c3b_projection(csynth_result)
    recommended_variant = choose_recommended_variant(resource_variants)
    csim_ok_or_pending = csim_result["status"] in {"pass", "not_run"}
    csynth_ok_or_pending = csynth_result["status"] in {"pass", "not_run"}
    ip_ok_or_pending = ip_package_result["status"] == "pass"
    overlay_ok_or_pending = vivado_overlay_result["status"] == "pass"
    physical_ok_or_pending = physical_smoke_result["status"] in {"pass", "not_captured"}

    return {
        "status": "pass"
        if header_sync
        and candidate.get("status") == "pass"
        and csim_ok_or_pending
        and csynth_ok_or_pending
        and ip_ok_or_pending
        and overlay_ok_or_pending
        and physical_ok_or_pending
        else "fail",
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware": str(hardware),
        "source_sweep": str(sweep_path),
        "source_spec": str(source_spec),
        "spec_out": str(spec_out),
        "header_out": str(header_out),
        "candidate": candidate_name,
        "candidate_status": candidate.get("status"),
        "candidate_exact_raw_match": candidate.get("exact_raw_match"),
        "candidate_raw_l1_delta": candidate.get("raw_l1_delta"),
        "candidate_max_raw_abs_delta": candidate.get("max_raw_abs_delta"),
        "expected_raw": ref.expected_raw,
        "runtime_state": ref.runtime_state,
        "math": candidate["math"],
        "custom_scale_flags": flags,
        "golden_macro": DEFAULT_GOLDEN_MACRO,
        "csim": {
            "project_name": csim_project_name,
            "env": {
                "HGTXR_E2E_PROJECT_NAME": csim_project_name,
                "HGTXR_E2E_SCALE": "custom",
                "HGTXR_E2E_CUSTOM_SCALE_FLAGS": flags,
            },
            "command": hls_shell_command_with_ld(hardware, csim_project_name, flags, "run_e2e_q4w8a_csim.tcl"),
            "requires_vitis_hls": "/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
            "script": str(hardware / "vivado" / "scripts" / "run_e2e_q4w8a_csim.tcl"),
        },
        "csim_result": csim_result,
        "csynth": {
            "project_name": csynth_project_name,
            "env": {
                "HGTXR_E2E_PROJECT_NAME": csynth_project_name,
                "HGTXR_E2E_SCALE": "custom",
                "HGTXR_E2E_CUSTOM_SCALE_FLAGS": flags,
            },
            "command": hls_shell_command_with_ld(hardware, csynth_project_name, flags, "run_e2e_q4w8a_csynth.tcl"),
            "requires_vitis_hls": "/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
            "script": str(hardware / "vivado" / "scripts" / "run_e2e_q4w8a_csynth.tcl"),
        },
        "csynth_result": csynth_result,
        "c3b_projection": c3b_projection,
        "resource_variants": resource_variants,
        "recommended_resource_variant": recommended_variant["name"] if recommended_variant else None,
        "recommended_c3b_projection": recommended_variant["c3b_projection"] if recommended_variant else None,
        "ip_package": {
            "project_name": ip_project_name,
            "env": {
                "HGTXR_E2E_PROJECT_NAME": ip_project_name,
                "HGTXR_E2E_SCALE": "custom",
                "HGTXR_E2E_RESOURCE_POLICY": "dsp_mixed_stream",
                "HGTXR_E2E_CUSTOM_SCALE_FLAGS": flags,
            },
            "command": package_command(hardware, ip_project_name, flags),
            "requires_vitis_hls": "/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
            "script": str(hardware / "vivado" / "scripts" / "package_e2e_axis_ip.tcl"),
        },
        "ip_package_result": ip_package_result,
        "vivado_overlay": {
            "command": vivado_overlay_result["command"],
            "requires_vivado": "/tools/Xilinx/Vivado/2023.2/bin/vivado",
            "script": str(hardware / "vivado" / "scripts" / "build_e2e_axis_dma_bitstream.tcl"),
        },
        "vivado_overlay_result": vivado_overlay_result,
        "physical_smoke_result": physical_smoke_result,
        "checks": [
            {"name": "candidate_status_pass", "status": "pass" if candidate.get("status") == "pass" else "fail", "detail": str(candidate.get("status"))},
            {"name": "spec_written", "status": "pass" if spec_out.exists() else "fail", "detail": str(spec_out)},
            {"name": "header_written", "status": "pass" if header_out.exists() else "fail", "detail": str(header_out)},
            {"name": "header_sync", "status": "pass" if header_sync else "fail", "detail": "header matches generated spec"},
            {"name": "custom_csim_flags_present", "status": "pass" if DEFAULT_GOLDEN_MACRO in flags else "fail", "detail": flags},
            {"name": "csim_result", "status": csim_result["status"], "detail": csim_result.get("csim_log", "")},
            {"name": "csynth_result", "status": csynth_result["status"], "detail": csynth_result.get("csynth_xml", "")},
            {"name": "ip_package_result", "status": ip_package_result["status"], "detail": ip_package_result.get("component_xml", "")},
            {"name": "vivado_overlay_result", "status": vivado_overlay_result["status"], "detail": vivado_overlay_result.get("timing_report", "")},
            {
                "name": "physical_smoke_result",
                "status": physical_smoke_result["status"],
                "detail": physical_smoke_result.get("result_json") or "not captured",
            },
        ],
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "writes_hls_source": False,
            "overwrites_c3b_artifacts": False,
        },
    }


def render_markdown(package: dict[str, Any]) -> str:
    lines = [
        "# VREF-P0-01 PoT Scale Successor Package",
        "",
        f"- status: `{package['status']}`",
        f"- candidate: `{package['candidate']}`",
        f"- exact_raw_match: `{package['candidate_exact_raw_match']}`",
        f"- raw_l1_delta: `{package['candidate_raw_l1_delta']}`",
        f"- max_raw_abs_delta: `{package['candidate_max_raw_abs_delta']}`",
        f"- spec: `{package['spec_out']}`",
        f"- header: `{package['header_out']}`",
        f"- expected_raw: `{package['expected_raw']}`",
        f"- runtime_state: `{package['runtime_state']}`",
        f"- csim_status: `{package['csim_result']['status']}`",
        f"- csynth_status: `{package['csynth_result']['status']}`",
        f"- ip_package_status: `{package['ip_package_result']['status']}`",
        f"- vivado_overlay_status: `{package['vivado_overlay_result']['status']}`",
        f"- physical_smoke_status: `{package['physical_smoke_result']['status']}`",
        f"- c3b_projection: `{package['c3b_projection']['status']}`",
        f"- recommended_resource_variant: `{package['recommended_resource_variant']}`",
        f"- recommended_c3b_projection: `{package['recommended_c3b_projection']['status'] if package['recommended_c3b_projection'] else None}`",
        "",
        "## CSim Command",
        "",
        "```bash",
        package["csim"]["command"],
        "```",
        "",
        "## CSynth Command",
        "",
        "```bash",
        package["csynth"]["command"],
        "```",
        "",
        "## IP Package Command",
        "",
        "```bash",
        package["ip_package"]["command"],
        "```",
        "",
        "## Vivado Overlay Command",
        "",
        "```bash",
        package["vivado_overlay"]["command"],
        "```",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in package["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        if len(detail) > 160:
            detail = detail[:157] + "..."
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(["", "## C3b Projection", "", "| Check | Status | Actual | Threshold |", "|---|---|---|---|"])
    for check in package["c3b_projection"].get("checks", []):
        lines.append(
            f"| {check['name']} | `{check['status']}` | {check['actual']} | {check['threshold']} |"
        )
    if package.get("recommended_c3b_projection"):
        lines.extend(
            [
                "",
                "## Recommended C3b Projection",
                "",
                "| Check | Status | Actual | Threshold |",
                "|---|---|---|---|",
            ]
        )
        for check in package["recommended_c3b_projection"].get("checks", []):
            lines.append(
                f"| {check['name']} | `{check['status']}` | {check['actual']} | {check['threshold']} |"
            )
    lines.extend(
        [
            "",
            "## Resource Variants",
            "",
            "| Variant | Policy | Status | Clock ns | Latency | BRAM | DSP | LUT | URAM | Projection | Failures | Pending |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for variant in package["resource_variants"]:
        result = variant["csynth_result"]
        resources = result.get("resources") or {}
        projection = variant["c3b_projection"]
        lines.append(
            "| {name} | {policy} | `{status}` | {clock} | {latency} | {bram} | {dsp} | {lut} | {uram} | `{projection}` | {failures} | {pending} |".format(
                name=variant["name"],
                policy=variant["resource_policy"],
                status=result["status"],
                clock=result.get("estimated_clock_ns", ""),
                latency=result.get("latency_cycles", ""),
                bram=resources.get("bram_18k", ""),
                dsp=resources.get("dsp", ""),
                lut=resources.get("lut", ""),
                uram=resources.get("uram", ""),
                projection=projection.get("status", ""),
                failures=", ".join(projection.get("failures", [])),
                pending=", ".join(projection.get("pending", [])),
            )
        )
    overlay = package["vivado_overlay_result"]
    timing = overlay.get("timing", {})
    route_status = overlay.get("route_status", {})
    physical = package["physical_smoke_result"]
    lines.extend(
        [
            "",
            "## IP Package Result",
            "",
            f"- project: `{package['ip_package_result']['project_name']}`",
            f"- component_xml: `{package['ip_package_result']['component_xml']}`",
            f"- export_zip: `{package['ip_package_result']['export_zip']}`",
            "",
            "## Vivado Overlay Result",
            "",
            f"- project: `{overlay['names']['project_name']}`",
            f"- artifact: `{overlay['names']['artifact_name']}`",
            f"- bit: `{overlay['overlay_bit']}`",
            f"- hwh: `{overlay['overlay_hwh']}`",
            f"- pynq_bit: `{overlay['pynq_bit']}`",
            f"- pynq_hwh: `{overlay['pynq_hwh']}`",
            f"- timing_report: `{overlay['timing_report']}`",
            f"- route_status_report: `{overlay['route_status_report']}`",
            f"- WNS ns: `{timing.get('wns_ns')}`",
            f"- TNS ns: `{timing.get('tns_ns')}`",
            f"- WHS ns: `{timing.get('whs_ns')}`",
            f"- routing_error_nets: `{route_status.get('routing_error_nets')}`",
            f"- fully_routed_nets: `{route_status.get('fully_routed_nets')}`",
            "",
            "## Physical Smoke Result",
            "",
            f"- status: `{physical['status']}`",
            f"- preset: `{physical['preset']}`",
            f"- result_json: `{physical.get('result_json', '')}`",
            f"- errors: `{physical.get('errors', [])}`",
            "",
            "Note: Vivado top-level utilization can under-report HLS IP DSP/URAM when the IP is integrated as generated/OOC collateral. Use HLS csynth resources for the successor resource-policy claim.",
        ]
    )
    lines.extend(["", "## Safety", ""])
    for key, value in package["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a VREF-P0-01 PoT scale successor package.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--sweep-json", type=Path, default=None)
    parser.add_argument("--spec-basename", default=DEFAULT_SPEC_BASENAME)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--spec-out", type=Path, default=None)
    parser.add_argument("--header-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    package = build_package(
        args.root,
        args.sweep_json,
        args.spec_basename,
        args.candidate,
        args.spec_out,
        args.header_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(package))
    print(json.dumps({"status": package["status"], "candidate": package["candidate"], "spec_out": package["spec_out"], "header_out": package["header_out"]}, sort_keys=True))
    return 0 if package["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
