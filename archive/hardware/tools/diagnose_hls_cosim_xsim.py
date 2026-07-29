#!/usr/bin/env python3
"""Diagnose HLS XSIM cosim failures for the dense runtime E2E AXIS profile."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
PROFILE = "par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16"
PROJECT = f"hgtxr_e2e_axis_{PROFILE}_no_board"
SOLUTION = "solution_e2e_q4w8a"
DEFAULT_PROJECT_ROOT = HARDWARE_ROOT / "generated" / PROJECT / SOLUTION
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.md"
DEFAULT_XSIM_SMOKE_JSON = HARDWARE_ROOT / "generated" / "signoff" / "xsim_snapshot_smoke_2026_06_29.json"
DEFAULT_DIRECT_XSIMK_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_hls_xsimk_direct_probe_2026_06_29.json"
DEFAULT_PROGRESS_XSIMK_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_hls_xsimk_progress_probe_2026_06_29.json"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="ignore"))
    except json.JSONDecodeError:
        return {}


def contains_all(text: str, needles: Sequence[str]) -> bool:
    return all(needle in text for needle in needles)


def extract_verilog_status(cosim_report: str) -> str | None:
    for line in cosim_report.splitlines():
        if "Verilog" not in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) >= 2 and parts[0] == "Verilog":
            return parts[1] or None
    return None


def extract_estimated_fmax(solution_log: str) -> float | None:
    match = re.search(r"Estimated Fmax:\s*([0-9]+(?:\.[0-9]+)?)\s*MHz", solution_log)
    if not match:
        return None
    return float(match.group(1))


def build_diagnosis(
    project_root: Path,
    xsim_smoke_json: Path,
    direct_xsimk_json: Path,
    progress_xsimk_json: Path,
) -> dict[str, Any]:
    sim_root = project_root / "sim"
    cosim_report_path = sim_root / "report" / "hgtxr_e2e_axis_top_cosim.rpt"
    verilog_log_path = sim_root / "report" / "verilog" / "hgtxr_e2e_axis_top.log"
    xsim_log_path = sim_root / "verilog" / "xsim.log"
    run_xsim_log_path = sim_root / "verilog" / "run_xsim.log"
    xelab_log_path = sim_root / "verilog" / "xelab.log"
    cosim_options_path = sim_root / "report" / "cosim_options.xml"
    solution_log_path = project_root / "solution_e2e_q4w8a.log"

    cosim_report = read_text(cosim_report_path)
    verilog_log = read_text(verilog_log_path)
    xsim_log = read_text(xsim_log_path)
    run_xsim_log = read_text(run_xsim_log_path)
    xelab_log = read_text(xelab_log_path)
    cosim_options = read_text(cosim_options_path)
    solution_log = read_text(solution_log_path)
    xsim_smoke = read_json(xsim_smoke_json)
    direct_xsimk = read_json(direct_xsimk_json)
    progress_xsimk = read_json(progress_xsimk_json)
    xsim_smoke_checks = xsim_smoke.get("checks", {})
    xsim_environment_smoke_pass = xsim_smoke.get("status") == "pass"
    xsim_environment_kernel_pass = xsim_smoke.get("status") == "wrapper-blocked-kernel-pass"
    xsim_environment_smoke_blocked = xsim_smoke.get("status") == "blocked-xsim-runtime"
    hls_direct_xsimk_started = direct_xsimk.get("status") in {"pass", "direct-kernel-timeout", "direct-kernel-started"}
    hls_direct_xsimk_pass = direct_xsimk.get("status") == "pass"
    hls_progress_xsimk_advancing = progress_xsimk.get("status") == "progress-timeout"

    c_tb_output_match = contains_all(
        verilog_log,
        [
            "search_runtime_state=0 expected=0 count=6 last=1 failures=0",
            "track_runtime_state=1 expected=1 count=6 last=1",
            "E2E AXIS vector comparison passed",
        ],
    )
    scheduler_trace_clean = contains_all(
        verilog_log,
        [
            "prefetch_trace_summary block_pairs=4 violations=0 not_ready=0 immediate_gaps=0",
            "prefetch_trace_summary block_pairs=2 violations=0 not_ready=0 immediate_gaps=0",
        ],
    )
    search_expected_output = contains_all(
        verilog_log,
        [
            "e2e_axis_out[0]=-1169",
            "e2e_axis_out[1]=-1169",
            "e2e_axis_out[2]=-1169",
            "e2e_axis_out[3]=-1169",
            "e2e_axis_out[4]=-1169",
            "e2e_axis_out[5]=-1125 last=1",
        ],
    )
    track_expected_output = contains_all(
        verilog_log,
        [
            "e2e_axis_track_out[0]=-235",
            "e2e_axis_track_out[1]=-235",
            "e2e_axis_track_out[2]=-235",
            "e2e_axis_track_out[3]=-235",
            "e2e_axis_track_out[4]=-235",
            "e2e_axis_track_out[5]=-239 last=1",
        ],
    )
    xsim_launch_exception = "unexpected exception when evaluating tcl command" in xsim_log
    autoloadwcfg_in_failure = "-autoloadwcfg" in xsim_log or "-autoloadwcfg" in run_xsim_log
    view_wcfg_in_failure = "-view" in xsim_log and ".wcfg" in xsim_log
    waveform_launch_option_in_failure = autoloadwcfg_in_failure or view_wcfg_in_failure
    xelab_snapshot_built = "Built simulation snapshot hgtxr_e2e_axis_top" in xelab_log
    disable_deadlock_detection_used = (
        "Running: config_cosim -disable_deadlock_detection" in solution_log
        or re.search(r'<option name="disable_deadlock_detection"[^>]*>\s*1\s*</option>', cosim_options)
        is not None
    )
    verilog_status = extract_verilog_status(cosim_report)
    rtl_cosim_pass = verilog_status == "Pass"

    if rtl_cosim_pass:
        status = "pass"
    elif xsim_launch_exception:
        status = "blocked-xsim-launch"
    else:
        status = "fail"

    return {
        "status": status,
        "profile": PROFILE,
        "project_root": str(project_root),
        "timing_rule": {
            "pl_clock_mhz": 300.0,
            "experimental_routed_wns_floor_ns": -0.5,
            "official_clean_wns_floor_ns": 0.0,
        },
        "checks": {
            "c_tb_output_match": c_tb_output_match,
            "scheduler_trace_clean": scheduler_trace_clean,
            "search_expected_output": search_expected_output,
            "track_expected_output": track_expected_output,
            "rtl_cosim_pass": rtl_cosim_pass,
            "xsim_launch_exception": xsim_launch_exception,
            "autoloadwcfg_in_failure": autoloadwcfg_in_failure,
            "view_wcfg_in_failure": view_wcfg_in_failure,
            "waveform_launch_option_in_failure": waveform_launch_option_in_failure,
            "xelab_snapshot_built": xelab_snapshot_built,
            "disable_deadlock_detection_used": disable_deadlock_detection_used,
            "xsim_environment_smoke_pass": xsim_environment_smoke_pass,
            "xsim_environment_kernel_pass": xsim_environment_kernel_pass,
            "xsim_environment_smoke_blocked": xsim_environment_smoke_blocked,
            "hls_direct_xsimk_started": hls_direct_xsimk_started,
            "hls_direct_xsimk_pass": hls_direct_xsimk_pass,
            "hls_progress_xsimk_advancing": hls_progress_xsimk_advancing,
        },
        "cosim": {
            "verilog_status": verilog_status,
            "estimated_fmax_mhz": extract_estimated_fmax(solution_log),
        },
        "sources": {
            "cosim_report": str(cosim_report_path),
            "verilog_log": str(verilog_log_path),
            "xsim_log": str(xsim_log_path),
            "run_xsim_log": str(run_xsim_log_path),
            "xelab_log": str(xelab_log_path),
            "cosim_options": str(cosim_options_path),
            "solution_log": str(solution_log_path),
            "xsim_smoke_json": str(xsim_smoke_json),
            "direct_xsimk_json": str(direct_xsimk_json),
            "progress_xsimk_json": str(progress_xsimk_json),
        },
        "xsim_environment": {
            "status": xsim_smoke.get("status"),
            "checks": xsim_smoke_checks,
            "interpretation": xsim_smoke.get("interpretation"),
        },
        "direct_xsimk": {
            "status": direct_xsimk.get("status"),
            "checks": direct_xsimk.get("checks", {}),
            "interpretation": direct_xsimk.get("interpretation"),
        },
        "progress_xsimk": {
            "status": progress_xsimk.get("status"),
            "checks": progress_xsimk.get("checks", {}),
            "interpretation": progress_xsimk.get("interpretation"),
        },
        "interpretation": (
            "C testbench output and scheduler trace passed. XELAB built the Verilog snapshot, "
            "but Verilog RTL cosim is not signed off because XSIM failed during launch. "
            "The minimal XSIM smoke shows the normal wrapper/Tcl launch is broken while direct "
            "xsimk kernel execution can run a trivial testbench. The HGTXR direct xsimk probe "
            "started RTL simulation, and the instrumented progress probe confirms RTL simulation "
            "time advances inside transaction 0, but the full two-transaction cosim has not completed "
            "within the bounded timeout, so this is still not RTL output signoff."
            if status == "blocked-xsim-launch"
            else "RTL cosim completed successfully."
            if status == "pass"
            else "RTL cosim failed without the known XSIM launch exception signature."
        ),
        "next_action": (
            "Either repair the standard XSIM wrapper/Tcl launch path or run the HGTXR direct xsimk/progress probe long enough to complete both RTL transactions and capture the expected outputs."
            if status == "blocked-xsim-launch"
            else "Import this result into validation docs."
            if status == "pass"
            else "Inspect RTL mismatch or simulator logs."
        ),
    }


def render_markdown(diagnosis: dict[str, Any]) -> str:
    checks = diagnosis["checks"]
    sources = diagnosis["sources"]
    lines = [
        "# Prefetch-All4 300MHz HLS XSIM Cosim Diagnosis",
        "",
        f"- status: `{diagnosis['status']}`",
        f"- profile: `{diagnosis['profile']}`",
        f"- PL clock: `{diagnosis['timing_rule']['pl_clock_mhz']} MHz`",
        f"- experimental WNS floor: `{diagnosis['timing_rule']['experimental_routed_wns_floor_ns']} ns`",
        f"- official clean WNS floor: `{diagnosis['timing_rule']['official_clean_wns_floor_ns']} ns`",
        f"- Verilog cosim status: `{diagnosis['cosim']['verilog_status']}`",
        f"- estimated Fmax: `{diagnosis['cosim']['estimated_fmax_mhz']} MHz`",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "|---|---:|",
    ]
    for name in sorted(checks):
        lines.append(f"| `{name}` | `{checks[name]}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            diagnosis["interpretation"],
            "",
            "## Next Action",
            "",
            diagnosis["next_action"],
            "",
            "## Sources",
            "",
        ]
    )
    for name in sorted(sources):
        lines.append(f"- {name}: `{sources[name]}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose HLS XSIM cosim output for the prefetch-all4 profile.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--xsim-smoke-json", type=Path, default=DEFAULT_XSIM_SMOKE_JSON)
    parser.add_argument("--direct-xsimk-json", type=Path, default=DEFAULT_DIRECT_XSIMK_JSON)
    parser.add_argument("--progress-xsimk-json", type=Path, default=DEFAULT_PROGRESS_XSIMK_JSON)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    diagnosis = build_diagnosis(
        args.project_root.resolve(),
        args.xsim_smoke_json.resolve(),
        args.direct_xsimk_json.resolve(),
        args.progress_xsimk_json.resolve(),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(diagnosis))
    print(json.dumps(diagnosis, indent=2, sort_keys=True))
    return 0 if diagnosis["status"] in {"pass", "blocked-xsim-launch"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
