#!/usr/bin/env python3
"""Run an instrumented direct xsimk progress probe for the HGTXR HLS snapshot."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

from check_hls_xsimk_direct_probe import (
    DEFAULT_PROFILE,
    default_output_path,
    parse_rtl_progress,
    sim_root_for_profile,
)
from check_xsim_snapshot_smoke import run_command, run_xsimk_mi_sequence, vivado_env


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XILINX_ROOT = Path("/tools/Xilinx")
DEFAULT_WORKDIR = Path("/tmp/hgtxr_xsim_progress_probe_full_300")
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_hls_xsimk_progress_probe_2026_06_29.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_hls_xsimk_progress_probe_2026_06_29.md"
XELAB_ARGS = [
    "xil_defaultlib.apatb_hgtxr_e2e_axis_top_top",
    "glbl",
    "-Oenable_linking_all_libraries",
    "-prj",
    "hgtxr_e2e_axis_top.prj",
    "-L",
    "smartconnect_v1_0",
    "-L",
    "axi_protocol_checker_v1_1_12",
    "-L",
    "axi_protocol_checker_v1_1_13",
    "-L",
    "axis_protocol_checker_v1_1_11",
    "-L",
    "axis_protocol_checker_v1_1_12",
    "-L",
    "xil_defaultlib",
    "-L",
    "unisims_ver",
    "-L",
    "xpm",
    "-L",
    "floating_point_v7_1_16",
    "-L",
    "floating_point_v7_0_21",
    "--lib",
    "ieee_proposed=./ieee_proposed",
    "-s",
    "hgtxr_e2e_axis_top",
]


def patch_progress_timeout(autotb: Path, progress_timeout: int) -> None:
    text = autotb.read_text(errors="ignore")
    patched = re.sub(r"parameter PROGRESS_TIMEOUT = [0-9]+;", f"parameter PROGRESS_TIMEOUT = {progress_timeout};", text)
    if patched == text:
        raise RuntimeError(f"Did not patch PROGRESS_TIMEOUT in {autotb}")
    autotb.write_text(patched)


def prepare_workdir(source_sim_root: Path, workdir: Path, progress_timeout: int) -> Path:
    sim_copy = workdir / "sim"
    verilog_copy = sim_copy / "verilog"
    if workdir.exists():
        shutil.rmtree(workdir)
    sim_copy.mkdir(parents=True)
    shutil.copytree(source_sim_root, verilog_copy)
    shutil.copytree(source_sim_root.parent / "tv", sim_copy / "tv")
    patch_progress_timeout(verilog_copy / "hgtxr_e2e_axis_top.autotb.v", progress_timeout)
    return verilog_copy


def build_probe(
    xilinx_root: Path,
    profile: str,
    workdir: Path,
    progress_timeout: int,
    timeout_s: int,
    search_tail: int,
    track_tail: int,
) -> dict[str, Any]:
    source_sim_root = sim_root_for_profile(profile)
    if not source_sim_root.exists():
        return {
            "status": "missing-sim-root",
            "profile": profile,
            "source_sim_root": str(source_sim_root),
            "workdir": str(workdir),
            "progress_timeout": progress_timeout,
            "timeout_s": timeout_s,
            "expected": {
                "search_tail": search_tail,
                "track_tail": track_tail,
            },
            "xelab": {
                "returncode": None,
                "snapshot_built": False,
                "stdout_tail": "sim/verilog directory not found for requested profile",
            },
            "xsimk": {
                "command": None,
                "returncode": None,
                "timed_out": False,
                "elapsed_s": None,
                "stdin": "",
                "stdout_tail": "",
            },
            "checks": {
                "entered_kernel": False,
                "exec_run_complete": False,
                "exec_continue_started": False,
                "rtl_progress_samples": 0,
                "rtl_first_progress": None,
                "rtl_last_progress": None,
                "rtl_max_progress_pct": None,
                "c_vector_pass_seen": False,
                "search_expected_seen": False,
                "track_expected_seen": False,
            },
            "interpretation": "The requested profile does not have an HLS sim/verilog snapshot yet. Run HLS cosim or generate the RTL simulation snapshot before using the progress probe.",
        }
    verilog_dir = prepare_workdir(source_sim_root, workdir, progress_timeout)
    env = vivado_env(xilinx_root)
    xelab = run_command(
        [str(xilinx_root / "Vivado" / "2023.2" / "bin" / "xelab"), *XELAB_ARGS],
        verilog_dir,
        env,
        timeout_s=180,
    )
    xsimk = verilog_dir / "xsim.dir" / "hgtxr_e2e_axis_top" / "xsimk"
    xsimk_result = (
        run_xsimk_mi_sequence([str(xsimk)], verilog_dir, env, timeout_s)
        if xsimk.exists() and xelab["returncode"] == 0
        else {"cmd": [str(xsimk)], "returncode": None, "stdin": "", "stdout": "xsimk not available", "timed_out": False}
    )
    stdout = xsimk_result.get("stdout", "")
    progress = parse_rtl_progress(stdout)
    last_progress = progress[-1] if progress else None
    elapsed_s = xsimk_result.get("elapsed_s")
    progress_rate_pct_per_s = None
    estimated_one_transaction_s = None
    estimated_two_transaction_s = None
    estimated_remaining_current_transaction_s = None
    if isinstance(elapsed_s, (int, float)) and elapsed_s > 0 and last_progress:
        progress_rate_pct_per_s = last_progress["pct"] / elapsed_s
        if progress_rate_pct_per_s > 0:
            estimated_one_transaction_s = 100.0 / progress_rate_pct_per_s
            estimated_two_transaction_s = 200.0 / progress_rate_pct_per_s
            estimated_remaining_current_transaction_s = max(0.0, 100.0 - last_progress["pct"]) / progress_rate_pct_per_s
    pass_seen = "E2E AXIS vector comparison passed" in stdout
    search_seen = f"e2e_axis_out[5]={search_tail}" in stdout
    track_seen = f"e2e_axis_track_out[5]={track_tail}" in stdout
    status = (
        "pass"
        if pass_seen and search_seen and track_seen
        else "progress-timeout"
        if xsimk_result.get("timed_out") and len(progress) > 1
        else "direct-kernel-timeout"
        if xsimk_result.get("timed_out")
        else "fail"
        if xelab["returncode"] != 0
        else "direct-kernel-ended-without-pass"
    )
    return {
        "status": status,
        "profile": profile,
        "source_sim_root": str(source_sim_root),
        "workdir": str(workdir),
        "progress_timeout": progress_timeout,
        "timeout_s": timeout_s,
        "expected": {
            "search_tail": search_tail,
            "track_tail": track_tail,
        },
        "xelab": {
            "returncode": xelab["returncode"],
            "snapshot_built": "Built simulation snapshot hgtxr_e2e_axis_top" in xelab["stdout"],
            "stdout_tail": xelab["stdout"][-4000:],
        },
        "xsimk": {
            "command": xsimk_result.get("cmd"),
            "returncode": xsimk_result.get("returncode"),
            "timed_out": bool(xsimk_result.get("timed_out")),
            "elapsed_s": elapsed_s,
            "stdin": xsimk_result.get("stdin"),
            "stdout_tail": stdout[-8000:],
        },
        "checks": {
            "entered_kernel": "elaboration-done" in stdout,
            "exec_run_complete": 'reason="run-complete"' in stdout,
            "exec_continue_started": xsimk_result.get("stdin", "").count("-exec-continue") > 0,
            "rtl_progress_samples": len(progress),
            "rtl_first_progress": progress[0] if progress else None,
            "rtl_last_progress": last_progress,
            "rtl_max_progress_pct": max([p["pct"] for p in progress], default=None),
            "c_vector_pass_seen": pass_seen,
            "search_expected_seen": search_seen,
            "track_expected_seen": track_seen,
        },
        "estimates": {
            "progress_rate_pct_per_s": progress_rate_pct_per_s,
            "estimated_one_transaction_s": estimated_one_transaction_s,
            "estimated_two_transaction_s": estimated_two_transaction_s,
            "estimated_remaining_current_transaction_s": estimated_remaining_current_transaction_s,
        },
        "interpretation": (
            "Instrumented direct xsimk execution completed with expected Search and Track RTL outputs."
            if status == "pass"
            else "Instrumented direct xsimk execution shows RTL simulation time is advancing, but did not complete the full HLS cosim within the bounded timeout."
            if status == "progress-timeout"
            else "Instrumented direct xsimk execution did not provide enough progress evidence."
        ),
    }


def render_markdown(probe: dict[str, Any]) -> str:
    checks = probe["checks"]
    estimates = probe.get("estimates", {})
    lines = [
        "# Prefetch-All4 300MHz HLS Instrumented XSIMK Progress Probe",
        "",
        f"- status: `{probe['status']}`",
        f"- profile: `{probe['profile']}`",
        f"- source sim root: `{probe['source_sim_root']}`",
        f"- workdir: `{probe['workdir']}`",
        f"- progress timeout: `{probe['progress_timeout']}`",
        f"- run timeout: `{probe['timeout_s']} s`",
        f"- xsimk elapsed: `{probe['xsimk'].get('elapsed_s')} s`",
        f"- expected search tail: `{probe['expected']['search_tail']}`",
        f"- expected track tail: `{probe['expected']['track_tail']}`",
        "",
        "## Checks",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]
    for name in sorted(checks):
        lines.append(f"| `{name}` | `{checks[name]}` |")
    if estimates:
        lines.extend(["", "## Estimates", "", "| Estimate | Value |", "|---|---:|"])
        for name in sorted(estimates):
            lines.append(f"| `{name}` | `{estimates[name]}` |")
    lines.extend(["", "## Interpretation", "", probe["interpretation"], "", "## Stdout Tail", "", "```text", probe["xsimk"]["stdout_tail"], "```"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an instrumented direct xsimk progress probe.")
    parser.add_argument("--xilinx-root", type=Path, default=DEFAULT_XILINX_ROOT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--progress-timeout", type=int, default=100000)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--search-tail", type=int, default=-1125)
    parser.add_argument("--track-tail", type=int, default=-239)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_out = args.json_out or default_output_path(args.profile, "hls_xsimk_progress_probe_2026_06_29.json", DEFAULT_JSON)
    markdown_out = args.markdown_out or default_output_path(args.profile, "hls_xsimk_progress_probe_2026_06_29.md", DEFAULT_MARKDOWN)
    probe = build_probe(
        args.xilinx_root.resolve(),
        args.profile,
        args.workdir,
        args.progress_timeout,
        args.timeout_s,
        args.search_tail,
        args.track_tail,
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(probe))
    print(json.dumps(probe, indent=2, sort_keys=True))
    return 0 if probe["status"] in {"pass", "progress-timeout", "direct-kernel-timeout"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
