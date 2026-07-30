#!/usr/bin/env python3
"""Probe the HGTXR HLS RTL snapshot through direct xsimk MI commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from check_xsim_snapshot_smoke import run_xsimk_mi_sequence, vivado_env


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = "par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16"
SOLUTION = "solution_e2e_q4w8a"
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_hls_xsimk_direct_probe_2026_06_29.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_hls_xsimk_direct_probe_2026_06_29.md"
DEFAULT_XILINX_ROOT = Path("/tools/Xilinx")
TV_OUTPUT_PORTS = (
    "axis_out_V_data_V",
    "axis_out_V_keep_V",
    "axis_out_V_strb_V",
    "axis_out_V_last_V",
    "gmem_e2e_runtime",
)


def parse_rtl_progress(stdout: str) -> list[dict[str, Any]]:
    progress = []
    pattern = re.compile(
        r"RTL Simulation :\s+([0-9]+)\s*/\s*([0-9]+)\s*\[([0-9.]+)%\]\s*@\s*\\?\"([0-9]+)\\?\""
    )
    for match in pattern.finditer(stdout):
        progress.append(
            {
                "completed_transactions": int(match.group(1)),
                "total_transactions": int(match.group(2)),
                "pct": float(match.group(3)),
                "sim_time": int(match.group(4)),
            }
        )
    return progress


def sim_root_for_profile(profile: str) -> Path:
    return HARDWARE_ROOT / "generated" / f"hgtxr_e2e_axis_{profile}_no_board" / SOLUTION / "sim" / "verilog"


def xsimk_for_sim_root(sim_root: Path) -> Path:
    return sim_root / "xsim.dir" / "hgtxr_e2e_axis_top" / "xsimk"


def normalize_tv_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("[[transaction]]"):
        parts = stripped.split()
        if len(parts) >= 2:
            return f"[[transaction]] {parts[-1]}"
    return stripped


def normalized_tv_lines(path: Path) -> list[str]:
    return [normalize_tv_line(line) for line in path.read_text(errors="replace").splitlines()]


def compare_tv_outputs(sim_root: Path) -> dict[str, Any]:
    tv_root = sim_root.parent / "tv"
    comparisons = []
    all_present = True
    all_equal = True
    for port in TV_OUTPUT_PORTS:
        c_path = tv_root / "cdatafile" / f"c.hgtxr_e2e_axis_top.autotvout_{port}.dat"
        rtl_path = tv_root / "rtldatafile" / f"rtl.hgtxr_e2e_axis_top.autotvout_{port}.dat"
        c_exists = c_path.exists()
        rtl_exists = rtl_path.exists()
        all_present = all_present and c_exists and rtl_exists
        entry: dict[str, Any] = {
            "port": port,
            "c_path": str(c_path),
            "rtl_path": str(rtl_path),
            "c_exists": c_exists,
            "rtl_exists": rtl_exists,
            "equal": False,
        }
        if c_exists and rtl_exists:
            c_bytes = c_path.read_bytes()
            rtl_bytes = rtl_path.read_bytes()
            c_lines = normalized_tv_lines(c_path)
            rtl_lines = normalized_tv_lines(rtl_path)
            equal = c_lines == rtl_lines
            entry.update(
                {
                    "c_bytes": len(c_bytes),
                    "rtl_bytes": len(rtl_bytes),
                    "c_sha256": hashlib.sha256(c_bytes).hexdigest(),
                    "rtl_sha256": hashlib.sha256(rtl_bytes).hexdigest(),
                    "normalized_equal": equal,
                    "equal": equal,
                    "c_line_count": len(c_lines),
                    "rtl_line_count": len(rtl_lines),
                }
            )
            if not equal:
                first_mismatch = None
                for idx, (c_line, rtl_line) in enumerate(zip(c_lines, rtl_lines), start=1):
                    if c_line != rtl_line:
                        first_mismatch = {
                            "line": idx,
                            "c": c_line,
                            "rtl": rtl_line,
                        }
                        break
                if first_mismatch is None and len(c_lines) != len(rtl_lines):
                    first_mismatch = {
                        "line": min(len(c_lines), len(rtl_lines)) + 1,
                        "c": "<missing>" if len(c_lines) < len(rtl_lines) else c_lines[-1],
                        "rtl": "<missing>" if len(rtl_lines) < len(c_lines) else rtl_lines[-1],
                    }
                entry["first_mismatch"] = first_mismatch
        else:
            entry["normalized_equal"] = False
        all_equal = all_equal and bool(entry["equal"])
        comparisons.append(entry)
    return {
        "tv_root": str(tv_root),
        "ports": comparisons,
        "all_present": all_present,
        "all_equal": all_present and all_equal,
    }


def default_output_path(profile: str, suffix: str, default_path: Path) -> Path:
    if profile == DEFAULT_PROFILE:
        return default_path
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile)
    return HARDWARE_ROOT / "generated" / "signoff" / f"{safe}_{suffix}"


def classify_probe(
    result: dict[str, Any],
    xsimk_path: Path,
    search_tail: int,
    track_tail: int,
    tv_comparison: dict[str, Any],
) -> dict[str, Any]:
    stdout = result.get("stdout", "")
    progress = parse_rtl_progress(stdout)
    last_progress = progress[-1] if progress else None
    checks = {
        "xsimk_exists": xsimk_path.exists(),
        "entered_kernel": "elaboration-done" in stdout,
        "exec_run_complete": 'reason="run-complete"' in stdout,
        "exec_continue_started": result.get("stdin", "").count("-exec-continue") > 0,
        "rtl_progress_banner_seen": "Inter-Transaction Progress" in stdout,
        "rtl_transaction_started": 'RTL Simulation : 0 / 2' in stdout,
        "rtl_progress_samples": len(progress),
        "rtl_last_completed_transactions": last_progress["completed_transactions"] if last_progress else None,
        "rtl_total_transactions": last_progress["total_transactions"] if last_progress else None,
        "rtl_last_progress_pct": last_progress["pct"] if last_progress else None,
        "rtl_last_sim_time": last_progress["sim_time"] if last_progress else None,
        "c_vector_pass_seen": "E2E AXIS vector comparison passed" in stdout,
        "search_expected_seen": f"e2e_axis_out[5]={search_tail}" in stdout,
        "track_expected_seen": f"e2e_axis_track_out[5]={track_tail}" in stdout,
        "tv_outputs_present": bool(tv_comparison.get("all_present")),
        "tv_outputs_equal": bool(tv_comparison.get("all_equal")),
        "timed_out": bool(result.get("timed_out")),
    }
    if checks["c_vector_pass_seen"] and checks["search_expected_seen"] and checks["track_expected_seen"]:
        status = "pass"
    elif (
        checks["rtl_last_completed_transactions"] == checks["rtl_total_transactions"]
        and checks["rtl_total_transactions"]
        and checks["tv_outputs_equal"]
    ):
        status = "pass-tv-output-match"
    elif (
        checks["rtl_last_completed_transactions"] == checks["rtl_total_transactions"]
        and checks["rtl_total_transactions"]
        and checks["tv_outputs_present"]
    ):
        status = "completed-tv-output-mismatch"
    elif checks["entered_kernel"] and checks["rtl_progress_banner_seen"] and checks["timed_out"]:
        status = "direct-kernel-timeout"
    elif checks["entered_kernel"]:
        status = "direct-kernel-started"
    else:
        status = "fail"
    return {"status": status, "checks": checks}


def build_probe(
    xilinx_root: Path,
    profile: str,
    timeout_s: int,
    search_tail: int,
    track_tail: int,
    compare_tv_only: bool,
) -> dict[str, Any]:
    sim_root = sim_root_for_profile(profile)
    xsimk_path = xsimk_for_sim_root(sim_root)
    if compare_tv_only:
        result = {
            "cmd": None,
            "returncode": None,
            "stdin": "",
            "stdout": "compare-tv-only",
            "timed_out": False,
            "elapsed_s": 0.0,
        }
    elif not xsimk_path.exists():
        result: dict[str, Any] = {
            "cmd": [str(xsimk_path)],
            "returncode": None,
            "stdin": "",
            "stdout": "xsimk not found",
            "timed_out": False,
        }
    else:
        result = run_xsimk_mi_sequence([str(xsimk_path)], sim_root, vivado_env(xilinx_root), timeout_s)
    tv_comparison = compare_tv_outputs(sim_root)
    classification = classify_probe(result, xsimk_path, search_tail, track_tail, tv_comparison)
    if compare_tv_only:
        classification["status"] = "pass-tv-output-match" if tv_comparison["all_equal"] else "tv-output-mismatch"
    stdout = result.get("stdout", "")
    return {
        "status": classification["status"],
        "profile": profile,
        "timeout_s": timeout_s,
        "expected": {
            "search_tail": search_tail,
            "track_tail": track_tail,
        },
        "checks": classification["checks"],
        "tv_comparison": tv_comparison,
        "command": result.get("cmd"),
        "returncode": result.get("returncode"),
        "stdin": result.get("stdin"),
        "elapsed_s": result.get("elapsed_s"),
        "stdout_tail": stdout[-8000:],
        "sources": {
            "sim_root": str(sim_root),
            "xsimk": str(xsimk_path),
            "xsim_script": str(sim_root / "xsim.dir" / "hgtxr_e2e_axis_top" / "xsim_script.tcl"),
            "xsim_log": str(sim_root / "xsim.log"),
            "run_xsim_log": str(sim_root / "run_xsim.log"),
        },
        "interpretation": (
            "Direct xsimk MI execution completed the RTL testbench with expected Search and Track outputs."
            if classification["status"] == "pass"
            else "Direct xsimk MI execution completed all RTL transactions and the RTL TV outputs match the C reference outputs."
            if classification["status"] == "pass-tv-output-match"
            else "Direct xsimk MI execution completed all RTL transactions, but RTL TV outputs do not match the C reference outputs."
            if classification["status"] == "completed-tv-output-mismatch"
            else "Existing RTL TV outputs do not match the C reference outputs."
            if classification["status"] == "tv-output-mismatch"
            else "Direct xsimk MI execution entered the HGTXR RTL snapshot and started simulation progress, "
            "but the full two-transaction HLS cosim did not finish within the timeout."
            if classification["status"] == "direct-kernel-timeout"
            else "Direct xsimk MI execution entered the snapshot, but did not provide enough evidence for RTL output signoff."
            if classification["status"] == "direct-kernel-started"
            else "Direct xsimk MI execution did not reach a usable RTL snapshot."
        ),
    }


def render_markdown(probe: dict[str, Any]) -> str:
    lines = [
        "# Prefetch-All4 300MHz HLS Direct XSIMK Probe",
        "",
        f"- status: `{probe['status']}`",
        f"- profile: `{probe['profile']}`",
        f"- timeout: `{probe['timeout_s']} s`",
        f"- elapsed: `{probe.get('elapsed_s')} s`",
        f"- expected search tail: `{probe['expected']['search_tail']}`",
        f"- expected track tail: `{probe['expected']['track_tail']}`",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "|---|---:|",
    ]
    for name, value in sorted(probe["checks"].items()):
        lines.append(f"| `{name}` | `{value}` |")
    lines.extend(["", "## TV Output Comparison", "", "| Port | Present | Match | First mismatch |", "|---|---:|---:|---|"])
    for entry in probe["tv_comparison"]["ports"]:
        mismatch = entry.get("first_mismatch")
        if mismatch:
            first = f"line {mismatch['line']}: C `{mismatch['c']}` vs RTL `{mismatch['rtl']}`"
        else:
            first = ""
        present = bool(entry.get("c_exists")) and bool(entry.get("rtl_exists"))
        lines.append(f"| `{entry['port']}` | `{present}` | `{entry.get('equal')}` | {first} |")
    lines.extend(["", "## Interpretation", "", probe["interpretation"], "", "## Sources", ""])
    for name, path in sorted(probe["sources"].items()):
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Stdout Tail", "", "```text", probe["stdout_tail"], "```"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct xsimk MI probe for the HGTXR HLS RTL snapshot.")
    parser.add_argument("--xilinx-root", type=Path, default=DEFAULT_XILINX_ROOT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--search-tail", type=int, default=-1125)
    parser.add_argument("--track-tail", type=int, default=-239)
    parser.add_argument("--compare-tv-only", action="store_true", help="Skip xsimk execution and compare existing C/RTL TV output files.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_out = args.json_out or default_output_path(args.profile, "hls_xsimk_direct_probe_2026_06_29.json", DEFAULT_JSON)
    markdown_out = args.markdown_out or default_output_path(args.profile, "hls_xsimk_direct_probe_2026_06_29.md", DEFAULT_MARKDOWN)
    probe = build_probe(
        args.xilinx_root.resolve(),
        args.profile,
        args.timeout_s,
        args.search_tail,
        args.track_tail,
        args.compare_tv_only,
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(probe))
    print(json.dumps(probe, indent=2, sort_keys=True))
    return 0 if probe["status"] in {"pass", "pass-tv-output-match", "direct-kernel-timeout", "direct-kernel-started"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
