#!/usr/bin/env python3
"""Audit HGTXR HLS AXIS payload RTL/TV contract for a generated profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
SOLUTION = "solution_e2e_q4w8a"
DEFAULT_PROFILE = (
    "par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_"
    "compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16"
)
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "hls_axis_payload_contract_2026_06_29.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "hls_axis_payload_contract_2026_06_29.md"


def project_root_for_profile(profile: str) -> Path:
    return HARDWARE_ROOT / "generated" / f"hgtxr_e2e_axis_{profile}_no_board" / SOLUTION


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_tv_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("[[transaction]]"):
        parts = stripped.split()
        if len(parts) >= 2:
            return f"[[transaction]] {parts[-1]}"
    return stripped


def read_tv_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [normalize_tv_line(line) for line in path.read_text(errors="replace").splitlines()]


def hex_payload_values(lines: list[str]) -> list[int]:
    values: list[int] = []
    for line in lines:
        if line.startswith("[[") or not line:
            continue
        match = re.fullmatch(r"0x([0-9A-Fa-f]+)", line)
        if match:
            values.append(int(match.group(1), 16))
    return values


def signed_low(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    low = value & mask
    sign = 1 << (bits - 1)
    return low - (1 << bits) if low & sign else low


def tv_payload_summary(tv_root: Path, payload_bits: int) -> dict[str, Any]:
    c_path = tv_root / "cdatafile" / "c.hgtxr_e2e_axis_top.autotvout_axis_out_V_data_V.dat"
    rtl_path = tv_root / "rtldatafile" / "rtl.hgtxr_e2e_axis_top.autotvout_axis_out_V_data_V.dat"
    c_lines = read_tv_lines(c_path)
    rtl_lines = read_tv_lines(rtl_path)
    c_values = hex_payload_values(c_lines)
    rtl_values = hex_payload_values(rtl_lines)
    c_low = [value & ((1 << payload_bits) - 1) for value in c_values]
    rtl_low = [value & ((1 << payload_bits) - 1) for value in rtl_values]
    first_mismatch = None
    for idx, (c_value, rtl_value) in enumerate(zip(c_low, rtl_low), start=1):
        if c_value != rtl_value:
            first_mismatch = {
                "payload_index": idx,
                "c_low_hex": f"0x{c_value:0{max(1, payload_bits // 4)}x}",
                "rtl_low_hex": f"0x{rtl_value:0{max(1, payload_bits // 4)}x}",
                "c_low_signed": signed_low(c_value, payload_bits),
                "rtl_low_signed": signed_low(rtl_value, payload_bits),
            }
            break
    if first_mismatch is None and len(c_low) != len(rtl_low):
        first_mismatch = {
            "payload_index": min(len(c_low), len(rtl_low)) + 1,
            "c_low_hex": "<missing>" if len(c_low) < len(rtl_low) else f"0x{c_low[-1]:x}",
            "rtl_low_hex": "<missing>" if len(rtl_low) < len(c_low) else f"0x{rtl_low[-1]:x}",
        }
    return {
        "c_path": str(c_path),
        "rtl_path": str(rtl_path),
        "c_exists": c_path.exists(),
        "rtl_exists": rtl_path.exists(),
        "c_sha256": sha256(c_path),
        "rtl_sha256": sha256(rtl_path),
        "c_line_count": len(c_lines),
        "rtl_line_count": len(rtl_lines),
        "c_payload_count": len(c_values),
        "rtl_payload_count": len(rtl_values),
        "full_width_equal": bool(c_lines and rtl_lines and c_lines == rtl_lines),
        "active_low_bits_equal": bool(c_low and rtl_low and c_low == rtl_low),
        "payload_bits": payload_bits,
        "c_active_low_signed": [signed_low(value, payload_bits) for value in c_low],
        "rtl_active_low_signed": [signed_low(value, payload_bits) for value in rtl_low],
        "first_active_mismatch": first_mismatch,
    }


def rtl_structural_summary(syn_root: Path) -> dict[str, Any]:
    top = syn_root / "hgtxr_e2e_axis_top.v"
    axis_writer_modules = sorted(syn_root.glob("*axis_write_state*.v"))
    text = top.read_text(errors="replace") if top.exists() else ""
    writer_text = "\n".join(path.read_text(errors="replace") for path in axis_writer_modules)
    combined = text + "\n" + writer_text
    state_widths = sorted(
        {
            int(match.group(1)) + 1
            for match in re.finditer(r"(?:output|reg|wire)\s+\[(\d+):0\]\s+out_state", combined)
        }
    )
    tdata_sources = sorted(
        set(re.findall(r"axis_out_TDATA(?:_int_regslice)?\s*=\s*([^;]+);", combined))
    )
    zero_assignments = [
        source for source in tdata_sources if re.fullmatch(r"(?:256')?d?0|256'd0", source.strip())
    ]
    has_out_state_source = any("out_state" in source or "zext_ln" in source for source in tdata_sources)
    return {
        "syn_top": str(top),
        "syn_top_exists": top.exists(),
        "axis_writer_modules": [str(path) for path in axis_writer_modules],
        "top_sha256": sha256(top),
        "axis_out_tdata_sources": tdata_sources,
        "axis_out_tdata_zero_assignment_count": len(zero_assignments),
        "axis_out_tdata_has_out_state_source": has_out_state_source,
        "out_state_widths": state_widths,
    }


def build_audit(profile: str, payload_bits: int) -> dict[str, Any]:
    project_root = project_root_for_profile(profile)
    syn_root = project_root / "syn" / "verilog"
    tv_root = project_root / "sim" / "tv"
    rtl = rtl_structural_summary(syn_root)
    tv = tv_payload_summary(tv_root, payload_bits)
    status = "missing-syn-verilog"
    if rtl["syn_top_exists"]:
        if rtl["axis_out_tdata_zero_assignment_count"] and not rtl["axis_out_tdata_has_out_state_source"]:
            status = "fail-constant-axis-tdata"
        elif tv["c_exists"] and tv["rtl_exists"] and tv["active_low_bits_equal"]:
            status = "pass-active-payload-tv"
        elif tv["c_exists"] and tv["rtl_exists"]:
            status = "fail-active-payload-tv"
        elif rtl["axis_out_tdata_has_out_state_source"]:
            status = "structural-pass-missing-rtl-tv"
        else:
            status = "inconclusive-rtl-structure"
    return {
        "status": status,
        "profile": profile,
        "project_root": str(project_root),
        "payload_bits": payload_bits,
        "rtl": rtl,
        "tv": tv,
        "interpretation": interpretation_for_status(status),
    }


def interpretation_for_status(status: str) -> str:
    return {
        "missing-syn-verilog": "No generated synthesis Verilog was found for this profile.",
        "fail-constant-axis-tdata": "Generated RTL still contains a constant-zero AXIS TDATA writer with no out_state source.",
        "pass-active-payload-tv": "C and RTL TV output data match in the configured active low payload bits.",
        "fail-active-payload-tv": "C and RTL TV output data are both present, but their active low payload bits differ.",
        "structural-pass-missing-rtl-tv": "Generated RTL structurally drives AXIS TDATA from out_state, but RTL TV output is missing.",
        "inconclusive-rtl-structure": "Generated RTL exists, but this tool could not prove an out_state AXIS TDATA source.",
    }[status]


def render_markdown(audit: dict[str, Any]) -> str:
    tv = audit["tv"]
    rtl = audit["rtl"]
    lines = [
        "# HLS AXIS Payload Contract Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- profile: `{audit['profile']}`",
        f"- payload bits checked: `{audit['payload_bits']}`",
        f"- interpretation: {audit['interpretation']}",
        "",
        "## RTL Structure",
        "",
        f"- syn top exists: `{rtl['syn_top_exists']}`",
        f"- out_state widths: `{rtl['out_state_widths']}`",
        f"- zero TDATA assignments: `{rtl['axis_out_tdata_zero_assignment_count']}`",
        f"- has out_state-like TDATA source: `{rtl['axis_out_tdata_has_out_state_source']}`",
        "",
        "## TV Payload",
        "",
        f"- C TV exists: `{tv['c_exists']}`",
        f"- RTL TV exists: `{tv['rtl_exists']}`",
        f"- full-width equal: `{tv['full_width_equal']}`",
        f"- active low-bit equal: `{tv['active_low_bits_equal']}`",
        f"- C active signed values: `{tv['c_active_low_signed']}`",
        f"- RTL active signed values: `{tv['rtl_active_low_signed']}`",
    ]
    mismatch = tv.get("first_active_mismatch")
    if mismatch:
        lines.extend(["", "## First Active Payload Mismatch", "", "```json", json.dumps(mismatch, indent=2), "```"])
    lines.extend(["", "## Sources", "", f"- project root: `{audit['project_root']}`", f"- syn top: `{rtl['syn_top']}`", f"- C TV: `{tv['c_path']}`", f"- RTL TV: `{tv['rtl_path']}`"])
    return "\n".join(lines) + "\n"


def default_output_path(profile: str, suffix: str, default_path: Path) -> Path:
    if profile == DEFAULT_PROFILE:
        return default_path
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile)
    return HARDWARE_ROOT / "generated" / "signoff" / f"{safe}_{suffix}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--payload-bits", type=int, default=8)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.profile, args.payload_bits)
    json_out = args.json_out or default_output_path(args.profile, "hls_axis_payload_contract_2026_06_29.json", DEFAULT_JSON)
    markdown_out = args.markdown_out or default_output_path(args.profile, "hls_axis_payload_contract_2026_06_29.md", DEFAULT_MARKDOWN)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] in {"pass-active-payload-tv", "structural-pass-missing-rtl-tv"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
