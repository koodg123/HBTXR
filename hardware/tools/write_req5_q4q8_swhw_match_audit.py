#!/usr/bin/env python3
"""Audit Req5 Q4/Q8 synthetic-weight SW/HW match evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"

CSIM_LOGS = [
    (
        "c3b_axis_csim",
        "generated/hgtxr_e2e_axis_par16_c3b_mem16_csim/solution_e2e_q4w8a/csim/report/hgtxr_e2e_axis_top_csim.log",
        "E2E AXIS vector comparison passed",
    ),
    (
        "vref_softmax_input_x2_csim",
        "generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_csim/solution_e2e_q4w8a/csim/report/hgtxr_e2e_axis_top_csim.log",
        "E2E AXIS vector comparison passed",
    ),
    (
        "qkv_uram_csim",
        "generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csim/solution_e2e_q4w8a/csim/report/hgtxr_e2e_axis_top_csim.log",
        "E2E AXIS vector comparison passed",
    ),
]


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def source_contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(errors="replace")


def parse_macro(text: str, name: str) -> int | None:
    match = re.search(rf"^\s*#\s*define\s+{re.escape(name)}\s+([0-9]+)\b", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_csim_log(hardware: Path, name: str, rel_path: str, pass_marker: str) -> dict[str, Any]:
    path = hardware / rel_path
    if not path.exists():
        return {"name": name, "path": rel_path, "status": "missing"}
    text = path.read_text(errors="replace")
    outputs = [int(match.group(2)) for match in re.finditer(r"e2e_axis_out\[(\d+)\]=(-?\d+)", text)]
    runtime_match = re.search(r"runtime_state=(\d+).*failures=(\d+)", text)
    runtime_state = int(runtime_match.group(1)) if runtime_match else None
    failures = int(runtime_match.group(2)) if runtime_match else None
    pass_status = (
        pass_marker in text
        and "CSim done with 0 errors" in text
        and "FAIL:" not in text
        and runtime_state == 2
        and failures == 0
        and len(outputs) == 6
    )
    return {
        "name": name,
        "path": rel_path,
        "status": "pass" if pass_status else "fail",
        "outputs": outputs,
        "runtime_state": runtime_state,
        "failures": failures,
        "has_pass_marker": pass_marker in text,
        "has_zero_errors": "CSim done with 0 errors" in text,
    }


def manifest_checks(hardware: Path) -> dict[str, Any]:
    rel_path = "refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path}
    payload = load_json(path)
    layout = payload.get("layout", {}) if isinstance(payload.get("layout"), dict) else {}
    known = payload.get("known_offset_checks", {})
    known_values = [int(value) for value in known.values()] if isinstance(known, dict) else []
    has_mixed_sign = any(value > 0 for value in known_values) and any(value < 0 for value in known_values)
    expected_raw = payload.get("expected_raw", [])
    expected_ok = isinstance(expected_raw, list) and len(expected_raw) == 6
    binary_value = payload.get("binary")
    binary_path = Path(binary_value) if isinstance(binary_value, str) and binary_value else None
    if binary_path is not None and not binary_path.is_absolute():
        binary_path = hardware.parent / binary_path if binary_value.startswith("hardware/") else hardware / binary_path
    binary_exists = binary_path is not None and binary_path.exists()
    binary_bytes = binary_path.stat().st_size if binary_exists else None
    binary_sha256 = hashlib.sha256(binary_path.read_bytes()).hexdigest() if binary_exists else None
    layout_ok = (
        payload.get("dtype") == "uint32"
        and layout.get("weight_bits") == 4
        and layout.get("bus_width_bits") == 256
        and int(payload.get("bytes", 0)) > 0
        and int(payload.get("required_u32", 0)) > 0
        and int(payload.get("tail_zero_start_u32", -1)) == int(payload.get("required_u32", -2))
    )
    sha_ok = binary_exists and binary_sha256 == payload.get("sha256")
    byte_count_ok = binary_exists and binary_bytes == payload.get("bytes")
    runtime_ok = payload.get("expected_runtime_state") == 2
    expected_c3b_ok = expected_raw == [32, -13, 26, -6, 14, -11]
    return {
        "status": "pass"
        if layout_ok and has_mixed_sign and expected_ok and sha_ok and byte_count_ok and runtime_ok and expected_c3b_ok
        else "fail",
        "path": rel_path,
        "dtype": payload.get("dtype"),
        "bytes": payload.get("bytes"),
        "sha256": payload.get("sha256"),
        "binary": str(binary_path) if binary_path is not None else None,
        "binary_exists": binary_exists,
        "binary_bytes": binary_bytes,
        "binary_sha256": binary_sha256,
        "binary_sha256_matches_manifest": sha_ok,
        "binary_bytes_matches_manifest": byte_count_ok,
        "layout": layout,
        "known_offset_checks": known,
        "expected_raw": expected_raw,
        "expected_runtime_state": payload.get("expected_runtime_state"),
        "expected_runtime_state_ok": runtime_ok,
        "expected_c3b_raw_ok": expected_c3b_ok,
        "has_mixed_sign_probe_weights": has_mixed_sign,
        "provenance": payload.get("provenance", {}),
    }


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    config_path = hardware / "configs" / "zcu104_e2e_q4w8a_defines.h"
    config_text = config_path.read_text(errors="replace") if config_path.exists() else ""
    weight_bits = parse_macro(config_text, "HGTXR_WEIGHT_BIT_WIDTH")
    activation_bits = parse_macro(config_text, "HGTXR_BIT_WIDTH")
    manifest = manifest_checks(hardware)
    axis_tb = hardware / "hls" / "tb" / "tb_hgtxr_e2e_axis_top.cpp"
    maxis_tb = hardware / "hls" / "tb" / "tb_hgtxr_e2e_m_axi_top.cpp"
    tb_checks = {
        "axis_strict_golden": source_contains(axis_tb, "HGTXR_E2E_STRICT_GOLDEN")
        and source_contains(axis_tb, "init_q4_vector_weights")
        and source_contains(axis_tb, "FAIL: e2e output"),
        "m_axi_strict_golden": source_contains(maxis_tb, "HGTXR_E2E_STRICT_GOLDEN")
        and source_contains(maxis_tb, "init_q4_vector_weights")
        and source_contains(maxis_tb, "FAIL: e2e_m_axi output"),
    }
    csim_logs = [parse_csim_log(hardware, name, rel_path, marker) for name, rel_path, marker in CSIM_LOGS]
    checks = [
        {
            "name": "config_q4w_q8a",
            "status": "pass" if weight_bits == 4 and activation_bits == 8 else "fail",
            "detail": f"weight_bits={weight_bits} activation_bits={activation_bits}",
        },
        {
            "name": "packed_weight_manifest",
            "status": manifest["status"],
            "detail": manifest["path"],
        },
        {
            "name": "packed_weight_binary_exists",
            "status": "pass" if manifest.get("binary_exists") is True else "fail",
            "detail": str(manifest.get("binary")),
        },
        {
            "name": "packed_weight_sha256_matches_manifest",
            "status": "pass" if manifest.get("binary_sha256_matches_manifest") is True else "fail",
            "detail": str(manifest.get("binary_sha256")),
        },
        {
            "name": "packed_weight_byte_count_matches_manifest",
            "status": "pass" if manifest.get("binary_bytes_matches_manifest") is True else "fail",
            "detail": f"binary={manifest.get('binary_bytes')} manifest={manifest.get('bytes')}",
        },
        {
            "name": "packed_weight_expected_runtime_state",
            "status": "pass" if manifest.get("expected_runtime_state_ok") is True else "fail",
            "detail": str(manifest.get("expected_runtime_state")),
        },
        {
            "name": "packed_weight_expected_c3b_output",
            "status": "pass" if manifest.get("expected_c3b_raw_ok") is True else "fail",
            "detail": str(manifest.get("expected_raw")),
        },
        {
            "name": "testbench_strict_golden_compare",
            "status": "pass" if all(tb_checks.values()) else "fail",
            "detail": str(tb_checks),
        },
    ]
    checks.extend(
        {
            "name": f"{log['name']}_strict_csim",
            "status": log["status"],
            "detail": f"runtime={log.get('runtime_state')} failures={log.get('failures')} outputs={log.get('outputs')}",
        }
        for log in csim_logs
    )
    failed = [check["name"] for check in checks if check["status"] != "pass"]
    return {
        "status": "pass" if not failed else "fail",
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "precision": {
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "weight_format": "signed-q4-packed-in-uint32/AXI-256",
            "activation_format": "q8/hgtxr_data_t boundary per config",
        },
        "manifest": manifest,
        "testbench_checks": tb_checks,
        "csim_logs": csim_logs,
        "checks": checks,
        "pass_count": len(checks) - len(failed),
        "fail_count": len(failed),
        "failed_checks": failed,
        "claim_supported": "Local strict CSim proves deterministic synthetic Q4 weight vectors and Q8 activation path match golden SW references for available E2E AXIS variants.",
        "remaining_scope": [
            "ZCU104 physical board smoke remains separate Req4/final gate.",
            "This is not exhaustive over every possible arbitrary weight tensor.",
        ],
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Req5 Q4/Q8 SW-HW Match Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- weight_bits: `{audit['precision']['weight_bits']}`",
        f"- activation_bits: `{audit['precision']['activation_bits']}`",
        f"- pass: `{audit['pass_count']}`",
        f"- fail: `{audit['fail_count']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(["", "## CSim Logs", "", "| Name | Status | Outputs |", "|---|---|---|"])
    for log in audit["csim_logs"]:
        lines.append(f"| {log['name']} | `{log['status']}` | `{log.get('outputs', [])}` |")
    lines.extend(["", "## Remaining Scope"])
    lines.extend(f"- {item}" for item in audit["remaining_scope"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Req5 Q4/Q8 SW-HW match audit.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _, hardware = normalize_roots(args.root)
    json_out = args.json_out or hardware / "generated" / "signoff" / f"req5_q4q8_swhw_match_audit_{DATE_TAG}.json"
    markdown_out = args.markdown_out or hardware / "generated" / "signoff" / f"req5_q4q8_swhw_match_audit_{DATE_TAG}.md"
    audit = build_audit(args.root)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))
    print(f"[req5-q4q8-swhw-match-audit] status={audit['status']} fail={audit['fail_count']}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
