#!/usr/bin/env python3
"""Write the C3b baseline protection checklist for third-goal VREF work."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from validate_pynq_smoke_result import validate_result


DATE_TAG = "2026_06_16"
HGTXR_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ID = "C3b"
SMOKE_PRESET = "axis-c3b-mem16"


def resolve_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware" and (root / "hls").exists():
        return root.parent, root
    return root, root / "hardware"


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def rows_by_id(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = matrix.get("rows", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("id")): row for row in rows if isinstance(row, dict) and row.get("id")}


def nested(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def number(row: dict[str, Any], *keys: str) -> float | None:
    value = nested(row, *keys)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def integer(row: dict[str, Any], *keys: str) -> int | None:
    value = nested(row, *keys)
    if isinstance(value, int):
        return value
    return None


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    detail: str,
    *,
    category: str,
    threshold: str | None = None,
    path: Path | None = None,
) -> None:
    row: dict[str, Any] = {
        "name": name,
        "status": status,
        "category": category,
        "detail": detail,
    }
    if threshold is not None:
        row["threshold"] = threshold
    if path is not None:
        row["path"] = str(path)
    checks.append(row)


def pass_fail(value: bool) -> str:
    return "pass" if value else "fail"


def pending_if_missing(path: Path, present_detail: str, missing_detail: str) -> tuple[str, str]:
    return ("pass", present_detail) if path.exists() else ("pending", missing_detail)


def parse_gate(text: str, key: str) -> float | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def existing_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    return Path(path_text)


def hwh_has_axis_top(hwh_path: Path) -> bool:
    text = read_text(hwh_path)
    return "hgtxr_e2e_axis_top_0" in text or "hgtxr_e2e_axis_top" in text


def validate_smoke(smoke_path: Path) -> tuple[str, str]:
    if not smoke_path.exists():
        return "pending", "missing physical board smoke JSON"
    try:
        payload = load_json(smoke_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return "fail", f"invalid smoke JSON: {exc}"
    errors = validate_result(payload, SMOKE_PRESET, require_paths=True)
    if errors:
        return "fail", "; ".join(errors)
    return "pass", f"validated with preset {SMOKE_PRESET}"


def validate_smoke_session(session_path: Path, smoke_path: Path) -> tuple[str, str]:
    if not session_path.exists():
        return "pending", "missing smoke session metadata"
    try:
        payload = load_json(session_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return "fail", f"invalid session JSON: {exc}"

    errors: list[str] = []
    expected = {
        "status": "pass",
        "target": "ZCU104 PYNQ",
        "preset": SMOKE_PRESET,
        "variant": "c3b-mem16",
        "canonical_result_path": str(smoke_path),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"{key}={payload.get(key)!r}, expected {value!r}")
    if errors:
        return "pending", "; ".join(errors)
    return "pass", "session metadata matches canonical C3b smoke capture"


def build_checklist(
    root: Path,
    matrix_path: Path | None = None,
    sweep_path: Path | None = None,
) -> dict[str, Any]:
    hgtxr, hardware = resolve_roots(root)
    matrix_path = matrix_path or hardware / "generated" / "signoff" / "e2e_resource_matrix_2026_06_10.json"
    sweep_path = sweep_path or hardware / "configs" / "sweeps" / "zcu104_cyclic_transformer_sweep.yaml"
    smoke_path = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
    session_path = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
    xr_vits_path = hgtxr.parent.parent / "XR-VITs"
    xr_vits_policy_path = hardware / "docs" / "resources" / "xr_vits_replacement_policy.json"

    matrix = load_json(matrix_path)
    sweep_text = read_text(sweep_path)
    rows = rows_by_id(matrix)
    c3b = rows.get(BASELINE_ID, {})
    c1 = rows.get("C1", {})
    a1 = rows.get("A1", {})

    c3b_latency = integer(c3b, "hls", "latency_cycles")
    c3b_wns = number(c3b, "vivado", "timing", "wns_ns")
    c3b_target_clock = number(c3b, "hls", "target_clock_ns")
    c3b_est_clock = number(c3b, "hls", "estimated_clock_ns")
    c3b_resources = nested(c3b, "hls", "resources") or {}
    c3b_util = nested(c3b, "hls", "utilization_pct") or {}
    c3b_sources = nested(c3b, "sources") or {}
    c3b_bit = existing_path(nested(c3b, "artifacts", "bit", "path"))
    c3b_hwh = existing_path(nested(c3b, "artifacts", "hwh", "path"))

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "matrix_policy_no_new_hls_or_vivado_run",
        pass_fail(nested(matrix, "policy", "no_new_hls_or_vivado_run") is True),
        str(nested(matrix, "policy", "no_new_hls_or_vivado_run")),
        category="baseline-preservation",
        threshold="true",
        path=matrix_path,
    )
    add_check(
        checks,
        "matrix_recommends_c3b",
        pass_fail(nested(matrix, "summary", "recommended_board_smoke_variant") == BASELINE_ID),
        str(nested(matrix, "summary", "recommended_board_smoke_variant")),
        category="baseline-preservation",
        threshold=BASELINE_ID,
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_row_present",
        pass_fail(bool(c3b)),
        "C3b row found" if c3b else "C3b row missing",
        category="baseline-preservation",
        threshold="row id C3b",
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_ready_for_board_smoke",
        pass_fail(c3b.get("board_status") == "ready-for-board-smoke"),
        str(c3b.get("board_status")),
        category="baseline-preservation",
        threshold="ready-for-board-smoke",
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_parallelism_16",
        pass_fail(c3b.get("parallelism") == 16),
        str(c3b.get("parallelism")),
        category="architecture",
        threshold="16",
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_memory_banks_16",
        pass_fail(c3b.get("memory_banks") == 16),
        str(c3b.get("memory_banks")),
        category="architecture",
        threshold="16",
        path=matrix_path,
    )

    for label, path in [
        ("c3b_hls_report_exists", existing_path(str(c3b_sources.get("hls_report", "")))),
        ("c3b_timing_report_exists", existing_path(str(c3b_sources.get("timing_report", "")))),
        ("c3b_power_report_exists", existing_path(str(c3b_sources.get("power_report", "")))),
        ("c3b_bit_exists", c3b_bit),
        ("c3b_hwh_exists", c3b_hwh),
    ]:
        ok = path is not None and path.exists()
        add_check(
            checks,
            label,
            pass_fail(ok),
            "exists" if ok else "missing",
            category="artifact",
            threshold="file exists",
            path=path,
        )
    if c3b_hwh is not None:
        add_check(
            checks,
            "c3b_hwh_axis_top_marker",
            pass_fail(c3b_hwh.exists() and hwh_has_axis_top(c3b_hwh)),
            "hgtxr_e2e_axis_top marker present" if c3b_hwh.exists() and hwh_has_axis_top(c3b_hwh) else "marker missing",
            category="artifact",
            threshold="hgtxr_e2e_axis_top",
            path=c3b_hwh,
        )

    add_check(
        checks,
        "c3b_wns_nonnegative",
        pass_fail(c3b_wns is not None and c3b_wns >= 0.0),
        str(c3b_wns),
        category="timing",
        threshold=">= 0.0 ns",
        path=matrix_path,
    )
    add_check(
        checks,
        "successor_wns_floor_is_current_c3b",
        pass_fail(c3b_wns is not None and c3b_wns >= 4.415),
        str(c3b_wns),
        category="successor-gate",
        threshold="successor must be >= 4.415 ns unless explicitly waived",
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_estimated_clock_within_target",
        pass_fail(c3b_est_clock is not None and c3b_target_clock is not None and c3b_est_clock <= c3b_target_clock),
        f"estimated={c3b_est_clock} target={c3b_target_clock}",
        category="timing",
        threshold="estimated_clock_ns <= target_clock_ns",
        path=matrix_path,
    )
    for name, threshold_key in [
        ("lut", "max_total_lut_pct"),
        ("ff", "max_total_ff_pct"),
        ("bram_18k", "max_total_bram_pct"),
        ("dsp", "max_total_dsp_pct"),
    ]:
        observed = c3b_util.get(name)
        gate = parse_gate(sweep_text, threshold_key)
        ok = isinstance(observed, (int, float)) and gate is not None and float(observed) <= gate
        add_check(
            checks,
            f"c3b_{name}_util_within_global_gate",
            pass_fail(ok),
            f"observed={observed} gate={gate}",
            category="resource",
            threshold=f"{threshold_key}",
            path=sweep_path,
        )
    add_check(
        checks,
        "c3b_uram_within_device_capacity",
        pass_fail(
            integer(c3b, "hls", "resources", "uram") is not None
            and integer(c3b, "hls", "available", "uram") is not None
            and integer(c3b, "hls", "resources", "uram") <= integer(c3b, "hls", "available", "uram")
        ),
        f"observed={integer(c3b, 'hls', 'resources', 'uram')} available={integer(c3b, 'hls', 'available', 'uram')}",
        category="resource",
        threshold="URAM <= available",
        path=matrix_path,
    )
    for resource_name, threshold in [("dsp", 604), ("lut", 126506), ("uram", 64)]:
        observed = c3b_resources.get(resource_name)
        ok = isinstance(observed, int) and observed <= threshold
        add_check(
            checks,
            f"successor_{resource_name}_ceiling_is_current_c3b",
            pass_fail(ok),
            f"observed={observed} ceiling={threshold}",
            category="successor-gate",
            threshold=f"successor must be <= {threshold}",
            path=matrix_path,
        )
    add_check(
        checks,
        "successor_latency_ceiling_is_current_c3b",
        pass_fail(c3b_latency is not None and c3b_latency <= 37_508_072),
        str(c3b_latency),
        category="successor-gate",
        threshold="successor must be <= 37508072 cycles",
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_latency_not_worse_than_c1",
        pass_fail(c3b_latency is not None and integer(c1, "hls", "latency_cycles") is not None and c3b_latency <= integer(c1, "hls", "latency_cycles")),
        f"C3b={c3b_latency} C1={integer(c1, 'hls', 'latency_cycles')}",
        category="resource",
        threshold="C3b <= C1",
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_latency_better_than_a1",
        pass_fail(c3b_latency is not None and integer(a1, "hls", "latency_cycles") is not None and c3b_latency < integer(a1, "hls", "latency_cycles")),
        f"C3b={c3b_latency} A1={integer(a1, 'hls', 'latency_cycles')}",
        category="resource",
        threshold="C3b < A1",
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_lut_lower_than_c1",
        pass_fail(integer(c3b, "hls", "resources", "lut") is not None and integer(c1, "hls", "resources", "lut") is not None and integer(c3b, "hls", "resources", "lut") < integer(c1, "hls", "resources", "lut")),
        f"C3b={integer(c3b, 'hls', 'resources', 'lut')} C1={integer(c1, 'hls', 'resources', 'lut')}",
        category="resource",
        threshold="C3b < C1",
        path=matrix_path,
    )
    add_check(
        checks,
        "c3b_dsp_higher_than_a1",
        pass_fail(integer(c3b, "hls", "resources", "dsp") is not None and integer(a1, "hls", "resources", "dsp") is not None and integer(c3b, "hls", "resources", "dsp") > integer(a1, "hls", "resources", "dsp")),
        f"C3b={integer(c3b, 'hls', 'resources', 'dsp')} A1={integer(a1, 'hls', 'resources', 'dsp')}",
        category="resource",
        threshold="C3b > A1",
        path=matrix_path,
    )
    add_check(
        checks,
        "vref_entries_preserve_c3b_signoff_path",
        pass_fail(sweep_text.count("preserves_c3b_signoff_path: true") >= 8),
        str(sweep_text.count("preserves_c3b_signoff_path: true")),
        category="baseline-preservation",
        threshold=">= 8",
        path=sweep_path,
    )

    smoke_status, smoke_detail = validate_smoke(smoke_path)
    add_check(
        checks,
        "c3b_physical_smoke_json_validated",
        smoke_status,
        smoke_detail,
        category="board-smoke",
        threshold=f"validate_pynq_smoke_result preset {SMOKE_PRESET}",
        path=smoke_path,
    )
    session_status, session_detail = validate_smoke_session(session_path, smoke_path)
    add_check(
        checks,
        "c3b_smoke_session_metadata_validated",
        session_status,
        session_detail,
        category="board-smoke",
        threshold="status/target/preset/variant/canonical_result_path match",
        path=session_path,
    )
    xr_status, xr_detail = pending_if_missing(
        xr_vits_path,
        "requested XR-VITs sibling exists",
        "requested XR-VITs sibling missing; replacement policy required",
    )
    if xr_status == "pending" and xr_vits_policy_path.exists():
        xr_status = "pass"
        xr_detail = "replacement policy file exists; final preflight will validate approval fields"
    add_check(
        checks,
        "xr_vits_reference_or_replacement_policy",
        xr_status,
        xr_detail,
        category="external-reference",
        threshold="XR-VITs sibling or approved replacement policy",
        path=xr_vits_path if xr_vits_path.exists() else xr_vits_policy_path,
    )

    pass_count = sum(1 for check in checks if check["status"] == "pass")
    fail_count = sum(1 for check in checks if check["status"] == "fail")
    pending_count = sum(1 for check in checks if check["status"] == "pending")
    status = "fail" if fail_count else ("ready-for-final-unblock" if pending_count == 0 else "ready-for-board-smoke")

    return {
        "status": status,
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware": str(hardware),
        "baseline": {
            "id": BASELINE_ID,
            "latency_cycles": c3b_latency,
            "wns_ns": c3b_wns,
            "resources": {
                "dsp": integer(c3b, "hls", "resources", "dsp"),
                "lut": integer(c3b, "hls", "resources", "lut"),
                "ff": integer(c3b, "hls", "resources", "ff"),
                "bram_18k": integer(c3b, "hls", "resources", "bram_18k"),
                "uram": integer(c3b, "hls", "resources", "uram"),
            },
            "board_status": c3b.get("board_status"),
        },
        "successor_gate": {
            "no_overwrite_existing_c3b": True,
            "max_latency_cycles": 37_508_072,
            "min_wns_ns": 4.415,
            "max_dsp": 604,
            "max_lut": 126_506,
            "max_uram": 64,
            "required_bit_hwh": ["hgtxr_e2e_axis_dma_c3b_mem16.bit", "hgtxr_e2e_axis_dma_c3b_mem16.hwh"],
            "required_smoke_result": str(smoke_path),
        },
        "source_files": {
            "resource_matrix": str(matrix_path),
            "sweep_config": str(sweep_path),
            "smoke_json": str(smoke_path),
            "smoke_session": str(session_path),
            "xr_vits_requested": str(xr_vits_path),
            "xr_vits_replacement_policy": str(xr_vits_policy_path),
        },
        "checks": checks,
        "check_count": len(checks),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pending_count": pending_count,
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "writes_hls_source": False,
            "overwrites_c3b_artifacts": False,
        },
    }


def render_markdown(checklist: dict[str, Any]) -> str:
    lines = [
        "# C3b Baseline Protection Checklist",
        "",
        f"- status: `{checklist['status']}`",
        f"- root: `{checklist['root']}`",
        f"- checks: `{checklist['pass_count']}/{checklist['check_count']} pass`",
        f"- fail_count: `{checklist['fail_count']}`",
        f"- pending_count: `{checklist['pending_count']}`",
        "",
        "## Baseline",
        "",
    ]
    for key, value in checklist["baseline"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Successor Gate", ""])
    for key, value in checklist["successor_gate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", "", "| Check | Status | Category | Threshold | Detail |", "|---|---|---|---|---|"])
    for check in checklist["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        threshold = str(check.get("threshold", "")).replace("|", "\\|")
        if len(detail) > 180:
            detail = detail[:177] + "..."
        lines.append(f"| {check['name']} | `{check['status']}` | {check['category']} | {threshold} | {detail} |")
    lines.extend(["", "## Safety", ""])
    for key, value in checklist["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write C3b baseline protection checklist.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--resource-matrix", type=Path, default=None)
    parser.add_argument("--sweep-config", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checklist = build_checklist(args.root, args.resource_matrix, args.sweep_config)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(checklist, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(checklist))
    print(json.dumps({"status": checklist["status"], "pass_count": checklist["pass_count"], "pending_count": checklist["pending_count"], "fail_count": checklist["fail_count"]}, sort_keys=True))
    return 0 if checklist["fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
