#!/usr/bin/env python3
"""Audit ME-ViT-style single-load/on-chip buffer policy for VREF-P0-02."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"
HGTXR_ROOT = Path(__file__).resolve().parents[2]

LARGE_BUFFERS = [
    "tokens",
    "gb.tokens",
    "gb.norm",
    "gb.q",
    "gb.k",
    "gb.v",
    "gb.attn",
    "gb.hidden",
]

SMALL_BUFFERS = [
    "gb.pooled",
    "score",
    "prob",
    "exp_raw",
]


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def macro_values(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for match in re.finditer(r"^\s*#define\s+([A-Za-z0-9_]+)\s+([0-9]+)\b", text, re.MULTILINE):
        values[match.group(1)] = int(match.group(2))
    return values


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def line_refs(text: str, pattern: str, path: Path) -> list[str]:
    refs: list[str] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            refs.append(f"{path}:{idx}")
    return refs


def bind_refs(text: str, variable: str, impl: str, path: Path) -> list[str]:
    pattern = f"bind_storage variable={variable}"
    return [
        ref
        for ref in line_refs(text, pattern, path)
        if f"impl={impl}" in read_line(path, int(ref.rsplit(":", 1)[1]))
    ]


def read_line(path: Path, line_no: int) -> str:
    try:
        return path.read_text(errors="ignore").splitlines()[line_no - 1]
    except (OSError, IndexError):
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


def resource(row: dict[str, Any], key: str) -> int | None:
    hls = row.get("hls", {}) if isinstance(row.get("hls"), dict) else {}
    resources = hls.get("resources", {}) if isinstance(hls.get("resources"), dict) else {}
    value = resources.get(key)
    return int(value) if isinstance(value, int) else None


def latency(row: dict[str, Any]) -> int | None:
    hls = row.get("hls", {}) if isinstance(row.get("hls"), dict) else {}
    value = hls.get("latency_cycles")
    return int(value) if isinstance(value, int) else None


def status_at(payload: dict[str, Any], *keys: str) -> str:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return "missing"
        current = current.get(key)
    return str(current) if current is not None else "missing"


def named_check_status(payload: dict[str, Any], name: str, section: str | None = None) -> str:
    checks_owner: Any = payload.get(section, {}) if section else payload
    checks = checks_owner.get("checks", []) if isinstance(checks_owner, dict) else []
    if not isinstance(checks, list):
        return "missing"
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return str(check.get("status", "missing"))
    return "missing"


def build_audit(root: Path, matrix_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    hardware = root / "hardware"
    source_path = hardware / "hls" / "include" / "hgtxr_e2e_vit.hpp"
    axis_path = hardware / "hls" / "src" / "hgtxr_e2e_axis_top.cpp"
    maxis_path = hardware / "hls" / "src" / "hgtxr_e2e_m_axi_top.cpp"
    rmu_smu_path = hardware / "hls" / "src" / "rmu_smu.cpp"
    config_path = hardware / "configs" / "zcu104_e2e_q4w8a_defines.h"
    matrix_path = matrix_path or hardware / "generated" / "signoff" / "e2e_resource_matrix_2026_06_10.json"
    qkv_successor_path = hardware / "generated" / "signoff" / f"vref_p0_qkv_uram_cache_successor_{DATE_TAG}.json"

    source = read_text(source_path)
    axis = read_text(axis_path)
    maxis = read_text(maxis_path)
    rmu_smu = read_text(rmu_smu_path)
    config = read_text(config_path)
    matrix = load_json(matrix_path)
    qkv_successor = load_json(qkv_successor_path) if qkv_successor_path.exists() else {}
    rows = rows_by_id(matrix)
    c3b = rows.get("C3b", {})
    c1 = rows.get("C1", {})
    a1 = rows.get("A1", {})
    macros = macro_values(source)
    macros.update(macro_values(config))

    checks: list[dict[str, Any]] = []
    large_axis_refs = {name: bind_refs(axis, name, "uram", axis_path) for name in LARGE_BUFFERS}
    large_maxis_refs = {name: bind_refs(maxis, name, "uram", maxis_path) for name in LARGE_BUFFERS}
    small_axis_refs = {"gb.pooled": bind_refs(axis, "gb.pooled", "lutram", axis_path)}
    small_maxis_refs = {"gb.pooled": bind_refs(maxis, "gb.pooled", "lutram", maxis_path)}
    small_attention_refs = {
        name: bind_refs(source, name, "lutram", source_path)
        for name in ["score", "prob", "exp_raw"]
    }
    rmu_smu_small_refs = {
        name: bind_refs(rmu_smu, name, "lutram", rmu_smu_path)
        for name in ["score", "prob"]
    }
    qkv_bram_refs = {
        name: bind_refs(source, name, "bram", source_path)
        for name in ["q_weight_cache", "k_weight_cache", "v_weight_cache"]
    }
    qkv_uram_refs = {
        name: bind_refs(source, name, "uram", source_path)
        for name in ["q_weight_cache", "k_weight_cache", "v_weight_cache"]
    }
    partitions_axis = line_refs(axis, "ARRAY_PARTITION variable=gb.", axis_path)
    partitions_maxis = line_refs(maxis, "ARRAY_PARTITION variable=gb.", maxis_path)
    qkv_csynth = qkv_successor.get("csynth", {}) if isinstance(qkv_successor.get("csynth"), dict) else {}
    qkv_resources = qkv_csynth.get("resources", {}) if isinstance(qkv_csynth.get("resources"), dict) else {}
    qkv_comparison = qkv_successor.get("comparison", {}) if isinstance(qkv_successor.get("comparison"), dict) else {}
    qkv_delta = (
        qkv_comparison.get("delta_vs_dsp_mixed_stream", {})
        if isinstance(qkv_comparison.get("delta_vs_dsp_mixed_stream"), dict)
        else {}
    )
    qkv_promotion = qkv_successor.get("promotion", {}) if isinstance(qkv_successor.get("promotion"), dict) else {}
    qkv_remaining = qkv_promotion.get("remaining", []) if isinstance(qkv_promotion.get("remaining"), list) else []

    add_check(checks, "force_uram_buffers_enabled", macros.get("HGTXR_E2E_FORCE_URAM_BUFFERS") == 1, str(macros.get("HGTXR_E2E_FORCE_URAM_BUFFERS")))
    add_check(checks, "small_mem_lutram_enabled", macros.get("HGTXR_E2E_SMALL_MEM_LUTRAM") == 1, str(macros.get("HGTXR_E2E_SMALL_MEM_LUTRAM")))
    add_check(checks, "qkv_weight_cache_enabled", macros.get("HGTXR_E2E_QKV_WEIGHT_CACHE") == 1, str(macros.get("HGTXR_E2E_QKV_WEIGHT_CACHE")))
    add_check(checks, "qkv_weight_cache_not_forced_to_uram", macros.get("HGTXR_E2E_URAM_QKV_WEIGHT_CACHE") == 0, str(macros.get("HGTXR_E2E_URAM_QKV_WEIGHT_CACHE")))
    for name in LARGE_BUFFERS:
        add_check(checks, f"axis_large_{name}_uram", bool(large_axis_refs[name]), ", ".join(large_axis_refs[name]))
        add_check(checks, f"maxis_large_{name}_uram", bool(large_maxis_refs[name]), ", ".join(large_maxis_refs[name]))
    add_check(checks, "axis_pooled_lutram", bool(small_axis_refs["gb.pooled"]), ", ".join(small_axis_refs["gb.pooled"]))
    add_check(checks, "maxis_pooled_lutram", bool(small_maxis_refs["gb.pooled"]), ", ".join(small_maxis_refs["gb.pooled"]))
    for name in ["score", "prob", "exp_raw"]:
        add_check(checks, f"attention_small_{name}_lutram", bool(small_attention_refs[name]), ", ".join(small_attention_refs[name]))
    for name in ["score", "prob"]:
        add_check(checks, f"rmu_smu_small_{name}_lutram", bool(rmu_smu_small_refs[name]), ", ".join(rmu_smu_small_refs[name]))
    for name in ["q_weight_cache", "k_weight_cache", "v_weight_cache"]:
        add_check(checks, f"qkv_cache_{name}_bram_default", bool(qkv_bram_refs[name]), ", ".join(qkv_bram_refs[name]))
    for name in ["q_weight_cache", "k_weight_cache", "v_weight_cache"]:
        add_check(checks, f"qkv_cache_{name}_uram_successor_branch", bool(qkv_uram_refs[name]), ", ".join(qkv_uram_refs[name]))
    add_check(checks, "qkv_successor_file_present", bool(qkv_successor), str(qkv_successor_path))
    add_check(checks, "qkv_successor_status_pass", qkv_successor.get("status") == "pass", str(qkv_successor.get("status", "missing")))
    add_check(checks, "qkv_successor_macro_uram_enabled", qkv_successor.get("macro") == "HGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1", str(qkv_successor.get("macro", "missing")))
    add_check(checks, "qkv_successor_csim_pass", status_at(qkv_successor, "csim", "status") == "pass", status_at(qkv_successor, "csim", "status"))
    add_check(checks, "qkv_successor_csynth_pass", status_at(qkv_successor, "csynth", "status") == "pass", status_at(qkv_successor, "csynth", "status"))
    add_check(checks, "qkv_successor_overlay_routed", named_check_status(qkv_successor, "route_errors_zero", "overlay") == "pass", named_check_status(qkv_successor, "route_errors_zero", "overlay"))
    add_check(checks, "qkv_successor_uram_increased_vs_dsp_mixed_stream", isinstance(qkv_delta.get("uram"), int) and qkv_delta["uram"] > 0, str(qkv_delta.get("uram", "missing")))
    add_check(checks, "qkv_successor_bram_reduced_vs_dsp_mixed_stream", isinstance(qkv_delta.get("bram_18k"), int) and qkv_delta["bram_18k"] < 0, str(qkv_delta.get("bram_18k", "missing")))
    add_check(checks, "qkv_successor_uram_positive", isinstance(qkv_resources.get("uram"), int) and qkv_resources["uram"] > 0, str(qkv_resources.get("uram", "missing")))
    add_check(checks, "qkv_successor_physical_smoke_pending_only", qkv_promotion.get("status") == "hls-resource-pass-routed-pending" and "physical_smoke_json" in [str(item) for item in qkv_remaining], str(qkv_promotion))
    add_check(checks, "axis_mem_bank_partition_present", len(partitions_axis) >= 7, str(len(partitions_axis)))
    add_check(checks, "maxis_dense_partition_present", len(partitions_maxis) >= 7, str(len(partitions_maxis)))
    add_check(checks, "c3b_parallelism_16", c3b.get("parallelism") == 16, str(c3b.get("parallelism")))
    add_check(checks, "c3b_memory_banks_16", c3b.get("memory_banks") == 16, str(c3b.get("memory_banks")))
    add_check(checks, "c3b_uram_positive", resource(c3b, "uram") is not None and resource(c3b, "uram") > 0, str(resource(c3b, "uram")))
    add_check(checks, "c3b_dsp_increased_vs_a1", resource(c3b, "dsp") is not None and resource(a1, "dsp") is not None and resource(c3b, "dsp") > resource(a1, "dsp"), f"C3b={resource(c3b, 'dsp')} A1={resource(a1, 'dsp')}")
    add_check(checks, "c3b_lut_lower_than_c1", resource(c3b, "lut") is not None and resource(c1, "lut") is not None and resource(c3b, "lut") < resource(c1, "lut"), f"C3b={resource(c3b, 'lut')} C1={resource(c1, 'lut')}")
    add_check(checks, "c3b_latency_not_worse_than_c1", latency(c3b) is not None and latency(c1) is not None and latency(c3b) <= latency(c1), f"C3b={latency(c3b)} C1={latency(c1)}")

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "date_tag": DATE_TAG,
        "root": str(root),
        "source_files": {
            "hgtxr_e2e_vit": str(source_path),
            "hgtxr_e2e_axis_top": str(axis_path),
            "hgtxr_e2e_m_axi_top": str(maxis_path),
            "rmu_smu": str(rmu_smu_path),
            "zcu104_e2e_q4w8a_defines": str(config_path),
            "resource_matrix": str(matrix_path),
            "qkv_uram_successor": str(qkv_successor_path),
        },
        "policy": {
            "vref": "VREF-P0-02",
            "source": "ME-ViT",
            "scope": "single-load/on-chip retention and resource placement audit; no HLS source mutation",
            "large_buffers": "tokens, norm, Q/K/V, attention, hidden stay URAM candidates",
            "small_buffers": "pooled and attention-row scratch stay LUTRAM/BRAM candidates",
            "promotion_gate": "C3b path unchanged; csynth/routed evidence required for any successor variant",
        },
        "observed": {
            "macros": {key: macros.get(key) for key in ["HGTXR_E2E_FORCE_URAM_BUFFERS", "HGTXR_E2E_SMALL_MEM_LUTRAM", "HGTXR_E2E_QKV_WEIGHT_CACHE", "HGTXR_E2E_URAM_QKV_WEIGHT_CACHE", "HGTXR_E2E_DENSE_PAR", "HGTXR_PARALLELISM_FACTOR", "HGTXR_BUFFER_SIZE", "HGTXR_FIFO_DEPTH"]},
            "large_axis_refs": large_axis_refs,
            "large_maxis_refs": large_maxis_refs,
            "small_axis_refs": small_axis_refs,
            "small_maxis_refs": small_maxis_refs,
            "small_attention_refs": small_attention_refs,
            "rmu_smu_small_refs": rmu_smu_small_refs,
            "qkv_bram_refs": qkv_bram_refs,
            "qkv_uram_refs": qkv_uram_refs,
            "qkv_uram_successor": {
                "status": qkv_successor.get("status"),
                "macro": qkv_successor.get("macro"),
                "csim_status": status_at(qkv_successor, "csim", "status"),
                "csynth_status": status_at(qkv_successor, "csynth", "status"),
                "overlay_status": status_at(qkv_successor, "overlay", "status"),
                "resources": qkv_resources,
                "delta_vs_dsp_mixed_stream": qkv_delta,
                "promotion": qkv_promotion,
            },
            "partition_refs": {
                "axis": partitions_axis,
                "m_axi": partitions_maxis,
            },
            "c3b": {
                "parallelism": c3b.get("parallelism"),
                "memory_banks": c3b.get("memory_banks"),
                "dsp": resource(c3b, "dsp"),
                "lut": resource(c3b, "lut"),
                "uram": resource(c3b, "uram"),
                "latency_cycles": latency(c3b),
            },
        },
        "checks": checks,
        "check_count": len(checks),
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "writes_hls_source": False,
            "overwrites_c3b_artifacts": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# VREF-P0-02 Buffer Lifetime And Placement Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- root: `{audit['root']}`",
        f"- checks: `{audit['pass_count']}/{audit['check_count']}`",
        f"- fail_count: `{audit['fail_count']}`",
        "",
        "## Policy",
        "",
    ]
    for key, value in audit["policy"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Observed C3b", ""])
    for key, value in audit["observed"]["c3b"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Macro Snapshot", "", "| Macro | Value |", "|---|---:|"])
    for key, value in audit["observed"]["macros"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        if len(detail) > 180:
            detail = detail[:177] + "..."
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(["", "## Safety", ""])
    for key, value in audit["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write VREF-P0-02 buffer lifetime and placement audit.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--resource-matrix", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root, args.resource_matrix)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audit))
    print(f"[vref-p0-buffer-lifetime-audit] status={audit['status']} checks={audit['pass_count']}/{audit['check_count']}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
