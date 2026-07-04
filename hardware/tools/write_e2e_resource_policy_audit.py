#!/usr/bin/env python3
"""Audit E2E DSP/URAM/LUTRAM/parallelism resource policy evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from write_e2e_resource_matrix import parse_hls_csynth_xml


DATE_TAG = "2026_06_10"
HGTXR_ROOT = Path(__file__).resolve().parents[2]
C3B_DSP_MAX = 604
C3B_LUT_MAX = 126_506
C3B_URAM_MAX = 64
C3B_LATENCY_MAX = 37_508_072
C3B_WNS_MIN = 4.415


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def macro_value(text: str, name: str) -> int | None:
    match = re.search(rf"#define\s+{re.escape(name)}\s+([0-9]+)\b", text)
    return int(match.group(1)) if match else None


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def rows_by_id(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = matrix.get("rows", [])
    result: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                result[str(row["id"])] = row
    return result


def resource(row: dict[str, Any], key: str) -> int | None:
    hls = row.get("hls", {}) if isinstance(row.get("hls"), dict) else {}
    resources = hls.get("resources", {}) if isinstance(hls.get("resources"), dict) else {}
    value = resources.get(key)
    return int(value) if isinstance(value, int) else None


def latency(row: dict[str, Any]) -> int | None:
    hls = row.get("hls", {}) if isinstance(row.get("hls"), dict) else {}
    value = hls.get("latency_cycles")
    return int(value) if isinstance(value, int) else None


def routed_wns(row: dict[str, Any]) -> float | None:
    vivado = row.get("vivado", {}) if isinstance(row.get("vivado"), dict) else {}
    timing = vivado.get("timing", {}) if isinstance(vivado.get("timing"), dict) else {}
    value = timing.get("wns_ns")
    return float(value) if isinstance(value, (int, float)) else None


def resolve_source_path(root: Path, row: dict[str, Any], key: str) -> Path | None:
    sources = row.get("sources", {}) if isinstance(row.get("sources"), dict) else {}
    value = sources.get(key)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def resource_values_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ["bram_18k", "dsp", "ff", "lut", "uram"]
    return all(left.get(key) == right.get(key) for key in keys)


def build_audit(root: Path, matrix_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    hardware = root / "hardware"
    source_path = hardware / "hls" / "include" / "hgtxr_e2e_vit.hpp"
    rmu_smu_path = hardware / "hls" / "src" / "rmu_smu.cpp"
    cyclic_top_path = hardware / "hls" / "src" / "hgtxr_top.cpp"
    cyclic_params_path = hardware / "hls" / "include" / "hgtxr_cyclic_transformer_params.hpp"
    config_path = hardware / "configs" / "zcu104_e2e_q4w8a_defines.h"
    matrix_path = matrix_path or root / "docs" / "resources" / f"e2e_resource_matrix_{DATE_TAG}.json"
    source = source_path.read_text()
    rmu_smu_source = rmu_smu_path.read_text()
    cyclic_top_source = cyclic_top_path.read_text()
    cyclic_params = cyclic_params_path.read_text()
    config = config_path.read_text()
    matrix = load_json(matrix_path)
    rows = rows_by_id(matrix)

    checks: list[dict[str, Any]] = []
    bind_op_dsp_count = source.count("impl=dsp")
    rmu_smu_bind_op_dsp_count = rmu_smu_source.count("impl=dsp")
    rmu_smu_dsp_helper_count = rmu_smu_source.count("rmu_smu_dsp_mul(")
    rmu_smu_dsp_acc_helper_count = rmu_smu_source.count("rmu_smu_dsp_mul_acc(")
    bind_storage_uram_count = source.count("impl=uram")
    bind_storage_lutram_count = source.count("impl=lutram")
    cyclic_weight_uram_count = sum(
        cyclic_top_source.count(f"variable={name} type=ram_2p impl=uram")
        for name in ["wq", "wk", "wv", "wo", "w1", "w2"]
    )
    cyclic_large_temp_uram_count = sum(
        cyclic_top_source.count(f"variable={name} type=ram_2p impl=uram")
        for name in ["attn_tiles", "hidden_tiles", "residual0", "residual1"]
    )
    cyclic_small_tile_lutram_count = cyclic_top_source.count("impl=lutram")
    c3b = rows.get("C3b", {})
    c1 = rows.get("C1", {})
    a1 = rows.get("A1", {})
    c3b_csynth_path = resolve_source_path(root, c3b, "hls_report")
    c3b_csynth: dict[str, Any] | None = None
    if c3b_csynth_path is not None and c3b_csynth_path.exists():
        c3b_csynth = parse_hls_csynth_xml(c3b_csynth_path)
    c3b_matrix_hls = c3b.get("hls", {}) if isinstance(c3b.get("hls"), dict) else {}
    c3b_matrix_resources = (
        c3b_matrix_hls.get("resources", {}) if isinstance(c3b_matrix_hls.get("resources"), dict) else {}
    )
    c3b_csynth_resources = (
        c3b_csynth.get("resources", {}) if isinstance(c3b_csynth, dict) and isinstance(c3b_csynth.get("resources"), dict) else {}
    )
    c3b_csynth_latency = c3b_csynth.get("latency_cycles") if isinstance(c3b_csynth, dict) else None

    add_check(checks, "force_dsp_macro_default", macro_value(source, "HGTXR_E2E_FORCE_DSP_MUL") == 1, str(macro_value(source, "HGTXR_E2E_FORCE_DSP_MUL")))
    add_check(checks, "force_uram_macro_default", macro_value(source, "HGTXR_E2E_FORCE_URAM_BUFFERS") == 1, str(macro_value(source, "HGTXR_E2E_FORCE_URAM_BUFFERS")))
    add_check(checks, "small_mem_lutram_macro_default", macro_value(source, "HGTXR_E2E_SMALL_MEM_LUTRAM") == 1, str(macro_value(source, "HGTXR_E2E_SMALL_MEM_LUTRAM")))
    add_check(checks, "cyclic_weight_tiles_uram_macro_default", macro_value(cyclic_params, "HGTXR_CYCLIC_FORCE_URAM_WEIGHT_TILES") == 1, str(macro_value(cyclic_params, "HGTXR_CYCLIC_FORCE_URAM_WEIGHT_TILES")))
    add_check(checks, "cyclic_large_temps_uram_macro_default", macro_value(cyclic_params, "HGTXR_CYCLIC_FORCE_URAM_LARGE_TEMPS") == 1, str(macro_value(cyclic_params, "HGTXR_CYCLIC_FORCE_URAM_LARGE_TEMPS")))
    add_check(checks, "cyclic_small_tile_lutram_macro_default", macro_value(cyclic_params, "HGTXR_CYCLIC_SMALL_TILE_LUTRAM") == 1, str(macro_value(cyclic_params, "HGTXR_CYCLIC_SMALL_TILE_LUTRAM")))
    add_check(checks, "dsp_bind_op_present", bind_op_dsp_count >= 2, str(bind_op_dsp_count))
    add_check(checks, "rmu_smu_dsp_bind_op_present", rmu_smu_bind_op_dsp_count >= 2, str(rmu_smu_bind_op_dsp_count))
    add_check(checks, "rmu_smu_dsp_helper_defined", "static hgtxr_acc_t rmu_smu_dsp_mul(" in rmu_smu_source, "rmu_smu_dsp_mul")
    add_check(checks, "rmu_smu_dsp_acc_helper_defined", "static hgtxr_acc_t rmu_smu_dsp_mul_acc(" in rmu_smu_source, "rmu_smu_dsp_mul_acc")
    add_check(checks, "rmu_projection_uses_dsp_helper", "rmu_smu_dsp_mul(cur, channel_gain(c))" in rmu_smu_source and "rmu_smu_dsp_mul_acc(token_mean" in rmu_smu_source, "RMU projection helper calls")
    add_check(checks, "smu_relation_uses_dsp_helper", "rmu_smu_dsp_mul(tokens[t][c], prob[t])" in rmu_smu_source and "rmu_smu_dsp_mul(score[t], token_gain(t))" in rmu_smu_source, "SMU relation helper calls")
    add_check(checks, "uram_bind_storage_present", bind_storage_uram_count >= 3, str(bind_storage_uram_count))
    add_check(checks, "lutram_bind_storage_present", bind_storage_lutram_count >= 3, str(bind_storage_lutram_count))
    add_check(checks, "cyclic_weight_tiles_uram_pragmas", cyclic_weight_uram_count >= 6, str(cyclic_weight_uram_count))
    add_check(checks, "cyclic_large_temps_uram_pragmas", cyclic_large_temp_uram_count >= 4, str(cyclic_large_temp_uram_count))
    add_check(checks, "cyclic_small_tile_lutram_pragmas", cyclic_small_tile_lutram_count >= 8, str(cyclic_small_tile_lutram_count))
    add_check(checks, "zcu104_parallelism_default", macro_value(config, "HGTXR_PARALLELISM_FACTOR") == 8, str(macro_value(config, "HGTXR_PARALLELISM_FACTOR")))
    add_check(checks, "zcu104_dense_parallelism_default", macro_value(config, "HGTXR_E2E_DENSE_PAR") == 8, str(macro_value(config, "HGTXR_E2E_DENSE_PAR")))
    add_check(checks, "zcu104_fifo_depth_default", macro_value(config, "HGTXR_FIFO_DEPTH") == 128, str(macro_value(config, "HGTXR_FIFO_DEPTH")))
    add_check(checks, "resource_matrix_pass", matrix.get("status") == "pass", str(matrix.get("status")))
    add_check(checks, "resource_matrix_recommends_c3b", matrix.get("summary", {}).get("recommended_board_smoke_variant") == "C3b" if isinstance(matrix.get("summary"), dict) else False, str(matrix.get("summary", {})))
    add_check(checks, "c3b_parallelism_16", c3b.get("parallelism") == 16, str(c3b.get("parallelism")))
    add_check(checks, "c3b_memory_banks_16", c3b.get("memory_banks") == 16, str(c3b.get("memory_banks")))
    add_check(checks, "c3b_dsp_increased_vs_a1", resource(c3b, "dsp") is not None and resource(a1, "dsp") is not None and resource(c3b, "dsp") > resource(a1, "dsp"), f"C3b={resource(c3b, 'dsp')} A1={resource(a1, 'dsp')}")
    add_check(checks, "c3b_lut_lower_than_c1", resource(c3b, "lut") is not None and resource(c1, "lut") is not None and resource(c3b, "lut") < resource(c1, "lut"), f"C3b={resource(c3b, 'lut')} C1={resource(c1, 'lut')}")
    add_check(checks, "c3b_uram_positive", resource(c3b, "uram") is not None and resource(c3b, "uram") > 0, str(resource(c3b, "uram")))
    add_check(checks, "c3b_latency_not_worse_than_c1", latency(c3b) is not None and latency(c1) is not None and latency(c3b) <= latency(c1), f"C3b={latency(c3b)} C1={latency(c1)}")
    add_check(checks, "c3b_csynth_xml_exists", c3b_csynth is not None, str(c3b_csynth_path))
    add_check(
        checks,
        "c3b_csynth_resources_match_matrix",
        bool(c3b_csynth_resources) and resource_values_match(c3b_csynth_resources, c3b_matrix_resources),
        f"csynth={c3b_csynth_resources} matrix={c3b_matrix_resources}",
    )
    add_check(
        checks,
        "c3b_csynth_latency_matches_matrix",
        c3b_csynth_latency is not None and c3b_csynth_latency == latency(c3b),
        f"csynth={c3b_csynth_latency} matrix={latency(c3b)}",
    )
    add_check(
        checks,
        "c3b_csynth_dsp_lte_threshold",
        isinstance(c3b_csynth_resources.get("dsp"), int) and c3b_csynth_resources["dsp"] <= C3B_DSP_MAX,
        f"{c3b_csynth_resources.get('dsp')} <= {C3B_DSP_MAX}",
    )
    add_check(
        checks,
        "c3b_csynth_uram_lte_threshold",
        isinstance(c3b_csynth_resources.get("uram"), int) and c3b_csynth_resources["uram"] <= C3B_URAM_MAX,
        f"{c3b_csynth_resources.get('uram')} <= {C3B_URAM_MAX}",
    )
    add_check(
        checks,
        "c3b_csynth_lut_lte_threshold",
        isinstance(c3b_csynth_resources.get("lut"), int) and c3b_csynth_resources["lut"] <= C3B_LUT_MAX,
        f"{c3b_csynth_resources.get('lut')} <= {C3B_LUT_MAX}",
    )
    add_check(
        checks,
        "c3b_csynth_latency_lte_threshold",
        isinstance(c3b_csynth_latency, int) and c3b_csynth_latency <= C3B_LATENCY_MAX,
        f"{c3b_csynth_latency} <= {C3B_LATENCY_MAX}",
    )
    add_check(
        checks,
        "c3b_routed_wns_gte_threshold",
        routed_wns(c3b) is not None and routed_wns(c3b) >= C3B_WNS_MIN,
        f"{routed_wns(c3b)} >= {C3B_WNS_MIN}",
    )

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "root": str(root),
        "date_tag": DATE_TAG,
        "source_files": {
            "hgtxr_e2e_vit": str(source_path),
            "rmu_smu": str(rmu_smu_path),
            "cyclic_top": str(cyclic_top_path),
            "cyclic_params": str(cyclic_params_path),
            "zcu104_e2e_config": str(config_path),
            "resource_matrix": str(matrix_path),
        },
        "policy": {
            "dsp": "multipliers route through hgtxr_e2e_dsp_mul helpers with bind_op impl=dsp",
            "uram": "large Q/K/V weight caches and report-level C3b design use URAM resources",
            "lutram": "small attention row scratch memories use bind_storage impl=lutram",
            "cyclic": "legacy cyclic weight tiles and large temporaries prefer URAM; small tile scratch prefers LUTRAM",
            "parallelism": "C3b uses PAR16 and MEM_BANK_PAR16 for the selected board-smoke candidate",
        },
        "observed": {
            "bind_op_dsp_count": bind_op_dsp_count,
            "rmu_smu_bind_op_dsp_count": rmu_smu_bind_op_dsp_count,
            "rmu_smu_dsp_helper_count": rmu_smu_dsp_helper_count,
            "rmu_smu_dsp_acc_helper_count": rmu_smu_dsp_acc_helper_count,
            "bind_storage_uram_count": bind_storage_uram_count,
            "bind_storage_lutram_count": bind_storage_lutram_count,
            "cyclic_weight_uram_count": cyclic_weight_uram_count,
            "cyclic_large_temp_uram_count": cyclic_large_temp_uram_count,
            "cyclic_small_tile_lutram_count": cyclic_small_tile_lutram_count,
            "c3b": {
                "parallelism": c3b.get("parallelism"),
                "memory_banks": c3b.get("memory_banks"),
                "dsp": resource(c3b, "dsp"),
                "lut": resource(c3b, "lut"),
                "uram": resource(c3b, "uram"),
                "latency_cycles": latency(c3b),
            },
            "c3b_csynth": {
                "path": str(c3b_csynth_path) if c3b_csynth_path else None,
                "latency_cycles": c3b_csynth_latency,
                "resources": c3b_csynth_resources,
            },
        },
        "checks": checks,
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "check_count": len(checks),
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "writes_hls_source": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# HGTXR E2E Resource Policy Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- root: `{audit['root']}`",
        f"- checks: `{audit['pass_count']}/{audit['check_count']}`",
        f"- fail_count: `{audit['fail_count']}`",
        "",
        "## Observed",
        "",
    ]
    for key, value in audit["observed"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(["", "## Safety", ""])
    for key, value in audit["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write E2E resource policy audit.")
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
    print(
        f"[e2e-resource-policy-audit] status={audit['status']} "
        f"checks={audit['pass_count']}/{audit['check_count']}"
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
