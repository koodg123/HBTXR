#!/usr/bin/env python3
"""Audit Req6 HLS parameterization coverage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import yaml


DATE_TAG = "2026_06_16"

REQ6_KNOBS = [
    "HGTXR_TILING_FACTOR",
    "HGTXR_PARALLELISM_FACTOR",
    "HGTXR_BUS_WIDTH",
    "HGTXR_BIT_WIDTH",
    "HGTXR_WEIGHT_BIT_WIDTH",
    "HGTXR_BUFFER_SIZE",
    "HGTXR_FIFO_DEPTH",
]

LEGACY_DEFAULTS = {
    "HGTXR_ACC_W": 32,
    "HGTXR_CYCLIC_MODEL_DIM": 192,
    "HGTXR_CYCLIC_MLP_RATIO": 4,
    "HGTXR_E2E_HEADS": 3,
    "HGTXR_E2E_HEAD_DIM": 64,
    "HGTXR_E2E_FF_DIM": 768,
    "HGTXR_E2E_DENSE_PAR": 8,
}


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def macro_values(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for match in re.finditer(r"^\s*#\s*define\s+([A-Za-z0-9_]+)\s+([0-9]+)\b", text, re.MULTILINE):
        values[match.group(1)] = int(match.group(2))
    return values


def line_refs(text: str, needle: str, rel_path: str) -> list[str]:
    return [
        f"{rel_path}:{idx}"
        for idx, line in enumerate(text.splitlines(), start=1)
        if needle in line
    ]


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def macro_int(macros: dict[str, int], name: str) -> int | None:
    value = macros.get(name)
    if isinstance(value, int):
        return value
    fallback = LEGACY_DEFAULTS.get(name)
    return fallback if isinstance(fallback, int) else None


def positive(value: int | None) -> bool:
    return isinstance(value, int) and value > 0


def divisible(numerator: int | None, denominator: int | None) -> bool:
    return positive(numerator) and positive(denominator) and numerator % denominator == 0


def sweep_has_values(text: str, name: str, values: list[int]) -> bool:
    pattern = rf"{re.escape(name)}:\s*(?:\n\s+.*)*?\n\s+values:\s*\[([^\]]+)\]"
    match = re.search(pattern, text)
    if not match:
        return False
    found = [int(token) for token in re.findall(r"\d+", match.group(1))]
    return found == values


def load_yaml_object(text: str) -> dict[str, Any]:
    payload = yaml.safe_load(text) if text.strip() else {}
    return payload if isinstance(payload, dict) else {}


def tcl_supports_par16_par32(text: str) -> bool:
    return (
        '$e2e_par ne "16" && $e2e_par ne "32"' in text
        and "supported: 16, 32" in text
        and "-DHGTXR_PARALLELISM_FACTOR=$e2e_par" in text
        and "-DHGTXR_E2E_DENSE_PAR=$e2e_par" in text
    )


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    paths = {
        "config": "configs/zcu104_e2e_q4w8a_defines.h",
        "params": "hls/include/hgtxr_cyclic_transformer_params.hpp",
        "e2e": "hls/include/hgtxr_e2e_vit.hpp",
        "axis": "hls/src/hgtxr_e2e_axis_top.cpp",
        "maxis": "hls/src/hgtxr_e2e_m_axi_top.cpp",
        "csim_tcl": "vivado/scripts/run_e2e_q4w8a_csim.tcl",
        "csynth_tcl": "vivado/scripts/run_e2e_q4w8a_csynth.tcl",
        "sweep": "configs/sweeps/zcu104_cyclic_transformer_sweep.yaml",
    }
    texts = {key: read_text(hardware / rel) for key, rel in paths.items()}
    sweep_doc = load_yaml_object(texts["sweep"])
    sweep_matrix = sweep_doc.get("sweep_matrix", {}) if isinstance(sweep_doc.get("sweep_matrix"), dict) else {}
    parallelism_extensions = (
        sweep_matrix.get("parallelism_extensions", {})
        if isinstance(sweep_matrix.get("parallelism_extensions"), dict)
        else {}
    )
    c3b_par16 = (
        parallelism_extensions.get("C3b_PAR16", {})
        if isinstance(parallelism_extensions.get("C3b_PAR16"), dict)
        else {}
    )
    par32 = (
        parallelism_extensions.get("PAR32", {})
        if isinstance(parallelism_extensions.get("PAR32"), dict)
        else {}
    )
    config_macros = macro_values(texts["config"])
    params_macros = macro_values(texts["params"])
    e2e_macros = macro_values(texts["e2e"])
    merged_macros = dict(params_macros)
    merged_macros.update(e2e_macros)
    merged_macros.update(config_macros)

    checks: list[dict[str, Any]] = []
    for knob in REQ6_KNOBS:
        add_check(checks, f"config_defines_{knob}", knob in config_macros, config_macros.get(knob))

    expected_defaults = {
        "HGTXR_TILING_FACTOR": 1,
        "HGTXR_PARALLELISM_FACTOR": 8,
        "HGTXR_BUS_WIDTH": 256,
        "HGTXR_BIT_WIDTH": 8,
        "HGTXR_WEIGHT_BIT_WIDTH": 4,
        "HGTXR_BUFFER_SIZE": 256,
        "HGTXR_FIFO_DEPTH": 128,
    }
    for knob, value in expected_defaults.items():
        add_check(checks, f"zcu104_q4w8a_default_{knob}", config_macros.get(knob) == value, config_macros.get(knob))

    derived_needles = {
        "tiling_to_tile_tokens": "#define HGTXR_TILE_TOKENS (16 * HGTXR_TILING_FACTOR)",
        "tiling_to_tile_channels": "#define HGTXR_TILE_CHANNELS (32 * HGTXR_TILING_FACTOR)",
        "parallelism_to_head_par": "#define HGTXR_HEAD_PAR HGTXR_PARALLELISM_FACTOR",
        "parallelism_to_pe_par": "#define HGTXR_PE_PAR HGTXR_PARALLELISM_FACTOR",
        "bus_width_to_bus_bits": "#define HGTXR_BUS_WIDTH_BITS HGTXR_BUS_WIDTH",
        "bit_width_to_data_w": "#define HGTXR_DATA_W HGTXR_BIT_WIDTH",
        "buffer_size_to_local_depth": "#define HGTXR_LOCAL_BUFFER_DEPTH (((HGTXR_BUFFER_SIZE) * 1024 * 8) / HGTXR_DATA_W)",
        "fifo_depth_to_stream_fifo": "#define HGTXR_STREAM_FIFO_DEPTH HGTXR_FIFO_DEPTH",
    }
    for name, needle in derived_needles.items():
        refs = line_refs(texts["params"], needle, paths["params"])
        add_check(checks, name, bool(refs), refs)

    e2e_needles = {
        "axis_word_uses_bus_width": "ap_axiu<HGTXR_BUS_WIDTH, 0, 0, 0>",
        "dense_par_defaults_to_parallelism": "#define HGTXR_E2E_DENSE_PAR HGTXR_PARALLELISM_FACTOR",
        "mem_bank_par_defaults_to_dense": "#define HGTXR_E2E_MEM_BANK_PAR HGTXR_E2E_DENSE_PAR",
        "q4_extract_specialized_for_bus": "#if HGTXR_WEIGHT_BIT_WIDTH == 4 && HGTXR_BUS_WIDTH_BITS == 256",
    }
    for name, needle in e2e_needles.items():
        refs = line_refs(texts["e2e"], needle, paths["e2e"])
        add_check(checks, name, bool(refs), refs)

    e2e_static_assert_needles = {
        "e2e_static_assert_active_tokens_fit": "static_assert(kActiveTokens > 0 && kActiveTokens <= HGTXR_TOKENS",
        "e2e_static_assert_heads_positive": "static_assert(kHeads > 0",
        "e2e_static_assert_head_dim_covers_embed": "static_assert(kHeadDim * kHeads == kEmbed",
        "e2e_static_assert_dense_par_positive": "static_assert(kDensePar > 0",
        "e2e_static_assert_dense_par_divides_embed": "static_assert(kEmbed % kDensePar == 0",
        "e2e_static_assert_dense_par_divides_ff_dim": "static_assert(kFfDim > 0 && kFfDim % kDensePar == 0",
        "e2e_static_assert_weight_lanes_positive": "static_assert(kWeightLanes > 0",
        "e2e_static_assert_dense_par_divides_weight_lanes": "static_assert(kWeightLanes % kDensePar == 0",
        "e2e_static_assert_axis_width_matches_cyclic_axi": "static_assert(HGTXR_BUS_WIDTH == hgtxr::cyclic_transformer::HGTXR_AXI_DATA_WIDTH",
    }
    for name, needle in e2e_static_assert_needles.items():
        refs = line_refs(texts["e2e"], needle, paths["e2e"])
        add_check(checks, name, bool(refs), refs)

    axis_refs = line_refs(texts["axis"], "depth=HGTXR_STREAM_FIFO_DEPTH", paths["axis"])
    add_check(checks, "axis_ports_use_fifo_depth", len(axis_refs) >= 2, axis_refs)
    axis_bank_refs = line_refs(texts["axis"], "factor=HGTXR_E2E_MEM_BANK_PAR", paths["axis"])
    add_check(checks, "axis_partitions_use_mem_bank_par", len(axis_bank_refs) >= 8, axis_bank_refs)
    maxis_bank_refs = line_refs(texts["maxis"], "factor=HGTXR_E2E_DENSE_PAR", paths["maxis"])
    add_check(checks, "m_axi_partitions_use_dense_par", len(maxis_bank_refs) >= 8, maxis_bank_refs)

    for key in ["csim_tcl", "csynth_tcl"]:
        text = texts[key]
        add_check(checks, f"{key}_includes_q4w8a_config", "zcu104_e2e_q4w8a_defines.h" in text, paths[key])
        add_check(
            checks,
            f"{key}_has_parallelism_override",
            "-DHGTXR_PARALLELISM_FACTOR=$e2e_par" in text
            and "-DHGTXR_E2E_DENSE_PAR=$e2e_par" in text,
            paths[key],
        )
        add_check(
            checks,
            f"{key}_supports_par16_par32",
            tcl_supports_par16_par32(text),
            paths[key],
        )

    bus_width = macro_int(merged_macros, "HGTXR_BUS_WIDTH")
    bit_width = macro_int(merged_macros, "HGTXR_BIT_WIDTH")
    weight_bit_width = macro_int(merged_macros, "HGTXR_WEIGHT_BIT_WIDTH")
    acc_width = macro_int(merged_macros, "HGTXR_ACC_W")
    model_dim = macro_int(merged_macros, "HGTXR_CYCLIC_MODEL_DIM")
    mlp_ratio = macro_int(merged_macros, "HGTXR_CYCLIC_MLP_RATIO")
    heads = macro_int(merged_macros, "HGTXR_E2E_HEADS")
    head_dim = macro_int(merged_macros, "HGTXR_E2E_HEAD_DIM")
    ff_dim = macro_int(merged_macros, "HGTXR_E2E_FF_DIM")
    parallelism = macro_int(merged_macros, "HGTXR_PARALLELISM_FACTOR")
    dense_par = macro_int(merged_macros, "HGTXR_E2E_DENSE_PAR")
    buffer_size = macro_int(merged_macros, "HGTXR_BUFFER_SIZE")
    fifo_depth = macro_int(merged_macros, "HGTXR_FIFO_DEPTH")
    weight_lanes = bus_width // weight_bit_width if divisible(bus_width, weight_bit_width) else None
    expected_head_dim = model_dim // heads if divisible(model_dim, heads) else None
    expected_ff_dim = model_dim * mlp_ratio if positive(model_dim) and positive(mlp_ratio) else None
    effective_dense_par = dense_par if positive(dense_par) else parallelism

    add_check(checks, "legal_bus_width_byte_aligned", divisible(bus_width, 8), {"bus_width": bus_width})
    add_check(checks, "legal_bus_width_data_divisible", divisible(bus_width, bit_width), {"bus_width": bus_width, "bit_width": bit_width})
    add_check(checks, "legal_bus_width_weight_divisible", divisible(bus_width, weight_bit_width), {"bus_width": bus_width, "weight_bit_width": weight_bit_width})
    add_check(checks, "legal_weight_width_lte_data_width", positive(weight_bit_width) and positive(bit_width) and weight_bit_width <= bit_width, {"weight_bit_width": weight_bit_width, "bit_width": bit_width})
    add_check(checks, "legal_data_width_lt_acc_width", positive(bit_width) and positive(acc_width) and bit_width < acc_width, {"bit_width": bit_width, "acc_width": acc_width})
    add_check(checks, "legal_model_dim_divides_heads", divisible(model_dim, heads), {"model_dim": model_dim, "heads": heads})
    add_check(checks, "legal_head_dim_matches_model_heads", positive(head_dim) and head_dim == expected_head_dim, {"head_dim": head_dim, "expected": expected_head_dim})
    add_check(checks, "legal_ff_dim_matches_mlp_ratio", positive(ff_dim) and ff_dim == expected_ff_dim, {"ff_dim": ff_dim, "expected": expected_ff_dim})
    add_check(checks, "legal_dense_parallelism_matches_req6_parallelism", positive(dense_par) and dense_par == parallelism, {"dense_par": dense_par, "parallelism": parallelism})
    add_check(checks, "legal_dense_parallelism_divides_embed", divisible(model_dim, effective_dense_par), {"model_dim": model_dim, "dense_par": effective_dense_par})
    add_check(checks, "legal_dense_parallelism_divides_ff_dim", divisible(ff_dim, effective_dense_par), {"ff_dim": ff_dim, "dense_par": effective_dense_par})
    add_check(checks, "legal_weight_lanes_divide_dense_parallelism", divisible(weight_lanes, effective_dense_par), {"weight_lanes": weight_lanes, "dense_par": effective_dense_par})
    add_check(checks, "legal_buffer_size_positive", positive(buffer_size), {"buffer_size": buffer_size})
    add_check(checks, "legal_fifo_depth_positive", positive(fifo_depth), {"fifo_depth": fifo_depth})

    sweep_expectations = {
        "tiling_factor": [1, 2, 4, 8],
        "parallelism_factor": [1, 2, 4, 8, 16, 32],
        "bus_width": [64, 128, 256, 512],
        "bit_width": [8, 16],
        "buffer_size": [64, 128, 256, 512],
        "fifo_depth": [16, 32, 64, 128],
    }
    for name, values in sweep_expectations.items():
        add_check(checks, f"sweep_covers_{name}", sweep_has_values(texts["sweep"], name, values), values)

    add_check(checks, "sweep_yaml_parsed", bool(sweep_matrix), paths["sweep"])
    add_check(
        checks,
        "parallelism_extension_c3b_par16_validated",
        c3b_par16.get("status") == "validated_resource_matrix",
        c3b_par16.get("status"),
    )
    add_check(
        checks,
        "parallelism_extension_c3b_par16_evidence_resource_matrix",
        "e2e_resource_matrix_2026_06_10.json:C3b" in str(c3b_par16.get("evidence", "")),
        c3b_par16.get("evidence"),
    )
    c3b_expected = str(c3b_par16.get("expected_result", ""))
    add_check(
        checks,
        "parallelism_extension_c3b_par16_expected_result",
        all(token in c3b_expected for token in ["DSP 604", "LUT 126506", "URAM 64", "latency 37508072"]),
        c3b_expected,
    )
    add_check(
        checks,
        "parallelism_extension_par32_exploratory_not_default",
        par32.get("status") == "exploratory_not_default",
        par32.get("status"),
    )
    par32_promotion = str(par32.get("promotion_rule", ""))
    add_check(
        checks,
        "parallelism_extension_par32_requires_fresh_reports",
        all(token in par32_promotion for token in ["fresh csynth", "routed timing", "resource-fit audit", "no C3b artifact overwrite"]),
        par32_promotion,
    )
    par32_expected = str(par32.get("expected_result", ""))
    add_check(
        checks,
        "parallelism_extension_par32_records_risk",
        all(token in par32_expected for token in ["latency reduction", "DSP", "timing risk"]),
        par32_expected,
    )

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "requirement": "Req6 parameterize tiling, parallelism, bus width, bit width, buffer size, FIFO depth",
        "source_files": {key: str(hardware / rel) for key, rel in paths.items()},
        "knobs": {
            "config_macros": {knob: config_macros.get(knob) for knob in REQ6_KNOBS},
            "legality_macros": {
                "HGTXR_ACC_W": acc_width,
                "HGTXR_CYCLIC_MODEL_DIM": model_dim,
                "HGTXR_CYCLIC_MLP_RATIO": mlp_ratio,
                "HGTXR_E2E_HEADS": heads,
                "HGTXR_E2E_HEAD_DIM": head_dim,
                "HGTXR_E2E_FF_DIM": ff_dim,
                "HGTXR_E2E_DENSE_PAR": dense_par,
                "HGTXR_WEIGHT_LANES": weight_lanes,
            },
            "expected_defaults": expected_defaults,
            "sweep_values": sweep_expectations,
            "parallelism_extensions": {
                "C3b_PAR16": c3b_par16,
                "PAR32": par32,
            },
        },
        "coverage": {
            "config": "top-level ZCU104 Q4/W8A defaults are override-safe #ifndef macros",
            "derived_params": "core tile, parallel, bus, bit, local-buffer, and FIFO constants derive from Req6 macros",
            "hls_pragmas": "AXIS FIFO depth and E2E memory partition factors consume derived macros",
            "tool_overrides": "CSim/CSynth Tcl can override E2E parallelism for PAR16/C3b-style builds and PAR32 exploration",
            "legality": "bus/bit widths, head dimensions, dense parallelism, packed weight lanes, buffer size, and FIFO depth are checked for HLS-legal relationships",
            "static_asserts": "E2E header has compile-time guards for active tokens, heads, dimensions, dense parallelism, packed weight lanes, and AXIS width",
            "sweep": "machine-readable sweep matrix covers all requested knobs",
            "parallelism_extensions": "C3b PAR16 is the validated resource-matrix path and PAR32 remains exploratory until fresh csynth, routing, resource-fit, and no-overwrite evidence exist",
        },
        "checks": checks,
        "check_count": len(checks),
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "writes_hls_source": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Req6 Parameterization Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- root: `{audit['root']}`",
        f"- checks: `{audit['pass_count']}/{audit['check_count']}`",
        f"- requirement: {audit['requirement']}",
        "",
        "## Knobs",
        "",
        "| Macro | Value |",
        "|---|---:|",
    ]
    for key, value in audit["knobs"]["config_macros"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Coverage", ""])
    for key, value in audit["coverage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        detail = check["detail"]
        if isinstance(detail, list):
            detail_text = "<br>".join(str(item) for item in detail) if detail else "-"
        else:
            detail_text = str(detail)
        lines.append(f"| {check['name']} | `{check['status']}` | {detail_text} |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Req6 parameterization coverage audit.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    _hgtxr, hardware = normalize_roots(args.root)
    json_out = args.json_out or hardware / "generated" / "signoff" / f"req6_parameterization_audit_{DATE_TAG}.json"
    markdown_out = args.markdown_out or hardware / "generated" / "signoff" / f"req6_parameterization_audit_{DATE_TAG}.md"
    audit = build_audit(args.root)
    write_outputs(audit, json_out, markdown_out)
    print(f"[req6-parameterization-audit] status={audit['status']} checks={audit['pass_count']}/{audit['check_count']}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
