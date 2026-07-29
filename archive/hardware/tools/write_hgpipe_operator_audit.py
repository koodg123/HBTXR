#!/usr/bin/env python3
"""Write HG-PIPE-derived operator implementation and validation audit."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import validate_hgpipe_lut_math  # noqa: E402


DATE_TAG = "2026_06_16"


OPERATORS: dict[str, dict[str, Any]] = {
    "LayerNorm": {
        "contract_prefixes": ["layernorm_attn", "layernorm_mlp", "layernorm_head"],
        "implementation_markers": [
            "hgtxr_e2e_layernorm",
            "HGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ",
            "hgtxr_e2e_hgpipe_lnq_requant_raw",
        ],
        "vref_markers": [
            "-DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1",
            "-DHGTXR_E2E_USE_HGPIPE_LNQ_GAMMA_RAW=1",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE=",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE=",
        ],
        "hgpipe_guide_markers": ["layernorm.h", "do_layernorm", "requant"],
    },
    "GeLU": {
        "contract_prefixes": ["gelu_quantized_mlp"],
        "implementation_markers": [
            "hgtxr_e2e_gelu",
            "HGTXR_E2E_USE_HGPIPE_INT_GELUQ",
            "hgtxr_hgpipe_geluq64_int",
        ],
        "vref_markers": [
            "-DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1",
            "-DHGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE=",
            "-DHGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE=",
        ],
        "hgpipe_guide_markers": ["gelu.h", "do_gelu", "GeLU"],
    },
    "Softmax": {
        "contract_prefixes": ["softmax_attn"],
        "implementation_markers": [
            "HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ",
            "hgtxr_e2e_hgpipe_softmax_exp_raw",
            "hgtxr_e2e_hgpipe_softmax_requant_raw",
        ],
        "vref_markers": [
            "-DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE=",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE=",
        ],
        "hgpipe_guide_markers": ["softmax.h", "do_softmax", "Softmax"],
    },
    "Quantization": {
        "contract_prefixes": ["quant_attn"],
        "implementation_markers": [
            "hgtxr_hgpipe_int_table_lookup",
            "hgtxr_clamp_int",
            "HGTXR_WEIGHT_BIT_WIDTH",
            "HGTXR_BIT_WIDTH",
        ],
        "vref_markers": [
            "-DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1",
            "-DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1",
            "-DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1",
        ],
        "hgpipe_guide_markers": ["quant.h", "quantize_clamp", "do_quant"],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def source_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def marker_results(text: str, markers: Sequence[str]) -> dict[str, bool]:
    return {marker: marker in text for marker in markers}


def matching_contracts(ref_checks: dict[str, Any], prefixes: Sequence[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in ref_checks.items()
        if any(key.startswith(prefix) or key == prefix for prefix in prefixes)
    }


def sample_count(ref_checks: dict[str, Any]) -> int:
    total = 0
    for value in ref_checks.values():
        if isinstance(value, dict):
            total += int(value.get("checked_samples", 0))
    return total


def add_property(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})


def parse_ap_int_range(type_name: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"ap_(u?)int<(\d+)>", str(type_name))
    if not match:
        return None
    unsigned = match.group(1) == "u"
    width = int(match.group(2))
    if width <= 0:
        return None
    if unsigned:
        return 0, (1 << width) - 1
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def numeric_table(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return []
        out.append(float(value))
    return out


def nonincreasing(values: Sequence[float]) -> bool:
    return all(values[index] >= values[index + 1] for index in range(len(values) - 1))


def operator_for_contract(name: str) -> str:
    if name.startswith("layernorm"):
        return "LayerNorm"
    if name.startswith("gelu"):
        return "GeLU"
    if name.startswith("softmax"):
        return "Softmax"
    if name.startswith("quant"):
        return "Quantization"
    return "global"


def build_property_checks(contract: dict[str, Any]) -> dict[str, Any]:
    cursor_contracts = contract.get("cursor_contracts", {})
    tables = contract.get("tables", {})
    if not isinstance(cursor_contracts, dict):
        cursor_contracts = {}
    if not isinstance(tables, dict):
        tables = {}

    checks: list[dict[str, Any]] = []
    operator_counts: dict[str, dict[str, int]] = {}

    for name, spec in cursor_contracts.items():
        if not isinstance(spec, dict):
            continue
        operator = operator_for_contract(str(name))
        operator_counts.setdefault(operator, {"checks": 0, "fail": 0})

        def record(suffix: str, passed: bool, detail: str) -> None:
            operator_counts[operator]["checks"] += 1
            if not passed:
                operator_counts[operator]["fail"] += 1
            add_property(checks, f"{name}_{suffix}", passed, detail)

        table = tables.get(name)
        if "entries" in spec and table is not None:
            expected = int(spec.get("entries", -1))
            actual = len(table) if isinstance(table, list) else -1
            record("table_length_matches_entries", actual == expected, f"{actual}/{expected}")
            scalars = spec.get("scalars", {}) if isinstance(spec.get("scalars"), dict) else {}
            if "bound" in scalars:
                record("cursor_bound_matches_entries", int(scalars["bound"]) == expected - 1, f"{scalars['bound']}/{expected - 1}")

        if "entries_exp" in spec:
            exp_table = tables.get(f"{name}_exp")
            expected = int(spec.get("entries_exp", -1))
            actual = len(exp_table) if isinstance(exp_table, list) else -1
            record("exp_table_length_matches_entries", actual == expected, f"{actual}/{expected}")

        if "entries_recip" in spec:
            recip_table = tables.get(f"{name}_recip")
            if recip_table is not None:
                expected = int(spec.get("entries_recip", -1))
                actual = len(recip_table) if isinstance(recip_table, list) else -1
                record("recip_table_length_matches_entries", actual == expected, f"{actual}/{expected}")

        output_range = parse_ap_int_range(str(spec.get("output_type", "")))
        values = numeric_table(table)
        if output_range and values:
            low, high = output_range
            in_range = all(low <= value <= high and float(value).is_integer() for value in values)
            record("integer_output_table_range", in_range, f"range=[{low},{high}] min={min(values)} max={max(values)}")

    for table_name in ["softmax_exp"]:
        values = numeric_table(tables.get(table_name))
        passed = bool(values) and all(math.isfinite(value) and value >= 0 for value in values) and nonincreasing(values)
        add_property(checks, f"{table_name}_nonnegative_finite_nonincreasing", passed, f"len={len(values)}")
        operator = operator_for_contract(table_name)
        operator_counts.setdefault(operator, {"checks": 0, "fail": 0})
        operator_counts[operator]["checks"] += 1
        if not passed:
            operator_counts[operator]["fail"] += 1

    values = numeric_table(tables.get("layernorm_rsqrt"))
    passed = bool(values) and all(math.isfinite(value) and value > 0 for value in values)
    add_property(checks, "layernorm_rsqrt_positive_finite", passed, f"len={len(values)}")
    operator_counts.setdefault("LayerNorm", {"checks": 0, "fail": 0})
    operator_counts["LayerNorm"]["checks"] += 1
    if not passed:
        operator_counts["LayerNorm"]["fail"] += 1

    for name, table in tables.items():
        if str(name).startswith("softmax_attn") and str(name).endswith("_exp"):
            values = numeric_table(table)
            passed = bool(values) and all(math.isfinite(value) and value >= 0 for value in values) and nonincreasing(values)
            add_property(checks, f"{name}_nonnegative_finite_nonincreasing", passed, f"len={len(values)}")
            operator_counts.setdefault("Softmax", {"checks": 0, "fail": 0})
            operator_counts["Softmax"]["checks"] += 1
            if not passed:
                operator_counts["Softmax"]["fail"] += 1

    quantize = cursor_contracts.get("quantize_clamp", {})
    if isinstance(quantize, dict):
        signed_range = quantize.get("signed_range", [])
        unsigned_range = quantize.get("unsigned_range", [])
        signed_bits = int(quantize.get("signed_bits", 0))
        unsigned_bits = int(quantize.get("unsigned_bits", 0))
        expected_signed = [-(1 << (signed_bits - 1)), (1 << (signed_bits - 1)) - 1] if signed_bits > 0 else []
        expected_unsigned = [0, (1 << unsigned_bits) - 1] if unsigned_bits > 0 else []
        add_property(checks, "quantize_clamp_signed_range_matches_bits", signed_range == expected_signed, f"{signed_range}/{expected_signed}")
        add_property(checks, "quantize_clamp_unsigned_range_matches_bits", unsigned_range == expected_unsigned, f"{unsigned_range}/{expected_unsigned}")
        operator_counts.setdefault("Quantization", {"checks": 0, "fail": 0})
        operator_counts["Quantization"]["checks"] += 2
        if signed_range != expected_signed:
            operator_counts["Quantization"]["fail"] += 1
        if unsigned_range != expected_unsigned:
            operator_counts["Quantization"]["fail"] += 1

    fail_count = len([check for check in checks if check["status"] == "fail"])
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "check_count": len(checks),
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "operator_counts": operator_counts,
        "checks": checks,
    }


def build_operator_audit(root: Path) -> dict[str, Any]:
    _hgtxr, hardware = normalize_roots(root)
    contract_path = hardware / "refs" / "hgpipe_lut_math_contract.json"
    contract = validate_hgpipe_lut_math.load_contract(contract_path)
    validated = validate_hgpipe_lut_math.validate_contract(contract)
    ref_checks = validated["hgpipe_ref_checks"]
    property_checks = build_property_checks(contract)

    math_header = hardware / "hls" / "include" / "hgtxr_cyclic_math.hpp"
    e2e_header = hardware / "hls" / "include" / "hgtxr_e2e_vit.hpp"
    guide = hardware / "docs" / "SRC_CASE_MODULE_GUIDE.md"
    hgpipe_analysis = hardware / "analysis" / "vit-accel" / "codebases" / "HG-PIPE" / "analysis.md"
    successor_json = hardware / f"generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_{DATE_TAG}.json"
    successor = load_json(successor_json) if successor_json.exists() else {}
    custom_flags = str(successor.get("custom_scale_flags", ""))
    csim_result = successor.get("csim_result", {}) if isinstance(successor.get("csim_result"), dict) else {}

    combined_impl_text = "\n".join([source_text(math_header), source_text(e2e_header)])
    combined_ref_text = "\n".join([source_text(guide), source_text(hgpipe_analysis)])

    operators: list[dict[str, Any]] = []
    for name, spec in OPERATORS.items():
        op_ref_checks = matching_contracts(ref_checks, spec["contract_prefixes"])
        op_property = property_checks["operator_counts"].get(name, {"checks": 0, "fail": 0})
        implementation_markers = marker_results(combined_impl_text, spec["implementation_markers"])
        vref_markers = marker_results(custom_flags, spec["vref_markers"])
        guide_markers = marker_results(combined_ref_text, spec["hgpipe_guide_markers"])
        missing = [
            f"implementation:{marker}"
            for marker, present in implementation_markers.items()
            if not present
        ]
        missing += [
            f"vref_flag:{marker}"
            for marker, present in vref_markers.items()
            if not present
        ]
        missing += [
            f"guide:{marker}"
            for marker, present in guide_markers.items()
            if not present
        ]
        if not op_ref_checks:
            missing.append("hgpipe_ref_contracts")
        if op_property["fail"]:
            missing.append("property_checks")
        status = "pass" if not missing and all(
            isinstance(check, dict) and check.get("status") == "pass"
            for check in op_ref_checks.values()
        ) else "partial"
        operators.append(
            {
                "operator": name,
                "status": status,
                "contract_count": len(op_ref_checks),
                "checked_samples": sample_count(op_ref_checks),
                "property_check_count": op_property["checks"],
                "property_fail_count": op_property["fail"],
                "implementation_markers": implementation_markers,
                "vref_markers": vref_markers,
                "guide_markers": guide_markers,
                "missing": missing,
            }
        )

    total_contracts = len(ref_checks)
    passed_contracts = len(
        [check for check in ref_checks.values() if isinstance(check, dict) and check.get("status") == "pass"]
    )
    status = (
        "pass"
        if all(item["status"] == "pass" for item in operators)
        and csim_result.get("status") == "pass"
        and property_checks["status"] == "pass"
        else "partial"
    )
    return {
        "status": status,
        "date_tag": DATE_TAG,
        "hardware_root": str(hardware),
        "contract": str(contract_path),
        "contract_schema": contract.get("schema"),
        "contract_summary": {
            "total_ref_checks": total_contracts,
            "passed_ref_checks": passed_contracts,
            "total_checked_samples": sample_count(ref_checks),
        },
        "property_summary": {
            "status": property_checks["status"],
            "check_count": property_checks["check_count"],
            "pass_count": property_checks["pass_count"],
            "fail_count": property_checks["fail_count"],
        },
        "property_checks": property_checks["checks"],
        "operators": operators,
        "implementation_sources": [
            str(math_header),
            str(e2e_header),
        ],
        "reference_sources": [
            str(guide),
            str(hgpipe_analysis),
        ],
        "vref_successor": {
            "artifact": str(successor_json),
            "custom_scale_flags_present": bool(custom_flags),
            "csim_status": csim_result.get("status"),
            "csim_log": csim_result.get("csim_log"),
            "expected_markers": csim_result.get("expected_markers", []),
            "missing_markers": csim_result.get("missing_markers", []),
        },
        "residual_risk": [
            "This is local sampled/reference-vector equivalence plus deterministic contract property checks, not a formal proof over every possible input.",
            "Final E2E hardware signoff still depends on C3b physical smoke and XR-VITs source/replacement policy.",
        ],
        "safety": {
            "executes_hls": False,
            "executes_network": False,
            "creates_board_result": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# HG-PIPE Operator Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- date_tag: `{audit['date_tag']}`",
        f"- total_ref_checks: `{audit['contract_summary']['total_ref_checks']}`",
        f"- passed_ref_checks: `{audit['contract_summary']['passed_ref_checks']}`",
        f"- total_checked_samples: `{audit['contract_summary']['total_checked_samples']}`",
        f"- property_checks: `{audit['property_summary']['pass_count']}/{audit['property_summary']['check_count']}`",
        f"- vref_csim_status: `{audit['vref_successor']['csim_status']}`",
        "",
        "## Operators",
        "",
        "| Operator | Status | Contracts | Samples | Properties | Missing |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in audit["operators"]:
        missing = "<br>".join(item["missing"]) if item["missing"] else "-"
        lines.append(
            f"| {item['operator']} | `{item['status']}` | {item['contract_count']} | {item['checked_samples']} | "
            f"{item['property_check_count'] - item['property_fail_count']}/{item['property_check_count']} | {missing} |"
        )
    lines.extend(["", "## Property Checks", ""])
    lines.append(
        f"- status: `{audit['property_summary']['status']}`, "
        f"pass `{audit['property_summary']['pass_count']}/{audit['property_summary']['check_count']}`, "
        f"fail `{audit['property_summary']['fail_count']}`"
    )
    lines.extend(["", "## Sources"])
    for path in audit["implementation_sources"]:
        lines.append(f"- implementation: `{path}`")
    for path in audit["reference_sources"]:
        lines.append(f"- reference: `{path}`")
    lines.extend(["", "## Residual Risk"])
    for risk in audit["residual_risk"]:
        lines.append(f"- {risk}")
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, default=Path(f"generated/signoff/hgpipe_operator_audit_{DATE_TAG}.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path(f"generated/signoff/hgpipe_operator_audit_{DATE_TAG}.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_operator_audit(args.root)
    write_outputs(audit, args.json_out, args.markdown_out)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "operators": {item["operator"]: item["status"] for item in audit["operators"]},
                "total_ref_checks": audit["contract_summary"]["total_ref_checks"],
                "total_checked_samples": audit["contract_summary"]["total_checked_samples"],
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
