#!/usr/bin/env python3
"""Audit P2-ViT-style power-of-two scale readiness for VREF-P0-01."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"
HGTXR_ROOT = Path(__file__).resolve().parents[2]


SCALE_MACROS = [
    "HGTXR_E2E_ACC_SCALE",
    "HGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE",
    "HGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE",
    "HGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE",
    "HGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE",
    "HGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE",
    "HGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE",
]


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def macro_values(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for match in re.finditer(r"^\s*#define\s+([A-Za-z0-9_]+)\s+([0-9]+)\b", text, re.MULTILINE):
        values[match.group(1)] = int(match.group(2))
    return values


def is_power_of_two(value: int | None) -> bool:
    return isinstance(value, int) and value > 0 and (value & (value - 1)) == 0


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def line_refs(text: str, pattern: str, path: Path) -> list[str]:
    refs: list[str] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            refs.append(f"{path}:{idx}")
    return refs


def build_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    hardware = root / "hardware"
    source_path = hardware / "hls" / "include" / "hgtxr_e2e_vit.hpp"
    config_path = hardware / "configs" / "zcu104_e2e_q4w8a_defines.h"
    source = read_text(source_path)
    config = read_text(config_path)
    source_macros = macro_values(source)
    config_macros = macro_values(config)
    merged = dict(source_macros)
    merged.update(config_macros)

    checks: list[dict[str, Any]] = []
    add_check(checks, "q4_weight_contract", merged.get("HGTXR_WEIGHT_BIT_WIDTH") == 4, str(merged.get("HGTXR_WEIGHT_BIT_WIDTH")))
    add_check(checks, "q8_activation_contract", merged.get("HGTXR_BIT_WIDTH") == 8, str(merged.get("HGTXR_BIT_WIDTH")))
    add_check(checks, "dense_parallelism_positive", (merged.get("HGTXR_E2E_DENSE_PAR") or merged.get("HGTXR_PARALLELISM_FACTOR") or 0) > 0, str(merged.get("HGTXR_E2E_DENSE_PAR") or merged.get("HGTXR_PARALLELISM_FACTOR")))
    for macro in SCALE_MACROS:
        value = merged.get(macro)
        add_check(checks, f"pot_scale_{macro}", is_power_of_two(value), str(value))

    dsp_bind_refs = line_refs(source, "impl=dsp", source_path)
    dsp_helper_refs = line_refs(source, "hgtxr_e2e_dsp_mul", source_path)
    add_check(checks, "dsp_bind_op_present", len(dsp_bind_refs) >= 2, str(len(dsp_bind_refs)))
    add_check(checks, "dsp_helper_used", len(dsp_helper_refs) >= 4, str(len(dsp_helper_refs)))
    add_check(checks, "small_mem_lutram_enabled", merged.get("HGTXR_E2E_SMALL_MEM_LUTRAM") == 1, str(merged.get("HGTXR_E2E_SMALL_MEM_LUTRAM")))
    add_check(checks, "uram_buffer_policy_enabled", merged.get("HGTXR_E2E_FORCE_URAM_BUFFERS") == 1, str(merged.get("HGTXR_E2E_FORCE_URAM_BUFFERS")))

    scale_refs = {
        macro: line_refs(source, macro, source_path) + line_refs(config, macro, config_path)
        for macro in SCALE_MACROS
    }
    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "date_tag": DATE_TAG,
        "root": str(root),
        "source_files": {
            "hgtxr_e2e_vit": str(source_path),
            "zcu104_e2e_q4w8a_defines": str(config_path),
        },
        "policy": {
            "vref": "VREF-P0-01",
            "source": "P2-ViT",
            "scope": "SW-first PoT scale calibration readiness; no HLS source mutation",
            "promotion_gate": "SW/HW exact-match, bounded LUT delta, DSP mapping preserved, C3b not overwritten",
        },
        "observed": {
            "macros": {macro: merged.get(macro) for macro in ["HGTXR_WEIGHT_BIT_WIDTH", "HGTXR_BIT_WIDTH", "HGTXR_E2E_DENSE_PAR", "HGTXR_PARALLELISM_FACTOR", *SCALE_MACROS]},
            "dsp_bind_refs": dsp_bind_refs,
            "dsp_helper_ref_count": len(dsp_helper_refs),
            "scale_refs": scale_refs,
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
        "# VREF-P0-01 PoT Scale Readiness Audit",
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
    lines.extend(["", "## Scale Macros", "", "| Macro | Value |", "|---|---:|"])
    for macro, value in audit["observed"]["macros"].items():
        lines.append(f"| `{macro}` | `{value}` |")
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(["", "## Safety", ""])
    for key, value in audit["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write VREF-P0-01 PoT scale readiness audit.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audit))
    print(f"[vref-p0-pot-scale-audit] status={audit['status']} checks={audit['pass_count']}/{audit['check_count']}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
