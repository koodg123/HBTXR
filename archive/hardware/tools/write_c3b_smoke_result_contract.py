#!/usr/bin/env python3
"""Write the expected C3b board smoke result JSON contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import import_pynq_smoke_result
from validate_pynq_smoke_result import VARIANT_PRESETS


DATE_TAG = "2026_06_10"
HGTXR_ROOT = Path(__file__).resolve().parents[2]
PRESET = "axis-c3b-mem16"


def build_contract(root: Path) -> dict[str, Any]:
    root = root.resolve()
    preset = VARIANT_PRESETS[PRESET]
    canonical_rel = import_pynq_smoke_result.DEFAULT_DEST_BY_PRESET[PRESET]
    canonical_path = root / canonical_rel
    required_fields = {
        "status": "pass",
        "variant": preset["variant"],
        "weights_mode": preset["weights_mode"],
        "expected_runtime_state": preset["expected_runtime_state"],
        "runtime_state": preset["expected_runtime_state"],
        "runtime_match": True,
        "expected_out_raw": preset["expected_out_raw"],
        "out_raw": preset["expected_out_raw"],
        "output_match": True,
        "dma_in_name": "non-empty",
        "dma_out_name": "non-empty",
        "bitfile": "non-empty",
        "hwhfile": "non-empty",
        "out_state": f"list length {len(preset['expected_out_raw'])}",
    }
    result_name = "e2e_axis_dma_c3b_mem16_file_smoke.json"
    validate_command = (
        f"python3 tools/validate_pynq_smoke_result.py /path/to/{result_name} "
        f"--preset {PRESET}"
    )
    dry_run_import_command = (
        f"python3 tools/import_pynq_smoke_result.py /path/to/{result_name} "
        f"--preset {PRESET} --root {root} --dry-run "
        f"--json-out /tmp/c3b_smoke_import_{DATE_TAG}.json "
        f"--validation-out /tmp/c3b_smoke_import_validation_{DATE_TAG}.json"
    )
    active_import_command = (
        f"python3 tools/import_pynq_smoke_result.py /path/to/{result_name} "
        f"--preset {PRESET} --root {root} "
        f"--json-out /tmp/c3b_smoke_import_{DATE_TAG}.json "
        f"--validation-out /tmp/c3b_smoke_import_validation_{DATE_TAG}.json"
    )
    return {
        "status": "pass",
        "root": str(root),
        "preset": PRESET,
        "variant": preset["variant"],
        "canonical_result_path": str(canonical_path),
        "canonical_result_relative_path": canonical_rel,
        "board_result_filename": result_name,
        "required_fields": required_fields,
        "validation": {
            "validator": "hardware/tools/validate_pynq_smoke_result.py",
            "validate_command": validate_command,
            "dry_run_import_command": dry_run_import_command,
            "active_import_command": active_import_command,
            "require_paths": True,
        },
        "expected_unblock_effect": {
            "dry_run": "validates without writing canonical smoke result",
            "active": "copies valid board JSON to canonical_result_path",
            "final_gate": "clears C3b AXIS/DMA physical smoke result after active import",
        },
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(contract: dict[str, Any]) -> str:
    lines = [
        "# HGTXR C3b Board Smoke Result Contract",
        "",
        f"- status: `{contract['status']}`",
        f"- preset: `{contract['preset']}`",
        f"- variant: `{contract['variant']}`",
        f"- canonical_result_path: `{contract['canonical_result_path']}`",
        "",
        "## Required Fields",
        "",
        "| Field | Expected |",
        "|---|---|",
    ]
    for field, expected in contract["required_fields"].items():
        lines.append(f"| `{field}` | `{expected}` |")
    validation = contract["validation"]
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "### Validate",
            "",
            "```sh",
            validation["validate_command"],
            "```",
            "",
            "### Dry-Run Import",
            "",
            "```sh",
            validation["dry_run_import_command"],
            "```",
            "",
            "### Active Import",
            "",
            "```sh",
            validation["active_import_command"],
            "```",
            "",
            "## Safety",
            "",
        ]
    )
    for key, value in contract["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write C3b board smoke result contract.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = build_contract(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(contract))
    print(f"[c3b-smoke-result-contract] status={contract['status']} preset={contract['preset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
