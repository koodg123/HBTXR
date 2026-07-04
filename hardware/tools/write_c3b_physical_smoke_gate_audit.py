#!/usr/bin/env python3
"""Write current C3b physical-smoke gate audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import validate_pynq_bundle_package  # noqa: E402
import validate_pynq_smoke_result  # noqa: E402


DATE_TAG = "2026_06_16"
PRESET = "axis-c3b-mem16"
VARIANT = "c3b-mem16"


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


def rel_or_abs(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def maybe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def validate_canonical_validation(path: Path) -> tuple[str, list[str], dict[str, Any]]:
    if not path.exists():
        return "missing", [], {}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return "fail", [f"C3b validation JSON invalid: {exc}"], {}

    errors: list[str] = []
    if payload.get("status") != "pass":
        errors.append(f"validation status: got {payload.get('status')!r}, expected 'pass'")
    if payload.get("preset") != PRESET:
        errors.append(f"validation preset: got {payload.get('preset')!r}, expected {PRESET!r}")
    payload_errors = payload.get("errors")
    if payload_errors not in ([], None):
        errors.append(f"validation errors not empty: {payload_errors!r}")
    return ("pass" if not errors else "fail"), errors, payload


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    canonical_result = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
    canonical_validation = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke_validation.json"
    bundle_dir = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
    bundle_tar = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
    session_json = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
    remote_run_json = hardware / "generated" / "signoff" / "zcu104_c3b_smoke_remote_run_2026_06_10.json"

    bundle_errors = validate_pynq_bundle_package.validate_bundle(bundle_dir, bundle_tar, VARIANT)
    session = maybe_json(session_json)
    manifest = maybe_json(bundle_dir / "BUNDLE_MANIFEST.json")

    validation_status, validation_errors, validation_payload = validate_canonical_validation(canonical_validation)

    if canonical_result.exists():
        try:
            result_payload = validate_pynq_smoke_result.load_json(canonical_result)
            smoke_errors = validate_pynq_smoke_result.validate_result(result_payload, PRESET, require_paths=True)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            smoke_errors = [f"C3b smoke JSON invalid: {exc}"]
        if validation_status == "fail":
            smoke_errors.extend(validation_errors)
        physical_status = "pass" if not smoke_errors else "fail"
    else:
        result_payload = {}
        smoke_errors = [f"C3b smoke JSON missing: {canonical_result}"]
        physical_status = "missing"

    bundle_status = "pass" if not bundle_errors else "fail"
    session_errors: list[str] = []
    if not session_json.exists():
        session_errors.append(f"session JSON missing: {session_json}")
    else:
        if session.get("status") != "pass":
            session_errors.append(f"session status: got {session.get('status')!r}, expected 'pass'")
        if session.get("preset") != PRESET:
            session_errors.append(f"session preset: got {session.get('preset')!r}, expected {PRESET!r}")
        if session.get("canonical_result_path") != str(canonical_result):
            session_errors.append("session canonical_result_path mismatch")
    session_status = "pass" if not session_errors else "fail"

    ready_for_board = bundle_status == "pass" and session_status == "pass"
    physical_smoke_pass = physical_status == "pass"
    status = "pass" if physical_smoke_pass else "blocked_missing_canonical_physical_smoke_result"
    return {
        "status": status,
        "date_tag": DATE_TAG,
        "task_id": "T-C3B-GATE-EVAL",
        "gate": "HGTXR third-goal Req4 C3b physical smoke",
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "preset": PRESET,
        "variant": VARIANT,
        "ready_for_board": ready_for_board,
        "physical_smoke_pass": physical_smoke_pass,
        "canonical_result": {
            "path": str(canonical_result),
            "exists": canonical_result.exists(),
            "status": physical_status,
            "errors": smoke_errors,
            "payload_status": result_payload.get("status") if result_payload else None,
        },
        "canonical_validation": {
            "path": str(canonical_validation),
            "exists": canonical_validation.exists(),
            "status": validation_status,
            "errors": validation_errors,
            "preset": validation_payload.get("preset") if validation_payload else None,
            "payload_status": validation_payload.get("status") if validation_payload else None,
            "result_json": validation_payload.get("result_json") if validation_payload else None,
            "source_sha256": validation_payload.get("source_sha256") if validation_payload else None,
        },
        "bundle": {
            "status": bundle_status,
            "bundle_dir": str(bundle_dir),
            "tar": str(bundle_tar),
            "errors": bundle_errors,
            "expected_out_raw": manifest.get("expected_out_raw"),
            "expected_runtime_state": manifest.get("expected_runtime_state"),
            "validation_command": manifest.get("validation_command"),
        },
        "session": {
            "status": session_status,
            "path": str(session_json),
            "exists": session_json.exists(),
            "errors": session_errors,
            "board_steps": session.get("board_steps", []),
            "host_steps": session.get("host_steps", []),
        },
        "remote_run": {
            "path": str(remote_run_json),
            "exists": remote_run_json.exists(),
        },
        "remaining_blockers": [] if physical_status == "pass" else ["C3b AXIS/DMA physical smoke result"],
        "next_actions": [
            "Run generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md on ZCU104.",
            "Copy e2e_axis_dma_c3b_mem16_file_smoke.json back to pynq/hgtxr/.",
            "Validate with tools/validate_pynq_smoke_result.py --preset axis-c3b-mem16.",
        ],
        "recommendation": (
            "Do not claim board smoke pass until ZCU104 result JSON is imported to the canonical path "
            "and final-signoff preflight passes."
        ),
        "reference_artifacts": [
            {"path": rel_or_abs(hardware, bundle_dir / "BUNDLE_MANIFEST.json"), "exists": (bundle_dir / "BUNDLE_MANIFEST.json").exists()},
            {"path": rel_or_abs(hardware, session_json), "exists": session_json.exists()},
            {"path": rel_or_abs(hardware, remote_run_json), "exists": remote_run_json.exists()},
        ],
        "safety": {
            "executes_commands": False,
            "executes_network": False,
            "creates_board_result": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# C3b Physical Smoke Gate Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- preset: `{audit['preset']}`",
        f"- variant: `{audit['variant']}`",
        f"- ready_for_board: `{audit['ready_for_board']}`",
        f"- physical_smoke_pass: `{audit['physical_smoke_pass']}`",
        f"- canonical_result_exists: `{audit['canonical_result']['exists']}`",
        f"- canonical_result_status: `{audit['canonical_result']['status']}`",
        f"- canonical_validation_exists: `{audit['canonical_validation']['exists']}`",
        f"- canonical_validation_status: `{audit['canonical_validation']['status']}`",
        f"- bundle_status: `{audit['bundle']['status']}`",
        f"- session_status: `{audit['session']['status']}`",
        f"- expected_out_raw: `{audit['bundle']['expected_out_raw']}`",
        f"- expected_runtime_state: `{audit['bundle']['expected_runtime_state']}`",
        "",
        "## Reference Artifacts",
    ]
    for artifact in audit["reference_artifacts"]:
        lines.append(f"- `{artifact['path']}`: `{artifact['exists']}`")
    lines.extend(["", "## Remaining Blockers"])
    if audit["remaining_blockers"]:
        for blocker in audit["remaining_blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Next Actions"])
    for action in audit["next_actions"]:
        lines.append(f"- {action}")
    lines.extend(["", "## Safety"])
    lines.append("- Does not run board smoke.")
    lines.append("- Does not create board result JSON.")
    lines.append("- Does not write canonical unblock inputs.")
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, default=Path(f"generated/signoff/c3b_physical_smoke_gate_audit_{DATE_TAG}.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path(f"generated/signoff/c3b_physical_smoke_gate_audit_{DATE_TAG}.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    write_outputs(audit, args.json_out, args.markdown_out)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "ready_for_board": audit["ready_for_board"],
                "physical_smoke_pass": audit["physical_smoke_pass"],
                "canonical_result_status": audit["canonical_result"]["status"],
                "bundle_status": audit["bundle"]["status"],
                "session_status": audit["session"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
