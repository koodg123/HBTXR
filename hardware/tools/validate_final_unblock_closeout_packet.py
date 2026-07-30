#!/usr/bin/env python3
"""Validate the final unblock closeout packet without executing operator commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import create_xr_vits_replacement_policy as xr_policy


DATE_TAG = "2026_06_10"
HGTXR_ROOT = Path(__file__).resolve().parents[2]
BASE_EXPECTED_BLOCKERS = {
    "C3b AXIS/DMA physical smoke result",
    "requested XR-VITs sibling",
}
EXPECTED_C3B_SHA256 = "3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


def safety_payload(packet: dict[str, Any]) -> dict[str, bool]:
    safety = packet.get("safety", {})
    if not isinstance(safety, dict):
        safety = {}
    executes_commands = bool(safety.get("executes_commands", safety.get("executes_network", True)))
    executes_network = bool(safety.get("executes_network", safety.get("executes_commands", True)))
    return {
        "executes_commands": executes_commands,
        "executes_network": executes_network,
        "creates_board_result": bool(safety.get("creates_board_result", True)),
        "creates_xr_vits_policy": bool(safety.get("creates_xr_vits_policy", True)),
        "writes_canonical_inputs": bool(safety.get("writes_canonical_inputs", True)),
    }


def command_groups(packet: dict[str, Any]) -> dict[str, Any]:
    commands = packet.get("operator_commands", {})
    return commands if isinstance(commands, dict) else {}


def load_policy_validation(root: Path, integrity: dict[str, Any]) -> dict[str, Any]:
    policy_path = Path(str(integrity.get("policy_path", root / "docs/resources/xr_vits_replacement_policy.json")))
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    if not policy_path.exists():
        return {
            "status": "pending-policy-creation",
            "policy_exists": False,
            "candidate_audit_path": str(root / xr_policy.DEFAULT_AUDIT_REL),
            "errors": [],
        }
    policy = load_json(policy_path)
    result = xr_policy.validate_policy_integrity(root, policy)
    result["policy_exists"] = True
    return result


def policy_validation_status(integrity: dict[str, Any]) -> str:
    validation = integrity.get("validation", {}) if isinstance(integrity.get("validation"), dict) else {}
    return str(validation.get("status", "missing"))


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers = set(str(item) for item in packet.get("remaining_blockers", []) if str(item))
    artifacts = packet.get("required_artifacts", [])
    artifact_entries = artifacts if isinstance(artifacts, list) else []
    board = packet.get("board_package", {}) if isinstance(packet.get("board_package"), dict) else {}
    contract = packet.get("c3b_smoke_contract", {}) if isinstance(packet.get("c3b_smoke_contract"), dict) else {}
    xr = packet.get("xr_vits_resolution", {}) if isinstance(packet.get("xr_vits_resolution"), dict) else {}
    closure = packet.get("closure_readiness", {}) if isinstance(packet.get("closure_readiness"), dict) else {}
    vref = packet.get("vref_successor_gate", {}) if isinstance(packet.get("vref_successor_gate"), dict) else {}
    qkv = packet.get("qkv_uram_successor_gate", {}) if isinstance(packet.get("qkv_uram_successor_gate"), dict) else {}
    vref_policy = packet.get("vref_successor_policy", {}) if isinstance(packet.get("vref_successor_policy"), dict) else {}
    xr_policy_integrity = xr.get("policy_integrity", {}) if isinstance(xr.get("policy_integrity"), dict) else {}
    root = Path(str(packet.get("root", HGTXR_ROOT))).resolve()
    require_vref = bool(vref_policy.get("require_physical_smoke_for_final_signoff")) or vref.get("required_for_final_signoff") is True
    expected_blockers = set(BASE_EXPECTED_BLOCKERS)
    if require_vref and vref.get("physical_smoke_status") != "pass":
        expected_blockers.add("VREF-P0 successor physical smoke result")
    commands = command_groups(packet)
    dry_run = commands.get("dry_run_readiness", [])
    combined = commands.get("combined_unblock", {})
    qkv_commands = commands.get("qkv_uram_successor", [])
    safety = safety_payload(packet)

    add_check(checks, "packet_status_ready", packet.get("status") == "ready-for-operator-unblock", str(packet.get("status")))
    add_check(checks, "remaining_blockers_expected", blockers == expected_blockers, str(sorted(blockers)))
    add_check(checks, "required_artifact_count", len(artifact_entries) >= 8, str(len(artifact_entries)))
    add_check(
        checks,
        "required_artifacts_all_pass",
        bool(artifact_entries) and all(isinstance(entry, dict) and entry.get("status") == "pass" for entry in artifact_entries),
        str([entry.get("status") for entry in artifact_entries if isinstance(entry, dict)]),
    )
    add_check(
        checks,
        "required_artifact_hashes_present",
        bool(artifact_entries) and all(isinstance(entry, dict) and is_sha256(entry.get("sha256", "")) for entry in artifact_entries),
        "sha256 on all required artifacts",
    )
    add_check(checks, "missing_required_artifacts_empty", packet.get("missing_required_artifacts") == [], str(packet.get("missing_required_artifacts")))
    add_check(checks, "board_package_ready", board.get("status") == "ready-for-board", str(board.get("status")))
    add_check(checks, "board_package_preset", board.get("preset") == "axis-c3b-mem16", str(board.get("preset")))
    add_check(checks, "board_package_sha", board.get("tar_sha256") == EXPECTED_C3B_SHA256, str(board.get("tar_sha256")))
    add_check(checks, "board_expected_runtime", board.get("expected_runtime_state") == 2, str(board.get("expected_runtime_state")))
    add_check(checks, "board_expected_output", board.get("expected_out_raw") == [32, -13, 26, -6, 14, -11], str(board.get("expected_out_raw")))
    add_check(checks, "c3b_contract_status", contract.get("status") == "pass", str(contract.get("status")))
    add_check(checks, "c3b_contract_preset", contract.get("preset") == "axis-c3b-mem16", str(contract.get("preset")))
    add_check(checks, "c3b_contract_variant", contract.get("variant") == "c3b-mem16", str(contract.get("variant")))
    add_check(
        checks,
        "c3b_contract_canonical_path",
        "e2e_axis_dma_c3b_mem16_file_smoke.json" in str(contract.get("canonical_result_path", "")),
        str(contract.get("canonical_result_path", "")),
    )
    add_check(checks, "c3b_contract_runtime", contract.get("expected_runtime_state") == 2, str(contract.get("expected_runtime_state")))
    add_check(checks, "c3b_contract_output", contract.get("expected_out_raw") == [32, -13, 26, -6, 14, -11], str(contract.get("expected_out_raw")))
    add_check(
        checks,
        "c3b_contract_commands_present",
        "validate_pynq_smoke_result.py" in str(contract.get("validate_command", ""))
        and "--dry-run" in str(contract.get("dry_run_import_command", ""))
        and "import_pynq_smoke_result.py" in str(contract.get("active_import_command", "")),
        str(contract),
    )
    add_check(checks, "xr_resolution_known", xr.get("status") == "candidate-ready-needs-approval", str(xr.get("status")))
    add_check(checks, "xr_candidate_score", xr.get("candidate_score") == 99, str(xr.get("candidate_score")))
    add_check(checks, "xr_candidate_path", "XR_Accel" in str(xr.get("candidate_replacement_path", "")), str(xr.get("candidate_replacement_path", "")))
    required_policy_fields = xr_policy_integrity.get("required_policy_fields", [])
    if not isinstance(required_policy_fields, list):
        required_policy_fields = []
    add_check(checks, "xr_policy_integrity_required", xr_policy_integrity.get("required") is True, str(xr_policy_integrity.get("required")))
    add_check(
        checks,
        "xr_policy_integrity_fields",
        {
            "candidate_audit_fingerprint",
            "candidate_audit_recommendation_snapshot",
            "candidate_audit_meta",
            "approval_event",
            "policy_fingerprint",
        }.issubset({str(field) for field in required_policy_fields}),
        str(required_policy_fields),
    )
    add_check(
        checks,
        "xr_policy_integrity_generator",
        "create_xr_vits_replacement_policy.py" in str(xr_policy_integrity.get("generator", "")),
        str(xr_policy_integrity.get("generator", "")),
    )
    add_check(
        checks,
        "xr_policy_integrity_validator",
        "check_final_blocker_closure_readiness.py" in str(xr_policy_integrity.get("validator", "")),
        str(xr_policy_integrity.get("validator", "")),
    )
    add_check(
        checks,
        "xr_legacy_policy_not_accepted",
        xr_policy_integrity.get("legacy_policy_clears_final_signoff") is False,
        str(xr_policy_integrity.get("legacy_policy_clears_final_signoff")),
    )
    stored_policy_status = policy_validation_status(xr_policy_integrity)
    live_policy_validation = load_policy_validation(root, xr_policy_integrity)
    live_policy_status = str(live_policy_validation.get("status", "missing"))
    pending_policy_allowed = (
        "requested XR-VITs sibling" in blockers
        and xr.get("status") == "candidate-ready-needs-approval"
        and xr_policy_integrity.get("policy_exists") is False
    )
    add_check(
        checks,
        "xr_policy_validation_status",
        live_policy_status == "pass" or (pending_policy_allowed and live_policy_status == "pending-policy-creation"),
        f"stored={stored_policy_status} live={live_policy_status}",
    )
    add_check(
        checks,
        "xr_policy_validation_not_legacy",
        live_policy_status != "legacy-warning",
        live_policy_status,
    )
    add_check(
        checks,
        "xr_policy_validation_embedded_matches_live",
        stored_policy_status == live_policy_status,
        f"stored={stored_policy_status} live={live_policy_status}",
    )
    add_check(checks, "closure_status_blocked", closure.get("status") == "blocked", str(closure.get("status")))
    add_check(checks, "closure_current_not_ready", closure.get("current_ready") is False, str(closure.get("current_ready")))
    add_check(
        checks,
        "vref_successor_not_final_blocker",
        (vref.get("required_for_final_signoff") is False and not require_vref)
        or (vref.get("required_for_final_signoff") is True and require_vref),
        f"required={vref.get('required_for_final_signoff')} policy={require_vref}",
    )
    if vref:
        add_check(
            checks,
            "vref_successor_gate_status_known",
            vref.get("status") in {"ready-for-physical-smoke", "blocked", "pass", "fail", "not_available"},
            str(vref.get("status")),
        )
        add_check(
            checks,
            "vref_successor_required_blocker_consistent",
            not require_vref
            or vref.get("physical_smoke_status") == "pass"
            or "VREF-P0 successor physical smoke result" in blockers,
            str(sorted(blockers)),
        )
        add_check(
            checks,
            "vref_successor_variant_selected",
            vref.get("recommended_resource_variant") in {"dsp_mixed_stream", None, ""},
            str(vref.get("recommended_resource_variant")),
        )
        add_check(
            checks,
            "vref_successor_physical_path",
            "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
            in str(vref.get("expected_result_json", "")),
            str(vref.get("expected_result_json", "")),
        )
    add_check(checks, "dry_run_commands_present", isinstance(dry_run, list) and len(dry_run) >= 3, str(len(dry_run) if isinstance(dry_run, list) else "missing"))
    add_check(
        checks,
        "dry_run_uses_closure_checker",
        isinstance(dry_run, list) and all("check_final_blocker_closure_readiness.py" in str(command) for command in dry_run),
        str(dry_run),
    )
    add_check(
        checks,
        "combined_unblock_options_present",
        isinstance(combined, dict) and {"U4a", "U4b"}.issubset(set(combined)),
        str(sorted(combined) if isinstance(combined, dict) else "missing"),
    )
    add_check(
        checks,
        "combined_replacement_has_dry_run",
        isinstance(combined, dict)
        and any("--dry-run-import-c3b-smoke" in str(command) for command in combined.get("U4b", []))
        and any("--dry-run-xr-vits-replacement" in str(command) for command in combined.get("U4b", [])),
        str(combined.get("U4b", []) if isinstance(combined, dict) else "missing"),
    )
    add_check(
        checks,
        "combined_replacement_has_active_command",
        isinstance(combined, dict)
        and any("--approve-xr-vits-replacement" in str(command) and "--dry-run-xr-vits-replacement" not in str(command) for command in combined.get("U4b", [])),
        str(combined.get("U4b", []) if isinstance(combined, dict) else "missing"),
    )
    add_check(
        checks,
        "combined_replacement_uses_policy_generator",
        isinstance(combined, dict)
        and any("create_xr_vits_replacement_policy.py" in str(command) for command in combined.get("U4b", [])),
        str(combined.get("U4b", []) if isinstance(combined, dict) else "missing"),
    )
    add_check(
        checks,
        "qkv_uram_successor_not_final_blocker",
        not qkv or qkv.get("required_for_final_signoff") is False,
        str(qkv.get("required_for_final_signoff")),
    )
    if qkv:
        resources = qkv.get("resources", {}) if isinstance(qkv.get("resources"), dict) else {}
        add_check(
            checks,
            "qkv_uram_successor_gate_status_known",
            qkv.get("status") in {"ready-for-physical-smoke", "pass", "not_available"},
            str(qkv.get("status")),
        )
        add_check(
            checks,
            "qkv_uram_successor_csynth_pass",
            qkv.get("csynth_status") in {"pass", None, ""},
            str(qkv.get("csynth_status")),
        )
        add_check(
            checks,
            "qkv_uram_successor_resources_present",
            all(key in resources for key in ("bram_18k", "dsp", "lut", "uram")),
            str(resources),
        )
        add_check(
            checks,
            "qkv_uram_successor_physical_path",
            "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
            in str(qkv.get("expected_result_json", "")),
            str(qkv.get("expected_result_json", "")),
        )
        add_check(
            checks,
            "qkv_uram_successor_command_group_present",
            isinstance(qkv_commands, list)
            and any("--execute-qkv-uram-smoke" in str(command) for command in qkv_commands)
            and any("--dry-run-import-qkv-uram-smoke" in str(command) for command in qkv_commands),
            str(qkv_commands),
        )
    add_check(
        checks,
        "no_side_effect_safety",
        safety == {
            "executes_commands": False,
            "executes_network": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
        str(safety),
    )

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "checks": checks,
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "check_count": len(checks),
        "remaining_blockers": sorted(blockers),
        "safety": safety,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Unblock Closeout Packet Validation",
        "",
        f"- status: `{result['status']}`",
        f"- checks: `{result['pass_count']}/{result['check_count']}`",
        f"- fail_count: `{result['fail_count']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in result["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final unblock closeout packet.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    packet_path = args.packet or root / "docs" / "resources" / f"final_unblock_closeout_packet_{DATE_TAG}.json"
    result = validate_packet(load_json(packet_path))
    result["root"] = str(root)
    result["packet"] = str(packet_path)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(result))
    print(
        f"[final-unblock-closeout-validation] status={result['status']} "
        f"fail={result['fail_count']} checks={result['check_count']}"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
