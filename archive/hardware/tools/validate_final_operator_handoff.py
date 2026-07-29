#!/usr/bin/env python3
"""Validate the final operator handoff without executing any commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import create_xr_vits_replacement_policy as xr_policy


DATE_TAG = "2026_06_10"
MIN_EVIDENCE_CONSISTENCY_COUNT = 347
MIN_EVIDENCE_REQUIRED_COUNT = 86
MIN_SOURCE_AUDIT_REQUIRED_COUNT = 86
MIN_SOURCE_AUDIT_SOURCE_COUNT = 224
MIN_RESOURCE_POLICY_CHECK_COUNT = 36
MIN_SPEC_PLAN_CHECK_COUNT = 86
MIN_REQ6_PARAMETERIZATION_CHECK_COUNT = 71
MIN_CLOSEOUT_VALIDATION_CHECK_COUNT = 49
EXPECTED_CURRENT_AUDIT_SUMMARY = {
    "requirements": 12,
    "reflected": 10,
    "partial": 1,
    "blocked": 1,
    "default_closeout_blockers": 2,
}
ALLOWED_BOOTSTRAP_EVIDENCE_FAILURES = {
    "spec_plan_current_doc_freshness_contract",
    "third_goal_completion_audit_req12_manifest_pass",
}
EXPECTED_FINAL_RUNNER_BLOCKERS = {
    "C3b AXIS/DMA physical smoke result",
    "requested XR-VITs sibling",
}
EXPECTED_REQ6_KNOBS = {
    "HGTXR_TILING_FACTOR": 1,
    "HGTXR_PARALLELISM_FACTOR": 8,
    "HGTXR_BUS_WIDTH": 256,
    "HGTXR_BIT_WIDTH": 8,
    "HGTXR_WEIGHT_BIT_WIDTH": 4,
    "HGTXR_BUFFER_SIZE": 256,
    "HGTXR_FIFO_DEPTH": 128,
}
REQUIRED_REQ5_CONSISTENCY_CHECKS = {
    "req5_q4q8_swhw_match_pass",
    "req5_q4q8_precision",
    "req5_q4q8_packed_weight_manifest",
    "req5_q4q8_packed_weight_binary_exists",
    "req5_q4q8_packed_weight_sha256_matches_manifest",
    "req5_q4q8_packed_weight_byte_count_matches_manifest",
    "req5_q4q8_packed_weight_expected_runtime_state",
    "req5_q4q8_packed_weight_expected_c3b_output",
    "req5_q4q8_testbench_strict_golden_compare",
    "req5_q4q8_c3b_axis_csim_strict_csim",
    "req5_q4q8_vref_softmax_input_x2_csim_strict_csim",
    "req5_q4q8_qkv_uram_csim_strict_csim",
}
REQUIRED_CYCLIC_RESOURCE_CONSISTENCY_CHECKS = {
    "resource_policy_canonical_pair_present",
    "resource_policy_docs_singularity",
    "resource_policy_generated_singularity",
    "resource_policy_filename_payload_date_match",
    "resource_policy_canonical_mirror_sha256",
    "resource_policy_audit_single_canonical_artifact_set",
    "resource_policy_audit_cyclic_weight_tiles_uram_macro_default",
    "resource_policy_audit_cyclic_large_temps_uram_macro_default",
    "resource_policy_audit_cyclic_small_tile_lutram_macro_default",
    "resource_policy_audit_cyclic_weight_tiles_uram_pragmas",
    "resource_policy_audit_cyclic_large_temps_uram_pragmas",
    "resource_policy_audit_cyclic_small_tile_lutram_pragmas",
}
REQUIRED_C3B_THRESHOLD_CONSISTENCY_CHECKS = {
    "resource_policy_audit_c3b_csynth_lut_lte_threshold",
    "resource_policy_audit_c3b_csynth_latency_lte_threshold",
    "resource_policy_audit_c3b_routed_wns_gte_threshold",
}
REQUIRED_LIVE_RESOURCE_POLICY_CHECKS = {
    "force_dsp_macro_default",
    "force_uram_macro_default",
    "small_mem_lutram_macro_default",
    "rmu_smu_dsp_helper_defined",
    "rmu_smu_dsp_acc_helper_defined",
    "rmu_projection_uses_dsp_helper",
    "smu_relation_uses_dsp_helper",
    "cyclic_weight_tiles_uram_pragmas",
    "cyclic_large_temps_uram_pragmas",
    "cyclic_small_tile_lutram_pragmas",
    "c3b_dsp_increased_vs_a1",
    "c3b_lut_lower_than_c1",
    "c3b_uram_positive",
    "c3b_latency_not_worse_than_c1",
    "c3b_csynth_dsp_lte_threshold",
    "c3b_csynth_uram_lte_threshold",
    "c3b_csynth_lut_lte_threshold",
    "c3b_csynth_latency_lte_threshold",
    "c3b_routed_wns_gte_threshold",
}
REQUIRED_LIVE_SPEC_PLAN_CHECKS = {
    "spec_records_spec_kit_unavailable",
    "spec_records_zcu104",
    "spec_records_q4w_q8a",
    "spec_records_param_knobs",
    "spec_records_a2_a1_c",
    "master_records_selected_paths",
    "choice_records_e_pending",
    "execution_records_xilinx_root",
    "current_docs_have_no_stale_final_evidence_counts",
    "current_docs_have_no_stale_validator_counts",
}
REQUIRED_LIVE_REQ6_CHECKS = {
    "config_defines_HGTXR_TILING_FACTOR",
    "config_defines_HGTXR_PARALLELISM_FACTOR",
    "config_defines_HGTXR_BUS_WIDTH",
    "config_defines_HGTXR_BIT_WIDTH",
    "config_defines_HGTXR_WEIGHT_BIT_WIDTH",
    "config_defines_HGTXR_BUFFER_SIZE",
    "config_defines_HGTXR_FIFO_DEPTH",
    "legal_dense_parallelism_matches_req6_parallelism",
    "csim_tcl_supports_par16_par32",
    "csynth_tcl_supports_par16_par32",
    "sweep_covers_parallelism_factor",
    "parallelism_extension_c3b_par16_validated",
    "parallelism_extension_par32_exploratory_not_default",
    "parallelism_extension_par32_requires_fresh_reports",
}
REQUIRED_FINAL_RUNNER_SUMMARY_CONSISTENCY_CHECKS = {
    "final_runner_remaining_blockers_present",
    "final_runner_blocker_count_matches_list",
    "final_runner_remaining_blocker_detail_count_matches",
    "final_runner_mirrored_artifact_counts_match",
    "final_runner_mirrored_artifact_integrity_present",
    "final_runner_mirrored_artifact_integrity_pass",
    "final_runner_mirrored_artifact_integrity_counts_match",
}
REQUIRED_SOURCE_AUDIT_CONSISTENCY_CHECKS = {
    "third_goal_source_audit_required_count",
    "third_goal_source_audit_source_count",
}
REQUIRED_UNBLOCK_INTAKE_OPERATOR_CONSISTENCY_CHECKS = {
    "final_unblock_intake_next_inputs_match_operator_plan_required_inputs",
    "final_unblock_intake_next_input_paths_match_operator_plan",
    "final_unblock_intake_operator_sequence_contains_expected_steps",
    "final_unblock_intake_operator_sequence_matches_operator_plan_commands",
    "final_unblock_intake_operator_sequence_side_effect_profile",
}
REQUIRED_LIVE_CLOSEOUT_VALIDATION_CHECKS = {
    "packet_status_ready",
    "remaining_blockers_expected",
    "required_artifact_count",
    "required_artifacts_all_pass",
    "required_artifact_hashes_present",
    "missing_required_artifacts_empty",
    "board_package_ready",
    "board_package_preset",
    "board_package_sha",
    "c3b_contract_status",
    "c3b_contract_canonical_path",
    "xr_policy_integrity_required",
    "xr_policy_integrity_fields",
    "xr_legacy_policy_not_accepted",
    "closure_status_blocked",
    "closure_current_not_ready",
    "dry_run_commands_present",
    "combined_unblock_options_present",
    "qkv_uram_successor_not_final_blocker",
    "no_side_effect_safety",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def contains(commands: Any, needle: str) -> bool:
    return isinstance(commands, list) and any(needle in str(command) for command in commands)


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def failed_consistency(contract: dict[str, Any]) -> list[str]:
    failed = contract.get("failed_consistency_checks", [])
    if not isinstance(failed, list):
        return ["<non-list>"]
    return [str(item) for item in failed]


def bootstrap_evidence_failures_only(contract: dict[str, Any]) -> bool:
    failed = set(failed_consistency(contract))
    return bool(failed) and failed <= ALLOWED_BOOTSTRAP_EVIDENCE_FAILURES


def spec_plan_bootstrap_failures_only(spec_plan: dict[str, Any] | None) -> bool:
    checks = (
        spec_plan.get("checks", [])
        if isinstance(spec_plan, dict) and isinstance(spec_plan.get("checks"), list)
        else []
    )
    failed = [
        str(check.get("name", ""))
        for check in checks
        if isinstance(check, dict) and check.get("status") == "fail"
    ]
    return bool(failed) and all(
        name.endswith("_records_live_operator_handoff_validation")
        or name.endswith("_records_live_bundle_validation")
        for name in failed
    )


def contract_safety(contract: dict[str, Any]) -> dict[str, Any]:
    safety = contract.get("safety", {})
    return safety if isinstance(safety, dict) else {}


def evidence_manifest_contract(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    consistency_checks = manifest.get("consistency_checks", [])
    consistency_count = len(consistency_checks) if isinstance(consistency_checks, list) else 0
    return {
        "status": manifest.get("status"),
        "source": "manifest-json",
        "path": str(root / "docs" / "resources" / f"final_evidence_manifest_{DATE_TAG}.json"),
        "required_count": manifest.get("required_count"),
        "present_required_count": manifest.get("present_required_count"),
        "consistency_count": consistency_count,
        "failed_consistency_checks": failed_consistency(manifest),
        "safety": contract_safety(manifest),
    }


def load_evidence_manifest_contract(root: Path) -> tuple[dict[str, Any] | None, str]:
    path = root / "docs" / "resources" / f"final_evidence_manifest_{DATE_TAG}.json"
    if not path.exists():
        return None, str(path)
    try:
        manifest = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{path}: {exc}"
    return evidence_manifest_contract(root, manifest), str(path)


def manifest_consistency_by_name(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    checks = (
        manifest.get("consistency_checks", [])
        if isinstance(manifest, dict) and isinstance(manifest.get("consistency_checks"), list)
        else []
    )
    return {
        str(check.get("name", "")): check
        for check in checks
        if isinstance(check, dict) and check.get("name")
    }


def checks_by_name(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    checks = (
        payload.get("checks", [])
        if isinstance(payload, dict) and isinstance(payload.get("checks"), list)
        else []
    )
    return {
        str(check.get("name", "")): check
        for check in checks
        if isinstance(check, dict) and check.get("name")
    }


def load_live_resource(root: Path, filename: str) -> tuple[dict[str, Any] | None, str]:
    path = root / "docs" / "resources" / filename
    if not path.exists():
        return None, str(path)
    try:
        return load_json(path), str(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{path}: {exc}"


def xr_resolution_contract(resolution: dict[str, Any]) -> dict[str, Any]:
    candidate = resolution.get("candidate", {}) if isinstance(resolution.get("candidate"), dict) else {}
    commands = resolution.get("commands", {}) if isinstance(resolution.get("commands"), dict) else {}
    safety = resolution.get("safety", {}) if isinstance(resolution.get("safety"), dict) else {}
    if candidate or commands or safety:
        return {
            "status": resolution.get("status"),
            "resolution_ready": bool(resolution.get("resolution_ready", False)),
            "approval_required": bool(resolution.get("approval_required", True)),
            "requested_path": resolution.get("requested_path", ""),
            "candidate_status": candidate.get("status", "missing"),
            "candidate_replacement_path": candidate.get("replacement_path", ""),
            "candidate_score": candidate.get("score", 0),
            "replacement_dry_run_command": commands.get("replacement_dry_run", ""),
            "replacement_approve_command": commands.get("replacement_approve", ""),
            "creates_xr_vits_policy": bool(safety.get("creates_xr_vits_policy", False)),
            "writes_canonical_inputs": bool(safety.get("writes_canonical_inputs", False)),
        }
    return {
        "status": resolution.get("status"),
        "resolution_ready": bool(resolution.get("resolution_ready", False)),
        "approval_required": bool(resolution.get("approval_required", True)),
        "requested_path": resolution.get("requested_path", ""),
        "candidate_status": resolution.get("candidate_status", "missing"),
        "candidate_replacement_path": resolution.get("candidate_replacement_path", ""),
        "candidate_score": resolution.get("candidate_score", 0),
        "replacement_dry_run_command": resolution.get("replacement_dry_run_command", ""),
        "replacement_approve_command": resolution.get("replacement_approve_command", ""),
        "creates_xr_vits_policy": bool(resolution.get("creates_xr_vits_policy", False)),
        "writes_canonical_inputs": bool(resolution.get("writes_canonical_inputs", False)),
    }


def intake_pending_blockers(intake: dict[str, Any]) -> set[str]:
    blocker_status = intake.get("blocker_status", {})
    if not isinstance(blocker_status, dict):
        return set()
    return {
        str(blocker)
        for blocker, status in blocker_status.items()
        if isinstance(status, dict) and not bool(status.get("ready_for_active_unblock"))
    }


def intake_candidate_states(intake: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blocker_status = intake.get("blocker_status", {})
    if not isinstance(blocker_status, dict):
        return {}
    states: dict[str, dict[str, Any]] = {}
    for blocker, status in blocker_status.items():
        if not isinstance(status, dict):
            continue
        states[str(blocker)] = {
            "candidate_status": normalized_candidate_status(status.get("candidate_status")),
            "candidate_would_clear": bool(status.get("candidate_would_clear", False)),
            "ready_for_active_unblock": bool(status.get("ready_for_active_unblock", False)),
        }
    return states


def normalized_candidate_status(value: Any) -> str:
    status = str(value)
    return "missing" if status == "not-supplied" else status


def handoff_candidate_states(board: dict[str, Any], xr: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "C3b AXIS/DMA physical smoke result": {
            "candidate_status": normalized_candidate_status(board.get("candidate_status")),
            "candidate_would_clear": bool(board.get("would_clear", False)),
            "ready_for_active_unblock": bool(board.get("would_clear", False)),
        },
        "requested XR-VITs sibling": {
            "candidate_status": normalized_candidate_status(xr.get("candidate_status")),
            "candidate_would_clear": bool(xr.get("would_clear", False)),
            "ready_for_active_unblock": bool(xr.get("would_clear", False)),
        },
    }


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


def validate_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    board = handoff.get("board_smoke", {}) if isinstance(handoff.get("board_smoke"), dict) else {}
    xr = handoff.get("xr_vits", {}) if isinstance(handoff.get("xr_vits"), dict) else {}
    xr_resolution = xr.get("reference_resolution", {}) if isinstance(xr.get("reference_resolution"), dict) else {}
    xr_policy_integrity = (
        xr_resolution.get("policy_integrity", {})
        if isinstance(xr_resolution.get("policy_integrity"), dict)
        else {}
    )
    combined = handoff.get("combined_unblock", {}) if isinstance(handoff.get("combined_unblock"), dict) else {}
    final = handoff.get("final_signoff", {}) if isinstance(handoff.get("final_signoff"), dict) else {}
    evidence_contract = (
        handoff.get("final_evidence_manifest_contract", {})
        if isinstance(handoff.get("final_evidence_manifest_contract"), dict)
        else {}
    )
    evidence_safety = (
        evidence_contract.get("safety", {})
        if isinstance(evidence_contract.get("safety"), dict)
        else {}
    )
    safety = handoff.get("safety", {}) if isinstance(handoff.get("safety"), dict) else {}
    remaining = handoff.get("remaining_blockers", [])
    remaining = remaining if isinstance(remaining, list) else []
    xr_vits_blocked = "requested XR-VITs sibling" in {str(item) for item in remaining}
    xr_resolution_status = str(xr_resolution.get("status", ""))
    xr_resolution_ready = bool(xr_resolution.get("resolution_ready", False))
    root = Path(str(handoff.get("root", Path.cwd()))).resolve()
    live_evidence_contract, live_evidence_detail = load_evidence_manifest_contract(root)
    live_evidence_manifest, live_evidence_manifest_detail = load_live_resource(
        root,
        f"final_evidence_manifest_{DATE_TAG}.json",
    )
    live_xr_resolution, live_xr_resolution_detail = load_live_resource(
        root,
        f"xr_vits_reference_resolution_{DATE_TAG}.json",
    )
    live_intake, live_intake_detail = load_live_resource(root, f"final_unblock_intake_{DATE_TAG}.json")
    live_closeout_packet, live_closeout_packet_detail = load_live_resource(
        root,
        f"final_unblock_closeout_packet_{DATE_TAG}.json",
    )
    live_closeout_validation, live_closeout_validation_detail = load_live_resource(
        root,
        f"final_unblock_closeout_packet_validation_{DATE_TAG}.json",
    )
    live_source_audit, live_source_audit_detail = load_live_resource(
        root,
        "third_goal_source_audit_2026_06_16.json",
    )
    live_resource_policy, live_resource_policy_detail = load_live_resource(
        root,
        f"e2e_resource_policy_audit_{DATE_TAG}.json",
    )
    live_spec_plan, live_spec_plan_detail = load_live_resource(
        root,
        f"spec_plan_conformance_audit_{DATE_TAG}.json",
    )
    live_final_runner, live_final_runner_detail = load_live_resource(
        root,
        f"third_goal_final_signoff_run_{DATE_TAG}.json",
    )
    live_req6_parameterization, live_req6_parameterization_detail = load_live_resource(
        root,
        "req6_parameterization_audit_2026_06_16.json",
    )
    live_current_audit, live_current_audit_detail = load_live_resource(
        root,
        "third_goal_current_audit_2026_06_16.json",
    )
    live_c3b_physical_gate, live_c3b_physical_gate_detail = load_live_resource(
        root,
        "c3b_physical_smoke_gate_audit_2026_06_16.json",
    )
    live_xr_vits_gate, live_xr_vits_gate_detail = load_live_resource(
        root,
        "xr_vits_gate_audit_2026_06_16.json",
    )

    add_check(checks, "status", handoff.get("status") in {"pending-operator-actions", "ready-for-final-signoff"}, str(handoff.get("status")))
    add_check(checks, "root", bool(handoff.get("root")), str(handoff.get("root", "")))
    add_check(checks, "remaining_blockers", isinstance(handoff.get("remaining_blockers"), list), "remaining_blockers is list")
    add_check(checks, "board_ready", board.get("ready") is True, str(board.get("ready")))
    add_check(checks, "board_preset", board.get("preset") == "axis-c3b-mem16", str(board.get("preset", "")))
    add_check(checks, "board_bundle", bool(board.get("bundle_tar")) and bool(board.get("bundle_sha256")), str(board.get("bundle_tar", "")))
    add_check(checks, "board_expected_runtime", board.get("expected_runtime_state") == 2, str(board.get("expected_runtime_state")))
    add_check(checks, "board_expected_output", isinstance(board.get("expected_out_raw"), list) and len(board.get("expected_out_raw", [])) == 6, str(board.get("expected_out_raw", [])))
    add_check(checks, "board_execute_command", contains(board.get("commands"), "--execute") or contains(board.get("commands"), "--execute-zcu104-smoke"), "board execute command present")
    add_check(checks, "board_import_command", contains(board.get("commands"), "--import-c3b-smoke-json") or contains(board.get("commands"), "import_pynq_smoke_result.py"), "board import command present")
    add_check(checks, "board_validate_command", contains(board.get("commands"), "validate_pynq_smoke_result.py"), "board validate command present")
    add_check(checks, "xr_requested_path", str(xr.get("requested_path", "")).endswith("XR-VITs"), str(xr.get("requested_path", "")))
    add_check(checks, "xr_exact_command", contains(xr.get("exact_restore_commands"), "test -d"), "exact restore check present")
    add_check(checks, "xr_replacement_command", contains(xr.get("replacement_approval_commands"), "--approve") and contains(xr.get("replacement_approval_commands"), "--approved-by"), "replacement approval command present")
    add_check(checks, "xr_resolution_present", bool(xr_resolution), "xr_vits.reference_resolution present")
    add_check(
        checks,
        "xr_resolution_status_known",
        xr_resolution_status in {
            "exact-ready",
            "approved-replacement-ready",
            "candidate-ready-needs-approval",
            "blocked",
        },
        xr_resolution_status,
    )
    add_check(
        checks,
        "xr_resolution_matches_blocker",
        (not xr_vits_blocked and xr_resolution_ready)
        or (xr_vits_blocked and not xr_resolution_ready),
        f"blocked={xr_vits_blocked} resolution_ready={xr_resolution_ready}",
    )
    add_check(
        checks,
        "xr_resolution_candidate_score",
        xr_resolution_status != "candidate-ready-needs-approval"
        or (xr_resolution.get("candidate_status") == "pass" and positive_int(xr_resolution.get("candidate_score"))),
        f"candidate_status={xr_resolution.get('candidate_status')} score={xr_resolution.get('candidate_score')}",
    )
    add_check(
        checks,
        "xr_resolution_dry_run_command",
        "--dry-run-xr-vits-replacement" in str(xr_resolution.get("replacement_dry_run_command", ""))
        and "--xr-vits-approved-by" in str(xr_resolution.get("replacement_dry_run_command", "")),
        "resolution dry-run approval command present",
    )
    add_check(
        checks,
        "xr_resolution_approve_command",
        "--approve-xr-vits-replacement" in str(xr_resolution.get("replacement_approve_command", ""))
        and "--xr-vits-approved-by" in str(xr_resolution.get("replacement_approve_command", "")),
        "resolution approval command present",
    )
    add_check(
        checks,
        "xr_resolution_no_policy_write",
        xr_resolution.get("creates_xr_vits_policy") is False,
        str(xr_resolution.get("creates_xr_vits_policy")),
    )
    add_check(
        checks,
        "xr_resolution_no_canonical_write",
        xr_resolution.get("writes_canonical_inputs") is False,
        str(xr_resolution.get("writes_canonical_inputs")),
    )
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
        "xr_legacy_policy_not_accepted",
        xr_policy_integrity.get("legacy_policy_clears_final_signoff") is False,
        str(xr_policy_integrity.get("legacy_policy_clears_final_signoff")),
    )
    stored_policy_status = policy_validation_status(xr_policy_integrity)
    live_policy_validation = load_policy_validation(root, xr_policy_integrity)
    live_policy_status = str(live_policy_validation.get("status", "missing"))
    pending_policy_allowed = (
        xr_vits_blocked
        and xr_resolution_status == "candidate-ready-needs-approval"
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
    embedded_xr_contract = xr_resolution_contract(xr_resolution)
    live_xr_contract = xr_resolution_contract(live_xr_resolution or {})
    live_xr_ready = bool(live_xr_contract.get("resolution_ready", False))
    add_check(
        checks,
        "live_xr_vits_reference_resolution_present",
        isinstance(live_xr_resolution, dict),
        live_xr_resolution_detail,
    )
    add_check(
        checks,
        "xr_resolution_embedded_matches_live",
        isinstance(live_xr_resolution, dict) and embedded_xr_contract == live_xr_contract,
        f"embedded={embedded_xr_contract.get('status')} live={live_xr_contract.get('status')}",
    )
    add_check(
        checks,
        "xr_resolution_live_status_matches_blocker",
        isinstance(live_xr_resolution, dict)
        and ((not xr_vits_blocked and live_xr_ready) or (xr_vits_blocked and not live_xr_ready)),
        f"blocked={xr_vits_blocked} live_resolution_ready={live_xr_ready}",
    )
    add_check(checks, "evidence_manifest_contract_present", bool(evidence_contract), "final_evidence_manifest_contract present")
    add_check(
        checks,
        "live_evidence_manifest_contract_present",
        isinstance(live_evidence_contract, dict),
        live_evidence_detail,
    )
    add_check(
        checks,
        "evidence_manifest_contract_matches_live",
        isinstance(live_evidence_contract, dict) and evidence_contract == live_evidence_contract,
        f"embedded={evidence_contract.get('consistency_count')} live={(live_evidence_contract or {}).get('consistency_count')}",
    )
    add_check(
        checks,
        "evidence_manifest_contract_pass",
        evidence_contract.get("status") == "pass" or bootstrap_evidence_failures_only(evidence_contract),
        str(evidence_contract.get("status")),
    )
    add_check(
        checks,
        "evidence_manifest_required_complete",
        positive_int(evidence_contract.get("required_count"))
        and int(evidence_contract.get("required_count")) >= MIN_EVIDENCE_REQUIRED_COUNT
        and evidence_contract.get("present_required_count") == evidence_contract.get("required_count"),
        f"{evidence_contract.get('present_required_count')}/{evidence_contract.get('required_count')}",
    )
    add_check(
        checks,
        "evidence_manifest_consistency_count",
        positive_int(evidence_contract.get("consistency_count"))
        and int(evidence_contract.get("consistency_count")) >= MIN_EVIDENCE_CONSISTENCY_COUNT,
        str(evidence_contract.get("consistency_count")),
    )
    failed_consistency = evidence_contract.get("failed_consistency_checks", [])
    add_check(
        checks,
        "evidence_manifest_failed_consistency_zero",
        isinstance(failed_consistency, list)
        and (len(failed_consistency) == 0 or set(map(str, failed_consistency)) <= ALLOWED_BOOTSTRAP_EVIDENCE_FAILURES),
        str(failed_consistency),
    )
    live_consistency = manifest_consistency_by_name(live_evidence_manifest)
    missing_req5_checks = sorted(REQUIRED_REQ5_CONSISTENCY_CHECKS - set(live_consistency))
    failing_req5_checks = sorted(
        name
        for name in REQUIRED_REQ5_CONSISTENCY_CHECKS
        if live_consistency.get(name, {}).get("status") != "pass"
    )
    add_check(
        checks,
        "live_evidence_manifest_req5_checks_present",
        isinstance(live_evidence_manifest, dict) and missing_req5_checks == [],
        f"{live_evidence_manifest_detail} missing={missing_req5_checks}",
    )
    add_check(
        checks,
        "live_evidence_manifest_req5_checks_pass",
        isinstance(live_evidence_manifest, dict) and failing_req5_checks == [],
        f"failing={failing_req5_checks}",
    )
    missing_cyclic_resource_checks = sorted(REQUIRED_CYCLIC_RESOURCE_CONSISTENCY_CHECKS - set(live_consistency))
    failing_cyclic_resource_checks = sorted(
        name
        for name in REQUIRED_CYCLIC_RESOURCE_CONSISTENCY_CHECKS
        if live_consistency.get(name, {}).get("status") != "pass"
    )
    add_check(
        checks,
        "live_evidence_manifest_cyclic_resource_checks_present",
        isinstance(live_evidence_manifest, dict) and missing_cyclic_resource_checks == [],
        f"{live_evidence_manifest_detail} missing={missing_cyclic_resource_checks}",
    )
    add_check(
        checks,
        "live_evidence_manifest_cyclic_resource_checks_pass",
        isinstance(live_evidence_manifest, dict) and failing_cyclic_resource_checks == [],
        f"failing={failing_cyclic_resource_checks}",
    )
    missing_c3b_threshold_checks = sorted(REQUIRED_C3B_THRESHOLD_CONSISTENCY_CHECKS - set(live_consistency))
    failing_c3b_threshold_checks = sorted(
        name
        for name in REQUIRED_C3B_THRESHOLD_CONSISTENCY_CHECKS
        if live_consistency.get(name, {}).get("status") != "pass"
    )
    add_check(
        checks,
        "live_evidence_manifest_c3b_threshold_checks_present",
        isinstance(live_evidence_manifest, dict) and missing_c3b_threshold_checks == [],
        f"{live_evidence_manifest_detail} missing={missing_c3b_threshold_checks}",
    )
    add_check(
        checks,
        "live_evidence_manifest_c3b_threshold_checks_pass",
        isinstance(live_evidence_manifest, dict) and failing_c3b_threshold_checks == [],
        f"failing={failing_c3b_threshold_checks}",
    )
    resource_policy_checks = checks_by_name(live_resource_policy)
    missing_resource_policy_checks = sorted(REQUIRED_LIVE_RESOURCE_POLICY_CHECKS - set(resource_policy_checks))
    failing_resource_policy_checks = sorted(
        name
        for name in REQUIRED_LIVE_RESOURCE_POLICY_CHECKS
        if resource_policy_checks.get(name, {}).get("status") != "pass"
    )
    add_check(
        checks,
        "live_resource_policy_present",
        isinstance(live_resource_policy, dict),
        live_resource_policy_detail,
    )
    add_check(
        checks,
        "live_resource_policy_pass",
        isinstance(live_resource_policy, dict) and live_resource_policy.get("status") == "pass",
        str(live_resource_policy.get("status") if isinstance(live_resource_policy, dict) else "missing"),
    )
    add_check(
        checks,
        "live_resource_policy_fail_zero",
        isinstance(live_resource_policy, dict) and live_resource_policy.get("fail_count") == 0,
        str(live_resource_policy.get("fail_count") if isinstance(live_resource_policy, dict) else "missing"),
    )
    add_check(
        checks,
        "live_resource_policy_check_count",
        isinstance(live_resource_policy, dict)
        and positive_int(live_resource_policy.get("check_count"))
        and int(live_resource_policy.get("check_count")) >= MIN_RESOURCE_POLICY_CHECK_COUNT,
        str(live_resource_policy.get("check_count") if isinstance(live_resource_policy, dict) else "missing"),
    )
    add_check(
        checks,
        "live_resource_policy_core_checks_present",
        isinstance(live_resource_policy, dict) and missing_resource_policy_checks == [],
        f"missing={missing_resource_policy_checks}",
    )
    add_check(
        checks,
        "live_resource_policy_core_checks_pass",
        isinstance(live_resource_policy, dict) and failing_resource_policy_checks == [],
        f"failing={failing_resource_policy_checks}",
    )
    spec_plan_checks = checks_by_name(live_spec_plan)
    missing_spec_plan_checks = sorted(REQUIRED_LIVE_SPEC_PLAN_CHECKS - set(spec_plan_checks))
    failing_spec_plan_checks = sorted(
        name
        for name in REQUIRED_LIVE_SPEC_PLAN_CHECKS
        if spec_plan_checks.get(name, {}).get("status") != "pass"
    )
    spec_observed = live_spec_plan.get("observed", {}) if isinstance(live_spec_plan, dict) else {}
    spec_safety = live_spec_plan.get("safety", {}) if isinstance(live_spec_plan, dict) else {}
    add_check(
        checks,
        "live_spec_plan_present",
        isinstance(live_spec_plan, dict),
        live_spec_plan_detail,
    )
    add_check(
        checks,
        "live_spec_plan_pass",
        isinstance(live_spec_plan, dict)
        and (live_spec_plan.get("status") == "pass" or spec_plan_bootstrap_failures_only(live_spec_plan)),
        str(live_spec_plan.get("status") if isinstance(live_spec_plan, dict) else "missing"),
    )
    add_check(
        checks,
        "live_spec_plan_check_count",
        isinstance(live_spec_plan, dict)
        and positive_int(live_spec_plan.get("check_count"))
        and int(live_spec_plan.get("check_count")) >= MIN_SPEC_PLAN_CHECK_COUNT,
        str(live_spec_plan.get("check_count") if isinstance(live_spec_plan, dict) else "missing"),
    )
    add_check(
        checks,
        "live_spec_plan_core_checks_present",
        isinstance(live_spec_plan, dict) and missing_spec_plan_checks == [],
        f"missing={missing_spec_plan_checks}",
    )
    add_check(
        checks,
        "live_spec_plan_core_checks_pass",
        isinstance(live_spec_plan, dict) and failing_spec_plan_checks == [],
        f"failing={failing_spec_plan_checks}",
    )
    add_check(
        checks,
        "live_spec_plan_observed_counts_current",
        isinstance(spec_observed, dict)
        and spec_observed.get("evidence_required") == MIN_EVIDENCE_REQUIRED_COUNT
        and spec_observed.get("evidence_consistency_count") == MIN_EVIDENCE_CONSISTENCY_COUNT
        and spec_observed.get("source_required_count") == MIN_SOURCE_AUDIT_REQUIRED_COUNT
        and positive_int(spec_observed.get("source_count"))
        and int(spec_observed.get("source_count")) >= MIN_SOURCE_AUDIT_SOURCE_COUNT,
        str(spec_observed),
    )
    add_check(
        checks,
        "live_spec_plan_safety_no_side_effects",
        isinstance(spec_safety, dict)
        and spec_safety.get("creates_board_result") is False
        and spec_safety.get("creates_xr_vits_policy") is False
        and spec_safety.get("executes_hls_or_vivado") is False
        and spec_safety.get("writes_canonical_inputs") is False,
        str(spec_safety),
    )
    req6_checks = checks_by_name(live_req6_parameterization)
    missing_req6_checks = sorted(REQUIRED_LIVE_REQ6_CHECKS - set(req6_checks))
    failing_req6_checks = sorted(
        name
        for name in REQUIRED_LIVE_REQ6_CHECKS
        if req6_checks.get(name, {}).get("status") != "pass"
    )
    req6_knobs = (
        live_req6_parameterization.get("knobs", {}).get("config_macros", {})
        if isinstance(live_req6_parameterization, dict)
        and isinstance(live_req6_parameterization.get("knobs"), dict)
        and isinstance(live_req6_parameterization["knobs"].get("config_macros"), dict)
        else {}
    )
    add_check(
        checks,
        "live_req6_parameterization_present",
        isinstance(live_req6_parameterization, dict),
        live_req6_parameterization_detail,
    )
    add_check(
        checks,
        "live_req6_parameterization_pass",
        isinstance(live_req6_parameterization, dict) and live_req6_parameterization.get("status") == "pass",
        str(live_req6_parameterization.get("status") if isinstance(live_req6_parameterization, dict) else "missing"),
    )
    add_check(
        checks,
        "live_req6_parameterization_fail_zero",
        isinstance(live_req6_parameterization, dict) and live_req6_parameterization.get("fail_count") == 0,
        str(live_req6_parameterization.get("fail_count") if isinstance(live_req6_parameterization, dict) else "missing"),
    )
    add_check(
        checks,
        "live_req6_parameterization_check_count",
        isinstance(live_req6_parameterization, dict)
        and positive_int(live_req6_parameterization.get("check_count"))
        and int(live_req6_parameterization.get("check_count")) >= MIN_REQ6_PARAMETERIZATION_CHECK_COUNT,
        str(live_req6_parameterization.get("check_count") if isinstance(live_req6_parameterization, dict) else "missing"),
    )
    add_check(
        checks,
        "live_req6_parameterization_knobs",
        req6_knobs == EXPECTED_REQ6_KNOBS,
        str(req6_knobs),
    )
    add_check(
        checks,
        "live_req6_parameterization_core_checks_present",
        isinstance(live_req6_parameterization, dict) and missing_req6_checks == [],
        f"missing={missing_req6_checks}",
    )
    add_check(
        checks,
        "live_req6_parameterization_core_checks_pass",
        isinstance(live_req6_parameterization, dict) and failing_req6_checks == [],
        f"failing={failing_req6_checks}",
    )
    current_summary = live_current_audit.get("summary", {}) if isinstance(live_current_audit, dict) else {}
    current_external_blockers = (
        {str(name) for name in live_current_audit.get("external_blocker_names", [])}
        if isinstance(live_current_audit, dict) and isinstance(live_current_audit.get("external_blocker_names"), list)
        else set()
    )
    current_paths = (
        live_current_audit.get("blocked_external_input_paths_by_blocker", {})
        if isinstance(live_current_audit, dict)
        and isinstance(live_current_audit.get("blocked_external_input_paths_by_blocker"), dict)
        else {}
    )
    current_details = (
        live_current_audit.get("remaining_external_input_details", [])
        if isinstance(live_current_audit, dict)
        and isinstance(live_current_audit.get("remaining_external_input_details"), list)
        else []
    )
    current_xr_policy = (
        live_current_audit.get("xr_vits_policy_integrity", {})
        if isinstance(live_current_audit, dict)
        and isinstance(live_current_audit.get("xr_vits_policy_integrity"), dict)
        else {}
    )
    current_qkv = (
        live_current_audit.get("qkv_uram_runner", {})
        if isinstance(live_current_audit, dict)
        and isinstance(live_current_audit.get("qkv_uram_runner"), dict)
        else {}
    )
    current_safety = (
        live_current_audit.get("safety", {})
        if isinstance(live_current_audit, dict)
        and isinstance(live_current_audit.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "live_current_audit_present",
        isinstance(live_current_audit, dict),
        live_current_audit_detail,
    )
    add_check(
        checks,
        "live_current_audit_status_blocked_external",
        isinstance(live_current_audit, dict) and live_current_audit.get("status") == "blocked-external",
        str(live_current_audit.get("status") if isinstance(live_current_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "live_current_audit_summary_counts",
        isinstance(current_summary, dict)
        and all(current_summary.get(key) == value for key, value in EXPECTED_CURRENT_AUDIT_SUMMARY.items()),
        str(current_summary),
    )
    add_check(
        checks,
        "live_current_audit_external_blockers",
        current_external_blockers == EXPECTED_FINAL_RUNNER_BLOCKERS,
        str(sorted(current_external_blockers)),
    )
    add_check(
        checks,
        "live_current_audit_blocker_paths",
        str(current_paths.get("C3b AXIS/DMA physical smoke result", "")).endswith(
            "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
        )
        and str(current_paths.get("requested XR-VITs sibling", "")).endswith("XR-VITs"),
        str(current_paths),
    )
    add_check(
        checks,
        "live_current_audit_external_detail_count",
        len(current_details) >= len(EXPECTED_FINAL_RUNNER_BLOCKERS)
        and all(detail.get("status") == "missing" for detail in current_details if isinstance(detail, dict)),
        str(current_details),
    )
    add_check(
        checks,
        "live_current_audit_xr_policy_integrity",
        current_xr_policy.get("consistent") is True
        and current_xr_policy.get("operator_handoff_policy_check_count") == 9,
        str(current_xr_policy),
    )
    add_check(
        checks,
        "live_current_audit_qkv_optional_state",
        current_qkv.get("qkv_uram_required_for_final_signoff") is False
        and current_qkv.get("qkv_uram_import_status") == "skipped"
        and current_qkv.get("qkv_uram_remote_status") == "skipped"
        and current_qkv.get("qkv_uram_smoke_discovery_status") == "missing",
        str(current_qkv),
    )
    add_check(
        checks,
        "live_current_audit_safety_no_side_effects",
        current_safety.get("creates_board_result") is False
        and current_safety.get("creates_xr_vits_policy") is False
        and current_safety.get("executes_network") is False
        and current_safety.get("writes_canonical_inputs") is False,
        str(current_safety),
    )
    c3b_gate_canonical = (
        live_c3b_physical_gate.get("canonical_result", {})
        if isinstance(live_c3b_physical_gate, dict)
        and isinstance(live_c3b_physical_gate.get("canonical_result"), dict)
        else {}
    )
    c3b_gate_validation = (
        live_c3b_physical_gate.get("canonical_validation", {})
        if isinstance(live_c3b_physical_gate, dict)
        and isinstance(live_c3b_physical_gate.get("canonical_validation"), dict)
        else {}
    )
    c3b_gate_bundle = (
        live_c3b_physical_gate.get("bundle", {})
        if isinstance(live_c3b_physical_gate, dict)
        and isinstance(live_c3b_physical_gate.get("bundle"), dict)
        else {}
    )
    c3b_gate_session = (
        live_c3b_physical_gate.get("session", {})
        if isinstance(live_c3b_physical_gate, dict)
        and isinstance(live_c3b_physical_gate.get("session"), dict)
        else {}
    )
    c3b_gate_safety = (
        live_c3b_physical_gate.get("safety", {})
        if isinstance(live_c3b_physical_gate, dict)
        and isinstance(live_c3b_physical_gate.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "live_c3b_physical_gate_present",
        isinstance(live_c3b_physical_gate, dict),
        live_c3b_physical_gate_detail,
    )
    add_check(
        checks,
        "live_c3b_physical_gate_status_known",
        isinstance(live_c3b_physical_gate, dict)
        and live_c3b_physical_gate.get("status") in {"pass", "blocked_missing_canonical_physical_smoke_result"},
        str(live_c3b_physical_gate.get("status") if isinstance(live_c3b_physical_gate, dict) else "missing"),
    )
    add_check(
        checks,
        "live_c3b_physical_gate_current_blocker_contract",
        isinstance(live_c3b_physical_gate, dict)
        and live_c3b_physical_gate.get("ready_for_board") is True
        and live_c3b_physical_gate.get("physical_smoke_pass") is False
        and "C3b AXIS/DMA physical smoke result"
        in {str(name) for name in live_c3b_physical_gate.get("remaining_blockers", [])}
        and c3b_gate_canonical.get("status") == "missing"
        and str(c3b_gate_canonical.get("path", "")).endswith(
            "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
        ),
        f"status={live_c3b_physical_gate.get('status') if isinstance(live_c3b_physical_gate, dict) else 'missing'} canonical={c3b_gate_canonical}",
    )
    add_check(
        checks,
        "live_c3b_physical_gate_validation_path",
        c3b_gate_validation.get("status") in {"missing", "pass", "fail"}
        and str(c3b_gate_validation.get("path", "")).endswith(
            "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke_validation.json"
        ),
        str(c3b_gate_validation),
    )
    add_check(
        checks,
        "live_c3b_physical_gate_bundle_session_ready",
        c3b_gate_bundle.get("status") == "pass"
        and c3b_gate_session.get("status") == "pass"
        and c3b_gate_bundle.get("expected_runtime_state") == 2
        and isinstance(c3b_gate_bundle.get("expected_out_raw"), list)
        and len(c3b_gate_bundle.get("expected_out_raw", [])) == 6,
        f"bundle={c3b_gate_bundle} session={c3b_gate_session}",
    )
    add_check(
        checks,
        "live_c3b_physical_gate_safety_no_side_effects",
        c3b_gate_safety.get("creates_board_result") is False
        and c3b_gate_safety.get("executes_commands") is False
        and c3b_gate_safety.get("executes_network") is False
        and c3b_gate_safety.get("writes_canonical_inputs") is False,
        str(c3b_gate_safety),
    )
    xr_gate_current = (
        live_xr_vits_gate.get("current_gate", {})
        if isinstance(live_xr_vits_gate, dict)
        and isinstance(live_xr_vits_gate.get("current_gate"), dict)
        else {}
    )
    xr_gate_exact = (
        live_xr_vits_gate.get("exact", {})
        if isinstance(live_xr_vits_gate, dict)
        and isinstance(live_xr_vits_gate.get("exact"), dict)
        else {}
    )
    xr_gate_candidate = (
        live_xr_vits_gate.get("replacement_candidate", {})
        if isinstance(live_xr_vits_gate, dict)
        and isinstance(live_xr_vits_gate.get("replacement_candidate"), dict)
        else {}
    )
    xr_gate_candidate_audit = (
        live_xr_vits_gate.get("candidate_audit", {})
        if isinstance(live_xr_vits_gate, dict)
        and isinstance(live_xr_vits_gate.get("candidate_audit"), dict)
        else {}
    )
    xr_gate_policy = (
        live_xr_vits_gate.get("active_policy_summary", {})
        if isinstance(live_xr_vits_gate, dict)
        and isinstance(live_xr_vits_gate.get("active_policy_summary"), dict)
        else {}
    )
    xr_gate_safety = (
        live_xr_vits_gate.get("safety", {})
        if isinstance(live_xr_vits_gate, dict)
        and isinstance(live_xr_vits_gate.get("safety"), dict)
        else {}
    )
    xr_gate_recommendation = (
        xr_gate_candidate_audit.get("recommendation", {})
        if isinstance(xr_gate_candidate_audit.get("recommendation"), dict)
        else {}
    )
    add_check(
        checks,
        "live_xr_vits_gate_present",
        isinstance(live_xr_vits_gate, dict),
        live_xr_vits_gate_detail,
    )
    add_check(
        checks,
        "live_xr_vits_gate_status_blocked_or_clear",
        isinstance(live_xr_vits_gate, dict)
        and live_xr_vits_gate.get("status") in {"blocked", "pass-exact", "pass-replacement-policy"},
        str(live_xr_vits_gate.get("status") if isinstance(live_xr_vits_gate, dict) else "missing"),
    )
    add_check(
        checks,
        "live_xr_vits_gate_current_blocker_contract",
        isinstance(live_xr_vits_gate, dict)
        and live_xr_vits_gate.get("resolution_mode") == "candidate-ready-needs-approval"
        and "requested XR-VITs sibling" in {str(name) for name in live_xr_vits_gate.get("remaining_blockers", [])}
        and xr_gate_current.get("would_clear") is False
        and xr_gate_current.get("status") == "missing"
        and xr_gate_exact.get("exists") is False
        and str(xr_gate_exact.get("path", "")).endswith("XR-VITs")
        and xr_gate_policy.get("status") == "missing"
        and xr_gate_policy.get("would_clear") is False,
        f"current={xr_gate_current} exact={xr_gate_exact} policy={xr_gate_policy}",
    )
    add_check(
        checks,
        "live_xr_vits_gate_candidate_contract",
        xr_gate_candidate.get("exists") is True
        and str(xr_gate_candidate.get("path", "")).endswith("XR_Accel")
        and xr_gate_candidate_audit.get("status") == "candidate-found"
        and xr_gate_candidate_audit.get("recommendation_matches") is True
        and positive_int(xr_gate_recommendation.get("score")),
        f"candidate={xr_gate_candidate} audit={xr_gate_candidate_audit}",
    )
    add_check(
        checks,
        "live_xr_vits_gate_safety_no_side_effects",
        xr_gate_safety.get("creates_xr_vits_policy") is False
        and xr_gate_safety.get("executes_commands") is False
        and xr_gate_safety.get("executes_network") is False
        and xr_gate_safety.get("writes_canonical_inputs") is False,
        str(xr_gate_safety),
    )
    closeout_validation_checks = checks_by_name(live_closeout_validation)
    missing_closeout_validation_checks = sorted(
        REQUIRED_LIVE_CLOSEOUT_VALIDATION_CHECKS - set(closeout_validation_checks)
    )
    failing_closeout_validation_checks = sorted(
        name
        for name in REQUIRED_LIVE_CLOSEOUT_VALIDATION_CHECKS
        if closeout_validation_checks.get(name, {}).get("status") != "pass"
    )
    closeout_validation_safety = (
        live_closeout_validation.get("safety", {})
        if isinstance(live_closeout_validation, dict)
        and isinstance(live_closeout_validation.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "live_closeout_validation_present",
        isinstance(live_closeout_validation, dict),
        live_closeout_validation_detail,
    )
    add_check(
        checks,
        "live_closeout_validation_pass",
        isinstance(live_closeout_validation, dict) and live_closeout_validation.get("status") == "pass",
        str(live_closeout_validation.get("status") if isinstance(live_closeout_validation, dict) else "missing"),
    )
    add_check(
        checks,
        "live_closeout_validation_fail_zero",
        isinstance(live_closeout_validation, dict) and live_closeout_validation.get("fail_count") == 0,
        str(live_closeout_validation.get("fail_count") if isinstance(live_closeout_validation, dict) else "missing"),
    )
    add_check(
        checks,
        "live_closeout_validation_check_count",
        isinstance(live_closeout_validation, dict)
        and positive_int(live_closeout_validation.get("check_count"))
        and int(live_closeout_validation.get("check_count")) >= MIN_CLOSEOUT_VALIDATION_CHECK_COUNT,
        str(live_closeout_validation.get("check_count") if isinstance(live_closeout_validation, dict) else "missing"),
    )
    add_check(
        checks,
        "live_closeout_validation_core_checks_present",
        isinstance(live_closeout_validation, dict) and missing_closeout_validation_checks == [],
        f"missing={missing_closeout_validation_checks}",
    )
    add_check(
        checks,
        "live_closeout_validation_core_checks_pass",
        isinstance(live_closeout_validation, dict) and failing_closeout_validation_checks == [],
        f"failing={failing_closeout_validation_checks}",
    )
    add_check(
        checks,
        "live_closeout_validation_safety_no_side_effects",
        closeout_validation_safety.get("creates_board_result") is False
        and closeout_validation_safety.get("creates_xr_vits_policy") is False
        and closeout_validation_safety.get("executes_commands") is False
        and closeout_validation_safety.get("executes_network") is False
        and closeout_validation_safety.get("writes_canonical_inputs") is False,
        str(closeout_validation_safety),
    )
    closeout_packet_blockers = (
        {str(name) for name in live_closeout_packet.get("remaining_blockers", [])}
        if isinstance(live_closeout_packet, dict)
        and isinstance(live_closeout_packet.get("remaining_blockers"), list)
        else set()
    )
    closeout_board = (
        live_closeout_packet.get("board_package", {})
        if isinstance(live_closeout_packet, dict)
        and isinstance(live_closeout_packet.get("board_package"), dict)
        else {}
    )
    closeout_c3b = (
        live_closeout_packet.get("c3b_smoke_contract", {})
        if isinstance(live_closeout_packet, dict)
        and isinstance(live_closeout_packet.get("c3b_smoke_contract"), dict)
        else {}
    )
    closeout_xr = (
        live_closeout_packet.get("xr_vits_resolution", {})
        if isinstance(live_closeout_packet, dict)
        and isinstance(live_closeout_packet.get("xr_vits_resolution"), dict)
        else {}
    )
    closeout_xr_policy = (
        closeout_xr.get("policy_integrity", {})
        if isinstance(closeout_xr.get("policy_integrity"), dict)
        else {}
    )
    closeout_qkv = (
        live_closeout_packet.get("qkv_uram_successor_gate", {})
        if isinstance(live_closeout_packet, dict)
        and isinstance(live_closeout_packet.get("qkv_uram_successor_gate"), dict)
        else {}
    )
    closeout_commands = (
        live_closeout_packet.get("operator_commands", {})
        if isinstance(live_closeout_packet, dict)
        and isinstance(live_closeout_packet.get("operator_commands"), dict)
        else {}
    )
    closeout_qkv_commands = (
        closeout_commands.get("qkv_uram_successor", [])
        if isinstance(closeout_commands.get("qkv_uram_successor"), list)
        else []
    )
    closeout_packet_safety = (
        live_closeout_packet.get("safety", {})
        if isinstance(live_closeout_packet, dict)
        and isinstance(live_closeout_packet.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "live_closeout_packet_present",
        isinstance(live_closeout_packet, dict),
        live_closeout_packet_detail,
    )
    add_check(
        checks,
        "live_closeout_packet_ready",
        isinstance(live_closeout_packet, dict)
        and live_closeout_packet.get("status") == "ready-for-operator-unblock",
        str(live_closeout_packet.get("status") if isinstance(live_closeout_packet, dict) else "missing"),
    )
    add_check(
        checks,
        "live_closeout_packet_blockers",
        closeout_packet_blockers == EXPECTED_FINAL_RUNNER_BLOCKERS,
        str(sorted(closeout_packet_blockers)),
    )
    add_check(
        checks,
        "live_closeout_packet_board_package",
        closeout_board.get("status") == "ready-for-board"
        and closeout_board.get("preset") == "axis-c3b-mem16"
        and closeout_board.get("expected_runtime_state") == 2
        and bool(closeout_board.get("tar"))
        and bool(closeout_board.get("tar_sha256")),
        str(closeout_board),
    )
    add_check(
        checks,
        "live_closeout_packet_c3b_contract",
        closeout_c3b.get("status") == "pass"
        and closeout_c3b.get("preset") == "axis-c3b-mem16"
        and str(closeout_c3b.get("canonical_result_path", "")).endswith(
            "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
        )
        and "validate_pynq_smoke_result.py" in str(closeout_c3b.get("validate_command", "")),
        str(closeout_c3b),
    )
    add_check(
        checks,
        "live_closeout_packet_xr_policy_integrity",
        closeout_xr.get("status") in {"candidate-ready-needs-approval", "blocked", "exact-ready", "approved-replacement-ready"}
        and closeout_xr_policy.get("required") is True
        and closeout_xr_policy.get("legacy_policy_clears_final_signoff") is False
        and closeout_xr_policy.get("policy_exists") is False,
        str(closeout_xr_policy),
    )
    add_check(
        checks,
        "live_closeout_packet_qkv_optional_commands",
        closeout_qkv.get("required_for_final_signoff") is False
        and closeout_qkv.get("physical_smoke_status") == "not_captured"
        and any("--execute-qkv-uram-smoke" in str(command) for command in closeout_qkv_commands)
        and any("--dry-run-import-qkv-uram-smoke" in str(command) for command in closeout_qkv_commands),
        f"qkv={closeout_qkv} commands={closeout_qkv_commands}",
    )
    add_check(
        checks,
        "live_closeout_packet_safety_no_side_effects",
        closeout_packet_safety.get("creates_board_result") is False
        and closeout_packet_safety.get("creates_xr_vits_policy") is False
        and closeout_packet_safety.get("executes_commands") is False
        and closeout_packet_safety.get("executes_network") is False
        and closeout_packet_safety.get("writes_canonical_inputs") is False,
        str(closeout_packet_safety),
    )
    runner_blockers = set()
    if isinstance(live_final_runner, dict) and isinstance(live_final_runner.get("remaining_blockers"), list):
        runner_blockers = {str(name) for name in live_final_runner.get("remaining_blockers", [])}
    runner_details = live_final_runner.get("remaining_blocker_details") if isinstance(live_final_runner, dict) else None
    runner_detail_count = len(runner_details) if isinstance(runner_details, (list, dict)) else None
    mirrored_artifacts = (
        live_final_runner.get("mirrored_artifacts", [])
        if isinstance(live_final_runner, dict) and isinstance(live_final_runner.get("mirrored_artifacts"), list)
        else []
    )
    mirror_integrity = (
        live_final_runner.get("mirrored_artifact_integrity", [])
        if isinstance(live_final_runner, dict) and isinstance(live_final_runner.get("mirrored_artifact_integrity"), list)
        else []
    )
    add_check(
        checks,
        "live_final_runner_present",
        isinstance(live_final_runner, dict),
        live_final_runner_detail,
    )
    add_check(
        checks,
        "live_final_runner_status_blocked",
        isinstance(live_final_runner, dict) and live_final_runner.get("status") == "blocked",
        str(live_final_runner.get("status") if isinstance(live_final_runner, dict) else "missing"),
    )
    add_check(
        checks,
        "live_final_runner_blocker_count",
        isinstance(live_final_runner, dict)
        and live_final_runner.get("blocker_count") == len(EXPECTED_FINAL_RUNNER_BLOCKERS)
        and live_final_runner.get("blocker_count") == len(runner_blockers),
        f"count={live_final_runner.get('blocker_count') if isinstance(live_final_runner, dict) else 'missing'} blockers={sorted(runner_blockers)}",
    )
    add_check(
        checks,
        "live_final_runner_blocker_names",
        isinstance(live_final_runner, dict) and runner_blockers == EXPECTED_FINAL_RUNNER_BLOCKERS,
        str(sorted(runner_blockers)),
    )
    add_check(
        checks,
        "live_final_runner_detail_count",
        isinstance(live_final_runner, dict)
        and live_final_runner.get("remaining_blocker_detail_count") == len(EXPECTED_FINAL_RUNNER_BLOCKERS)
        and runner_detail_count == len(EXPECTED_FINAL_RUNNER_BLOCKERS),
        f"count={live_final_runner.get('remaining_blocker_detail_count') if isinstance(live_final_runner, dict) else 'missing'} details={runner_detail_count}",
    )
    add_check(
        checks,
        "live_final_runner_mirror_counts",
        isinstance(live_final_runner, dict)
        and live_final_runner.get("mirrored_artifact_count") == len(mirrored_artifacts)
        and live_final_runner.get("mirrored_artifact_unique_count") == len(set(str(path) for path in mirrored_artifacts))
        and live_final_runner.get("mirrored_artifact_duplicate_count") == 0,
        (
            f"count={live_final_runner.get('mirrored_artifact_count') if isinstance(live_final_runner, dict) else 'missing'} "
            f"unique={live_final_runner.get('mirrored_artifact_unique_count') if isinstance(live_final_runner, dict) else 'missing'} "
            f"duplicates={live_final_runner.get('mirrored_artifact_duplicate_count') if isinstance(live_final_runner, dict) else 'missing'}"
        ),
    )
    add_check(
        checks,
        "live_final_runner_mirror_integrity_pass",
        isinstance(live_final_runner, dict)
        and live_final_runner.get("mirrored_artifact_integrity_status") == "pass"
        and live_final_runner.get("mirrored_artifact_integrity_fail_count") == 0
        and live_final_runner.get("mirrored_artifact_integrity_failures") == [],
        (
            f"status={live_final_runner.get('mirrored_artifact_integrity_status') if isinstance(live_final_runner, dict) else 'missing'} "
            f"fail_count={live_final_runner.get('mirrored_artifact_integrity_fail_count') if isinstance(live_final_runner, dict) else 'missing'}"
        ),
    )
    add_check(
        checks,
        "live_final_runner_mirror_integrity_counts",
        isinstance(live_final_runner, dict)
        and live_final_runner.get("mirrored_artifact_integrity_count") == len(mirrored_artifacts)
        and live_final_runner.get("mirrored_artifact_integrity_count") == len(mirror_integrity)
        and isinstance(live_final_runner.get("mirrored_artifact_integrity_checked_count"), int)
        and isinstance(live_final_runner.get("mirrored_artifact_integrity_excluded_count"), int)
        and live_final_runner.get("mirrored_artifact_integrity_checked_count")
        + live_final_runner.get("mirrored_artifact_integrity_excluded_count")
        == live_final_runner.get("mirrored_artifact_integrity_count"),
        (
            f"mirrored={len(mirrored_artifacts)} "
            f"integrity={live_final_runner.get('mirrored_artifact_integrity_count') if isinstance(live_final_runner, dict) else 'missing'}"
        ),
    )
    missing_runner_summary_checks = sorted(REQUIRED_FINAL_RUNNER_SUMMARY_CONSISTENCY_CHECKS - set(live_consistency))
    failing_runner_summary_checks = sorted(
        name
        for name in REQUIRED_FINAL_RUNNER_SUMMARY_CONSISTENCY_CHECKS
        if live_consistency.get(name, {}).get("status") != "pass"
    )
    add_check(
        checks,
        "live_evidence_manifest_runner_summary_checks_present",
        isinstance(live_evidence_manifest, dict) and missing_runner_summary_checks == [],
        f"{live_evidence_manifest_detail} missing={missing_runner_summary_checks}",
    )
    add_check(
        checks,
        "live_evidence_manifest_runner_summary_checks_pass",
        isinstance(live_evidence_manifest, dict) and failing_runner_summary_checks == [],
        f"failing={failing_runner_summary_checks}",
    )
    missing_source_audit_checks = sorted(REQUIRED_SOURCE_AUDIT_CONSISTENCY_CHECKS - set(live_consistency))
    failing_source_audit_checks = sorted(
        name
        for name in REQUIRED_SOURCE_AUDIT_CONSISTENCY_CHECKS
        if live_consistency.get(name, {}).get("status") != "pass"
    )
    add_check(
        checks,
        "live_evidence_manifest_source_audit_checks_present",
        isinstance(live_evidence_manifest, dict) and missing_source_audit_checks == [],
        f"{live_evidence_manifest_detail} missing={missing_source_audit_checks}",
    )
    add_check(
        checks,
        "live_evidence_manifest_source_audit_checks_pass",
        isinstance(live_evidence_manifest, dict) and failing_source_audit_checks == [],
        f"failing={failing_source_audit_checks}",
    )
    missing_unblock_operator_checks = sorted(REQUIRED_UNBLOCK_INTAKE_OPERATOR_CONSISTENCY_CHECKS - set(live_consistency))
    failing_unblock_operator_checks = sorted(
        name
        for name in REQUIRED_UNBLOCK_INTAKE_OPERATOR_CONSISTENCY_CHECKS
        if live_consistency.get(name, {}).get("status") != "pass"
    )
    add_check(
        checks,
        "live_evidence_manifest_unblock_intake_operator_checks_present",
        isinstance(live_evidence_manifest, dict) and missing_unblock_operator_checks == [],
        f"{live_evidence_manifest_detail} missing={missing_unblock_operator_checks}",
    )
    add_check(
        checks,
        "live_evidence_manifest_unblock_intake_operator_checks_pass",
        isinstance(live_evidence_manifest, dict) and failing_unblock_operator_checks == [],
        f"failing={failing_unblock_operator_checks}",
    )
    add_check(
        checks,
        "live_source_audit_present",
        isinstance(live_source_audit, dict),
        live_source_audit_detail,
    )
    add_check(
        checks,
        "live_source_audit_pass",
        isinstance(live_source_audit, dict) and live_source_audit.get("status") == "pass",
        str(live_source_audit.get("status") if isinstance(live_source_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "live_source_audit_required_count",
        isinstance(live_source_audit, dict)
        and positive_int(live_source_audit.get("required_count"))
        and int(live_source_audit.get("required_count")) >= MIN_SOURCE_AUDIT_REQUIRED_COUNT,
        str(live_source_audit.get("required_count") if isinstance(live_source_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "live_source_audit_source_count",
        isinstance(live_source_audit, dict)
        and positive_int(live_source_audit.get("source_count"))
        and int(live_source_audit.get("source_count")) >= MIN_SOURCE_AUDIT_SOURCE_COUNT,
        str(live_source_audit.get("source_count") if isinstance(live_source_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "evidence_manifest_no_policy_write",
        evidence_safety.get("creates_xr_vits_policy") is False,
        str(evidence_safety.get("creates_xr_vits_policy")),
    )
    add_check(
        checks,
        "evidence_manifest_no_board_create",
        evidence_safety.get("creates_board_result") is False,
        str(evidence_safety.get("creates_board_result")),
    )
    add_check(
        checks,
        "evidence_manifest_no_canonical_write",
        evidence_safety.get("writes_canonical_inputs") is False,
        str(evidence_safety.get("writes_canonical_inputs")),
    )
    embedded_remaining = {str(item) for item in remaining}
    live_intake_pending = intake_pending_blockers(live_intake or {})
    embedded_states = handoff_candidate_states(board, xr)
    live_intake_states = intake_candidate_states(live_intake or {})
    add_check(
        checks,
        "live_final_unblock_intake_present",
        isinstance(live_intake, dict),
        live_intake_detail,
    )
    add_check(
        checks,
        "handoff_remaining_blockers_match_intake",
        isinstance(live_intake, dict) and embedded_remaining == live_intake_pending,
        f"handoff={sorted(embedded_remaining)} intake={sorted(live_intake_pending)}",
    )
    add_check(
        checks,
        "handoff_candidate_states_match_intake",
        isinstance(live_intake, dict) and embedded_states == live_intake_states,
        f"handoff={embedded_states} intake={live_intake_states}",
    )
    add_check(checks, "combined_exact", contains(combined.get("exact_restore_commands"), "--import-c3b-smoke-json"), "combined exact import command present")
    add_check(checks, "combined_replacement", contains(combined.get("replacement_approval_commands"), "--dry-run-import-c3b-smoke") and contains(combined.get("replacement_approval_commands"), "--approve-xr-vits-replacement"), "combined replacement dry-run and approval command present")
    add_check(checks, "final_signoff_command", contains(final.get("commands"), "run_third_goal_final_signoff.py"), "final signoff command present")
    add_check(checks, "safety_no_board_create", safety.get("creates_board_result") is False, str(safety.get("creates_board_result")))
    add_check(checks, "safety_no_policy_create", safety.get("creates_xr_vits_policy") is False, str(safety.get("creates_xr_vits_policy")))
    add_check(checks, "safety_no_network_exec", safety.get("executes_network") is False, str(safety.get("executes_network")))
    add_check(checks, "operator_required", safety.get("requires_operator_action") is True, str(safety.get("requires_operator_action")))

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "fail_count": fail_count,
        "pass_count": len(checks) - fail_count,
        "check_count": len(checks),
        "checks": checks,
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Operator Handoff Validation",
        "",
        f"- status: `{result['status']}`",
        f"- pass: `{result['pass_count']}`",
        f"- fail: `{result['fail_count']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in result["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Does not execute commands.",
            "- Does not create board smoke results.",
            "- Does not create XR-VITs replacement policy.",
            "- Does not write canonical unblock inputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final operator handoff.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--handoff", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    handoff_path = args.handoff or root / f"docs/resources/final_operator_handoff_{DATE_TAG}.json"
    result = validate_handoff(load_json(handoff_path))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(result))
    print(f"[final-operator-handoff-validation] status={result['status']} fail={result['fail_count']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
