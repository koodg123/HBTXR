#!/usr/bin/env python3
"""Validate consistency of the HGTXR final signoff evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import create_xr_vits_replacement_policy as xr_policy


DATE_TAG = "2026_06_10"
HGTXR_ROOT = Path(__file__).resolve().parents[2]
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
REQUIRED_FINAL_EVIDENCE_IDS = {
    "c3b_bundle_sha256",
    "c3b_transfer_manifest_json",
    "c3b_transfer_manifest_md",
    "hgpipe_operator_audit_json",
    "hgpipe_operator_audit_md",
    "qkv_uram_successor_json",
    "qkv_uram_successor_md",
    "qkv_uram_smoke_discovery_json",
    "qkv_uram_smoke_discovery_md",
    "pynq_c3b_smoke_discovery_json",
    "pynq_c3b_smoke_discovery_md",
    "pynq_vref_smoke_discovery_json",
    "pynq_vref_smoke_discovery_md",
    "vref_p0_pot_scale_audit_json",
    "vref_p0_pot_scale_audit_md",
    "vref_p0_pot_scale_sweep_json",
    "vref_p0_pot_scale_sweep_md",
    "req5_q4q8_swhw_json",
    "req5_q4q8_swhw_md",
    "req6_parameterization_json",
    "req6_parameterization_md",
    "req9_deit_image_json",
    "req9_deit_image_md",
    "xr_vits_gate_json",
    "xr_vits_gate_md",
    "c3b_physical_smoke_gate_json",
    "c3b_physical_smoke_gate_md",
    "vref_p0_buffer_lifetime_json",
    "vref_p0_buffer_lifetime_md",
}
REQUIRED_TRACE_POLICY_CONSISTENCY_CHECKS = {
    "third_goal_requirements_trace_status_known",
    "third_goal_requirements_trace_req11_blocked",
    "third_goal_requirements_trace_xr_vits_policy_fields_complete",
    "third_goal_requirements_trace_xr_vits_legacy_policy_rejected",
    "third_goal_requirements_trace_hgpipe_property_summary_status_pass",
    "third_goal_requirements_trace_hgpipe_property_summary_no_fail",
    "third_goal_requirements_trace_hgpipe_operators_all_pass",
    "vref_p0_pot_scale_audit_pass",
    "vref_p0_pot_scale_audit_checks",
    "vref_p0_pot_scale_sweep_pass",
    "vref_p0_pot_scale_sweep_candidate_coverage",
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
    "resource_policy_audit_cyclic_weight_tiles_uram_macro_default",
    "resource_policy_audit_cyclic_large_temps_uram_macro_default",
    "resource_policy_audit_cyclic_small_tile_lutram_macro_default",
    "resource_policy_audit_cyclic_weight_tiles_uram_pragmas",
    "resource_policy_audit_cyclic_large_temps_uram_pragmas",
    "resource_policy_audit_cyclic_small_tile_lutram_pragmas",
    "resource_policy_audit_c3b_csynth_lut_lte_threshold",
    "resource_policy_audit_c3b_csynth_latency_lte_threshold",
    "resource_policy_audit_c3b_routed_wns_gte_threshold",
    "req9_deit_image_audit_pass",
    "req9_deit_image_audit_fail_zero",
    "req9_deit_image_requested_image_exists",
    "xr_vits_gate_status_blocked_or_clear",
    "xr_vits_gate_replacement_candidate_present",
    "xr_vits_gate_candidate_recommendation_matches",
    "c3b_physical_smoke_gate_status_known",
    "c3b_physical_smoke_gate_ready_for_board",
    "c3b_physical_smoke_gate_bundle_pass",
    "c3b_physical_smoke_gate_session_pass",
    "vref_p0_pot_scale_sweep_current_recommended",
    "resource_policy_canonical_pair_present",
    "resource_policy_docs_singularity",
    "resource_policy_generated_singularity",
    "resource_policy_filename_payload_date_match",
    "resource_policy_canonical_mirror_sha256",
    "resource_policy_audit_single_canonical_artifact_set",
    "pynq_smoke_validator_overlay_prefix_contracts",
    "pynq_smoke_validator_basename_check",
    "pynq_smoke_validator_require_paths_gate",
    "final_runner_remaining_blocker_input_paths_contract",
    "final_runner_remaining_blocker_details_contract",
    "xr_vits_policy_tool_placeholder_guard_contract",
    "xr_vits_policy_tool_dry_run_placeholder_allowed",
    "final_runner_active_policy_failure_blocks",
    "final_runner_remaining_blocker_input_paths_present",
    "final_runner_remaining_blocker_details_present",
    "final_runner_remaining_blocker_keys_align",
    "final_runner_remaining_blocker_input_paths_non_empty",
    "final_runner_remaining_blockers_present",
    "final_runner_blocker_count_matches_list",
    "final_runner_remaining_blocker_detail_count_matches",
    "final_runner_mirrored_artifact_counts_match",
    "final_runner_mirrored_artifact_integrity_present",
    "final_runner_mirrored_artifact_integrity_pass",
    "final_runner_mirrored_artifact_integrity_counts_match",
    "third_goal_source_audit_required_count",
    "third_goal_source_audit_source_count",
    "final_blocker_closure_c3b_canonical_path",
    "final_blocker_closure_xr_vits_exact_path",
    "final_blocker_closure_xr_vits_policy_path",
    "final_blocker_closure_discovery_only_safety",
    "final_blocker_closure_pynq_c3b_discovery_status_known",
    "final_blocker_closure_pynq_c3b_discovery_preset",
    "final_blocker_closure_pynq_c3b_discovery_candidate_count_numeric",
    "final_blocker_closure_pynq_c3b_discovery_pass_count_numeric",
    "final_blocker_closure_pynq_c3b_discovery_pass_count_lte_candidate_count",
    "final_blocker_closure_pynq_c3b_discovery_missing_has_no_recommended_candidate_path",
    "final_blocker_closure_pynq_c3b_discovery_safe_no_side_effects",
    "final_blocker_closure_pynq_vref_discovery_status_known",
    "final_blocker_closure_pynq_vref_discovery_preset",
    "final_blocker_closure_pynq_vref_discovery_candidate_count_numeric",
    "final_blocker_closure_pynq_vref_discovery_pass_count_numeric",
    "final_blocker_closure_pynq_vref_discovery_pass_count_lte_candidate_count",
    "final_blocker_closure_pynq_vref_discovery_missing_has_no_recommended_candidate_path",
    "final_blocker_closure_pynq_vref_discovery_safe_no_side_effects",
    "final_blocker_closure_qkv_uram_discovery_status_known",
    "final_blocker_closure_qkv_uram_discovery_preset",
    "final_blocker_closure_qkv_uram_discovery_candidate_count_numeric",
    "final_blocker_closure_qkv_uram_discovery_pass_count_numeric",
    "final_blocker_closure_qkv_uram_discovery_pass_count_lte_candidate_count",
    "final_blocker_closure_qkv_uram_discovery_missing_has_no_recommended_candidate_path",
    "final_blocker_closure_qkv_uram_discovery_safe_no_side_effects",
    "pynq_c3b_smoke_discovery_status_known",
    "pynq_c3b_smoke_discovery_preset",
    "pynq_c3b_smoke_discovery_candidate_count_numeric",
    "pynq_c3b_smoke_discovery_pass_count_numeric",
    "pynq_c3b_smoke_discovery_pass_count_lte_candidate_count",
    "pynq_c3b_smoke_discovery_missing_has_no_recommended_candidate",
    "pynq_c3b_smoke_discovery_safe_no_side_effects",
    "pynq_vref_smoke_discovery_status_known",
    "pynq_vref_smoke_discovery_preset",
    "pynq_vref_smoke_discovery_candidate_count_numeric",
    "pynq_vref_smoke_discovery_pass_count_numeric",
    "pynq_vref_smoke_discovery_pass_count_lte_candidate_count",
    "pynq_vref_smoke_discovery_missing_has_no_recommended_candidate",
    "pynq_vref_smoke_discovery_safe_no_side_effects",
    "qkv_uram_smoke_discovery_status_known",
    "qkv_uram_smoke_discovery_preset",
    "qkv_uram_smoke_discovery_candidate_count_numeric",
    "qkv_uram_smoke_discovery_pass_count_numeric",
    "qkv_uram_smoke_discovery_pass_count_lte_candidate_count",
    "qkv_uram_smoke_discovery_missing_has_no_recommended_candidate",
    "qkv_uram_smoke_discovery_safe_no_side_effects",
    "final_runner_c3b_transfer_manifest_status_pass",
    "spec_plan_conformance_pass",
    "spec_plan_manual_fallback_contract",
    "spec_plan_zcu104_q4wq8a_contract",
    "spec_plan_param_knobs_contract",
    "spec_plan_selected_paths_contract",
    "spec_plan_xilinx_root_contract",
    "xr_vits_policy_preview_active_policy_path",
    "xr_vits_policy_preview_candidate_audit_rel",
    "xr_vits_policy_preview_integrity_candidate_path",
    "xr_vits_policy_preview_approval_event_schema",
    "xr_vits_policy_preview_approval_event_matches_policy",
    "xr_vits_policy_preview_approval_event_id",
    "c3b_transfer_manifest_loaded_and_typed",
    "c3b_transfer_manifest_pass",
    "c3b_transfer_manifest_preset_variant",
    "c3b_transfer_manifest_tar_shape",
    "c3b_transfer_manifest_files_contract",
    "c3b_transfer_manifest_bundle_validation_clean",
    "c3b_transfer_manifest_sha256_line_matches",
    "c3b_transfer_manifest_expected_outputs_contract",
    "c3b_transfer_manifest_board_verify_command",
    "c3b_transfer_manifest_board_run_commands",
    "c3b_transfer_manifest_copyback_contract",
    "c3b_transfer_manifest_sha256_file_format",
    "c3b_transfer_manifest_sha_matches_readiness",
    "c3b_transfer_manifest_bundle_to_readiness_alignment",
    "final_unblock_intake_next_inputs_match_operator_plan_required_inputs",
    "final_unblock_intake_next_input_paths_match_operator_plan",
    "final_unblock_intake_operator_sequence_contains_expected_steps",
    "final_unblock_intake_operator_sequence_matches_operator_plan_commands",
    "final_unblock_intake_operator_sequence_side_effect_profile",
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


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def blocker_names(final_audit: dict[str, Any]) -> list[str]:
    blockers = final_audit.get("blockers", [])
    if not isinstance(blockers, list):
        return []
    return sorted(str(blocker.get("name", "")) for blocker in blockers if isinstance(blocker, dict))


def section_ids(command_card: dict[str, Any]) -> set[str]:
    sections = command_card.get("sections", [])
    if not isinstance(sections, list):
        return set()
    return {str(section.get("id", "")) for section in sections if isinstance(section, dict)}


def command_card_blockers(command_card: dict[str, Any]) -> list[str]:
    blockers = command_card.get("remaining_blockers", [])
    if not isinstance(blockers, list):
        return []
    return sorted(str(item) for item in blockers)


def trace_blockers(requirements_trace: dict[str, Any]) -> list[str]:
    final = requirements_trace.get("final_signoff", {})
    blockers = final.get("remaining_blockers", []) if isinstance(final, dict) else []
    if not isinstance(blockers, list):
        return []
    return sorted(str(item) for item in blockers)


def candidate_blockers(candidate_audit: dict[str, Any]) -> list[str]:
    blockers = candidate_audit.get("remaining_blockers", [])
    if not isinstance(blockers, list):
        return []
    return sorted(str(item) for item in blockers)


def handoff_blockers(handoff: dict[str, Any]) -> list[str]:
    blockers = handoff.get("remaining_blockers", [])
    if not isinstance(blockers, list):
        return []
    return sorted(str(item) for item in blockers)


def evidence_contract(payload: dict[str, Any]) -> dict[str, Any]:
    contract = payload.get("final_evidence_manifest_contract", {})
    return contract if isinstance(contract, dict) else {}


def failed_consistency(contract: dict[str, Any]) -> list[str]:
    failed = contract.get("failed_consistency_checks", [])
    if not isinstance(failed, list):
        return ["<non-list>"]
    return [str(item) for item in failed]


def bootstrap_evidence_failures_only(contract: dict[str, Any]) -> bool:
    failed = set(failed_consistency(contract))
    return bool(failed) and failed <= ALLOWED_BOOTSTRAP_EVIDENCE_FAILURES


def spec_plan_bootstrap_failures_only(spec_plan: dict[str, Any]) -> bool:
    checks = spec_plan.get("checks", []) if isinstance(spec_plan.get("checks"), list) else []
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


def safety_flag(safety: dict[str, Any], primary: str, alias: str | None = None) -> bool | None:
    if primary in safety:
        return safety.get(primary)
    if alias and alias in safety:
        return safety.get(alias)
    return None


def manifest_contract(final_evidence_manifest: dict[str, Any]) -> dict[str, Any]:
    consistency_checks = final_evidence_manifest.get("consistency_checks", [])
    consistency_count = len(consistency_checks) if isinstance(consistency_checks, list) else 0
    return {
        "status": final_evidence_manifest.get("status"),
        "source": "manifest-json",
        "path": str(Path(str(final_evidence_manifest.get("root", ""))) / "docs" / "resources" / f"final_evidence_manifest_{DATE_TAG}.json"),
        "required_count": final_evidence_manifest.get("required_count"),
        "present_required_count": final_evidence_manifest.get("present_required_count"),
        "consistency_count": consistency_count,
        "failed_consistency_checks": failed_consistency(final_evidence_manifest),
        "safety": contract_safety(final_evidence_manifest),
    }


def manifest_artifact_ids(final_evidence_manifest: dict[str, Any]) -> set[str]:
    artifacts = final_evidence_manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        return set()
    return {str(artifact.get("id", "")) for artifact in artifacts if isinstance(artifact, dict)}


def manifest_consistency_by_name(final_evidence_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = final_evidence_manifest.get("consistency_checks", [])
    if not isinstance(checks, list):
        return {}
    return {
        str(check.get("name")): check
        for check in checks
        if isinstance(check, dict) and check.get("name")
    }


def checks_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = payload.get("checks", []) if isinstance(payload.get("checks"), list) else []
    return {
        str(check.get("name", "")): check
        for check in checks
        if isinstance(check, dict) and check.get("name")
    }


def int_at_least(value: Any, minimum: int) -> bool:
    try:
        return int(value) >= minimum
    except (TypeError, ValueError):
        return False


def required_policy_fields_present(integrity: dict[str, Any]) -> bool:
    fields = integrity.get("required_policy_fields", [])
    if not isinstance(fields, list):
        return False
    return {
        "candidate_audit_fingerprint",
        "candidate_audit_recommendation_snapshot",
        "candidate_audit_meta",
        "approval_event",
        "policy_fingerprint",
    }.issubset({str(field) for field in fields})


def handoff_policy_integrity(handoff: dict[str, Any]) -> dict[str, Any]:
    xr = handoff.get("xr_vits", {}) if isinstance(handoff.get("xr_vits"), dict) else {}
    resolution = xr.get("reference_resolution", {}) if isinstance(xr.get("reference_resolution"), dict) else {}
    integrity = resolution.get("policy_integrity", {}) if isinstance(resolution.get("policy_integrity"), dict) else {}
    return integrity


def policy_validation_status(integrity: dict[str, Any]) -> str:
    validation = integrity.get("validation", {}) if isinstance(integrity.get("validation"), dict) else {}
    return str(validation.get("status", "missing"))


def live_policy_validation(root: Path, integrity: dict[str, Any]) -> dict[str, Any]:
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


def build_bundle_paths(root: Path) -> dict[str, Path]:
    resources = root / "docs" / "resources"
    return {
        "final_audit": resources / f"final_signoff_audit_{DATE_TAG}.json",
        "completion": resources / f"third_goal_completion_audit_{DATE_TAG}.json",
        "unblock": resources / f"third_goal_unblock_checklist_{DATE_TAG}.json",
        "command_card": resources / f"final_unblock_commands_{DATE_TAG}.json",
        "resource_matrix": resources / f"e2e_resource_matrix_{DATE_TAG}.json",
        "resource_policy": resources / f"e2e_resource_policy_audit_{DATE_TAG}.json",
        "spec_plan": resources / f"spec_plan_conformance_audit_{DATE_TAG}.json",
        "requirements_trace": resources / f"third_goal_requirements_trace_{DATE_TAG}.json",
        "candidate_audit": resources / f"final_unblock_candidate_audit_{DATE_TAG}.json",
        "xr_vits_reference_resolution": resources / f"xr_vits_reference_resolution_{DATE_TAG}.json",
        "operator_handoff": resources / f"final_operator_handoff_{DATE_TAG}.json",
        "operator_handoff_validation": resources / f"final_operator_handoff_validation_{DATE_TAG}.json",
        "final_evidence_manifest": resources / f"final_evidence_manifest_{DATE_TAG}.json",
        "final_runner": resources / f"third_goal_final_signoff_run_{DATE_TAG}.json",
        "closeout_packet": resources / f"final_unblock_closeout_packet_{DATE_TAG}.json",
        "closeout_validation": resources / f"final_unblock_closeout_packet_validation_{DATE_TAG}.json",
        "source_audit": resources / "third_goal_source_audit_2026_06_16.json",
        "req6_parameterization": resources / "req6_parameterization_audit_2026_06_16.json",
        "current_audit": resources / "third_goal_current_audit_2026_06_16.json",
        "c3b_physical_gate": resources / "c3b_physical_smoke_gate_audit_2026_06_16.json",
        "xr_vits_gate": resources / "xr_vits_gate_audit_2026_06_16.json",
    }


def validate_bundle(root: Path, paths: dict[str, Path] | None = None) -> dict[str, Any]:
    root = root.resolve()
    paths = paths or build_bundle_paths(root)
    loaded: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []

    for key, path in paths.items():
        exists = path.exists()
        add_check(checks, f"{key}_exists", exists, str(path))
        if exists:
            try:
                loaded[key] = load_json(path)
                add_check(checks, f"{key}_json_object", True, "json object")
            except (json.JSONDecodeError, ValueError) as exc:
                add_check(checks, f"{key}_json_object", False, str(exc))

    required_keys = set(paths)
    if set(loaded) != required_keys:
        fail_count = sum(1 for check in checks if check["status"] == "fail")
        return {
            "status": "fail",
            "root": str(root),
            "checks": checks,
            "check_count": len(checks),
            "pass_count": len(checks) - fail_count,
            "fail_count": fail_count,
            "source_files": {key: str(path) for key, path in paths.items()},
            "safety": safety_payload(),
        }

    final_audit = loaded["final_audit"]
    completion = loaded["completion"]
    unblock = loaded["unblock"]
    command_card = loaded["command_card"]
    resource_matrix = loaded["resource_matrix"]
    resource_policy = loaded["resource_policy"]
    spec_plan = loaded["spec_plan"]
    requirements_trace = loaded["requirements_trace"]
    candidate_audit = loaded["candidate_audit"]
    xr_vits_resolution = loaded["xr_vits_reference_resolution"]
    handoff = loaded["operator_handoff"]
    handoff_validation = loaded["operator_handoff_validation"]
    final_evidence_manifest = loaded["final_evidence_manifest"]
    final_runner = loaded["final_runner"]
    closeout_packet = loaded["closeout_packet"]
    closeout_validation = loaded["closeout_validation"]
    source_audit = loaded["source_audit"]
    req6_parameterization = loaded["req6_parameterization"]
    current_audit = loaded["current_audit"]
    c3b_physical_gate = loaded["c3b_physical_gate"]
    xr_vits_gate = loaded["xr_vits_gate"]

    final_status = str(final_audit.get("status", ""))
    blockers = blocker_names(final_audit)
    blocker_set = set(blockers)
    xr_vits_blocked = "requested XR-VITs sibling" in blocker_set
    xr_vits_resolution_status = str(xr_vits_resolution.get("status", ""))
    xr_vits_resolution_ready = bool(xr_vits_resolution.get("resolution_ready"))
    expected_unblock_status = "pending-unblock" if blocker_set else "ready-for-final-signoff"
    expected_handoff_status = "pending-operator-actions" if blocker_set else "ready-for-final-signoff"
    trace_contract = evidence_contract(requirements_trace)
    handoff_contract = evidence_contract(handoff)
    live_contract = manifest_contract(final_evidence_manifest)
    live_artifact_ids = manifest_artifact_ids(final_evidence_manifest)
    live_consistency = manifest_consistency_by_name(final_evidence_manifest)
    trace_safety = contract_safety(trace_contract)
    handoff_safety = contract_safety(handoff_contract)
    command_policy_integrity = (
        command_card.get("xr_vits_policy_integrity", {})
        if isinstance(command_card.get("xr_vits_policy_integrity"), dict)
        else {}
    )
    handoff_xr_policy_integrity = handoff_policy_integrity(handoff)
    live_xr_policy_status = str(live_policy_validation(root, handoff_xr_policy_integrity).get("status", "missing"))
    stored_xr_policy_status = policy_validation_status(handoff_xr_policy_integrity)
    pending_xr_policy_allowed = (
        xr_vits_blocked
        and xr_vits_resolution_status == "candidate-ready-needs-approval"
        and handoff_xr_policy_integrity.get("policy_exists") is False
    )

    add_check(checks, "final_status_known", final_status in {"blocked", "pass"}, final_status)
    add_check(checks, "blocker_count", final_audit.get("blocker_count") == len(blockers), str(blockers))
    add_check(checks, "completion_matches_final", completion.get("status") == final_status, str(completion.get("status")))
    add_check(checks, "unblock_status", unblock.get("status") == expected_unblock_status, str(unblock.get("status")))
    add_check(checks, "command_card_status", command_card.get("status") == expected_unblock_status, str(command_card.get("status")))
    add_check(checks, "resource_matrix_pass", resource_matrix.get("status") == "pass", str(resource_matrix.get("status")))
    add_check(
        checks,
        "resource_matrix_recommends_c3b",
        (resource_matrix.get("summary", {}) or {}).get("recommended_board_smoke_variant") == "C3b",
        str((resource_matrix.get("summary", {}) or {}).get("recommended_board_smoke_variant")),
    )
    add_check(checks, "requirements_trace_matches_final", requirements_trace.get("status") == final_status, str(requirements_trace.get("status")))
    trace_final = requirements_trace.get("final_signoff", {})
    add_check(
        checks,
        "requirements_trace_final_status",
        isinstance(trace_final, dict) and trace_final.get("status") == final_status,
        str(trace_final.get("status") if isinstance(trace_final, dict) else ""),
    )
    add_check(checks, "command_card_blockers_match", set(command_card_blockers(command_card)) == blocker_set, str(command_card_blockers(command_card)))
    add_check(checks, "trace_blockers_match", set(trace_blockers(requirements_trace)) == blocker_set, str(trace_blockers(requirements_trace)))
    add_check(checks, "candidate_blockers_match", set(candidate_blockers(candidate_audit)) == blocker_set, str(candidate_blockers(candidate_audit)))
    add_check(
        checks,
        "xr_vits_resolution_status_known",
        xr_vits_resolution_status in {
            "exact-ready",
            "approved-replacement-ready",
            "candidate-ready-needs-approval",
            "blocked",
        },
        xr_vits_resolution_status,
    )
    add_check(
        checks,
        "xr_vits_resolution_matches_blocker",
        (not xr_vits_blocked and xr_vits_resolution_ready)
        or (xr_vits_blocked and not xr_vits_resolution_ready),
        f"blocked={xr_vits_blocked} resolution_ready={xr_vits_resolution_ready}",
    )
    add_check(
        checks,
        "xr_vits_candidate_needs_approval_when_blocked",
        not xr_vits_blocked
        or xr_vits_resolution_status in {"candidate-ready-needs-approval", "blocked"},
        xr_vits_resolution_status,
    )
    add_check(checks, "handoff_blockers_match", set(handoff_blockers(handoff)) == blocker_set, str(handoff_blockers(handoff)))
    add_check(checks, "handoff_status", handoff.get("status") == expected_handoff_status, str(handoff.get("status")))
    add_check(checks, "handoff_validation_pass", handoff_validation.get("status") == "pass", str(handoff_validation.get("status")))
    add_check(checks, "handoff_validation_fail_zero", handoff_validation.get("fail_count") == 0, str(handoff_validation.get("fail_count")))
    add_check(
        checks,
        "handoff_validation_check_count",
        int_at_least(handoff_validation.get("check_count"), 131),
        str(handoff_validation.get("check_count")),
    )
    resource_policy_checks = checks_by_name(resource_policy)
    missing_resource_policy_checks = sorted(REQUIRED_LIVE_RESOURCE_POLICY_CHECKS - set(resource_policy_checks))
    failing_resource_policy_checks = sorted(
        name
        for name in REQUIRED_LIVE_RESOURCE_POLICY_CHECKS
        if resource_policy_checks.get(name, {}).get("status") != "pass"
    )
    add_check(checks, "resource_policy_pass", resource_policy.get("status") == "pass", str(resource_policy.get("status")))
    add_check(checks, "resource_policy_fail_zero", resource_policy.get("fail_count") == 0, str(resource_policy.get("fail_count")))
    add_check(
        checks,
        "resource_policy_check_count",
        int_at_least(resource_policy.get("check_count"), MIN_RESOURCE_POLICY_CHECK_COUNT),
        str(resource_policy.get("check_count")),
    )
    add_check(
        checks,
        "resource_policy_core_checks_present",
        missing_resource_policy_checks == [],
        f"missing={missing_resource_policy_checks}",
    )
    add_check(
        checks,
        "resource_policy_core_checks_pass",
        failing_resource_policy_checks == [],
        f"failing={failing_resource_policy_checks}",
    )
    spec_plan_checks = checks_by_name(spec_plan)
    missing_spec_plan_checks = sorted(REQUIRED_LIVE_SPEC_PLAN_CHECKS - set(spec_plan_checks))
    failing_spec_plan_checks = sorted(
        name
        for name in REQUIRED_LIVE_SPEC_PLAN_CHECKS
        if spec_plan_checks.get(name, {}).get("status") != "pass"
    )
    spec_observed = spec_plan.get("observed", {}) if isinstance(spec_plan.get("observed"), dict) else {}
    spec_safety = spec_plan.get("safety", {}) if isinstance(spec_plan.get("safety"), dict) else {}
    add_check(
        checks,
        "spec_plan_pass",
        spec_plan.get("status") == "pass" or spec_plan_bootstrap_failures_only(spec_plan),
        str(spec_plan.get("status")),
    )
    add_check(
        checks,
        "spec_plan_check_count",
        int_at_least(spec_plan.get("check_count"), MIN_SPEC_PLAN_CHECK_COUNT),
        str(spec_plan.get("check_count")),
    )
    add_check(
        checks,
        "spec_plan_core_checks_present",
        missing_spec_plan_checks == [],
        f"missing={missing_spec_plan_checks}",
    )
    add_check(
        checks,
        "spec_plan_core_checks_pass",
        failing_spec_plan_checks == [],
        f"failing={failing_spec_plan_checks}",
    )
    add_check(
        checks,
        "spec_plan_observed_counts_current",
        spec_observed.get("evidence_required") == MIN_EVIDENCE_REQUIRED_COUNT
        and spec_observed.get("evidence_consistency_count") == MIN_EVIDENCE_CONSISTENCY_COUNT
        and spec_observed.get("source_required_count") == MIN_SOURCE_AUDIT_REQUIRED_COUNT
        and int_at_least(spec_observed.get("source_count"), MIN_SOURCE_AUDIT_SOURCE_COUNT),
        str(spec_observed),
    )
    add_check(
        checks,
        "spec_plan_safety_no_side_effects",
        spec_safety.get("creates_board_result") is False
        and spec_safety.get("creates_xr_vits_policy") is False
        and spec_safety.get("executes_hls_or_vivado") is False
        and spec_safety.get("writes_canonical_inputs") is False,
        str(spec_safety),
    )
    req6_checks = checks_by_name(req6_parameterization)
    missing_req6_checks = sorted(REQUIRED_LIVE_REQ6_CHECKS - set(req6_checks))
    failing_req6_checks = sorted(
        name
        for name in REQUIRED_LIVE_REQ6_CHECKS
        if req6_checks.get(name, {}).get("status") != "pass"
    )
    req6_knobs = (
        req6_parameterization.get("knobs", {}).get("config_macros", {})
        if isinstance(req6_parameterization.get("knobs"), dict)
        and isinstance(req6_parameterization["knobs"].get("config_macros"), dict)
        else {}
    )
    add_check(checks, "req6_parameterization_pass", req6_parameterization.get("status") == "pass", str(req6_parameterization.get("status")))
    add_check(
        checks,
        "req6_parameterization_fail_zero",
        req6_parameterization.get("fail_count") == 0,
        str(req6_parameterization.get("fail_count")),
    )
    add_check(
        checks,
        "req6_parameterization_check_count",
        int_at_least(req6_parameterization.get("check_count"), MIN_REQ6_PARAMETERIZATION_CHECK_COUNT),
        str(req6_parameterization.get("check_count")),
    )
    add_check(
        checks,
        "req6_parameterization_knobs",
        req6_knobs == EXPECTED_REQ6_KNOBS,
        str(req6_knobs),
    )
    add_check(
        checks,
        "req6_parameterization_core_checks_present",
        missing_req6_checks == [],
        f"missing={missing_req6_checks}",
    )
    add_check(
        checks,
        "req6_parameterization_core_checks_pass",
        failing_req6_checks == [],
        f"failing={failing_req6_checks}",
    )
    current_summary = current_audit.get("summary", {}) if isinstance(current_audit.get("summary"), dict) else {}
    current_external_blockers = (
        {str(name) for name in current_audit.get("external_blocker_names", [])}
        if isinstance(current_audit.get("external_blocker_names"), list)
        else set()
    )
    current_paths = (
        current_audit.get("blocked_external_input_paths_by_blocker", {})
        if isinstance(current_audit.get("blocked_external_input_paths_by_blocker"), dict)
        else {}
    )
    current_details = (
        current_audit.get("remaining_external_input_details", [])
        if isinstance(current_audit.get("remaining_external_input_details"), list)
        else []
    )
    current_xr_policy = (
        current_audit.get("xr_vits_policy_integrity", {})
        if isinstance(current_audit.get("xr_vits_policy_integrity"), dict)
        else {}
    )
    current_qkv = (
        current_audit.get("qkv_uram_runner", {})
        if isinstance(current_audit.get("qkv_uram_runner"), dict)
        else {}
    )
    current_safety = current_audit.get("safety", {}) if isinstance(current_audit.get("safety"), dict) else {}
    add_check(
        checks,
        "current_audit_status_blocked_external",
        current_audit.get("status") == "blocked-external",
        str(current_audit.get("status")),
    )
    add_check(
        checks,
        "current_audit_summary_counts",
        all(current_summary.get(key) == value for key, value in EXPECTED_CURRENT_AUDIT_SUMMARY.items()),
        str(current_summary),
    )
    add_check(
        checks,
        "current_audit_external_blockers",
        current_external_blockers == EXPECTED_FINAL_RUNNER_BLOCKERS,
        str(sorted(current_external_blockers)),
    )
    add_check(
        checks,
        "current_audit_blocker_paths",
        str(current_paths.get("C3b AXIS/DMA physical smoke result", "")).endswith(
            "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
        )
        and str(current_paths.get("requested XR-VITs sibling", "")).endswith("XR-VITs"),
        str(current_paths),
    )
    add_check(
        checks,
        "current_audit_external_detail_count",
        len(current_details) >= len(EXPECTED_FINAL_RUNNER_BLOCKERS)
        and all(detail.get("status") == "missing" for detail in current_details if isinstance(detail, dict)),
        str(current_details),
    )
    add_check(
        checks,
        "current_audit_xr_policy_integrity",
        current_xr_policy.get("consistent") is True
        and current_xr_policy.get("operator_handoff_policy_check_count") == 9,
        str(current_xr_policy),
    )
    add_check(
        checks,
        "current_audit_qkv_optional_state",
        current_qkv.get("qkv_uram_required_for_final_signoff") is False
        and current_qkv.get("qkv_uram_import_status") == "skipped"
        and current_qkv.get("qkv_uram_remote_status") == "skipped"
        and current_qkv.get("qkv_uram_smoke_discovery_status") == "missing",
        str(current_qkv),
    )
    add_check(
        checks,
        "current_audit_safety_no_side_effects",
        current_safety.get("creates_board_result") is False
        and current_safety.get("creates_xr_vits_policy") is False
        and current_safety.get("executes_network") is False
        and current_safety.get("writes_canonical_inputs") is False,
        str(current_safety),
    )
    c3b_gate_canonical = (
        c3b_physical_gate.get("canonical_result", {})
        if isinstance(c3b_physical_gate.get("canonical_result"), dict)
        else {}
    )
    c3b_gate_validation = (
        c3b_physical_gate.get("canonical_validation", {})
        if isinstance(c3b_physical_gate.get("canonical_validation"), dict)
        else {}
    )
    c3b_gate_bundle = c3b_physical_gate.get("bundle", {}) if isinstance(c3b_physical_gate.get("bundle"), dict) else {}
    c3b_gate_session = c3b_physical_gate.get("session", {}) if isinstance(c3b_physical_gate.get("session"), dict) else {}
    c3b_gate_safety = c3b_physical_gate.get("safety", {}) if isinstance(c3b_physical_gate.get("safety"), dict) else {}
    add_check(
        checks,
        "c3b_physical_gate_status_known",
        c3b_physical_gate.get("status") in {"pass", "blocked_missing_canonical_physical_smoke_result"},
        str(c3b_physical_gate.get("status")),
    )
    add_check(
        checks,
        "c3b_physical_gate_current_blocker_contract",
        c3b_physical_gate.get("ready_for_board") is True
        and c3b_physical_gate.get("physical_smoke_pass") is False
        and "C3b AXIS/DMA physical smoke result" in {str(name) for name in c3b_physical_gate.get("remaining_blockers", [])}
        and c3b_gate_canonical.get("status") == "missing"
        and str(c3b_gate_canonical.get("path", "")).endswith(
            "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
        ),
        f"status={c3b_physical_gate.get('status')} canonical={c3b_gate_canonical}",
    )
    add_check(
        checks,
        "c3b_physical_gate_validation_path",
        c3b_gate_validation.get("status") in {"missing", "pass", "fail"}
        and str(c3b_gate_validation.get("path", "")).endswith(
            "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke_validation.json"
        ),
        str(c3b_gate_validation),
    )
    add_check(
        checks,
        "c3b_physical_gate_bundle_session_ready",
        c3b_gate_bundle.get("status") == "pass"
        and c3b_gate_session.get("status") == "pass"
        and c3b_gate_bundle.get("expected_runtime_state") == 2
        and isinstance(c3b_gate_bundle.get("expected_out_raw"), list)
        and len(c3b_gate_bundle.get("expected_out_raw", [])) == 6,
        f"bundle={c3b_gate_bundle} session={c3b_gate_session}",
    )
    add_check(
        checks,
        "c3b_physical_gate_safety_no_side_effects",
        c3b_gate_safety.get("creates_board_result") is False
        and c3b_gate_safety.get("executes_commands") is False
        and c3b_gate_safety.get("executes_network") is False
        and c3b_gate_safety.get("writes_canonical_inputs") is False,
        str(c3b_gate_safety),
    )
    xr_gate_current = xr_vits_gate.get("current_gate", {}) if isinstance(xr_vits_gate.get("current_gate"), dict) else {}
    xr_gate_exact = xr_vits_gate.get("exact", {}) if isinstance(xr_vits_gate.get("exact"), dict) else {}
    xr_gate_candidate = (
        xr_vits_gate.get("replacement_candidate", {})
        if isinstance(xr_vits_gate.get("replacement_candidate"), dict)
        else {}
    )
    xr_gate_candidate_audit = (
        xr_vits_gate.get("candidate_audit", {}) if isinstance(xr_vits_gate.get("candidate_audit"), dict) else {}
    )
    xr_gate_policy = (
        xr_vits_gate.get("active_policy_summary", {})
        if isinstance(xr_vits_gate.get("active_policy_summary"), dict)
        else {}
    )
    xr_gate_safety = xr_vits_gate.get("safety", {}) if isinstance(xr_vits_gate.get("safety"), dict) else {}
    xr_gate_recommendation = (
        xr_gate_candidate_audit.get("recommendation", {})
        if isinstance(xr_gate_candidate_audit.get("recommendation"), dict)
        else {}
    )
    add_check(
        checks,
        "xr_vits_gate_status_blocked_or_clear",
        xr_vits_gate.get("status") in {"blocked", "pass-exact", "pass-replacement-policy"},
        str(xr_vits_gate.get("status")),
    )
    add_check(
        checks,
        "xr_vits_gate_current_blocker_contract",
        xr_vits_gate.get("resolution_mode") == "candidate-ready-needs-approval"
        and "requested XR-VITs sibling" in {str(name) for name in xr_vits_gate.get("remaining_blockers", [])}
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
        "xr_vits_gate_candidate_contract",
        xr_gate_candidate.get("exists") is True
        and str(xr_gate_candidate.get("path", "")).endswith("XR_Accel")
        and xr_gate_candidate_audit.get("status") == "candidate-found"
        and xr_gate_candidate_audit.get("recommendation_matches") is True
        and int_at_least(xr_gate_recommendation.get("score"), 1),
        f"candidate={xr_gate_candidate} audit={xr_gate_candidate_audit}",
    )
    add_check(
        checks,
        "xr_vits_gate_safety_no_side_effects",
        xr_gate_safety.get("creates_xr_vits_policy") is False
        and xr_gate_safety.get("executes_commands") is False
        and xr_gate_safety.get("executes_network") is False
        and xr_gate_safety.get("writes_canonical_inputs") is False,
        str(xr_gate_safety),
    )
    closeout_packet_blockers = (
        {str(name) for name in closeout_packet.get("remaining_blockers", [])}
        if isinstance(closeout_packet.get("remaining_blockers"), list)
        else set()
    )
    closeout_board = closeout_packet.get("board_package", {}) if isinstance(closeout_packet.get("board_package"), dict) else {}
    closeout_c3b = (
        closeout_packet.get("c3b_smoke_contract", {})
        if isinstance(closeout_packet.get("c3b_smoke_contract"), dict)
        else {}
    )
    closeout_xr = (
        closeout_packet.get("xr_vits_resolution", {})
        if isinstance(closeout_packet.get("xr_vits_resolution"), dict)
        else {}
    )
    closeout_xr_policy = (
        closeout_xr.get("policy_integrity", {})
        if isinstance(closeout_xr.get("policy_integrity"), dict)
        else {}
    )
    closeout_qkv = (
        closeout_packet.get("qkv_uram_successor_gate", {})
        if isinstance(closeout_packet.get("qkv_uram_successor_gate"), dict)
        else {}
    )
    closeout_commands = (
        closeout_packet.get("operator_commands", {})
        if isinstance(closeout_packet.get("operator_commands"), dict)
        else {}
    )
    closeout_qkv_commands = (
        closeout_commands.get("qkv_uram_successor", [])
        if isinstance(closeout_commands.get("qkv_uram_successor"), list)
        else []
    )
    closeout_packet_safety = closeout_packet.get("safety", {}) if isinstance(closeout_packet.get("safety"), dict) else {}
    add_check(
        checks,
        "closeout_packet_ready",
        closeout_packet.get("status") == "ready-for-operator-unblock",
        str(closeout_packet.get("status")),
    )
    add_check(
        checks,
        "closeout_packet_blockers",
        closeout_packet_blockers == EXPECTED_FINAL_RUNNER_BLOCKERS,
        str(sorted(closeout_packet_blockers)),
    )
    add_check(
        checks,
        "closeout_packet_board_package",
        closeout_board.get("status") == "ready-for-board"
        and closeout_board.get("preset") == "axis-c3b-mem16"
        and closeout_board.get("expected_runtime_state") == 2
        and bool(closeout_board.get("tar"))
        and bool(closeout_board.get("tar_sha256")),
        str(closeout_board),
    )
    add_check(
        checks,
        "closeout_packet_c3b_contract",
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
        "closeout_packet_xr_policy_integrity",
        closeout_xr.get("status") in {"candidate-ready-needs-approval", "blocked", "exact-ready", "approved-replacement-ready"}
        and closeout_xr_policy.get("required") is True
        and closeout_xr_policy.get("legacy_policy_clears_final_signoff") is False
        and closeout_xr_policy.get("policy_exists") is False,
        str(closeout_xr_policy),
    )
    add_check(
        checks,
        "closeout_packet_qkv_optional_commands",
        closeout_qkv.get("required_for_final_signoff") is False
        and closeout_qkv.get("physical_smoke_status") == "not_captured"
        and any("--execute-qkv-uram-smoke" in str(command) for command in closeout_qkv_commands)
        and any("--dry-run-import-qkv-uram-smoke" in str(command) for command in closeout_qkv_commands),
        f"qkv={closeout_qkv} commands={closeout_qkv_commands}",
    )
    add_check(
        checks,
        "closeout_packet_safety_no_side_effects",
        closeout_packet_safety.get("creates_board_result") is False
        and closeout_packet_safety.get("creates_xr_vits_policy") is False
        and closeout_packet_safety.get("executes_commands") is False
        and closeout_packet_safety.get("executes_network") is False
        and closeout_packet_safety.get("writes_canonical_inputs") is False,
        str(closeout_packet_safety),
    )
    closeout_validation_checks = checks_by_name(closeout_validation)
    missing_closeout_validation_checks = sorted(
        REQUIRED_LIVE_CLOSEOUT_VALIDATION_CHECKS - set(closeout_validation_checks)
    )
    failing_closeout_validation_checks = sorted(
        name
        for name in REQUIRED_LIVE_CLOSEOUT_VALIDATION_CHECKS
        if closeout_validation_checks.get(name, {}).get("status") != "pass"
    )
    closeout_validation_safety = (
        closeout_validation.get("safety", {})
        if isinstance(closeout_validation.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "closeout_validation_pass",
        closeout_validation.get("status") == "pass",
        str(closeout_validation.get("status")),
    )
    add_check(
        checks,
        "closeout_validation_fail_zero",
        closeout_validation.get("fail_count") == 0,
        str(closeout_validation.get("fail_count")),
    )
    add_check(
        checks,
        "closeout_validation_check_count",
        int_at_least(closeout_validation.get("check_count"), MIN_CLOSEOUT_VALIDATION_CHECK_COUNT),
        str(closeout_validation.get("check_count")),
    )
    add_check(
        checks,
        "closeout_validation_core_checks_present",
        missing_closeout_validation_checks == [],
        f"missing={missing_closeout_validation_checks}",
    )
    add_check(
        checks,
        "closeout_validation_core_checks_pass",
        failing_closeout_validation_checks == [],
        f"failing={failing_closeout_validation_checks}",
    )
    add_check(
        checks,
        "closeout_validation_safety_no_side_effects",
        closeout_validation_safety.get("creates_board_result") is False
        and closeout_validation_safety.get("creates_xr_vits_policy") is False
        and closeout_validation_safety.get("executes_commands") is False
        and closeout_validation_safety.get("executes_network") is False
        and closeout_validation_safety.get("writes_canonical_inputs") is False,
        str(closeout_validation_safety),
    )
    runner_blockers = set()
    if isinstance(final_runner.get("remaining_blockers"), list):
        runner_blockers = {str(name) for name in final_runner.get("remaining_blockers", [])}
    runner_details = final_runner.get("remaining_blocker_details")
    runner_detail_count = len(runner_details) if isinstance(runner_details, (list, dict)) else None
    mirrored_artifacts = final_runner.get("mirrored_artifacts", []) if isinstance(final_runner.get("mirrored_artifacts"), list) else []
    mirror_integrity = (
        final_runner.get("mirrored_artifact_integrity", [])
        if isinstance(final_runner.get("mirrored_artifact_integrity"), list)
        else []
    )
    add_check(checks, "final_runner_status_blocked", final_runner.get("status") == "blocked", str(final_runner.get("status")))
    add_check(
        checks,
        "final_runner_blocker_count",
        final_runner.get("blocker_count") == len(EXPECTED_FINAL_RUNNER_BLOCKERS)
        and final_runner.get("blocker_count") == len(runner_blockers),
        f"count={final_runner.get('blocker_count')} blockers={sorted(runner_blockers)}",
    )
    add_check(
        checks,
        "final_runner_blocker_names",
        runner_blockers == EXPECTED_FINAL_RUNNER_BLOCKERS,
        str(sorted(runner_blockers)),
    )
    add_check(
        checks,
        "final_runner_detail_count",
        final_runner.get("remaining_blocker_detail_count") == len(EXPECTED_FINAL_RUNNER_BLOCKERS)
        and runner_detail_count == len(EXPECTED_FINAL_RUNNER_BLOCKERS),
        f"count={final_runner.get('remaining_blocker_detail_count')} details={runner_detail_count}",
    )
    add_check(
        checks,
        "final_runner_mirror_counts",
        final_runner.get("mirrored_artifact_count") == len(mirrored_artifacts)
        and final_runner.get("mirrored_artifact_unique_count") == len(set(str(path) for path in mirrored_artifacts))
        and final_runner.get("mirrored_artifact_duplicate_count") == 0,
        (
            f"count={final_runner.get('mirrored_artifact_count')} "
            f"unique={final_runner.get('mirrored_artifact_unique_count')} "
            f"duplicates={final_runner.get('mirrored_artifact_duplicate_count')}"
        ),
    )
    add_check(
        checks,
        "final_runner_mirror_integrity_pass",
        final_runner.get("mirrored_artifact_integrity_status") == "pass"
        and final_runner.get("mirrored_artifact_integrity_fail_count") == 0
        and final_runner.get("mirrored_artifact_integrity_failures") == [],
        (
            f"status={final_runner.get('mirrored_artifact_integrity_status')} "
            f"fail_count={final_runner.get('mirrored_artifact_integrity_fail_count')}"
        ),
    )
    add_check(
        checks,
        "final_runner_mirror_integrity_counts",
        final_runner.get("mirrored_artifact_integrity_count") == len(mirrored_artifacts)
        and final_runner.get("mirrored_artifact_integrity_count") == len(mirror_integrity)
        and isinstance(final_runner.get("mirrored_artifact_integrity_checked_count"), int)
        and isinstance(final_runner.get("mirrored_artifact_integrity_excluded_count"), int)
        and final_runner.get("mirrored_artifact_integrity_checked_count")
        + final_runner.get("mirrored_artifact_integrity_excluded_count")
        == final_runner.get("mirrored_artifact_integrity_count"),
        f"mirrored={len(mirrored_artifacts)} integrity={final_runner.get('mirrored_artifact_integrity_count')}",
    )
    add_check(
        checks,
        "command_policy_integrity_required",
        command_policy_integrity.get("required") is True,
        str(command_policy_integrity.get("required")),
    )
    add_check(
        checks,
        "command_policy_integrity_fields",
        required_policy_fields_present(command_policy_integrity),
        str(command_policy_integrity.get("required_policy_fields", [])),
    )
    add_check(
        checks,
        "command_policy_integrity_generator",
        "create_xr_vits_replacement_policy.py" in str(command_policy_integrity.get("generator", "")),
        str(command_policy_integrity.get("generator", "")),
    )
    add_check(
        checks,
        "handoff_policy_integrity_required",
        handoff_xr_policy_integrity.get("required") is True,
        str(handoff_xr_policy_integrity.get("required")),
    )
    add_check(
        checks,
        "handoff_policy_integrity_matches_command_card",
        all(
            handoff_xr_policy_integrity.get(key) == command_policy_integrity.get(key)
            for key in (
                "required",
                "policy_path",
                "candidate_audit",
                "generator",
                "validator",
                "required_policy_fields",
                "legacy_policy_clears_final_signoff",
            )
        ),
        "handoff policy integrity agrees with command card",
    )
    add_check(
        checks,
        "handoff_policy_validation_status",
        live_xr_policy_status == "pass"
        or (pending_xr_policy_allowed and live_xr_policy_status == "pending-policy-creation"),
        f"stored={stored_xr_policy_status} live={live_xr_policy_status}",
    )
    add_check(
        checks,
        "handoff_policy_validation_not_legacy",
        live_xr_policy_status != "legacy-warning",
        live_xr_policy_status,
    )
    add_check(
        checks,
        "handoff_policy_validation_embedded_matches_live",
        stored_xr_policy_status == live_xr_policy_status,
        f"stored={stored_xr_policy_status} live={live_xr_policy_status}",
    )
    add_check(checks, "trace_evidence_contract_present", bool(trace_contract), "trace contract present")
    add_check(checks, "handoff_evidence_contract_present", bool(handoff_contract), "handoff contract present")
    add_check(
        checks,
        "trace_evidence_contract_pass",
        trace_contract.get("status") == "pass" or bootstrap_evidence_failures_only(trace_contract),
        str(trace_contract.get("status")),
    )
    add_check(
        checks,
        "handoff_evidence_contract_pass",
        handoff_contract.get("status") == "pass" or bootstrap_evidence_failures_only(handoff_contract),
        str(handoff_contract.get("status")),
    )
    add_check(
        checks,
        "handoff_evidence_contract_matches_trace",
        handoff_contract == trace_contract,
        "handoff contract equals requirements trace contract",
    )
    add_check(
        checks,
        "trace_evidence_contract_matches_manifest",
        trace_contract == live_contract,
        "requirements trace contract equals live final evidence manifest",
    )
    add_check(
        checks,
        "handoff_evidence_contract_matches_manifest",
        handoff_contract == live_contract,
        "operator handoff contract equals live final evidence manifest",
    )
    add_check(
        checks,
        "evidence_contract_required_complete",
        trace_contract.get("present_required_count") == trace_contract.get("required_count")
        and int_at_least(trace_contract.get("required_count"), MIN_EVIDENCE_REQUIRED_COUNT),
        f"{trace_contract.get('present_required_count')}/{trace_contract.get('required_count')}",
    )
    add_check(
        checks,
        "evidence_contract_consistency_count",
        int_at_least(trace_contract.get("consistency_count"), MIN_EVIDENCE_CONSISTENCY_COUNT),
        str(trace_contract.get("consistency_count")),
    )
    add_check(
        checks,
        "evidence_contract_failed_consistency_zero",
        failed_consistency(trace_contract) == []
        or set(failed_consistency(trace_contract)) <= ALLOWED_BOOTSTRAP_EVIDENCE_FAILURES,
        str(failed_consistency(trace_contract)),
    )
    add_check(
        checks,
        "evidence_contract_no_board_create",
        trace_safety.get("creates_board_result") is False and handoff_safety.get("creates_board_result") is False,
        f"trace={trace_safety.get('creates_board_result')} handoff={handoff_safety.get('creates_board_result')}",
    )
    add_check(
        checks,
        "evidence_contract_no_policy_write",
        trace_safety.get("creates_xr_vits_policy") is False and handoff_safety.get("creates_xr_vits_policy") is False,
        f"trace={trace_safety.get('creates_xr_vits_policy')} handoff={handoff_safety.get('creates_xr_vits_policy')}",
    )
    add_check(
        checks,
        "evidence_contract_no_canonical_write",
        trace_safety.get("writes_canonical_inputs") is False and handoff_safety.get("writes_canonical_inputs") is False,
        f"trace={trace_safety.get('writes_canonical_inputs')} handoff={handoff_safety.get('writes_canonical_inputs')}",
    )
    missing_evidence_ids = sorted(REQUIRED_FINAL_EVIDENCE_IDS - live_artifact_ids)
    add_check(
        checks,
        "final_evidence_manifest_qkv_artifacts_present",
        missing_evidence_ids == [],
        f"missing={missing_evidence_ids}",
    )
    missing_trace_policy_checks = sorted(REQUIRED_TRACE_POLICY_CONSISTENCY_CHECKS - set(live_consistency))
    failing_trace_policy_checks = sorted(
        name
        for name in REQUIRED_TRACE_POLICY_CONSISTENCY_CHECKS
        if live_consistency.get(name, {}).get("status") != "pass"
    )
    add_check(
        checks,
        "final_evidence_manifest_trace_policy_checks_present",
        missing_trace_policy_checks == [],
        f"missing={missing_trace_policy_checks}",
    )
    add_check(
        checks,
        "final_evidence_manifest_trace_policy_checks_pass",
        failing_trace_policy_checks == [],
        f"failing={failing_trace_policy_checks}",
    )
    add_check(
        checks,
        "live_source_audit_pass",
        source_audit.get("status") == "pass",
        str(source_audit.get("status")),
    )
    add_check(
        checks,
        "live_source_audit_required_count",
        int_at_least(source_audit.get("required_count"), MIN_SOURCE_AUDIT_REQUIRED_COUNT),
        str(source_audit.get("required_count")),
    )
    add_check(
        checks,
        "live_source_audit_source_count",
        int_at_least(source_audit.get("source_count"), MIN_SOURCE_AUDIT_SOURCE_COUNT),
        str(source_audit.get("source_count")),
    )
    missing_required = source_audit.get("missing_required", [])
    add_check(
        checks,
        "live_source_audit_missing_zero",
        isinstance(missing_required, list) and missing_required == [],
        str(missing_required),
    )
    add_check(checks, "unblock_sections", {"U1", "U2", "U3", "U4"}.issubset(section_ids(command_card)), str(sorted(section_ids(command_card))))
    safety = command_card.get("safety", {}) if isinstance(command_card.get("safety"), dict) else {}
    add_check(checks, "safety_no_board_result", safety.get("creates_board_result") is False, str(safety.get("creates_board_result")))
    add_check(checks, "safety_no_xr_policy", safety.get("creates_xr_vits_policy") is False, str(safety.get("creates_xr_vits_policy")))
    add_check(
        checks,
        "safety_no_command_exec",
        safety_flag(safety, "executes_commands", "executes_network") is False,
        str(safety_flag(safety, "executes_commands", "executes_network")),
    )
    add_check(
        checks,
        "safety_no_network_exec",
        safety_flag(safety, "executes_network", "executes_commands") is False,
        str(safety_flag(safety, "executes_network", "executes_commands")),
    )
    add_check(checks, "safety_no_canonical_write", safety.get("writes_canonical_inputs") is False, str(safety.get("writes_canonical_inputs")))
    resolution_safety = xr_vits_resolution.get("safety", {}) if isinstance(xr_vits_resolution.get("safety"), dict) else {}
    add_check(
        checks,
        "xr_vits_resolution_no_policy_write",
        resolution_safety.get("creates_xr_vits_policy") is False,
        str(resolution_safety.get("creates_xr_vits_policy")),
    )
    add_check(
        checks,
        "xr_vits_resolution_no_canonical_write",
        resolution_safety.get("writes_canonical_inputs") is False,
        str(resolution_safety.get("writes_canonical_inputs")),
    )

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "root": str(root),
        "final_status": final_status,
        "remaining_blockers": blockers,
        "checks": checks,
        "check_count": len(checks),
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "source_files": {key: str(path) for key, path in paths.items()},
        "safety": safety_payload(),
    }


def safety_payload() -> dict[str, bool]:
    return {
        "executes_commands": False,
        "creates_board_result": False,
        "creates_xr_vits_policy": False,
        "writes_canonical_inputs": False,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Signoff Bundle Validation",
        "",
        f"- status: `{result['status']}`",
        f"- final_status: `{result.get('final_status', 'unknown')}`",
        f"- checks: `{result['check_count']}`",
        f"- pass: `{result['pass_count']}`",
        f"- fail: `{result['fail_count']}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    blockers = result.get("remaining_blockers", [])
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None.")
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
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
    parser = argparse.ArgumentParser(description="Validate final HGTXR signoff evidence bundle.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_bundle(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(result))
    print(f"[final-signoff-bundle-validation] status={result['status']} fail={result['fail_count']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
