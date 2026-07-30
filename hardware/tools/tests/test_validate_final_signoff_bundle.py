#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_final_signoff_bundle as validator  # noqa: E402


BLOCKERS = ["C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"]
RESOURCE_POLICY_CHECKS = [
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
]
SPEC_PLAN_CHECKS = [
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
]
REQ6_CHECKS = [
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
]
CLOSEOUT_VALIDATION_CHECKS = [
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
]
REQ6_KNOBS = {
    "HGTXR_TILING_FACTOR": 1,
    "HGTXR_PARALLELISM_FACTOR": 8,
    "HGTXR_BUS_WIDTH": 256,
    "HGTXR_BIT_WIDTH": 8,
    "HGTXR_WEIGHT_BIT_WIDTH": 4,
    "HGTXR_BUFFER_SIZE": 256,
    "HGTXR_FIFO_DEPTH": 128,
}
CURRENT_AUDIT_SUMMARY = {
    "requirements": 12,
    "reflected": 10,
    "partial": 1,
    "blocked": 1,
    "default_closeout_blockers": 2,
}


def evidence_contract(root: Path, *, required_count: int = 86) -> dict[str, object]:
    return {
        "status": "pass",
        "source": "manifest-json",
        "path": str(root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json"),
        "required_count": required_count,
        "present_required_count": required_count,
        "consistency_count": 347,
        "failed_consistency_checks": [],
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def source_audit_payload(*, required_count: int = 86, source_count: int = 224) -> dict[str, object]:
    return {
        "status": "pass",
        "required_count": required_count,
        "source_count": source_count,
        "missing_required": [],
    }


def resource_policy_payload(*, check_count: int = 36, fail_name: str | None = None) -> dict[str, object]:
    filler_count = max(0, check_count - len(RESOURCE_POLICY_CHECKS))
    checks = [
        {"name": f"resource_check_{index}", "status": "pass", "detail": "pass"}
        for index in range(filler_count)
    ] + [
        {
            "name": name,
            "status": "fail" if name == fail_name else "pass",
            "detail": "test",
        }
        for name in RESOURCE_POLICY_CHECKS
    ]
    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "fail" if fail_count else "pass",
        "check_count": check_count,
        "pass_count": check_count - fail_count,
        "fail_count": fail_count,
        "checks": checks,
    }


def spec_plan_payload(*, check_count: int = 86, fail_name: str | None = None) -> dict[str, object]:
    filler_count = max(0, check_count - len(SPEC_PLAN_CHECKS))
    checks = [
        {"name": f"spec_check_{index}", "status": "pass", "detail": "pass"}
        for index in range(filler_count)
    ] + [
        {
            "name": name,
            "status": "fail" if name == fail_name else "pass",
            "detail": "test",
        }
        for name in SPEC_PLAN_CHECKS
    ]
    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "fail" if fail_count else "pass",
        "check_count": check_count,
        "pass_count": check_count - fail_count,
        "fail_count": fail_count,
        "checks": checks,
        "observed": {
            "evidence_required": 86,
            "evidence_consistency_count": 347,
            "source_required_count": 86,
            "source_count": 224,
        },
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_hls_or_vivado": False,
            "writes_canonical_inputs": False,
        },
    }


def final_runner_payload(
    *,
    status: str = "blocked",
    blocker_count: int = 2,
    detail_count: int = 2,
    duplicate_count: int = 0,
) -> dict[str, object]:
    mirrored = ["docs/resources/final_evidence_manifest_2026_06_10.json", "docs/resources/spec_plan_conformance_audit_2026_06_10.json"]
    if duplicate_count:
        mirrored.append(mirrored[0])
    integrity = [
        {
            "status": "pass",
            "source": path.replace("docs/resources", "hardware/generated/signoff"),
            "mirror": path,
            "source_exists": True,
            "mirror_exists": True,
            "sha256_match": True,
        }
        for path in mirrored
    ]
    return {
        "status": status,
        "blocker_count": blocker_count,
        "remaining_blockers": BLOCKERS,
        "remaining_blocker_detail_count": detail_count,
        "remaining_blocker_details": [{"name": name} for name in BLOCKERS],
        "mirrored_artifact_count": len(mirrored),
        "mirrored_artifact_unique_count": len(set(mirrored)),
        "mirrored_artifact_duplicate_count": duplicate_count,
        "mirrored_artifacts": mirrored,
        "mirrored_artifact_integrity_status": "pass",
        "mirrored_artifact_integrity_count": len(mirrored),
        "mirrored_artifact_integrity_checked_count": len(mirrored),
        "mirrored_artifact_integrity_excluded_count": 0,
        "mirrored_artifact_integrity_fail_count": 0,
        "mirrored_artifact_integrity_failures": [],
        "mirrored_artifact_integrity": integrity,
    }


def req6_parameterization_payload(*, check_count: int = 71, fail_name: str | None = None) -> dict[str, object]:
    filler_count = max(0, check_count - len(REQ6_CHECKS))
    checks = [
        {"name": f"req6_check_{index}", "status": "pass", "detail": "pass"}
        for index in range(filler_count)
    ] + [
        {
            "name": name,
            "status": "fail" if name == fail_name else "pass",
            "detail": "test",
        }
        for name in REQ6_CHECKS
    ]
    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "fail" if fail_count else "pass",
        "check_count": check_count,
        "pass_count": check_count - fail_count,
        "fail_count": fail_count,
        "knobs": {"config_macros": REQ6_KNOBS},
        "checks": checks,
    }


def current_audit_payload(
    root: Path,
    *,
    status: str = "blocked-external",
    summary: dict[str, object] | None = None,
    xr_policy_consistent: bool = True,
    qkv_required: bool = False,
) -> dict[str, object]:
    return {
        "status": status,
        "summary": summary or CURRENT_AUDIT_SUMMARY,
        "external_blocker_names": BLOCKERS,
        "blocked_external_input_paths_by_blocker": {
            "C3b AXIS/DMA physical smoke result": str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
            "requested XR-VITs sibling": "/home/kjm26/project/PRJXR/XR-VITs",
        },
        "remaining_external_input_details": [
            {
                "blocker_name": "C3b AXIS/DMA physical smoke result",
                "kind": "canonical-board-smoke-json",
                "path": str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                "required_for_default_final_signoff": True,
                "status": "missing",
            },
            {
                "blocker_name": "requested XR-VITs sibling",
                "kind": "requested-xr-vits-source-tree",
                "path": "/home/kjm26/project/PRJXR/XR-VITs",
                "required_for_default_final_signoff": True,
                "status": "missing",
            },
        ],
        "xr_vits_policy_integrity": {
            "consistent": xr_policy_consistent,
            "operator_handoff_policy_check_count": 9,
        },
        "qkv_uram_runner": {
            "status": "blocked",
            "qkv_uram_required_for_final_signoff": qkv_required,
            "qkv_uram_import_status": "skipped",
            "qkv_uram_remote_status": "skipped",
            "qkv_uram_remote_execute": False,
            "qkv_uram_smoke_discovery_status": "missing",
        },
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }


def closeout_validation_payload(*, check_count: int = 49, fail_name: str | None = None) -> dict[str, object]:
    filler_count = max(0, check_count - len(CLOSEOUT_VALIDATION_CHECKS))
    checks = [
        {"name": f"closeout_check_{index}", "status": "pass", "detail": "pass"}
        for index in range(filler_count)
    ] + [
        {
            "name": name,
            "status": "fail" if name == fail_name else "pass",
            "detail": "test",
        }
        for name in CLOSEOUT_VALIDATION_CHECKS
    ]
    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "fail" if fail_count else "pass",
        "check_count": check_count,
        "pass_count": check_count - fail_count,
        "fail_count": fail_count,
        "remaining_blockers": BLOCKERS,
        "checks": checks,
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_commands": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }


def closeout_packet_payload(
    root: Path,
    *,
    status: str = "ready-for-operator-unblock",
    board_status: str = "ready-for-board",
    qkv_required: bool = False,
    qkv_commands: bool = True,
) -> dict[str, object]:
    qkv_command_list = [
        "python3 tools/run_third_goal_final_signoff.py --execute-qkv-uram-smoke --allow-blocked",
        "python3 tools/run_third_goal_final_signoff.py --import-qkv-uram-smoke-json /tmp/qkv.json --dry-run-import-qkv-uram-smoke --allow-blocked",
    ] if qkv_commands else []
    return {
        "status": status,
        "remaining_blockers": BLOCKERS,
        "board_package": {
            "status": board_status,
            "preset": "axis-c3b-mem16",
            "expected_runtime_state": 2,
            "expected_out_raw": [32, -13, 26, -6, 14, -11],
            "tar": str(root / "hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"),
            "tar_sha256": "a" * 64,
        },
        "c3b_smoke_contract": {
            "status": "pass",
            "preset": "axis-c3b-mem16",
            "canonical_result_path": str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
            "validate_command": "python3 tools/validate_pynq_smoke_result.py /tmp/smoke.json --preset axis-c3b-mem16",
        },
        "xr_vits_resolution": {
            "status": "candidate-ready-needs-approval",
            "policy_integrity": {
                "required": True,
                "legacy_policy_clears_final_signoff": False,
                "policy_exists": False,
            },
        },
        "qkv_uram_successor_gate": {
            "required_for_final_signoff": qkv_required,
            "physical_smoke_status": "not_captured",
        },
        "operator_commands": {
            "qkv_uram_successor": qkv_command_list,
        },
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_commands": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }


def c3b_physical_gate_payload(
    root: Path,
    *,
    status: str = "blocked_missing_canonical_physical_smoke_result",
    ready_for_board: bool = True,
    physical_smoke_pass: bool = False,
    canonical_status: str = "missing",
    bundle_status: str = "pass",
) -> dict[str, object]:
    return {
        "status": status,
        "ready_for_board": ready_for_board,
        "physical_smoke_pass": physical_smoke_pass,
        "remaining_blockers": ["C3b AXIS/DMA physical smoke result"],
        "canonical_result": {
            "status": canonical_status,
            "exists": canonical_status == "pass",
            "path": str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
        },
        "canonical_validation": {
            "status": "missing",
            "path": str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke_validation.json"),
        },
        "bundle": {
            "status": bundle_status,
            "expected_runtime_state": 2,
            "expected_out_raw": [32, -13, 26, -6, 14, -11],
        },
        "session": {"status": "pass"},
        "safety": {
            "creates_board_result": False,
            "executes_commands": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }


def xr_vits_gate_payload(
    *,
    status: str = "blocked",
    resolution_mode: str = "candidate-ready-needs-approval",
    candidate_exists: bool = True,
    recommendation_matches: bool = True,
    safety_policy_write: bool = False,
) -> dict[str, object]:
    return {
        "status": status,
        "resolution_mode": resolution_mode,
        "remaining_blockers": ["requested XR-VITs sibling"],
        "current_gate": {"status": "missing", "would_clear": False},
        "exact": {
            "exists": False,
            "path": "/home/kjm26/project/PRJXR/XR-VITs",
            "would_clear": False,
        },
        "replacement_candidate": {
            "exists": candidate_exists,
            "path": "/home/kjm26/project/PRJXR/XR-VIT/XR_Accel",
        },
        "active_policy_summary": {"status": "missing", "would_clear": False},
        "candidate_audit": {
            "status": "candidate-found",
            "recommendation_matches": recommendation_matches,
            "recommendation": {"score": 99},
        },
        "safety": {
            "creates_xr_vits_policy": safety_policy_write,
            "executes_commands": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }


def evidence_manifest(root: Path) -> dict[str, object]:
    artifact_ids = [
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
    ]
    required_policy_checks = [
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
        "vref_p0_pot_scale_sweep_current_recommended",
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
        "pynq_smoke_validator_overlay_prefix_contracts",
        "pynq_smoke_validator_basename_check",
        "pynq_smoke_validator_require_paths_gate",
        "final_runner_remaining_blocker_input_paths_contract",
        "final_runner_remaining_blocker_details_contract",
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
        "xr_vits_policy_tool_placeholder_guard_contract",
        "xr_vits_policy_tool_dry_run_placeholder_allowed",
        "final_runner_active_policy_failure_blocks",
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
    ]
    filler_checks = [
        {"name": f"check_{index}", "status": "pass"}
        for index in range(347 - len(required_policy_checks))
    ]
    return {
        "status": "pass",
        "root": str(root),
        "required_count": 86,
        "present_required_count": 86,
        "failed_consistency_checks": [],
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
        "consistency_checks": filler_checks + [{"name": name, "status": "pass"} for name in required_policy_checks],
        "artifacts": [{"id": artifact_id, "required": True, "status": "pass"} for artifact_id in artifact_ids],
    }


def policy_integrity(root: Path) -> dict[str, object]:
    return {
        "required": True,
        "policy_path": str(root / "docs/resources/xr_vits_replacement_policy.json"),
        "candidate_audit": str(root / "docs/resources/xr_vits_candidate_audit_2026_06_10.json"),
        "generator": "tools/create_xr_vits_replacement_policy.py",
        "validator": "tools/check_final_blocker_closure_readiness.py",
        "required_policy_fields": [
            "candidate_audit_fingerprint",
            "candidate_audit_recommendation_snapshot",
            "candidate_audit_meta",
            "approval_event",
            "policy_fingerprint",
        ],
        "legacy_policy_clears_final_signoff": False,
        "policy_exists": False,
        "validation": {
            "status": "pending-policy-creation",
            "policy_exists": False,
            "candidate_audit_path": str(root / "docs/resources/xr_vits_candidate_audit_2026_06_10.json"),
            "errors": [],
        },
    }


def operator_handoff_payload(root: Path, *, status: str = "pending-operator-actions", blockers: list[str] | None = None) -> dict[str, object]:
    return {
        "status": status,
        "root": str(root),
        "remaining_blockers": BLOCKERS if blockers is None else blockers,
        "xr_vits": {
            "reference_resolution": {
                "status": "candidate-ready-needs-approval",
                "policy_integrity": policy_integrity(root),
            }
        },
        "final_evidence_manifest_contract": evidence_contract(root),
    }


def populate_bundle(root: Path) -> dict[str, Path]:
    paths = validator.build_bundle_paths(root)
    contract = evidence_contract(root)
    write_json(
        paths["final_audit"],
        {
            "status": "blocked",
            "blocker_count": 2,
            "blockers": [{"name": "requested XR-VITs sibling"}, {"name": "C3b AXIS/DMA physical smoke result"}],
        },
    )
    write_json(paths["completion"], {"status": "blocked"})
    write_json(paths["unblock"], {"status": "pending-unblock"})
    write_json(
        paths["command_card"],
        {
            "status": "pending-unblock",
            "remaining_blockers": BLOCKERS,
            "xr_vits_policy_integrity": policy_integrity(root),
            "sections": [{"id": "U1"}, {"id": "U2"}, {"id": "U3"}, {"id": "U4"}],
            "safety": {
                "creates_board_result": False,
                "creates_xr_vits_policy": False,
                "executes_commands": False,
                "executes_network": False,
                "writes_canonical_inputs": False,
            },
        },
    )
    write_json(paths["resource_matrix"], {"status": "pass", "summary": {"recommended_board_smoke_variant": "C3b"}})
    write_json(
        paths["requirements_trace"],
        {
            "status": "blocked",
            "final_signoff": {"status": "blocked", "remaining_blockers": BLOCKERS},
            "final_evidence_manifest_contract": contract,
        },
    )
    write_json(paths["candidate_audit"], {"status": "blocked", "remaining_blockers": BLOCKERS})
    write_json(
        paths["xr_vits_reference_resolution"],
        {
            "status": "candidate-ready-needs-approval",
            "resolution_ready": False,
            "safety": {
                "creates_xr_vits_policy": False,
                "writes_canonical_inputs": False,
            },
        },
    )
    write_json(paths["operator_handoff"], operator_handoff_payload(root))
    write_json(paths["operator_handoff_validation"], {"status": "pass", "fail_count": 0, "check_count": 131})
    write_json(paths["final_evidence_manifest"], evidence_manifest(root))
    write_json(paths["final_runner"], final_runner_payload())
    write_json(paths["closeout_packet"], closeout_packet_payload(root))
    write_json(paths["closeout_validation"], closeout_validation_payload())
    write_json(paths["source_audit"], source_audit_payload())
    write_json(paths["resource_policy"], resource_policy_payload())
    write_json(paths["spec_plan"], spec_plan_payload())
    write_json(paths["req6_parameterization"], req6_parameterization_payload())
    write_json(paths["current_audit"], current_audit_payload(root))
    write_json(paths["c3b_physical_gate"], c3b_physical_gate_payload(root))
    write_json(paths["xr_vits_gate"], xr_vits_gate_payload())
    return paths


class ValidateFinalSignoffBundleTests(unittest.TestCase):
    def test_consistent_blocked_bundle_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "pass")
            self.assertGreaterEqual(result["check_count"], 95)
            self.assertEqual(result["fail_count"], 0)
            self.assertEqual(set(result["remaining_blockers"]), set(BLOCKERS))
            self.assertFalse(result["safety"]["executes_commands"])
            self.assertFalse(result["safety"]["creates_board_result"])

    def test_xr_vits_resolution_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["final_audit"], {"status": "pass", "blocker_count": 0, "blockers": []})
            write_json(paths["completion"], {"status": "pass"})
            write_json(paths["unblock"], {"status": "ready-for-final-signoff"})
            write_json(
                paths["command_card"],
                {
                    "status": "ready-for-final-signoff",
                    "remaining_blockers": [],
                    "xr_vits_policy_integrity": policy_integrity(root),
                    "sections": [{"id": "U1"}, {"id": "U2"}, {"id": "U3"}, {"id": "U4"}],
                    "safety": {
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                        "executes_commands": False,
                        "executes_network": False,
                        "writes_canonical_inputs": False,
                    },
                },
            )
            write_json(
                paths["requirements_trace"],
                {
                    "status": "pass",
                    "final_signoff": {"status": "pass", "remaining_blockers": []},
                    "final_evidence_manifest_contract": evidence_contract(root),
                },
            )
            write_json(paths["candidate_audit"], {"status": "would-clear", "remaining_blockers": []})
            write_json(paths["operator_handoff"], operator_handoff_payload(root, status="ready-for-final-signoff", blockers=[]))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("xr_vits_resolution_matches_blocker", failed)

    def test_blocker_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["operator_handoff"], operator_handoff_payload(root, blockers=["requested XR-VITs sibling"]))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("handoff_blockers_match", failed)

    def test_stale_operator_handoff_validation_check_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["operator_handoff_validation"], {"status": "pass", "fail_count": 0, "check_count": 128})

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("handoff_validation_check_count", failed)

    def test_stale_closeout_packet_status_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["closeout_packet"], closeout_packet_payload(root, status="stale"))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("closeout_packet_ready", failed)

    def test_closeout_packet_qkv_command_drift_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["closeout_packet"], closeout_packet_payload(root, qkv_commands=False))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("closeout_packet_qkv_optional_commands", failed)

    def test_c3b_physical_gate_drift_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["c3b_physical_gate"], c3b_physical_gate_payload(root, ready_for_board=False))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("c3b_physical_gate_current_blocker_contract", failed)

    def test_xr_vits_gate_candidate_drift_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["xr_vits_gate"], xr_vits_gate_payload(candidate_exists=False))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("xr_vits_gate_candidate_contract", failed)

    def test_stale_closeout_validation_check_count_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["closeout_validation"], closeout_validation_payload(check_count=48))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("closeout_validation_check_count", failed)

    def test_closeout_validation_core_failure_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["closeout_validation"], closeout_validation_payload(fail_name="board_package_ready"))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("closeout_validation_pass", failed)
            self.assertIn("closeout_validation_fail_zero", failed)
            self.assertIn("closeout_validation_core_checks_pass", failed)

    def test_evidence_contract_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            bad_contract = evidence_contract(root)
            bad_contract["failed_consistency_checks"] = ["operator_handoff_validation_pass"]
            write_json(
                paths["operator_handoff"],
                {
                    "status": "pending-operator-actions",
                    "root": str(root),
                    "remaining_blockers": BLOCKERS,
                    "xr_vits": {
                        "reference_resolution": {
                            "status": "candidate-ready-needs-approval",
                            "policy_integrity": policy_integrity(root),
                        }
                    },
                    "final_evidence_manifest_contract": bad_contract,
                },
            )

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("handoff_evidence_contract_matches_trace", failed)

    def test_stale_56_of_56_contract_fails_against_live_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            stale_contract = evidence_contract(root, required_count=56)
            write_json(
                paths["requirements_trace"],
                {
                    "status": "blocked",
                    "final_signoff": {"status": "blocked", "remaining_blockers": BLOCKERS},
                    "final_evidence_manifest_contract": stale_contract,
                },
            )
            write_json(
                paths["operator_handoff"],
                {
                    "status": "pending-operator-actions",
                    "root": str(root),
                    "remaining_blockers": BLOCKERS,
                    "xr_vits": {
                        "reference_resolution": {
                            "status": "candidate-ready-needs-approval",
                            "policy_integrity": policy_integrity(root),
                        }
                    },
                    "final_evidence_manifest_contract": stale_contract,
                },
            )

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("trace_evidence_contract_matches_manifest", failed)
            self.assertIn("handoff_evidence_contract_matches_manifest", failed)

    def test_stale_340_consistency_contract_fails_even_when_artifacts_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "check_0"
            ]
            write_json(paths["final_evidence_manifest"], manifest)
            stale_contract = evidence_contract(root)
            stale_contract["consistency_count"] = len(manifest["consistency_checks"])
            write_json(
                paths["requirements_trace"],
                {
                    "status": "blocked",
                    "final_signoff": {"status": "blocked", "remaining_blockers": BLOCKERS},
                    "final_evidence_manifest_contract": stale_contract,
                },
            )
            write_json(
                paths["operator_handoff"],
                {
                    "status": "pending-operator-actions",
                    "root": str(root),
                    "remaining_blockers": BLOCKERS,
                    "xr_vits": {
                        "reference_resolution": {
                            "status": "candidate-ready-needs-approval",
                            "policy_integrity": policy_integrity(root),
                        }
                    },
                    "final_evidence_manifest_contract": stale_contract,
                },
            )

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertNotIn("trace_evidence_contract_matches_manifest", failed)
            self.assertNotIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("evidence_contract_consistency_count", failed)

    def test_stale_85_required_contract_fails_even_when_artifacts_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["required_count"] = 85
            manifest["present_required_count"] = 85
            write_json(paths["final_evidence_manifest"], manifest)
            stale_contract = evidence_contract(root, required_count=85)
            write_json(
                paths["requirements_trace"],
                {
                    "status": "blocked",
                    "final_signoff": {"status": "blocked", "remaining_blockers": BLOCKERS},
                    "final_evidence_manifest_contract": stale_contract,
                },
            )
            write_json(
                paths["operator_handoff"],
                {
                    "status": "pending-operator-actions",
                    "root": str(root),
                    "remaining_blockers": BLOCKERS,
                    "xr_vits": {
                        "reference_resolution": {
                            "status": "candidate-ready-needs-approval",
                            "policy_integrity": policy_integrity(root),
                        }
                    },
                    "final_evidence_manifest_contract": stale_contract,
                },
            )

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertNotIn("trace_evidence_contract_matches_manifest", failed)
            self.assertNotIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("evidence_contract_required_complete", failed)

    def test_missing_trace_policy_manifest_checks_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "third_goal_requirements_trace_xr_vits_policy_fields_complete"
            ]
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("trace_evidence_contract_matches_manifest", failed)
            self.assertIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_pynq_overlay_provenance_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "pynq_smoke_validator_basename_check"
            ]
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("trace_evidence_contract_matches_manifest", failed)
            self.assertIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_qkv_discovery_semantic_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "qkv_uram_smoke_discovery_safe_no_side_effects"
            ]
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("trace_evidence_contract_matches_manifest", failed)
            self.assertIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_runner_blocker_summary_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "final_runner_remaining_blocker_input_paths_present"
            ]
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("trace_evidence_contract_matches_manifest", failed)
            self.assertIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_runner_summary_count_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "final_runner_mirrored_artifact_counts_match"
            ]
            manifest["consistency_checks"].append(
                {"name": "replacement_filler_keeps_count_326_runner_summary", "status": "pass"}
            )
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_source_audit_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "third_goal_source_audit_source_count"
            ]
            manifest["consistency_checks"].append(
                {"name": "replacement_filler_keeps_count_326_source_audit", "status": "pass"}
            )
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_stale_live_source_audit_counts_fail_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["source_audit"], source_audit_payload(required_count=85, source_count=223))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("live_source_audit_required_count", failed)
            self.assertIn("live_source_audit_source_count", failed)

    def test_stale_live_resource_policy_count_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["resource_policy"], resource_policy_payload(check_count=35))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("resource_policy_check_count", failed)

    def test_live_resource_policy_core_failure_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["resource_policy"], resource_policy_payload(fail_name="c3b_csynth_lut_lte_threshold"))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("resource_policy_pass", failed)
            self.assertIn("resource_policy_fail_zero", failed)
            self.assertIn("resource_policy_core_checks_pass", failed)

    def test_stale_live_spec_plan_count_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["spec_plan"], spec_plan_payload(check_count=65))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("spec_plan_check_count", failed)

    def test_live_spec_plan_core_failure_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["spec_plan"], spec_plan_payload(fail_name="execution_records_xilinx_root"))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("spec_plan_pass", failed)
            self.assertIn("spec_plan_core_checks_pass", failed)

    def test_stale_live_final_runner_counts_fail_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["final_runner"], final_runner_payload(blocker_count=1, detail_count=1))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("final_runner_blocker_count", failed)
            self.assertIn("final_runner_detail_count", failed)

    def test_live_final_runner_duplicate_mirror_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["final_runner"], final_runner_payload(duplicate_count=1))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("final_runner_mirror_counts", failed)

    def test_stale_live_req6_parameterization_count_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["req6_parameterization"], req6_parameterization_payload(check_count=70))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("req6_parameterization_check_count", failed)

    def test_live_req6_parameterization_core_failure_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(
                paths["req6_parameterization"],
                req6_parameterization_payload(fail_name="parallelism_extension_par32_requires_fresh_reports"),
            )

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("req6_parameterization_pass", failed)
            self.assertIn("req6_parameterization_fail_zero", failed)
            self.assertIn("req6_parameterization_core_checks_pass", failed)

    def test_stale_live_current_audit_summary_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            stale_summary = dict(CURRENT_AUDIT_SUMMARY)
            stale_summary["reflected"] = 9
            write_json(paths["current_audit"], current_audit_payload(root, summary=stale_summary))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("current_audit_summary_counts", failed)

    def test_live_current_audit_policy_drift_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["current_audit"], current_audit_payload(root, xr_policy_consistent=False))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("current_audit_xr_policy_integrity", failed)

    def test_live_current_audit_qkv_required_drift_fails_even_when_manifest_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            write_json(paths["current_audit"], current_audit_payload(root, qkv_required=True))

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("current_audit_qkv_optional_state", failed)

    def test_missing_blocker_readiness_discovery_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "final_blocker_closure_qkv_uram_discovery_safe_no_side_effects"
            ]
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("trace_evidence_contract_matches_manifest", failed)
            self.assertIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_c3b_transfer_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "c3b_transfer_manifest_sha_matches_readiness"
            ]
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("trace_evidence_contract_matches_manifest", failed)
            self.assertIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_req5_packed_weight_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "req5_q4q8_packed_weight_sha256_matches_manifest"
            ]
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("trace_evidence_contract_matches_manifest", failed)
            self.assertIn("handoff_evidence_contract_matches_manifest", failed)
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_cyclic_resource_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "resource_policy_audit_cyclic_weight_tiles_uram_pragmas"
            ]
            manifest["consistency_checks"].append(
                {"name": "replacement_filler_keeps_count_321_cyclic", "status": "pass"}
            )
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_missing_c3b_threshold_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            manifest["consistency_checks"] = [
                check
                for check in manifest["consistency_checks"]
                if check["name"] != "resource_policy_audit_c3b_routed_wns_gte_threshold"
            ]
            manifest["consistency_checks"].append(
                {"name": "replacement_filler_keeps_count_321_c3b_threshold", "status": "pass"}
            )
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("final_evidence_manifest_trace_policy_checks_present", failed)

    def test_failed_trace_policy_manifest_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = populate_bundle(root)
            manifest = evidence_manifest(root)
            for check in manifest["consistency_checks"]:
                if check["name"] == "third_goal_requirements_trace_xr_vits_legacy_policy_rejected":
                    check["status"] = "fail"
            write_json(paths["final_evidence_manifest"], manifest)

            result = validator.validate_bundle(root, paths)

            self.assertEqual(result["status"], "fail")
            failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
            self.assertIn("final_evidence_manifest_trace_policy_checks_pass", failed)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            populate_bundle(root)
            json_out = root / "generated" / "bundle.json"
            markdown_out = root / "generated" / "bundle.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = validator.main(
                    [
                        "--root",
                        str(root),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
            self.assertIn("HGTXR Final Signoff Bundle Validation", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
