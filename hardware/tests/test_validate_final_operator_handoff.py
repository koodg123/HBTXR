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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_final_operator_handoff as validator  # noqa: E402
import write_final_operator_handoff as handoff_tool  # noqa: E402
from test_write_final_operator_handoff import sample_readiness, sample_trace  # noqa: E402
from test_write_third_goal_requirements_trace import sample_candidate_audit, sample_xr_vits_resolution  # noqa: E402


REQ5_CHECKS = [
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
]
CYCLIC_RESOURCE_CHECKS = [
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
]
C3B_THRESHOLD_CHECKS = [
    "resource_policy_audit_c3b_csynth_lut_lte_threshold",
    "resource_policy_audit_c3b_csynth_latency_lte_threshold",
    "resource_policy_audit_c3b_routed_wns_gte_threshold",
]
FINAL_RUNNER_SUMMARY_CHECKS = [
    "final_runner_remaining_blockers_present",
    "final_runner_blocker_count_matches_list",
    "final_runner_remaining_blocker_detail_count_matches",
    "final_runner_mirrored_artifact_counts_match",
    "final_runner_mirrored_artifact_integrity_present",
    "final_runner_mirrored_artifact_integrity_pass",
    "final_runner_mirrored_artifact_integrity_counts_match",
]
SOURCE_AUDIT_CHECKS = [
    "third_goal_source_audit_required_count",
    "third_goal_source_audit_source_count",
]
UNBLOCK_INTAKE_OPERATOR_CHECKS = [
    "final_unblock_intake_next_inputs_match_operator_plan_required_inputs",
    "final_unblock_intake_next_input_paths_match_operator_plan",
    "final_unblock_intake_operator_sequence_contains_expected_steps",
    "final_unblock_intake_operator_sequence_matches_operator_plan_commands",
    "final_unblock_intake_operator_sequence_side_effect_profile",
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


def evidence_contract(root: Path, *, consistency_count: int = 347, required_count: int = 86) -> dict[str, object]:
    return {
        "status": "pass",
        "source": "manifest-json",
        "path": str(root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json"),
        "required_count": required_count,
        "present_required_count": required_count,
        "consistency_count": consistency_count,
        "failed_consistency_checks": [],
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def write_live_manifest(root: Path, *, consistency_count: int = 347, required_count: int = 86) -> None:
    named_checks = (
        REQ5_CHECKS
        + CYCLIC_RESOURCE_CHECKS
        + C3B_THRESHOLD_CHECKS
        + FINAL_RUNNER_SUMMARY_CHECKS
        + SOURCE_AUDIT_CHECKS
        + UNBLOCK_INTAKE_OPERATOR_CHECKS
    )
    filler_count = max(0, consistency_count - len(named_checks))
    payload = {
        "status": "pass",
        "root": str(root),
        "required_count": required_count,
        "present_required_count": required_count,
        "failed_consistency_checks": [],
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
        "consistency_checks": [
            {"name": f"check_{index}", "status": "pass", "detail": "pass"}
            for index in range(filler_count)
        ] + [{"name": name, "status": "pass", "detail": "pass"} for name in named_checks],
    }
    path = root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_live_source_audit(root: Path, *, required_count: int = 86, source_count: int = 224) -> None:
    path = root / "docs" / "resources" / "third_goal_source_audit_2026_06_16.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "required_count": required_count,
                "source_count": source_count,
                "missing_required": [],
            }
        )
        + "\n"
    )


def write_live_resource_policy(root: Path, *, check_count: int = 36, fail_name: str | None = None) -> None:
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
    path = root / "docs" / "resources" / "e2e_resource_policy_audit_2026_06_10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "fail" if fail_count else "pass",
                "check_count": check_count,
                "pass_count": check_count - fail_count,
                "fail_count": fail_count,
                "checks": checks,
            }
        )
        + "\n"
    )


def write_live_spec_plan(root: Path, *, check_count: int = 86, fail_name: str | None = None) -> None:
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
    path = root / "docs" / "resources" / "spec_plan_conformance_audit_2026_06_10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
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
        )
        + "\n"
    )


def write_live_final_runner(
    root: Path,
    *,
    status: str = "blocked",
    blocker_count: int = 2,
    detail_count: int = 2,
    duplicate_count: int = 0,
) -> None:
    mirrored = ["docs/resources/final_evidence_manifest_2026_06_10.json", "docs/resources/spec_plan_conformance_audit_2026_06_10.json"]
    if duplicate_count:
        mirrored.append(mirrored[0])
    integrity = [
        {
            "status": "pass",
            "source": item.replace("docs/resources", "hardware/generated/signoff"),
            "mirror": item,
            "source_exists": True,
            "mirror_exists": True,
            "sha256_match": True,
        }
        for item in mirrored
    ]
    path = root / "docs" / "resources" / "third_goal_final_signoff_run_2026_06_10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "blocker_count": blocker_count,
                "remaining_blockers": ["requested XR-VITs sibling", "C3b AXIS/DMA physical smoke result"],
                "remaining_blocker_detail_count": detail_count,
                "remaining_blocker_details": [
                    {"name": "requested XR-VITs sibling"},
                    {"name": "C3b AXIS/DMA physical smoke result"},
                ],
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
        )
        + "\n"
    )


def write_live_req6_parameterization(root: Path, *, check_count: int = 71, fail_name: str | None = None) -> None:
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
    path = root / "docs" / "resources" / "req6_parameterization_audit_2026_06_16.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "fail" if fail_count else "pass",
                "check_count": check_count,
                "pass_count": check_count - fail_count,
                "fail_count": fail_count,
                "knobs": {"config_macros": REQ6_KNOBS},
                "checks": checks,
            }
        )
        + "\n"
    )


def write_live_current_audit(
    root: Path,
    *,
    status: str = "blocked-external",
    summary: dict[str, object] | None = None,
    xr_policy_consistent: bool = True,
    qkv_required: bool = False,
) -> None:
    payload = {
        "status": status,
        "summary": summary or CURRENT_AUDIT_SUMMARY,
        "external_blocker_names": ["C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"],
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
    path = root / "docs" / "resources" / "third_goal_current_audit_2026_06_16.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_live_closeout_validation(root: Path, *, check_count: int = 49, fail_name: str | None = None) -> None:
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
    payload = {
        "status": "fail" if fail_count else "pass",
        "check_count": check_count,
        "pass_count": check_count - fail_count,
        "fail_count": fail_count,
        "remaining_blockers": ["C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"],
        "checks": checks,
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_commands": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }
    path = root / "docs" / "resources" / "final_unblock_closeout_packet_validation_2026_06_10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_live_closeout_packet(
    root: Path,
    *,
    status: str = "ready-for-operator-unblock",
    board_status: str = "ready-for-board",
    qkv_required: bool = False,
    qkv_commands: bool = True,
) -> None:
    qkv_command_list = [
        "python3 tools/run_third_goal_final_signoff.py --execute-qkv-uram-smoke --allow-blocked",
        "python3 tools/run_third_goal_final_signoff.py --import-qkv-uram-smoke-json /tmp/qkv.json --dry-run-import-qkv-uram-smoke --allow-blocked",
    ] if qkv_commands else []
    payload = {
        "status": status,
        "remaining_blockers": ["C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"],
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
    path = root / "docs" / "resources" / "final_unblock_closeout_packet_2026_06_10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_live_c3b_physical_gate(
    root: Path,
    *,
    status: str = "blocked_missing_canonical_physical_smoke_result",
    ready_for_board: bool = True,
    physical_smoke_pass: bool = False,
    canonical_status: str = "missing",
    bundle_status: str = "pass",
) -> None:
    payload = {
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
    path = root / "docs" / "resources" / "c3b_physical_smoke_gate_audit_2026_06_16.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_live_xr_vits_gate(
    root: Path,
    *,
    status: str = "blocked",
    resolution_mode: str = "candidate-ready-needs-approval",
    candidate_exists: bool = True,
    recommendation_matches: bool = True,
    safety_policy_write: bool = False,
) -> None:
    payload = {
        "status": status,
        "resolution_mode": resolution_mode,
        "remaining_blockers": ["requested XR-VITs sibling"],
        "current_gate": {
            "status": "missing",
            "would_clear": False,
        },
        "exact": {
            "exists": False,
            "path": "/home/kjm26/project/PRJXR/XR-VITs",
            "would_clear": False,
        },
        "replacement_candidate": {
            "exists": candidate_exists,
            "path": "/home/kjm26/project/PRJXR/XR-VIT/XR_Accel",
        },
        "active_policy_summary": {
            "status": "missing",
            "would_clear": False,
        },
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
    path = root / "docs" / "resources" / "xr_vits_gate_audit_2026_06_16.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_live_xr_resolution(root: Path) -> None:
    path = root / "docs" / "resources" / "xr_vits_reference_resolution_2026_06_10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample_xr_vits_resolution()) + "\n")


def write_live_intake(root: Path) -> None:
    payload = {
        "status": "blocked",
        "blocker_status": {
            "C3b AXIS/DMA physical smoke result": {
                "candidate_status": "missing",
                "candidate_would_clear": False,
                "ready_for_active_unblock": False,
            },
            "requested XR-VITs sibling": {
                "candidate_status": "fail",
                "candidate_would_clear": False,
                "ready_for_active_unblock": False,
            },
        },
        "next_inputs": [
            {"blocker": "C3b AXIS/DMA physical smoke result"},
            {"blocker": "requested XR-VITs sibling"},
        ],
    }
    path = root / "docs" / "resources" / "final_unblock_intake_2026_06_10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def trace_with_contract(root: Path, *, consistency_count: int = 347) -> dict[str, object]:
    trace = sample_trace()
    trace["final_evidence_manifest_contract"] = evidence_contract(root, consistency_count=consistency_count)
    return trace


def validator_command_card(root: Path) -> dict[str, object]:
    return {
        "xr_vits_policy_integrity": {
            "required": True,
            "policy_path": str(root / "docs" / "resources" / "xr_vits_replacement_policy.json"),
            "candidate_audit": str(root / "docs" / "resources" / "xr_vits_candidate_audit_2026_06_10.json"),
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
        },
        "sections": [
            {
                "id": "U1",
                "title": "Run board smoke",
                "status": "pending-board-run",
                "reason": "needs board",
                "commands": [
                    f"python3 tools/run_zcu104_c3b_smoke_remote.py --root {root} --host <host> --execute",
                    f"python3 tools/run_third_goal_final_signoff.py --root {root} --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json",
                    "python3 tools/validate_pynq_smoke_result.py pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
                ],
            },
            {
                "id": "U2",
                "title": "Resolve XR-VITs",
                "status": "pending-user-choice",
                "reason": "needs choice",
                "options": [
                    {"id": "U2a", "commands": ["test -d /home/kjm26/project/PRJXR/XR-VITs"]},
                    {
                        "id": "U2b",
                        "commands": [
                            "python3 tools/create_xr_vits_replacement_policy.py --dry-run",
                            "python3 tools/create_xr_vits_replacement_policy.py --approve --approved-by <approved-by>",
                        ],
                    },
                ],
            },
            {
                "id": "U4",
                "title": "Combined one-shot unblock",
                "status": "pending-user-choice",
                "reason": "combined",
                "options": [
                    {
                        "id": "U4a",
                        "commands": [
                            "python3 tools/run_third_goal_final_signoff.py --root /tmp/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json"
                        ],
                    },
                    {
                        "id": "U4b",
                        "commands": [
                            f"python3 tools/run_third_goal_final_signoff.py --root {root} --dry-run-import-c3b-smoke --approve-xr-vits-replacement",
                            "python3 tools/create_xr_vits_replacement_policy.py --dry-run",
                            "python3 tools/create_xr_vits_replacement_policy.py --approve",
                        ],
                    },
                ],
            },
            {
                "id": "U3",
                "title": "Run final signoff",
                "status": "pending",
                "reason": "after unblock",
                "commands": [f"python3 tools/run_third_goal_final_signoff.py --root {root}"],
                "expected_result": "status=pass, final_preflight_failures=0",
            },
        ]
    }


def sample_handoff(root: Path | None = None, *, write_manifest: bool = True) -> dict[str, object]:
    root = root or (Path(tempfile.mkdtemp()) / "HGTXR")
    if write_manifest:
        write_live_manifest(root)
        write_live_source_audit(root)
        write_live_resource_policy(root)
        write_live_spec_plan(root)
        write_live_final_runner(root)
        write_live_req6_parameterization(root)
        write_live_current_audit(root)
        write_live_closeout_packet(root)
        write_live_closeout_validation(root)
        write_live_c3b_physical_gate(root)
        write_live_xr_vits_gate(root)
        write_live_xr_resolution(root)
        write_live_intake(root)
    handoff = handoff_tool.build_handoff(
        root=root,
        command_card=validator_command_card(root),
        requirements_trace=trace_with_contract(root),
        readiness=sample_readiness(),
        candidate_audit=sample_candidate_audit(),
        xr_vits_resolution=sample_xr_vits_resolution(),
    )
    return handoff


class ValidateFinalOperatorHandoffTests(unittest.TestCase):
    def test_valid_handoff_passes_without_side_effects(self) -> None:
        result = validator.validate_handoff(sample_handoff())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["fail_count"], 0)
        self.assertGreaterEqual(result["check_count"], 79)
        self.assertFalse(result["safety"]["executes_commands"])
        self.assertFalse(result["safety"]["creates_board_result"])
        self.assertFalse(result["safety"]["creates_xr_vits_policy"])

    def test_invalid_xr_resolution_fails(self) -> None:
        handoff = sample_handoff()
        handoff["xr_vits"]["reference_resolution"]["candidate_score"] = 0
        handoff["xr_vits"]["reference_resolution"]["replacement_dry_run_command"] = ""

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("xr_resolution_candidate_score", failed)
        self.assertIn("xr_resolution_dry_run_command", failed)

    def test_invalid_evidence_manifest_contract_fails(self) -> None:
        handoff = sample_handoff()
        handoff["final_evidence_manifest_contract"]["status"] = "fail"
        handoff["final_evidence_manifest_contract"]["present_required_count"] = 35
        handoff["final_evidence_manifest_contract"]["failed_consistency_checks"] = ["final_bundle_validation_pass"]

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("evidence_manifest_contract_pass", failed)
        self.assertIn("evidence_manifest_required_complete", failed)
        self.assertIn("evidence_manifest_failed_consistency_zero", failed)

    def test_stale_evidence_manifest_consistency_count_fails(self) -> None:
        handoff = sample_handoff()
        handoff["final_evidence_manifest_contract"]["consistency_count"] = 340

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("evidence_manifest_consistency_count", failed)
        self.assertIn("evidence_manifest_contract_matches_live", failed)

    def test_stale_evidence_manifest_required_count_fails(self) -> None:
        handoff = sample_handoff()
        handoff["final_evidence_manifest_contract"]["required_count"] = 85
        handoff["final_evidence_manifest_contract"]["present_required_count"] = 85

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("evidence_manifest_required_complete", failed)
        self.assertIn("evidence_manifest_contract_matches_live", failed)

    def test_missing_live_evidence_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handoff = sample_handoff(Path(tmp) / "HGTXR", write_manifest=False)

            result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_evidence_manifest_contract_present", failed)
        self.assertIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_evidence_manifest_req5_checks_present", failed)
        self.assertIn("live_evidence_manifest_cyclic_resource_checks_present", failed)

    def test_missing_req5_live_manifest_check_fails_even_when_contract_count_matches(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        path = root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json"
        live = json.loads(path.read_text())
        live["consistency_checks"] = [
            check
            for check in live["consistency_checks"]
            if check["name"] != "req5_q4q8_packed_weight_sha256_matches_manifest"
        ]
        live["consistency_checks"].append(
            {"name": "replacement_filler_keeps_count_321", "status": "pass", "detail": "pass"}
        )
        path.write_text(json.dumps(live) + "\n")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_evidence_manifest_req5_checks_present", failed)

    def test_missing_cyclic_resource_live_manifest_check_fails_even_when_contract_count_matches(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        path = root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json"
        live = json.loads(path.read_text())
        live["consistency_checks"] = [
            check
            for check in live["consistency_checks"]
            if check["name"] != "resource_policy_audit_cyclic_weight_tiles_uram_pragmas"
        ]
        live["consistency_checks"].append(
            {"name": "replacement_filler_keeps_count_321_cyclic", "status": "pass", "detail": "pass"}
        )
        path.write_text(json.dumps(live) + "\n")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_evidence_manifest_cyclic_resource_checks_present", failed)

    def test_missing_c3b_threshold_live_manifest_check_fails_even_when_contract_count_matches(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        path = root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json"
        live = json.loads(path.read_text())
        live["consistency_checks"] = [
            check
            for check in live["consistency_checks"]
            if check["name"] != "resource_policy_audit_c3b_routed_wns_gte_threshold"
        ]
        live["consistency_checks"].append(
            {"name": "replacement_filler_keeps_count_321_c3b_threshold", "status": "pass", "detail": "pass"}
        )
        path.write_text(json.dumps(live) + "\n")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_evidence_manifest_c3b_threshold_checks_present", failed)

    def test_missing_runner_summary_live_manifest_check_fails_even_when_contract_count_matches(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        path = root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json"
        live = json.loads(path.read_text())
        live["consistency_checks"] = [
            check
            for check in live["consistency_checks"]
            if check["name"] != "final_runner_mirrored_artifact_counts_match"
        ]
        live["consistency_checks"].append(
            {"name": "replacement_filler_keeps_count_326_runner_summary", "status": "pass", "detail": "pass"}
        )
        path.write_text(json.dumps(live) + "\n")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_evidence_manifest_runner_summary_checks_present", failed)

    def test_missing_source_audit_live_manifest_check_fails_even_when_contract_count_matches(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        path = root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json"
        live = json.loads(path.read_text())
        live["consistency_checks"] = [
            check
            for check in live["consistency_checks"]
            if check["name"] != "third_goal_source_audit_source_count"
        ]
        live["consistency_checks"].append(
            {"name": "replacement_filler_keeps_count_326_source_audit", "status": "pass", "detail": "pass"}
        )
        path.write_text(json.dumps(live) + "\n")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_evidence_manifest_source_audit_checks_present", failed)

    def test_stale_live_source_audit_counts_fail_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_source_audit(root, required_count=85, source_count=223)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_source_audit_required_count", failed)
        self.assertIn("live_source_audit_source_count", failed)

    def test_stale_live_resource_policy_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_resource_policy(root, check_count=35)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_resource_policy_check_count", failed)

    def test_live_resource_policy_core_failure_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_resource_policy(root, fail_name="c3b_csynth_lut_lte_threshold")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_resource_policy_pass", failed)
        self.assertIn("live_resource_policy_fail_zero", failed)
        self.assertIn("live_resource_policy_core_checks_pass", failed)

    def test_stale_live_spec_plan_counts_fail_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_spec_plan(root, check_count=65)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_spec_plan_check_count", failed)

    def test_live_spec_plan_core_failure_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_spec_plan(root, fail_name="execution_records_xilinx_root")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_spec_plan_pass", failed)
        self.assertIn("live_spec_plan_core_checks_pass", failed)

    def test_stale_live_final_runner_counts_fail_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_final_runner(root, blocker_count=1, detail_count=1)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_final_runner_blocker_count", failed)
        self.assertIn("live_final_runner_detail_count", failed)

    def test_live_final_runner_duplicate_mirror_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_final_runner(root, duplicate_count=1)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_final_runner_mirror_counts", failed)

    def test_stale_live_req6_parameterization_count_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_req6_parameterization(root, check_count=70)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_req6_parameterization_check_count", failed)

    def test_live_req6_parameterization_core_failure_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_req6_parameterization(root, fail_name="parallelism_extension_par32_requires_fresh_reports")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_req6_parameterization_pass", failed)
        self.assertIn("live_req6_parameterization_fail_zero", failed)
        self.assertIn("live_req6_parameterization_core_checks_pass", failed)

    def test_stale_live_current_audit_summary_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        stale_summary = dict(CURRENT_AUDIT_SUMMARY)
        stale_summary["reflected"] = 9
        write_live_current_audit(root, summary=stale_summary)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_current_audit_summary_counts", failed)

    def test_live_current_audit_policy_drift_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_current_audit(root, xr_policy_consistent=False)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_current_audit_xr_policy_integrity", failed)

    def test_live_current_audit_qkv_required_drift_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_current_audit(root, qkv_required=True)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_current_audit_qkv_optional_state", failed)

    def test_stale_live_closeout_validation_count_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_closeout_validation(root, check_count=48)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_closeout_validation_check_count", failed)

    def test_live_closeout_validation_core_failure_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_closeout_validation(root, fail_name="board_package_ready")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_closeout_validation_pass", failed)
        self.assertIn("live_closeout_validation_fail_zero", failed)
        self.assertIn("live_closeout_validation_core_checks_pass", failed)

    def test_stale_live_closeout_packet_status_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_closeout_packet(root, status="stale")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_closeout_packet_ready", failed)

    def test_live_closeout_packet_qkv_command_drift_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_closeout_packet(root, qkv_commands=False)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_closeout_packet_qkv_optional_commands", failed)

    def test_live_c3b_physical_gate_drift_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_c3b_physical_gate(root, ready_for_board=False)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertNotIn("evidence_manifest_contract_matches_live", failed)
        self.assertIn("live_c3b_physical_gate_current_blocker_contract", failed)

    def test_live_xr_vits_gate_candidate_drift_fails_even_when_manifest_checks_pass(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        write_live_xr_vits_gate(root, candidate_exists=False)

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("live_xr_vits_gate_candidate_contract", failed)

    def test_stale_xr_resolution_fails(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        path = root / "docs" / "resources" / "xr_vits_reference_resolution_2026_06_10.json"
        live = json.loads(path.read_text())
        live["candidate"]["score"] = 1
        path.write_text(json.dumps(live) + "\n")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("xr_resolution_embedded_matches_live", failed)

    def test_stale_unblock_intake_fails(self) -> None:
        handoff = sample_handoff()
        root = Path(str(handoff["root"]))
        path = root / "docs" / "resources" / "final_unblock_intake_2026_06_10.json"
        live = json.loads(path.read_text())
        live["blocker_status"]["requested XR-VITs sibling"]["ready_for_active_unblock"] = True
        live["blocker_status"]["requested XR-VITs sibling"]["candidate_status"] = "pass"
        path.write_text(json.dumps(live) + "\n")

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("handoff_remaining_blockers_match_intake", failed)
        self.assertIn("handoff_candidate_states_match_intake", failed)

    def test_missing_board_import_command_fails(self) -> None:
        handoff = sample_handoff()
        handoff["board_smoke"]["commands"] = ["python3 tools/run_zcu104_c3b_smoke_remote.py --execute"]

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("board_import_command", failed)
        self.assertIn("board_validate_command", failed)

    def test_missing_policy_integrity_fails(self) -> None:
        handoff = sample_handoff()
        handoff["xr_vits"]["reference_resolution"].pop("policy_integrity")  # type: ignore[index, union-attr]

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("xr_policy_integrity_required", failed)
        self.assertIn("xr_policy_validation_embedded_matches_live", failed)

    def test_legacy_policy_integrity_fails(self) -> None:
        handoff = sample_handoff()
        integrity = handoff["xr_vits"]["reference_resolution"]["policy_integrity"]  # type: ignore[index]
        integrity["legacy_policy_clears_final_signoff"] = True  # type: ignore[index]
        integrity["validation"]["status"] = "legacy-warning"  # type: ignore[index]

        result = validator.validate_handoff(handoff)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("xr_legacy_policy_not_accepted", failed)
        self.assertIn("xr_policy_validation_embedded_matches_live", failed)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            resources = root / "docs" / "resources"
            resources.mkdir(parents=True)
            handoff_path = resources / "final_operator_handoff_2026_06_10.json"
            handoff_path.write_text(json.dumps(sample_handoff(root)) + "\n")
            json_out = root / "generated" / "handoff_validation.json"
            markdown_out = root / "generated" / "handoff_validation.md"
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
            self.assertIn("HGTXR Final Operator Handoff Validation", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
