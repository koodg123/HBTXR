#!/usr/bin/env python3
"""Regenerate third-goal final signoff evidence in one host-side pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import audit_final_unblock_candidates
import check_c3b_board_smoke_readiness
import check_final_blocker_closure_readiness
import check_third_goal_preflight
import create_xr_vits_replacement_policy
import discover_c3b_smoke_candidates
import discover_pynq_smoke_candidates
import check_xr_vits_reference_resolution
import import_pynq_smoke_result
import run_zcu104_c3b_smoke_remote
import validate_final_operator_handoff
import validate_final_signoff_bundle
import validate_final_unblock_closeout_packet
import write_e2e_resource_matrix
import write_e2e_resource_policy_audit
import write_final_evidence_manifest
import write_final_operator_handoff
import write_final_signoff_audit
import write_final_unblock_commands
import write_final_unblock_closeout_packet
import write_final_unblock_intake
import write_c3b_smoke_result_contract
import write_c3b_smoke_transfer_manifest
import write_c3b_physical_smoke_gate_audit
import write_hgpipe_operator_audit
import write_p2_vit_scale_calibration_report
import write_req5_q4q8_swhw_match_audit
import write_req1_environment_audit
import write_req6_parameterization_audit
import write_req9_deit_image_reference_audit
import write_selected_path_execution_audit
import write_spec_plan_conformance_audit
import write_third_goal_completion_audit
import write_third_goal_current_audit
import write_third_goal_requirements_trace
import write_third_goal_source_audit
import write_third_goal_unblock_checklist
import write_vref_p0_buffer_lifetime_audit
import write_vref_p0_pot_scale_audit
import write_vref_p0_pot_scale_sweep
import write_vref_p0_qkv_uram_cache_successor
import write_xr_vits_gate_audit
import write_xr_vits_unblock_packet


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
DATE_TAG = "2026_06_10"
CURRENT_AUDIT_DATE_TAG = "2026_06_16"

ToolMain = Callable[[list[str]], int]


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def copy_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def unique_ordered(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mirror_integrity(root: Path, mirrored_artifacts: list[str]) -> dict[str, Any]:
    resources = (root / "docs" / "resources").resolve()
    generated = (root / "hardware" / "generated" / "signoff").resolve()
    self_referential = {
        f"third_goal_final_signoff_run_{DATE_TAG}.json",
        f"third_goal_final_signoff_run_{DATE_TAG}.md",
    }
    contracts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    checked = 0
    excluded = 0

    for artifact in unique_ordered(mirrored_artifacts):
        mirror = Path(artifact)
        mirror_resolved = mirror.resolve()
        try:
            relative = mirror_resolved.relative_to(resources)
        except ValueError:
            contract = {
                "status": "fail",
                "reason": "outside-canonical-docs-resources",
                "source": "",
                "mirror": str(mirror),
                "mirror_exists": mirror.exists(),
            }
            contracts.append(contract)
            failures.append(contract)
            continue

        source = generated / relative
        if relative.name in self_referential:
            excluded += 1
            contracts.append(
                {
                    "status": "excluded",
                    "reason": "self-referential-final-runner-summary",
                    "source": str(source),
                    "mirror": str(mirror),
                    "source_exists": source.exists(),
                    "mirror_exists": mirror.exists(),
                }
            )
            continue

        source_exists = source.exists()
        mirror_exists = mirror.exists()
        source_sha256 = sha256_file(source) if source_exists and source.is_file() else ""
        mirror_sha256 = sha256_file(mirror) if mirror_exists and mirror.is_file() else ""
        source_size = source.stat().st_size if source_exists and source.is_file() else None
        mirror_size = mirror.stat().st_size if mirror_exists and mirror.is_file() else None
        ok = bool(
            source_exists
            and mirror_exists
            and source_size == mirror_size
            and source_sha256
            and source_sha256 == mirror_sha256
        )
        checked += 1
        contract = {
            "status": "pass" if ok else "fail",
            "source": str(source),
            "mirror": str(mirror),
            "source_exists": source_exists,
            "mirror_exists": mirror_exists,
            "source_size": source_size,
            "mirror_size": mirror_size,
            "source_sha256": source_sha256,
            "mirror_sha256": mirror_sha256,
            "sha256_match": bool(source_sha256 and source_sha256 == mirror_sha256),
        }
        contracts.append(contract)
        if not ok:
            failures.append(contract)

    return {
        "canonical_evidence_root": str(resources),
        "generated_signoff_root": str(generated),
        "status": "pass" if not failures else "fail",
        "artifact_count": len(unique_ordered(mirrored_artifacts)),
        "checked_count": checked,
        "excluded_count": excluded,
        "fail_count": len(failures),
        "failures": failures,
        "contracts": contracts,
    }


def default_paths(root: Path) -> dict[str, Path]:
    hardware = root / "hardware"
    generated = hardware / "generated" / "signoff"
    resources = root / "docs" / "resources"
    return {
        "readiness_json": generated / f"c3b_board_smoke_readiness_{DATE_TAG}.json",
        "readiness_md": generated / f"c3b_board_smoke_readiness_{DATE_TAG}.md",
        "preflight_json": generated / f"third_goal_final_signoff_{DATE_TAG}.json",
        "final_audit_json": generated / f"final_signoff_audit_{DATE_TAG}.json",
        "final_audit_md": generated / f"final_signoff_audit_{DATE_TAG}.md",
        "completion_json": generated / f"third_goal_completion_audit_{DATE_TAG}.json",
        "completion_md": generated / f"third_goal_completion_audit_{DATE_TAG}.md",
        "unblock_json": generated / f"third_goal_unblock_checklist_{DATE_TAG}.json",
        "unblock_md": generated / f"third_goal_unblock_checklist_{DATE_TAG}.md",
        "zcu104_remote_json": generated / f"zcu104_c3b_smoke_remote_run_{DATE_TAG}.json",
        "zcu104_remote_md": generated / f"zcu104_c3b_smoke_remote_run_{DATE_TAG}.md",
        "vref_successor_remote_json": generated / f"zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_{DATE_TAG}.json",
        "vref_successor_remote_md": generated / f"zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_{DATE_TAG}.md",
        "qkv_uram_remote_json": generated / f"zcu104_vref_p0_softmax_input_x2_qkv_uram_smoke_remote_run_{DATE_TAG}.json",
        "qkv_uram_remote_md": generated / f"zcu104_vref_p0_softmax_input_x2_qkv_uram_smoke_remote_run_{DATE_TAG}.md",
        "c3b_transfer_manifest_json": generated / f"c3b_smoke_transfer_manifest_{DATE_TAG}.json",
        "c3b_transfer_manifest_md": generated / f"c3b_smoke_transfer_manifest_{DATE_TAG}.md",
        "c3b_bundle_sha256": generated / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
        "c3b_smoke_contract_json": generated / f"c3b_smoke_result_contract_{DATE_TAG}.json",
        "c3b_smoke_contract_md": generated / f"c3b_smoke_result_contract_{DATE_TAG}.md",
        "xr_vits_packet_json": generated / f"xr_vits_unblock_packet_{DATE_TAG}.json",
        "xr_vits_packet_md": generated / f"xr_vits_unblock_packet_{DATE_TAG}.md",
        "xr_vits_policy_preview_json": generated / f"xr_vits_replacement_policy_preview_{DATE_TAG}.json",
        "xr_vits_policy_preview_md": generated / f"xr_vits_replacement_policy_preview_{DATE_TAG}.md",
        "xr_vits_reference_resolution_json": generated / f"xr_vits_reference_resolution_{DATE_TAG}.json",
        "xr_vits_reference_resolution_md": generated / f"xr_vits_reference_resolution_{DATE_TAG}.md",
        "command_card_json": generated / f"final_unblock_commands_{DATE_TAG}.json",
        "command_card_md": generated / f"final_unblock_commands_{DATE_TAG}.md",
        "unblock_intake_json": generated / f"final_unblock_intake_{DATE_TAG}.json",
        "unblock_intake_md": generated / f"final_unblock_intake_{DATE_TAG}.md",
        "closeout_packet_json": generated / f"final_unblock_closeout_packet_{DATE_TAG}.json",
        "closeout_packet_md": generated / f"final_unblock_closeout_packet_{DATE_TAG}.md",
        "closeout_validation_json": generated / f"final_unblock_closeout_packet_validation_{DATE_TAG}.json",
        "closeout_validation_md": generated / f"final_unblock_closeout_packet_validation_{DATE_TAG}.md",
        "resource_matrix_json": generated / f"e2e_resource_matrix_{DATE_TAG}.json",
        "resource_matrix_md": generated / f"e2e_resource_matrix_{DATE_TAG}.md",
        "selected_path_json": generated / f"selected_path_execution_audit_{DATE_TAG}.json",
        "selected_path_md": generated / f"selected_path_execution_audit_{DATE_TAG}.md",
        "resource_policy_json": generated / f"e2e_resource_policy_audit_{DATE_TAG}.json",
        "resource_policy_md": generated / f"e2e_resource_policy_audit_{DATE_TAG}.md",
        "requirements_trace_json": generated / f"third_goal_requirements_trace_{DATE_TAG}.json",
        "requirements_trace_md": generated / f"third_goal_requirements_trace_{DATE_TAG}.md",
        "spec_plan_json": generated / f"spec_plan_conformance_audit_{DATE_TAG}.json",
        "spec_plan_md": generated / f"spec_plan_conformance_audit_{DATE_TAG}.md",
        "candidate_audit_json": generated / f"final_unblock_candidate_audit_{DATE_TAG}.json",
        "candidate_audit_md": generated / f"final_unblock_candidate_audit_{DATE_TAG}.md",
        "operator_handoff_json": generated / f"final_operator_handoff_{DATE_TAG}.json",
        "operator_handoff_md": generated / f"final_operator_handoff_{DATE_TAG}.md",
        "operator_handoff_validation_json": generated / f"final_operator_handoff_validation_{DATE_TAG}.json",
        "operator_handoff_validation_md": generated / f"final_operator_handoff_validation_{DATE_TAG}.md",
        "bundle_validation_json": generated / f"final_signoff_bundle_validation_{DATE_TAG}.json",
        "bundle_validation_md": generated / f"final_signoff_bundle_validation_{DATE_TAG}.md",
        "c3b_smoke_discovery_json": generated / f"c3b_smoke_candidate_discovery_{DATE_TAG}.json",
        "c3b_smoke_discovery_md": generated / f"c3b_smoke_candidate_discovery_{DATE_TAG}.md",
        "pynq_c3b_smoke_discovery_json": generated / f"pynq_smoke_candidate_discovery_c3b_{CURRENT_AUDIT_DATE_TAG}.json",
        "pynq_c3b_smoke_discovery_md": generated / f"pynq_smoke_candidate_discovery_c3b_{CURRENT_AUDIT_DATE_TAG}.md",
        "vref_smoke_discovery_json": generated / f"vref_successor_smoke_candidate_discovery_{DATE_TAG}.json",
        "vref_smoke_discovery_md": generated / f"vref_successor_smoke_candidate_discovery_{DATE_TAG}.md",
        "pynq_vref_smoke_discovery_json": generated / f"pynq_smoke_candidate_discovery_vref_p0_{CURRENT_AUDIT_DATE_TAG}.json",
        "pynq_vref_smoke_discovery_md": generated / f"pynq_smoke_candidate_discovery_vref_p0_{CURRENT_AUDIT_DATE_TAG}.md",
        "qkv_uram_smoke_discovery_json": generated / f"qkv_uram_smoke_candidate_discovery_{CURRENT_AUDIT_DATE_TAG}.json",
        "qkv_uram_smoke_discovery_md": generated / f"qkv_uram_smoke_candidate_discovery_{CURRENT_AUDIT_DATE_TAG}.md",
        "qkv_uram_successor_json": generated / f"vref_p0_qkv_uram_cache_successor_{CURRENT_AUDIT_DATE_TAG}.json",
        "qkv_uram_successor_md": generated / f"vref_p0_qkv_uram_cache_successor_{CURRENT_AUDIT_DATE_TAG}.md",
        "vref_p0_pot_scale_audit_json": generated / f"vref_p0_pot_scale_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "vref_p0_pot_scale_audit_md": generated / f"vref_p0_pot_scale_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "vref_p0_pot_scale_sweep_json": generated / f"vref_p0_pot_scale_sweep_{CURRENT_AUDIT_DATE_TAG}.json",
        "vref_p0_pot_scale_sweep_md": generated / f"vref_p0_pot_scale_sweep_{CURRENT_AUDIT_DATE_TAG}.md",
        "req5_q4q8_swhw_json": generated / f"req5_q4q8_swhw_match_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "req5_q4q8_swhw_md": generated / f"req5_q4q8_swhw_match_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "p2_vit_scale_calibration_json": generated / f"p2_vit_scale_calibration_report_{CURRENT_AUDIT_DATE_TAG}.json",
        "p2_vit_scale_calibration_md": generated / f"p2_vit_scale_calibration_report_{CURRENT_AUDIT_DATE_TAG}.md",
        "req1_environment_json": generated / f"req1_environment_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "req1_environment_md": generated / f"req1_environment_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "req6_parameterization_json": generated / f"req6_parameterization_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "req6_parameterization_md": generated / f"req6_parameterization_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "req9_deit_image_json": generated / f"req9_deit_image_reference_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "req9_deit_image_md": generated / f"req9_deit_image_reference_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "xr_vits_gate_json": generated / f"xr_vits_gate_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "xr_vits_gate_md": generated / f"xr_vits_gate_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "c3b_physical_smoke_gate_json": generated / f"c3b_physical_smoke_gate_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "c3b_physical_smoke_gate_md": generated / f"c3b_physical_smoke_gate_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "vref_p0_buffer_lifetime_json": generated / f"vref_p0_buffer_lifetime_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "vref_p0_buffer_lifetime_md": generated / f"vref_p0_buffer_lifetime_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "hgpipe_operator_audit_json": generated / f"hgpipe_operator_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "hgpipe_operator_audit_md": generated / f"hgpipe_operator_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "blocker_closure_json": generated / f"final_blocker_closure_readiness_{DATE_TAG}.json",
        "blocker_closure_md": generated / f"final_blocker_closure_readiness_{DATE_TAG}.md",
        "evidence_manifest_json": generated / f"final_evidence_manifest_{DATE_TAG}.json",
        "evidence_manifest_md": generated / f"final_evidence_manifest_{DATE_TAG}.md",
        "source_audit_json": generated / f"third_goal_source_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "source_audit_md": generated / f"third_goal_source_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "current_audit_json": generated / f"third_goal_current_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "current_audit_md": generated / f"third_goal_current_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "summary_json": generated / f"third_goal_final_signoff_run_{DATE_TAG}.json",
        "summary_md": generated / f"third_goal_final_signoff_run_{DATE_TAG}.md",
        "c3b_import_json": generated / f"c3b_smoke_import_{DATE_TAG}.json",
        "c3b_import_validation_json": generated / f"c3b_smoke_import_validation_{DATE_TAG}.json",
        "vref_successor_import_json": generated / f"vref_successor_smoke_import_{DATE_TAG}.json",
        "vref_successor_import_validation_json": generated / f"vref_successor_smoke_import_validation_{DATE_TAG}.json",
        "qkv_uram_import_json": generated / f"qkv_uram_smoke_import_{CURRENT_AUDIT_DATE_TAG}.json",
        "qkv_uram_import_validation_json": generated / f"qkv_uram_smoke_import_validation_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_readiness_json": resources / f"c3b_board_smoke_readiness_{DATE_TAG}.json",
        "doc_readiness_md": resources / f"c3b_board_smoke_readiness_{DATE_TAG}.md",
        "doc_preflight_json": resources / f"third_goal_final_signoff_{DATE_TAG}.json",
        "doc_final_audit_json": resources / f"final_signoff_audit_{DATE_TAG}.json",
        "doc_final_audit_md": resources / f"final_signoff_audit_{DATE_TAG}.md",
        "doc_completion_json": resources / f"third_goal_completion_audit_{DATE_TAG}.json",
        "doc_completion_md": resources / f"third_goal_completion_audit_{DATE_TAG}.md",
        "doc_unblock_json": resources / f"third_goal_unblock_checklist_{DATE_TAG}.json",
        "doc_unblock_md": resources / f"third_goal_unblock_checklist_{DATE_TAG}.md",
        "doc_zcu104_remote_json": resources / f"zcu104_c3b_smoke_remote_run_{DATE_TAG}.json",
        "doc_zcu104_remote_md": resources / f"zcu104_c3b_smoke_remote_run_{DATE_TAG}.md",
        "doc_vref_successor_remote_json": resources / f"zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_{DATE_TAG}.json",
        "doc_vref_successor_remote_md": resources / f"zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_{DATE_TAG}.md",
        "doc_qkv_uram_remote_json": resources / f"zcu104_vref_p0_softmax_input_x2_qkv_uram_smoke_remote_run_{DATE_TAG}.json",
        "doc_qkv_uram_remote_md": resources / f"zcu104_vref_p0_softmax_input_x2_qkv_uram_smoke_remote_run_{DATE_TAG}.md",
        "doc_c3b_transfer_manifest_json": resources / f"c3b_smoke_transfer_manifest_{DATE_TAG}.json",
        "doc_c3b_transfer_manifest_md": resources / f"c3b_smoke_transfer_manifest_{DATE_TAG}.md",
        "doc_c3b_bundle_sha256": resources / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
        "doc_c3b_smoke_contract_json": resources / f"c3b_smoke_result_contract_{DATE_TAG}.json",
        "doc_c3b_smoke_contract_md": resources / f"c3b_smoke_result_contract_{DATE_TAG}.md",
        "doc_xr_vits_packet_json": resources / f"xr_vits_unblock_packet_{DATE_TAG}.json",
        "doc_xr_vits_packet_md": resources / f"xr_vits_unblock_packet_{DATE_TAG}.md",
        "doc_xr_vits_policy_preview_json": resources / f"xr_vits_replacement_policy_preview_{DATE_TAG}.json",
        "doc_xr_vits_policy_preview_md": resources / f"xr_vits_replacement_policy_preview_{DATE_TAG}.md",
        "doc_xr_vits_reference_resolution_json": resources / f"xr_vits_reference_resolution_{DATE_TAG}.json",
        "doc_xr_vits_reference_resolution_md": resources / f"xr_vits_reference_resolution_{DATE_TAG}.md",
        "doc_command_card_json": resources / f"final_unblock_commands_{DATE_TAG}.json",
        "doc_command_card_md": resources / f"final_unblock_commands_{DATE_TAG}.md",
        "doc_unblock_intake_json": resources / f"final_unblock_intake_{DATE_TAG}.json",
        "doc_unblock_intake_md": resources / f"final_unblock_intake_{DATE_TAG}.md",
        "doc_closeout_packet_json": resources / f"final_unblock_closeout_packet_{DATE_TAG}.json",
        "doc_closeout_packet_md": resources / f"final_unblock_closeout_packet_{DATE_TAG}.md",
        "doc_closeout_validation_json": resources / f"final_unblock_closeout_packet_validation_{DATE_TAG}.json",
        "doc_closeout_validation_md": resources / f"final_unblock_closeout_packet_validation_{DATE_TAG}.md",
        "doc_resource_matrix_json": resources / f"e2e_resource_matrix_{DATE_TAG}.json",
        "doc_resource_matrix_md": resources / f"e2e_resource_matrix_{DATE_TAG}.md",
        "doc_selected_path_json": resources / f"selected_path_execution_audit_{DATE_TAG}.json",
        "doc_selected_path_md": resources / f"selected_path_execution_audit_{DATE_TAG}.md",
        "doc_resource_policy_json": resources / f"e2e_resource_policy_audit_{DATE_TAG}.json",
        "doc_resource_policy_md": resources / f"e2e_resource_policy_audit_{DATE_TAG}.md",
        "doc_requirements_trace_json": resources / f"third_goal_requirements_trace_{DATE_TAG}.json",
        "doc_requirements_trace_md": resources / f"third_goal_requirements_trace_{DATE_TAG}.md",
        "doc_spec_plan_json": resources / f"spec_plan_conformance_audit_{DATE_TAG}.json",
        "doc_spec_plan_md": resources / f"spec_plan_conformance_audit_{DATE_TAG}.md",
        "doc_candidate_audit_json": resources / f"final_unblock_candidate_audit_{DATE_TAG}.json",
        "doc_candidate_audit_md": resources / f"final_unblock_candidate_audit_{DATE_TAG}.md",
        "doc_operator_handoff_json": resources / f"final_operator_handoff_{DATE_TAG}.json",
        "doc_operator_handoff_md": resources / f"final_operator_handoff_{DATE_TAG}.md",
        "doc_operator_handoff_validation_json": resources / f"final_operator_handoff_validation_{DATE_TAG}.json",
        "doc_operator_handoff_validation_md": resources / f"final_operator_handoff_validation_{DATE_TAG}.md",
        "doc_bundle_validation_json": resources / f"final_signoff_bundle_validation_{DATE_TAG}.json",
        "doc_bundle_validation_md": resources / f"final_signoff_bundle_validation_{DATE_TAG}.md",
        "doc_c3b_smoke_discovery_json": resources / f"c3b_smoke_candidate_discovery_{DATE_TAG}.json",
        "doc_c3b_smoke_discovery_md": resources / f"c3b_smoke_candidate_discovery_{DATE_TAG}.md",
        "doc_pynq_c3b_smoke_discovery_json": resources / f"pynq_smoke_candidate_discovery_c3b_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_pynq_c3b_smoke_discovery_md": resources / f"pynq_smoke_candidate_discovery_c3b_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_vref_smoke_discovery_json": resources / f"vref_successor_smoke_candidate_discovery_{DATE_TAG}.json",
        "doc_vref_smoke_discovery_md": resources / f"vref_successor_smoke_candidate_discovery_{DATE_TAG}.md",
        "doc_pynq_vref_smoke_discovery_json": resources / f"pynq_smoke_candidate_discovery_vref_p0_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_pynq_vref_smoke_discovery_md": resources / f"pynq_smoke_candidate_discovery_vref_p0_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_qkv_uram_smoke_discovery_json": resources / f"qkv_uram_smoke_candidate_discovery_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_qkv_uram_smoke_discovery_md": resources / f"qkv_uram_smoke_candidate_discovery_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_qkv_uram_successor_json": resources / f"vref_p0_qkv_uram_cache_successor_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_qkv_uram_successor_md": resources / f"vref_p0_qkv_uram_cache_successor_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_vref_p0_pot_scale_audit_json": resources / f"vref_p0_pot_scale_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_vref_p0_pot_scale_audit_md": resources / f"vref_p0_pot_scale_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_vref_p0_pot_scale_sweep_json": resources / f"vref_p0_pot_scale_sweep_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_vref_p0_pot_scale_sweep_md": resources / f"vref_p0_pot_scale_sweep_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_req5_q4q8_swhw_json": resources / f"req5_q4q8_swhw_match_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_req5_q4q8_swhw_md": resources / f"req5_q4q8_swhw_match_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_p2_vit_scale_calibration_json": resources / f"p2_vit_scale_calibration_report_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_p2_vit_scale_calibration_md": resources / f"p2_vit_scale_calibration_report_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_req1_environment_json": resources / f"req1_environment_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_req1_environment_md": resources / f"req1_environment_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_req6_parameterization_json": resources / f"req6_parameterization_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_req6_parameterization_md": resources / f"req6_parameterization_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_req9_deit_image_json": resources / f"req9_deit_image_reference_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_req9_deit_image_md": resources / f"req9_deit_image_reference_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_xr_vits_gate_json": resources / f"xr_vits_gate_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_xr_vits_gate_md": resources / f"xr_vits_gate_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_c3b_physical_smoke_gate_json": resources / f"c3b_physical_smoke_gate_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_c3b_physical_smoke_gate_md": resources / f"c3b_physical_smoke_gate_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_vref_p0_buffer_lifetime_json": resources / f"vref_p0_buffer_lifetime_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_vref_p0_buffer_lifetime_md": resources / f"vref_p0_buffer_lifetime_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_hgpipe_operator_audit_json": resources / f"hgpipe_operator_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_hgpipe_operator_audit_md": resources / f"hgpipe_operator_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_blocker_closure_json": resources / f"final_blocker_closure_readiness_{DATE_TAG}.json",
        "doc_blocker_closure_md": resources / f"final_blocker_closure_readiness_{DATE_TAG}.md",
        "doc_evidence_manifest_json": resources / f"final_evidence_manifest_{DATE_TAG}.json",
        "doc_evidence_manifest_md": resources / f"final_evidence_manifest_{DATE_TAG}.md",
        "doc_source_audit_json": resources / f"third_goal_source_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_source_audit_md": resources / f"third_goal_source_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_current_audit_json": resources / f"third_goal_current_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_current_audit_md": resources / f"third_goal_current_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "doc_summary_json": resources / f"third_goal_final_signoff_run_{DATE_TAG}.json",
        "doc_summary_md": resources / f"third_goal_final_signoff_run_{DATE_TAG}.md",
        "doc_c3b_import_json": resources / f"c3b_smoke_import_{DATE_TAG}.json",
        "doc_c3b_import_validation_json": resources / f"c3b_smoke_import_validation_{DATE_TAG}.json",
        "doc_vref_successor_import_json": resources / f"vref_successor_smoke_import_{DATE_TAG}.json",
        "doc_vref_successor_import_validation_json": resources / f"vref_successor_smoke_import_validation_{DATE_TAG}.json",
        "doc_qkv_uram_import_json": resources / f"qkv_uram_smoke_import_{CURRENT_AUDIT_DATE_TAG}.json",
        "doc_qkv_uram_import_validation_json": resources / f"qkv_uram_smoke_import_validation_{CURRENT_AUDIT_DATE_TAG}.json",
    }


def run_step(name: str, tool: ToolMain, argv: list[str]) -> dict:
    code = tool(argv)
    return {
        "name": name,
        "exit_code": code,
        "argv": argv,
    }


def run_preflight(argv: list[str]) -> int:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["check_third_goal_preflight.py"] + argv
        return check_third_goal_preflight.main()
    finally:
        sys.argv = old_argv


def render_markdown(summary: dict) -> str:
    lines = [
        "# HGTXR Third Goal Final Signoff Run",
        "",
        f"- status: `{summary['status']}`",
        f"- root: `{summary['root']}`",
        f"- allow_blocked: `{summary['allow_blocked']}`",
        f"- final_preflight_failures: `{summary['final_preflight_summary'].get('fail', 'n/a')}`",
        f"- final_audit_status: `{summary['final_audit_status']}`",
        f"- completion_status: `{summary['completion_status']}`",
        f"- unblock_status: `{summary['unblock_status']}`",
        f"- readiness_status: `{summary['readiness_status']}`",
        f"- c3b_import_status: `{summary['c3b_import_status']}`",
        f"- zcu104_remote_status: `{summary['zcu104_remote_status']}`",
        f"- zcu104_remote_execute: `{summary['zcu104_remote_execute']}`",
        f"- vref_successor_required_for_final_signoff: `{summary['vref_successor_required_for_final_signoff']}`",
        f"- vref_successor_import_status: `{summary['vref_successor_import_status']}`",
        f"- vref_successor_remote_status: `{summary['vref_successor_remote_status']}`",
        f"- vref_successor_remote_execute: `{summary['vref_successor_remote_execute']}`",
        f"- qkv_uram_required_for_final_signoff: `{summary['qkv_uram_required_for_final_signoff']}`",
        f"- qkv_uram_import_status: `{summary['qkv_uram_import_status']}`",
        f"- qkv_uram_remote_status: `{summary['qkv_uram_remote_status']}`",
        f"- qkv_uram_remote_execute: `{summary['qkv_uram_remote_execute']}`",
        f"- c3b_transfer_manifest_status: `{summary['c3b_transfer_manifest_status']}`",
        f"- c3b_smoke_contract_status: `{summary['c3b_smoke_contract_status']}`",
        f"- xr_vits_policy_status: `{summary['xr_vits_policy_status']}`",
        f"- xr_vits_policy_preview_status: `{summary['xr_vits_policy_preview_status']}`",
        f"- xr_vits_packet_status: `{summary['xr_vits_packet_status']}`",
        f"- xr_vits_reference_resolution_status: `{summary['xr_vits_reference_resolution_status']}`",
        f"- resource_matrix_status: `{summary['resource_matrix_status']}`",
        f"- selected_path_status: `{summary['selected_path_status']}`",
        f"- resource_policy_status: `{summary['resource_policy_status']}`",
        f"- requirements_trace_status: `{summary['requirements_trace_status']}`",
        f"- spec_plan_status: `{summary['spec_plan_status']}`",
        f"- candidate_audit_status: `{summary['candidate_audit_status']}`",
        f"- operator_handoff_status: `{summary['operator_handoff_status']}`",
        f"- operator_handoff_validation_status: `{summary['operator_handoff_validation_status']}`",
        f"- final_bundle_validation_status: `{summary['final_bundle_validation_status']}`",
        f"- c3b_smoke_discovery_status: `{summary['c3b_smoke_discovery_status']}`",
        f"- pynq_c3b_smoke_discovery_status: `{summary['pynq_c3b_smoke_discovery_status']}`",
        f"- vref_smoke_discovery_status: `{summary['vref_smoke_discovery_status']}`",
        f"- pynq_vref_smoke_discovery_status: `{summary['pynq_vref_smoke_discovery_status']}`",
        f"- qkv_uram_smoke_discovery_status: `{summary['qkv_uram_smoke_discovery_status']}`",
        f"- p2_vit_scale_calibration_status: `{summary['p2_vit_scale_calibration_status']}`",
        f"- final_blocker_closure_status: `{summary['final_blocker_closure_status']}`",
        f"- final_unblock_intake_status: `{summary['final_unblock_intake_status']}`",
        f"- final_unblock_closeout_packet_status: `{summary['final_unblock_closeout_packet_status']}`",
        f"- final_unblock_closeout_validation_status: `{summary['final_unblock_closeout_validation_status']}`",
        f"- third_goal_source_audit_status: `{summary['third_goal_source_audit_status']}`",
        f"- third_goal_current_audit_status: `{summary['third_goal_current_audit_status']}`",
        f"- final_evidence_manifest_status: `{summary['final_evidence_manifest_status']}`",
        f"- blocker_count: `{summary['blocker_count']}`",
        f"- remaining_blocker_detail_count: `{summary['remaining_blocker_detail_count']}`",
        f"- mirrored_artifact_count: `{summary['mirrored_artifact_count']}`",
        f"- mirrored_artifact_unique_count: `{summary['mirrored_artifact_unique_count']}`",
        f"- mirrored_artifact_duplicate_count: `{summary['mirrored_artifact_duplicate_count']}`",
        f"- mirrored_artifact_integrity_status: `{summary.get('mirrored_artifact_integrity_status', 'n/a')}`",
        f"- mirrored_artifact_integrity_checked_count: `{summary.get('mirrored_artifact_integrity_checked_count', 'n/a')}`",
        f"- mirrored_artifact_integrity_fail_count: `{summary.get('mirrored_artifact_integrity_fail_count', 'n/a')}`",
        "",
        "## Steps",
        "",
    ]
    for step in summary["steps"]:
        lines.append(f"- `{step['name']}` exit `{step['exit_code']}`")
    lines.extend(["", "## Mirrored Artifacts", ""])
    for artifact in summary["mirrored_artifacts"]:
        lines.append(f"- `{artifact}`")
    integrity_failures = summary.get("mirrored_artifact_integrity_failures", [])
    lines.extend(["", "## Mirrored Artifact Integrity Failures", ""])
    if integrity_failures:
        for failure in integrity_failures:
            lines.append(f"- `{failure.get('mirror', 'unknown')}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = summary["remaining_blockers"]
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Remaining Blocker Input Paths", ""])
    blocker_paths = summary.get("remaining_blocker_input_paths", {})
    if blocker_paths:
        for blocker, paths in blocker_paths.items():
            if not paths:
                lines.append(f"- `{blocker}`: None.")
                continue
            rendered_paths = ", ".join(f"`{path}`" for path in paths)
            lines.append(f"- `{blocker}`: {rendered_paths}")
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def extract_path(payload: dict[str, Any], *keys: str) -> str:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return str(current) if current else ""


def build_remaining_blocker_details(
    *,
    root: Path,
    remaining_blockers: list[str],
    blocker_closure: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    c3b_path = (
        extract_path(blocker_closure, "current", "c3b_smoke", "path")
        or str(check_final_blocker_closure_readiness.canonical_c3b_path(root.resolve()))
    )
    xr_exact_path = (
        extract_path(blocker_closure, "current", "xr_vits", "exact", "path")
        or str(create_xr_vits_replacement_policy.default_requested_path(root.resolve()).resolve())
    )
    xr_policy_path = (
        extract_path(blocker_closure, "current", "xr_vits", "replacement_policy", "path")
        or str(check_final_blocker_closure_readiness.active_policy_path(root.resolve()))
    )

    path_map: dict[str, list[str]] = {}
    details: dict[str, Any] = {}
    for blocker in remaining_blockers:
        if blocker == "C3b AXIS/DMA physical smoke result":
            path_map[blocker] = [c3b_path]
            details[blocker] = {
                "kind": "canonical-pynq-smoke-json",
                "paths": [c3b_path],
                "status": extract_path(blocker_closure, "current", "c3b_smoke", "status") or "missing",
                "would_clear": bool(
                    blocker_closure.get("current", {})
                    .get("c3b_smoke", {})
                    .get("would_clear", False)
                ),
            }
        elif blocker == "requested XR-VITs sibling":
            path_map[blocker] = [xr_exact_path, xr_policy_path]
            current_xr = blocker_closure.get("current", {}).get("xr_vits", {})
            details[blocker] = {
                "kind": "exact-source-or-approved-replacement-policy",
                "paths": [xr_exact_path, xr_policy_path],
                "mode": current_xr.get("mode", "missing") if isinstance(current_xr, dict) else "missing",
                "would_clear": bool(current_xr.get("would_clear", False)) if isinstance(current_xr, dict) else False,
                "closure_options": [
                    {
                        "kind": "exact-source",
                        "path": xr_exact_path,
                        "status": extract_path(blocker_closure, "current", "xr_vits", "exact", "status")
                        or "missing",
                    },
                    {
                        "kind": "approved-replacement-policy",
                        "path": xr_policy_path,
                        "status": extract_path(
                            blocker_closure,
                            "current",
                            "xr_vits",
                            "replacement_policy",
                            "status",
                        )
                        or "missing",
                    },
                ],
            }
        elif blocker == "VREF-P0-02 QKV URAM physical smoke result":
            qkv_path = str(
                root.resolve()
                / "hardware"
                / "pynq"
                / "hgtxr"
                / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
            )
            path_map[blocker] = [qkv_path]
            details[blocker] = {
                "kind": "optional-qkv-uram-pynq-smoke-json",
                "paths": [qkv_path],
                "status": "missing",
                "would_clear": False,
            }
        else:
            path_map[blocker] = []
            details[blocker] = {"kind": "unknown", "paths": [], "status": "unknown", "would_clear": False}
    return path_map, details


def run_pipeline(
    *,
    root: Path,
    allow_blocked: bool,
    zcu104_host: str = "zcu104.local",
    zcu104_user: str | None = "xilinx",
    zcu104_port: int | None = None,
    zcu104_identity_file: Path | None = None,
    zcu104_remote_dir: str = run_zcu104_c3b_smoke_remote.DEFAULT_REMOTE_DIR,
    vref_successor_remote_dir: str | None = None,
    qkv_uram_remote_dir: str | None = None,
    execute_zcu104_smoke: bool = False,
    import_c3b_smoke_json: Path | None = None,
    import_c3b_no_require_paths: bool = False,
    dry_run_import_c3b_smoke: bool = False,
    require_vref_successor_physical_smoke: bool = False,
    execute_vref_successor_smoke: bool = False,
    import_vref_successor_smoke_json: Path | None = None,
    import_vref_successor_no_require_paths: bool = False,
    dry_run_import_vref_successor_smoke: bool = False,
    require_qkv_uram_physical_smoke: bool = False,
    execute_qkv_uram_smoke: bool = False,
    import_qkv_uram_smoke_json: Path | None = None,
    import_qkv_uram_no_require_paths: bool = False,
    dry_run_import_qkv_uram_smoke: bool = False,
    approve_xr_vits_replacement: bool = False,
    dry_run_xr_vits_replacement: bool = False,
    xr_vits_approved_by: str = "",
    xr_vits_replacement_reason: str = "",
    xr_vits_replacement_path: Path | None = None,
    tools: dict[str, ToolMain] | None = None,
    paths: dict[str, Path] | None = None,
) -> tuple[int, dict]:
    root = root.resolve()
    paths = paths or default_paths(root)
    tools = tools or {
        "readiness": check_c3b_board_smoke_readiness.main,
        "c3b_import": import_pynq_smoke_result.main,
        "preflight": run_preflight,
        "final_audit": write_final_signoff_audit.main,
        "completion": write_third_goal_completion_audit.main,
        "unblock": write_third_goal_unblock_checklist.main,
        "zcu104_remote": run_zcu104_c3b_smoke_remote.main,
        "c3b_transfer_manifest": write_c3b_smoke_transfer_manifest.main,
        "c3b_smoke_contract": write_c3b_smoke_result_contract.main,
        "xr_vits_policy": create_xr_vits_replacement_policy.main,
        "xr_vits_packet": write_xr_vits_unblock_packet.main,
        "xr_vits_reference_resolution": check_xr_vits_reference_resolution.main,
        "command_card": write_final_unblock_commands.main,
        "unblock_intake": write_final_unblock_intake.main,
        "closeout_packet": write_final_unblock_closeout_packet.main,
        "closeout_validation": validate_final_unblock_closeout_packet.main,
        "resource_matrix": write_e2e_resource_matrix.main,
        "selected_path": write_selected_path_execution_audit.main,
        "resource_policy": write_e2e_resource_policy_audit.main,
        "requirements_trace": write_third_goal_requirements_trace.main,
        "spec_plan": write_spec_plan_conformance_audit.main,
        "candidate_audit": audit_final_unblock_candidates.main,
        "operator_handoff": write_final_operator_handoff.main,
        "operator_handoff_validation": validate_final_operator_handoff.main,
        "bundle_validation": validate_final_signoff_bundle.main,
        "c3b_smoke_discovery": discover_c3b_smoke_candidates.main,
        "pynq_smoke_discovery": discover_pynq_smoke_candidates.main,
        "blocker_closure": check_final_blocker_closure_readiness.main,
        "evidence_manifest": write_final_evidence_manifest.main,
        "source_audit": write_third_goal_source_audit.main,
        "current_audit": write_third_goal_current_audit.main,
        "qkv_uram_successor": write_vref_p0_qkv_uram_cache_successor.main,
        "vref_p0_pot_scale_audit": write_vref_p0_pot_scale_audit.main,
        "vref_p0_pot_scale_sweep": write_vref_p0_pot_scale_sweep.main,
        "req5_q4q8_swhw": write_req5_q4q8_swhw_match_audit.main,
        "p2_vit_scale_calibration": write_p2_vit_scale_calibration_report.main,
        "req1_environment": write_req1_environment_audit.main,
        "req6_parameterization": write_req6_parameterization_audit.main,
        "req9_deit_image": write_req9_deit_image_reference_audit.main,
        "xr_vits_gate": write_xr_vits_gate_audit.main,
        "c3b_physical_smoke_gate": write_c3b_physical_smoke_gate_audit.main,
        "vref_p0_buffer_lifetime": write_vref_p0_buffer_lifetime_audit.main,
        "hgpipe_operator_audit": write_hgpipe_operator_audit.main,
    }

    steps: list[dict] = []
    c3b_import_status = "skipped"
    if import_c3b_smoke_json is not None:
        c3b_import_argv = [
            str(import_c3b_smoke_json),
            "--preset",
            "axis-c3b-mem16",
            "--root",
            str(root),
            "--json-out",
            str(paths["c3b_import_json"]),
            "--validation-out",
            str(paths["c3b_import_validation_json"]),
        ]
        if import_c3b_no_require_paths:
            c3b_import_argv.append("--no-require-paths")
        if dry_run_import_c3b_smoke:
            c3b_import_argv.append("--dry-run")
        import_step = run_step("c3b-smoke-import", tools["c3b_import"], c3b_import_argv)
        steps.append(import_step)
        if import_step["exit_code"] == 0:
            c3b_import_status = "dry-run-pass" if dry_run_import_c3b_smoke else "pass"
        else:
            c3b_import_status = "fail"

    vref_successor_import_status = "skipped"
    if import_vref_successor_smoke_json is not None:
        vref_import_argv = [
            str(import_vref_successor_smoke_json),
            "--preset",
            "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
            "--root",
            str(root),
            "--json-out",
            str(paths["vref_successor_import_json"]),
            "--validation-out",
            str(paths["vref_successor_import_validation_json"]),
        ]
        if import_vref_successor_no_require_paths:
            vref_import_argv.append("--no-require-paths")
        if dry_run_import_vref_successor_smoke:
            vref_import_argv.append("--dry-run")
        vref_import_step = run_step("vref-successor-smoke-import", tools["c3b_import"], vref_import_argv)
        steps.append(vref_import_step)
        if vref_import_step["exit_code"] == 0:
            vref_successor_import_status = "dry-run-pass" if dry_run_import_vref_successor_smoke else "pass"
        else:
            vref_successor_import_status = "fail"

    qkv_uram_import_status = "skipped"
    if import_qkv_uram_smoke_json is not None:
        qkv_import_argv = [
            str(import_qkv_uram_smoke_json),
            "--preset",
            "axis-vref-p0-softmax-input-x2-qkv-uram",
            "--root",
            str(root),
            "--json-out",
            str(paths["qkv_uram_import_json"]),
            "--validation-out",
            str(paths["qkv_uram_import_validation_json"]),
        ]
        if import_qkv_uram_no_require_paths:
            qkv_import_argv.append("--no-require-paths")
        if dry_run_import_qkv_uram_smoke:
            qkv_import_argv.append("--dry-run")
        qkv_import_step = run_step("qkv-uram-smoke-import", tools["c3b_import"], qkv_import_argv)
        steps.append(qkv_import_step)
        if qkv_import_step["exit_code"] == 0:
            qkv_uram_import_status = "dry-run-pass" if dry_run_import_qkv_uram_smoke else "pass"
        else:
            qkv_uram_import_status = "fail"

    zcu104_argv = [
        "--root",
        str(root),
        "--host",
        zcu104_host,
        "--remote-dir",
        zcu104_remote_dir,
        "--json-out",
        str(paths["zcu104_remote_json"]),
        "--markdown-out",
        str(paths["zcu104_remote_md"]),
    ]
    if zcu104_user:
        zcu104_argv.extend(["--user", zcu104_user])
    if zcu104_port is not None:
        zcu104_argv.extend(["--port", str(zcu104_port)])
    if zcu104_identity_file is not None:
        zcu104_argv.extend(["--identity-file", str(zcu104_identity_file)])
    if execute_zcu104_smoke:
        zcu104_argv.append("--execute")
    steps.append(
        run_step(
            "zcu104-remote-execute" if execute_zcu104_smoke else "zcu104-remote-dry-run",
            tools["zcu104_remote"],
            zcu104_argv,
        )
    )
    vref_successor_remote_status = "skipped"
    vref_successor_remote_execute = False
    if require_vref_successor_physical_smoke or execute_vref_successor_smoke:
        vref_remote_argv = [
            "--profile",
            "vref-p0-softmax-input-x2-dsp-mixed-stream",
            "--root",
            str(root),
            "--host",
            zcu104_host,
            "--json-out",
            str(paths["vref_successor_remote_json"]),
            "--markdown-out",
            str(paths["vref_successor_remote_md"]),
        ]
        if vref_successor_remote_dir is not None:
            vref_remote_argv.extend(["--remote-dir", vref_successor_remote_dir])
        if zcu104_user:
            vref_remote_argv.extend(["--user", zcu104_user])
        if zcu104_port is not None:
            vref_remote_argv.extend(["--port", str(zcu104_port)])
        if zcu104_identity_file is not None:
            vref_remote_argv.extend(["--identity-file", str(zcu104_identity_file)])
        if execute_vref_successor_smoke:
            vref_remote_argv.append("--execute")
        vref_step = run_step(
            "vref-successor-remote-execute" if execute_vref_successor_smoke else "vref-successor-remote-dry-run",
            tools["zcu104_remote"],
            vref_remote_argv,
        )
        steps.append(vref_step)
        if paths["vref_successor_remote_json"].exists():
            vref_remote = load_json(paths["vref_successor_remote_json"])
            vref_successor_remote_status = str(vref_remote.get("status", "unknown"))
            vref_successor_remote_execute = bool(vref_remote.get("execute", False))
        else:
            vref_successor_remote_status = "fail" if vref_step["exit_code"] else "missing-output"
    qkv_uram_remote_status = "skipped"
    qkv_uram_remote_execute = False
    if require_qkv_uram_physical_smoke or execute_qkv_uram_smoke:
        qkv_remote_argv = [
            "--profile",
            "vref-p0-softmax-input-x2-qkv-uram",
            "--root",
            str(root),
            "--host",
            zcu104_host,
            "--json-out",
            str(paths["qkv_uram_remote_json"]),
            "--markdown-out",
            str(paths["qkv_uram_remote_md"]),
        ]
        if qkv_uram_remote_dir is not None:
            qkv_remote_argv.extend(["--remote-dir", qkv_uram_remote_dir])
        if zcu104_user:
            qkv_remote_argv.extend(["--user", zcu104_user])
        if zcu104_port is not None:
            qkv_remote_argv.extend(["--port", str(zcu104_port)])
        if zcu104_identity_file is not None:
            qkv_remote_argv.extend(["--identity-file", str(zcu104_identity_file)])
        if execute_qkv_uram_smoke:
            qkv_remote_argv.append("--execute")
        qkv_step = run_step(
            "qkv-uram-remote-execute" if execute_qkv_uram_smoke else "qkv-uram-remote-dry-run",
            tools["zcu104_remote"],
            qkv_remote_argv,
        )
        steps.append(qkv_step)
        if paths["qkv_uram_remote_json"].exists():
            qkv_remote = load_json(paths["qkv_uram_remote_json"])
            qkv_uram_remote_status = str(qkv_remote.get("status", "unknown"))
            qkv_uram_remote_execute = bool(qkv_remote.get("execute", False))
        else:
            qkv_uram_remote_status = "fail" if qkv_step["exit_code"] else "missing-output"
    steps.append(
        run_step(
            "c3b-smoke-transfer-manifest",
            tools["c3b_transfer_manifest"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["c3b_transfer_manifest_json"]),
                "--markdown-out",
                str(paths["c3b_transfer_manifest_md"]),
                "--sha256-out",
                str(paths["c3b_bundle_sha256"]),
            ],
        )
    )
    steps.append(
        run_step(
            "c3b-smoke-result-contract",
            tools["c3b_smoke_contract"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["c3b_smoke_contract_json"]),
                "--markdown-out",
                str(paths["c3b_smoke_contract_md"]),
            ],
        )
    )
    xr_vits_policy_status = "skipped"
    if approve_xr_vits_replacement:
        policy_argv = [
            "--root",
            str(root),
            "--approve",
            "--approved-by",
            xr_vits_approved_by,
            "--reason",
            xr_vits_replacement_reason,
            "--json-out",
            str(root / create_xr_vits_replacement_policy.DEFAULT_POLICY_REL),
        ]
        if xr_vits_replacement_path is not None:
            policy_argv.extend(["--replacement-path", str(xr_vits_replacement_path)])
        if dry_run_xr_vits_replacement:
            policy_argv.append("--dry-run")
        policy_step = run_step(
            "xr-vits-replacement-policy-dry-run" if dry_run_xr_vits_replacement else "xr-vits-replacement-policy",
            tools["xr_vits_policy"],
            policy_argv,
        )
        steps.append(policy_step)
        if policy_step["exit_code"] == 0:
            xr_vits_policy_status = "dry-run-pass" if dry_run_xr_vits_replacement else "pass"
        else:
            xr_vits_policy_status = "fail"
    preview_policy_argv = [
        "--root",
        str(root),
        "--approve",
        "--approved-by",
        "<approved-by>",
        "--approved-at",
        "1970-01-01T00:00:00Z",
        "--reason",
        "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff",
        "--dry-run",
        "--preview-json-out",
        str(paths["xr_vits_policy_preview_json"]),
        "--preview-markdown-out",
        str(paths["xr_vits_policy_preview_md"]),
    ]
    if xr_vits_replacement_path is not None:
        preview_policy_argv.extend(["--replacement-path", str(xr_vits_replacement_path)])
    preview_policy_step = run_step("xr-vits-replacement-policy-preview", tools["xr_vits_policy"], preview_policy_argv)
    steps.append(preview_policy_step)
    xr_vits_policy_preview_status = "pass" if preview_policy_step["exit_code"] == 0 else "fail"
    steps.append(
        run_step(
            "xr-vits-unblock-packet",
            tools["xr_vits_packet"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["xr_vits_packet_json"]),
                "--markdown-out",
                str(paths["xr_vits_packet_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "xr-vits-reference-resolution",
            tools["xr_vits_reference_resolution"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["xr_vits_reference_resolution_json"]),
                "--markdown-out",
                str(paths["xr_vits_reference_resolution_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "c3b-readiness",
            tools["readiness"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["readiness_json"]),
                "--markdown-out",
                str(paths["readiness_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "final-preflight",
            tools["preflight"],
            [
                "--root",
                str(root),
                "--mode",
                "final-signoff",
                "--json-out",
                str(paths["preflight_json"]),
            ],
        )
    )
    steps.append(
        run_step(
            "final-audit",
            tools["final_audit"],
            [
                "--preflight-json",
                str(paths["preflight_json"]),
                "--json-out",
                str(paths["final_audit_json"]),
                "--markdown-out",
                str(paths["final_audit_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "completion-audit",
            tools["completion"],
            [
                "--root",
                str(root),
                "--preflight-json",
                str(paths["preflight_json"]),
                "--json-out",
                str(paths["completion_json"]),
                "--markdown-out",
                str(paths["completion_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "unblock-checklist",
            tools["unblock"],
            [
                "--root",
                str(root),
                "--final-audit",
                str(paths["final_audit_json"]),
                "--completion-audit",
                str(paths["completion_json"]),
                "--json-out",
                str(paths["unblock_json"]),
                "--markdown-out",
                str(paths["unblock_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "final-unblock-command-card",
            tools["command_card"],
            [
                "--root",
                str(root),
                "--final-audit",
                str(paths["final_audit_json"]),
                "--readiness-json",
                str(paths["readiness_json"]),
                "--xr-vits-packet-json",
                str(paths["xr_vits_packet_json"]),
                "--c3b-contract-json",
                str(paths["c3b_smoke_contract_json"]),
                "--json-out",
                str(paths["command_card_json"]),
                "--markdown-out",
                str(paths["command_card_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "resource-matrix",
            tools["resource_matrix"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["resource_matrix_json"]),
                "--markdown-out",
                str(paths["resource_matrix_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "resource-policy-audit",
            tools["resource_policy"],
            [
                "--root",
                str(root),
                "--resource-matrix",
                str(paths["resource_matrix_json"]),
                "--json-out",
                str(paths["resource_policy_json"]),
                "--markdown-out",
                str(paths["resource_policy_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "selected-path-execution-audit",
            tools["selected_path"],
            [
                "--root",
                str(root),
                "--resource-matrix",
                str(paths["resource_matrix_json"]),
                "--json-out",
                str(paths["selected_path_json"]),
                "--markdown-out",
                str(paths["selected_path_md"]),
            ],
        )
    )

    candidate_audit_argv = [
        "--root",
        str(root),
        "--json-out",
        str(paths["candidate_audit_json"]),
        "--markdown-out",
        str(paths["candidate_audit_md"]),
    ]
    if import_c3b_smoke_json is not None:
        candidate_audit_argv.extend(["--c3b-smoke-json", str(import_c3b_smoke_json)])
        if import_c3b_no_require_paths:
            candidate_audit_argv.append("--c3b-no-require-paths")
    if approve_xr_vits_replacement:
        candidate_audit_argv.extend(
            [
                "--xr-vits-mode",
                "replacement",
                "--approved-by",
                xr_vits_approved_by,
                "--reason",
                xr_vits_replacement_reason,
            ]
        )
        if xr_vits_replacement_path is not None:
            candidate_audit_argv.extend(["--replacement-path", str(xr_vits_replacement_path)])
    else:
        candidate_audit_argv.extend(["--xr-vits-mode", "exact"])
    steps.append(run_step("final-unblock-candidate-audit", tools["candidate_audit"], candidate_audit_argv))

    mirrored_pairs = [
        ("readiness_json", "doc_readiness_json"),
        ("readiness_md", "doc_readiness_md"),
        ("preflight_json", "doc_preflight_json"),
        ("final_audit_json", "doc_final_audit_json"),
        ("final_audit_md", "doc_final_audit_md"),
        ("completion_json", "doc_completion_json"),
        ("completion_md", "doc_completion_md"),
        ("unblock_json", "doc_unblock_json"),
        ("unblock_md", "doc_unblock_md"),
        ("zcu104_remote_json", "doc_zcu104_remote_json"),
        ("zcu104_remote_md", "doc_zcu104_remote_md"),
        ("vref_successor_remote_json", "doc_vref_successor_remote_json"),
        ("vref_successor_remote_md", "doc_vref_successor_remote_md"),
        ("qkv_uram_remote_json", "doc_qkv_uram_remote_json"),
        ("qkv_uram_remote_md", "doc_qkv_uram_remote_md"),
        ("c3b_transfer_manifest_json", "doc_c3b_transfer_manifest_json"),
        ("c3b_transfer_manifest_md", "doc_c3b_transfer_manifest_md"),
        ("c3b_bundle_sha256", "doc_c3b_bundle_sha256"),
        ("c3b_smoke_contract_json", "doc_c3b_smoke_contract_json"),
        ("c3b_smoke_contract_md", "doc_c3b_smoke_contract_md"),
        ("xr_vits_packet_json", "doc_xr_vits_packet_json"),
        ("xr_vits_packet_md", "doc_xr_vits_packet_md"),
        ("xr_vits_policy_preview_json", "doc_xr_vits_policy_preview_json"),
        ("xr_vits_policy_preview_md", "doc_xr_vits_policy_preview_md"),
        ("xr_vits_reference_resolution_json", "doc_xr_vits_reference_resolution_json"),
        ("xr_vits_reference_resolution_md", "doc_xr_vits_reference_resolution_md"),
        ("command_card_json", "doc_command_card_json"),
        ("command_card_md", "doc_command_card_md"),
        ("unblock_intake_json", "doc_unblock_intake_json"),
        ("unblock_intake_md", "doc_unblock_intake_md"),
        ("closeout_packet_json", "doc_closeout_packet_json"),
        ("closeout_packet_md", "doc_closeout_packet_md"),
        ("closeout_validation_json", "doc_closeout_validation_json"),
        ("closeout_validation_md", "doc_closeout_validation_md"),
        ("resource_matrix_json", "doc_resource_matrix_json"),
        ("resource_matrix_md", "doc_resource_matrix_md"),
        ("selected_path_json", "doc_selected_path_json"),
        ("selected_path_md", "doc_selected_path_md"),
        ("resource_policy_json", "doc_resource_policy_json"),
        ("resource_policy_md", "doc_resource_policy_md"),
        ("candidate_audit_json", "doc_candidate_audit_json"),
        ("candidate_audit_md", "doc_candidate_audit_md"),
        ("c3b_import_json", "doc_c3b_import_json"),
        ("c3b_import_validation_json", "doc_c3b_import_validation_json"),
        ("vref_successor_import_json", "doc_vref_successor_import_json"),
        ("vref_successor_import_validation_json", "doc_vref_successor_import_validation_json"),
        ("qkv_uram_import_json", "doc_qkv_uram_import_json"),
        ("qkv_uram_import_validation_json", "doc_qkv_uram_import_validation_json"),
    ]
    mirrored: list[str] = []
    for src_key, dst_key in mirrored_pairs:
        if paths[src_key].exists():
            copy_artifact(paths[src_key], paths[dst_key])
            mirrored.append(str(paths[dst_key]))

    preflight = load_json(paths["preflight_json"])
    final_audit = load_json(paths["final_audit_json"])
    completion = load_json(paths["completion_json"])
    unblock = load_json(paths["unblock_json"])
    readiness = load_json(paths["readiness_json"])
    zcu104_remote = load_json(paths["zcu104_remote_json"])
    c3b_transfer_manifest = (
        load_json(paths["c3b_transfer_manifest_json"]) if paths["c3b_transfer_manifest_json"].exists() else {}
    )
    c3b_smoke_contract = load_json(paths["c3b_smoke_contract_json"])
    xr_vits_packet = load_json(paths["xr_vits_packet_json"])
    xr_vits_reference_resolution = load_json(paths["xr_vits_reference_resolution_json"])
    resource_matrix = load_json(paths["resource_matrix_json"]) if paths["resource_matrix_json"].exists() else {}
    selected_path = load_json(paths["selected_path_json"]) if paths["selected_path_json"].exists() else {}
    resource_policy = load_json(paths["resource_policy_json"]) if paths["resource_policy_json"].exists() else {}
    candidate_audit = load_json(paths["candidate_audit_json"]) if paths["candidate_audit_json"].exists() else {}
    spec_plan: dict = {}
    blocker_closure: dict[str, Any] = {}
    remaining_blockers = [
        str(blocker.get("name", "unknown"))
        for blocker in final_audit.get("blockers", [])
        if isinstance(blocker, dict)
    ]
    if require_qkv_uram_physical_smoke:
        qkv_smoke_ok = any(
            isinstance(check, dict)
            and check.get("name") == "VREF-P0-02 QKV URAM physical smoke result"
            and check.get("status") == "ok"
            for check in preflight.get("checks", [])
        )
        if not qkv_smoke_ok and "VREF-P0-02 QKV URAM physical smoke result" not in remaining_blockers:
            remaining_blockers.append("VREF-P0-02 QKV URAM physical smoke result")

    status = "pass"
    if (
        preflight.get("summary", {}).get("fail", 0)
        or final_audit.get("status") != "pass"
        or completion.get("status") != "pass"
        or (
            approve_xr_vits_replacement
            and not dry_run_xr_vits_replacement
            and xr_vits_policy_status == "fail"
        )
        or (require_qkv_uram_physical_smoke and "VREF-P0-02 QKV URAM physical smoke result" in remaining_blockers)
    ):
        status = "blocked"

    def write_summary(
        requirements_trace_status: str,
        spec_plan_status: str,
        operator_handoff_status: str,
        operator_handoff_validation_status: str,
        final_bundle_validation_status: str,
        c3b_smoke_contract_status: str,
        c3b_smoke_discovery_status: str,
        pynq_c3b_smoke_discovery_status: str,
        vref_smoke_discovery_status: str,
        pynq_vref_smoke_discovery_status: str,
        qkv_uram_smoke_discovery_status: str,
        p2_vit_scale_calibration_status: str,
        final_blocker_closure_status: str,
        final_unblock_intake_status: str,
        final_unblock_closeout_packet_status: str,
        final_unblock_closeout_validation_status: str,
        source_audit_status: str,
        current_audit_status: str,
        final_evidence_manifest_status: str,
    ) -> dict:
        remaining_blocker_input_paths, remaining_blocker_details = build_remaining_blocker_details(
            root=root,
            remaining_blockers=remaining_blockers,
            blocker_closure=blocker_closure,
        )
        mirrored_artifacts = unique_ordered(mirrored)
        mirror_integrity = build_mirror_integrity(root, mirrored_artifacts)
        summary_payload = {
            "status": status,
            "root": str(root),
            "allow_blocked": allow_blocked,
            "steps": steps,
            "final_preflight_summary": preflight.get("summary", {}),
            "final_audit_status": final_audit.get("status"),
            "completion_status": completion.get("status"),
            "unblock_status": unblock.get("status"),
            "readiness_status": readiness.get("status"),
            "c3b_import_status": c3b_import_status,
            "zcu104_remote_status": zcu104_remote.get("status"),
            "zcu104_remote_execute": zcu104_remote.get("execute"),
            "vref_successor_required_for_final_signoff": require_vref_successor_physical_smoke,
            "vref_successor_import_status": vref_successor_import_status,
            "vref_successor_remote_status": vref_successor_remote_status,
            "vref_successor_remote_execute": vref_successor_remote_execute,
            "qkv_uram_required_for_final_signoff": require_qkv_uram_physical_smoke,
            "qkv_uram_import_status": qkv_uram_import_status,
            "qkv_uram_remote_status": qkv_uram_remote_status,
            "qkv_uram_remote_execute": qkv_uram_remote_execute,
            "c3b_transfer_manifest_status": c3b_transfer_manifest.get("status", "missing"),
            "c3b_smoke_contract_status": c3b_smoke_contract_status,
            "xr_vits_policy_status": xr_vits_policy_status,
            "xr_vits_policy_preview_status": xr_vits_policy_preview_status,
            "xr_vits_packet_status": xr_vits_packet.get("status"),
            "xr_vits_reference_resolution_status": xr_vits_reference_resolution.get("status"),
            "resource_matrix_status": resource_matrix.get("status", "missing"),
            "selected_path_status": selected_path.get("status", "missing"),
            "resource_policy_status": resource_policy.get("status", "missing"),
            "requirements_trace_status": requirements_trace_status,
            "spec_plan_status": spec_plan_status,
            "candidate_audit_status": candidate_audit.get("status", "missing"),
            "operator_handoff_status": operator_handoff_status,
            "operator_handoff_validation_status": operator_handoff_validation_status,
            "final_bundle_validation_status": final_bundle_validation_status,
            "c3b_smoke_discovery_status": c3b_smoke_discovery_status,
            "pynq_c3b_smoke_discovery_status": pynq_c3b_smoke_discovery_status,
            "vref_smoke_discovery_status": vref_smoke_discovery_status,
            "pynq_vref_smoke_discovery_status": pynq_vref_smoke_discovery_status,
            "qkv_uram_smoke_discovery_status": qkv_uram_smoke_discovery_status,
            "p2_vit_scale_calibration_status": p2_vit_scale_calibration_status,
            "final_blocker_closure_status": final_blocker_closure_status,
            "final_unblock_intake_status": final_unblock_intake_status,
            "final_unblock_closeout_packet_status": final_unblock_closeout_packet_status,
            "final_unblock_closeout_validation_status": final_unblock_closeout_validation_status,
            "third_goal_source_audit_status": source_audit_status,
            "third_goal_current_audit_status": current_audit_status,
            "final_evidence_manifest_status": final_evidence_manifest_status,
            "remaining_blockers": remaining_blockers,
            "blocker_count": len(remaining_blockers),
            "remaining_blocker_input_paths": remaining_blocker_input_paths,
            "remaining_blocker_details": remaining_blocker_details,
            "remaining_blocker_detail_count": len(remaining_blocker_details),
            "mirrored_artifacts": mirrored_artifacts,
            "mirrored_artifact_count": len(mirrored),
            "mirrored_artifact_unique_count": len(mirrored_artifacts),
            "mirrored_artifact_duplicate_count": len(mirrored) - len(mirrored_artifacts),
            "canonical_evidence_root": mirror_integrity["canonical_evidence_root"],
            "generated_signoff_root": mirror_integrity["generated_signoff_root"],
            "mirrored_artifact_integrity_status": mirror_integrity["status"],
            "mirrored_artifact_integrity_count": mirror_integrity["artifact_count"],
            "mirrored_artifact_integrity_checked_count": mirror_integrity["checked_count"],
            "mirrored_artifact_integrity_excluded_count": mirror_integrity["excluded_count"],
            "mirrored_artifact_integrity_fail_count": mirror_integrity["fail_count"],
            "mirrored_artifact_integrity_failures": mirror_integrity["failures"],
            "mirrored_artifact_integrity": mirror_integrity["contracts"],
        }
        paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["summary_json"].write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
        paths["summary_md"].write_text(render_markdown(summary_payload))
        return summary_payload

    write_summary(
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
        "not-run",
    )
    copy_artifact(paths["summary_json"], paths["doc_summary_json"])
    copy_artifact(paths["summary_md"], paths["doc_summary_md"])

    def mirror_if_exists(src_key: str, dst_key: str) -> None:
        if paths[src_key].exists():
            copy_artifact(paths[src_key], paths[dst_key])
            mirrored.append(str(paths[dst_key]))

    def run_requirements_trace_step(step_name: str) -> dict:
        step = run_step(
            step_name,
            tools["requirements_trace"],
            [
                "--root",
                str(root),
                "--completion-audit",
                str(paths["completion_json"]),
                "--resource-matrix",
                str(paths["resource_matrix_json"]),
                "--command-card",
                str(paths["command_card_json"]),
                "--signoff-run",
                str(paths["summary_json"]),
                "--candidate-audit",
                str(paths["candidate_audit_json"]),
                "--xr-vits-resolution",
                str(paths["xr_vits_reference_resolution_json"]),
                "--evidence-manifest",
                str(paths["doc_evidence_manifest_json"]),
                "--json-out",
                str(paths["requirements_trace_json"]),
                "--markdown-out",
                str(paths["requirements_trace_md"]),
            ],
        )
        steps.append(step)
        mirror_if_exists("requirements_trace_json", "doc_requirements_trace_json")
        mirror_if_exists("requirements_trace_md", "doc_requirements_trace_md")
        return load_json(paths["requirements_trace_json"]) if paths["requirements_trace_json"].exists() else {}

    def run_spec_plan_step(step_name: str) -> dict:
        step = run_step(
            step_name,
            tools["spec_plan"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["spec_plan_json"]),
                "--markdown-out",
                str(paths["spec_plan_md"]),
            ],
        )
        steps.append(step)
        mirror_if_exists("spec_plan_json", "doc_spec_plan_json")
        mirror_if_exists("spec_plan_md", "doc_spec_plan_md")
        return load_json(paths["spec_plan_json"]) if paths["spec_plan_json"].exists() else {}

    def run_completion_step(step_name: str) -> dict:
        step = run_step(
            step_name,
            tools["completion"],
            [
                "--root",
                str(root),
                "--preflight-json",
                str(paths["preflight_json"]),
                "--json-out",
                str(paths["completion_json"]),
                "--markdown-out",
                str(paths["completion_md"]),
            ],
        )
        steps.append(step)
        mirror_if_exists("completion_json", "doc_completion_json")
        mirror_if_exists("completion_md", "doc_completion_md")
        return load_json(paths["completion_json"]) if paths["completion_json"].exists() else {}

    def run_operator_handoff_step(step_name: str) -> dict:
        step = run_step(
            step_name,
            tools["operator_handoff"],
            [
                "--root",
                str(root),
                "--command-card",
                str(paths["command_card_json"]),
                "--requirements-trace",
                str(paths["requirements_trace_json"]),
                "--readiness",
                str(paths["readiness_json"]),
                "--candidate-audit",
                str(paths["candidate_audit_json"]),
                "--xr-vits-resolution",
                str(paths["xr_vits_reference_resolution_json"]),
                "--json-out",
                str(paths["operator_handoff_json"]),
                "--markdown-out",
                str(paths["operator_handoff_md"]),
            ],
        )
        steps.append(step)
        mirror_if_exists("operator_handoff_json", "doc_operator_handoff_json")
        mirror_if_exists("operator_handoff_md", "doc_operator_handoff_md")
        return load_json(paths["operator_handoff_json"]) if paths["operator_handoff_json"].exists() else {}

    def run_operator_handoff_validation_step(step_name: str) -> dict:
        step = run_step(
            step_name,
            tools["operator_handoff_validation"],
            [
                "--root",
                str(root),
                "--handoff",
                str(paths["operator_handoff_json"]),
                "--json-out",
                str(paths["operator_handoff_validation_json"]),
                "--markdown-out",
                str(paths["operator_handoff_validation_md"]),
            ],
        )
        steps.append(step)
        mirror_if_exists("operator_handoff_validation_json", "doc_operator_handoff_validation_json")
        mirror_if_exists("operator_handoff_validation_md", "doc_operator_handoff_validation_md")
        return (
            load_json(paths["operator_handoff_validation_json"])
            if paths["operator_handoff_validation_json"].exists()
            else {}
        )

    def run_bundle_validation_step(step_name: str) -> dict:
        step = run_step(
            step_name,
            tools["bundle_validation"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["bundle_validation_json"]),
                "--markdown-out",
                str(paths["bundle_validation_md"]),
            ],
        )
        steps.append(step)
        mirror_if_exists("bundle_validation_json", "doc_bundle_validation_json")
        mirror_if_exists("bundle_validation_md", "doc_bundle_validation_md")
        return load_json(paths["bundle_validation_json"]) if paths["bundle_validation_json"].exists() else {}

    def run_evidence_manifest_step(step_name: str) -> dict:
        step = run_step(
            step_name,
            tools["evidence_manifest"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["evidence_manifest_json"]),
                "--markdown-out",
                str(paths["evidence_manifest_md"]),
            ],
        )
        steps.append(step)
        mirror_if_exists("evidence_manifest_json", "doc_evidence_manifest_json")
        mirror_if_exists("evidence_manifest_md", "doc_evidence_manifest_md")
        return load_json(paths["evidence_manifest_json"]) if paths["evidence_manifest_json"].exists() else {}

    def refresh_final_manifest_dependents() -> tuple[dict, dict, dict, dict, dict, dict]:
        trace = run_requirements_trace_step("requirements-trace-final-refresh")
        handoff = run_operator_handoff_step("final-operator-handoff-final-refresh")
        handoff_validation = run_operator_handoff_validation_step("final-operator-handoff-validation-final-refresh")
        bundle = run_bundle_validation_step("final-signoff-bundle-validation-final-refresh")
        manifest = run_evidence_manifest_step("final-evidence-manifest-final-refresh")
        spec = run_spec_plan_step("spec-plan-conformance-final-refresh")
        manifest = run_evidence_manifest_step("final-evidence-manifest-post-spec-refresh")
        run_completion_step("completion-audit-final-refresh")
        trace = run_requirements_trace_step("requirements-trace-post-completion-refresh")
        handoff = run_operator_handoff_step("final-operator-handoff-post-completion-refresh")
        handoff_validation = run_operator_handoff_validation_step("final-operator-handoff-validation-post-completion-refresh")
        bundle = run_bundle_validation_step("final-signoff-bundle-validation-post-completion-refresh")
        manifest = run_evidence_manifest_step("final-evidence-manifest-post-completion-refresh")
        spec = run_spec_plan_step("spec-plan-conformance-post-completion-refresh")
        manifest = run_evidence_manifest_step("final-evidence-manifest-post-spec-completion-refresh")
        trace = run_requirements_trace_step("requirements-trace-final-contract-refresh")
        handoff = run_operator_handoff_step("final-operator-handoff-final-contract-refresh")
        handoff_validation = run_operator_handoff_validation_step(
            "final-operator-handoff-validation-final-contract-refresh"
        )
        bundle = run_bundle_validation_step("final-signoff-bundle-validation-final-contract-refresh")
        manifest = run_evidence_manifest_step("final-evidence-manifest-final-contract-refresh")
        spec = run_spec_plan_step("spec-plan-conformance-final-contract-refresh")
        manifest = run_evidence_manifest_step("final-evidence-manifest-post-final-contract-refresh")
        return trace, spec, handoff, handoff_validation, bundle, manifest

    steps.append(
        run_step(
            "requirements-trace",
            tools["requirements_trace"],
            [
                "--root",
                str(root),
                "--completion-audit",
                str(paths["completion_json"]),
                "--resource-matrix",
                str(paths["resource_matrix_json"]),
                "--command-card",
                str(paths["command_card_json"]),
                "--signoff-run",
                str(paths["summary_json"]),
                "--candidate-audit",
                str(paths["candidate_audit_json"]),
                "--xr-vits-resolution",
                str(paths["xr_vits_reference_resolution_json"]),
                "--evidence-manifest",
                str(paths["doc_evidence_manifest_json"]),
                "--json-out",
                str(paths["requirements_trace_json"]),
                "--markdown-out",
                str(paths["requirements_trace_md"]),
            ],
        )
    )
    if paths["requirements_trace_json"].exists():
        copy_artifact(paths["requirements_trace_json"], paths["doc_requirements_trace_json"])
        mirrored.append(str(paths["doc_requirements_trace_json"]))
    if paths["requirements_trace_md"].exists():
        copy_artifact(paths["requirements_trace_md"], paths["doc_requirements_trace_md"])
        mirrored.append(str(paths["doc_requirements_trace_md"]))
    requirements_trace = load_json(paths["requirements_trace_json"]) if paths["requirements_trace_json"].exists() else {}

    spec_plan = run_spec_plan_step("spec-plan-conformance-audit")

    steps.append(
        run_step(
            "final-operator-handoff",
            tools["operator_handoff"],
            [
                "--root",
                str(root),
                "--command-card",
                str(paths["command_card_json"]),
                "--requirements-trace",
                str(paths["requirements_trace_json"]),
                "--readiness",
                str(paths["readiness_json"]),
                "--candidate-audit",
                str(paths["candidate_audit_json"]),
                "--xr-vits-resolution",
                str(paths["xr_vits_reference_resolution_json"]),
                "--json-out",
                str(paths["operator_handoff_json"]),
                "--markdown-out",
                str(paths["operator_handoff_md"]),
            ],
        )
    )
    if paths["operator_handoff_json"].exists():
        copy_artifact(paths["operator_handoff_json"], paths["doc_operator_handoff_json"])
        mirrored.append(str(paths["doc_operator_handoff_json"]))
    if paths["operator_handoff_md"].exists():
        copy_artifact(paths["operator_handoff_md"], paths["doc_operator_handoff_md"])
        mirrored.append(str(paths["doc_operator_handoff_md"]))
    operator_handoff = load_json(paths["operator_handoff_json"]) if paths["operator_handoff_json"].exists() else {}
    steps.append(
        run_step(
            "final-operator-handoff-validation",
            tools["operator_handoff_validation"],
            [
                "--root",
                str(root),
                "--handoff",
                str(paths["operator_handoff_json"]),
                "--json-out",
                str(paths["operator_handoff_validation_json"]),
                "--markdown-out",
                str(paths["operator_handoff_validation_md"]),
            ],
        )
    )
    if paths["operator_handoff_validation_json"].exists():
        copy_artifact(paths["operator_handoff_validation_json"], paths["doc_operator_handoff_validation_json"])
        mirrored.append(str(paths["doc_operator_handoff_validation_json"]))
    if paths["operator_handoff_validation_md"].exists():
        copy_artifact(paths["operator_handoff_validation_md"], paths["doc_operator_handoff_validation_md"])
        mirrored.append(str(paths["doc_operator_handoff_validation_md"]))
    operator_handoff_validation = (
        load_json(paths["operator_handoff_validation_json"]) if paths["operator_handoff_validation_json"].exists() else {}
    )

    steps.append(
        run_step(
            "final-signoff-bundle-validation",
            tools["bundle_validation"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["bundle_validation_json"]),
                "--markdown-out",
                str(paths["bundle_validation_md"]),
            ],
        )
    )
    if paths["bundle_validation_json"].exists():
        copy_artifact(paths["bundle_validation_json"], paths["doc_bundle_validation_json"])
        mirrored.append(str(paths["doc_bundle_validation_json"]))
    if paths["bundle_validation_md"].exists():
        copy_artifact(paths["bundle_validation_md"], paths["doc_bundle_validation_md"])
        mirrored.append(str(paths["doc_bundle_validation_md"]))
    bundle_validation = load_json(paths["bundle_validation_json"]) if paths["bundle_validation_json"].exists() else {}

    steps.append(
        run_step(
            "c3b-smoke-candidate-discovery",
            tools["c3b_smoke_discovery"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["c3b_smoke_discovery_json"]),
                "--markdown-out",
                str(paths["c3b_smoke_discovery_md"]),
            ],
        )
    )
    if paths["c3b_smoke_discovery_json"].exists():
        copy_artifact(paths["c3b_smoke_discovery_json"], paths["doc_c3b_smoke_discovery_json"])
        mirrored.append(str(paths["doc_c3b_smoke_discovery_json"]))
    if paths["c3b_smoke_discovery_md"].exists():
        copy_artifact(paths["c3b_smoke_discovery_md"], paths["doc_c3b_smoke_discovery_md"])
        mirrored.append(str(paths["doc_c3b_smoke_discovery_md"]))
    c3b_smoke_discovery = (
        load_json(paths["c3b_smoke_discovery_json"]) if paths["c3b_smoke_discovery_json"].exists() else {}
    )
    steps.append(
        run_step(
            "pynq-c3b-smoke-candidate-discovery",
            tools["pynq_smoke_discovery"],
            [
                "--root",
                str(root),
                "--preset",
                "axis-c3b-mem16",
                "--json-out",
                str(paths["pynq_c3b_smoke_discovery_json"]),
                "--markdown-out",
                str(paths["pynq_c3b_smoke_discovery_md"]),
            ],
        )
    )
    if paths["pynq_c3b_smoke_discovery_json"].exists():
        copy_artifact(paths["pynq_c3b_smoke_discovery_json"], paths["doc_pynq_c3b_smoke_discovery_json"])
        mirrored.append(str(paths["doc_pynq_c3b_smoke_discovery_json"]))
    if paths["pynq_c3b_smoke_discovery_md"].exists():
        copy_artifact(paths["pynq_c3b_smoke_discovery_md"], paths["doc_pynq_c3b_smoke_discovery_md"])
        mirrored.append(str(paths["doc_pynq_c3b_smoke_discovery_md"]))
    pynq_c3b_smoke_discovery = (
        load_json(paths["pynq_c3b_smoke_discovery_json"])
        if paths["pynq_c3b_smoke_discovery_json"].exists()
        else {}
    )
    steps.append(
        run_step(
            "vref-successor-smoke-candidate-discovery",
            tools["pynq_smoke_discovery"],
            [
                "--root",
                str(root),
                "--preset",
                "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                "--json-out",
                str(paths["vref_smoke_discovery_json"]),
                "--markdown-out",
                str(paths["vref_smoke_discovery_md"]),
            ],
        )
    )
    if paths["vref_smoke_discovery_json"].exists():
        copy_artifact(paths["vref_smoke_discovery_json"], paths["doc_vref_smoke_discovery_json"])
        mirrored.append(str(paths["doc_vref_smoke_discovery_json"]))
    if paths["vref_smoke_discovery_md"].exists():
        copy_artifact(paths["vref_smoke_discovery_md"], paths["doc_vref_smoke_discovery_md"])
        mirrored.append(str(paths["doc_vref_smoke_discovery_md"]))
    vref_smoke_discovery = (
        load_json(paths["vref_smoke_discovery_json"]) if paths["vref_smoke_discovery_json"].exists() else {}
    )
    steps.append(
        run_step(
            "pynq-vref-smoke-candidate-discovery",
            tools["pynq_smoke_discovery"],
            [
                "--root",
                str(root),
                "--preset",
                "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                "--json-out",
                str(paths["pynq_vref_smoke_discovery_json"]),
                "--markdown-out",
                str(paths["pynq_vref_smoke_discovery_md"]),
            ],
        )
    )
    if paths["pynq_vref_smoke_discovery_json"].exists():
        copy_artifact(paths["pynq_vref_smoke_discovery_json"], paths["doc_pynq_vref_smoke_discovery_json"])
        mirrored.append(str(paths["doc_pynq_vref_smoke_discovery_json"]))
    if paths["pynq_vref_smoke_discovery_md"].exists():
        copy_artifact(paths["pynq_vref_smoke_discovery_md"], paths["doc_pynq_vref_smoke_discovery_md"])
        mirrored.append(str(paths["doc_pynq_vref_smoke_discovery_md"]))
    pynq_vref_smoke_discovery = (
        load_json(paths["pynq_vref_smoke_discovery_json"])
        if paths["pynq_vref_smoke_discovery_json"].exists()
        else {}
    )
    steps.append(
        run_step(
            "qkv-uram-smoke-candidate-discovery",
            tools["pynq_smoke_discovery"],
            [
                "--root",
                str(root),
                "--preset",
                "axis-vref-p0-softmax-input-x2-qkv-uram",
                "--json-out",
                str(paths["qkv_uram_smoke_discovery_json"]),
                "--markdown-out",
                str(paths["qkv_uram_smoke_discovery_md"]),
            ],
        )
    )
    if paths["qkv_uram_smoke_discovery_json"].exists():
        copy_artifact(paths["qkv_uram_smoke_discovery_json"], paths["doc_qkv_uram_smoke_discovery_json"])
        mirrored.append(str(paths["doc_qkv_uram_smoke_discovery_json"]))
    if paths["qkv_uram_smoke_discovery_md"].exists():
        copy_artifact(paths["qkv_uram_smoke_discovery_md"], paths["doc_qkv_uram_smoke_discovery_md"])
        mirrored.append(str(paths["doc_qkv_uram_smoke_discovery_md"]))
    qkv_uram_smoke_discovery = (
        load_json(paths["qkv_uram_smoke_discovery_json"]) if paths["qkv_uram_smoke_discovery_json"].exists() else {}
    )

    steps.append(
        run_step(
            "vref-p0-qkv-uram-cache-successor",
            tools["qkv_uram_successor"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["qkv_uram_successor_json"]),
                "--markdown-out",
                str(paths["qkv_uram_successor_md"]),
            ],
        )
    )
    if paths["qkv_uram_successor_json"].exists():
        copy_artifact(paths["qkv_uram_successor_json"], paths["doc_qkv_uram_successor_json"])
        mirrored.append(str(paths["doc_qkv_uram_successor_json"]))
    if paths["qkv_uram_successor_md"].exists():
        copy_artifact(paths["qkv_uram_successor_md"], paths["doc_qkv_uram_successor_md"])
        mirrored.append(str(paths["doc_qkv_uram_successor_md"]))

    steps.append(
        run_step(
            "vref-p0-pot-scale-audit",
            tools["vref_p0_pot_scale_audit"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["vref_p0_pot_scale_audit_json"]),
                "--markdown-out",
                str(paths["vref_p0_pot_scale_audit_md"]),
            ],
        )
    )
    if paths["vref_p0_pot_scale_audit_json"].exists():
        copy_artifact(paths["vref_p0_pot_scale_audit_json"], paths["doc_vref_p0_pot_scale_audit_json"])
        mirrored.append(str(paths["doc_vref_p0_pot_scale_audit_json"]))
    if paths["vref_p0_pot_scale_audit_md"].exists():
        copy_artifact(paths["vref_p0_pot_scale_audit_md"], paths["doc_vref_p0_pot_scale_audit_md"])
        mirrored.append(str(paths["doc_vref_p0_pot_scale_audit_md"]))

    steps.append(
        run_step(
            "vref-p0-pot-scale-sweep",
            tools["vref_p0_pot_scale_sweep"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["vref_p0_pot_scale_sweep_json"]),
                "--markdown-out",
                str(paths["vref_p0_pot_scale_sweep_md"]),
            ],
        )
    )
    if paths["vref_p0_pot_scale_sweep_json"].exists():
        copy_artifact(paths["vref_p0_pot_scale_sweep_json"], paths["doc_vref_p0_pot_scale_sweep_json"])
        mirrored.append(str(paths["doc_vref_p0_pot_scale_sweep_json"]))
    if paths["vref_p0_pot_scale_sweep_md"].exists():
        copy_artifact(paths["vref_p0_pot_scale_sweep_md"], paths["doc_vref_p0_pot_scale_sweep_md"])
        mirrored.append(str(paths["doc_vref_p0_pot_scale_sweep_md"]))

    steps.append(
        run_step(
            "req5-q4q8-swhw-match-audit",
            tools["req5_q4q8_swhw"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["req5_q4q8_swhw_json"]),
                "--markdown-out",
                str(paths["req5_q4q8_swhw_md"]),
            ],
        )
    )
    if paths["req5_q4q8_swhw_json"].exists():
        copy_artifact(paths["req5_q4q8_swhw_json"], paths["doc_req5_q4q8_swhw_json"])
        mirrored.append(str(paths["doc_req5_q4q8_swhw_json"]))
    if paths["req5_q4q8_swhw_md"].exists():
        copy_artifact(paths["req5_q4q8_swhw_md"], paths["doc_req5_q4q8_swhw_md"])
        mirrored.append(str(paths["doc_req5_q4q8_swhw_md"]))

    steps.append(
        run_step(
            "p2-vit-scale-calibration-report",
            tools["p2_vit_scale_calibration"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["p2_vit_scale_calibration_json"]),
                "--markdown-out",
                str(paths["p2_vit_scale_calibration_md"]),
            ],
        )
    )
    if paths["p2_vit_scale_calibration_json"].exists():
        copy_artifact(paths["p2_vit_scale_calibration_json"], paths["doc_p2_vit_scale_calibration_json"])
        mirrored.append(str(paths["doc_p2_vit_scale_calibration_json"]))
    if paths["p2_vit_scale_calibration_md"].exists():
        copy_artifact(paths["p2_vit_scale_calibration_md"], paths["doc_p2_vit_scale_calibration_md"])
        mirrored.append(str(paths["doc_p2_vit_scale_calibration_md"]))
    p2_vit_scale_calibration = (
        load_json(paths["p2_vit_scale_calibration_json"])
        if paths["p2_vit_scale_calibration_json"].exists()
        else {}
    )

    steps.append(
        run_step(
            "req1-environment-audit",
            tools["req1_environment"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["req1_environment_json"]),
                "--markdown-out",
                str(paths["req1_environment_md"]),
            ],
        )
    )
    if paths["req1_environment_json"].exists():
        copy_artifact(paths["req1_environment_json"], paths["doc_req1_environment_json"])
        mirrored.append(str(paths["doc_req1_environment_json"]))
    if paths["req1_environment_md"].exists():
        copy_artifact(paths["req1_environment_md"], paths["doc_req1_environment_md"])
        mirrored.append(str(paths["doc_req1_environment_md"]))

    steps.append(
        run_step(
            "req6-parameterization-audit",
            tools["req6_parameterization"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["req6_parameterization_json"]),
                "--markdown-out",
                str(paths["req6_parameterization_md"]),
            ],
        )
    )
    if paths["req6_parameterization_json"].exists():
        copy_artifact(paths["req6_parameterization_json"], paths["doc_req6_parameterization_json"])
        mirrored.append(str(paths["doc_req6_parameterization_json"]))
    if paths["req6_parameterization_md"].exists():
        copy_artifact(paths["req6_parameterization_md"], paths["doc_req6_parameterization_md"])
        mirrored.append(str(paths["doc_req6_parameterization_md"]))

    steps.append(
        run_step(
            "req9-deit-image-reference-audit",
            tools["req9_deit_image"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["req9_deit_image_json"]),
                "--markdown-out",
                str(paths["req9_deit_image_md"]),
            ],
        )
    )
    if paths["req9_deit_image_json"].exists():
        copy_artifact(paths["req9_deit_image_json"], paths["doc_req9_deit_image_json"])
        mirrored.append(str(paths["doc_req9_deit_image_json"]))
    if paths["req9_deit_image_md"].exists():
        copy_artifact(paths["req9_deit_image_md"], paths["doc_req9_deit_image_md"])
        mirrored.append(str(paths["doc_req9_deit_image_md"]))

    steps.append(
        run_step(
            "xr-vits-gate-audit",
            tools["xr_vits_gate"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["xr_vits_gate_json"]),
                "--markdown-out",
                str(paths["xr_vits_gate_md"]),
            ],
        )
    )
    if paths["xr_vits_gate_json"].exists():
        copy_artifact(paths["xr_vits_gate_json"], paths["doc_xr_vits_gate_json"])
        mirrored.append(str(paths["doc_xr_vits_gate_json"]))
    if paths["xr_vits_gate_md"].exists():
        copy_artifact(paths["xr_vits_gate_md"], paths["doc_xr_vits_gate_md"])
        mirrored.append(str(paths["doc_xr_vits_gate_md"]))

    steps.append(
        run_step(
            "c3b-physical-smoke-gate-audit",
            tools["c3b_physical_smoke_gate"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["c3b_physical_smoke_gate_json"]),
                "--markdown-out",
                str(paths["c3b_physical_smoke_gate_md"]),
            ],
        )
    )
    if paths["c3b_physical_smoke_gate_json"].exists():
        copy_artifact(paths["c3b_physical_smoke_gate_json"], paths["doc_c3b_physical_smoke_gate_json"])
        mirrored.append(str(paths["doc_c3b_physical_smoke_gate_json"]))
    if paths["c3b_physical_smoke_gate_md"].exists():
        copy_artifact(paths["c3b_physical_smoke_gate_md"], paths["doc_c3b_physical_smoke_gate_md"])
        mirrored.append(str(paths["doc_c3b_physical_smoke_gate_md"]))

    steps.append(
        run_step(
            "vref-p0-buffer-lifetime-audit",
            tools["vref_p0_buffer_lifetime"],
            [
                "--root",
                str(root),
                "--resource-matrix",
                str(paths["resource_matrix_json"]),
                "--json-out",
                str(paths["vref_p0_buffer_lifetime_json"]),
                "--markdown-out",
                str(paths["vref_p0_buffer_lifetime_md"]),
            ],
        )
    )
    if paths["vref_p0_buffer_lifetime_json"].exists():
        copy_artifact(paths["vref_p0_buffer_lifetime_json"], paths["doc_vref_p0_buffer_lifetime_json"])
        mirrored.append(str(paths["doc_vref_p0_buffer_lifetime_json"]))
    if paths["vref_p0_buffer_lifetime_md"].exists():
        copy_artifact(paths["vref_p0_buffer_lifetime_md"], paths["doc_vref_p0_buffer_lifetime_md"])
        mirrored.append(str(paths["doc_vref_p0_buffer_lifetime_md"]))

    steps.append(
        run_step(
            "hgpipe-operator-audit",
            tools["hgpipe_operator_audit"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["hgpipe_operator_audit_json"]),
                "--markdown-out",
                str(paths["hgpipe_operator_audit_md"]),
            ],
        )
    )
    if paths["hgpipe_operator_audit_json"].exists():
        copy_artifact(paths["hgpipe_operator_audit_json"], paths["doc_hgpipe_operator_audit_json"])
        mirrored.append(str(paths["doc_hgpipe_operator_audit_json"]))
    if paths["hgpipe_operator_audit_md"].exists():
        copy_artifact(paths["hgpipe_operator_audit_md"], paths["doc_hgpipe_operator_audit_md"])
        mirrored.append(str(paths["doc_hgpipe_operator_audit_md"]))

    blocker_closure_argv = [
        "--root",
        str(root),
        "--json-out",
        str(paths["blocker_closure_json"]),
        "--markdown-out",
        str(paths["blocker_closure_md"]),
    ]
    if import_c3b_smoke_json is not None:
        blocker_closure_argv.extend(["--c3b-smoke-json", str(import_c3b_smoke_json)])
        if import_c3b_no_require_paths:
            blocker_closure_argv.append("--c3b-no-require-paths")
    if approve_xr_vits_replacement:
        blocker_closure_argv.extend(
            [
                "--xr-vits-mode",
                "replacement",
                "--approved-by",
                xr_vits_approved_by,
                "--reason",
                xr_vits_replacement_reason,
            ]
        )
        if xr_vits_replacement_path is not None:
            blocker_closure_argv.extend(["--replacement-path", str(xr_vits_replacement_path)])
    else:
        blocker_closure_argv.extend(["--xr-vits-mode", "exact"])
    steps.append(run_step("final-blocker-closure-readiness", tools["blocker_closure"], blocker_closure_argv))
    if paths["blocker_closure_json"].exists():
        copy_artifact(paths["blocker_closure_json"], paths["doc_blocker_closure_json"])
        mirrored.append(str(paths["doc_blocker_closure_json"]))
    if paths["blocker_closure_md"].exists():
        copy_artifact(paths["blocker_closure_md"], paths["doc_blocker_closure_md"])
        mirrored.append(str(paths["doc_blocker_closure_md"]))
    blocker_closure = load_json(paths["blocker_closure_json"]) if paths["blocker_closure_json"].exists() else {}

    unblock_intake_argv = [
        "--root",
        str(root),
        "--json-out",
        str(paths["unblock_intake_json"]),
        "--markdown-out",
        str(paths["unblock_intake_md"]),
    ]
    if import_c3b_smoke_json is not None:
        unblock_intake_argv.extend(["--c3b-smoke-json", str(import_c3b_smoke_json)])
        if import_c3b_no_require_paths:
            unblock_intake_argv.append("--c3b-no-require-paths")
    if approve_xr_vits_replacement:
        unblock_intake_argv.extend(
            [
                "--xr-vits-mode",
                "replacement",
                "--approved-by",
                xr_vits_approved_by,
                "--reason",
                xr_vits_replacement_reason,
            ]
        )
        if xr_vits_replacement_path is not None:
            unblock_intake_argv.extend(["--replacement-path", str(xr_vits_replacement_path)])
    else:
        unblock_intake_argv.extend(["--xr-vits-mode", "exact"])
    steps.append(run_step("final-unblock-intake", tools["unblock_intake"], unblock_intake_argv))
    if paths["unblock_intake_json"].exists():
        copy_artifact(paths["unblock_intake_json"], paths["doc_unblock_intake_json"])
        mirrored.append(str(paths["doc_unblock_intake_json"]))
    if paths["unblock_intake_md"].exists():
        copy_artifact(paths["unblock_intake_md"], paths["doc_unblock_intake_md"])
        mirrored.append(str(paths["doc_unblock_intake_md"]))
    unblock_intake = load_json(paths["unblock_intake_json"]) if paths["unblock_intake_json"].exists() else {}

    closeout_packet_argv = [
        "--root",
        str(root),
        "--json-out",
        str(paths["closeout_packet_json"]),
        "--markdown-out",
        str(paths["closeout_packet_md"]),
    ]
    if require_vref_successor_physical_smoke:
        closeout_packet_argv.append("--require-vref-successor-physical-smoke")
    steps.append(
        run_step(
            "final-unblock-closeout-packet",
            tools["closeout_packet"],
            closeout_packet_argv,
        )
    )
    if paths["closeout_packet_json"].exists():
        copy_artifact(paths["closeout_packet_json"], paths["doc_closeout_packet_json"])
        mirrored.append(str(paths["doc_closeout_packet_json"]))
    if paths["closeout_packet_md"].exists():
        copy_artifact(paths["closeout_packet_md"], paths["doc_closeout_packet_md"])
        mirrored.append(str(paths["doc_closeout_packet_md"]))
    closeout_packet = load_json(paths["closeout_packet_json"]) if paths["closeout_packet_json"].exists() else {}

    steps.append(
        run_step(
            "final-unblock-closeout-validation",
            tools["closeout_validation"],
            [
                "--root",
                str(root),
                "--packet",
                str(paths["closeout_packet_json"]),
                "--json-out",
                str(paths["closeout_validation_json"]),
                "--markdown-out",
                str(paths["closeout_validation_md"]),
            ],
        )
    )
    if paths["closeout_validation_json"].exists():
        copy_artifact(paths["closeout_validation_json"], paths["doc_closeout_validation_json"])
        mirrored.append(str(paths["doc_closeout_validation_json"]))
    if paths["closeout_validation_md"].exists():
        copy_artifact(paths["closeout_validation_md"], paths["doc_closeout_validation_md"])
        mirrored.append(str(paths["doc_closeout_validation_md"]))
    closeout_validation = load_json(paths["closeout_validation_json"]) if paths["closeout_validation_json"].exists() else {}

    steps.append(
        run_step(
            "third-goal-source-audit",
            tools["source_audit"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["source_audit_json"]),
                "--markdown-out",
                str(paths["source_audit_md"]),
            ],
        )
    )
    steps.append(
        run_step(
            "third-goal-current-audit",
            tools["current_audit"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["current_audit_json"]),
                "--markdown-out",
                str(paths["current_audit_md"]),
            ],
        )
    )
    # Current audit consumes source-audit state; refresh source audit once more so
    # final evidence hashes include the current-audit artifact it just produced.
    steps.append(
        run_step(
            "third-goal-source-audit-refresh",
            tools["source_audit"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["source_audit_json"]),
                "--markdown-out",
                str(paths["source_audit_md"]),
            ],
        )
    )
    if paths["source_audit_json"].exists():
        copy_artifact(paths["source_audit_json"], paths["doc_source_audit_json"])
        mirrored.append(str(paths["doc_source_audit_json"]))
    if paths["source_audit_md"].exists():
        copy_artifact(paths["source_audit_md"], paths["doc_source_audit_md"])
        mirrored.append(str(paths["doc_source_audit_md"]))
    if paths["current_audit_json"].exists():
        copy_artifact(paths["current_audit_json"], paths["doc_current_audit_json"])
        mirrored.append(str(paths["doc_current_audit_json"]))
    if paths["current_audit_md"].exists():
        copy_artifact(paths["current_audit_md"], paths["doc_current_audit_md"])
        mirrored.append(str(paths["doc_current_audit_md"]))
    source_audit = load_json(paths["source_audit_json"]) if paths["source_audit_json"].exists() else {}
    current_audit = load_json(paths["current_audit_json"]) if paths["current_audit_json"].exists() else {}

    steps.append(
        run_step(
            "final-evidence-manifest",
            tools["evidence_manifest"],
            [
                "--root",
                str(root),
                "--json-out",
                str(paths["evidence_manifest_json"]),
                "--markdown-out",
                str(paths["evidence_manifest_md"]),
            ],
        )
    )
    if paths["evidence_manifest_json"].exists():
        copy_artifact(paths["evidence_manifest_json"], paths["doc_evidence_manifest_json"])
        mirrored.append(str(paths["doc_evidence_manifest_json"]))
    if paths["evidence_manifest_md"].exists():
        copy_artifact(paths["evidence_manifest_md"], paths["doc_evidence_manifest_md"])
        mirrored.append(str(paths["doc_evidence_manifest_md"]))
    evidence_manifest = load_json(paths["evidence_manifest_json"]) if paths["evidence_manifest_json"].exists() else {}
    (
        requirements_trace,
        spec_plan,
        operator_handoff,
        operator_handoff_validation,
        bundle_validation,
        evidence_manifest,
    ) = refresh_final_manifest_dependents()

    summary = write_summary(
        str(requirements_trace.get("status", "missing")),
        str(spec_plan.get("status", "missing")),
        str(operator_handoff.get("status", "missing")),
        str(operator_handoff_validation.get("status", "missing")),
        str(bundle_validation.get("status", "missing")),
        str(c3b_smoke_contract.get("status", "missing")),
        str(c3b_smoke_discovery.get("status", "missing")),
        str(pynq_c3b_smoke_discovery.get("status", "missing")),
        str(vref_smoke_discovery.get("status", "missing")),
        str(pynq_vref_smoke_discovery.get("status", "missing")),
        str(qkv_uram_smoke_discovery.get("status", "missing")),
        str(p2_vit_scale_calibration.get("status", "missing")),
        str(blocker_closure.get("status", "missing")),
        str(unblock_intake.get("status", "missing")),
        str(closeout_packet.get("status", "missing")),
        str(closeout_validation.get("status", "missing")),
        str(source_audit.get("status", "missing")),
        str(current_audit.get("status", "missing")),
        str(evidence_manifest.get("status", "missing")),
    )
    copy_artifact(paths["summary_json"], paths["doc_summary_json"])
    copy_artifact(paths["summary_md"], paths["doc_summary_md"])
    summary["mirrored_artifacts"].extend([str(paths["doc_summary_json"]), str(paths["doc_summary_md"])])
    summary["mirrored_artifacts"] = unique_ordered(summary["mirrored_artifacts"])
    summary["mirrored_artifact_count"] = len(summary["mirrored_artifacts"])
    summary["mirrored_artifact_unique_count"] = len(set(summary["mirrored_artifacts"]))
    summary["mirrored_artifact_duplicate_count"] = (
        summary["mirrored_artifact_count"] - summary["mirrored_artifact_unique_count"]
    )
    mirror_integrity = build_mirror_integrity(root, summary["mirrored_artifacts"])
    summary.update(
        {
            "canonical_evidence_root": mirror_integrity["canonical_evidence_root"],
            "generated_signoff_root": mirror_integrity["generated_signoff_root"],
            "mirrored_artifact_integrity_status": mirror_integrity["status"],
            "mirrored_artifact_integrity_count": mirror_integrity["artifact_count"],
            "mirrored_artifact_integrity_checked_count": mirror_integrity["checked_count"],
            "mirrored_artifact_integrity_excluded_count": mirror_integrity["excluded_count"],
            "mirrored_artifact_integrity_fail_count": mirror_integrity["fail_count"],
            "mirrored_artifact_integrity_failures": mirror_integrity["failures"],
            "mirrored_artifact_integrity": mirror_integrity["contracts"],
        }
    )
    paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    paths["summary_md"].write_text(render_markdown(summary))
    copy_artifact(paths["summary_json"], paths["doc_summary_json"])
    copy_artifact(paths["summary_md"], paths["doc_summary_md"])

    if status == "pass" or allow_blocked:
        return 0, summary
    return 1, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate HGTXR third-goal final signoff evidence.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--allow-blocked", action="store_true", help="return 0 after writing blocked evidence")
    parser.add_argument("--zcu104-host", default="zcu104.local")
    parser.add_argument("--zcu104-user", default="xilinx")
    parser.add_argument("--zcu104-port", type=int, default=None)
    parser.add_argument("--zcu104-identity-file", type=Path, default=None)
    parser.add_argument("--zcu104-remote-dir", default=run_zcu104_c3b_smoke_remote.DEFAULT_REMOTE_DIR)
    parser.add_argument("--vref-successor-remote-dir", default=None, help="remote dir override for VREF successor smoke; omitted uses the profile default")
    parser.add_argument("--qkv-uram-remote-dir", default=None, help="remote dir override for QKV URAM successor smoke; omitted uses the profile default")
    parser.add_argument("--execute-zcu104-smoke", action="store_true", help="run the remote C3b ZCU104 smoke instead of dry-run")
    parser.add_argument("--import-c3b-smoke-json", type=Path, default=None, help="validate and import an existing C3b board smoke JSON before signoff checks")
    parser.add_argument("--import-c3b-no-require-paths", action="store_true", help="do not require bitfile/hwhfile fields when importing C3b smoke JSON")
    parser.add_argument("--dry-run-import-c3b-smoke", action="store_true", help="validate C3b smoke JSON without copying it into the canonical path")
    parser.add_argument("--require-vref-successor-physical-smoke", action="store_true", help="treat VREF-P0 successor physical smoke as a hard closeout blocker")
    parser.add_argument("--execute-vref-successor-smoke", action="store_true", help="run the remote VREF-P0 successor ZCU104 smoke instead of dry-run")
    parser.add_argument("--import-vref-successor-smoke-json", type=Path, default=None, help="validate and import an existing VREF-P0 successor board smoke JSON before signoff checks")
    parser.add_argument("--import-vref-successor-no-require-paths", action="store_true", help="do not require bitfile/hwhfile fields when importing VREF successor smoke JSON")
    parser.add_argument("--dry-run-import-vref-successor-smoke", action="store_true", help="validate VREF successor smoke JSON without copying it into the canonical path")
    parser.add_argument("--require-qkv-uram-physical-smoke", action="store_true", help="treat VREF-P0-02 QKV URAM physical smoke as a hard closeout blocker")
    parser.add_argument("--execute-qkv-uram-smoke", action="store_true", help="run the remote QKV URAM ZCU104 smoke instead of dry-run")
    parser.add_argument("--import-qkv-uram-smoke-json", type=Path, default=None, help="validate and import an existing QKV URAM board smoke JSON before signoff checks")
    parser.add_argument("--import-qkv-uram-no-require-paths", action="store_true", help="do not require bitfile/hwhfile fields when importing QKV URAM smoke JSON")
    parser.add_argument("--dry-run-import-qkv-uram-smoke", action="store_true", help="validate QKV URAM smoke JSON without copying it into the canonical path")
    parser.add_argument("--approve-xr-vits-replacement", action="store_true", help="create active XR-VITs replacement policy before signoff checks")
    parser.add_argument("--dry-run-xr-vits-replacement", action="store_true", help="validate XR-VITs replacement policy inputs without writing active policy")
    parser.add_argument("--xr-vits-approved-by", default="")
    parser.add_argument("--xr-vits-replacement-reason", default="")
    parser.add_argument("--xr-vits-replacement-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    code, summary = run_pipeline(
        root=args.root,
        allow_blocked=args.allow_blocked,
        zcu104_host=args.zcu104_host,
        zcu104_user=args.zcu104_user,
        zcu104_port=args.zcu104_port,
        zcu104_identity_file=args.zcu104_identity_file,
        zcu104_remote_dir=args.zcu104_remote_dir,
        vref_successor_remote_dir=args.vref_successor_remote_dir,
        qkv_uram_remote_dir=args.qkv_uram_remote_dir,
        execute_zcu104_smoke=args.execute_zcu104_smoke,
        import_c3b_smoke_json=args.import_c3b_smoke_json,
        import_c3b_no_require_paths=args.import_c3b_no_require_paths,
        dry_run_import_c3b_smoke=args.dry_run_import_c3b_smoke,
        require_vref_successor_physical_smoke=args.require_vref_successor_physical_smoke,
        execute_vref_successor_smoke=args.execute_vref_successor_smoke,
        import_vref_successor_smoke_json=args.import_vref_successor_smoke_json,
        import_vref_successor_no_require_paths=args.import_vref_successor_no_require_paths,
        dry_run_import_vref_successor_smoke=args.dry_run_import_vref_successor_smoke,
        require_qkv_uram_physical_smoke=args.require_qkv_uram_physical_smoke,
        execute_qkv_uram_smoke=args.execute_qkv_uram_smoke,
        import_qkv_uram_smoke_json=args.import_qkv_uram_smoke_json,
        import_qkv_uram_no_require_paths=args.import_qkv_uram_no_require_paths,
        dry_run_import_qkv_uram_smoke=args.dry_run_import_qkv_uram_smoke,
        approve_xr_vits_replacement=args.approve_xr_vits_replacement,
        dry_run_xr_vits_replacement=args.dry_run_xr_vits_replacement,
        xr_vits_approved_by=args.xr_vits_approved_by,
        xr_vits_replacement_reason=args.xr_vits_replacement_reason,
        xr_vits_replacement_path=args.xr_vits_replacement_path,
    )
    print(
        f"[third-goal-final-signoff-run] status={summary['status']} "
        f"fail={summary['final_preflight_summary'].get('fail', 'n/a')} "
        f"blockers={len(summary['remaining_blockers'])}"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
