#!/usr/bin/env python3
"""Write a hash manifest for final signoff evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_10"
CURRENT_AUDIT_DATE_TAG = "2026_06_16"
HGTXR_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ARTIFACTS = [
    ("c3b_readiness_json", f"docs/resources/c3b_board_smoke_readiness_{DATE_TAG}.json", "json"),
    ("c3b_readiness_md", f"docs/resources/c3b_board_smoke_readiness_{DATE_TAG}.md", "markdown"),
    ("final_preflight_json", f"docs/resources/third_goal_final_signoff_{DATE_TAG}.json", "json"),
    ("final_audit_json", f"docs/resources/final_signoff_audit_{DATE_TAG}.json", "json"),
    ("final_audit_md", f"docs/resources/final_signoff_audit_{DATE_TAG}.md", "markdown"),
    ("completion_json", f"docs/resources/third_goal_completion_audit_{DATE_TAG}.json", "json"),
    ("completion_md", f"docs/resources/third_goal_completion_audit_{DATE_TAG}.md", "markdown"),
    ("unblock_json", f"docs/resources/third_goal_unblock_checklist_{DATE_TAG}.json", "json"),
    ("unblock_md", f"docs/resources/third_goal_unblock_checklist_{DATE_TAG}.md", "markdown"),
    ("zcu104_remote_json", f"docs/resources/zcu104_c3b_smoke_remote_run_{DATE_TAG}.json", "json"),
    ("zcu104_remote_md", f"docs/resources/zcu104_c3b_smoke_remote_run_{DATE_TAG}.md", "markdown"),
    ("c3b_smoke_contract_json", f"docs/resources/c3b_smoke_result_contract_{DATE_TAG}.json", "json"),
    ("c3b_smoke_contract_md", f"docs/resources/c3b_smoke_result_contract_{DATE_TAG}.md", "markdown"),
    ("xr_vits_packet_json", f"docs/resources/xr_vits_unblock_packet_{DATE_TAG}.json", "json"),
    ("xr_vits_packet_md", f"docs/resources/xr_vits_unblock_packet_{DATE_TAG}.md", "markdown"),
    ("xr_vits_policy_preview_json", f"docs/resources/xr_vits_replacement_policy_preview_{DATE_TAG}.json", "json"),
    ("xr_vits_policy_preview_md", f"docs/resources/xr_vits_replacement_policy_preview_{DATE_TAG}.md", "markdown"),
    ("xr_vits_reference_resolution_json", f"docs/resources/xr_vits_reference_resolution_{DATE_TAG}.json", "json"),
    ("xr_vits_reference_resolution_md", f"docs/resources/xr_vits_reference_resolution_{DATE_TAG}.md", "markdown"),
    ("command_card_json", f"docs/resources/final_unblock_commands_{DATE_TAG}.json", "json"),
    ("command_card_md", f"docs/resources/final_unblock_commands_{DATE_TAG}.md", "markdown"),
    ("unblock_intake_json", f"docs/resources/final_unblock_intake_{DATE_TAG}.json", "json"),
    ("unblock_intake_md", f"docs/resources/final_unblock_intake_{DATE_TAG}.md", "markdown"),
    ("closeout_packet_json", f"docs/resources/final_unblock_closeout_packet_{DATE_TAG}.json", "json"),
    ("closeout_packet_md", f"docs/resources/final_unblock_closeout_packet_{DATE_TAG}.md", "markdown"),
    (
        "closeout_validation_json",
        f"docs/resources/final_unblock_closeout_packet_validation_{DATE_TAG}.json",
        "json",
    ),
    (
        "closeout_validation_md",
        f"docs/resources/final_unblock_closeout_packet_validation_{DATE_TAG}.md",
        "markdown",
    ),
    ("resource_matrix_json", f"docs/resources/e2e_resource_matrix_{DATE_TAG}.json", "json"),
    ("resource_matrix_md", f"docs/resources/e2e_resource_matrix_{DATE_TAG}.md", "markdown"),
    ("selected_path_json", f"docs/resources/selected_path_execution_audit_{DATE_TAG}.json", "json"),
    ("selected_path_md", f"docs/resources/selected_path_execution_audit_{DATE_TAG}.md", "markdown"),
    ("resource_policy_json", f"docs/resources/e2e_resource_policy_audit_{DATE_TAG}.json", "json"),
    ("resource_policy_md", f"docs/resources/e2e_resource_policy_audit_{DATE_TAG}.md", "markdown"),
    ("requirements_trace_json", f"docs/resources/third_goal_requirements_trace_{DATE_TAG}.json", "json"),
    ("requirements_trace_md", f"docs/resources/third_goal_requirements_trace_{DATE_TAG}.md", "markdown"),
    ("spec_plan_json", f"docs/resources/spec_plan_conformance_audit_{DATE_TAG}.json", "json"),
    ("spec_plan_md", f"docs/resources/spec_plan_conformance_audit_{DATE_TAG}.md", "markdown"),
    ("candidate_audit_json", f"docs/resources/final_unblock_candidate_audit_{DATE_TAG}.json", "json"),
    ("candidate_audit_md", f"docs/resources/final_unblock_candidate_audit_{DATE_TAG}.md", "markdown"),
    ("operator_handoff_json", f"docs/resources/final_operator_handoff_{DATE_TAG}.json", "json"),
    ("operator_handoff_md", f"docs/resources/final_operator_handoff_{DATE_TAG}.md", "markdown"),
    (
        "operator_handoff_validation_json",
        f"docs/resources/final_operator_handoff_validation_{DATE_TAG}.json",
        "json",
    ),
    (
        "operator_handoff_validation_md",
        f"docs/resources/final_operator_handoff_validation_{DATE_TAG}.md",
        "markdown",
    ),
    ("final_bundle_validation_json", f"docs/resources/final_signoff_bundle_validation_{DATE_TAG}.json", "json"),
    ("final_bundle_validation_md", f"docs/resources/final_signoff_bundle_validation_{DATE_TAG}.md", "markdown"),
    ("c3b_smoke_discovery_json", f"docs/resources/c3b_smoke_candidate_discovery_{DATE_TAG}.json", "json"),
    ("c3b_smoke_discovery_md", f"docs/resources/c3b_smoke_candidate_discovery_{DATE_TAG}.md", "markdown"),
    (
        "pynq_c3b_smoke_discovery_json",
        f"docs/resources/pynq_smoke_candidate_discovery_c3b_{CURRENT_AUDIT_DATE_TAG}.json",
        "json",
    ),
    (
        "pynq_c3b_smoke_discovery_md",
        f"docs/resources/pynq_smoke_candidate_discovery_c3b_{CURRENT_AUDIT_DATE_TAG}.md",
        "markdown",
    ),
    (
        "pynq_vref_smoke_discovery_json",
        f"docs/resources/pynq_smoke_candidate_discovery_vref_p0_{CURRENT_AUDIT_DATE_TAG}.json",
        "json",
    ),
    (
        "pynq_vref_smoke_discovery_md",
        f"docs/resources/pynq_smoke_candidate_discovery_vref_p0_{CURRENT_AUDIT_DATE_TAG}.md",
        "markdown",
    ),
    ("final_blocker_closure_json", f"docs/resources/final_blocker_closure_readiness_{DATE_TAG}.json", "json"),
    ("final_blocker_closure_md", f"docs/resources/final_blocker_closure_readiness_{DATE_TAG}.md", "markdown"),
    ("third_goal_source_audit_json", f"docs/resources/third_goal_source_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("third_goal_source_audit_md", f"docs/resources/third_goal_source_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("third_goal_current_audit_json", f"docs/resources/third_goal_current_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("third_goal_current_audit_md", f"docs/resources/third_goal_current_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("req1_environment_json", f"docs/resources/req1_environment_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("req1_environment_md", f"docs/resources/req1_environment_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("req9_deit_image_json", f"docs/resources/req9_deit_image_reference_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("req9_deit_image_md", f"docs/resources/req9_deit_image_reference_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("qkv_uram_successor_json", f"docs/resources/vref_p0_qkv_uram_cache_successor_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("qkv_uram_successor_md", f"docs/resources/vref_p0_qkv_uram_cache_successor_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("vref_p0_pot_scale_audit_json", f"docs/resources/vref_p0_pot_scale_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("vref_p0_pot_scale_audit_md", f"docs/resources/vref_p0_pot_scale_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("vref_p0_pot_scale_sweep_json", f"docs/resources/vref_p0_pot_scale_sweep_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("vref_p0_pot_scale_sweep_md", f"docs/resources/vref_p0_pot_scale_sweep_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("qkv_uram_smoke_discovery_json", f"docs/resources/qkv_uram_smoke_candidate_discovery_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("qkv_uram_smoke_discovery_md", f"docs/resources/qkv_uram_smoke_candidate_discovery_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("req5_q4q8_swhw_json", f"docs/resources/req5_q4q8_swhw_match_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("req5_q4q8_swhw_md", f"docs/resources/req5_q4q8_swhw_match_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("p2_vit_scale_calibration_json", f"docs/resources/p2_vit_scale_calibration_report_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("p2_vit_scale_calibration_md", f"docs/resources/p2_vit_scale_calibration_report_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("req6_parameterization_json", f"docs/resources/req6_parameterization_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("req6_parameterization_md", f"docs/resources/req6_parameterization_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("xr_vits_gate_json", f"docs/resources/xr_vits_gate_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("xr_vits_gate_md", f"docs/resources/xr_vits_gate_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    (
        "c3b_physical_smoke_gate_json",
        f"docs/resources/c3b_physical_smoke_gate_audit_{CURRENT_AUDIT_DATE_TAG}.json",
        "json",
    ),
    (
        "c3b_physical_smoke_gate_md",
        f"docs/resources/c3b_physical_smoke_gate_audit_{CURRENT_AUDIT_DATE_TAG}.md",
        "markdown",
    ),
    ("vref_p0_buffer_lifetime_json", f"docs/resources/vref_p0_buffer_lifetime_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("vref_p0_buffer_lifetime_md", f"docs/resources/vref_p0_buffer_lifetime_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("hgpipe_operator_audit_json", f"docs/resources/hgpipe_operator_audit_{CURRENT_AUDIT_DATE_TAG}.json", "json"),
    ("hgpipe_operator_audit_md", f"docs/resources/hgpipe_operator_audit_{CURRENT_AUDIT_DATE_TAG}.md", "markdown"),
    ("c3b_transfer_manifest_json", f"docs/resources/c3b_smoke_transfer_manifest_{DATE_TAG}.json", "json"),
    ("c3b_transfer_manifest_md", f"docs/resources/c3b_smoke_transfer_manifest_{DATE_TAG}.md", "markdown"),
    ("c3b_bundle_sha256", "docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256", "sha256-text"),
]

OPTIONAL_ARTIFACTS = [
    ("c3b_import_json", f"docs/resources/c3b_smoke_import_{DATE_TAG}.json", "json"),
    ("c3b_import_validation_json", f"docs/resources/c3b_smoke_import_validation_{DATE_TAG}.json", "json"),
]

VOLATILE_ARTIFACTS = [
    (
        "final_signoff_summary_json",
        f"docs/resources/third_goal_final_signoff_run_{DATE_TAG}.json",
        "excluded to avoid self-referential summary hash churn",
    ),
    (
        "final_signoff_summary_md",
        f"docs/resources/third_goal_final_signoff_run_{DATE_TAG}.md",
        "excluded to avoid self-referential summary hash churn",
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def json_status(path: Path) -> tuple[str, str]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return "fail", f"json decode error: {exc}"
    if not isinstance(payload, dict):
        return "fail", "json root is not object"
    return "pass", str(payload.get("status", "no-status-field"))


def build_entry(root: Path, artifact_id: str, rel: str, kind: str, required: bool) -> dict[str, Any]:
    path = root / rel
    entry: dict[str, Any] = {
        "id": artifact_id,
        "path": str(path),
        "relative_path": rel,
        "kind": kind,
        "required": required,
        "exists": path.exists(),
        "source_date_tag": artifact_source_date_tag(rel),
    }
    if not path.exists():
        entry["status"] = "missing" if required else "absent-optional"
        return entry

    entry.update(
        {
            "status": "pass",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
    if kind == "json":
        parse_status, source_status = json_status(path)
        entry["json_parse_status"] = parse_status
        entry["source_status"] = source_status
        if parse_status != "pass":
            entry["status"] = "fail"
    return entry


def artifact_source_date_tag(relative_path: str) -> str:
    if CURRENT_AUDIT_DATE_TAG in relative_path:
        return CURRENT_AUDIT_DATE_TAG
    if DATE_TAG in relative_path:
        return DATE_TAG
    return "undated"


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def resource_policy_artifact_set(root: Path) -> dict[str, Any]:
    expected = {
        f"e2e_resource_policy_audit_{DATE_TAG}.json",
        f"e2e_resource_policy_audit_{DATE_TAG}.md",
    }
    locations = {
        "docs_resources": root / "docs" / "resources",
        "generated_signoff": root / "hardware" / "generated" / "signoff",
    }
    observed: dict[str, list[str]] = {}
    unexpected: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for name, directory in locations.items():
        files = sorted(path.name for path in directory.glob("e2e_resource_policy_audit_*.json"))
        files += sorted(path.name for path in directory.glob("e2e_resource_policy_audit_*.md"))
        observed[name] = files
        unexpected[name] = sorted(set(files) - expected)
        missing[name] = sorted(expected - set(files))
    filename_payload_mismatches: list[dict[str, str]] = []
    for location_name, directory in locations.items():
        for path in sorted(directory.glob("e2e_resource_policy_audit_*.json")):
            payload = load_json_object(path)
            filename_tag = path.name.removeprefix("e2e_resource_policy_audit_").removesuffix(".json")
            payload_tag = str(payload.get("date_tag")) if isinstance(payload, dict) else "<invalid-json>"
            if filename_tag != payload_tag:
                filename_payload_mismatches.append(
                    {
                        "location": location_name,
                        "file": path.name,
                        "filename_date_tag": filename_tag,
                        "payload_date_tag": payload_tag,
                    }
                )
    mirror_hash_mismatches: list[dict[str, str]] = []
    for filename in expected:
        docs_path = locations["docs_resources"] / filename
        generated_path = locations["generated_signoff"] / filename
        if not docs_path.exists() or not generated_path.exists():
            continue
        docs_sha = sha256_file(docs_path)
        generated_sha = sha256_file(generated_path)
        if docs_sha != generated_sha:
            mirror_hash_mismatches.append(
                {
                    "file": filename,
                    "docs_sha256": docs_sha,
                    "generated_sha256": generated_sha,
                }
            )
    canonical_pair_present = all(not values for values in missing.values())
    docs_singularity = missing["docs_resources"] == [] and unexpected["docs_resources"] == []
    generated_singularity = missing["generated_signoff"] == [] and unexpected["generated_signoff"] == []
    filename_payload_date_match = filename_payload_mismatches == []
    canonical_mirror_sha256 = mirror_hash_mismatches == [] and canonical_pair_present
    return {
        "expected": sorted(expected),
        "observed": observed,
        "unexpected": unexpected,
        "missing": missing,
        "filename_payload_mismatches": filename_payload_mismatches,
        "mirror_hash_mismatches": mirror_hash_mismatches,
        "canonical_pair_present": canonical_pair_present,
        "docs_singularity": docs_singularity,
        "generated_singularity": generated_singularity,
        "filename_payload_date_match": filename_payload_date_match,
        "canonical_mirror_sha256": canonical_mirror_sha256,
        "status": "pass"
        if (
            canonical_pair_present
            and docs_singularity
            and generated_singularity
            and filename_payload_date_match
            and canonical_mirror_sha256
        )
        else "fail",
    }


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def int_at_least(value: Any, minimum: int) -> bool:
    try:
        return int(value) >= minimum
    except (TypeError, ValueError):
        return False


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def is_plain_int(value: Any) -> bool:
    return type(value) is int


def add_pynq_discovery_checks(
    checks: list[dict[str, Any]],
    *,
    label: str,
    payload: dict[str, Any] | None,
    expected_preset: str,
) -> None:
    status = payload.get("status") if isinstance(payload, dict) else "missing"
    candidate_count = payload.get("candidate_count") if isinstance(payload, dict) else None
    pass_count = payload.get("pass_count") if isinstance(payload, dict) else None
    recommended = payload.get("recommended_candidate") if isinstance(payload, dict) else "missing"
    safety = payload.get("safety", {}) if isinstance(payload, dict) and isinstance(payload.get("safety"), dict) else {}
    add_check(
        checks,
        f"{label}_status_known",
        isinstance(payload, dict) and status in {"found", "missing"},
        str(status),
    )
    add_check(
        checks,
        f"{label}_preset",
        isinstance(payload, dict) and payload.get("preset") == expected_preset,
        str(payload.get("preset") if isinstance(payload, dict) else "missing"),
    )
    add_check(
        checks,
        f"{label}_candidate_count_numeric",
        is_plain_int(candidate_count),
        str(candidate_count),
    )
    add_check(
        checks,
        f"{label}_pass_count_numeric",
        is_plain_int(pass_count),
        str(pass_count),
    )
    add_check(
        checks,
        f"{label}_pass_count_lte_candidate_count",
        is_plain_int(pass_count) and is_plain_int(candidate_count) and pass_count <= candidate_count,
        f"pass={pass_count} candidates={candidate_count}",
    )
    add_check(
        checks,
        f"{label}_missing_has_no_recommended_candidate",
        status != "missing" or recommended is None,
        str(recommended),
    )
    add_check(
        checks,
        f"{label}_safe_no_side_effects",
        safety.get("executes_commands") is False
        and safety.get("creates_board_result") is False
        and safety.get("creates_xr_vits_policy") is False
        and safety.get("writes_canonical_inputs") is False,
        str(safety),
    )


def add_pynq_discovery_summary_checks(
    checks: list[dict[str, Any]],
    *,
    label: str,
    payload: dict[str, Any] | None,
    expected_preset: str,
) -> None:
    status = payload.get("status") if isinstance(payload, dict) else "missing"
    candidate_count = payload.get("candidate_count") if isinstance(payload, dict) else None
    pass_count = payload.get("pass_count") if isinstance(payload, dict) else None
    recommended_path = payload.get("recommended_candidate_path") if isinstance(payload, dict) else "missing"
    safety = payload.get("safety", {}) if isinstance(payload, dict) and isinstance(payload.get("safety"), dict) else {}
    add_check(
        checks,
        f"{label}_status_known",
        isinstance(payload, dict) and status in {"found", "missing"},
        str(status),
    )
    add_check(
        checks,
        f"{label}_preset",
        isinstance(payload, dict) and payload.get("preset") == expected_preset,
        str(payload.get("preset") if isinstance(payload, dict) else "missing"),
    )
    add_check(
        checks,
        f"{label}_candidate_count_numeric",
        is_plain_int(candidate_count),
        str(candidate_count),
    )
    add_check(
        checks,
        f"{label}_pass_count_numeric",
        is_plain_int(pass_count),
        str(pass_count),
    )
    add_check(
        checks,
        f"{label}_pass_count_lte_candidate_count",
        is_plain_int(pass_count) and is_plain_int(candidate_count) and pass_count <= candidate_count,
        f"pass={pass_count} candidates={candidate_count}",
    )
    add_check(
        checks,
        f"{label}_missing_has_no_recommended_candidate_path",
        status != "missing" or recommended_path == "",
        str(recommended_path),
    )
    add_check(
        checks,
        f"{label}_safe_no_side_effects",
        safety.get("executes_commands") is False
        and safety.get("creates_board_result") is False
        and safety.get("creates_xr_vits_policy") is False
        and safety.get("writes_canonical_inputs") is False,
        str(safety),
    )


def add_final_runner_blocker_summary_checks(
    checks: list[dict[str, Any]],
    *,
    payload: dict[str, Any] | None,
) -> None:
    input_paths = payload.get("remaining_blocker_input_paths") if isinstance(payload, dict) else None
    details = payload.get("remaining_blocker_details") if isinstance(payload, dict) else None
    remaining = payload.get("remaining_blockers") if isinstance(payload, dict) else None
    mirrored = payload.get("mirrored_artifacts") if isinstance(payload, dict) else None
    input_ok = isinstance(input_paths, dict)
    details_ok = isinstance(details, dict)
    remaining_ok = isinstance(remaining, list) and all(isinstance(item, str) and item for item in remaining)
    mirrored_ok = isinstance(mirrored, list) and all(isinstance(item, str) and item for item in mirrored)
    mirrored_unique = mirrored_ok and len(mirrored) == len(set(mirrored))
    blocker_count = payload.get("blocker_count") if isinstance(payload, dict) else None
    detail_count = payload.get("remaining_blocker_detail_count") if isinstance(payload, dict) else None
    mirrored_count = payload.get("mirrored_artifact_count") if isinstance(payload, dict) else None
    mirrored_unique_count = payload.get("mirrored_artifact_unique_count") if isinstance(payload, dict) else None
    mirrored_duplicate_count = payload.get("mirrored_artifact_duplicate_count") if isinstance(payload, dict) else None
    integrity = payload.get("mirrored_artifact_integrity") if isinstance(payload, dict) else None
    integrity_status = payload.get("mirrored_artifact_integrity_status") if isinstance(payload, dict) else None
    integrity_count = payload.get("mirrored_artifact_integrity_count") if isinstance(payload, dict) else None
    integrity_checked_count = payload.get("mirrored_artifact_integrity_checked_count") if isinstance(payload, dict) else None
    integrity_excluded_count = payload.get("mirrored_artifact_integrity_excluded_count") if isinstance(payload, dict) else None
    integrity_fail_count = payload.get("mirrored_artifact_integrity_fail_count") if isinstance(payload, dict) else None
    integrity_failures = payload.get("mirrored_artifact_integrity_failures") if isinstance(payload, dict) else None
    integrity_ok = isinstance(integrity, list) and all(isinstance(item, dict) for item in integrity)
    keys_match = set(input_paths) == set(details) == set(remaining) if input_ok and details_ok and remaining_ok else False
    values_non_empty = (
        input_ok
        and all(isinstance(value, list) and len(value) > 0 and all(str(item) for item in value) for value in input_paths.values())
    )
    add_check(
        checks,
        "final_runner_remaining_blockers_present",
        remaining_ok,
        str(remaining if remaining_ok else "missing"),
    )
    add_check(
        checks,
        "final_runner_blocker_count_matches_list",
        remaining_ok and blocker_count == len(remaining),
        f"blocker_count={blocker_count} list={len(remaining) if remaining_ok else 'missing'}",
    )
    add_check(
        checks,
        "final_runner_remaining_blocker_detail_count_matches",
        details_ok and detail_count == len(details),
        f"detail_count={detail_count} details={len(details) if details_ok else 'missing'}",
    )
    add_check(
        checks,
        "final_runner_remaining_blocker_input_paths_present",
        input_ok,
        str(input_paths if input_ok else "missing"),
    )
    add_check(
        checks,
        "final_runner_remaining_blocker_details_present",
        details_ok,
        str(details if details_ok else "missing"),
    )
    add_check(
        checks,
        "final_runner_remaining_blocker_keys_align",
        keys_match,
        str(sorted((set(input_paths) if input_ok else set()) ^ (set(details) if details_ok else set()))),
    )
    add_check(
        checks,
        "final_runner_remaining_blocker_input_paths_non_empty",
        values_non_empty,
        "all remaining blocker path lists are non-empty",
    )
    add_check(
        checks,
        "final_runner_mirrored_artifacts_present",
        mirrored_ok,
        str(len(mirrored) if isinstance(mirrored, list) else "missing"),
    )
    add_check(
        checks,
        "final_runner_mirrored_artifacts_unique",
        mirrored_unique,
        f"total={len(mirrored) if isinstance(mirrored, list) else 'missing'} "
        f"unique={len(set(mirrored)) if mirrored_ok else 'missing'}",
    )
    add_check(
        checks,
        "final_runner_mirrored_artifact_counts_match",
        mirrored_ok
        and mirrored_count == len(mirrored)
        and mirrored_unique_count == len(set(mirrored))
        and mirrored_duplicate_count == len(mirrored) - len(set(mirrored)),
        f"fields=({mirrored_count},{mirrored_unique_count},{mirrored_duplicate_count}) "
        f"actual=({len(mirrored) if mirrored_ok else 'missing'},"
        f"{len(set(mirrored)) if mirrored_ok else 'missing'},"
        f"{(len(mirrored) - len(set(mirrored))) if mirrored_ok else 'missing'})",
    )
    add_check(
        checks,
        "final_runner_mirrored_artifact_integrity_present",
        integrity_ok,
        f"count={len(integrity) if isinstance(integrity, list) else 'missing'}",
    )
    add_check(
        checks,
        "final_runner_mirrored_artifact_integrity_pass",
        integrity_status == "pass" and integrity_fail_count == 0 and integrity_failures == [],
        f"status={integrity_status} fail_count={integrity_fail_count}",
    )
    add_check(
        checks,
        "final_runner_mirrored_artifact_integrity_counts_match",
        mirrored_ok
        and integrity_ok
        and integrity_count == len(mirrored)
        and integrity_count == len(integrity)
        and isinstance(integrity_checked_count, int)
        and isinstance(integrity_excluded_count, int)
        and integrity_checked_count + integrity_excluded_count == integrity_count,
        (
            f"mirrored={len(mirrored) if mirrored_ok else 'missing'} integrity={integrity_count} "
            f"checked={integrity_checked_count} excluded={integrity_excluded_count}"
        ),
    )


def named_checks(payload: dict[str, Any] | None, section: str | None = None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    source: Any = payload.get(section, {}) if section else payload
    if not isinstance(source, dict):
        return {}
    checks = source.get("checks", [])
    if not isinstance(checks, list):
        return {}
    return {
        str(check.get("name")): check
        for check in checks
        if isinstance(check, dict) and check.get("name")
    }


def build_consistency_checks(root: Path) -> list[dict[str, Any]]:
    resources = root / "docs" / "resources"
    checks: list[dict[str, Any]] = []

    xr_resolution = load_json_object(resources / f"xr_vits_reference_resolution_{DATE_TAG}.json")
    c3b_readiness = load_json_object(resources / f"c3b_board_smoke_readiness_{DATE_TAG}.json")
    blocker_closure = load_json_object(resources / f"final_blocker_closure_readiness_{DATE_TAG}.json")
    c3b_discovery = load_json_object(resources / f"c3b_smoke_candidate_discovery_{DATE_TAG}.json")
    pynq_c3b_discovery = load_json_object(
        resources / f"pynq_smoke_candidate_discovery_c3b_{CURRENT_AUDIT_DATE_TAG}.json"
    )
    pynq_vref_discovery = load_json_object(
        resources / f"pynq_smoke_candidate_discovery_vref_p0_{CURRENT_AUDIT_DATE_TAG}.json"
    )
    qkv_uram_discovery = load_json_object(
        resources / f"qkv_uram_smoke_candidate_discovery_{CURRENT_AUDIT_DATE_TAG}.json"
    )
    c3b_contract = load_json_object(resources / f"c3b_smoke_result_contract_{DATE_TAG}.json")
    xr_policy_preview = load_json_object(resources / f"xr_vits_replacement_policy_preview_{DATE_TAG}.json")
    unblock_intake = load_json_object(resources / f"final_unblock_intake_{DATE_TAG}.json")
    closeout_packet = load_json_object(resources / f"final_unblock_closeout_packet_{DATE_TAG}.json")
    closeout_validation = load_json_object(resources / f"final_unblock_closeout_packet_validation_{DATE_TAG}.json")
    operator_handoff_validation = load_json_object(resources / f"final_operator_handoff_validation_{DATE_TAG}.json")
    final_bundle_validation = load_json_object(resources / f"final_signoff_bundle_validation_{DATE_TAG}.json")
    selected_path = load_json_object(resources / f"selected_path_execution_audit_{DATE_TAG}.json")
    resource_policy = load_json_object(resources / f"e2e_resource_policy_audit_{DATE_TAG}.json")
    completion_audit = load_json_object(resources / f"third_goal_completion_audit_{DATE_TAG}.json")
    source_audit = load_json_object(resources / f"third_goal_source_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    current_audit = load_json_object(resources / f"third_goal_current_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    req1_environment = load_json_object(resources / f"req1_environment_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    req9_deit_image = load_json_object(resources / f"req9_deit_image_reference_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    requirements_trace = load_json_object(resources / f"third_goal_requirements_trace_{DATE_TAG}.json")
    spec_plan = load_json_object(resources / f"spec_plan_conformance_audit_{DATE_TAG}.json")
    qkv_uram = load_json_object(resources / f"vref_p0_qkv_uram_cache_successor_{CURRENT_AUDIT_DATE_TAG}.json")
    pot_scale_audit = load_json_object(resources / f"vref_p0_pot_scale_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    pot_scale_sweep = load_json_object(resources / f"vref_p0_pot_scale_sweep_{CURRENT_AUDIT_DATE_TAG}.json")
    req5_q4q8 = load_json_object(resources / f"req5_q4q8_swhw_match_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    p2_scale_calibration = load_json_object(resources / f"p2_vit_scale_calibration_report_{CURRENT_AUDIT_DATE_TAG}.json")
    req6_parameterization = load_json_object(resources / f"req6_parameterization_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    xr_vits_gate = load_json_object(resources / f"xr_vits_gate_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    c3b_physical_gate = load_json_object(resources / f"c3b_physical_smoke_gate_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    c3b_transfer_manifest = load_json_object(resources / f"c3b_smoke_transfer_manifest_{DATE_TAG}.json")
    c3b_bundle_sha256_text = read_text_or_empty(resources / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256").strip()
    vref_buffer_lifetime = load_json_object(resources / f"vref_p0_buffer_lifetime_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    hgpipe_operator = load_json_object(resources / f"hgpipe_operator_audit_{CURRENT_AUDIT_DATE_TAG}.json")
    final_runner_summary = load_json_object(
        root / "hardware" / "generated" / "signoff" / f"third_goal_final_signoff_run_{DATE_TAG}.json"
    )
    xr_policy_tool_text = read_text_or_empty(root / "hardware" / "tools" / "create_xr_vits_replacement_policy.py")
    pynq_validator_text = read_text_or_empty(root / "hardware" / "tools" / "validate_pynq_smoke_result.py")
    final_runner_text = read_text_or_empty(root / "hardware" / "tools" / "run_third_goal_final_signoff.py")
    spec_plan_checks = named_checks(spec_plan)

    expected_overlay_prefixes = [
        "hgtxr_e2e_axis_dma_c3b_mem16",
        "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream",
        "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram",
        "hgtxr_e2e_m_axi",
    ]
    add_check(
        checks,
        "pynq_smoke_validator_overlay_prefix_contracts",
        all(prefix in pynq_validator_text for prefix in expected_overlay_prefixes),
        "required overlay prefixes present in validate_pynq_smoke_result.py",
    )
    add_check(
        checks,
        "pynq_smoke_validator_basename_check",
        "Path(str(value)).name" in pynq_validator_text and "expected_artifact_name" in pynq_validator_text,
        "validator compares reported bit/hwh basename to expected artifact name",
    )
    add_check(
        checks,
        "pynq_smoke_validator_require_paths_gate",
        "if require_paths:" in pynq_validator_text and "artifact_prefix" in pynq_validator_text,
        "overlay basename gate is tied to require_paths",
    )
    add_check(
        checks,
        "final_runner_remaining_blocker_input_paths_contract",
        "remaining_blocker_input_paths" in final_runner_text
        and "build_remaining_blocker_details" in final_runner_text,
        "runner summary exposes exact blocker input paths",
    )
    add_check(
        checks,
        "final_runner_remaining_blocker_details_contract",
        "remaining_blocker_details" in final_runner_text
        and "blocker_count" in final_runner_text
        and "remaining_blocker_detail_count" in final_runner_text
        and "exact-source-or-approved-replacement-policy" in final_runner_text
        and "canonical-pynq-smoke-json" in final_runner_text,
        "runner summary exposes structured blocker details and blocker counts",
    )
    add_check(
        checks,
        "final_runner_mirrored_artifacts_dedupe_contract",
        "summary[\"mirrored_artifacts\"].extend" in final_runner_text
        and "mirrored_artifact_count" in final_runner_text
        and "mirrored_artifact_unique_count" in final_runner_text
        and "mirrored_artifact_duplicate_count" in final_runner_text
        and "doc_summary_json" in final_runner_text
        and "doc_summary_md" in final_runner_text
        and "summary[\"mirrored_artifacts\"] = unique_ordered(summary[\"mirrored_artifacts\"])" in final_runner_text,
        "runner appends summary mirrors, dedupes, and exposes mirror counts",
    )
    add_check(
        checks,
        "final_runner_mirrored_artifact_integrity_contract",
        "def build_mirror_integrity" in final_runner_text
        and "canonical_evidence_root" in final_runner_text
        and "generated_signoff_root" in final_runner_text
        and "mirrored_artifact_integrity_status" in final_runner_text
        and "mirrored_artifact_integrity_checked_count" in final_runner_text
        and "sha256_match" in final_runner_text,
        "runner records generated/signoff to docs/resources mirror integrity",
    )
    add_check(
        checks,
        "xr_vits_policy_tool_placeholder_guard_contract",
        "PLACEHOLDER_APPROVER_VALUES" in xr_policy_tool_text
        and "def approver_is_placeholder" in xr_policy_tool_text
        and "--approved-by must be a real approver identifier for active policy writes" in xr_policy_tool_text,
        "policy tool defines placeholder approver rejection for active writes",
    )
    add_check(
        checks,
        "xr_vits_policy_tool_dry_run_placeholder_allowed",
        "not dry_run and approver_is_placeholder(approved_by)" in xr_policy_tool_text
        and "dry_run=args.dry_run" in xr_policy_tool_text,
        "placeholder approver rejection is gated by non-dry-run active writes",
    )
    add_check(
        checks,
        "final_runner_active_policy_failure_blocks",
        "xr_vits_policy_status == \"fail\"" in final_runner_text
        and "not dry_run_xr_vits_replacement" in final_runner_text,
        "final runner blocks when active replacement-policy creation fails",
    )
    add_final_runner_blocker_summary_checks(checks, payload=final_runner_summary)
    add_check(
        checks,
        "final_runner_c3b_transfer_manifest_status_pass",
        isinstance(final_runner_summary, dict)
        and final_runner_summary.get("c3b_transfer_manifest_status") == "pass",
        str(final_runner_summary.get("c3b_transfer_manifest_status") if isinstance(final_runner_summary, dict) else "missing"),
    )
    add_check(
        checks,
        "spec_plan_conformance_pass",
        isinstance(spec_plan, dict)
        and isinstance(spec_plan.get("checks"), list)
        and int_at_least(spec_plan.get("check_count"), 86),
        f"status={spec_plan.get('status') if isinstance(spec_plan, dict) else 'missing'} fail={spec_plan.get('fail_count') if isinstance(spec_plan, dict) else 'missing'}",
    )
    add_check(
        checks,
        "spec_plan_manual_fallback_contract",
        spec_plan_checks.get("spec_records_spec_kit_unavailable", {}).get("status") == "pass",
        str(spec_plan_checks.get("spec_records_spec_kit_unavailable", {}).get("detail", "missing")),
    )
    add_check(
        checks,
        "spec_plan_zcu104_q4wq8a_contract",
        spec_plan_checks.get("spec_records_zcu104", {}).get("status") == "pass"
        and spec_plan_checks.get("spec_records_q4w_q8a", {}).get("status") == "pass",
        f"zcu104={spec_plan_checks.get('spec_records_zcu104', {}).get('status', 'missing')} q4q8={spec_plan_checks.get('spec_records_q4w_q8a', {}).get('status', 'missing')}",
    )
    add_check(
        checks,
        "spec_plan_param_knobs_contract",
        spec_plan_checks.get("spec_records_param_knobs", {}).get("status") == "pass",
        str(spec_plan_checks.get("spec_records_param_knobs", {}).get("detail", "missing")),
    )
    add_check(
        checks,
        "spec_plan_selected_paths_contract",
        spec_plan_checks.get("spec_records_a2_a1_c", {}).get("status") == "pass"
        and spec_plan_checks.get("master_records_selected_paths", {}).get("status") == "pass"
        and spec_plan_checks.get("choice_records_e_pending", {}).get("status") == "pass",
        (
            f"spec={spec_plan_checks.get('spec_records_a2_a1_c', {}).get('status', 'missing')} "
            f"master={spec_plan_checks.get('master_records_selected_paths', {}).get('status', 'missing')} "
            f"choice={spec_plan_checks.get('choice_records_e_pending', {}).get('status', 'missing')}"
        ),
    )
    add_check(
        checks,
        "spec_plan_xilinx_root_contract",
        spec_plan_checks.get("execution_records_xilinx_root", {}).get("status") == "pass",
        str(spec_plan_checks.get("execution_records_xilinx_root", {}).get("detail", "missing")),
    )
    current_doc_freshness_names = [
        f"{doc}_{suffix}"
        for doc in ["master_plan", "sub_plan", "spec", "validation", "progress", "current_handover", "choice", "log"]
        for suffix in [
            "records_live_manifest_required",
            "records_live_manifest_consistency",
            "records_live_source_count",
            "records_live_current_summary",
            "records_live_operator_handoff_validation",
            "records_live_bundle_validation",
        ]
    ]
    current_doc_freshness_names.extend(
        [
            "current_handover_records_live_operator_handoff_validation_artifact_summary",
            "current_handover_records_live_bundle_validation_artifact_summary",
        ]
    )
    add_check(
        checks,
        "spec_plan_current_doc_freshness_contract",
        all(spec_plan_checks.get(name, {}).get("status") == "pass" for name in current_doc_freshness_names),
        ",".join(
            name
            for name in current_doc_freshness_names
            if spec_plan_checks.get(name, {}).get("status") != "pass"
        )
        or "all-current-docs-fresh",
    )

    add_check(
        checks,
        "xr_vits_resolution_status_known",
        isinstance(xr_resolution, dict)
        and xr_resolution.get("status")
        in {"exact-ready", "approved-replacement-ready", "candidate-ready-needs-approval", "blocked"},
        str(xr_resolution.get("status") if isinstance(xr_resolution, dict) else "missing"),
    )
    xr_safety = xr_resolution.get("safety", {}) if isinstance(xr_resolution, dict) and isinstance(xr_resolution.get("safety"), dict) else {}
    add_check(
        checks,
        "xr_vits_resolution_no_policy_write",
        xr_safety.get("creates_xr_vits_policy") is False,
        str(xr_safety.get("creates_xr_vits_policy")),
    )
    add_check(
        checks,
        "xr_vits_resolution_no_canonical_write",
        xr_safety.get("writes_canonical_inputs") is False,
        str(xr_safety.get("writes_canonical_inputs")),
    )
    add_check(
        checks,
        "final_blocker_closure_status_known",
        isinstance(blocker_closure, dict)
        and blocker_closure.get("status") in {"ready-to-run-final", "would-clear-with-candidates", "blocked"},
        str(blocker_closure.get("status") if isinstance(blocker_closure, dict) else "missing"),
    )
    blocker_current = (
        blocker_closure.get("current", {})
        if isinstance(blocker_closure, dict) and isinstance(blocker_closure.get("current"), dict)
        else {}
    )
    blocker_c3b = blocker_current.get("c3b_smoke", {}) if isinstance(blocker_current.get("c3b_smoke"), dict) else {}
    blocker_xr = blocker_current.get("xr_vits", {}) if isinstance(blocker_current.get("xr_vits"), dict) else {}
    blocker_xr_exact = blocker_xr.get("exact", {}) if isinstance(blocker_xr.get("exact"), dict) else {}
    blocker_xr_policy = (
        blocker_xr.get("replacement_policy", {}) if isinstance(blocker_xr.get("replacement_policy"), dict) else {}
    )
    blocker_safety = (
        blocker_closure.get("safety", {})
        if isinstance(blocker_closure, dict) and isinstance(blocker_closure.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "final_blocker_closure_c3b_canonical_path",
        str(blocker_c3b.get("path", "")).endswith(
            "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
        ),
        str(blocker_c3b.get("path", "missing")),
    )
    add_check(
        checks,
        "final_blocker_closure_xr_vits_exact_path",
        str(blocker_xr_exact.get("path", "")) == str(root.parent.parent / "XR-VITs"),
        str(blocker_xr_exact.get("path", "missing")),
    )
    add_check(
        checks,
        "final_blocker_closure_xr_vits_policy_path",
        str(blocker_xr_policy.get("path", "")) == str(root / "docs/resources/xr_vits_replacement_policy.json"),
        str(blocker_xr_policy.get("path", "missing")),
    )
    add_check(
        checks,
        "final_blocker_closure_discovery_only_safety",
        blocker_safety.get("executes_commands") is False
        and blocker_safety.get("creates_board_result") is False
        and blocker_safety.get("creates_xr_vits_policy") is False
        and blocker_safety.get("writes_canonical_inputs") is False
        and blocker_safety.get("discovers_pynq_candidates_only") is True,
        str(blocker_safety),
    )
    operator_plan = (
        blocker_closure.get("operator_unblock_plan", {})
        if isinstance(blocker_closure, dict) and isinstance(blocker_closure.get("operator_unblock_plan"), dict)
        else {}
    )
    plan_inputs = (
        operator_plan.get("required_inputs", [])
        if isinstance(operator_plan.get("required_inputs"), list)
        else []
    )
    plan_input_by_blocker = {
        str(item.get("blocker", "")): item for item in plan_inputs if isinstance(item, dict)
    }
    plan_dry_run = (
        operator_plan.get("dry_run_sequence", [])
        if isinstance(operator_plan.get("dry_run_sequence"), list)
        else []
    )
    plan_dry_run_by_step = {
        str(item.get("step", "")): item for item in plan_dry_run if isinstance(item, dict)
    }
    plan_safety = (
        operator_plan.get("safety", {})
        if isinstance(operator_plan.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "final_blocker_closure_operator_plan_status_known",
        operator_plan.get("status") in {"ready-to-run-final", "ready-with-candidates", "needs-external-inputs"},
        str(operator_plan.get("status", "missing")),
    )
    add_check(
        checks,
        "final_blocker_closure_operator_plan_required_inputs",
        {"C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"}.issubset(
            set(plan_input_by_blocker)
        ),
        str(sorted(plan_input_by_blocker)),
    )
    add_check(
        checks,
        "final_blocker_closure_operator_plan_c3b_contract",
        str(
            plan_input_by_blocker.get("C3b AXIS/DMA physical smoke result", {}).get("canonical_path", "")
        ).endswith("hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json")
        and "axis-c3b-mem16"
        in ";".join(
            str(item)
            for item in plan_input_by_blocker.get("C3b AXIS/DMA physical smoke result", {}).get(
                "acceptance", []
            )
        ),
        str(plan_input_by_blocker.get("C3b AXIS/DMA physical smoke result", {})),
    )
    add_check(
        checks,
        "final_blocker_closure_operator_plan_xr_contract",
        str(plan_input_by_blocker.get("requested XR-VITs sibling", {}).get("exact_path", ""))
        == str(root.parent.parent / "XR-VITs")
        and str(plan_input_by_blocker.get("requested XR-VITs sibling", {}).get("policy_path", ""))
        == str(root / "docs/resources/xr_vits_replacement_policy.json")
        and "policy_fingerprint"
        in ";".join(
            str(item)
            for item in plan_input_by_blocker.get("requested XR-VITs sibling", {}).get("acceptance", [])
        ),
        str(plan_input_by_blocker.get("requested XR-VITs sibling", {})),
    )
    add_check(
        checks,
        "final_blocker_closure_operator_plan_dry_run_templates",
        "--dry-run-import-c3b-smoke"
        in str(plan_dry_run_by_step.get("c3b-import-dry-run", {}).get("command", ""))
        and "--dry-run-xr-vits-replacement"
        in str(plan_dry_run_by_step.get("xr-vits-policy-preview", {}).get("command", ""))
        and plan_dry_run_by_step.get("c3b-import-dry-run", {}).get("side_effects") is False
        and plan_dry_run_by_step.get("xr-vits-policy-preview", {}).get("side_effects") is False,
        str(plan_dry_run_by_step),
    )
    add_check(
        checks,
        "final_blocker_closure_operator_plan_no_side_effects",
        plan_safety.get("executes_commands") is False
        and plan_safety.get("creates_board_result") is False
        and plan_safety.get("creates_xr_vits_policy") is False
        and plan_safety.get("writes_canonical_inputs") is False,
        str(plan_safety),
    )
    blocker_discovery = (
        blocker_closure.get("pynq_discovery", {})
        if isinstance(blocker_closure, dict) and isinstance(blocker_closure.get("pynq_discovery"), dict)
        else {}
    )
    add_pynq_discovery_summary_checks(
        checks,
        label="final_blocker_closure_pynq_c3b_discovery",
        payload=blocker_discovery.get("c3b") if isinstance(blocker_discovery, dict) else None,
        expected_preset="axis-c3b-mem16",
    )
    add_pynq_discovery_summary_checks(
        checks,
        label="final_blocker_closure_pynq_vref_discovery",
        payload=blocker_discovery.get("vref_p0") if isinstance(blocker_discovery, dict) else None,
        expected_preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
    )
    add_pynq_discovery_summary_checks(
        checks,
        label="final_blocker_closure_qkv_uram_discovery",
        payload=blocker_discovery.get("qkv_uram") if isinstance(blocker_discovery, dict) else None,
        expected_preset="axis-vref-p0-softmax-input-x2-qkv-uram",
    )
    add_check(
        checks,
        "c3b_smoke_discovery_status_known",
        isinstance(c3b_discovery, dict) and c3b_discovery.get("status") in {"found", "missing"},
        str(c3b_discovery.get("status") if isinstance(c3b_discovery, dict) else "missing"),
    )
    add_check(
        checks,
        "c3b_smoke_discovery_pass_count_numeric",
        isinstance(c3b_discovery, dict) and isinstance(c3b_discovery.get("pass_count"), int),
        str(c3b_discovery.get("pass_count") if isinstance(c3b_discovery, dict) else "missing"),
    )
    add_pynq_discovery_checks(
        checks,
        label="pynq_c3b_smoke_discovery",
        payload=pynq_c3b_discovery,
        expected_preset="axis-c3b-mem16",
    )
    add_pynq_discovery_checks(
        checks,
        label="pynq_vref_smoke_discovery",
        payload=pynq_vref_discovery,
        expected_preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
    )
    add_pynq_discovery_checks(
        checks,
        label="qkv_uram_smoke_discovery",
        payload=qkv_uram_discovery,
        expected_preset="axis-vref-p0-softmax-input-x2-qkv-uram",
    )
    add_check(
        checks,
        "c3b_smoke_contract_pass",
        isinstance(c3b_contract, dict) and c3b_contract.get("status") == "pass",
        str(c3b_contract.get("status") if isinstance(c3b_contract, dict) else "missing"),
    )
    add_check(
        checks,
        "c3b_smoke_contract_preset",
        isinstance(c3b_contract, dict) and c3b_contract.get("preset") == "axis-c3b-mem16",
        str(c3b_contract.get("preset") if isinstance(c3b_contract, dict) else "missing"),
    )
    c3b_contract_safety = (
        c3b_contract.get("safety", {})
        if isinstance(c3b_contract, dict) and isinstance(c3b_contract.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "c3b_smoke_contract_no_side_effects",
        c3b_contract_safety.get("executes_commands") is False
        and c3b_contract_safety.get("creates_board_result") is False
        and c3b_contract_safety.get("creates_xr_vits_policy") is False
        and c3b_contract_safety.get("writes_canonical_inputs") is False,
        str(c3b_contract_safety),
    )
    c3b_transfer_tar = (
        c3b_transfer_manifest.get("tar", {})
        if isinstance(c3b_transfer_manifest, dict) and isinstance(c3b_transfer_manifest.get("tar"), dict)
        else {}
    )
    c3b_readiness_tar = (
        c3b_readiness.get("tar", {})
        if isinstance(c3b_readiness, dict) and isinstance(c3b_readiness.get("tar"), dict)
        else {}
    )
    c3b_transfer_copyback = (
        c3b_transfer_manifest.get("host_copyback", {})
        if isinstance(c3b_transfer_manifest, dict) and isinstance(c3b_transfer_manifest.get("host_copyback"), dict)
        else {}
    )
    c3b_transfer_expected = (
        c3b_transfer_manifest.get("board_expected_outputs", {})
        if isinstance(c3b_transfer_manifest, dict)
        and isinstance(c3b_transfer_manifest.get("board_expected_outputs"), dict)
        else {}
    )
    c3b_transfer_files = (
        c3b_transfer_manifest.get("transfer_files", [])
        if isinstance(c3b_transfer_manifest, dict)
        else []
    )
    c3b_transfer_verify = (
        c3b_transfer_manifest.get("board_verify_commands", [])
        if isinstance(c3b_transfer_manifest, dict)
        else []
    )
    c3b_transfer_run = (
        c3b_transfer_manifest.get("board_run_commands", [])
        if isinstance(c3b_transfer_manifest, dict)
        else []
    )
    c3b_transfer_errors = (
        c3b_transfer_manifest.get("bundle_validation_errors", [])
        if isinstance(c3b_transfer_manifest, dict)
        else []
    )
    c3b_transfer_expected_sha_line = c3b_transfer_tar.get("sha256_line")
    c3b_transfer_tar_name = c3b_transfer_tar.get("name")
    c3b_transfer_tar_sha = c3b_transfer_tar.get("sha256")
    add_check(
        checks,
        "c3b_transfer_manifest_loaded_and_typed",
        isinstance(c3b_transfer_manifest, dict),
        str(type(c3b_transfer_manifest).__name__),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_pass",
        isinstance(c3b_transfer_manifest, dict) and c3b_transfer_manifest.get("status") == "pass",
        str(c3b_transfer_manifest.get("status") if isinstance(c3b_transfer_manifest, dict) else "missing"),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_preset_variant",
        isinstance(c3b_transfer_manifest, dict)
        and c3b_transfer_manifest.get("preset") == "axis-c3b-mem16"
        and c3b_transfer_manifest.get("variant") == "c3b-mem16"
        and c3b_transfer_manifest.get("target") == "ZCU104 PYNQ",
        f"preset={c3b_transfer_manifest.get('preset') if isinstance(c3b_transfer_manifest, dict) else 'missing'} variant={c3b_transfer_manifest.get('variant') if isinstance(c3b_transfer_manifest, dict) else 'missing'} target={c3b_transfer_manifest.get('target') if isinstance(c3b_transfer_manifest, dict) else 'missing'}",
    )
    add_check(
        checks,
        "c3b_transfer_manifest_tar_shape",
        isinstance(c3b_transfer_tar.get("path"), str)
        and str(c3b_transfer_tar.get("path")).endswith("e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz")
        and c3b_transfer_tar_name == "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
        and int_at_least(c3b_transfer_tar.get("bytes"), 1)
        and is_sha256_hex(c3b_transfer_tar_sha)
        and c3b_transfer_tar.get("sha256_file") == "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
        and c3b_transfer_expected_sha_line == f"{c3b_transfer_tar_sha}  {c3b_transfer_tar_name}",
        str(c3b_transfer_tar),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_files_contract",
        isinstance(c3b_transfer_files, list)
        and any("e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz" in str(path) for path in c3b_transfer_files)
        and any("e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256" in str(path) for path in c3b_transfer_files),
        str(c3b_transfer_files),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_bundle_validation_clean",
        isinstance(c3b_transfer_errors, list) and c3b_transfer_errors == [],
        str(c3b_transfer_errors),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_sha256_line_matches",
        isinstance(c3b_transfer_expected_sha_line, str)
        and c3b_transfer_expected_sha_line == c3b_bundle_sha256_text
        and is_sha256_hex(c3b_transfer_tar_sha)
        and c3b_transfer_tar_sha in c3b_bundle_sha256_text,
        f"manifest={c3b_transfer_expected_sha_line} sha256_file={c3b_bundle_sha256_text}",
    )
    add_check(
        checks,
        "c3b_transfer_manifest_expected_outputs_contract",
        c3b_transfer_expected.get("expected_runtime_state") == 2
        and c3b_transfer_expected.get("expected_out_raw") == [32, -13, 26, -6, 14, -11]
        and c3b_transfer_expected.get("result_json") == "e2e_axis_dma_c3b_mem16_file_smoke.json"
        and c3b_transfer_expected.get("validation_json") == "e2e_axis_dma_c3b_mem16_file_smoke_validation.json",
        str(c3b_transfer_expected),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_board_verify_command",
        isinstance(c3b_transfer_verify, list)
        and any("sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256" in str(command) for command in c3b_transfer_verify),
        str(c3b_transfer_verify),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_board_run_commands",
        isinstance(c3b_transfer_run, list)
        and any("./run_e2e_axis_dma_c3b_mem16_file_smoke.sh" in str(command) for command in c3b_transfer_run)
        and any("./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh" in str(command) for command in c3b_transfer_run),
        str(c3b_transfer_run),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_copyback_contract",
        str(c3b_transfer_copyback.get("canonical_result_path", "")).endswith(
            "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
        )
        and "tools/import_pynq_smoke_result.py" in str(c3b_transfer_copyback.get("import_command", ""))
        and "--preset axis-c3b-mem16" in str(c3b_transfer_copyback.get("import_command", ""))
        and "tools/validate_pynq_smoke_result.py" in str(c3b_transfer_copyback.get("validate_command", ""))
        and "--preset axis-c3b-mem16" in str(c3b_transfer_copyback.get("validate_command", ""))
        and "final-signoff" in str(c3b_transfer_copyback.get("final_signoff_command", "")),
        str(c3b_transfer_copyback),
    )
    add_check(
        checks,
        "c3b_transfer_manifest_sha256_file_format",
        len(c3b_bundle_sha256_text.split("  ")) == 2
        and is_sha256_hex(c3b_bundle_sha256_text.split("  ")[0])
        and c3b_bundle_sha256_text.split("  ")[1] == "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
        c3b_bundle_sha256_text,
    )
    add_check(
        checks,
        "c3b_transfer_manifest_sha_matches_readiness",
        is_sha256_hex(c3b_transfer_tar_sha)
        and c3b_transfer_tar_sha == c3b_readiness_tar.get("sha256")
        and c3b_transfer_tar_sha in c3b_bundle_sha256_text,
        f"transfer={c3b_transfer_tar_sha} readiness={c3b_readiness_tar.get('sha256')} sha_file={c3b_bundle_sha256_text}",
    )
    add_check(
        checks,
        "c3b_transfer_manifest_bundle_to_readiness_alignment",
        c3b_transfer_tar.get("name") == c3b_readiness_tar.get("name")
        and c3b_transfer_tar.get("path") == c3b_readiness_tar.get("path")
        and c3b_transfer_tar.get("sha256_line") == c3b_readiness_tar.get("sha256_line"),
        f"transfer={c3b_transfer_tar} readiness={c3b_readiness_tar}",
    )
    preview_policy = xr_policy_preview.get("policy", {}) if isinstance(xr_policy_preview, dict) and isinstance(xr_policy_preview.get("policy"), dict) else {}
    preview_integrity = (
        xr_policy_preview.get("integrity", {})
        if isinstance(xr_policy_preview, dict) and isinstance(xr_policy_preview.get("integrity"), dict)
        else {}
    )
    preview_safety = (
        xr_policy_preview.get("safety", {})
        if isinstance(xr_policy_preview, dict) and isinstance(xr_policy_preview.get("safety"), dict)
        else {}
    )
    preview_approval_event = (
        preview_policy.get("approval_event", {})
        if isinstance(preview_policy, dict) and isinstance(preview_policy.get("approval_event"), dict)
        else {}
    )
    add_check(
        checks,
        "xr_vits_policy_preview_pass",
        isinstance(xr_policy_preview, dict) and xr_policy_preview.get("status") == "pass",
        str(xr_policy_preview.get("status") if isinstance(xr_policy_preview, dict) else "missing"),
    )
    add_check(
        checks,
        "xr_vits_policy_preview_only",
        isinstance(xr_policy_preview, dict)
        and xr_policy_preview.get("preview_only") is True
        and xr_policy_preview.get("active_policy_written") is False,
        f"preview_only={xr_policy_preview.get('preview_only') if isinstance(xr_policy_preview, dict) else 'missing'} active_policy_written={xr_policy_preview.get('active_policy_written') if isinstance(xr_policy_preview, dict) else 'missing'}",
    )
    add_check(
        checks,
        "xr_vits_policy_preview_integrity_pass",
        preview_integrity.get("status") == "pass",
        str(preview_integrity.get("status", "missing")),
    )
    add_check(
        checks,
        "xr_vits_policy_preview_required_fields",
        all(
            preview_policy.get(field)
            for field in [
                "candidate_audit_fingerprint",
                "candidate_audit_recommendation_snapshot",
                "candidate_audit_meta",
                "approval_event",
                "policy_fingerprint",
            ]
        ),
        "fingerprint-bound preview fields present",
    )
    required_approval_event_fields = {
        "event_type",
        "protocol_version",
        "approver_id",
        "approved_at",
        "reason_code",
        "reason",
        "replacement_role",
        "requested_path",
        "replacement_path",
        "candidate_audit",
        "candidate_audit_fingerprint",
        "generator",
        "event_id",
    }
    add_check(
        checks,
        "xr_vits_policy_preview_approval_event_schema",
        required_approval_event_fields.issubset(set(preview_approval_event))
        and preview_approval_event.get("event_type") == "xr_vits_replacement_approval"
        and preview_approval_event.get("protocol_version") == "hgtxr-xr-vits-replacement-v1"
        and preview_approval_event.get("reason_code") == "xr_vits_replacement"
        and str(preview_approval_event.get("approved_at", "")).endswith("Z"),
        str(preview_approval_event),
    )
    add_check(
        checks,
        "xr_vits_policy_preview_approval_event_matches_policy",
        preview_approval_event.get("approver_id") == preview_policy.get("approved_by")
        and preview_approval_event.get("approved_at") == preview_policy.get("approved_at")
        and preview_approval_event.get("reason") == preview_policy.get("reason")
        and preview_approval_event.get("replacement_role") == preview_policy.get("replacement_role")
        and preview_approval_event.get("requested_path") == preview_policy.get("requested_path")
        and preview_approval_event.get("replacement_path") == preview_policy.get("replacement_path")
        and preview_approval_event.get("candidate_audit") == preview_policy.get("candidate_audit")
        and preview_approval_event.get("candidate_audit_fingerprint")
        == preview_policy.get("candidate_audit_fingerprint"),
        str(preview_approval_event),
    )
    preview_approval_event_without_id = {
        key: value for key, value in preview_approval_event.items() if key != "event_id"
    }
    add_check(
        checks,
        "xr_vits_policy_preview_approval_event_id",
        bool(preview_approval_event_without_id)
        and preview_approval_event.get("event_id") == stable_sha256(preview_approval_event_without_id),
        str(preview_approval_event.get("event_id", "missing")),
    )
    add_check(
        checks,
        "xr_vits_policy_preview_active_policy_path",
        isinstance(xr_policy_preview, dict)
        and xr_policy_preview.get("active_policy_path") == str(root / "docs/resources/xr_vits_replacement_policy.json"),
        str(xr_policy_preview.get("active_policy_path") if isinstance(xr_policy_preview, dict) else "missing"),
    )
    add_check(
        checks,
        "xr_vits_policy_preview_candidate_audit_rel",
        preview_policy.get("candidate_audit") == "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
        str(preview_policy.get("candidate_audit", "missing")),
    )
    add_check(
        checks,
        "xr_vits_policy_preview_integrity_candidate_path",
        preview_integrity.get("candidate_audit_path")
        == str((root / "docs/resources/xr_vits_candidate_audit_2026_06_10.json").resolve()),
        str(preview_integrity.get("candidate_audit_path", "missing")),
    )
    add_check(
        checks,
        "xr_vits_policy_preview_no_side_effects",
        preview_safety.get("creates_xr_vits_policy") is False
        and preview_safety.get("writes_canonical_inputs") is False
        and preview_safety.get("executes_commands") is False,
        str(preview_safety),
    )
    add_check(
        checks,
        "final_unblock_closeout_packet_ready",
        isinstance(closeout_packet, dict) and closeout_packet.get("status") == "ready-for-operator-unblock",
        str(closeout_packet.get("status") if isinstance(closeout_packet, dict) else "missing"),
    )
    add_check(
        checks,
        "final_unblock_intake_status_known",
        isinstance(unblock_intake, dict) and unblock_intake.get("status") in {"blocked", "ready-for-active-unblock"},
        str(unblock_intake.get("status") if isinstance(unblock_intake, dict) else "missing"),
    )
    intake_blockers = (
        unblock_intake.get("blocker_status", {})
        if isinstance(unblock_intake, dict) and isinstance(unblock_intake.get("blocker_status"), dict)
        else {}
    )
    intake_next_inputs = (
        unblock_intake.get("next_inputs", [])
        if isinstance(unblock_intake, dict) and isinstance(unblock_intake.get("next_inputs"), list)
        else []
    )
    intake_operator_sequence = (
        unblock_intake.get("operator_sequence", [])
        if isinstance(unblock_intake, dict) and isinstance(unblock_intake.get("operator_sequence"), list)
        else []
    )
    intake_sequence_by_step = {
        str(item.get("step", "")): item for item in intake_operator_sequence if isinstance(item, dict)
    }
    intake_next_input_by_blocker = {
        str(item.get("blocker", "")): item for item in intake_next_inputs if isinstance(item, dict)
    }
    intake_c3b = (
        intake_blockers.get("C3b AXIS/DMA physical smoke result", {})
        if isinstance(intake_blockers.get("C3b AXIS/DMA physical smoke result"), dict)
        else {}
    )
    intake_xr = (
        intake_blockers.get("requested XR-VITs sibling", {})
        if isinstance(intake_blockers.get("requested XR-VITs sibling"), dict)
        else {}
    )
    expected_c3b_smoke_path = str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json")
    expected_xr_exact_path = str(root.parent.parent / "XR-VITs")
    expected_xr_policy_path = str(root / "docs/resources/xr_vits_replacement_policy.json")
    add_check(
        checks,
        "final_unblock_intake_blocker_status_keys",
        set(intake_blockers) == {"C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"},
        str(sorted(intake_blockers)),
    )
    add_check(
        checks,
        "final_unblock_intake_c3b_next_input_path",
        intake_c3b.get("canonical_path") == expected_c3b_smoke_path
        and intake_c3b.get("next_input_path") == expected_c3b_smoke_path
        and intake_c3b.get("next_input") == "board-produced C3b smoke JSON",
        str(intake_c3b),
    )
    add_check(
        checks,
        "final_unblock_intake_xr_vits_next_input_path",
        intake_xr.get("exact_path") == expected_xr_exact_path
        and intake_xr.get("policy_path") == expected_xr_policy_path
        and intake_xr.get("next_input_path") == f"{expected_xr_exact_path} OR {expected_xr_policy_path}",
        str(intake_xr),
    )
    add_check(
        checks,
        "final_unblock_intake_next_inputs_match_blockers",
        isinstance(intake_next_inputs, list)
        and {str(item.get("blocker")) for item in intake_next_inputs if isinstance(item, dict)}
        == {
            blocker
            for blocker, status in intake_blockers.items()
            if isinstance(status, dict) and not bool(status.get("ready_for_active_unblock"))
        },
        str(intake_next_inputs),
    )
    plan_not_ready_blockers = {
        blocker for blocker, item in plan_input_by_blocker.items() if not bool(item.get("ready"))
    }
    add_check(
        checks,
        "final_unblock_intake_next_inputs_match_operator_plan_required_inputs",
        set(intake_next_input_by_blocker) == plan_not_ready_blockers,
        f"intake={sorted(intake_next_input_by_blocker)} plan={sorted(plan_not_ready_blockers)}",
    )
    add_check(
        checks,
        "final_unblock_intake_next_input_paths_match_operator_plan",
        str(intake_next_input_by_blocker.get("C3b AXIS/DMA physical smoke result", {}).get("path", ""))
        == str(plan_input_by_blocker.get("C3b AXIS/DMA physical smoke result", {}).get("canonical_path", ""))
        and str(intake_next_input_by_blocker.get("requested XR-VITs sibling", {}).get("path", ""))
        == (
            str(plan_input_by_blocker.get("requested XR-VITs sibling", {}).get("exact_path", ""))
            + " OR "
            + str(plan_input_by_blocker.get("requested XR-VITs sibling", {}).get("policy_path", ""))
        ),
        f"intake={intake_next_input_by_blocker} plan={plan_input_by_blocker}",
    )
    plan_combined_dry_run = plan_dry_run_by_step.get("combined-final-runner-dry-run", {})
    plan_active_sequence = (
        operator_plan.get("active_sequence", [])
        if isinstance(operator_plan.get("active_sequence"), list)
        else []
    )
    plan_active_by_step = {
        str(item.get("step", "")): item for item in plan_active_sequence if isinstance(item, dict)
    }
    plan_combined_active = plan_active_by_step.get("combined-final-runner-active", {})
    add_check(
        checks,
        "final_unblock_intake_operator_sequence_contains_expected_steps",
        set(intake_sequence_by_step)
        == {"candidate-audit", "dry-run-final-runner", "active-final-runner"},
        str(sorted(intake_sequence_by_step)),
    )
    add_check(
        checks,
        "final_unblock_intake_operator_sequence_matches_operator_plan_commands",
        str(intake_sequence_by_step.get("dry-run-final-runner", {}).get("command", ""))
        == str(plan_combined_dry_run.get("command", ""))
        and str(intake_sequence_by_step.get("active-final-runner", {}).get("command", ""))
        == str(plan_combined_active.get("command", "")),
        f"intake={intake_sequence_by_step} plan_dry={plan_combined_dry_run} plan_active={plan_combined_active}",
    )
    add_check(
        checks,
        "final_unblock_intake_operator_sequence_side_effect_profile",
        intake_sequence_by_step.get("candidate-audit", {}).get("side_effects") is False
        and intake_sequence_by_step.get("dry-run-final-runner", {}).get("side_effects") is False
        and intake_sequence_by_step.get("active-final-runner", {}).get("side_effects")
        == plan_combined_active.get("side_effects"),
        f"intake={intake_sequence_by_step} plan_active={plan_combined_active}",
    )
    intake_safety = (
        unblock_intake.get("safety", {})
        if isinstance(unblock_intake, dict) and isinstance(unblock_intake.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "final_unblock_intake_no_side_effects",
        intake_safety.get("executes_commands") is False
        and intake_safety.get("creates_board_result") is False
        and intake_safety.get("creates_xr_vits_policy") is False
        and intake_safety.get("writes_canonical_inputs") is False,
        str(intake_safety),
    )
    closeout_safety = (
        closeout_packet.get("safety", {})
        if isinstance(closeout_packet, dict) and isinstance(closeout_packet.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "final_unblock_closeout_packet_no_side_effects",
        closeout_safety.get("executes_commands") is False
        and closeout_safety.get("creates_board_result") is False
        and closeout_safety.get("creates_xr_vits_policy") is False
        and closeout_safety.get("writes_canonical_inputs") is False,
        str(closeout_safety),
    )
    add_check(
        checks,
        "final_unblock_closeout_validation_pass",
        isinstance(closeout_validation, dict) and closeout_validation.get("status") == "pass",
        str(closeout_validation.get("status") if isinstance(closeout_validation, dict) else "missing"),
    )
    add_check(
        checks,
        "final_unblock_closeout_validation_fail_zero",
        isinstance(closeout_validation, dict) and closeout_validation.get("fail_count") == 0,
        str(closeout_validation.get("fail_count") if isinstance(closeout_validation, dict) else "missing"),
    )
    add_check(
        checks,
        "final_unblock_closeout_validation_check_count",
        isinstance(closeout_validation, dict) and int_at_least(closeout_validation.get("check_count", 0), 49),
        str(closeout_validation.get("check_count") if isinstance(closeout_validation, dict) else "missing"),
    )
    add_check(
        checks,
        "final_operator_handoff_validation_check_count",
        isinstance(operator_handoff_validation, dict)
        and int_at_least(operator_handoff_validation.get("check_count"), 131),
        str(operator_handoff_validation.get("check_count") if isinstance(operator_handoff_validation, dict) else "missing"),
    )
    add_check(
        checks,
        "final_signoff_bundle_validation_check_count",
        isinstance(final_bundle_validation, dict) and int_at_least(final_bundle_validation.get("check_count"), 152),
        str(final_bundle_validation.get("check_count") if isinstance(final_bundle_validation, dict) else "missing"),
    )
    closeout_qkv_gate = (
        closeout_packet.get("qkv_uram_successor_gate", {})
        if isinstance(closeout_packet, dict) and isinstance(closeout_packet.get("qkv_uram_successor_gate"), dict)
        else {}
    )
    closeout_commands = (
        closeout_packet.get("operator_commands", {})
        if isinstance(closeout_packet, dict) and isinstance(closeout_packet.get("operator_commands"), dict)
        else {}
    )
    closeout_qkv_commands = closeout_commands.get("qkv_uram_successor", [])
    add_check(
        checks,
        "final_unblock_closeout_qkv_gate_optional",
        isinstance(closeout_qkv_gate, dict) and closeout_qkv_gate.get("required_for_final_signoff") is False,
        str(closeout_qkv_gate.get("required_for_final_signoff") if isinstance(closeout_qkv_gate, dict) else "missing"),
    )
    add_check(
        checks,
        "final_unblock_closeout_qkv_gate_status_known",
        isinstance(closeout_qkv_gate, dict)
        and closeout_qkv_gate.get("status") in {"ready-for-physical-smoke", "pass", "not_available"},
        str(closeout_qkv_gate.get("status") if isinstance(closeout_qkv_gate, dict) else "missing"),
    )
    add_check(
        checks,
        "final_unblock_closeout_qkv_u5_commands_present",
        isinstance(closeout_qkv_commands, list)
        and any("--execute-qkv-uram-smoke" in str(command) for command in closeout_qkv_commands)
        and any("--dry-run-import-qkv-uram-smoke" in str(command) for command in closeout_qkv_commands),
        str(closeout_qkv_commands),
    )
    closeout_qkv_required = (
        closeout_qkv_gate.get("required_for_final_signoff") is True if isinstance(closeout_qkv_gate, dict) else False
    )
    closeout_qkv_smoke_status = closeout_qkv_gate.get("physical_smoke_status") if isinstance(closeout_qkv_gate, dict) else None
    closeout_qkv_smoke_json = closeout_qkv_gate.get("physical_smoke_result_json") if isinstance(closeout_qkv_gate, dict) else None
    add_check(
        checks,
        "final_unblock_closeout_qkv_required_smoke_status_consistent",
        (not closeout_qkv_required) or closeout_qkv_smoke_status == "pass",
        f"required={closeout_qkv_required} status={closeout_qkv_smoke_status}",
    )
    add_check(
        checks,
        "final_unblock_closeout_qkv_required_smoke_json_present",
        (not closeout_qkv_required) or bool(closeout_qkv_smoke_json),
        f"required={closeout_qkv_required} json={closeout_qkv_smoke_json}",
    )
    add_check(
        checks,
        "selected_path_audit_pass",
        isinstance(selected_path, dict) and selected_path.get("status") == "pass",
        str(selected_path.get("status") if isinstance(selected_path, dict) else "missing"),
    )
    add_check(
        checks,
        "selected_path_audit_fail_zero",
        isinstance(selected_path, dict) and selected_path.get("fail_count") == 0,
        str(selected_path.get("fail_count") if isinstance(selected_path, dict) else "missing"),
    )
    add_check(
        checks,
        "selected_path_audit_check_count",
        isinstance(selected_path, dict) and int_at_least(selected_path.get("check_count", 0), 22),
        str(selected_path.get("check_count") if isinstance(selected_path, dict) else "missing"),
    )
    add_check(
        checks,
        "selected_path_e_pending",
        isinstance(selected_path, dict)
        and isinstance(selected_path.get("selection"), dict)
        and selected_path["selection"].get("e") == "pending",
        str(selected_path.get("selection", {}).get("e") if isinstance(selected_path, dict) else "missing"),
    )
    add_check(
        checks,
        "resource_policy_audit_pass",
        isinstance(resource_policy, dict) and resource_policy.get("status") == "pass",
        str(resource_policy.get("status") if isinstance(resource_policy, dict) else "missing"),
    )
    add_check(
        checks,
        "resource_policy_audit_fail_zero",
        isinstance(resource_policy, dict) and resource_policy.get("fail_count") == 0,
        str(resource_policy.get("fail_count") if isinstance(resource_policy, dict) else "missing"),
    )
    add_check(
        checks,
        "resource_policy_audit_check_count",
        isinstance(resource_policy, dict) and int_at_least(resource_policy.get("check_count", 0), 36),
        str(resource_policy.get("check_count") if isinstance(resource_policy, dict) else "missing"),
    )
    resource_policy_artifacts = resource_policy_artifact_set(root)
    add_check(
        checks,
        "resource_policy_canonical_pair_present",
        bool(resource_policy_artifacts["canonical_pair_present"]),
        json.dumps(resource_policy_artifacts, sort_keys=True),
    )
    add_check(
        checks,
        "resource_policy_docs_singularity",
        bool(resource_policy_artifacts["docs_singularity"]),
        json.dumps(resource_policy_artifacts, sort_keys=True),
    )
    add_check(
        checks,
        "resource_policy_generated_singularity",
        bool(resource_policy_artifacts["generated_singularity"]),
        json.dumps(resource_policy_artifacts, sort_keys=True),
    )
    add_check(
        checks,
        "resource_policy_filename_payload_date_match",
        bool(resource_policy_artifacts["filename_payload_date_match"]),
        json.dumps(resource_policy_artifacts, sort_keys=True),
    )
    add_check(
        checks,
        "resource_policy_canonical_mirror_sha256",
        bool(resource_policy_artifacts["canonical_mirror_sha256"]),
        json.dumps(resource_policy_artifacts, sort_keys=True),
    )
    add_check(
        checks,
        "resource_policy_audit_single_canonical_artifact_set",
        resource_policy_artifacts["status"] == "pass",
        json.dumps(resource_policy_artifacts, sort_keys=True),
    )
    resource_policy_checks = named_checks(resource_policy)
    for name in [
        "force_dsp_macro_default",
        "force_uram_macro_default",
        "small_mem_lutram_macro_default",
        "cyclic_weight_tiles_uram_macro_default",
        "cyclic_large_temps_uram_macro_default",
        "cyclic_small_tile_lutram_macro_default",
        "dsp_bind_op_present",
        "rmu_smu_dsp_bind_op_present",
        "rmu_smu_dsp_helper_defined",
        "rmu_smu_dsp_acc_helper_defined",
        "rmu_projection_uses_dsp_helper",
        "smu_relation_uses_dsp_helper",
        "uram_bind_storage_present",
        "lutram_bind_storage_present",
        "cyclic_weight_tiles_uram_pragmas",
        "cyclic_large_temps_uram_pragmas",
        "cyclic_small_tile_lutram_pragmas",
        "zcu104_parallelism_default",
        "zcu104_dense_parallelism_default",
        "zcu104_fifo_depth_default",
        "resource_matrix_pass",
        "resource_matrix_recommends_c3b",
        "c3b_parallelism_16",
        "c3b_memory_banks_16",
        "c3b_dsp_increased_vs_a1",
        "c3b_lut_lower_than_c1",
        "c3b_uram_positive",
        "c3b_latency_not_worse_than_c1",
        "c3b_csynth_xml_exists",
        "c3b_csynth_resources_match_matrix",
        "c3b_csynth_latency_matches_matrix",
        "c3b_csynth_dsp_lte_threshold",
        "c3b_csynth_uram_lte_threshold",
        "c3b_csynth_lut_lte_threshold",
        "c3b_csynth_latency_lte_threshold",
        "c3b_routed_wns_gte_threshold",
    ]:
        check = resource_policy_checks.get(name, {})
        add_check(
            checks,
            f"resource_policy_audit_{name}",
            check.get("status") == "pass",
            str(check.get("detail", "missing")),
        )
    add_check(
        checks,
        "third_goal_source_audit_pass",
        isinstance(source_audit, dict) and source_audit.get("status") == "pass",
        str(source_audit.get("status") if isinstance(source_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "third_goal_source_audit_required_count",
        isinstance(source_audit, dict) and int_at_least(source_audit.get("required_count", 0), 86),
        str(source_audit.get("required_count") if isinstance(source_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "third_goal_source_audit_source_count",
        isinstance(source_audit, dict) and int_at_least(source_audit.get("source_count", 0), 224),
        str(source_audit.get("source_count") if isinstance(source_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "third_goal_source_audit_missing_zero",
        isinstance(source_audit, dict) and source_audit.get("missing_required") == [],
        str(source_audit.get("missing_required") if isinstance(source_audit, dict) else "missing"),
    )
    current_summary = (
        current_audit.get("summary", {})
        if isinstance(current_audit, dict) and isinstance(current_audit.get("summary"), dict)
        else {}
    )
    requirements_trace_blocked_ids = (
        requirements_trace.get("blocked_requirement_ids", [])
        if isinstance(requirements_trace, dict) and isinstance(requirements_trace.get("blocked_requirement_ids"), list)
        else []
    )
    requirements_trace_partial_ids = (
        requirements_trace.get("partial_requirement_ids", [])
        if isinstance(requirements_trace, dict) and isinstance(requirements_trace.get("partial_requirement_ids"), list)
        else []
    )
    add_check(
        checks,
        "third_goal_current_audit_status_known",
        isinstance(current_audit, dict) and current_audit.get("status") in {"pass", "partial", "blocked-external"},
        str(current_audit.get("status") if isinstance(current_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "third_goal_current_audit_requirements_count",
        int_at_least(current_summary.get("requirements", 0), 12),
        str(current_summary.get("requirements", "missing")),
    )
    completion_items = (
        completion_audit.get("items", [])
        if isinstance(completion_audit, dict) and isinstance(completion_audit.get("items"), list)
        else []
    )
    completion_by_id = {
        str(item.get("id")): item
        for item in completion_items
        if isinstance(item, dict) and item.get("id") is not None
    }
    completion_pass_count = sum(
        1 for item in completion_items if isinstance(item, dict) and item.get("status") == "pass"
    )
    completion_partial_count = sum(
        1 for item in completion_items if isinstance(item, dict) and item.get("status") == "partial"
    )
    completion_blocked_count = sum(
        1 for item in completion_items if isinstance(item, dict) and item.get("status") == "blocked"
    )
    add_check(
        checks,
        "third_goal_completion_audit_status_known",
        isinstance(completion_audit, dict) and completion_audit.get("status") in {"pass", "partial", "blocked"},
        str(completion_audit.get("status") if isinstance(completion_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "third_goal_completion_audit_item_count",
        isinstance(completion_audit, dict) and completion_audit.get("item_count") == 14,
        str(completion_audit.get("item_count") if isinstance(completion_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "third_goal_completion_audit_counts_match_items",
        isinstance(completion_audit, dict)
        and completion_audit.get("pass_count") == completion_pass_count
        and completion_audit.get("partial_count") == completion_partial_count
        and completion_audit.get("blocked_count") == completion_blocked_count,
        (
            f"stored={completion_audit.get('pass_count') if isinstance(completion_audit, dict) else 'missing'}/"
            f"{completion_audit.get('partial_count') if isinstance(completion_audit, dict) else 'missing'}/"
            f"{completion_audit.get('blocked_count') if isinstance(completion_audit, dict) else 'missing'} "
            f"computed={completion_pass_count}/{completion_partial_count}/{completion_blocked_count}"
        ),
    )
    add_check(
        checks,
        "third_goal_completion_audit_req12_manifest_pass",
        completion_by_id.get("12", {}).get("status") == "pass",
        str(completion_by_id.get("12", {}).get("status", "missing")),
    )
    add_check(
        checks,
        "third_goal_completion_audit_req11_matches_trace",
        (
            ("11" in [str(item) for item in requirements_trace_blocked_ids])
            == (completion_by_id.get("11", {}).get("status") == "blocked")
        ),
        f"trace_blocked={requirements_trace_blocked_ids} completion_req11={completion_by_id.get('11', {}).get('status', 'missing')}",
    )
    add_check(
        checks,
        "third_goal_completion_audit_req4_matches_trace",
        ("4" in [str(item) for item in requirements_trace_partial_ids])
        == (completion_by_id.get("4", {}).get("status") == "partial"),
        f"trace_partial={requirements_trace_partial_ids} completion_req4={completion_by_id.get('4', {}).get('status', 'missing')}",
    )
    add_check(
        checks,
        "req1_environment_audit_pass",
        isinstance(req1_environment, dict) and req1_environment.get("status") == "pass",
        str(req1_environment.get("status") if isinstance(req1_environment, dict) else "missing"),
    )
    add_check(
        checks,
        "req1_environment_audit_fail_zero",
        isinstance(req1_environment, dict) and req1_environment.get("fail_count") == 0,
        str(req1_environment.get("fail_count") if isinstance(req1_environment, dict) else "missing"),
    )
    add_check(
        checks,
        "req1_environment_audit_check_count",
        isinstance(req1_environment, dict) and int_at_least(req1_environment.get("check_count"), 13),
        str(req1_environment.get("check_count") if isinstance(req1_environment, dict) else "missing"),
    )
    req1_checks = named_checks(req1_environment)
    for name in [
        "platform_linux",
        "ubuntu_id",
        "ubuntu_version_22_04",
        "not_wsl_kernel",
        "hardware_root_expected_prefix",
        "no_legacy_wsl_or_xilinx_paths",
        "vitis_hls_tools_xilinx_executable",
        "vivado_tools_xilinx_executable",
    ]:
        check = req1_checks.get(name, {})
        add_check(
            checks,
            f"req1_environment_audit_{name}",
            check.get("status") == "pass",
            str(check.get("detail", "missing")),
        )
    req9_requested = (
        req9_deit_image.get("requested_image", {})
        if isinstance(req9_deit_image, dict) and isinstance(req9_deit_image.get("requested_image"), dict)
        else {}
    )
    req9_safety = (
        req9_deit_image.get("safety", {})
        if isinstance(req9_deit_image, dict) and isinstance(req9_deit_image.get("safety"), dict)
        else {}
    )
    req9_checks = named_checks(req9_deit_image)
    add_check(
        checks,
        "req9_deit_image_audit_pass",
        isinstance(req9_deit_image, dict) and req9_deit_image.get("status") == "pass",
        str(req9_deit_image.get("status") if isinstance(req9_deit_image, dict) else "missing"),
    )
    add_check(
        checks,
        "req9_deit_image_audit_fail_zero",
        isinstance(req9_deit_image, dict) and req9_deit_image.get("fail_count") == 0,
        str(req9_deit_image.get("fail_count") if isinstance(req9_deit_image, dict) else "missing"),
    )
    add_check(
        checks,
        "req9_deit_image_audit_check_count",
        isinstance(req9_deit_image, dict) and int_at_least(req9_deit_image.get("check_count"), 11),
        str(req9_deit_image.get("check_count") if isinstance(req9_deit_image, dict) else "missing"),
    )
    add_check(
        checks,
        "req9_deit_image_requested_sha256_valid",
        isinstance(req9_requested.get("sha256"), str) and len(req9_requested.get("sha256", "")) == 64,
        str(req9_requested.get("sha256", "missing")),
    )
    for name in [
        "requested_image_exists",
        "requested_image_png_magic",
        "hgpipe_substitute_matches_requested_sha256",
        "hardware_docs_copy_matches_requested_sha256",
    ]:
        check = req9_checks.get(name, {})
        add_check(
            checks,
            f"req9_deit_image_{name}",
            check.get("status") == "pass",
            str(check.get("detail", "missing")),
        )
    add_check(
        checks,
        "req9_deit_image_no_side_effects",
        req9_safety.get("executes_commands") is False
        and req9_safety.get("executes_network") is False
        and req9_safety.get("creates_board_result") is False
        and req9_safety.get("creates_xr_vits_policy") is False
        and req9_safety.get("writes_canonical_inputs") is False,
        str(req9_safety),
    )
    xr_gate_exact = (
        xr_vits_gate.get("exact", {})
        if isinstance(xr_vits_gate, dict) and isinstance(xr_vits_gate.get("exact"), dict)
        else {}
    )
    xr_gate_replacement = (
        xr_vits_gate.get("replacement_candidate", {})
        if isinstance(xr_vits_gate, dict) and isinstance(xr_vits_gate.get("replacement_candidate"), dict)
        else {}
    )
    xr_gate_active_policy = (
        xr_vits_gate.get("active_policy_summary", {})
        if isinstance(xr_vits_gate, dict) and isinstance(xr_vits_gate.get("active_policy_summary"), dict)
        else {}
    )
    xr_gate_candidate = (
        xr_vits_gate.get("candidate_audit", {})
        if isinstance(xr_vits_gate, dict) and isinstance(xr_vits_gate.get("candidate_audit"), dict)
        else {}
    )
    xr_gate_safety = (
        xr_vits_gate.get("safety", {})
        if isinstance(xr_vits_gate, dict) and isinstance(xr_vits_gate.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "xr_vits_gate_status_blocked_or_clear",
        isinstance(xr_vits_gate, dict)
        and xr_vits_gate.get("status") in {"blocked", "pass-exact", "pass-replacement-policy"},
        str(xr_vits_gate.get("status") if isinstance(xr_vits_gate, dict) else "missing"),
    )
    add_check(
        checks,
        "xr_vits_gate_current_resolution_expected",
        isinstance(xr_vits_gate, dict)
        and xr_vits_gate.get("resolution_mode")
        in {"candidate-ready-needs-approval", "exact", "replacement-policy"},
        str(xr_vits_gate.get("resolution_mode") if isinstance(xr_vits_gate, dict) else "missing"),
    )
    add_check(
        checks,
        "xr_vits_gate_exact_path_absent_or_clear",
        xr_gate_exact.get("exists") in {False, True} and isinstance(xr_gate_exact.get("path"), str),
        f"exists={xr_gate_exact.get('exists')} path={xr_gate_exact.get('path')}",
    )
    add_check(
        checks,
        "xr_vits_gate_replacement_candidate_present",
        xr_gate_replacement.get("exists") is True and isinstance(xr_gate_replacement.get("path"), str),
        f"exists={xr_gate_replacement.get('exists')} path={xr_gate_replacement.get('path')}",
    )
    add_check(
        checks,
        "xr_vits_gate_candidate_recommendation_matches",
        xr_gate_candidate.get("recommendation_matches") is True,
        str(xr_gate_candidate.get("recommendation_matches", "missing")),
    )
    add_check(
        checks,
        "xr_vits_gate_active_policy_status_known",
        xr_gate_active_policy.get("status") in {"missing", "present"},
        str(xr_gate_active_policy.get("status", "missing")),
    )
    add_check(
        checks,
        "xr_vits_gate_no_side_effects",
        xr_gate_safety.get("creates_xr_vits_policy") is False
        and xr_gate_safety.get("executes_commands") is False
        and xr_gate_safety.get("executes_network") is False
        and xr_gate_safety.get("writes_canonical_inputs") is False,
        str(xr_gate_safety),
    )
    c3b_canonical = (
        c3b_physical_gate.get("canonical_result", {})
        if isinstance(c3b_physical_gate, dict) and isinstance(c3b_physical_gate.get("canonical_result"), dict)
        else {}
    )
    c3b_validation = (
        c3b_physical_gate.get("canonical_validation", {})
        if isinstance(c3b_physical_gate, dict) and isinstance(c3b_physical_gate.get("canonical_validation"), dict)
        else {}
    )
    c3b_bundle = (
        c3b_physical_gate.get("bundle", {})
        if isinstance(c3b_physical_gate, dict) and isinstance(c3b_physical_gate.get("bundle"), dict)
        else {}
    )
    c3b_session = (
        c3b_physical_gate.get("session", {})
        if isinstance(c3b_physical_gate, dict) and isinstance(c3b_physical_gate.get("session"), dict)
        else {}
    )
    c3b_safety = (
        c3b_physical_gate.get("safety", {})
        if isinstance(c3b_physical_gate, dict) and isinstance(c3b_physical_gate.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "c3b_physical_smoke_gate_status_known",
        isinstance(c3b_physical_gate, dict)
        and c3b_physical_gate.get("status") in {"pass", "blocked_missing_canonical_physical_smoke_result"},
        str(c3b_physical_gate.get("status") if isinstance(c3b_physical_gate, dict) else "missing"),
    )
    add_check(
        checks,
        "c3b_physical_smoke_gate_ready_for_board",
        isinstance(c3b_physical_gate, dict) and c3b_physical_gate.get("ready_for_board") is True,
        str(c3b_physical_gate.get("ready_for_board") if isinstance(c3b_physical_gate, dict) else "missing"),
    )
    add_check(
        checks,
        "c3b_physical_smoke_gate_canonical_result_status_known",
        c3b_canonical.get("status") in {"missing", "pass", "fail"},
        str(c3b_canonical.get("status", "missing")),
    )
    add_check(
        checks,
        "c3b_physical_smoke_gate_canonical_validation_status_known",
        c3b_validation.get("status") in {"missing", "pass", "fail"},
        str(c3b_validation.get("status", "missing")),
    )
    add_check(
        checks,
        "c3b_physical_smoke_gate_bundle_pass",
        c3b_bundle.get("status") == "pass",
        str(c3b_bundle.get("status", "missing")),
    )
    add_check(
        checks,
        "c3b_physical_smoke_gate_session_pass",
        c3b_session.get("status") == "pass",
        str(c3b_session.get("status", "missing")),
    )
    add_check(
        checks,
        "c3b_physical_smoke_gate_no_side_effects",
        c3b_safety.get("executes_commands") is False
        and c3b_safety.get("executes_network") is False
        and c3b_safety.get("creates_board_result") is False
        and c3b_safety.get("writes_canonical_inputs") is False,
        str(c3b_safety),
    )
    requirements_trace_policy = (
        requirements_trace.get("xr_vits_policy_integrity", {})
        if isinstance(requirements_trace, dict) and isinstance(requirements_trace.get("xr_vits_policy_integrity"), dict)
        else {}
    )
    requirements_trace_blocked_ids = (
        requirements_trace.get("blocked_requirement_ids", [])
        if isinstance(requirements_trace, dict) and isinstance(requirements_trace.get("blocked_requirement_ids"), list)
        else []
    )
    add_check(
        checks,
        "third_goal_requirements_trace_status_known",
        isinstance(requirements_trace, dict) and requirements_trace.get("status") in {"pass", "partial", "blocked"},
        str(requirements_trace.get("status") if isinstance(requirements_trace, dict) else "missing"),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_req11_blocked",
        "11" in [str(item) for item in requirements_trace_blocked_ids],
        str(requirements_trace_blocked_ids),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_xr_vits_policy_fields_complete",
        requirements_trace_policy.get("required_policy_fields_complete") is True,
        str(requirements_trace_policy.get("required_policy_fields_complete", "missing")),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_xr_vits_legacy_policy_rejected",
        requirements_trace_policy.get("legacy_policy_clears_final_signoff") is False,
        str(requirements_trace_policy.get("legacy_policy_clears_final_signoff", "missing")),
    )
    qkv_csynth = qkv_uram.get("csynth", {}) if isinstance(qkv_uram, dict) and isinstance(qkv_uram.get("csynth"), dict) else {}
    qkv_resources = qkv_csynth.get("resources", {}) if isinstance(qkv_csynth.get("resources"), dict) else {}
    qkv_uram_value = qkv_resources.get("uram")
    qkv_comparison = (
        qkv_uram.get("comparison", {})
        if isinstance(qkv_uram, dict) and isinstance(qkv_uram.get("comparison"), dict)
        else {}
    )
    qkv_delta = (
        qkv_comparison.get("delta_vs_dsp_mixed_stream", {})
        if isinstance(qkv_comparison.get("delta_vs_dsp_mixed_stream"), dict)
        else {}
    )
    qkv_overlay = (
        qkv_uram.get("overlay", {})
        if isinstance(qkv_uram, dict) and isinstance(qkv_uram.get("overlay"), dict)
        else {}
    )
    qkv_route_status = (
        qkv_overlay.get("route_status", {})
        if isinstance(qkv_overlay.get("route_status"), dict)
        else {}
    )
    qkv_top_checks = named_checks(qkv_uram)
    qkv_overlay_checks = named_checks(qkv_uram, "overlay")
    add_check(
        checks,
        "qkv_uram_successor_pass",
        isinstance(qkv_uram, dict) and qkv_uram.get("status") == "pass",
        str(qkv_uram.get("status") if isinstance(qkv_uram, dict) else "missing"),
    )
    add_check(
        checks,
        "qkv_uram_successor_uram_lte_c3b",
        isinstance(qkv_uram_value, int) and qkv_uram_value <= 64,
        str(qkv_uram_value if qkv_uram_value is not None else "missing"),
    )
    for check_name in [
        "latency_lte_c3b",
        "dsp_lte_c3b",
        "lut_lte_c3b",
        "bram_reduced_vs_dsp_mixed_stream",
    ]:
        check = qkv_top_checks.get(check_name, {})
        add_check(
            checks,
            f"qkv_uram_successor_{check_name}",
            check.get("status") == "pass",
            str(check.get("detail", "missing")),
        )
    for check_name in ["timing_wns_gte_c3b", "route_errors_zero"]:
        check = qkv_overlay_checks.get(check_name, {})
        add_check(
            checks,
            f"qkv_uram_overlay_{check_name}",
            check.get("status") == "pass",
            str(check.get("detail", "missing")),
        )
    add_check(
        checks,
        "qkv_uram_successor_bram_delta_le_zero",
        isinstance(qkv_delta.get("bram_18k"), int) and qkv_delta["bram_18k"] <= 0,
        str(qkv_delta.get("bram_18k", "missing")),
    )
    add_check(
        checks,
        "qkv_uram_overlay_fully_routed",
        isinstance(qkv_route_status.get("fully_routed_nets"), int)
        and isinstance(qkv_route_status.get("routable_nets"), int)
        and qkv_route_status["fully_routed_nets"] == qkv_route_status["routable_nets"],
        str(qkv_route_status),
    )
    req5_precision = (
        req5_q4q8.get("precision", {})
        if isinstance(req5_q4q8, dict) and isinstance(req5_q4q8.get("precision"), dict)
        else {}
    )
    add_check(
        checks,
        "req5_q4q8_swhw_match_pass",
        isinstance(req5_q4q8, dict) and req5_q4q8.get("status") == "pass",
        str(req5_q4q8.get("status") if isinstance(req5_q4q8, dict) else "missing"),
    )
    add_check(
        checks,
        "req5_q4q8_precision",
        req5_precision.get("weight_bits") == 4 and req5_precision.get("activation_bits") == 8,
        str(req5_precision),
    )
    req5_checks = named_checks(req5_q4q8)
    for name in [
        "packed_weight_manifest",
        "packed_weight_binary_exists",
        "packed_weight_sha256_matches_manifest",
        "packed_weight_byte_count_matches_manifest",
        "packed_weight_expected_runtime_state",
        "packed_weight_expected_c3b_output",
        "testbench_strict_golden_compare",
        "c3b_axis_csim_strict_csim",
        "vref_softmax_input_x2_csim_strict_csim",
        "qkv_uram_csim_strict_csim",
    ]:
        check = req5_checks.get(name, {})
        add_check(
            checks,
            f"req5_q4q8_{name}",
            check.get("status") == "pass",
            str(check.get("detail", "missing")),
        )
    pot_summary = (
        pot_scale_sweep.get("summary", {})
        if isinstance(pot_scale_sweep, dict) and isinstance(pot_scale_sweep.get("summary"), dict)
        else {}
    )
    add_check(
        checks,
        "vref_p0_pot_scale_audit_pass",
        isinstance(pot_scale_audit, dict) and pot_scale_audit.get("status") == "pass",
        str(pot_scale_audit.get("status") if isinstance(pot_scale_audit, dict) else "missing"),
    )
    add_check(
        checks,
        "vref_p0_pot_scale_audit_checks",
        isinstance(pot_scale_audit, dict)
        and int_at_least(pot_scale_audit.get("check_count"), 14)
        and pot_scale_audit.get("fail_count") == 0,
        f"checks={pot_scale_audit.get('check_count') if isinstance(pot_scale_audit, dict) else 'missing'} fail={pot_scale_audit.get('fail_count') if isinstance(pot_scale_audit, dict) else 'missing'}",
    )
    add_check(
        checks,
        "vref_p0_pot_scale_sweep_pass",
        isinstance(pot_scale_sweep, dict) and pot_scale_sweep.get("status") == "pass",
        str(pot_scale_sweep.get("status") if isinstance(pot_scale_sweep, dict) else "missing"),
    )
    add_check(
        checks,
        "vref_p0_pot_scale_sweep_candidate_coverage",
        int_at_least(pot_scale_sweep.get("spec_count") if isinstance(pot_scale_sweep, dict) else None, 3)
        and int_at_least(pot_summary.get("total_candidate_count"), 45)
        and pot_summary.get("total_fail_count") == 0,
        f"specs={pot_scale_sweep.get('spec_count') if isinstance(pot_scale_sweep, dict) else 'missing'} candidates={pot_summary.get('total_candidate_count', 'missing')} fail={pot_summary.get('total_fail_count', 'missing')}",
    )
    add_check(
        checks,
        "vref_p0_pot_scale_sweep_current_recommended",
        int_at_least(pot_summary.get("specs_recommending_current"), 3)
        and "keep current PoT scales" in str(pot_summary.get("next_action", "")),
        str(pot_summary),
    )
    p2_policy = (
        p2_scale_calibration.get("policy", {})
        if isinstance(p2_scale_calibration, dict) and isinstance(p2_scale_calibration.get("policy"), dict)
        else {}
    )
    p2_precision = (
        p2_scale_calibration.get("precision", {})
        if isinstance(p2_scale_calibration, dict) and isinstance(p2_scale_calibration.get("precision"), dict)
        else {}
    )
    p2_summary = (
        p2_scale_calibration.get("sweep_summary", {})
        if isinstance(p2_scale_calibration, dict) and isinstance(p2_scale_calibration.get("sweep_summary"), dict)
        else {}
    )
    p2_safety = (
        p2_scale_calibration.get("safety", {})
        if isinstance(p2_scale_calibration, dict) and isinstance(p2_scale_calibration.get("safety"), dict)
        else {}
    )
    add_check(
        checks,
        "p2_vit_scale_calibration_pass",
        isinstance(p2_scale_calibration, dict) and p2_scale_calibration.get("status") == "pass",
        str(p2_scale_calibration.get("status") if isinstance(p2_scale_calibration, dict) else "missing"),
    )
    add_check(
        checks,
        "p2_vit_scale_calibration_q4w8a",
        p2_precision.get("weight_bits") == 4 and p2_precision.get("activation_bits") == 8,
        str(p2_precision),
    )
    add_check(
        checks,
        "p2_vit_scale_calibration_current_decision",
        p2_policy.get("decision") == "keep_current_pot_scales_for_c3b"
        and int_at_least(p2_summary.get("specs_recommending_current"), 3),
        f"policy={p2_policy} summary={p2_summary}",
    )
    add_check(
        checks,
        "p2_vit_scale_calibration_candidate_coverage",
        int_at_least(p2_summary.get("total_candidate_count"), 45) and p2_summary.get("total_fail_count") == 0,
        str(p2_summary),
    )
    add_check(
        checks,
        "p2_vit_scale_calibration_no_side_effects",
        all(
            p2_safety.get(key) is False
            for key in [
                "executes_hls",
                "executes_vivado",
                "writes_hls_source",
                "overwrites_c3b_artifacts",
                "creates_board_result",
                "creates_xr_vits_policy",
            ]
        ),
        str(p2_safety),
    )
    req6_knobs = (
        req6_parameterization.get("knobs", {}).get("config_macros", {})
        if isinstance(req6_parameterization, dict)
        and isinstance(req6_parameterization.get("knobs"), dict)
        and isinstance(req6_parameterization["knobs"].get("config_macros"), dict)
        else {}
    )
    add_check(
        checks,
        "req6_parameterization_pass",
        isinstance(req6_parameterization, dict) and req6_parameterization.get("status") == "pass",
        str(req6_parameterization.get("status") if isinstance(req6_parameterization, dict) else "missing"),
    )
    add_check(
        checks,
        "req6_parameterization_knobs",
        req6_knobs.get("HGTXR_TILING_FACTOR") == 1
        and req6_knobs.get("HGTXR_PARALLELISM_FACTOR") == 8
        and req6_knobs.get("HGTXR_BUS_WIDTH") == 256
        and req6_knobs.get("HGTXR_BIT_WIDTH") == 8
        and req6_knobs.get("HGTXR_WEIGHT_BIT_WIDTH") == 4
        and req6_knobs.get("HGTXR_BUFFER_SIZE") == 256
        and req6_knobs.get("HGTXR_FIFO_DEPTH") == 128,
        str(req6_knobs),
    )
    req6_checks = named_checks(req6_parameterization)
    add_check(
        checks,
        "req6_parameterization_check_count",
        isinstance(req6_parameterization, dict) and int_at_least(req6_parameterization.get("check_count"), 71),
        str(req6_parameterization.get("check_count") if isinstance(req6_parameterization, dict) else "missing"),
    )
    add_check(
        checks,
        "req6_parameterization_fail_zero",
        isinstance(req6_parameterization, dict) and req6_parameterization.get("fail_count") == 0,
        str(req6_parameterization.get("fail_count") if isinstance(req6_parameterization, dict) else "missing"),
    )
    for name in [
        "legal_bus_width_byte_aligned",
        "legal_bus_width_data_divisible",
        "legal_bus_width_weight_divisible",
        "legal_weight_width_lte_data_width",
        "legal_data_width_lt_acc_width",
        "legal_model_dim_divides_heads",
        "legal_head_dim_matches_model_heads",
        "legal_ff_dim_matches_mlp_ratio",
        "legal_dense_parallelism_matches_req6_parallelism",
        "legal_dense_parallelism_divides_embed",
        "legal_dense_parallelism_divides_ff_dim",
        "legal_weight_lanes_divide_dense_parallelism",
        "legal_buffer_size_positive",
        "legal_fifo_depth_positive",
        "e2e_static_assert_active_tokens_fit",
        "e2e_static_assert_heads_positive",
        "e2e_static_assert_head_dim_covers_embed",
        "e2e_static_assert_dense_par_positive",
        "e2e_static_assert_dense_par_divides_embed",
        "e2e_static_assert_dense_par_divides_ff_dim",
        "e2e_static_assert_weight_lanes_positive",
        "e2e_static_assert_dense_par_divides_weight_lanes",
        "e2e_static_assert_axis_width_matches_cyclic_axi",
        "csim_tcl_supports_par16_par32",
        "csynth_tcl_supports_par16_par32",
        "sweep_yaml_parsed",
        "parallelism_extension_c3b_par16_validated",
        "parallelism_extension_c3b_par16_evidence_resource_matrix",
        "parallelism_extension_c3b_par16_expected_result",
        "parallelism_extension_par32_exploratory_not_default",
        "parallelism_extension_par32_requires_fresh_reports",
        "parallelism_extension_par32_records_risk",
    ]:
        check = req6_checks.get(name, {})
        add_check(
            checks,
            f"req6_parameterization_{name}",
            check.get("status") == "pass",
            str(check.get("detail", "missing")),
        )
    buffer_checks = named_checks(vref_buffer_lifetime)
    add_check(
        checks,
        "vref_p0_buffer_lifetime_pass",
        isinstance(vref_buffer_lifetime, dict) and vref_buffer_lifetime.get("status") == "pass",
        str(vref_buffer_lifetime.get("status") if isinstance(vref_buffer_lifetime, dict) else "missing"),
    )
    add_check(
        checks,
        "vref_p0_buffer_lifetime_check_count",
        isinstance(vref_buffer_lifetime, dict) and int_at_least(vref_buffer_lifetime.get("check_count"), 51),
        str(vref_buffer_lifetime.get("check_count") if isinstance(vref_buffer_lifetime, dict) else "missing"),
    )
    add_check(
        checks,
        "vref_p0_buffer_lifetime_fail_zero",
        isinstance(vref_buffer_lifetime, dict) and vref_buffer_lifetime.get("fail_count") == 0,
        str(vref_buffer_lifetime.get("fail_count") if isinstance(vref_buffer_lifetime, dict) else "missing"),
    )
    for name in [
        "force_uram_buffers_enabled",
        "small_mem_lutram_enabled",
        "axis_large_gb.q_uram",
        "axis_large_gb.k_uram",
        "axis_large_gb.v_uram",
        "axis_pooled_lutram",
        "attention_small_score_lutram",
        "attention_small_prob_lutram",
        "attention_small_exp_raw_lutram",
        "rmu_smu_small_score_lutram",
        "rmu_smu_small_prob_lutram",
        "qkv_cache_q_weight_cache_uram_successor_branch",
        "qkv_cache_k_weight_cache_uram_successor_branch",
        "qkv_cache_v_weight_cache_uram_successor_branch",
        "qkv_successor_file_present",
        "qkv_successor_status_pass",
        "qkv_successor_macro_uram_enabled",
        "qkv_successor_csim_pass",
        "qkv_successor_csynth_pass",
        "qkv_successor_overlay_routed",
        "qkv_successor_uram_increased_vs_dsp_mixed_stream",
        "qkv_successor_bram_reduced_vs_dsp_mixed_stream",
        "qkv_successor_uram_positive",
        "qkv_successor_physical_smoke_pending_only",
        "c3b_dsp_increased_vs_a1",
        "c3b_lut_lower_than_c1",
    ]:
        check = buffer_checks.get(name, {})
        add_check(
            checks,
            f"vref_p0_buffer_lifetime_{name}",
            check.get("status") == "pass",
            str(check.get("detail", "missing")),
        )
    hgpipe_contract = (
        hgpipe_operator.get("contract_summary", {})
        if isinstance(hgpipe_operator, dict) and isinstance(hgpipe_operator.get("contract_summary"), dict)
        else {}
    )
    hgpipe_properties = (
        hgpipe_operator.get("property_summary", {})
        if isinstance(hgpipe_operator, dict) and isinstance(hgpipe_operator.get("property_summary"), dict)
        else {}
    )
    hgpipe_operators = (
        hgpipe_operator.get("operators", [])
        if isinstance(hgpipe_operator, dict) and isinstance(hgpipe_operator.get("operators"), list)
        else []
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_operator_audit_file_present",
        isinstance(hgpipe_operator, dict),
        "json object" if isinstance(hgpipe_operator, dict) else "missing",
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_operator_audit_status_pass",
        isinstance(hgpipe_operator, dict) and hgpipe_operator.get("status") == "pass",
        str(hgpipe_operator.get("status") if isinstance(hgpipe_operator, dict) else "missing"),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_ref_checks_pass",
        hgpipe_contract.get("passed_ref_checks") == hgpipe_contract.get("total_ref_checks"),
        f"{hgpipe_contract.get('passed_ref_checks')}/{hgpipe_contract.get('total_ref_checks')}",
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_ref_check_count",
        int_at_least(hgpipe_contract.get("total_ref_checks"), 97),
        str(hgpipe_contract.get("total_ref_checks", "missing")),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_checked_sample_count",
        int_at_least(hgpipe_contract.get("total_checked_samples"), 5_899_008),
        str(hgpipe_contract.get("total_checked_samples", "missing")),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_property_summary_status_pass",
        hgpipe_properties.get("status") == "pass",
        str(hgpipe_properties.get("status", "missing")),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_property_summary_no_fail",
        hgpipe_properties.get("fail_count") == 0,
        str(hgpipe_properties.get("fail_count", "missing")),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_property_summary_counts_consistent",
        hgpipe_properties.get("pass_count") == hgpipe_properties.get("check_count"),
        f"{hgpipe_properties.get('pass_count')}/{hgpipe_properties.get('check_count')}",
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_property_checks_present",
        int_at_least(hgpipe_properties.get("check_count"), 211),
        str(hgpipe_properties.get("check_count", "missing")),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_operators_all_pass",
        bool(hgpipe_operators) and all(
            isinstance(operator, dict) and operator.get("status") == "pass"
            for operator in hgpipe_operators
        ),
        str([
            operator.get("operator", "unknown")
            for operator in hgpipe_operators
            if isinstance(operator, dict) and operator.get("status") != "pass"
        ]),
    )
    add_check(
        checks,
        "third_goal_requirements_trace_hgpipe_operator_property_fail_zero",
        bool(hgpipe_operators) and all(
            isinstance(operator, dict) and operator.get("property_fail_count") == 0
            for operator in hgpipe_operators
        ),
        str([
            operator.get("operator", "unknown")
            for operator in hgpipe_operators
            if isinstance(operator, dict) and operator.get("property_fail_count") != 0
        ]),
    )
    return checks


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    artifacts = [
        build_entry(root, artifact_id, rel, kind, True)
        for artifact_id, rel, kind in REQUIRED_ARTIFACTS
    ]
    artifacts.extend(
        build_entry(root, artifact_id, rel, kind, False)
        for artifact_id, rel, kind in OPTIONAL_ARTIFACTS
    )
    missing_required = [
        entry["relative_path"]
        for entry in artifacts
        if entry["required"] and entry["status"] != "pass"
    ]
    failed = [
        entry["relative_path"]
        for entry in artifacts
        if entry["exists"] and entry["status"] == "fail"
    ]
    consistency_checks = build_consistency_checks(root)
    failed_consistency = [check["name"] for check in consistency_checks if check["status"] == "fail"]
    artifact_date_tags = sorted({str(entry["source_date_tag"]) for entry in artifacts})
    return {
        "status": "pass" if not missing_required and not failed and not failed_consistency else "fail",
        "root": str(root),
        "date_tag": DATE_TAG,
        "current_audit_date_tag": CURRENT_AUDIT_DATE_TAG,
        "artifact_date_tags": artifact_date_tags,
        "date_tag_policy": {
            "canonical_signoff": DATE_TAG,
            "current_audit": CURRENT_AUDIT_DATE_TAG,
            "note": "Manifest intentionally combines canonical signoff artifacts with newer current-audit evidence.",
        },
        "date_tag_profile": {
            "canonical": DATE_TAG,
            "current_audit": CURRENT_AUDIT_DATE_TAG,
        },
        "required_count": len(REQUIRED_ARTIFACTS),
        "present_required_count": sum(1 for entry in artifacts if entry["required"] and entry["status"] == "pass"),
        "optional_count": len(OPTIONAL_ARTIFACTS),
        "present_optional_count": sum(1 for entry in artifacts if not entry["required"] and entry["exists"]),
        "missing_required": missing_required,
        "failed_artifacts": failed,
        "consistency_checks": consistency_checks,
        "failed_consistency_checks": failed_consistency,
        "artifacts": artifacts,
        "volatile_excluded": [
            {"id": artifact_id, "path": str(root / rel), "relative_path": rel, "reason": reason}
            for artifact_id, rel, reason in VOLATILE_ARTIFACTS
        ],
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Evidence Manifest",
        "",
        f"- status: `{manifest['status']}`",
        f"- root: `{manifest['root']}`",
        f"- required: `{manifest['present_required_count']}/{manifest['required_count']}`",
        f"- optional_present: `{manifest['present_optional_count']}/{manifest['optional_count']}`",
        "",
        "## Artifacts",
        "",
        "| ID | Required | Status | Size | SHA256 |",
        "|---|---:|---|---:|---|",
    ]
    for entry in manifest["artifacts"]:
        sha = str(entry.get("sha256", ""))
        size = entry.get("size_bytes", "")
        lines.append(
            f"| {entry['id']} | `{entry['required']}` | `{entry['status']}` | {size} | `{sha}` |"
        )
    lines.extend(
        [
            "",
            "## Date Tags",
            "",
            f"- canonical_signoff: `{manifest['date_tag_policy']['canonical_signoff']}`",
            f"- current_audit: `{manifest['date_tag_policy']['current_audit']}`",
            f"- artifact_date_tags: `{', '.join(manifest['artifact_date_tags'])}`",
        ]
    )
    lines.extend(["", "## Volatile Excluded", ""])
    for entry in manifest["volatile_excluded"]:
        lines.append(f"- `{entry['relative_path']}`: {entry['reason']}")
    lines.extend(["", "## Consistency Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in manifest["consistency_checks"]:
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
    parser = argparse.ArgumentParser(description="Write final HGTXR evidence manifest.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(manifest))
    print(
        f"[final-evidence-manifest] status={manifest['status']} "
        f"required={manifest['present_required_count']}/{manifest['required_count']}"
    )
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
