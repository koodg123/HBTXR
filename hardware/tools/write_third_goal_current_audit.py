#!/usr/bin/env python3
"""Write a current third-goal requirement audit from local signoff evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        hardware = root
        hgtxr = root.parent
    else:
        hgtxr = root
        hardware = root / "hardware"
    return hgtxr, hardware


def rel(hardware: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(hardware.resolve()).as_posix()
    except ValueError:
        return str(path)


def file_evidence(hardware: Path, relative_path: str, *, required: bool = True) -> dict[str, Any]:
    path = hardware / relative_path
    status = "pass" if path.exists() else ("missing" if required else "absent-optional")
    return {
        "path": relative_path,
        "exists": path.exists(),
        "required": required,
        "status": status,
    }


def source_contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(errors="replace")


def closeout_summary(hardware: Path, *, vref_required: bool) -> dict[str, Any]:
    suffix = "_vref_required" if vref_required else ""
    rel_path = f"generated/signoff/final_unblock_closeout_packet{suffix}_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "remaining_blockers": [],
            "blocker_count": 0,
        }
    payload = load_json(path)
    blockers = payload.get("remaining_blockers", [])
    if not isinstance(blockers, list):
        blockers = []
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "remaining_blockers": [str(item) for item in blockers],
        "blocker_count": len(blockers),
        "vref_required": vref_required,
        "vref_successor_gate": payload.get("vref_successor_gate", {})
        if isinstance(payload.get("vref_successor_gate"), dict)
        else {},
        "qkv_uram_successor_gate": payload.get("qkv_uram_successor_gate", {})
        if isinstance(payload.get("qkv_uram_successor_gate"), dict)
        else {},
        "qkv_uram_command_count": len(
            payload.get("operator_commands", {}).get("qkv_uram_successor", [])
            if isinstance(payload.get("operator_commands"), dict)
            else []
        ),
    }


def vref_successor_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path}
    payload = load_json(path)
    physical = payload.get("physical_smoke_result", {})
    if not isinstance(physical, dict):
        physical = {}
    projection = payload.get("recommended_c3b_projection", {})
    if not isinstance(projection, dict):
        projection = {}
    expected_raw = payload.get("expected_raw")
    if expected_raw is None:
        expected = payload.get("expected_output", {})
        if not isinstance(expected, dict):
            expected = {}
        expected_raw = expected.get("out_raw")
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "candidate": payload.get("candidate"),
        "recommended_resource_variant": payload.get("recommended_resource_variant"),
        "expected_out_raw": expected_raw,
        "projection_status": projection.get("status"),
        "projection_pending": projection.get("pending", []),
        "physical_smoke_status": physical.get("status"),
        "physical_smoke_result_json": physical.get("result_json"),
        "physical_smoke_preset": physical.get("preset"),
    }


def qkv_uram_successor_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/vref_p0_qkv_uram_cache_successor_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path}
    payload = load_json(path)
    csynth = payload.get("csynth", {}) if isinstance(payload.get("csynth"), dict) else {}
    resources = csynth.get("resources", {}) if isinstance(csynth.get("resources"), dict) else {}
    promotion = payload.get("promotion", {}) if isinstance(payload.get("promotion"), dict) else {}
    physical_smoke = payload.get("physical_smoke", {}) if isinstance(payload.get("physical_smoke"), dict) else {}
    pynq_plumbing = payload.get("pynq_plumbing", {}) if isinstance(payload.get("pynq_plumbing"), dict) else {}
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "candidate": payload.get("candidate"),
        "macro": payload.get("macro"),
        "resource_policy": payload.get("resource_policy"),
        "latency_cycles": csynth.get("latency_cycles"),
        "estimated_clock_ns": csynth.get("estimated_clock_ns"),
        "resources": resources,
        "promotion_status": promotion.get("status"),
        "remaining": promotion.get("remaining", []),
        "physical_smoke_status": physical_smoke.get("status"),
        "physical_smoke_result_json": physical_smoke.get("result_json"),
        "pynq_plumbing_status": pynq_plumbing.get("status"),
        "dry_run_import_command": pynq_plumbing.get("dry_run_import_command", ""),
        "remote_dry_run_command": pynq_plumbing.get("remote_dry_run_command", ""),
        "expected_result_json": pynq_plumbing.get("expected_result_json", ""),
        "pass_count": payload.get("pass_count"),
        "check_count": payload.get("check_count"),
        "fail_count": payload.get("fail_count"),
    }


def smoke_discovery_summary(hardware: Path, rel_path: str) -> dict[str, Any]:
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "preset": None,
            "candidate_count": 0,
            "pass_count": 0,
            "recommended_candidate": None,
            "dry_run_import_command": "",
            "import_command": "",
        }
    payload = load_json(path)
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "preset": payload.get("preset"),
        "candidate_count": payload.get("candidate_count"),
        "pass_count": payload.get("pass_count"),
        "recommended_candidate": payload.get("recommended_candidate"),
        "dry_run_import_command": payload.get("dry_run_import_command", ""),
        "import_command": payload.get("import_command", ""),
    }


def vref_smoke_discovery_summary(hardware: Path) -> dict[str, Any]:
    return smoke_discovery_summary(
        hardware,
        "generated/signoff/vref_successor_smoke_candidate_discovery_2026_06_10.json",
    )


def smoke_candidate_discovery_summary(hardware: Path) -> dict[str, Any]:
    return {
        "c3b_generic": smoke_discovery_summary(
            hardware,
            "generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.json",
        ),
        "vref_generic": smoke_discovery_summary(
            hardware,
            "generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.json",
        ),
        "vref_runner": vref_smoke_discovery_summary(hardware),
        "qkv_runner": smoke_discovery_summary(
            hardware,
            "generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.json",
        ),
    }


def source_audit_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/third_goal_source_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path}
    payload = load_json(path)
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "required_count": payload.get("required_count"),
        "source_count": payload.get("source_count"),
        "missing_required": payload.get("missing_required", []),
    }


def final_signoff_runner_summary(hardware: Path) -> dict[str, Any]:
    rel_path = "generated/signoff/third_goal_final_signoff_run_2026_06_10.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "qkv_uram_required_for_final_signoff": False,
            "qkv_uram_import_status": "missing",
            "qkv_uram_remote_status": "missing",
            "qkv_uram_remote_execute": False,
            "pynq_c3b_smoke_discovery_status": "missing",
            "pynq_vref_smoke_discovery_status": "missing",
            "qkv_uram_smoke_discovery_status": "missing",
        }
    payload = load_json(path)
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "qkv_uram_required_for_final_signoff": bool(payload.get("qkv_uram_required_for_final_signoff", False)),
        "qkv_uram_import_status": str(payload.get("qkv_uram_import_status", "unknown")),
        "qkv_uram_remote_status": str(payload.get("qkv_uram_remote_status", "unknown")),
        "qkv_uram_remote_execute": bool(payload.get("qkv_uram_remote_execute", False)),
        "pynq_c3b_smoke_discovery_status": str(payload.get("pynq_c3b_smoke_discovery_status", "unknown")),
        "pynq_vref_smoke_discovery_status": str(payload.get("pynq_vref_smoke_discovery_status", "unknown")),
        "qkv_uram_smoke_discovery_status": str(payload.get("qkv_uram_smoke_discovery_status", "unknown")),
    }


def final_blocker_closure_summary(hardware: Path) -> dict[str, Any]:
    rel_path = "generated/signoff/final_blocker_closure_readiness_2026_06_10.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "current_ready": False,
            "candidate_ready": False,
            "missing_current_blockers": [],
            "dry_run_final_runner_command": "",
            "final_runner_command": "",
        }
    payload = load_json(path)
    missing = payload.get("missing_current_blockers", [])
    if not isinstance(missing, list):
        missing = []
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "current_ready": bool(payload.get("current_ready")),
        "candidate_ready": bool(payload.get("candidate_ready")),
        "missing_current_blockers": [str(item) for item in missing],
        "dry_run_final_runner_command": payload.get("dry_run_final_runner_command", ""),
        "final_runner_command": payload.get("final_runner_command", ""),
    }


def final_unblock_intake_summary(hardware: Path) -> dict[str, Any]:
    rel_path = "generated/signoff/final_unblock_intake_2026_06_10.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "candidate_ready": False,
            "would_clear_all": False,
            "remaining_blockers": [],
            "next_action": "",
            "dry_run_final_runner_command": "",
            "active_final_runner_command": "",
        }
    payload = load_json(path)
    candidate_audit = payload.get("candidate_audit", {})
    if not isinstance(candidate_audit, dict):
        candidate_audit = {}
    closure = payload.get("closure_readiness", {})
    if not isinstance(closure, dict):
        closure = {}
    remaining = candidate_audit.get("remaining_blockers", [])
    if not isinstance(remaining, list):
        remaining = []
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "candidate_ready": bool(closure.get("candidate_ready")),
        "would_clear_all": bool(candidate_audit.get("would_clear_all")),
        "remaining_blockers": [str(item) for item in remaining],
        "next_action": str(payload.get("next_action", "")),
        "dry_run_final_runner_command": payload.get("dry_run_final_runner_command", ""),
        "active_final_runner_command": payload.get("active_final_runner_command", ""),
    }


def final_unblock_commands_summary(hardware: Path) -> dict[str, Any]:
    rel_path = "generated/signoff/final_unblock_commands_2026_06_10.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "remaining_blockers": [],
            "section_count": 0,
            "readiness_status": "missing",
            "xr_vits_packet_status": "missing",
        }
    payload = load_json(path)
    blockers = payload.get("remaining_blockers", [])
    if not isinstance(blockers, list):
        blockers = []
    sections = payload.get("sections", [])
    if not isinstance(sections, list):
        sections = []
    xr_policy_integrity = (
        payload.get("xr_vits_policy_integrity", {})
        if isinstance(payload.get("xr_vits_policy_integrity"), dict)
        else {}
    )
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "remaining_blockers": [str(item) for item in blockers],
        "section_count": len(sections),
        "section_ids": [str(section.get("id")) for section in sections if isinstance(section, dict) and section.get("id")],
        "has_qkv_uram_u5": any(isinstance(section, dict) and section.get("id") == "U5" for section in sections),
        "readiness_status": str(payload.get("readiness_status", "unknown")),
        "xr_vits_packet_status": str(payload.get("xr_vits_packet_status", "unknown")),
        "c3b_smoke_contract": payload.get("c3b_smoke_contract", {})
        if isinstance(payload.get("c3b_smoke_contract"), dict)
        else {},
        "xr_vits_policy_integrity": policy_integrity_summary(xr_policy_integrity),
    }


def final_unblock_candidate_audit_summary(hardware: Path) -> dict[str, Any]:
    rel_path = "generated/signoff/final_unblock_candidate_audit_2026_06_10.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "would_clear_all": False,
            "remaining_blockers": [],
            "c3b_status": "missing",
            "xr_vits_status": "missing",
        }
    payload = load_json(path)
    remaining = payload.get("remaining_blockers", [])
    if not isinstance(remaining, list):
        remaining = []
    c3b = payload.get("c3b_smoke", {})
    if not isinstance(c3b, dict):
        c3b = {}
    xr_vits = payload.get("xr_vits", {})
    if not isinstance(xr_vits, dict):
        xr_vits = {}
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "would_clear_all": bool(payload.get("would_clear_all")),
        "remaining_blockers": [str(item) for item in remaining],
        "c3b_status": str(c3b.get("status", "unknown")),
        "c3b_would_clear": bool(c3b.get("would_clear")),
        "xr_vits_status": str(xr_vits.get("status", "unknown")),
        "xr_vits_mode": str(xr_vits.get("mode", "")),
        "xr_vits_would_clear": bool(xr_vits.get("would_clear")),
    }


def final_operator_handoff_summary(hardware: Path) -> dict[str, Any]:
    rel_path = "generated/signoff/final_operator_handoff_2026_06_10.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "remaining_blockers": [],
            "board_smoke_status": "missing",
            "xr_vits_candidate_status": "missing",
            "evidence_manifest_status": "missing",
        }
    payload = load_json(path)
    remaining = payload.get("remaining_blockers", [])
    if not isinstance(remaining, list):
        remaining = []
    board = payload.get("board_smoke", {})
    if not isinstance(board, dict):
        board = {}
    xr_vits = payload.get("xr_vits", {})
    if not isinstance(xr_vits, dict):
        xr_vits = {}
    xr_resolution = (
        xr_vits.get("reference_resolution", {})
        if isinstance(xr_vits.get("reference_resolution"), dict)
        else {}
    )
    xr_policy_integrity = (
        xr_resolution.get("policy_integrity", {})
        if isinstance(xr_resolution.get("policy_integrity"), dict)
        else {}
    )
    evidence = payload.get("final_evidence_manifest_contract", {})
    if not isinstance(evidence, dict):
        evidence = {}
    safety = payload.get("safety", {})
    if not isinstance(safety, dict):
        safety = {}
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "remaining_blockers": [str(item) for item in remaining],
        "board_smoke_status": str(board.get("status", "unknown")),
        "board_smoke_ready": bool(board.get("ready")),
        "xr_vits_candidate_status": str(xr_vits.get("candidate_status", "unknown")),
        "xr_vits_would_clear": bool(xr_vits.get("would_clear")),
        "evidence_manifest_status": str(evidence.get("status", "unknown")),
        "evidence_manifest_required_count": evidence.get("required_count"),
        "evidence_manifest_present_required_count": evidence.get("present_required_count"),
        "requires_operator_action": bool(safety.get("requires_operator_action")),
        "xr_vits_policy_integrity": policy_integrity_summary(xr_policy_integrity),
    }


def final_operator_handoff_validation_summary(hardware: Path) -> dict[str, Any]:
    rel_path = "generated/signoff/final_operator_handoff_validation_2026_06_10.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "pass_count": 0,
            "fail_count": 0,
            "check_count": 0,
        }
    payload = load_json(path)
    checks = payload.get("checks", [])
    policy_checks = []
    if isinstance(checks, list):
        policy_checks = [
            str(check.get("name"))
            for check in checks
            if isinstance(check, dict)
            and (
                "xr_policy" in str(check.get("name", ""))
                or "xr_legacy_policy" in str(check.get("name", ""))
            )
        ]
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "pass_count": payload.get("pass_count"),
        "fail_count": payload.get("fail_count"),
        "check_count": payload.get("check_count"),
        "policy_check_count": len(policy_checks),
        "policy_checks": policy_checks,
        "safety": payload.get("safety", {}) if isinstance(payload.get("safety"), dict) else {},
    }


def policy_integrity_summary(integrity: dict[str, Any]) -> dict[str, Any]:
    validation = integrity.get("validation", {}) if isinstance(integrity.get("validation"), dict) else {}
    fields = integrity.get("required_policy_fields", [])
    if not isinstance(fields, list):
        fields = []
    required_fields = {
        "candidate_audit_fingerprint",
        "candidate_audit_recommendation_snapshot",
        "candidate_audit_meta",
        "approval_event",
        "policy_fingerprint",
    }
    return {
        "required": integrity.get("required") is True,
        "policy_path": integrity.get("policy_path", ""),
        "candidate_audit": integrity.get("candidate_audit", ""),
        "generator": integrity.get("generator", ""),
        "validator": integrity.get("validator", ""),
        "legacy_policy_clears_final_signoff": integrity.get("legacy_policy_clears_final_signoff"),
        "required_policy_fields": [str(field) for field in fields],
        "required_policy_fields_complete": required_fields.issubset({str(field) for field in fields}),
        "policy_exists": bool(integrity.get("policy_exists", False)),
        "validation_status": str(validation.get("status", "not-embedded")),
        "candidate_audit_fingerprint_present": bool(integrity.get("candidate_audit_fingerprint")),
        "policy_fingerprint_present": bool(integrity.get("policy_fingerprint")),
    }


def xr_vits_unblock_packet_summary(hardware: Path) -> dict[str, Any]:
    rel_path = "generated/signoff/xr_vits_unblock_packet_2026_06_10.json"
    path = hardware / rel_path
    if not path.exists():
        return {
            "status": "missing",
            "path": rel_path,
            "requested_path_exists": False,
            "active_policy_approved": False,
            "option_count": 0,
            "next_required_action": "",
        }
    payload = load_json(path)
    options = payload.get("options", [])
    if not isinstance(options, list):
        options = []
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "requested_path": payload.get("requested_path"),
        "requested_path_exists": bool(payload.get("requested_path_exists")),
        "active_policy_approved": bool(payload.get("active_policy_approved")),
        "option_count": len(options),
        "next_required_action": str(payload.get("next_required_action", "")),
        "recommendation": payload.get("recommendation", {})
        if isinstance(payload.get("recommendation"), dict)
        else {},
    }


def hgpipe_operator_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/hgpipe_operator_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path, "operators": []}
    payload = load_json(path)
    operators = payload.get("operators", [])
    if not isinstance(operators, list):
        operators = []
    contract_summary = payload.get("contract_summary", {})
    if not isinstance(contract_summary, dict):
        contract_summary = {}
    residual_risk = payload.get("residual_risk", [])
    if not isinstance(residual_risk, list):
        residual_risk = []
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "operators": operators,
        "contract_summary": contract_summary,
        "residual_risk": residual_risk,
    }


def xr_vits_gate_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/xr_vits_gate_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path, "resolution_mode": "missing"}
    payload = load_json(path)
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "resolution_mode": payload.get("resolution_mode", "unknown"),
        "requested_path": payload.get("requested_path"),
        "remaining_blockers": payload.get("remaining_blockers", []),
        "candidate_audit": payload.get("candidate_audit", {})
        if isinstance(payload.get("candidate_audit"), dict)
        else {},
        "active_policy": payload.get("active_policy", {})
        if isinstance(payload.get("active_policy"), dict)
        else {},
    }


def c3b_smoke_gate_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/c3b_physical_smoke_gate_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path, "canonical_result": {"status": "missing"}}
    payload = load_json(path)
    canonical = payload.get("canonical_result", {})
    if not isinstance(canonical, dict):
        canonical = {"status": "unknown"}
    bundle = payload.get("bundle", {})
    if not isinstance(bundle, dict):
        bundle = {}
    session = payload.get("session", {})
    if not isinstance(session, dict):
        session = {}
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "canonical_result": canonical,
        "bundle": bundle,
        "session": session,
        "remaining_blockers": payload.get("remaining_blockers", [])
        if isinstance(payload.get("remaining_blockers"), list)
        else [],
    }


def req2_spec_subagent_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/req2_spec_subagent_gate_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path}
    payload = load_json(path)
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "spec_kit": payload.get("spec_kit", {}) if isinstance(payload.get("spec_kit"), dict) else {},
        "subagents": payload.get("subagents", {}) if isinstance(payload.get("subagents"), dict) else {},
        "plan_spec": payload.get("plan_spec", {}) if isinstance(payload.get("plan_spec"), dict) else {},
    }


def req1_environment_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/req1_environment_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path, "check_count": 0, "fail_count": None}
    payload = load_json(path)
    environment = payload.get("environment", {}) if isinstance(payload.get("environment"), dict) else {}
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "check_count": payload.get("check_count"),
        "pass_count": payload.get("pass_count"),
        "fail_count": payload.get("fail_count"),
        "environment": environment,
    }


def req5_q4q8_swhw_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/req5_q4q8_swhw_match_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path}
    payload = load_json(path)
    precision = payload.get("precision", {}) if isinstance(payload.get("precision"), dict) else {}
    manifest = payload.get("manifest", {}) if isinstance(payload.get("manifest"), dict) else {}
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "precision": precision,
        "manifest_status": manifest.get("status"),
        "pass_count": payload.get("pass_count"),
        "fail_count": payload.get("fail_count"),
        "claim_supported": payload.get("claim_supported", ""),
        "remaining_scope": payload.get("remaining_scope", [])
        if isinstance(payload.get("remaining_scope"), list)
        else [],
    }


def req6_parameterization_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/req6_parameterization_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path}
    payload = load_json(path)
    knobs = payload.get("knobs", {}) if isinstance(payload.get("knobs"), dict) else {}
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "config_macros": knobs.get("config_macros", {})
        if isinstance(knobs.get("config_macros"), dict)
        else {},
        "check_count": payload.get("check_count"),
        "pass_count": payload.get("pass_count"),
        "fail_count": payload.get("fail_count"),
        "coverage": payload.get("coverage", {}) if isinstance(payload.get("coverage"), dict) else {},
    }


def req9_deit_image_summary(hardware: Path) -> dict[str, Any]:
    rel_path = f"generated/signoff/req9_deit_image_reference_audit_{DATE_TAG}.json"
    path = hardware / rel_path
    if not path.exists():
        return {"status": "missing", "path": rel_path, "check_count": 0, "fail_count": None}
    payload = load_json(path)
    return {
        "status": str(payload.get("status", "unknown")),
        "path": rel_path,
        "check_count": payload.get("check_count"),
        "pass_count": payload.get("pass_count"),
        "fail_count": payload.get("fail_count"),
        "requested_user_path": payload.get("requested_user_path", ""),
        "normalized_requested_path": payload.get("normalized_requested_path", ""),
        "requested_image": payload.get("requested_image", {})
        if isinstance(payload.get("requested_image"), dict)
        else {},
        "hardware_docs_copy": payload.get("hardware_docs_copy", {})
        if isinstance(payload.get("hardware_docs_copy"), dict)
        else {},
    }


def evidence_item(path: str, kind: str, note: str = "") -> dict[str, str]:
    return {"path": path, "kind": kind, "note": note}


def evidence_paths(items: list[dict[str, str]]) -> list[str]:
    return [item["path"] for item in items]


def requirement(
    req_id: str,
    title: str,
    status: str,
    evidence: list[str] | None = None,
    missing: list[str] | None = None,
    note: str = "",
    evidence_items: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if evidence_items is None:
        evidence_items = [evidence_item(path, "doc-reported") for path in (evidence or [])]
    return {
        "id": req_id,
        "title": title,
        "status": status,
        "evidence": evidence_paths(evidence_items),
        "evidence_items": evidence_items,
        "missing": missing or [],
        "note": note,
    }


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    default_closeout = closeout_summary(hardware, vref_required=False)
    vref_required_closeout = closeout_summary(hardware, vref_required=True)
    vref = vref_successor_summary(hardware)
    qkv_uram_successor = qkv_uram_successor_summary(hardware)
    vref_discovery = vref_smoke_discovery_summary(hardware)
    smoke_discovery = smoke_candidate_discovery_summary(hardware)
    source_audit = source_audit_summary(hardware)
    final_runner_summary = final_signoff_runner_summary(hardware)
    if final_runner_summary.get("pynq_c3b_smoke_discovery_status") == "not-run":
        c3b_discovery = smoke_discovery.get("c3b_generic", {}) if isinstance(smoke_discovery, dict) else {}
        if isinstance(c3b_discovery, dict):
            final_runner_summary["pynq_c3b_smoke_discovery_status"] = str(c3b_discovery.get("status", "not-run"))
    if final_runner_summary.get("pynq_vref_smoke_discovery_status") == "not-run":
        vref_generic_discovery = smoke_discovery.get("vref_generic", {}) if isinstance(smoke_discovery, dict) else {}
        if isinstance(vref_generic_discovery, dict):
            final_runner_summary["pynq_vref_smoke_discovery_status"] = str(
                vref_generic_discovery.get("status", "not-run")
            )
    if final_runner_summary.get("qkv_uram_smoke_discovery_status") == "not-run":
        qkv_discovery = smoke_discovery.get("qkv_runner", {}) if isinstance(smoke_discovery, dict) else {}
        if isinstance(qkv_discovery, dict):
            final_runner_summary["qkv_uram_smoke_discovery_status"] = str(qkv_discovery.get("status", "not-run"))
    blocker_closure = final_blocker_closure_summary(hardware)
    unblock_intake = final_unblock_intake_summary(hardware)
    unblock_commands = final_unblock_commands_summary(hardware)
    unblock_candidate_audit = final_unblock_candidate_audit_summary(hardware)
    operator_handoff = final_operator_handoff_summary(hardware)
    operator_handoff_validation = final_operator_handoff_validation_summary(hardware)
    command_policy_integrity = unblock_commands.get("xr_vits_policy_integrity", {})
    handoff_policy_integrity = operator_handoff.get("xr_vits_policy_integrity", {})
    policy_integrity_consistent = (
        isinstance(command_policy_integrity, dict)
        and isinstance(handoff_policy_integrity, dict)
        and command_policy_integrity.get("required") is True
        and handoff_policy_integrity.get("required") is True
        and command_policy_integrity.get("required_policy_fields_complete") is True
        and handoff_policy_integrity.get("required_policy_fields_complete") is True
        and command_policy_integrity.get("legacy_policy_clears_final_signoff") is False
        and handoff_policy_integrity.get("legacy_policy_clears_final_signoff") is False
        and operator_handoff_validation.get("policy_check_count", 0) >= 7
    )
    xr_vits_unblock = xr_vits_unblock_packet_summary(hardware)
    hgpipe_operator_audit = hgpipe_operator_summary(hardware)
    xr_vits_gate = xr_vits_gate_summary(hardware)
    c3b_smoke_gate = c3b_smoke_gate_summary(hardware)
    req2_gate = req2_spec_subagent_summary(hardware)
    req1_environment = req1_environment_summary(hardware)
    req5_q4q8 = req5_q4q8_swhw_summary(hardware)
    req6_parameterization = req6_parameterization_summary(hardware)
    req9_deit_image = req9_deit_image_summary(hardware)

    docs = {
        "master": file_evidence(hardware, "docs/Master-Plan.md"),
        "sub": file_evidence(hardware, "docs/Sub-Plan.md"),
        "spec": file_evidence(hardware, "docs/Spec.md"),
        "execution": file_evidence(hardware, "docs/Execution.md"),
        "validation": file_evidence(hardware, "docs/Validation.md"),
        "progress": file_evidence(hardware, "docs/track/PROGRESS.md"),
        "requirements": file_evidence(hardware, f"docs/track/THIRD_GOAL_REQUIREMENTS_{DATE_TAG}.md"),
    }
    all_plan_docs_exist = all(item["exists"] for item in docs.values())
    final_runner = hardware / "tools" / "run_third_goal_final_signoff.py"
    preflight = hardware / "tools" / "check_third_goal_preflight.py"
    config = hardware / "configs" / "zcu104_e2e_q4w8a_defines.h"
    hgpipe_analysis = hardware / "docs" / "SRC_CASE_MODULE_GUIDE.md"
    exact_xr_vits = hgtxr.parent.parent / "XR-VITs"
    c3b_smoke = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
    vref_smoke = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"

    default_blockers = set(default_closeout["remaining_blockers"])
    vref_required_blockers = set(vref_required_closeout["remaining_blockers"])

    items = [
        requirement(
            "0",
            "Analyze plan/progress/HANDOVER/docs and continue work",
            "reflected" if source_audit["status"] == "pass" else "partial",
            missing=[entry["path"] for entry in docs.values() if not entry["exists"]]
            + ([] if source_audit["status"] == "pass" else ["direct HANDOVER/source audit artifact"]),
            note=(
                f"Source audit status={source_audit['status']}, "
                f"required={source_audit.get('required_count')}, sources={source_audit.get('source_count')}."
            ),
            evidence_items=[
                evidence_item(entry["path"], "file-exists" if entry["exists"] else "external-blocker")
                for entry in docs.values()
            ]
            + [evidence_item(source_audit["path"], "validated-by-tool")],
        ),
        requirement(
            "1",
            "Ubuntu Linux and /tools/Xilinx path changes",
            "reflected" if req1_environment["status"] == "pass" else "partial",
            evidence_items=[
                evidence_item(req1_environment["path"], "validated-by-tool"),
                evidence_item("tools/check_third_goal_preflight.py", "validated-by-tool"),
                evidence_item("docs/Validation.md", "doc-reported"),
            ],
            missing=[] if req1_environment["status"] == "pass" else ["Req1 environment audit"],
            note=(
                f"Req1 environment audit status={req1_environment['status']}; "
                f"checks={req1_environment.get('pass_count')}/{req1_environment.get('check_count')}; "
                f"os={req1_environment.get('environment', {}).get('os_release', {}).get('PRETTY_NAME', 'unknown')}; "
                f"vitis={req1_environment.get('environment', {}).get('vitis_hls', 'missing')}; "
                f"vivado={req1_environment.get('environment', {}).get('vivado', 'missing')}."
            ),
        ),
        requirement(
            "2",
            "Plan/spec first and use GPT5.3-Codex-Spark where possible",
            "reflected"
            if req2_gate["status"].startswith("pass")
            else ("partial" if all_plan_docs_exist else "missing"),
            evidence_items=[
                evidence_item("docs/Master-Plan.md", "file-exists"),
                evidence_item("docs/Sub-Plan.md", "file-exists"),
                evidence_item("docs/Spec.md", "file-exists"),
                evidence_item("docs/Execution.md", "doc-reported"),
                evidence_item(req2_gate["path"], "validated-by-tool"),
            ],
            missing=[],
            note=(
                "Req2 gate status="
                f"{req2_gate['status']}; "
                f"spec_kit_available={req2_gate.get('spec_kit', {}).get('available')}; "
                f"manual_fallback={req2_gate.get('spec_kit', {}).get('manual_fallback_recorded')}; "
                f"spark_first={req2_gate.get('subagents', {}).get('spark_first_recorded')}; "
                f"gpt55_fallback={req2_gate.get('subagents', {}).get('gpt55_fallback_recorded')}."
            ),
        ),
        requirement(
            "3",
            "Prioritize ZCU104 cyclic hardware accelerator baseline",
            "reflected",
            evidence_items=[
                evidence_item("configs/sweeps/zcu104_cyclic_transformer_sweep.yaml", "file-exists"),
                evidence_item("generated/signoff/c3b_protection_checklist_2026_06_16.md", "validated-by-tool"),
                evidence_item(default_closeout["path"], "validated-by-tool"),
            ],
            missing=["pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"]
            if not c3b_smoke.exists()
            else [],
            note="C3b prioritization is reflected; physical board result is a separate final hardware-signoff gate.",
        ),
        requirement(
            "4",
            "Run paper ViT model E2E",
            "reflected" if c3b_smoke_gate["status"] == "pass" else "partial",
            evidence_items=[
                evidence_item("hls/tb/tb_hgtxr_e2e_axis_top.cpp", "file-exists"),
                evidence_item("hls/tb/tb_hgtxr_e2e_m_axi_top.cpp", "file-exists"),
                evidence_item("pynq/hgtxr/run_e2e_axis_dma_smoke.py", "file-exists"),
                evidence_item(c3b_smoke_gate["path"], "validated-by-tool"),
                evidence_item(default_closeout["path"], "validated-by-tool"),
            ]
            + (
                [evidence_item("pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json", "external-blocker")]
                if not c3b_smoke.exists()
                else []
            ),
            missing=["C3b ZCU104 physical smoke result"] if not c3b_smoke.exists() else [],
            note=(
                "C3b smoke gate status="
                f"{c3b_smoke_gate['status']}; "
                f"canonical_result={c3b_smoke_gate['canonical_result'].get('status')}; "
                f"bundle={c3b_smoke_gate['bundle'].get('status')}; "
                f"session={c3b_smoke_gate['session'].get('status')}."
            ),
        ),
        requirement(
            "5",
            "Q4 weights and Q8 activations with SW/HW match",
            "reflected" if req5_q4q8["status"] == "pass" else "partial",
            evidence_items=[
                evidence_item("configs/zcu104_e2e_q4w8a_defines.h", "file-exists"),
                evidence_item(req5_q4q8["path"], "validated-by-tool"),
                evidence_item(
                    "generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.md",
                    "validated-by-tool",
                ),
            ],
            missing=[] if req5_q4q8["status"] == "pass" else ["Req5 Q4/Q8 SW-HW match audit"],
            note=(
                f"Req5 audit status={req5_q4q8['status']}; "
                f"precision={req5_q4q8.get('precision')}; "
                f"VREF successor expected raw={vref.get('expected_out_raw')}. "
                "Physical smoke remains Req4/final gate."
            ),
        ),
        requirement(
            "6",
            "Parameterize tiling, parallelism, bus width, bit width, buffer size, FIFO depth",
            "reflected"
            if config.exists()
            and req6_parameterization["status"] == "pass"
            and source_contains(final_runner, "--require-vref-successor-physical-smoke")
            and source_contains(final_runner, "--vref-successor-remote-dir")
            else "partial",
            evidence_items=[
                evidence_item("configs/zcu104_e2e_q4w8a_defines.h", "file-exists"),
                evidence_item(req6_parameterization["path"], "validated-by-tool"),
                evidence_item("tools/run_third_goal_final_signoff.py", "validated-by-tool"),
                evidence_item(qkv_uram_successor["path"], "validated-by-tool"),
            ],
            missing=[] if req6_parameterization["status"] == "pass" else ["Req6 parameterization audit"],
            note=(
                "Req6 parameterization audit status="
                f"{req6_parameterization['status']}, "
                f"checks={req6_parameterization.get('pass_count')}/{req6_parameterization.get('check_count')}; "
                "QKV URAM successor status="
                f"{qkv_uram_successor['status']}; "
                f"resources={qkv_uram_successor.get('resources')}; "
                f"remaining={qkv_uram_successor.get('remaining')}."
            ),
        ),
        requirement(
            "7",
            "Analyze HG-PIPE and reflect it for ZCU104 cyclic accelerator",
            "reflected" if hgpipe_analysis.exists() else "partial",
            evidence_items=[
                evidence_item("docs/SRC_CASE_MODULE_GUIDE.md", "file-exists"),
                evidence_item("analysis/vit-accel/codebases/HG-PIPE/analysis.md", "file-exists"),
            ],
            missing=[] if hgpipe_analysis.exists() else ["docs/SRC_CASE_MODULE_GUIDE.md"],
        ),
        requirement(
            "8",
            "Implement LayerNorm, GeLU, Softmax, Quantization from HG-PIPE paper/code",
            (
                "reflected"
                if hgpipe_operator_audit["status"] == "pass"
                else (
                    "partial"
                    if (hardware / "hls" / "include" / "hgtxr_cyclic_math.hpp").exists()
                    and (hardware / "hls" / "include" / "hgtxr_e2e_vit.hpp").exists()
                    else "missing"
                )
            ),
            evidence_items=[
                evidence_item("hls/include/hgtxr_cyclic_math.hpp", "file-exists"),
                evidence_item("hls/include/hgtxr_e2e_vit.hpp", "file-exists"),
                evidence_item("tools/validate_hgpipe_lut_math.py", "validated-by-tool"),
                evidence_item(hgpipe_operator_audit["path"], "validated-by-tool"),
            ],
            missing=[]
            if hgpipe_operator_audit["status"] == "pass"
            else ["full HG-PIPE paper/code operator-equivalence proof across all blocks"],
            note=(
                "Operator audit status="
                f"{hgpipe_operator_audit['status']}, "
                f"checked_samples={hgpipe_operator_audit['contract_summary'].get('total_checked_samples')}, "
                f"ref_checks={hgpipe_operator_audit['contract_summary'].get('total_ref_checks')}. "
                "Residual risk remains local sampled/reference-vector equivalence, not formal exhaustive proof."
            )
            if hgpipe_operator_audit["status"] == "pass"
            else "",
        ),
        requirement(
            "9",
            "Refer to DeiT-Tiny C-Syn image",
            "reflected" if req9_deit_image["status"] == "pass" else "partial",
            evidence_items=[
                evidence_item(req9_deit_image["path"], "validated-by-tool"),
                evidence_item(
                    req9_deit_image.get("normalized_requested_path", ""),
                    "file-exists" if req9_deit_image["status"] == "pass" else "external-blocker",
                ),
                evidence_item("docs/DeiT-Tiny C-Syn Results.png", "file-exists"),
                evidence_item("tools/check_third_goal_preflight.py", "validated-by-tool"),
            ],
            missing=[]
            if req9_deit_image["status"] == "pass"
            else ["Req9 DeiT image reference audit"],
            note=(
                f"Req9 audit status={req9_deit_image['status']}; "
                f"checks={req9_deit_image.get('pass_count')}/{req9_deit_image.get('check_count')}; "
                f"requested_sha={req9_deit_image.get('requested_image', {}).get('sha256', '')}."
            ),
        ),
        requirement(
            "10",
            "Refer to XR-VIT experiment results/code",
            "reflected"
            if (hardware / "docs" / "legacy" / "legacy_experiment_analysis_2026_06_12.md").exists()
            and (hardware / "analysis" / "vit-accel").exists()
            and xr_vits_gate["status"] != "missing"
            else "partial",
            evidence_items=[
                evidence_item("docs/legacy/legacy_experiment_analysis_2026_06_12.md", "doc-reported"),
                evidence_item("analysis/vit-accel", "file-exists"),
                evidence_item(xr_vits_gate["path"], "validated-by-tool"),
            ],
            missing=[]
            if (hardware / "docs" / "legacy" / "legacy_experiment_analysis_2026_06_12.md").exists()
            and (hardware / "analysis" / "vit-accel").exists()
            and xr_vits_gate["status"] != "missing"
            else ["XR-VIT reference evidence or candidate audit"],
            note=(
                "Past XR-VIT evidence is organized; XR-VITs gate audit status="
                f"{xr_vits_gate['status']} / {xr_vits_gate['resolution_mode']}. "
                "Exact XR-VITs source/replacement approval remains tracked by Req11."
            ),
        ),
        requirement(
            "11",
            "Use /home/kjm26/project/PRJXR/XR-VITs HLS code for ZCU104 fit/low latency",
            "reflected"
            if xr_vits_gate["status"] in {"pass-exact", "pass-replacement-policy"}
            else "blocked",
            evidence_items=[
                evidence_item("generated/signoff/xr_vits_reference_resolution_2026_06_10.md", "validated-by-tool"),
                evidence_item(xr_vits_gate["path"], "validated-by-tool"),
                evidence_item(default_closeout["path"], "validated-by-tool"),
                evidence_item(unblock_commands["path"], "validated-by-tool"),
                evidence_item(operator_handoff["path"], "validated-by-tool"),
                evidence_item(operator_handoff_validation["path"], "validated-by-tool"),
                evidence_item(str(exact_xr_vits), "external-blocker"),
            ],
            missing=[]
            if xr_vits_gate["status"] in {"pass-exact", "pass-replacement-policy"}
            else ["/home/kjm26/project/PRJXR/XR-VITs or approved replacement policy"],
            note=(
                "XR-VITs gate audit status="
                f"{xr_vits_gate['status']}; mode={xr_vits_gate['resolution_mode']}; "
                f"requested_path={xr_vits_gate.get('requested_path')}; "
                f"operator_policy_integrity_consistent={policy_integrity_consistent}; "
                f"policy_validation={handoff_policy_integrity.get('validation_status')}."
            ),
        ),
    ]

    if "VREF-P0 successor physical smoke result" in vref_required_blockers or not vref_smoke.exists():
        vref_status = "pending-physical-smoke"
    else:
        vref_status = "pass"

    remaining_external_inputs = [
        str(c3b_smoke),
        str(exact_xr_vits),
        str(vref_smoke),
    ]
    remaining_external_input_details = [
        {
            "path": str(c3b_smoke),
            "kind": "canonical-board-smoke-json",
            "status": "missing" if not c3b_smoke.exists() else "present",
            "blocker_name": "C3b AXIS/DMA physical smoke result",
            "required_for_default_final_signoff": True,
        },
        {
            "path": str(exact_xr_vits),
            "kind": "requested-xr-vits-source-tree",
            "status": "missing" if not exact_xr_vits.exists() else "present",
            "blocker_name": "requested XR-VITs sibling",
            "required_for_default_final_signoff": True,
        },
        {
            "path": str(vref_smoke),
            "kind": "successor-board-smoke-json",
            "status": "missing" if not vref_smoke.exists() else "present",
            "blocker_name": "VREF-P0 successor physical smoke result",
            "required_for_default_final_signoff": False,
        },
    ]
    blocker_to_external_input = {
        "C3b AXIS/DMA physical smoke result": str(c3b_smoke),
        "requested XR-VITs sibling": str(exact_xr_vits),
        "VREF-P0 successor physical smoke result": str(vref_smoke),
    }
    blocked_external_input_paths_by_blocker = {
        blocker: blocker_to_external_input[blocker]
        for blocker in sorted(default_blockers)
        if blocker in blocker_to_external_input
    }
    blocked = [item for item in items if item["status"] == "blocked"]
    partial = [item for item in items if item["status"] == "partial"]
    status = "blocked-external" if blocked or default_closeout["blocker_count"] else ("partial" if partial else "pass")
    return {
        "status": status,
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "items": items,
        "summary": {
            "requirements": len(items),
            "reflected": len([item for item in items if item["status"] == "reflected"]),
            "partial": len(partial),
            "blocked": len(blocked),
            "default_closeout_blockers": default_closeout["blocker_count"],
            "vref_required_closeout_blockers": vref_required_closeout["blocker_count"],
            "evidence_classes": sorted(
                {
                    evidence["kind"]
                    for item in items
                    for evidence in item.get("evidence_items", [])
                    if isinstance(evidence, dict)
                }
            ),
        },
        "blocker_schema": {
            "external_blocker_names_source": "gate_modes.default.remaining_blockers",
            "remaining_external_inputs_kind": "absolute paths for missing or optional external inputs",
            "remaining_external_input_details_kind": "path records mapped to blocker_name and default signoff requirement",
        },
        "external_blocker_names": sorted(default_blockers),
        "blocked_external_inputs": list(blocked_external_input_paths_by_blocker.values()),
        "blocked_external_input_paths_by_blocker": blocked_external_input_paths_by_blocker,
        "gate_modes": {
            "default": default_closeout,
            "vref_required": vref_required_closeout,
            "vref_successor_status": vref_status,
            "vref_successor": vref,
            "qkv_uram_successor": qkv_uram_successor,
            "qkv_uram_runner": final_runner_summary,
            "vref_smoke_discovery": vref_discovery,
            "smoke_candidate_discovery": smoke_discovery,
            "final_blocker_closure": blocker_closure,
            "final_unblock_intake": unblock_intake,
            "final_unblock_commands": unblock_commands,
            "final_unblock_candidate_audit": unblock_candidate_audit,
            "final_operator_handoff": operator_handoff,
            "final_operator_handoff_validation": operator_handoff_validation,
            "xr_vits_unblock_packet": xr_vits_unblock,
            "xr_vits_policy_integrity": {
                "command_card": command_policy_integrity,
                "operator_handoff": handoff_policy_integrity,
                "operator_handoff_policy_check_count": operator_handoff_validation.get("policy_check_count", 0),
                "consistent": policy_integrity_consistent,
            },
        },
        "source_audit": source_audit,
        "qkv_uram_runner": final_runner_summary,
        "final_blocker_closure_readiness": blocker_closure,
        "final_unblock_intake": unblock_intake,
        "final_unblock_commands": unblock_commands,
        "final_unblock_candidate_audit": unblock_candidate_audit,
        "final_operator_handoff": operator_handoff,
        "final_operator_handoff_validation": operator_handoff_validation,
        "xr_vits_unblock_packet": xr_vits_unblock,
        "xr_vits_policy_integrity": {
            "command_card": command_policy_integrity,
            "operator_handoff": handoff_policy_integrity,
            "operator_handoff_policy_check_count": operator_handoff_validation.get("policy_check_count", 0),
            "consistent": policy_integrity_consistent,
        },
        "qkv_uram_successor": qkv_uram_successor,
        "req2_spec_subagent_gate_audit": req2_gate,
        "req5_q4q8_swhw_match_audit": req5_q4q8,
        "req6_parameterization_audit": req6_parameterization,
        "req9_deit_image_reference_audit": req9_deit_image,
        "req1_environment_audit": req1_environment,
        "hgpipe_operator_audit": hgpipe_operator_audit,
        "xr_vits_gate_audit": xr_vits_gate,
        "c3b_physical_smoke_gate_audit": c3b_smoke_gate,
        "remaining_external_inputs": remaining_external_inputs,
        "remaining_external_input_details": remaining_external_input_details,
        "pending_optional_or_internal_evidence": [
            vref_discovery["path"],
            smoke_discovery["c3b_generic"]["path"],
            smoke_discovery["vref_generic"]["path"],
            smoke_discovery["qkv_runner"]["path"],
            blocker_closure["path"],
            unblock_intake["path"],
            unblock_commands["path"],
            unblock_candidate_audit["path"],
            operator_handoff["path"],
            operator_handoff_validation["path"],
            xr_vits_unblock["path"],
        ],
        "safety": {
            "executes_network": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Third Goal Current Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- date_tag: `{audit['date_tag']}`",
        f"- hardware_root: `{audit['hardware_root']}`",
        "",
        "## Summary",
    ]
    summary = audit["summary"]
    for key in [
        "requirements",
        "reflected",
        "partial",
        "blocked",
        "default_closeout_blockers",
        "vref_required_closeout_blockers",
    ]:
        lines.append(f"- {key}: `{summary[key]}`")
    lines.append(f"- evidence_classes: `{', '.join(summary['evidence_classes'])}`")
    lines.extend(["", "## Requirement Status", "", "| ID | Status | Missing | Evidence |"])
    lines.append("|---|---|---|---|")
    for item in audit["items"]:
        missing = "<br>".join(item["missing"]) if item["missing"] else "-"
        evidence = "<br>".join(
            f"{evidence_item['path']} ({evidence_item['kind']})"
            for evidence_item in item.get("evidence_items", [])
        ) or "-"
        lines.append(f"| {item['id']} | `{item['status']}` | {missing} | {evidence} |")
    lines.extend(["", "## Gate Modes"])
    default = audit["gate_modes"]["default"]
    vref_required = audit["gate_modes"]["vref_required"]
    lines.append(f"- default_closeout: `{default['status']}`, blockers `{default['blocker_count']}`")
    for blocker in default["remaining_blockers"]:
        lines.append(f"  - {blocker}")
    lines.append(
        f"- vref_required_closeout: `{vref_required['status']}`, blockers `{vref_required['blocker_count']}`"
    )
    for blocker in vref_required["remaining_blockers"]:
        lines.append(f"  - {blocker}")
    lines.append(f"- vref_successor_status: `{audit['gate_modes']['vref_successor_status']}`")
    qkv_uram = audit["gate_modes"].get("qkv_uram_successor", {})
    if isinstance(qkv_uram, dict):
        lines.append(
            f"- qkv_uram_successor: `{qkv_uram.get('status')}`, "
            f"resources `{qkv_uram.get('resources')}`, remaining `{qkv_uram.get('remaining')}`"
        )
        lines.append(
            f"- qkv_uram_physical_smoke: `{qkv_uram.get('physical_smoke_status')}`, "
            f"pynq_plumbing `{qkv_uram.get('pynq_plumbing_status')}`"
        )
    qkv_runner = audit["gate_modes"].get("qkv_uram_runner", {})
    if isinstance(qkv_runner, dict):
        lines.append(
            f"- qkv_uram_runner: import `{qkv_runner.get('qkv_uram_import_status')}`, "
            f"remote `{qkv_runner.get('qkv_uram_remote_status')}`, "
            f"c3b_generic_discovery `{qkv_runner.get('pynq_c3b_smoke_discovery_status')}`, "
            f"vref_generic_discovery `{qkv_runner.get('pynq_vref_smoke_discovery_status')}`, "
            f"discovery `{qkv_runner.get('qkv_uram_smoke_discovery_status')}`, "
            f"required `{qkv_runner.get('qkv_uram_required_for_final_signoff')}`"
        )
    vref_discovery = audit["gate_modes"].get("vref_smoke_discovery", {})
    if isinstance(vref_discovery, dict):
        lines.append(
            f"- vref_smoke_discovery: `{vref_discovery.get('status')}`, "
            f"pass `{vref_discovery.get('pass_count')}`, candidates `{vref_discovery.get('candidate_count')}`"
        )
    smoke_discovery = audit["gate_modes"].get("smoke_candidate_discovery", {})
    if isinstance(smoke_discovery, dict):
        for key in ["c3b_generic", "vref_generic", "vref_runner", "qkv_runner"]:
            item = smoke_discovery.get(key, {})
            if isinstance(item, dict):
                lines.append(
                    f"- smoke_candidate_discovery.{key}: `{item.get('status')}`, "
                    f"pass `{item.get('pass_count')}`, candidates `{item.get('candidate_count')}`"
                )
    blocker_closure = audit["gate_modes"].get("final_blocker_closure", {})
    if isinstance(blocker_closure, dict):
        lines.append(
            f"- final_blocker_closure: `{blocker_closure.get('status')}`, "
            f"current_ready `{blocker_closure.get('current_ready')}`, "
            f"candidate_ready `{blocker_closure.get('candidate_ready')}`"
        )
    unblock_intake = audit["gate_modes"].get("final_unblock_intake", {})
    if isinstance(unblock_intake, dict):
        lines.append(
            f"- final_unblock_intake: `{unblock_intake.get('status')}`, "
            f"candidate_ready `{unblock_intake.get('candidate_ready')}`, "
            f"would_clear_all `{unblock_intake.get('would_clear_all')}`"
        )
    unblock_commands = audit["gate_modes"].get("final_unblock_commands", {})
    if isinstance(unblock_commands, dict):
        lines.append(
            f"- final_unblock_commands: `{unblock_commands.get('status')}`, "
            f"sections `{unblock_commands.get('section_count')}`, "
            f"has_qkv_u5 `{unblock_commands.get('has_qkv_uram_u5')}`, "
            f"blockers `{len(unblock_commands.get('remaining_blockers', []))}`"
        )
    candidate_audit = audit["gate_modes"].get("final_unblock_candidate_audit", {})
    if isinstance(candidate_audit, dict):
        lines.append(
            f"- final_unblock_candidate_audit: `{candidate_audit.get('status')}`, "
            f"would_clear_all `{candidate_audit.get('would_clear_all')}`, "
            f"blockers `{len(candidate_audit.get('remaining_blockers', []))}`"
        )
    operator_handoff = audit["gate_modes"].get("final_operator_handoff", {})
    if isinstance(operator_handoff, dict):
        lines.append(
            f"- final_operator_handoff: `{operator_handoff.get('status')}`, "
            f"operator_action `{operator_handoff.get('requires_operator_action')}`, "
            f"blockers `{len(operator_handoff.get('remaining_blockers', []))}`"
        )
    operator_handoff_validation = audit["gate_modes"].get("final_operator_handoff_validation", {})
    if isinstance(operator_handoff_validation, dict):
        lines.append(
            f"- final_operator_handoff_validation: `{operator_handoff_validation.get('status')}`, "
            f"pass `{operator_handoff_validation.get('pass_count')}`, "
            f"fail `{operator_handoff_validation.get('fail_count')}`, "
            f"policy_checks `{operator_handoff_validation.get('policy_check_count')}`"
        )
    xr_policy_integrity = audit["gate_modes"].get("xr_vits_policy_integrity", {})
    if isinstance(xr_policy_integrity, dict):
        command_policy = xr_policy_integrity.get("command_card", {})
        handoff_policy = xr_policy_integrity.get("operator_handoff", {})
        if not isinstance(command_policy, dict):
            command_policy = {}
        if not isinstance(handoff_policy, dict):
            handoff_policy = {}
        lines.append(
            f"- xr_vits_policy_integrity: consistent `{xr_policy_integrity.get('consistent')}`, "
            f"command_required `{command_policy.get('required')}`, "
            f"handoff_validation `{handoff_policy.get('validation_status')}`, "
            f"policy_exists `{handoff_policy.get('policy_exists')}`"
        )
    xr_vits_unblock = audit["gate_modes"].get("xr_vits_unblock_packet", {})
    if isinstance(xr_vits_unblock, dict):
        lines.append(
            f"- xr_vits_unblock_packet: `{xr_vits_unblock.get('status')}`, "
            f"options `{xr_vits_unblock.get('option_count')}`, "
            f"policy_approved `{xr_vits_unblock.get('active_policy_approved')}`"
        )
    req5 = audit.get("req5_q4q8_swhw_match_audit", {})
    if isinstance(req5, dict):
        lines.append(
            f"- req5_q4q8_swhw_match: `{req5.get('status')}`, "
            f"precision `{req5.get('precision')}`, fail `{req5.get('fail_count')}`"
        )
    req1 = audit.get("req1_environment_audit", {})
    if isinstance(req1, dict):
        env = req1.get("environment", {})
        os_release = env.get("os_release", {}) if isinstance(env, dict) and isinstance(env.get("os_release"), dict) else {}
        lines.append(
            f"- req1_environment: `{req1.get('status')}`, "
            f"checks `{req1.get('pass_count')}/{req1.get('check_count')}`, "
            f"os `{os_release.get('PRETTY_NAME')}`"
        )
    req6 = audit.get("req6_parameterization_audit", {})
    if isinstance(req6, dict):
        lines.append(
            f"- req6_parameterization: `{req6.get('status')}`, "
            f"checks `{req6.get('pass_count')}/{req6.get('check_count')}`, "
            f"macros `{req6.get('config_macros')}`"
        )
    req9 = audit.get("req9_deit_image_reference_audit", {})
    if isinstance(req9, dict):
        lines.append(
            f"- req9_deit_image: `{req9.get('status')}`, "
            f"checks `{req9.get('pass_count')}/{req9.get('check_count')}`, "
            f"requested `{req9.get('normalized_requested_path')}`"
        )
    lines.extend(["", "## Remaining External Inputs"])
    blocker_names = audit.get("external_blocker_names", [])
    if blocker_names:
        lines.append("")
        lines.append("External blocker names:")
        for blocker in blocker_names:
            lines.append(f"- `{blocker}`")
        mapping = audit.get("blocked_external_input_paths_by_blocker", {})
        if isinstance(mapping, dict):
            lines.append("")
            lines.append("Blocked external input paths by blocker:")
            for blocker, path in mapping.items():
                lines.append(f"- `{blocker}` -> `{path}`")
        lines.append("")
        lines.append("External input paths:")
    for item in audit["remaining_external_inputs"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Pending Optional Or Internal Evidence"])
    for item in audit.get("pending_optional_or_internal_evidence", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, default=Path(f"generated/signoff/third_goal_current_audit_{DATE_TAG}.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path(f"generated/signoff/third_goal_current_audit_{DATE_TAG}.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    write_outputs(audit, args.json_out, args.markdown_out)
    print(json.dumps({"status": audit["status"], "summary": audit["summary"]}, sort_keys=True))
    return 0 if audit["status"] in {"pass", "partial", "blocked-external"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
