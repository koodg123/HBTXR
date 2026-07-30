#!/usr/bin/env python3
"""Write a compact evidence trace for every explicit third-goal requirement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import write_final_evidence_manifest


DATE_TAG = "2026_06_10"
REQUIRED_XR_VITS_POLICY_FIELDS = [
    "candidate_audit_fingerprint",
    "candidate_audit_recommendation_snapshot",
    "candidate_audit_meta",
    "approval_event",
    "policy_fingerprint",
]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def index_completion_items(completion: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = completion.get("items", [])
    if not isinstance(items, list):
        return {}
    return {str(item.get("id")): item for item in items if isinstance(item, dict)}


def pick_row(matrix: dict[str, Any], row_id: str) -> dict[str, Any] | None:
    rows = matrix.get("rows", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("id") == row_id:
            return row
    return None


def resource_snapshot(matrix: dict[str, Any]) -> dict[str, Any]:
    c3b = pick_row(matrix, "C3b") or {}
    hls = c3b.get("hls", {}) if isinstance(c3b.get("hls"), dict) else {}
    vivado = c3b.get("vivado", {}) if isinstance(c3b.get("vivado"), dict) else {}
    resources = hls.get("resources", {}) if isinstance(hls.get("resources"), dict) else {}
    timing = vivado.get("timing", {}) if isinstance(vivado.get("timing"), dict) else {}
    power = vivado.get("power", {}) if isinstance(vivado.get("power"), dict) else {}
    return {
        "selected_variant": "C3b",
        "latency_cycles": hls.get("latency_cycles"),
        "dsp": resources.get("dsp"),
        "lut": resources.get("lut"),
        "ff": resources.get("ff"),
        "bram_18k": resources.get("bram_18k"),
        "uram": resources.get("uram"),
        "wns_ns": timing.get("wns_ns"),
        "power_w": power.get("total_on_chip_w"),
        "matrix_recommendation": (matrix.get("summary") or {}).get("recommended_board_smoke_variant"),
    }


def candidate_audit_summary(candidate_audit: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate_audit:
        return {
            "status": "missing",
            "would_clear_all": False,
            "remaining_blockers": [],
            "c3b_smoke_status": "missing",
            "c3b_smoke_would_clear": False,
            "xr_vits_status": "missing",
            "xr_vits_mode": "",
            "xr_vits_would_clear": False,
            "safety": {
                "creates_board_result": False,
                "creates_xr_vits_policy": False,
                "executes_network": False,
                "writes_canonical_inputs": False,
            },
        }
    c3b = candidate_audit.get("c3b_smoke", {}) if isinstance(candidate_audit.get("c3b_smoke"), dict) else {}
    xr_vits = candidate_audit.get("xr_vits", {}) if isinstance(candidate_audit.get("xr_vits"), dict) else {}
    safety = candidate_audit.get("safety", {}) if isinstance(candidate_audit.get("safety"), dict) else {}
    return {
        "status": candidate_audit.get("status", "unknown"),
        "would_clear_all": bool(candidate_audit.get("would_clear_all")),
        "remaining_blockers": candidate_audit.get("remaining_blockers", [])
        if isinstance(candidate_audit.get("remaining_blockers"), list)
        else [],
        "c3b_smoke_status": c3b.get("status", "missing"),
        "c3b_smoke_would_clear": bool(c3b.get("would_clear")),
        "xr_vits_status": xr_vits.get("status", "missing"),
        "xr_vits_mode": xr_vits.get("mode", ""),
        "xr_vits_would_clear": bool(xr_vits.get("would_clear")),
        "safety": {
            "creates_board_result": bool(safety.get("creates_board_result", False)),
            "creates_xr_vits_policy": bool(safety.get("creates_xr_vits_policy", False)),
            "executes_network": bool(safety.get("executes_network", False)),
            "writes_canonical_inputs": bool(safety.get("writes_canonical_inputs", False)),
        },
    }


def xr_vits_resolution_summary(resolution: dict[str, Any] | None) -> dict[str, Any]:
    if not resolution:
        return {
            "status": "missing",
            "resolution_ready": False,
            "approval_required": True,
            "requested_path": "",
            "exact_status": "missing",
            "active_policy_status": "missing",
            "candidate_status": "missing",
            "candidate_replacement_path": "",
            "candidate_score": 0,
            "replacement_dry_run_command": "",
            "replacement_approve_command": "",
            "safety": {
                "creates_xr_vits_policy": False,
                "writes_canonical_inputs": False,
                "executes_commands": False,
            },
        }
    exact = resolution.get("exact", {}) if isinstance(resolution.get("exact"), dict) else {}
    active_policy = resolution.get("active_policy", {}) if isinstance(resolution.get("active_policy"), dict) else {}
    candidate = resolution.get("candidate", {}) if isinstance(resolution.get("candidate"), dict) else {}
    commands = resolution.get("commands", {}) if isinstance(resolution.get("commands"), dict) else {}
    safety = resolution.get("safety", {}) if isinstance(resolution.get("safety"), dict) else {}
    return {
        "status": resolution.get("status", "unknown"),
        "resolution_ready": bool(resolution.get("resolution_ready")),
        "approval_required": bool(resolution.get("approval_required")),
        "requested_path": resolution.get("requested_path", ""),
        "exact_status": exact.get("status", "missing"),
        "active_policy_status": active_policy.get("status", "missing"),
        "candidate_status": candidate.get("status", "missing"),
        "candidate_replacement_path": candidate.get("replacement_path", ""),
        "candidate_score": candidate.get("score", 0),
        "replacement_dry_run_command": commands.get("replacement_dry_run", ""),
        "replacement_approve_command": commands.get("replacement_approve", ""),
        "safety": {
            "creates_xr_vits_policy": bool(safety.get("creates_xr_vits_policy", False)),
            "writes_canonical_inputs": bool(safety.get("writes_canonical_inputs", False)),
            "executes_commands": bool(safety.get("executes_commands", False)),
        },
    }


def failed_check_names(checks: list[dict[str, Any]]) -> list[str]:
    return [
        str(check.get("name", "unknown"))
        for check in checks
        if isinstance(check, dict) and check.get("status") == "fail"
    ]


def evidence_manifest_contract_summary(root: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest:
        consistency_checks = manifest.get("consistency_checks", [])
        if not isinstance(consistency_checks, list):
            consistency_checks = []
        failed_consistency = manifest.get("failed_consistency_checks", failed_check_names(consistency_checks))
        if not isinstance(failed_consistency, list):
            failed_consistency = failed_check_names(consistency_checks)
        return {
            "status": manifest.get("status", "unknown"),
            "source": "manifest-json",
            "path": str(root / f"docs/resources/final_evidence_manifest_{DATE_TAG}.json"),
            "required_count": manifest.get("required_count"),
            "present_required_count": manifest.get("present_required_count"),
            "consistency_count": len(consistency_checks),
            "failed_consistency_checks": [str(item) for item in failed_consistency],
            "safety": manifest.get("safety", {}) if isinstance(manifest.get("safety"), dict) else {},
        }

    consistency_checks = write_final_evidence_manifest.build_consistency_checks(root)
    failed_consistency = failed_check_names(consistency_checks)
    return {
        "status": "pass" if not failed_consistency else "fail",
        "source": "live-consistency-checks",
        "path": str(root / f"docs/resources/final_evidence_manifest_{DATE_TAG}.json"),
        "required_count": None,
        "present_required_count": None,
        "consistency_count": len(consistency_checks),
        "failed_consistency_checks": failed_consistency,
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def unblock_actions(command_card: dict[str, Any]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    sections = command_card.get("sections", [])
    if not isinstance(sections, list):
        return actions
    for section in sections:
        if not isinstance(section, dict):
            continue
        status = str(section.get("status", "unknown"))
        if status in {"pending-board-run", "pending-user-choice", "pending"}:
            actions.append(
                {
                    "id": str(section.get("id", "")),
                    "title": str(section.get("title", "")),
                    "status": status,
                    "reason": str(section.get("reason", "")),
                }
            )
    return actions


def policy_integrity_summary(command_card: dict[str, Any]) -> dict[str, Any]:
    policy = command_card.get("xr_vits_policy_integrity", {})
    if not isinstance(policy, dict):
        policy = {}
    required_fields = policy.get("required_policy_fields", [])
    if not isinstance(required_fields, list):
        required_fields = []
    required_field_names = [str(field) for field in required_fields]
    required_fields_complete = all(field in required_field_names for field in REQUIRED_XR_VITS_POLICY_FIELDS)
    return {
        "required": bool(policy.get("required", False)),
        "policy_path": str(policy.get("policy_path", "")),
        "candidate_audit": str(policy.get("candidate_audit", "")),
        "generator": str(policy.get("generator", "")),
        "validator": str(policy.get("validator", "")),
        "required_policy_fields": required_field_names,
        "required_policy_fields_complete": bool(policy.get("required_policy_fields_complete", required_fields_complete)),
        "legacy_policy_clears_final_signoff": bool(policy.get("legacy_policy_clears_final_signoff", True)),
        "validation_status": str(policy.get("validation_status", "unknown")),
        "policy_exists": bool(policy.get("policy_exists", False)),
        "policy_fingerprint_present": bool(policy.get("policy_fingerprint_present", False)),
        "candidate_audit_fingerprint_present": bool(policy.get("candidate_audit_fingerprint_present", False)),
    }


def option_commands_by_id(command_card: dict[str, Any]) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    sections = command_card.get("sections", [])
    if not isinstance(sections, list):
        return commands
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id", ""))
        section_commands = section.get("commands", [])
        if section_id and isinstance(section_commands, list):
            commands[section_id] = [str(cmd) for cmd in section_commands]
        for option in section.get("options", []) if isinstance(section.get("options"), list) else []:
            if not isinstance(option, dict):
                continue
            option_id = str(option.get("id", ""))
            option_commands = option.get("commands", [])
            if option_id and isinstance(option_commands, list):
                commands[option_id] = [str(cmd) for cmd in option_commands]
    return commands


def blocker_resolution_conditions(root: Path, blockers: list[Any], command_card: dict[str, Any]) -> list[dict[str, Any]]:
    command_by_id = option_commands_by_id(command_card)
    policy_integrity = policy_integrity_summary(command_card)
    normalized = [str(blocker) for blocker in blockers]
    conditions: list[dict[str, Any]] = []
    for blocker in normalized:
        if blocker == "C3b AXIS/DMA physical smoke result":
            conditions.append(
                {
                    "blocker": blocker,
                    "clears_requirement_ids": ["3", "4"],
                    "required_evidence": [
                        str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                        str(root / f"docs/resources/c3b_smoke_import_{DATE_TAG}.json"),
                        str(root / f"docs/resources/c3b_board_smoke_readiness_{DATE_TAG}.json"),
                    ],
                    "acceptance": [
                        "validate_pynq_smoke_result reports pass for preset axis-c3b-mem16",
                        "c3b_import_status is pass after active import, not dry-run-pass",
                        "c3b_board_smoke_readiness physical_result.status is pass or complete",
                    ],
                    "command_options": {
                        "remote_execute": command_by_id.get("U1", []),
                        "combined_exact": command_by_id.get("U4a", []),
                        "combined_replacement": command_by_id.get("U4b", []),
                    },
                }
            )
        elif blocker == "requested XR-VITs sibling":
            conditions.append(
                {
                    "blocker": blocker,
                    "clears_requirement_ids": ["10", "11"],
                    "required_evidence": [
                        "/home/kjm26/project/PRJXR/XR-VITs",
                        str(root / "docs/resources/xr_vits_replacement_policy.json"),
                        str(root / "docs/resources/xr_vits_candidate_audit_2026_06_10.json"),
                        str(root / f"docs/resources/xr_vits_unblock_packet_{DATE_TAG}.json"),
                        str(root / f"docs/resources/xr_vits_reference_resolution_{DATE_TAG}.json"),
                    ],
                    "acceptance": [
                        "exact XR-VITs sibling exists, or active replacement policy is approved",
                        "replacement policy is generated by tools/create_xr_vits_replacement_policy.py",
                        "replacement policy includes candidate_audit_fingerprint, candidate_audit_recommendation_snapshot, candidate_audit_meta, and policy_fingerprint",
                        "replacement policy validates against current candidate audit; legacy or drifted policy does not clear final signoff",
                        "xr_vits_reference_resolution status is exact-ready or approved-replacement-ready",
                        "xr_vits_packet_status is exact-restored or approved-replacement",
                        "final preflight no longer reports requested XR-VITs sibling as fail",
                    ],
                    "policy_integrity": policy_integrity,
                    "command_options": {
                        "restore_exact": command_by_id.get("U2a", []),
                        "approve_replacement": command_by_id.get("U2b", []),
                        "combined_exact": command_by_id.get("U4a", []),
                        "combined_replacement": command_by_id.get("U4b", []),
                    },
                }
            )
        else:
            conditions.append(
                {
                    "blocker": blocker,
                    "clears_requirement_ids": [],
                    "required_evidence": [],
                    "acceptance": ["Re-run final signoff and verify blocker is absent."],
                    "command_options": {},
                }
            )
    return conditions


def build_trace(
    *,
    root: Path,
    completion: dict[str, Any],
    matrix: dict[str, Any],
    command_card: dict[str, Any],
    signoff_run: dict[str, Any],
    candidate_audit: dict[str, Any] | None = None,
    xr_vits_resolution: dict[str, Any] | None = None,
    evidence_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items_by_id = index_completion_items(completion)
    requirements: list[dict[str, Any]] = []
    for req_id in [str(i) for i in range(12)]:
        item = items_by_id.get(req_id, {})
        requirements.append(
            {
                "id": req_id,
                "title": item.get("title", f"Requirement {req_id}"),
                "status": item.get("status", "missing"),
                "evidence_count": len(item.get("evidence", [])) if isinstance(item.get("evidence"), list) else 0,
                "gap_count": len(item.get("gaps", [])) if isinstance(item.get("gaps"), list) else 0,
                "evidence": item.get("evidence", []),
                "gaps": item.get("gaps", []),
            }
        )

    blocked_requirements = [item for item in requirements if item["status"] == "blocked"]
    partial_requirements = [item for item in requirements if item["status"] == "partial"]
    external_actions = unblock_actions(command_card)
    remaining_blockers = signoff_run.get("remaining_blockers", [])
    if not isinstance(remaining_blockers, list):
        remaining_blockers = []
    evidence_contract = evidence_manifest_contract_summary(root, evidence_manifest)
    status = str(completion.get("status", "unknown"))
    if evidence_contract.get("status") == "fail":
        status = "blocked"

    return {
        "status": status,
        "root": str(root.resolve()),
        "completion_counts": {
            "pass": completion.get("pass_count"),
            "partial": completion.get("partial_count"),
            "blocked": completion.get("blocked_count"),
            "items": completion.get("item_count"),
        },
        "final_signoff": {
            "status": signoff_run.get("status"),
            "summary": signoff_run.get("final_preflight_summary"),
            "remaining_blockers": remaining_blockers,
        },
        "blocker_resolution_conditions": blocker_resolution_conditions(root, remaining_blockers, command_card),
        "xr_vits_policy_integrity": policy_integrity_summary(command_card),
        "candidate_unblock_audit": candidate_audit_summary(candidate_audit),
        "xr_vits_reference_resolution": xr_vits_resolution_summary(xr_vits_resolution),
        "final_evidence_manifest_contract": evidence_contract,
        "selected_resource_snapshot": resource_snapshot(matrix),
        "requirements": requirements,
        "blocked_requirement_ids": [item["id"] for item in blocked_requirements],
        "partial_requirement_ids": [item["id"] for item in partial_requirements],
        "external_actions": external_actions,
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_network": False,
        },
        "source_files": {
            "completion_audit": str(root / f"docs/resources/third_goal_completion_audit_{DATE_TAG}.json"),
            "resource_matrix": str(root / f"docs/resources/e2e_resource_matrix_{DATE_TAG}.json"),
            "command_card": str(root / f"docs/resources/final_unblock_commands_{DATE_TAG}.json"),
            "signoff_run": str(root / f"docs/resources/third_goal_final_signoff_run_{DATE_TAG}.json"),
            "candidate_audit": str(root / f"docs/resources/final_unblock_candidate_audit_{DATE_TAG}.json"),
            "xr_vits_reference_resolution": str(root / f"docs/resources/xr_vits_reference_resolution_{DATE_TAG}.json"),
            "final_evidence_manifest": str(root / f"docs/resources/final_evidence_manifest_{DATE_TAG}.json"),
        },
    }


def render_markdown(trace: dict[str, Any]) -> str:
    snap = trace["selected_resource_snapshot"]
    lines = [
        "# HGTXR Third Goal Requirements Trace",
        "",
        f"- status: `{trace['status']}`",
        f"- root: `{trace['root']}`",
        f"- pass: `{trace['completion_counts']['pass']}`",
        f"- partial: `{trace['completion_counts']['partial']}`",
        f"- blocked: `{trace['completion_counts']['blocked']}`",
        f"- final signoff: `{trace['final_signoff']['status']}`",
        f"- candidate audit: `{trace['candidate_unblock_audit']['status']}`",
        f"- XR-VITs resolution: `{trace['xr_vits_reference_resolution']['status']}`",
        f"- final evidence manifest contract: `{trace['final_evidence_manifest_contract']['status']}`",
        "",
        "## Selected Resource Snapshot",
        "",
        f"- variant: `{snap['selected_variant']}`",
        f"- latency_cycles: `{snap['latency_cycles']}`",
        f"- DSP/LUT/FF/BRAM18K/URAM: `{snap['dsp']}/{snap['lut']}/{snap['ff']}/{snap['bram_18k']}/{snap['uram']}`",
        f"- WNS ns: `{snap['wns_ns']}`",
        f"- power W: `{snap['power_w']}`",
        "",
        "## Requirements",
        "",
        "| ID | Status | Evidence | Gaps | Title |",
        "|---|---|---:|---:|---|",
    ]
    for item in trace["requirements"]:
        title = str(item["title"]).replace("|", "\\|")
        lines.append(f"| {item['id']} | `{item['status']}` | {item['evidence_count']} | {item['gap_count']} | {title} |")

    lines.extend(["", "## Remaining External Actions", ""])
    if trace["external_actions"]:
        for action in trace["external_actions"]:
            lines.append(f"- `{action['id']}` `{action['status']}`: {action['title']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Remaining Blockers", ""])
    blockers = trace["final_signoff"]["remaining_blockers"]
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None.")
    candidate = trace.get("candidate_unblock_audit", {})
    lines.extend(["", "## Candidate Unblock Audit", ""])
    lines.append(f"- status: `{candidate.get('status', 'missing')}`")
    lines.append(f"- would_clear_all: `{candidate.get('would_clear_all', False)}`")
    lines.append(f"- C3b smoke: `{candidate.get('c3b_smoke_status', 'missing')}` clear=`{candidate.get('c3b_smoke_would_clear', False)}`")
    lines.append(
        f"- XR-VITs: `{candidate.get('xr_vits_status', 'missing')}` "
        f"mode=`{candidate.get('xr_vits_mode', '')}` clear=`{candidate.get('xr_vits_would_clear', False)}`"
    )
    candidate_blockers = candidate.get("remaining_blockers", []) if isinstance(candidate.get("remaining_blockers", []), list) else []
    if candidate_blockers:
        lines.append("- candidate remaining blockers:")
        for blocker in candidate_blockers:
            lines.append(f"  - `{blocker}`")
    xr_resolution = trace.get("xr_vits_reference_resolution", {})
    lines.extend(["", "## XR-VITs Reference Resolution", ""])
    lines.append(f"- status: `{xr_resolution.get('status', 'missing')}`")
    lines.append(f"- resolution_ready: `{xr_resolution.get('resolution_ready', False)}`")
    lines.append(f"- approval_required: `{xr_resolution.get('approval_required', True)}`")
    lines.append(f"- requested_path: `{xr_resolution.get('requested_path', '')}`")
    lines.append(f"- exact_status: `{xr_resolution.get('exact_status', 'missing')}`")
    lines.append(f"- active_policy_status: `{xr_resolution.get('active_policy_status', 'missing')}`")
    lines.append(
        f"- candidate: `{xr_resolution.get('candidate_status', 'missing')}` "
        f"score=`{xr_resolution.get('candidate_score', 0)}` "
        f"path=`{xr_resolution.get('candidate_replacement_path', '')}`"
    )
    policy_integrity = trace.get("xr_vits_policy_integrity", {})
    lines.extend(["", "## XR-VITs Policy Integrity", ""])
    lines.append(f"- required: `{policy_integrity.get('required', False)}`")
    lines.append(f"- policy_path: `{policy_integrity.get('policy_path', '')}`")
    lines.append(f"- candidate_audit: `{policy_integrity.get('candidate_audit', '')}`")
    lines.append(f"- generator: `{policy_integrity.get('generator', '')}`")
    lines.append(f"- validator: `{policy_integrity.get('validator', '')}`")
    lines.append(f"- validation_status: `{policy_integrity.get('validation_status', 'unknown')}`")
    lines.append(f"- policy_exists: `{policy_integrity.get('policy_exists', False)}`")
    lines.append(f"- legacy_policy_clears_final_signoff: `{policy_integrity.get('legacy_policy_clears_final_signoff', True)}`")
    required_fields = (
        policy_integrity.get("required_policy_fields", [])
        if isinstance(policy_integrity.get("required_policy_fields", []), list)
        else []
    )
    if required_fields:
        lines.append("- required policy fields:")
        for field in required_fields:
            lines.append(f"  - `{field}`")
    evidence_contract = trace.get("final_evidence_manifest_contract", {})
    lines.extend(["", "## Final Evidence Manifest Contract", ""])
    lines.append(f"- status: `{evidence_contract.get('status', 'missing')}`")
    lines.append(f"- source: `{evidence_contract.get('source', '')}`")
    lines.append(f"- path: `{evidence_contract.get('path', '')}`")
    lines.append(
        f"- required: `{evidence_contract.get('present_required_count')}/"
        f"{evidence_contract.get('required_count')}`"
    )
    lines.append(f"- consistency_checks: `{evidence_contract.get('consistency_count', 0)}`")
    failed_consistency = (
        evidence_contract.get("failed_consistency_checks", [])
        if isinstance(evidence_contract.get("failed_consistency_checks", []), list)
        else []
    )
    if failed_consistency:
        lines.append("- failed consistency checks:")
        for check in failed_consistency:
            lines.append(f"  - `{check}`")
    else:
        lines.append("- failed consistency checks: `[]`")
    lines.extend(["", "## Blocker Resolution Conditions", ""])
    conditions = trace.get("blocker_resolution_conditions", [])
    if conditions:
        for condition in conditions:
            lines.append(f"### {condition['blocker']}")
            clears = ", ".join(condition.get("clears_requirement_ids", [])) or "none"
            lines.append(f"- clears requirements: `{clears}`")
            lines.append("- required evidence:")
            for evidence in condition.get("required_evidence", []):
                lines.append(f"  - `{evidence}`")
            lines.append("- acceptance:")
            for item in condition.get("acceptance", []):
                lines.append(f"  - {item}")
            lines.append("")
    else:
        lines.append("- None.")
    lines.extend(["", "## Safety", ""])
    lines.append("- Does not create board smoke results.")
    lines.append("- Does not create XR-VITs replacement policy.")
    lines.append("- Does not execute network commands.")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write third-goal requirements trace.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--completion-audit", type=Path, default=None)
    parser.add_argument("--resource-matrix", type=Path, default=None)
    parser.add_argument("--command-card", type=Path, default=None)
    parser.add_argument("--signoff-run", type=Path, default=None)
    parser.add_argument("--candidate-audit", type=Path, default=None)
    parser.add_argument("--xr-vits-resolution", type=Path, default=None)
    parser.add_argument("--evidence-manifest", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    resources = root / "docs" / "resources"
    completion_path = args.completion_audit or resources / f"third_goal_completion_audit_{DATE_TAG}.json"
    matrix_path = args.resource_matrix or resources / f"e2e_resource_matrix_{DATE_TAG}.json"
    command_path = args.command_card or resources / f"final_unblock_commands_{DATE_TAG}.json"
    signoff_path = args.signoff_run or resources / f"third_goal_final_signoff_run_{DATE_TAG}.json"
    candidate_path = args.candidate_audit or resources / f"final_unblock_candidate_audit_{DATE_TAG}.json"
    xr_vits_resolution_path = args.xr_vits_resolution or resources / f"xr_vits_reference_resolution_{DATE_TAG}.json"
    evidence_manifest_path = args.evidence_manifest or resources / f"final_evidence_manifest_{DATE_TAG}.json"
    candidate_audit = load_json(candidate_path) if candidate_path.exists() else None
    xr_vits_resolution = load_json(xr_vits_resolution_path) if xr_vits_resolution_path.exists() else None
    evidence_manifest = load_json(evidence_manifest_path) if evidence_manifest_path.exists() else None
    trace = build_trace(
        root=root,
        completion=load_json(completion_path),
        matrix=load_json(matrix_path),
        command_card=load_json(command_path),
        signoff_run=load_json(signoff_path),
        candidate_audit=candidate_audit,
        xr_vits_resolution=xr_vits_resolution,
        evidence_manifest=evidence_manifest,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(trace))
    print(
        f"[third-goal-requirements-trace] status={trace['status']} "
        f"blocked={len(trace['blocked_requirement_ids'])} partial={len(trace['partial_requirement_ids'])}"
    )
    return 0 if trace["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
