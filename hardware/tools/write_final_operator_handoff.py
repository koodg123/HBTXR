#!/usr/bin/env python3
"""Write a concise operator handoff for the remaining final unblock actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import create_xr_vits_replacement_policy as xr_policy


DATE_TAG = "2026_06_10"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def section_by_id(command_card: dict[str, Any], section_id: str) -> dict[str, Any]:
    sections = command_card.get("sections", [])
    if not isinstance(sections, list):
        return {}
    for section in sections:
        if isinstance(section, dict) and section.get("id") == section_id:
            return section
    return {}


def option_by_id(section: dict[str, Any], option_id: str) -> dict[str, Any]:
    options = section.get("options", [])
    if not isinstance(options, list):
        return {}
    for option in options:
        if isinstance(option, dict) and option.get("id") == option_id:
            return option
    return {}


def resolve_path(root: Path, value: Any, fallback: str) -> Path:
    path = Path(str(value or fallback))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def build_xr_policy_integrity(root: Path, command_card: dict[str, Any]) -> dict[str, Any]:
    base = (
        command_card.get("xr_vits_policy_integrity", {})
        if isinstance(command_card.get("xr_vits_policy_integrity"), dict)
        else {}
    )
    policy_path = resolve_path(root, base.get("policy_path"), "docs/resources/xr_vits_replacement_policy.json")
    candidate_audit_path = resolve_path(root, base.get("candidate_audit"), xr_policy.DEFAULT_AUDIT_REL)
    result = dict(base)
    result.update(
        {
            "required": base.get("required") is True,
            "policy_path": str(policy_path),
            "candidate_audit": str(candidate_audit_path),
            "policy_exists": policy_path.exists(),
        }
    )
    if not policy_path.exists():
        result["validation"] = {
            "status": "pending-policy-creation",
            "policy_exists": False,
            "candidate_audit_path": str(candidate_audit_path),
            "errors": [],
            "warnings": ["replacement policy has not been written yet; operator approval is still required"],
        }
        return result

    policy = load_json(policy_path)
    validation = xr_policy.validate_policy_integrity(root, policy)
    validation["policy_exists"] = True
    result["validation"] = validation
    for field in sorted(xr_policy.INTEGRITY_FIELD_NAMES):
        if field in policy:
            result[field] = policy[field]
    return result


def build_handoff(
    *,
    root: Path,
    command_card: dict[str, Any],
    requirements_trace: dict[str, Any],
    readiness: dict[str, Any],
    candidate_audit: dict[str, Any],
    xr_vits_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    u1 = section_by_id(command_card, "U1")
    u2 = section_by_id(command_card, "U2")
    u3 = section_by_id(command_card, "U3")
    u4 = section_by_id(command_card, "U4")
    u2a = option_by_id(u2, "U2a")
    u2b = option_by_id(u2, "U2b")
    u4a = option_by_id(u4, "U4a")
    u4b = option_by_id(u4, "U4b")
    candidate_xr = candidate_audit.get("xr_vits", {}) if isinstance(candidate_audit.get("xr_vits"), dict) else {}
    candidate_c3b = candidate_audit.get("c3b_smoke", {}) if isinstance(candidate_audit.get("c3b_smoke"), dict) else {}
    xr_resolution = xr_vits_resolution or {}
    xr_resolution_candidate = (
        xr_resolution.get("candidate", {}) if isinstance(xr_resolution.get("candidate"), dict) else {}
    )
    xr_resolution_commands = (
        xr_resolution.get("commands", {}) if isinstance(xr_resolution.get("commands"), dict) else {}
    )
    xr_resolution_safety = (
        xr_resolution.get("safety", {}) if isinstance(xr_resolution.get("safety"), dict) else {}
    )
    xr_policy_integrity = build_xr_policy_integrity(root, command_card)
    final_signoff = requirements_trace.get("final_signoff", {}) if isinstance(requirements_trace.get("final_signoff"), dict) else {}
    evidence_contract = (
        requirements_trace.get("final_evidence_manifest_contract", {})
        if isinstance(requirements_trace.get("final_evidence_manifest_contract"), dict)
        else {}
    )
    remaining = final_signoff.get("remaining_blockers", [])
    if not isinstance(remaining, list):
        remaining = []

    return {
        "status": "pending-operator-actions" if remaining else "ready-for-final-signoff",
        "root": str(root),
        "remaining_blockers": [str(item) for item in remaining],
        "board_smoke": {
            "status": readiness.get("status", "unknown"),
            "ready": bool(readiness.get("ready", False)),
            "variant": readiness.get("variant", ""),
            "preset": readiness.get("preset", ""),
            "bundle_tar": (readiness.get("tar", {}) or {}).get("path") if isinstance(readiness.get("tar"), dict) else "",
            "bundle_sha256": (readiness.get("tar", {}) or {}).get("sha256") if isinstance(readiness.get("tar"), dict) else "",
            "expected_runtime_state": readiness.get("expected_runtime_state"),
            "expected_out_raw": readiness.get("expected_out_raw", []),
            "physical_result": (readiness.get("physical_result", {}) or {}).get("path")
            if isinstance(readiness.get("physical_result"), dict)
            else "",
            "candidate_status": candidate_c3b.get("status", "missing"),
            "would_clear": bool(candidate_c3b.get("would_clear", False)),
            "commands": u1.get("commands", []) if isinstance(u1.get("commands"), list) else [],
        },
        "xr_vits": {
            "candidate_status": candidate_xr.get("status", "missing"),
            "candidate_mode": candidate_xr.get("mode", ""),
            "would_clear": bool(candidate_xr.get("would_clear", False)),
            "requested_path": candidate_xr.get("requested_path", "/home/kjm26/project/PRJXR/XR-VITs"),
            "replacement_path": candidate_xr.get("replacement_path", ""),
            "reference_resolution": {
                "status": xr_resolution.get("status", "missing"),
                "resolution_ready": bool(xr_resolution.get("resolution_ready", False)),
                "approval_required": bool(xr_resolution.get("approval_required", True)),
                "requested_path": xr_resolution.get("requested_path", ""),
                "candidate_status": xr_resolution_candidate.get("status", "missing"),
                "candidate_replacement_path": xr_resolution_candidate.get("replacement_path", ""),
                "candidate_score": xr_resolution_candidate.get("score", 0),
                "replacement_dry_run_command": xr_resolution_commands.get("replacement_dry_run", ""),
                "replacement_approve_command": xr_resolution_commands.get("replacement_approve", ""),
                "creates_xr_vits_policy": bool(xr_resolution_safety.get("creates_xr_vits_policy", False)),
                "writes_canonical_inputs": bool(xr_resolution_safety.get("writes_canonical_inputs", False)),
                "policy_integrity": xr_policy_integrity,
            },
            "exact_restore_commands": u2a.get("commands", []) if isinstance(u2a.get("commands"), list) else [],
            "replacement_approval_commands": u2b.get("commands", []) if isinstance(u2b.get("commands"), list) else [],
        },
        "combined_unblock": {
            "exact_restore_commands": u4a.get("commands", []) if isinstance(u4a.get("commands"), list) else [],
            "replacement_approval_commands": u4b.get("commands", []) if isinstance(u4b.get("commands"), list) else [],
        },
        "final_signoff": {
            "status": final_signoff.get("status", "unknown"),
            "summary": final_signoff.get("summary", {}),
            "commands": u3.get("commands", []) if isinstance(u3.get("commands"), list) else [],
            "expected_result": u3.get("expected_result", "status=pass"),
        },
        "final_evidence_manifest_contract": {
            "status": evidence_contract.get("status", "missing"),
            "source": evidence_contract.get("source", ""),
            "path": evidence_contract.get("path", ""),
            "required_count": evidence_contract.get("required_count"),
            "present_required_count": evidence_contract.get("present_required_count"),
            "consistency_count": evidence_contract.get("consistency_count", 0),
            "failed_consistency_checks": evidence_contract.get("failed_consistency_checks", [])
            if isinstance(evidence_contract.get("failed_consistency_checks"), list)
            else [],
            "safety": evidence_contract.get("safety", {})
            if isinstance(evidence_contract.get("safety"), dict)
            else {},
        },
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_network": False,
            "requires_operator_action": True,
        },
        "source_files": {
            "command_card": str(root / f"docs/resources/final_unblock_commands_{DATE_TAG}.json"),
            "requirements_trace": str(root / f"docs/resources/third_goal_requirements_trace_{DATE_TAG}.json"),
            "readiness": str(root / f"docs/resources/c3b_board_smoke_readiness_{DATE_TAG}.json"),
            "candidate_audit": str(root / f"docs/resources/final_unblock_candidate_audit_{DATE_TAG}.json"),
            "xr_vits_reference_resolution": str(root / f"docs/resources/xr_vits_reference_resolution_{DATE_TAG}.json"),
            "final_evidence_manifest": str(root / f"docs/resources/final_evidence_manifest_{DATE_TAG}.json"),
        },
    }


def render_markdown(handoff: dict[str, Any]) -> str:
    board = handoff["board_smoke"]
    xr = handoff["xr_vits"]
    xr_resolution = xr.get("reference_resolution", {})
    final = handoff["final_signoff"]
    evidence_contract = handoff.get("final_evidence_manifest_contract", {})
    lines = [
        "# HGTXR Final Operator Handoff",
        "",
        f"- status: `{handoff['status']}`",
        f"- root: `{handoff['root']}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    if handoff["remaining_blockers"]:
        lines.extend(f"- `{item}`" for item in handoff["remaining_blockers"])
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Board Smoke",
            "",
            f"- status: `{board['status']}`",
            f"- ready: `{board['ready']}`",
            f"- variant/preset: `{board['variant']}` / `{board['preset']}`",
            f"- bundle_tar: `{board['bundle_tar']}`",
            f"- bundle_sha256: `{board['bundle_sha256']}`",
            f"- physical_result: `{board['physical_result']}`",
            f"- expected_runtime_state: `{board['expected_runtime_state']}`",
            f"- expected_out_raw: `{board['expected_out_raw']}`",
            "",
            "```sh",
        ]
    )
    lines.extend(board["commands"] or ["# No board commands available."])
    lines.extend(
        [
            "```",
            "",
            "## XR-VITs Gate",
            "",
            f"- candidate_status: `{xr['candidate_status']}`",
            f"- candidate_mode: `{xr['candidate_mode']}`",
            f"- requested_path: `{xr['requested_path']}`",
            f"- replacement_path: `{xr['replacement_path']}`",
            f"- reference_resolution_status: `{xr_resolution.get('status', 'missing')}`",
            f"- reference_resolution_ready: `{xr_resolution.get('resolution_ready', False)}`",
            f"- reference_approval_required: `{xr_resolution.get('approval_required', True)}`",
            f"- reference_candidate_score: `{xr_resolution.get('candidate_score', 0)}`",
            "",
            "### Exact Restore",
            "",
            "```sh",
        ]
    )
    lines.extend(xr["exact_restore_commands"] or ["# No exact-restore commands available."])
    lines.extend(["```", "", "### Replacement Approval", "", "```sh"])
    lines.extend(xr["replacement_approval_commands"] or ["# No replacement commands available."])
    lines.extend(["```", "", "### Reference Resolution Commands", "", "```sh"])
    lines.append(xr_resolution.get("replacement_dry_run_command", "") or "# No resolution dry-run command available.")
    lines.append(xr_resolution.get("replacement_approve_command", "") or "# No resolution approval command available.")
    xr_integrity = xr_resolution.get("policy_integrity", {}) if isinstance(xr_resolution.get("policy_integrity"), dict) else {}
    lines.extend(
        [
            "```",
            "",
            "### Policy Integrity",
            "",
            f"- required: `{xr_integrity.get('required', False)}`",
            f"- policy_exists: `{xr_integrity.get('policy_exists', False)}`",
            f"- validation_status: `{(xr_integrity.get('validation', {}) or {}).get('status', 'missing')}`",
            f"- generator: `{xr_integrity.get('generator', '')}`",
            f"- validator: `{xr_integrity.get('validator', '')}`",
            f"- required_policy_fields: `{xr_integrity.get('required_policy_fields', [])}`",
            f"- candidate_audit_fingerprint: `{xr_integrity.get('candidate_audit_fingerprint', '')}`",
            f"- policy_fingerprint: `{xr_integrity.get('policy_fingerprint', '')}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Final Evidence Manifest Contract",
            "",
            f"- status: `{evidence_contract.get('status', 'missing')}`",
            f"- source: `{evidence_contract.get('source', '')}`",
            f"- path: `{evidence_contract.get('path', '')}`",
            f"- required: `{evidence_contract.get('present_required_count')}/"
            f"{evidence_contract.get('required_count')}`",
            f"- consistency_checks: `{evidence_contract.get('consistency_count', 0)}`",
            f"- failed_consistency_checks: `{evidence_contract.get('failed_consistency_checks', [])}`",
            "",
            "## Combined One-Shot",
            "",
            "### Exact Restore + C3b Import",
            "",
            "```sh",
        ]
    )
    lines.extend(handoff["combined_unblock"]["exact_restore_commands"] or ["# No combined exact commands available."])
    lines.extend(["```", "", "### Replacement Approval + C3b Import", "", "```sh"])
    lines.extend(handoff["combined_unblock"]["replacement_approval_commands"] or ["# No combined replacement commands available."])
    lines.extend(
        [
            "```",
            "",
            "## Final Signoff",
            "",
            f"- current_status: `{final['status']}`",
            f"- expected_result: `{final['expected_result']}`",
            "",
            "```sh",
        ]
    )
    lines.extend(final["commands"] or ["# No final signoff command available."])
    lines.extend(
        [
            "```",
            "",
            "## Safety",
            "",
            "- This handoff does not create board smoke results.",
            "- This handoff does not create XR-VITs replacement policy.",
            "- This handoff does not execute network commands.",
            "- Commands with placeholders require operator replacement before execution.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write final operator handoff for HGTXR unblock.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--command-card", type=Path, default=None)
    parser.add_argument("--requirements-trace", type=Path, default=None)
    parser.add_argument("--readiness", type=Path, default=None)
    parser.add_argument("--candidate-audit", type=Path, default=None)
    parser.add_argument("--xr-vits-resolution", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    resources = root / "docs" / "resources"
    command_path = args.command_card or resources / f"final_unblock_commands_{DATE_TAG}.json"
    trace_path = args.requirements_trace or resources / f"third_goal_requirements_trace_{DATE_TAG}.json"
    readiness_path = args.readiness or resources / f"c3b_board_smoke_readiness_{DATE_TAG}.json"
    candidate_path = args.candidate_audit or resources / f"final_unblock_candidate_audit_{DATE_TAG}.json"
    xr_vits_resolution_path = args.xr_vits_resolution or resources / f"xr_vits_reference_resolution_{DATE_TAG}.json"
    handoff = build_handoff(
        root=root,
        command_card=load_json(command_path),
        requirements_trace=load_json(trace_path),
        readiness=load_json(readiness_path),
        candidate_audit=load_json(candidate_path),
        xr_vits_resolution=load_json(xr_vits_resolution_path) if xr_vits_resolution_path.exists() else None,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(handoff))
    print(
        f"[final-operator-handoff] status={handoff['status']} "
        f"blockers={len(handoff['remaining_blockers'])}"
    )
    return 0 if handoff["status"] == "ready-for-final-signoff" else 1


if __name__ == "__main__":
    raise SystemExit(main())
