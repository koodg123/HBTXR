#!/usr/bin/env python3
"""Write a current XR-VITs source/replacement gate audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_final_blocker_closure_readiness  # noqa: E402
import create_xr_vits_replacement_policy  # noqa: E402


DATE_TAG = "2026_06_16"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def rel_or_abs(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def summarize_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "approved_replacement": False,
            "would_clear": False,
            "errors": [f"active replacement policy missing: {path}"],
        }
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "fail",
            "path": str(path),
            "approved_replacement": False,
            "would_clear": False,
            "errors": [f"active replacement policy invalid: {exc}"],
        }
    return {
        "status": "present",
        "path": str(path),
        "approved_replacement": payload.get("approved_replacement") is True,
        "requested_path": payload.get("requested_path"),
        "replacement_path": payload.get("replacement_path"),
        "approved_by": payload.get("approved_by", ""),
        "reason": payload.get("reason", ""),
        "candidate_audit_fingerprint": payload.get("candidate_audit_fingerprint"),
        "candidate_audit_recommendation_snapshot": payload.get("candidate_audit_recommendation_snapshot"),
        "policy_fingerprint": payload.get("policy_fingerprint"),
        "would_clear": payload.get("approved_replacement") is True,
        "errors": [],
    }


def summarize_candidate_audit(path: Path, replacement_path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "recommendation_matches": False,
            "errors": [f"candidate audit missing: {path}"],
        }
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "fail",
            "path": str(path),
            "recommendation_matches": False,
            "errors": [f"candidate audit invalid: {exc}"],
        }
    recommendation = payload.get("recommendation", {})
    if not isinstance(recommendation, dict):
        recommendation = {}
    recommended = Path(str(recommendation.get("path", ""))).resolve()
    return {
        "status": str(payload.get("status", "unknown")),
        "path": str(path),
        "recommendation": recommendation,
        "recommendation_matches": recommended == replacement_path.resolve(),
        "candidate_count": len(payload.get("candidates", [])) if isinstance(payload.get("candidates"), list) else 0,
        "approved_replacement": payload.get("approved_replacement") is True,
        "policy_note": payload.get("policy", ""),
        "errors": [],
    }


def decision_options(hgtxr: Path, requested_path: Path, replacement_path: Path) -> list[dict[str, Any]]:
    active_policy = hgtxr / create_xr_vits_replacement_policy.DEFAULT_POLICY_REL
    return [
        {
            "id": "X1",
            "title": "Restore exact XR-VITs checkout",
            "action": f"Make {requested_path} available as a real directory.",
            "expected_result": "Req11 clears as exact-source evidence; no replacement policy needed.",
            "risk": "Requires locating/restoring the missing repository or checkout.",
            "writes_policy": False,
            "clears_final_signoff_when_done": True,
        },
        {
            "id": "X2",
            "title": "Approve XR_Accel as replacement source",
            "action": "Create an explicit replacement policy after user approval.",
            "expected_result": "Req11 clears through approved replacement-policy evidence while preserving provenance.",
            "risk": "Scientifically weaker than exact XR-VITs source; approval metadata must be explicit.",
            "writes_policy": True,
            "clears_final_signoff_when_done": True,
            "dry_run_command": (
                "python3 tools/create_xr_vits_replacement_policy.py "
                f"--root {hgtxr} "
                "--approve --approved-by USER "
                "--reason \"User approved XR_Accel as XR-VITs replacement for HGTXR third-goal continuation\" "
                "--dry-run"
            ),
            "active_command": (
                "python3 tools/create_xr_vits_replacement_policy.py "
                f"--root {hgtxr} "
                "--approve --approved-by USER "
                "--reason \"User approved XR_Accel as XR-VITs replacement for HGTXR third-goal continuation\""
            ),
            "policy_path": str(active_policy),
            "replacement_path": str(replacement_path),
        },
        {
            "id": "X3",
            "title": "Keep Req11 blocked",
            "action": "Do not restore XR-VITs and do not approve replacement yet.",
            "expected_result": "Final signoff remains blocked; implementation can continue only on non-Req11 work.",
            "risk": "Third-goal completion cannot be claimed.",
            "writes_policy": False,
            "clears_final_signoff_when_done": False,
        },
    ]


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    requested_path = create_xr_vits_replacement_policy.default_requested_path(hgtxr).resolve()
    replacement_path = create_xr_vits_replacement_policy.default_replacement_path(hgtxr).resolve()
    active_policy = check_final_blocker_closure_readiness.check_active_policy(hgtxr)
    current_gate = check_final_blocker_closure_readiness.check_current_xr_vits(hgtxr)
    policy_path = check_final_blocker_closure_readiness.active_policy_path(hgtxr)
    candidate_audit_rel = create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL
    candidate_audit_path = (hgtxr / candidate_audit_rel).resolve()
    template_path = hgtxr / "docs" / "resources" / "xr_vits_replacement_policy.template.json"
    reference_resolution_path = hardware / "generated" / "signoff" / "xr_vits_reference_resolution_2026_06_10.json"
    unblock_packet_path = hardware / "generated" / "signoff" / "xr_vits_unblock_packet_2026_06_10.json"

    exact = {
        "path": str(requested_path),
        "exists": requested_path.is_dir(),
        "would_clear": requested_path.is_dir(),
    }
    replacement = {
        "path": str(replacement_path),
        "exists": replacement_path.is_dir(),
        "role": "candidate-xr-accel",
    }
    candidate_audit = summarize_candidate_audit(candidate_audit_path, replacement_path)
    policy_summary = summarize_policy(policy_path)
    policy_integrity = (
        create_xr_vits_replacement_policy.validate_policy_integrity(hgtxr, load_json(policy_path))
        if policy_path.exists()
        else {
            "status": "missing",
            "has_integrity_fields": False,
            "candidate_audit_path": str(candidate_audit_path),
            "errors": [f"active replacement policy missing: {policy_path}"],
            "warnings": [],
        }
    )

    if exact["would_clear"]:
        status = "pass-exact"
        resolution_mode = "exact"
        remaining_blockers: list[str] = []
    elif active_policy["would_clear"]:
        status = "pass-replacement-policy"
        resolution_mode = "replacement-policy"
        remaining_blockers = []
    else:
        status = "blocked"
        resolution_mode = "candidate-ready-needs-approval" if replacement["exists"] and candidate_audit["recommendation_matches"] else "missing"
        remaining_blockers = ["requested XR-VITs sibling"]

    return {
        "status": status,
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "requested_path": str(requested_path),
        "resolution_mode": resolution_mode,
        "exact": exact,
        "replacement_candidate": replacement,
        "active_policy": active_policy,
        "active_policy_summary": policy_summary,
        "active_policy_integrity": policy_integrity,
        "candidate_audit": candidate_audit,
        "template": {
            "path": str(template_path),
            "exists": template_path.exists(),
        },
        "reference_artifacts": [
            {
                "path": rel_or_abs(hardware, reference_resolution_path),
                "exists": reference_resolution_path.exists(),
            },
            {
                "path": rel_or_abs(hardware, unblock_packet_path),
                "exists": unblock_packet_path.exists(),
            },
        ],
        "decision_options": decision_options(hgtxr, requested_path, replacement_path),
        "current_gate": current_gate,
        "remaining_blockers": remaining_blockers,
        "next_actions": [
            "Restore exact /home/kjm26/project/PRJXR/XR-VITs checkout.",
            "Or explicitly approve XR_Accel as replacement with approved_by and reason metadata.",
        ],
        "safety": {
            "executes_commands": False,
            "executes_network": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# XR-VITs Gate Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- resolution_mode: `{audit['resolution_mode']}`",
        f"- requested_path: `{audit['requested_path']}`",
        f"- exact_exists: `{audit['exact']['exists']}`",
        f"- replacement_path: `{audit['replacement_candidate']['path']}`",
        f"- replacement_exists: `{audit['replacement_candidate']['exists']}`",
        f"- active_policy_status: `{audit['active_policy']['status']}`",
        f"- active_policy_integrity_status: `{audit['active_policy_integrity']['status']}`",
        f"- policy_fingerprint: `{audit['active_policy_integrity'].get('policy_fingerprint')}`",
        f"- candidate_audit_fingerprint: `{audit['active_policy_integrity'].get('candidate_audit_fingerprint')}`",
        f"- candidate_audit_status: `{audit['candidate_audit']['status']}`",
        f"- candidate_recommendation_matches: `{audit['candidate_audit']['recommendation_matches']}`",
        "",
        "## Reference Artifacts",
    ]
    for artifact in audit["reference_artifacts"]:
        lines.append(f"- `{artifact['path']}`: `{artifact['exists']}`")
    lines.extend(["", "## Remaining Blockers"])
    if audit["remaining_blockers"]:
        for blocker in audit["remaining_blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Decision Options"])
    for option in audit["decision_options"]:
        lines.append(f"- `{option['id']}` {option['title']}: {option['expected_result']}")
    lines.extend(["", "## Next Actions"])
    for action in audit["next_actions"]:
        lines.append(f"- {action}")
    lines.extend(["", "## Safety"])
    lines.append("- Does not create an XR-VITs replacement policy.")
    lines.append("- Does not write canonical unblock inputs.")
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, default=Path(f"generated/signoff/xr_vits_gate_audit_{DATE_TAG}.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path(f"generated/signoff/xr_vits_gate_audit_{DATE_TAG}.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    write_outputs(audit, args.json_out, args.markdown_out)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "resolution_mode": audit["resolution_mode"],
                "exact_exists": audit["exact"]["exists"],
                "replacement_exists": audit["replacement_candidate"]["exists"],
                "active_policy_status": audit["active_policy"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
