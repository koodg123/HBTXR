#!/usr/bin/env python3
"""Write a user-choice packet for resolving the XR-VITs reference gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
DATE_TAG = "2026_06_10"
DEFAULT_CANDIDATE_AUDIT = HGTXR_ROOT / "docs" / "resources" / f"xr_vits_candidate_audit_{DATE_TAG}.json"
DEFAULT_POLICY_TEMPLATE = HGTXR_ROOT / "docs" / "resources" / "xr_vits_replacement_policy.template.json"
DEFAULT_ACTIVE_POLICY = HGTXR_ROOT / "docs" / "resources" / "xr_vits_replacement_policy.json"
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / f"xr_vits_unblock_packet_{DATE_TAG}.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / f"xr_vits_unblock_packet_{DATE_TAG}.md"


def default_requested_path(root: Path) -> Path:
    return root.parent.parent / "XR-VITs"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def build_packet(
    *,
    root: Path,
    requested_path: Path,
    candidate_audit_path: Path,
    policy_template_path: Path,
    active_policy_path: Path,
) -> dict[str, Any]:
    candidate_audit = load_json(candidate_audit_path)
    policy_template = load_optional_json(policy_template_path)
    active_policy = load_optional_json(active_policy_path)
    recommendation = candidate_audit.get("recommendation", {})
    if not isinstance(recommendation, dict):
        recommendation = {}
    candidates = candidate_audit.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    exact_exists = requested_path.exists()
    active_approved = bool(active_policy and active_policy.get("approved_replacement") is True)
    if exact_exists:
        status = "exact-restored"
    elif active_approved:
        status = "replacement-approved"
    else:
        status = "pending-user-choice"

    replacement_path = str(recommendation.get("path", root.parent / "XR_Accel"))
    approved_by_placeholder = "<approved-by>"
    reason = "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"
    return {
        "status": status,
        "root": str(root),
        "requested_path": str(requested_path),
        "requested_path_exists": exact_exists,
        "candidate_audit": str(candidate_audit_path),
        "policy_template": str(policy_template_path),
        "active_policy": str(active_policy_path),
        "active_policy_exists": active_policy_path.exists(),
        "active_policy_approved": active_approved,
        "recommendation": recommendation,
        "top_candidates": [
            {
                "path": candidate.get("path"),
                "role": candidate.get("role"),
                "score": candidate.get("score"),
                "hls_file_count": candidate.get("hls_file_count"),
                "readme_exists": candidate.get("readme_exists"),
            }
            for candidate in candidates[:3]
            if isinstance(candidate, dict)
        ],
        "policy_template_summary": policy_template,
        "safety": [
            "This packet does not create docs/resources/xr_vits_replacement_policy.json.",
            "Active replacement policy requires explicit user approval.",
            "Exact XR-VITs restore remains the strictest path.",
        ],
        "options": [
            {
                "id": "restore-exact-xr-vits",
                "title": "Restore exact XR-VITs checkout",
                "effect": "Final signoff can use the requested source path directly.",
                "commands": [
                    "test -d /home/kjm26/project/PRJXR/XR-VITs",
                    "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked",
                ],
            },
            {
                "id": "approve-xr-accel-replacement",
                "title": "Explicitly approve XR_Accel as XR-VITs replacement",
                "effect": "Creates active replacement policy only after explicit approval metadata is supplied.",
                "commands": [
                    f"python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --replacement-path {replacement_path} --approve --approved-by {approved_by_placeholder} --reason \"{reason}\" --dry-run",
                    f"python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --replacement-path {replacement_path} --approve --approved-by {approved_by_placeholder} --reason \"{reason}\"",
                    "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked",
                ],
            },
        ],
        "next_required_action": "Restore exact XR-VITs or explicitly approve XR_Accel replacement." if status == "pending-user-choice" else "Rerun final signoff.",
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# HGTXR XR-VITs Unblock Packet",
        "",
        f"- status: `{packet['status']}`",
        f"- requested_path: `{packet['requested_path']}`",
        f"- requested_path_exists: `{packet['requested_path_exists']}`",
        f"- active_policy_exists: `{packet['active_policy_exists']}`",
        f"- active_policy_approved: `{packet['active_policy_approved']}`",
        f"- next_required_action: {packet['next_required_action']}",
        "",
        "## Recommendation",
        "",
    ]
    rec = packet["recommendation"]
    if rec:
        lines.append(f"- role: `{rec.get('role', '')}`")
        lines.append(f"- path: `{rec.get('path', '')}`")
        lines.append(f"- score: `{rec.get('score', '')}`")
        lines.append(f"- reason: {rec.get('reason', '')}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Options", ""])
    for option in packet["options"]:
        lines.append(f"### {option['id']}")
        lines.append(f"- title: {option['title']}")
        lines.append(f"- effect: {option['effect']}")
        lines.append("```sh")
        lines.extend(option["commands"])
        lines.append("```")
        lines.append("")
    lines.extend(["## Safety", ""])
    for item in packet["safety"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def write_outputs(packet: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(packet))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write XR-VITs unblock choice packet.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--requested-path", type=Path, default=None)
    parser.add_argument("--candidate-audit", type=Path, default=DEFAULT_CANDIDATE_AUDIT)
    parser.add_argument("--policy-template", type=Path, default=DEFAULT_POLICY_TEMPLATE)
    parser.add_argument("--active-policy", type=Path, default=DEFAULT_ACTIVE_POLICY)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    requested_path = (args.requested_path or default_requested_path(root)).resolve()
    packet = build_packet(
        root=root,
        requested_path=requested_path,
        candidate_audit_path=args.candidate_audit,
        policy_template_path=args.policy_template,
        active_policy_path=args.active_policy,
    )
    write_outputs(packet, args.json_out, args.markdown_out)
    print(
        f"[xr-vits-unblock-packet] status={packet['status']} "
        f"requested_exists={packet['requested_path_exists']} "
        f"policy_approved={packet['active_policy_approved']}"
    )
    return 0 if packet["status"] != "pending-user-choice" else 1


if __name__ == "__main__":
    raise SystemExit(main())
