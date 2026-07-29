#!/usr/bin/env python3
"""Write a no-side-effect intake report for final unblock candidate inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import audit_final_unblock_candidates
import check_final_blocker_closure_readiness


DATE_TAG = "2026_06_10"
HGTXR_ROOT = Path(__file__).resolve().parents[2]


def build_intake(
    *,
    root: Path,
    c3b_smoke_json: Path | None,
    c3b_require_paths: bool,
    xr_vits_mode: str,
    approved_by: str,
    reason: str,
    replacement_path: Path | None,
) -> dict[str, Any]:
    root = root.resolve()
    candidate_audit = audit_final_unblock_candidates.build_audit(
        root=root,
        c3b_smoke_json=c3b_smoke_json,
        c3b_require_paths=c3b_require_paths,
        xr_vits_mode=xr_vits_mode,
        approved_by=approved_by,
        reason=reason,
        replacement_path=replacement_path,
    )
    closure_readiness = check_final_blocker_closure_readiness.build_readiness(
        root=root,
        c3b_smoke_json=c3b_smoke_json,
        c3b_require_paths=c3b_require_paths,
        xr_vits_mode=xr_vits_mode,
        approved_by=approved_by,
        reason=reason,
        replacement_path=replacement_path,
    )
    dry_run_command = str(closure_readiness.get("dry_run_final_runner_command", ""))
    active_command = str(closure_readiness.get("final_runner_command", ""))
    ready = bool(candidate_audit.get("would_clear_all") and closure_readiness.get("candidate_ready"))
    status = "ready-for-active-unblock" if ready else "blocked"
    blocker_status = build_blocker_status(candidate_audit, closure_readiness)
    operator_sequence = [
        {
            "step": "candidate-audit",
            "command": build_candidate_audit_command(
                root=root,
                c3b_smoke_json=c3b_smoke_json,
                c3b_require_paths=c3b_require_paths,
                xr_vits_mode=xr_vits_mode,
                approved_by=approved_by,
                reason=reason,
                replacement_path=replacement_path,
            ),
            "side_effects": False,
        },
        {
            "step": "dry-run-final-runner",
            "command": dry_run_command,
            "side_effects": False,
        },
        {
            "step": "active-final-runner",
            "command": active_command,
            "side_effects": True,
        },
    ]
    return {
        "status": status,
        "root": str(root),
        "candidate_audit": candidate_audit,
        "closure_readiness": closure_readiness,
        "blocker_status": blocker_status,
        "next_inputs": build_next_inputs(blocker_status),
        "dry_run_final_runner_command": dry_run_command,
        "active_final_runner_command": active_command,
        "operator_sequence": operator_sequence,
        "next_action": next_action(status, candidate_audit, dry_run_command, active_command),
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def build_blocker_status(
    candidate_audit: dict[str, Any],
    closure_readiness: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    current = closure_readiness.get("current", {}) if isinstance(closure_readiness.get("current"), dict) else {}
    candidates = closure_readiness.get("candidates", {}) if isinstance(closure_readiness.get("candidates"), dict) else {}
    current_c3b = current.get("c3b_smoke", {}) if isinstance(current.get("c3b_smoke"), dict) else {}
    candidate_c3b = candidates.get("c3b_smoke", {}) if isinstance(candidates.get("c3b_smoke"), dict) else {}
    current_xr = current.get("xr_vits", {}) if isinstance(current.get("xr_vits"), dict) else {}
    candidate_xr = candidates.get("xr_vits", {}) if isinstance(candidates.get("xr_vits"), dict) else {}
    exact = current_xr.get("exact", {}) if isinstance(current_xr.get("exact"), dict) else {}
    policy = current_xr.get("replacement_policy", {}) if isinstance(current_xr.get("replacement_policy"), dict) else {}
    candidate_ready = {
        "C3b AXIS/DMA physical smoke result": bool(candidate_c3b.get("would_clear")),
        "requested XR-VITs sibling": bool(candidate_xr.get("would_clear")),
    }
    current_ready = {
        "C3b AXIS/DMA physical smoke result": bool(current_c3b.get("would_clear")),
        "requested XR-VITs sibling": bool(current_xr.get("would_clear")),
    }
    remaining = set(str(item) for item in candidate_audit.get("remaining_blockers", []))
    return {
        "C3b AXIS/DMA physical smoke result": {
            "current_status": str(current_c3b.get("status", "unknown")),
            "current_would_clear": current_ready["C3b AXIS/DMA physical smoke result"],
            "canonical_path": str(current_c3b.get("path", "")),
            "candidate_status": str(candidate_c3b.get("status", "unknown")),
            "candidate_would_clear": candidate_ready["C3b AXIS/DMA physical smoke result"],
            "candidate_path": str(candidate_c3b.get("path", "")),
            "ready_for_active_unblock": current_ready["C3b AXIS/DMA physical smoke result"]
            or candidate_ready["C3b AXIS/DMA physical smoke result"],
            "remaining_after_candidates": "C3b AXIS/DMA physical smoke result" in remaining,
            "next_input": "board-produced C3b smoke JSON",
            "next_input_path": str(current_c3b.get("path", "")),
            "unblock_action": "capture ZCU104 C3b smoke, dry-run import it, then import to canonical path",
        },
        "requested XR-VITs sibling": {
            "current_status": str(current_xr.get("status", "unknown")),
            "current_mode": str(current_xr.get("mode", "unknown")),
            "current_would_clear": current_ready["requested XR-VITs sibling"],
            "exact_path": str(exact.get("path", "")),
            "policy_path": str(policy.get("path", "")),
            "candidate_status": str(candidate_xr.get("status", "unknown")),
            "candidate_mode": str(candidate_xr.get("mode", "unknown")),
            "candidate_would_clear": candidate_ready["requested XR-VITs sibling"],
            "candidate_replacement_path": str(candidate_xr.get("replacement_path", "")),
            "ready_for_active_unblock": current_ready["requested XR-VITs sibling"]
            or candidate_ready["requested XR-VITs sibling"],
            "remaining_after_candidates": "requested XR-VITs sibling" in remaining,
            "next_input": "exact XR-VITs directory or approved replacement policy",
            "next_input_path": str(exact.get("path", "")) + " OR " + str(policy.get("path", "")),
            "unblock_action": "restore exact XR-VITs source or approve fingerprint-bound XR_Accel replacement policy",
        },
    }


def build_next_inputs(blocker_status: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "blocker": blocker,
            "input": str(status.get("next_input", "")),
            "path": str(status.get("next_input_path", "")),
            "action": str(status.get("unblock_action", "")),
        }
        for blocker, status in blocker_status.items()
        if not bool(status.get("ready_for_active_unblock"))
    ]


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._:-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_candidate_audit_command(
    *,
    root: Path,
    c3b_smoke_json: Path | None,
    c3b_require_paths: bool,
    xr_vits_mode: str,
    approved_by: str,
    reason: str,
    replacement_path: Path | None,
) -> str:
    parts = [
        "python3",
        "tools/audit_final_unblock_candidates.py",
        "--root",
        str(root),
        "--xr-vits-mode",
        xr_vits_mode,
        "--json-out",
        f"/tmp/final_unblock_candidate_audit_{DATE_TAG}.json",
        "--markdown-out",
        f"/tmp/final_unblock_candidate_audit_{DATE_TAG}.md",
    ]
    if c3b_smoke_json is not None:
        parts.extend(["--c3b-smoke-json", str(c3b_smoke_json)])
    if not c3b_require_paths:
        parts.append("--c3b-no-require-paths")
    if xr_vits_mode == "replacement":
        parts.extend(["--approved-by", approved_by or "<approved-by>"])
        parts.extend(["--reason", reason or "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"])
        if replacement_path is not None:
            parts.extend(["--replacement-path", str(replacement_path)])
    return " ".join(shell_quote(part) for part in parts)


def next_action(status: str, candidate_audit: dict[str, Any], dry_run_command: str, active_command: str) -> str:
    if status == "ready-for-active-unblock":
        return (
            "Run the dry-run final runner command first; if it passes with expected candidate readiness, "
            "run the active final runner command."
        )
    missing = candidate_audit.get("remaining_blockers", [])
    if isinstance(missing, list) and missing:
        return "Supply inputs for: " + ", ".join(str(item) for item in missing)
    if not dry_run_command or not active_command:
        return "Supply both a valid C3b smoke JSON and a valid XR-VITs exact/replacement choice."
    return "Review candidate audit errors before running active unblock."


def render_markdown(intake: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Unblock Intake",
        "",
        f"- status: `{intake['status']}`",
        f"- root: `{intake['root']}`",
        f"- next_action: {intake['next_action']}",
        "",
        "## Candidate Audit",
        "",
        f"- status: `{intake['candidate_audit']['status']}`",
        f"- would_clear_all: `{intake['candidate_audit']['would_clear_all']}`",
        f"- remaining_blockers: `{intake['candidate_audit']['remaining_blockers']}`",
        "",
        "## Blocker Status",
        "",
        "| Blocker | Current | Candidate | Ready | Next Input |",
        "|---|---:|---:|---:|---|",
    ]
    for blocker, status in intake["blocker_status"].items():
        lines.append(
            f"| `{blocker}` | `{status['current_status']}` | `{status['candidate_status']}` | "
            f"`{status['ready_for_active_unblock']}` | `{status['next_input_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Next Inputs",
            "",
        ]
    )
    if intake["next_inputs"]:
        for item in intake["next_inputs"]:
            lines.append(f"- `{item['blocker']}`: {item['action']} -> `{item['path']}`")
    else:
        lines.append("- None; dry-run final runner can be executed before active unblock.")
    lines.extend(
        [
            "",
            "## Closure Readiness",
            "",
            f"- status: `{intake['closure_readiness']['status']}`",
            f"- current_ready: `{intake['closure_readiness']['current_ready']}`",
            f"- candidate_ready: `{intake['closure_readiness']['candidate_ready']}`",
            "",
            "## Dry-Run Final Runner",
            "",
            "```sh",
            intake["dry_run_final_runner_command"] or "# Not ready.",
            "```",
            "",
            "## Active Final Runner",
            "",
            "```sh",
            intake["active_final_runner_command"] or "# Not ready.",
            "```",
            "",
            "## Safety",
            "",
        ]
    )
    for key, value in intake["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write final unblock candidate intake report.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--c3b-smoke-json", type=Path, default=None)
    parser.add_argument("--c3b-no-require-paths", action="store_true")
    parser.add_argument("--xr-vits-mode", choices=["exact", "replacement"], default="replacement")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--replacement-path", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    intake = build_intake(
        root=args.root,
        c3b_smoke_json=args.c3b_smoke_json,
        c3b_require_paths=not args.c3b_no_require_paths,
        xr_vits_mode=args.xr_vits_mode,
        approved_by=args.approved_by,
        reason=args.reason,
        replacement_path=args.replacement_path,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(intake, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(intake))
    print(
        f"[final-unblock-intake] status={intake['status']} "
        f"candidate_ready={intake['closure_readiness']['candidate_ready']}"
    )
    return 0 if intake["status"] == "ready-for-active-unblock" else 1


if __name__ == "__main__":
    raise SystemExit(main())
