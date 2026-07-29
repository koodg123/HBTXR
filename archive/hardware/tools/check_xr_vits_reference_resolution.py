#!/usr/bin/env python3
"""Check XR-VITs reference resolution without creating approval policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import create_xr_vits_replacement_policy
from check_final_blocker_closure_readiness import check_active_policy
from validate_pynq_smoke_result import load_json


DATE_TAG = "2026_06_10"


def default_hgtxr_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_exact(root: Path) -> dict[str, Any]:
    requested = create_xr_vits_replacement_policy.default_requested_path(root).resolve()
    exists = requested.is_dir()
    return {
        "status": "pass" if exists else "missing",
        "path": str(requested),
        "would_clear": exists,
        "errors": [] if exists else [f"requested XR-VITs path missing: {requested}"],
    }


def check_candidate_audit(root: Path, audit_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    payload: dict[str, Any] | None = None
    try:
        loaded = load_json(audit_path)
        if not isinstance(loaded, dict):
            errors.append("candidate audit root is not object")
        else:
            payload = loaded
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"candidate audit invalid: {exc}")

    requested = create_xr_vits_replacement_policy.default_requested_path(root).resolve()
    recommendation: dict[str, Any] = {}
    replacement_path = Path("")
    if payload is not None:
        if payload.get("status") != "candidate-found":
            errors.append(f"candidate audit status is not candidate-found: {payload.get('status')}")
        if payload.get("requested_path") != str(requested):
            errors.append(f"candidate audit requested_path mismatch: {payload.get('requested_path')}")
        if payload.get("approved_replacement") is not False:
            errors.append("candidate audit approved_replacement must be false")
        raw_recommendation = payload.get("recommendation")
        if not isinstance(raw_recommendation, dict):
            errors.append("candidate audit recommendation missing")
        else:
            recommendation = raw_recommendation
            replacement_path = Path(str(recommendation.get("path", ""))).resolve()
            if not replacement_path.is_dir():
                errors.append(f"recommended replacement path missing or not directory: {replacement_path}")
            if not str(recommendation.get("role", "")).strip():
                errors.append("recommendation role missing")
            try:
                score = int(recommendation.get("score", 0))
            except (TypeError, ValueError):
                score = 0
            if score <= 0:
                errors.append(f"recommendation score must be positive: {recommendation.get('score')}")

    return {
        "status": "pass" if not errors else "fail",
        "path": str(audit_path),
        "would_clear_after_approval": not errors,
        "replacement_path": str(replacement_path) if recommendation else "",
        "replacement_role": str(recommendation.get("role", "")) if recommendation else "",
        "score": recommendation.get("score", 0) if recommendation else 0,
        "errors": errors,
    }


def build_resolution(root: Path, candidate_audit_path: Path) -> dict[str, Any]:
    root = root.resolve()
    exact = check_exact(root)
    active_policy = check_active_policy(root)
    candidate = check_candidate_audit(root, candidate_audit_path.resolve())

    if exact["would_clear"]:
        status = "exact-ready"
        resolution_ready = True
        approval_required = False
    elif active_policy["would_clear"]:
        status = "approved-replacement-ready"
        resolution_ready = True
        approval_required = False
    elif candidate["would_clear_after_approval"]:
        status = "candidate-ready-needs-approval"
        resolution_ready = False
        approval_required = True
    else:
        status = "blocked"
        resolution_ready = False
        approval_required = True

    requested = create_xr_vits_replacement_policy.default_requested_path(root).resolve()
    replacement = candidate["replacement_path"] or str(create_xr_vits_replacement_policy.default_replacement_path(root).resolve())
    return {
        "status": status,
        "root": str(root),
        "requested_path": str(requested),
        "resolution_ready": resolution_ready,
        "approval_required": approval_required,
        "exact": exact,
        "active_policy": active_policy,
        "candidate": candidate,
        "commands": {
            "exact_restore": [
                f"restore or mount requested tree at {requested}",
                f"python3 tools/run_third_goal_final_signoff.py --root {root} --allow-blocked",
            ],
            "replacement_dry_run": (
                f"python3 tools/run_third_goal_final_signoff.py --root {root} --allow-blocked "
                "--approve-xr-vits-replacement --dry-run-xr-vits-replacement "
                "--xr-vits-approved-by <approved-by> "
                "--xr-vits-replacement-reason \"Approve XR_Accel as XR-VITs replacement\" "
                f"--xr-vits-replacement-path {replacement}"
            ),
            "replacement_approve": (
                f"python3 tools/run_third_goal_final_signoff.py --root {root} --allow-blocked "
                "--approve-xr-vits-replacement "
                "--xr-vits-approved-by <approved-by> "
                "--xr-vits-replacement-reason \"Approve XR_Accel as XR-VITs replacement\" "
                f"--xr-vits-replacement-path {replacement}"
            ),
        },
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(resolution: dict[str, Any]) -> str:
    lines = [
        "# HGTXR XR-VITs Reference Resolution",
        "",
        f"- status: `{resolution['status']}`",
        f"- root: `{resolution['root']}`",
        f"- requested_path: `{resolution['requested_path']}`",
        f"- resolution_ready: `{resolution['resolution_ready']}`",
        f"- approval_required: `{resolution['approval_required']}`",
        "",
        "## Evidence",
        "",
        f"- exact: `{resolution['exact']['status']}`",
        f"- active_policy: `{resolution['active_policy']['status']}`",
        f"- candidate: `{resolution['candidate']['status']}`",
        f"- candidate_role: `{resolution['candidate']['replacement_role']}`",
        f"- candidate_path: `{resolution['candidate']['replacement_path']}`",
        "",
        "## Commands",
        "",
        "```sh",
        resolution["commands"]["replacement_dry_run"],
        resolution["commands"]["replacement_approve"],
        "```",
        "",
        "## Safety",
        "",
        "- Does not execute commands.",
        "- Does not create board smoke results.",
        "- Does not create XR-VITs replacement policy.",
        "- Does not write canonical unblock inputs.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check XR-VITs reference resolution without side effects.")
    parser.add_argument("--root", type=Path, default=default_hgtxr_root())
    parser.add_argument("--candidate-audit", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    candidate_audit = args.candidate_audit or root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL
    resolution = build_resolution(root, candidate_audit)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(resolution, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(resolution))
    print(
        f"[xr-vits-reference-resolution] status={resolution['status']} "
        f"approval_required={resolution['approval_required']}"
    )
    return 0 if resolution["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
