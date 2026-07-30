#!/usr/bin/env python3
"""Create an explicit XR-VITs replacement policy after approval."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_AUDIT_REL = "docs/resources/xr_vits_candidate_audit_2026_06_10.json"
DEFAULT_POLICY_REL = "docs/resources/xr_vits_replacement_policy.json"
APPROVAL_PROTOCOL_VERSION = "hgtxr-xr-vits-replacement-v1"
DEFAULT_REASON_CODE = "xr_vits_replacement"
ALLOWED_REASON_CODES = {DEFAULT_REASON_CODE}
PLACEHOLDER_APPROVER_VALUES = {
    "<approved-by>",
    "<name>",
    "<user>",
    "approved-by",
    "name",
    "user",
    "todo",
    "tbd",
    "replace-me",
}
INTEGRITY_FIELD_NAMES = {
    "approval_event",
    "candidate_audit_fingerprint",
    "candidate_audit_meta",
    "candidate_audit_recommendation_snapshot",
    "policy_fingerprint",
}


def default_hgtxr_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_requested_path(root: Path) -> Path:
    return root.parent.parent / "XR-VITs"


def default_replacement_path(root: Path) -> Path:
    return root.parent / "XR_Accel"


def default_approved_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def approved_at_is_zulu(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return True


def approver_is_placeholder(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    return not stripped or lowered in PLACEHOLDER_APPROVER_VALUES or (stripped.startswith("<") and stripped.endswith(">"))


def load_candidate_audit(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"candidate audit is unreadable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("candidate audit root is not an object")
    return payload


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def stable_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def candidate_audit_fingerprint(audit: dict[str, Any]) -> str:
    return stable_sha256(audit)


def candidate_audit_meta(path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": stat.st_size,
        "sha256": candidate_audit_fingerprint(audit),
    }


def candidate_recommendation(audit: dict[str, Any]) -> dict[str, Any]:
    recommendation = audit.get("recommendation")
    if not isinstance(recommendation, dict):
        raise ValueError("candidate audit has no recommendation object")
    return recommendation


def candidate_recommendation_snapshot(audit: dict[str, Any]) -> dict[str, Any]:
    recommendation = candidate_recommendation(audit)
    return {
        "path": recommendation.get("path"),
        "role": recommendation.get("role"),
        "score": recommendation.get("score"),
        "reason": recommendation.get("reason"),
        "requested_path": audit.get("requested_path"),
    }


def policy_fingerprint_payload(policy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in policy.items() if key != "policy_fingerprint"}


def policy_fingerprint(policy: dict[str, Any]) -> str:
    return stable_sha256(policy_fingerprint_payload(policy))


def add_policy_integrity_fields(policy: dict[str, Any], *, candidate_audit_path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(policy)
    enriched["candidate_audit_fingerprint"] = candidate_audit_fingerprint(audit)
    enriched["candidate_audit_recommendation_snapshot"] = candidate_recommendation_snapshot(audit)
    enriched["candidate_audit_meta"] = candidate_audit_meta(candidate_audit_path, audit)
    enriched["policy_fingerprint"] = policy_fingerprint(enriched)
    return enriched


def approval_event(
    *,
    requested_path: Path,
    replacement_path: Path,
    approved_by: str,
    approved_at: str,
    reason: str,
    reason_code: str,
    candidate_audit_rel: str,
    candidate_fingerprint: str,
    replacement_role: str,
) -> dict[str, Any]:
    event = {
        "event_type": "xr_vits_replacement_approval",
        "protocol_version": APPROVAL_PROTOCOL_VERSION,
        "approver_id": approved_by,
        "approved_at": approved_at,
        "reason_code": reason_code,
        "reason": reason,
        "replacement_role": replacement_role,
        "requested_path": str(requested_path),
        "replacement_path": str(replacement_path),
        "candidate_audit": candidate_audit_rel,
        "candidate_audit_fingerprint": candidate_fingerprint,
        "generator": "tools/create_xr_vits_replacement_policy.py",
    }
    event["event_id"] = stable_sha256(event)
    return event


def validate_approval_event(policy: dict[str, Any]) -> list[str]:
    event = policy.get("approval_event")
    if not isinstance(event, dict):
        return ["approval_event missing"]
    errors: list[str] = []
    expected = {
        "event_type": "xr_vits_replacement_approval",
        "protocol_version": APPROVAL_PROTOCOL_VERSION,
        "approver_id": policy.get("approved_by"),
        "approved_at": policy.get("approved_at"),
        "reason": policy.get("reason"),
        "replacement_role": policy.get("replacement_role"),
        "requested_path": policy.get("requested_path"),
        "replacement_path": policy.get("replacement_path"),
        "candidate_audit": policy.get("candidate_audit"),
        "candidate_audit_fingerprint": policy.get("candidate_audit_fingerprint"),
        "generator": "tools/create_xr_vits_replacement_policy.py",
    }
    for key, value in expected.items():
        if event.get(key) != value:
            errors.append(f"approval_event {key} mismatch")
    if event.get("reason_code") not in ALLOWED_REASON_CODES:
        errors.append("approval_event reason_code invalid")
    if not approved_at_is_zulu(str(event.get("approved_at", ""))):
        errors.append("approval_event approved_at is not Zulu ISO-8601")
    event_without_id = {key: value for key, value in event.items() if key != "event_id"}
    if event.get("event_id") != stable_sha256(event_without_id):
        errors.append("approval_event event_id mismatch")
    return errors


def resolve_candidate_audit_path(root: Path, policy: dict[str, Any]) -> Path:
    value = policy.get("candidate_audit", DEFAULT_AUDIT_REL)
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def validate_policy_integrity(root: Path, policy: dict[str, Any]) -> dict[str, Any]:
    has_integrity = any(field in policy for field in INTEGRITY_FIELD_NAMES)
    candidate_path = resolve_candidate_audit_path(root, policy)
    result: dict[str, Any] = {
        "status": "legacy-warning",
        "has_integrity_fields": has_integrity,
        "candidate_audit_path": str(candidate_path),
        "errors": [],
        "warnings": [],
    }
    if not has_integrity:
        result["warnings"].append("policy has no candidate audit fingerprint; regenerate policy to bind approval to audit content")
        return result

    if not candidate_path.exists():
        result["errors"].append(f"candidate audit missing: {candidate_path}")
        result["status"] = "fail"
        return result

    try:
        audit = load_candidate_audit(candidate_path)
    except ValueError as exc:
        result["errors"].append(str(exc))
        result["status"] = "fail"
        return result

    current_audit_fingerprint = candidate_audit_fingerprint(audit)
    current_snapshot = candidate_recommendation_snapshot(audit)
    expected_audit_fingerprint = policy.get("candidate_audit_fingerprint")
    expected_snapshot = policy.get("candidate_audit_recommendation_snapshot")
    expected_policy_fingerprint = policy.get("policy_fingerprint")
    current_policy_fingerprint = policy_fingerprint(policy)

    result.update(
        {
            "status": "pass",
            "candidate_audit_fingerprint": current_audit_fingerprint,
            "expected_candidate_audit_fingerprint": expected_audit_fingerprint,
            "candidate_audit_recommendation_snapshot": current_snapshot,
            "expected_candidate_audit_recommendation_snapshot": expected_snapshot,
            "policy_fingerprint": current_policy_fingerprint,
            "expected_policy_fingerprint": expected_policy_fingerprint,
        }
    )
    if expected_audit_fingerprint != current_audit_fingerprint:
        result["errors"].append("candidate_audit_fingerprint mismatch")
    if expected_snapshot != current_snapshot:
        result["errors"].append("candidate_audit_recommendation_snapshot mismatch")
    if expected_policy_fingerprint != current_policy_fingerprint:
        result["errors"].append("policy_fingerprint mismatch")
    result["approval_event"] = policy.get("approval_event")
    result["approval_protocol_version"] = (
        policy.get("approval_event", {}).get("protocol_version")
        if isinstance(policy.get("approval_event"), dict)
        else None
    )
    result["errors"].extend(validate_approval_event(policy))
    if result["errors"]:
        result["status"] = "fail"
    return result


def validate_inputs(
    *,
    approve: bool,
    dry_run: bool,
    requested_path: Path,
    replacement_path: Path,
    approved_by: str,
    approved_at: str,
    reason: str,
    reason_code: str,
    candidate_audit_path: Path,
    candidate_audit_rel: str,
) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    audit: dict[str, Any] | None = None

    if not approve:
        errors.append("--approve is required before writing an active replacement policy")
    if requested_path.exists():
        errors.append(f"requested path already exists; replacement policy is unnecessary: {requested_path}")
    if not replacement_path.exists():
        errors.append(f"replacement path does not exist: {replacement_path}")
    elif not replacement_path.is_dir():
        errors.append(f"replacement path is not a directory: {replacement_path}")
    if not approved_by.strip():
        errors.append("--approved-by is required")
    elif not dry_run and approver_is_placeholder(approved_by):
        errors.append("--approved-by must be a real approver identifier for active policy writes")
    if not approved_at_is_zulu(approved_at):
        errors.append("--approved-at must be UTC ISO-8601 ending with Z")
    if not reason.strip():
        errors.append("--reason is required")
    if reason_code not in ALLOWED_REASON_CODES:
        errors.append(f"--reason-code must be one of: {', '.join(sorted(ALLOWED_REASON_CODES))}")
    if not candidate_audit_path.exists():
        errors.append(f"candidate audit does not exist: {candidate_audit_path}")
    else:
        try:
            audit = load_candidate_audit(candidate_audit_path)
            recommendation = candidate_recommendation(audit)
            recommended_path = recommendation.get("path")
            if Path(str(recommended_path)).resolve() != replacement_path.resolve():
                errors.append(f"replacement path does not match candidate audit recommendation: {recommended_path}")
            if audit.get("requested_path") != str(requested_path):
                errors.append(f"requested path does not match candidate audit: {audit.get('requested_path')}")
            if candidate_audit_rel != DEFAULT_AUDIT_REL:
                errors.append(f"candidate audit reference must remain tracked relative path: {candidate_audit_rel}")
        except ValueError as exc:
            errors.append(str(exc))

    return errors, audit


def build_policy(
    *,
    requested_path: Path,
    replacement_path: Path,
    approved_by: str,
    approved_at: str,
    reason: str,
    reason_code: str,
    candidate_audit_rel: str,
    candidate_audit_path: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    recommendation = candidate_recommendation(audit)
    audit_fp = candidate_audit_fingerprint(audit)
    replacement_role = recommendation.get("role", "candidate-xr-accel")
    policy = {
        "approved_replacement": True,
        "requested_path": str(requested_path),
        "replacement_path": str(replacement_path),
        "replacement_role": replacement_role,
        "candidate_audit": candidate_audit_rel,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "reason": reason,
    }
    policy["approval_event"] = approval_event(
        requested_path=requested_path,
        replacement_path=replacement_path,
        approved_by=approved_by,
        approved_at=approved_at,
        reason=reason,
        reason_code=reason_code,
        candidate_audit_rel=candidate_audit_rel,
        candidate_fingerprint=audit_fp,
        replacement_role=str(replacement_role),
    )
    return add_policy_integrity_fields(policy, candidate_audit_path=candidate_audit_path, audit=audit)


def preview_payload(
    *,
    root: Path,
    policy_path: Path,
    policy: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    integrity = validate_policy_integrity(root, policy)
    return {
        "status": "pass" if integrity.get("status") == "pass" else "fail",
        "preview_only": True,
        "dry_run": dry_run,
        "active_policy_path": str(policy_path),
        "active_policy_written": False,
        "policy": policy,
        "integrity": integrity,
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_preview_markdown(preview: dict[str, Any]) -> str:
    policy = preview["policy"]
    integrity = preview["integrity"]
    lines = [
        "# HGTXR XR-VITs Replacement Policy Preview",
        "",
        f"- status: `{preview['status']}`",
        f"- preview_only: `{preview['preview_only']}`",
        f"- dry_run: `{preview['dry_run']}`",
        f"- active_policy_written: `{preview['active_policy_written']}`",
        f"- active_policy_path: `{preview['active_policy_path']}`",
        f"- requested_path: `{policy.get('requested_path', '')}`",
        f"- replacement_path: `{policy.get('replacement_path', '')}`",
        f"- replacement_role: `{policy.get('replacement_role', '')}`",
        f"- approved_by: `{policy.get('approved_by', '')}`",
        f"- reason: {policy.get('reason', '')}",
        f"- approval_protocol_version: `{policy.get('approval_event', {}).get('protocol_version', '') if isinstance(policy.get('approval_event'), dict) else ''}`",
        f"- approval_reason_code: `{policy.get('approval_event', {}).get('reason_code', '') if isinstance(policy.get('approval_event'), dict) else ''}`",
        f"- candidate_audit_fingerprint: `{policy.get('candidate_audit_fingerprint', '')}`",
        f"- policy_fingerprint: `{policy.get('policy_fingerprint', '')}`",
        f"- integrity_status: `{integrity.get('status', '')}`",
        "",
        "## Recommendation Snapshot",
        "",
    ]
    snapshot = policy.get("candidate_audit_recommendation_snapshot", {})
    if isinstance(snapshot, dict):
        for key in ["path", "role", "score", "reason", "requested_path"]:
            lines.append(f"- {key}: `{snapshot.get(key, '')}`")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Does not execute commands.",
            "- Does not create board smoke results.",
            "- Does not write active replacement policy.",
            "- Does not write canonical unblock inputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_preview_outputs(preview: dict[str, Any], json_out: Path | None, markdown_out: Path | None) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(preview, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_preview_markdown(preview))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create XR-VITs replacement policy only after explicit approval.")
    parser.add_argument("--root", type=Path, default=default_hgtxr_root())
    parser.add_argument("--requested-path", type=Path, default=None)
    parser.add_argument("--replacement-path", type=Path, default=None)
    parser.add_argument("--candidate-audit", default=DEFAULT_AUDIT_REL)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--preview-json-out", type=Path, default=None)
    parser.add_argument("--preview-markdown-out", type=Path, default=None)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--approved-at", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--reason-code", default=DEFAULT_REASON_CODE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    requested_path = (args.requested_path or default_requested_path(root)).resolve()
    replacement_path = (args.replacement_path or default_replacement_path(root)).resolve()
    candidate_audit_rel = str(args.candidate_audit)
    candidate_audit_path = (root / candidate_audit_rel).resolve()
    json_out = args.json_out or root / DEFAULT_POLICY_REL
    approved_at = args.approved_at or default_approved_at()
    if (args.preview_json_out is not None or args.preview_markdown_out is not None) and not args.dry_run:
        result = {
            "status": "fail",
            "policy_path": str(json_out),
            "errors": ["preview outputs require --dry-run to avoid active policy confusion"],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    errors, audit = validate_inputs(
        approve=args.approve,
        dry_run=args.dry_run,
        requested_path=requested_path,
        replacement_path=replacement_path,
        approved_by=args.approved_by,
        approved_at=approved_at,
        reason=args.reason,
        reason_code=args.reason_code,
        candidate_audit_path=candidate_audit_path,
        candidate_audit_rel=candidate_audit_rel,
    )
    if errors:
        result = {
            "status": "fail",
            "policy_path": str(json_out),
            "errors": errors,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    assert audit is not None
    policy = build_policy(
        requested_path=requested_path,
        replacement_path=replacement_path,
        approved_by=args.approved_by,
        approved_at=approved_at,
        reason=args.reason,
        reason_code=args.reason_code,
        candidate_audit_rel=candidate_audit_rel,
        candidate_audit_path=candidate_audit_path,
        audit=audit,
    )
    result = {
        "status": "pass",
        "policy_path": str(json_out),
        "dry_run": args.dry_run,
        "policy": policy,
    }
    preview = preview_payload(root=root, policy_path=json_out, policy=policy, dry_run=args.dry_run)
    if args.preview_json_out is not None or args.preview_markdown_out is not None:
        write_preview_outputs(preview, args.preview_json_out, args.preview_markdown_out)
        result["preview_json_out"] = str(args.preview_json_out) if args.preview_json_out is not None else ""
        result["preview_markdown_out"] = str(args.preview_markdown_out) if args.preview_markdown_out is not None else ""
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.dry_run:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
