#!/usr/bin/env python3
"""Validate the fail-closed HANDOVER source registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ALLOWED_CLASSES = {
    "NO-OP",
    "REFERENCE_ONLY",
    "ADAPT",
    "ACTIVE_CANDIDATE",
    "EXCLUDED",
    "LICENSE_BLOCKED",
}
ALLOWED_RIGHTS = {"VERIFIED", "UNVERIFIED", "NOT_APPLICABLE"}
ALLOWED_LICENSE_STATUS = {"VERIFIED", "UNVERIFIED", "PRESENT_UNREVIEWED"}
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CREDENTIAL_RE = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://[^\s/@]+@|"
    r"(?:token|api[_-]?key|secret|password)\s*[=:]\s*[^\s&]+|"
    r"github_pat_[a-z0-9_]{16,}|gh[opusr]_[a-z0-9]{16,}|"
    r"sk-(?:proj-)?[a-z0-9_-]{16,}|(?:AKIA|ASIA)[0-9A-Z]{16})"
)
SENSITIVE_KEY_RE = re.compile(
    r"(?i)^(?:.*[_-])?(?:password|passwd|pwd|secret|token|api[_-]?key|"
    r"access[_-]?key|private[_-]?key)$"
)

REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "class",
    "source",
    "license",
    "notice",
    "rights",
    "destination",
    "adaptation_summary",
    "owner",
    "reviewer",
    "review_date",
    "validation",
    "ambiguity",
    "promotion_decision",
}
REQUIRED_SOURCE_FIELDS = {"repo", "branch", "revision", "path", "blob_sha256"}
REQUIRED_LICENSE_FIELDS = {"path", "scope", "status"}
REQUIRED_NOTICE_FIELDS = {"status", "required", "text"}
REQUIRED_RIGHT_FIELDS = {"code", "model", "data"}


def _missing_fields(value: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required - value.keys())


def _has_credentials(value: Any) -> bool:
    if isinstance(value, str):
        return CREDENTIAL_RE.search(value) is not None
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and SENSITIVE_KEY_RE.fullmatch(key) is not None)
            or _has_credentials(key)
            or _has_credentials(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_credentials(item) for item in value)
    return False


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve_allowed_repo(workspace_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        return None
    resolved = (workspace_root / value).resolve()
    allowed_roots = (workspace_root.resolve(), (workspace_root.parent / "HANDOVER").resolve())
    if any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        return resolved
    return None


def validate_registry(
    document: Any,
    *,
    workspace_root: Path,
    require_local: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["registry root must be an object"]
    if document.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if _has_credentials(document):
        errors.append("registry contains a credential-like key, URI, or value")
    candidates = document.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("candidates must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        prefix = f"candidates[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in _missing_fields(candidate, REQUIRED_CANDIDATE_FIELDS):
            errors.append(f"{prefix} missing field: {field}")

        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            errors.append(f"{prefix}.candidate_id must be a non-empty string")
        elif candidate_id in seen_ids:
            errors.append(f"{prefix}.candidate_id is duplicated: {candidate_id}")
        else:
            seen_ids.add(candidate_id)

        candidate_class = candidate.get("class")
        if candidate_class not in ALLOWED_CLASSES:
            errors.append(f"{prefix}.class is unknown: {candidate_class!r}")

        source = candidate.get("source")
        if not isinstance(source, dict):
            errors.append(f"{prefix}.source must be an object")
            source = {}
        for field in _missing_fields(source, REQUIRED_SOURCE_FIELDS):
            errors.append(f"{prefix}.source missing field: {field}")
        revision = source.get("revision")
        if not isinstance(revision, str) or SHA1_RE.fullmatch(revision) is None:
            errors.append(f"{prefix}.source.revision must be a 40-character lowercase hex SHA")
        blob_hash = source.get("blob_sha256")
        if not isinstance(blob_hash, str) or SHA256_RE.fullmatch(blob_hash) is None:
            errors.append(f"{prefix}.source.blob_sha256 must be a 64-character lowercase hex SHA-256")
        if not _safe_relative_path(source.get("path")):
            errors.append(f"{prefix}.source.path must be a safe relative path")
        if not isinstance(source.get("branch"), str) or not source["branch"].strip():
            errors.append(f"{prefix}.source.branch must be a non-empty string")
        repo_path = _resolve_allowed_repo(workspace_root, source.get("repo"))
        if repo_path is None:
            errors.append(
                f"{prefix}.source.repo must resolve within the HBTXR workspace or sibling HANDOVER"
            )

        license_record = candidate.get("license")
        if not isinstance(license_record, dict):
            errors.append(f"{prefix}.license must be an object")
            license_record = {}
        for field in _missing_fields(license_record, REQUIRED_LICENSE_FIELDS):
            errors.append(f"{prefix}.license missing field: {field}")
        license_status = license_record.get("status")
        if license_status not in ALLOWED_LICENSE_STATUS:
            errors.append(f"{prefix}.license.status is invalid: {license_status!r}")
        if not isinstance(license_record.get("scope"), str) or not license_record["scope"].strip():
            errors.append(f"{prefix}.license.scope must be a non-empty string")
        license_path = license_record.get("path")
        if license_path is not None and not _safe_relative_path(license_path):
            errors.append(f"{prefix}.license.path must be null or a safe relative path")

        notice = candidate.get("notice")
        if not isinstance(notice, dict):
            errors.append(f"{prefix}.notice must be an object")
            notice = {}
        for field in _missing_fields(notice, REQUIRED_NOTICE_FIELDS):
            errors.append(f"{prefix}.notice missing field: {field}")
        if not isinstance(notice.get("required"), bool):
            errors.append(f"{prefix}.notice.required must be boolean")
        for field in ("status", "text"):
            if not isinstance(notice.get(field), str) or not notice[field].strip():
                errors.append(f"{prefix}.notice.{field} must be a non-empty string")

        rights = candidate.get("rights")
        if not isinstance(rights, dict):
            errors.append(f"{prefix}.rights must be an object")
            rights = {}
        for field in _missing_fields(rights, REQUIRED_RIGHT_FIELDS):
            errors.append(f"{prefix}.rights missing field: {field}")
        for field in REQUIRED_RIGHT_FIELDS:
            if rights.get(field) not in ALLOWED_RIGHTS:
                errors.append(f"{prefix}.rights.{field} is invalid: {rights.get(field)!r}")

        for field in ("destination", "adaptation_summary", "owner", "reviewer", "review_date"):
            if not isinstance(candidate.get(field), str) or not candidate[field].strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
        if isinstance(candidate.get("review_date"), str) and DATE_RE.fullmatch(candidate["review_date"]) is None:
            errors.append(f"{prefix}.review_date must use YYYY-MM-DD")
        ambiguity = candidate.get("ambiguity")
        if not isinstance(ambiguity, list) or any(
            not isinstance(item, str) or not item.strip() for item in ambiguity
        ):
            errors.append(f"{prefix}.ambiguity must be a list of non-empty strings")
        validation = candidate.get("validation")
        if not isinstance(validation, dict):
            errors.append(f"{prefix}.validation must be an object")
        else:
            for field in ("status", "method"):
                if not isinstance(validation.get(field), str) or not validation[field].strip():
                    errors.append(f"{prefix}.validation.{field} must be a non-empty string")
        if candidate.get("promotion_decision") not in {"BLOCKED", "APPROVED"}:
            errors.append(f"{prefix}.promotion_decision must be BLOCKED or APPROVED")

        unresolved = license_status != "VERIFIED" or any(
            rights.get(field) == "UNVERIFIED" for field in REQUIRED_RIGHT_FIELDS
        )
        if candidate_class == "ACTIVE_CANDIDATE" and unresolved:
            errors.append(f"{prefix} ACTIVE_CANDIDATE has unresolved license or rights")
        if unresolved and candidate.get("promotion_decision") != "BLOCKED":
            errors.append(f"{prefix} unresolved license or rights must keep promotion BLOCKED")

        if repo_path is not None:
            source_path = (repo_path / str(source.get("path", ""))).resolve()
            if not source_path.is_relative_to(repo_path):
                errors.append(f"{prefix}.source.path resolves outside source.repo")
                continue
            if source_path.is_file():
                actual_hash = _sha256(source_path)
                if actual_hash != blob_hash:
                    errors.append(
                        f"{prefix}.source.blob_sha256 mismatch: expected {blob_hash}, got {actual_hash}"
                    )
                if license_path is not None:
                    resolved_license = (repo_path / license_path).resolve()
                    if not resolved_license.is_relative_to(repo_path):
                        errors.append(f"{prefix}.license.path resolves outside source.repo")
                    elif not resolved_license.is_file():
                        errors.append(f"{prefix}.license.path does not exist: {license_path}")
                if require_local:
                    try:
                        local_revision = subprocess.run(
                            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout.strip()
                        committed_blob = subprocess.run(
                            ["git", "-C", str(repo_path), "show", f"{revision}:{source['path']}"],
                            check=True,
                            capture_output=True,
                        ).stdout
                    except (OSError, subprocess.CalledProcessError) as exc:
                        errors.append(f"{prefix}.source.repo revision check failed: {exc}")
                    else:
                        if local_revision != revision:
                            errors.append(
                                f"{prefix}.source.revision mismatch: expected {revision}, got {local_revision}"
                            )
                        committed_hash = _sha256_bytes(committed_blob)
                        if committed_hash != blob_hash:
                            errors.append(
                                f"{prefix}.source.blob_sha256 is not the blob at declared revision: "
                                f"expected {blob_hash}, got {committed_hash}"
                            )
            elif require_local:
                errors.append(f"{prefix}.source.path does not exist: {source_path}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--require-local", action="store_true")
    args = parser.parse_args(argv)

    try:
        document = json.loads(args.registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"registry read failed: {exc}", file=sys.stderr)
        return 2

    errors = validate_registry(
        document,
        workspace_root=args.workspace_root.resolve(),
        require_local=args.require_local,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"validated {len(document['candidates'])} HANDOVER source candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
