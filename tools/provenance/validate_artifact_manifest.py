#!/usr/bin/env python3
"""Validate HBTXR external-artifact metadata without accessing the artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
CREDENTIAL_RE = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://[^\s/@]+@|"
    r"(?:token|api[_-]?key|secret|password)\s*[=:]\s*[^\s&]+|"
    r"github_pat_[a-z0-9_]{16,}|gh[opusr]_[a-z0-9]{16,}|"
    r"sk-(?:proj-)?[a-z0-9_-]{16,}|(?:AKIA|ASIA)[0-9A-Z]{16})"
)
SECRET_KEY_RE = re.compile(
    r"(?i)^(?:.*[_-])?(?:password|passwd|pwd|secret|token|api[_-]?key|"
    r"access[_-]?key|private[_-]?key)$"
)
SUBJECT_RE = re.compile(
    r"(?i)(?:subject|participant|patient)[_-]?(?:id|ids|identifier|identifiers)"
)

ARTIFACT_TYPES = {
    "MODEL_CHECKPOINT", "DATASET", "HDF5", "RAW_PREDICTIONS", "RUN_LOG",
    "HLS_PROJECT", "BUILD_OUTPUT", "BITSTREAM", "HWH", "XSA", "DCP", "IP_ARCHIVE",
}
STATUSES = {"DECLARED", "AVAILABLE", "VALIDATED", "BLOCKED", "RETIRED"}
ACCESS = {"PUBLIC", "RESTRICTED", "PRIVATE", "UNAVAILABLE"}
PRIVACY = {"PUBLIC_NO_SUBJECT_DATA", "DEIDENTIFIED", "SUBJECT_LEVEL_SENSITIVE", "UNKNOWN"}
LICENSES = {"VERIFIED", "RESTRICTED", "UNVERIFIED", "NOT_APPLICABLE"}
GENERATION = {"NOT_GENERATED", "GENERATED", "FAILED", "UNKNOWN"}
CLAIMS = {
    "HASH_VERIFIED", "SIZE_VERIFIED", "URI_SYNTAX_VERIFIED",
    "LICENSE_REVIEWED", "PRIVACY_REVIEWED",
}
TOP_KEYS = {
    "schema_version", "artifact_id", "artifact_type", "status", "content",
    "inputs", "environment", "governance", "generation", "validation",
}
OBJECT_KEYS = {
    "content": {"sha256", "byte_count", "uri", "replacement_sha256"},
    "inputs": {"source_sha256", "config_sha256", "split_sha256"},
    "environment": {"architecture_version", "tool_version", "board_version"},
    "governance": {"access", "privacy", "license"},
    "generation": {"command", "status"},
    "validation": {"owner", "date", "claims"},
}


def _contains_sensitive_data(value: Any) -> bool:
    if isinstance(value, str):
        return CREDENTIAL_RE.search(value) is not None or SUBJECT_RE.search(value) is not None
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and (SECRET_KEY_RE.fullmatch(key) or SUBJECT_RE.search(key)))
            or _contains_sensitive_data(key)
            or _contains_sensitive_data(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sensitive_data(item) for item in value)
    return False


def _check_keys(name: str, value: Any, expected: set[str], errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object")
        return {}
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing:
        errors.append(f"{name} missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{name} has unsupported fields: {', '.join(extra)}")
    return value


def _check_sha(name: str, value: Any, errors: list[str], *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        errors.append(f"{name} must be a lowercase SHA-256")


def _check_nonempty(name: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{name} must be a non-empty string")


def _check_uri(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append("content.uri must be a non-empty URI")
        return
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "s3", "file", "artifact"}:
        errors.append(f"content.uri scheme is unsupported: {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        errors.append("content.uri must not contain userinfo")
    if parsed.query or parsed.fragment:
        errors.append("content.uri must not contain query or fragment")
    if parsed.scheme in {"https", "s3", "artifact"} and not parsed.netloc:
        errors.append("content.uri must include an authority")


def validate_manifest(document: Any) -> list[str]:
    errors: list[str] = []
    root = _check_keys("manifest", document, TOP_KEYS, errors)
    if not root:
        return errors
    if _contains_sensitive_data(root):
        errors.append("manifest contains a credential or subject identifier")
    if root.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    artifact_id = root.get("artifact_id")
    if not isinstance(artifact_id, str) or ID_RE.fullmatch(artifact_id) is None:
        errors.append("artifact_id has an invalid format")
    if root.get("artifact_type") not in ARTIFACT_TYPES:
        errors.append("artifact_type is unsupported")
    status = root.get("status")
    if status not in STATUSES:
        errors.append("status is unsupported")

    objects = {
        name: _check_keys(name, root.get(name), keys, errors)
        for name, keys in OBJECT_KEYS.items()
    }
    content = objects["content"]
    inputs = objects["inputs"]
    environment = objects["environment"]
    governance = objects["governance"]
    generation = objects["generation"]
    validation = objects["validation"]

    _check_sha("content.sha256", content.get("sha256"), errors)
    _check_sha("content.replacement_sha256", content.get("replacement_sha256"), errors, nullable=True)
    for name in ("source_sha256", "config_sha256", "split_sha256"):
        _check_sha(f"inputs.{name}", inputs.get(name), errors)
    byte_count = content.get("byte_count")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        errors.append("content.byte_count must be a non-negative integer")
    _check_uri(content.get("uri"), errors)
    for name in ("architecture_version", "tool_version", "board_version"):
        _check_nonempty(f"environment.{name}", environment.get(name), errors)
    if governance.get("access") not in ACCESS:
        errors.append("governance.access is unsupported")
    if governance.get("privacy") not in PRIVACY:
        errors.append("governance.privacy is unsupported")
    if governance.get("license") not in LICENSES:
        errors.append("governance.license is unsupported")
    _check_nonempty("generation.command", generation.get("command"), errors)
    if generation.get("status") not in GENERATION:
        errors.append("generation.status is unsupported")
    _check_nonempty("validation.owner", validation.get("owner"), errors)
    date = validation.get("date")
    try:
        dt.date.fromisoformat(date)
    except (TypeError, ValueError):
        errors.append("validation.date must be a valid YYYY-MM-DD date")
    claims = validation.get("claims")
    if not isinstance(claims, list) or any(claim not in CLAIMS for claim in claims):
        errors.append("validation.claims contains an unsupported claim")
        claims = []
    elif len(claims) != len(set(claims)):
        errors.append("validation.claims must be unique")

    unresolved = governance.get("privacy") in {"SUBJECT_LEVEL_SENSITIVE", "UNKNOWN"} or governance.get("license") == "UNVERIFIED"
    if unresolved and status != "BLOCKED":
        errors.append("unresolved privacy or license requires BLOCKED status")
    if status == "VALIDATED" and not {"HASH_VERIFIED", "SIZE_VERIFIED"}.issubset(set(claims)):
        errors.append("VALIDATED requires HASH_VERIFIED and SIZE_VERIFIED")
    replacement = content.get("replacement_sha256")
    if status == "RETIRED" and (replacement is None or replacement == content.get("sha256")):
        errors.append("RETIRED requires a different replacement_sha256")
    if status != "RETIRED" and replacement is not None:
        errors.append("replacement_sha256 is only valid for RETIRED artifacts")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"manifest read failed: {exc}", file=sys.stderr)
        return 2
    errors = validate_manifest(document)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"validated external artifact manifest: {document['artifact_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
