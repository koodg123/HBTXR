#!/usr/bin/env python3
"""Python fallback for the installed CRLF/jq codebase-recon validator."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def resolve_evidence(value: str, repo_root: Path, artifact_dir: Path) -> bool:
    candidate = re.sub(r":[0-9]+$", "", value)
    path = Path(candidate)
    if path.is_absolute():
        return path.exists()
    return (repo_root / path).exists() or (artifact_dir / path).exists()


def validate(artifact: Path) -> None:
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "codebase-recon.v1"
    assert data.get("mode") in {"baseline", "delta"}
    assert nonempty_text(data.get("commit"))

    flows = data.get("flows")
    assert isinstance(flows, list)
    for flow in flows:
        assert isinstance(flow, dict)
        assert all(nonempty_text(flow.get(key)) for key in ("entry", "domain", "integration", "tests"))

    claims = data.get("claims")
    assert isinstance(claims, list)
    repo_root = artifact.parent.parents[2]
    for claim in claims:
        assert isinstance(claim, dict)
        assert claim.get("kind") in {"fact", "inference", "unknown"}
        assert nonempty_text(claim.get("text"))
        assert claim.get("confidence") in {"high", "medium", "low"}
        evidence = claim.get("evidence")
        assert isinstance(evidence, list) and all(nonempty_text(item) for item in evidence)
        if claim["kind"] != "unknown":
            assert evidence
        for item in evidence:
            assert resolve_evidence(item, repo_root, artifact.parent), f"missing evidence: {item}"

    coverage = data.get("coverage")
    assert isinstance(coverage, dict)
    for key in ("inspected", "uninspected"):
        values = coverage.get(key)
        assert isinstance(values, list) and values and all(nonempty_text(item) for item in values)

    if data["mode"] == "baseline":
        assert flows
        assert data.get("prior_recon") in {None, ""}
    else:
        assert nonempty_text(data.get("prior_recon"))
        assert data.get("baseline_verified") is True
        delta = data.get("delta")
        assert isinstance(delta, list) and delta
        for item in delta:
            assert nonempty_text(item.get("path")) and nonempty_text(item.get("change"))
        assert resolve_evidence(data["prior_recon"], repo_root, artifact.parent)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    validate(artifact)
    print(f"valid codebase-recon.v1: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
