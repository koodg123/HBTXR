from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_snapshot_guard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hbtxr_ci_snapshot_guard", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_sha_detection():
    mod = _load_module()
    assert mod._is_zero_sha("")
    assert mod._is_zero_sha("0" * 40)
    assert not mod._is_zero_sha("1" * 40)


def test_snapshot_guard_requires_contract_doc_when_snapshot_changes(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "_resolve_changed_files",
        lambda repo_root, base_sha, head_sha: {"tests/snapshots/output_contracts_v3.json"},
    )
    with pytest.raises(SystemExit):
        mod.enforce_snapshot_guard(
            repo_root=Path("."),
            base_sha="a" * 40,
            head_sha="b" * 40,
            snapshot_path=Path("tests/snapshots/output_contracts_v3.json"),
            contract_doc_path=Path("docs/prj/20260408_190000_tracker_dataset_output_contracts.md"),
        )


def test_snapshot_guard_allows_doc_and_snapshot_together(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "_resolve_changed_files",
        lambda repo_root, base_sha, head_sha: {
            "tests/snapshots/output_contracts_v3.json",
            "docs/prj/20260408_190000_tracker_dataset_output_contracts.md",
        },
    )
    mod.enforce_snapshot_guard(
        repo_root=Path("."),
        base_sha="a" * 40,
        head_sha="b" * 40,
        snapshot_path=Path("tests/snapshots/output_contracts_v3.json"),
        contract_doc_path=Path("docs/prj/20260408_190000_tracker_dataset_output_contracts.md"),
    )
