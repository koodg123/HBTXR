from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "tools" / "provenance" / "validate_source_registry.py"
SPEC = importlib.util.spec_from_file_location("validate_source_registry", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceRegistryValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        repo = self.root / "source"
        repo.mkdir()
        (repo / "README.md").write_text("source\n", encoding="utf-8")
        self.source_hash = hashlib.sha256((repo / "README.md").read_bytes()).hexdigest()
        self.candidate = {
            "candidate_id": "EXAMPLE",
            "class": "REFERENCE_ONLY",
            "source": {
                "repo": "source",
                "branch": "main",
                "revision": "a" * 40,
                "path": "README.md",
                "blob_sha256": self.source_hash,
            },
            "license": {"path": None, "scope": "reference", "status": "UNVERIFIED"},
            "notice": {"status": "PENDING", "required": True, "text": "Review notice."},
            "rights": {"code": "UNVERIFIED", "model": "NOT_APPLICABLE", "data": "NOT_APPLICABLE"},
            "destination": "none",
            "adaptation_summary": "Reference only.",
            "owner": "maintainer",
            "reviewer": "reviewer",
            "review_date": "2026-07-15",
            "validation": {"status": "HASH_VERIFIED", "method": "sha256"},
            "ambiguity": ["license"],
            "promotion_decision": "BLOCKED",
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def validate(self, candidate: dict) -> list[str]:
        return MODULE.validate_registry(
            {"schema_version": 1, "candidates": [candidate]},
            workspace_root=self.root,
            require_local=False,
        )

    def test_valid_blocked_reference_passes(self) -> None:
        self.assertEqual([], self.validate(self.candidate))

    def test_missing_required_field_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        del candidate["owner"]
        self.assertTrue(any("missing field: owner" in error for error in self.validate(candidate)))

    def test_unknown_class_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["class"] = "MAYBE"
        self.assertTrue(any("class is unknown" in error for error in self.validate(candidate)))

    def test_hash_mismatch_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["source"]["blob_sha256"] = "0" * 64
        self.assertTrue(any("blob_sha256 mismatch" in error for error in self.validate(candidate)))

    def test_credential_like_value_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["source"]["repo"] = "https://user:secret@example.invalid/repo"
        errors = MODULE.validate_registry(
            {"schema_version": 1, "candidates": [candidate]},
            workspace_root=self.root,
        )
        self.assertTrue(any("credential-like" in error for error in errors))

    def test_token_as_uri_userinfo_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["source"]["repo"] = "https://ghp_abcdefghijklmnopqrstuvwxyz@example.invalid/repo"
        errors = MODULE.validate_registry(
            {"schema_version": 1, "candidates": [candidate]},
            workspace_root=self.root,
        )
        self.assertTrue(any("credential-like" in error for error in errors))

    def test_raw_secret_token_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["ambiguity"] = ["sk-proj-abcdefghijklmnopqrstuvwxyz"]
        errors = MODULE.validate_registry(
            {"schema_version": 1, "candidates": [candidate]},
            workspace_root=self.root,
        )
        self.assertTrue(any("credential-like" in error for error in errors))

    def test_aws_temporary_access_key_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["ambiguity"] = ["ASIAIOSFODNN7EXAMPLE"]
        errors = MODULE.validate_registry(
            {"schema_version": 1, "candidates": [candidate]},
            workspace_root=self.root,
        )
        self.assertTrue(any("credential-like" in error for error in errors))

    def test_active_candidate_with_unresolved_license_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["class"] = "ACTIVE_CANDIDATE"
        self.assertTrue(any("ACTIVE_CANDIDATE" in error for error in self.validate(candidate)))

    def test_unresolved_rights_cannot_be_approved(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["promotion_decision"] = "APPROVED"
        self.assertTrue(any("must keep promotion BLOCKED" in error for error in self.validate(candidate)))

    def test_missing_local_license_path_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["license"]["path"] = "LICENSE"
        candidate["license"]["status"] = "PRESENT_UNREVIEWED"
        self.assertTrue(any("license.path does not exist" in error for error in self.validate(candidate)))

    def test_repo_escape_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["source"]["repo"] = "../../etc"
        self.assertTrue(any("source.repo must resolve" in error for error in self.validate(candidate)))

    def test_validation_record_requires_status_and_method(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["validation"] = {"status": "HASH_VERIFIED"}
        self.assertTrue(any("validation.method" in error for error in self.validate(candidate)))

    def test_source_symlink_cannot_escape_repo(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("outside\n", encoding="utf-8")
        (self.root / "source" / "escape.txt").symlink_to(outside)
        candidate = copy.deepcopy(self.candidate)
        candidate["source"]["path"] = "escape.txt"
        candidate["source"]["blob_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
        self.assertTrue(any("source.path resolves outside" in error for error in self.validate(candidate)))

    def test_license_symlink_cannot_escape_repo(self) -> None:
        outside = self.root / "outside-license.txt"
        outside.write_text("license\n", encoding="utf-8")
        (self.root / "source" / "LICENSE").symlink_to(outside)
        candidate = copy.deepcopy(self.candidate)
        candidate["license"] = {
            "path": "LICENSE",
            "scope": "unreviewed",
            "status": "PRESENT_UNREVIEWED",
        }
        self.assertTrue(any("license.path resolves outside" in error for error in self.validate(candidate)))

    def test_require_local_requires_git_revision(self) -> None:
        errors = MODULE.validate_registry(
            {"schema_version": 1, "candidates": [self.candidate]},
            workspace_root=self.root,
            require_local=True,
        )
        self.assertTrue(any("revision check failed" in error for error in errors))

    def test_credential_like_mapping_key_fails(self) -> None:
        document = {
            "schema_version": 1,
            "ghp_abcdefghijklmnopqrstuvwxyz": "redacted",
            "candidates": [self.candidate],
        }
        errors = MODULE.validate_registry(document, workspace_root=self.root)
        self.assertTrue(any("credential-like key" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
