from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "tools" / "provenance" / "validate_artifact_manifest.py"
EXAMPLE_PATH = ROOT / "docs" / "provenance" / "examples" / "external-artifact-manifest.example.json"
SPEC = importlib.util.spec_from_file_location("validate_artifact_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ArtifactManifestValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))

    def errors(self, update) -> list[str]:
        manifest = copy.deepcopy(self.manifest)
        update(manifest)
        return MODULE.validate_manifest(manifest)

    def test_example_passes(self) -> None:
        self.assertEqual([], MODULE.validate_manifest(self.manifest))

    def test_missing_hash_fails(self) -> None:
        errors = self.errors(lambda value: value["inputs"].pop("source_sha256"))
        self.assertTrue(any("missing fields" in error for error in errors))

    def test_bad_hash_fails(self) -> None:
        errors = self.errors(lambda value: value["content"].update(sha256="ABC"))
        self.assertTrue(any("lowercase SHA-256" in error for error in errors))

    def test_negative_size_fails(self) -> None:
        errors = self.errors(lambda value: value["content"].update(byte_count=-1))
        self.assertTrue(any("non-negative integer" in error for error in errors))

    def test_credential_uri_fails(self) -> None:
        errors = self.errors(
            lambda value: value["content"].update(uri="https://user:secret@example.invalid/file.pt")
        )
        self.assertTrue(any("credential" in error for error in errors))

    def test_subject_identifier_in_uri_fails(self) -> None:
        errors = self.errors(
            lambda value: value["content"].update(uri="s3://bucket/participant_id-42/file.pt")
        )
        self.assertTrue(any("subject identifier" in error for error in errors))

    def test_extra_field_fails(self) -> None:
        errors = self.errors(lambda value: value.update(extra="unsupported"))
        self.assertTrue(any("unsupported fields" in error for error in errors))

    def test_unsupported_claim_fails(self) -> None:
        errors = self.errors(
            lambda value: value["validation"].update(claims=["BEHAVIOR_REPRODUCED"])
        )
        self.assertTrue(any("unsupported claim" in error for error in errors))

    def test_unresolved_governance_requires_blocked(self) -> None:
        errors = self.errors(lambda value: value["governance"].update(license="UNVERIFIED"))
        self.assertTrue(any("requires BLOCKED" in error for error in errors))

    def test_validated_requires_hash_and_size_claims(self) -> None:
        errors = self.errors(lambda value: value.update(status="VALIDATED"))
        self.assertTrue(any("HASH_VERIFIED and SIZE_VERIFIED" in error for error in errors))

    def test_retired_replacement_must_differ(self) -> None:
        def update(value) -> None:
            value["status"] = "RETIRED"
            value["content"]["replacement_sha256"] = value["content"]["sha256"]

        errors = self.errors(update)
        self.assertTrue(any("different replacement_sha256" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
