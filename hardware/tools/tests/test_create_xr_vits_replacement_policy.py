#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import create_xr_vits_replacement_policy as policy_tool  # noqa: E402


class CreateXrVitsReplacementPolicyTests(unittest.TestCase):
    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name) / "XR-VIT"
        root = base / "HGTXR"
        replacement = base / "XR_Accel"
        audit_path = root / "docs" / "resources" / "xr_vits_candidate_audit_2026_06_10.json"
        replacement.mkdir(parents=True)
        audit_path.parent.mkdir(parents=True)
        audit_path.write_text(
            json.dumps(
                {
                    "status": "candidate-found",
                    "requested_path": str(base.parent / "XR-VITs"),
                    "approved_replacement": False,
                    "recommendation": {
                        "role": "candidate-xr-accel",
                        "path": str(replacement),
                        "score": 99,
                        "reason": "highest score for ZCU104/cyclic/DeiT/HLS evidence",
                    },
                    "candidates": [],
                }
            )
        )
        return tmp, root, replacement

    def run_tool(self, argv: list[str]) -> tuple[int, dict[str, object], str]:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = policy_tool.main(argv)
        text = stream.getvalue()
        return code, json.loads(text), text

    def test_rejects_without_explicit_approval_and_does_not_write(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        policy_path = root / "docs" / "resources" / "xr_vits_replacement_policy.json"

        code, payload, _text = self.run_tool(["--root", str(root), "--approved-by", "unit-test", "--reason", "test"])

        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("--approve is required", "\n".join(payload["errors"]))
        self.assertFalse(policy_path.exists())

    def test_writes_policy_when_approved_and_candidate_matches(self) -> None:
        tmp, root, replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        policy_path = root / "docs" / "resources" / "xr_vits_replacement_policy.json"

        code, payload, _text = self.run_tool(
            [
                "--root",
                str(root),
                "--approve",
                "--approved-by",
                "unit-test",
                "--approved-at",
                "2026-06-10T00:00:00Z",
                "--reason",
                "unit test approval",
            ]
        )

        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "pass")
        self.assertTrue(policy_path.exists())
        policy = json.loads(policy_path.read_text())
        self.assertTrue(policy["approved_replacement"])
        self.assertEqual(policy["replacement_path"], str(replacement.resolve()))
        self.assertEqual(policy["replacement_role"], "candidate-xr-accel")
        self.assertEqual(policy["candidate_audit"], policy_tool.DEFAULT_AUDIT_REL)
        self.assertEqual(policy["approved_by"], "unit-test")
        self.assertEqual(policy["approved_at"], "2026-06-10T00:00:00Z")
        self.assertIn("candidate_audit_fingerprint", policy)
        self.assertIn("candidate_audit_recommendation_snapshot", policy)
        self.assertIn("candidate_audit_meta", policy)
        self.assertIn("approval_event", policy)
        self.assertIn("policy_fingerprint", policy)
        self.assertEqual(policy["candidate_audit_meta"]["sha256"], policy["candidate_audit_fingerprint"])
        self.assertEqual(policy["candidate_audit_recommendation_snapshot"]["path"], str(replacement))
        approval_event = policy["approval_event"]
        self.assertEqual(approval_event["event_type"], "xr_vits_replacement_approval")
        self.assertEqual(approval_event["protocol_version"], policy_tool.APPROVAL_PROTOCOL_VERSION)
        self.assertEqual(approval_event["reason_code"], policy_tool.DEFAULT_REASON_CODE)
        self.assertEqual(approval_event["approver_id"], policy["approved_by"])
        self.assertEqual(approval_event["candidate_audit_fingerprint"], policy["candidate_audit_fingerprint"])
        self.assertEqual(approval_event["event_id"], policy_tool.stable_sha256({key: value for key, value in approval_event.items() if key != "event_id"}))
        integrity = policy_tool.validate_policy_integrity(root, policy)
        self.assertEqual(integrity["status"], "pass")
        self.assertEqual(integrity["policy_fingerprint"], policy["policy_fingerprint"])

    def test_dry_run_validates_without_writing(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        policy_path = root / "docs" / "resources" / "xr_vits_replacement_policy.json"

        code, payload, _text = self.run_tool(
            [
                "--root",
                str(root),
                "--approve",
                "--approved-by",
                "unit-test",
                "--reason",
                "unit test approval",
                "--dry-run",
            ]
        )

        self.assertEqual(code, 0)
        self.assertTrue(payload["dry_run"])
        self.assertFalse(policy_path.exists())

    def test_rejects_placeholder_approver_for_active_policy_write(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        policy_path = root / "docs" / "resources" / "xr_vits_replacement_policy.json"

        code, payload, _text = self.run_tool(
            [
                "--root",
                str(root),
                "--approve",
                "--approved-by",
                "<approved-by>",
                "--approved-at",
                "2026-06-10T00:00:00Z",
                "--reason",
                "unit test approval",
            ]
        )

        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("real approver identifier", "\n".join(payload["errors"]))
        self.assertFalse(policy_path.exists())

    def test_dry_run_preview_writes_non_active_review_artifacts(self) -> None:
        tmp, root, replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        policy_path = root / "docs" / "resources" / "xr_vits_replacement_policy.json"
        preview_json = root / "generated" / "signoff" / "xr_vits_replacement_policy_preview.json"
        preview_md = root / "generated" / "signoff" / "xr_vits_replacement_policy_preview.md"

        code, payload, _text = self.run_tool(
            [
                "--root",
                str(root),
                "--approve",
                "--approved-by",
                "<approved-by>",
                "--reason",
                "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff",
                "--dry-run",
                "--preview-json-out",
                str(preview_json),
                "--preview-markdown-out",
                str(preview_md),
            ]
        )

        self.assertEqual(code, 0)
        self.assertTrue(payload["dry_run"])
        self.assertFalse(policy_path.exists())
        self.assertTrue(preview_json.exists())
        self.assertTrue(preview_md.exists())
        preview = json.loads(preview_json.read_text())
        self.assertEqual(preview["status"], "pass")
        self.assertTrue(preview["preview_only"])
        self.assertFalse(preview["active_policy_written"])
        self.assertEqual(preview["active_policy_path"], str(policy_path))
        self.assertEqual(preview["policy"]["replacement_path"], str(replacement.resolve()))
        self.assertEqual(preview["integrity"]["status"], "pass")
        self.assertFalse(preview["safety"]["creates_xr_vits_policy"])
        self.assertIn("Replacement Policy Preview", preview_md.read_text())

    def test_preview_outputs_require_dry_run(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        preview_json = root / "generated" / "signoff" / "xr_vits_replacement_policy_preview.json"

        code, payload, _text = self.run_tool(
            [
                "--root",
                str(root),
                "--approve",
                "--approved-by",
                "unit-test",
                "--reason",
                "unit test approval",
                "--preview-json-out",
                str(preview_json),
            ]
        )

        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("preview outputs require --dry-run", "\n".join(payload["errors"]))
        self.assertFalse(preview_json.exists())

    def test_rejects_replacement_that_does_not_match_audit(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        other = root.parent / "ViT_Accel"
        other.mkdir()

        code, payload, _text = self.run_tool(
            [
                "--root",
                str(root),
                "--replacement-path",
                str(other),
                "--approve",
                "--approved-by",
                "unit-test",
                "--reason",
                "unit test approval",
            ]
        )

        self.assertEqual(code, 1)
        self.assertIn("does not match candidate audit recommendation", "\n".join(payload["errors"]))

    def test_policy_integrity_detects_candidate_audit_drift(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)

        code, payload, _text = self.run_tool(
            [
                "--root",
                str(root),
                "--approve",
                "--approved-by",
                "unit-test",
                "--approved-at",
                "2026-06-10T00:00:00Z",
                "--reason",
                "unit test approval",
            ]
        )

        self.assertEqual(code, 0)
        policy = payload["policy"]
        audit_path = root / policy_tool.DEFAULT_AUDIT_REL
        audit = json.loads(audit_path.read_text())
        audit["recommendation"]["score"] = 1
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

        integrity = policy_tool.validate_policy_integrity(root, policy)

        self.assertEqual(integrity["status"], "fail")
        self.assertIn("candidate_audit_fingerprint mismatch", integrity["errors"])
        self.assertIn("candidate_audit_recommendation_snapshot mismatch", integrity["errors"])

    def test_policy_integrity_detects_approval_event_drift(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)

        code, payload, _text = self.run_tool(
            [
                "--root",
                str(root),
                "--approve",
                "--approved-by",
                "unit-test",
                "--approved-at",
                "2026-06-10T00:00:00Z",
                "--reason",
                "unit test approval",
            ]
        )

        self.assertEqual(code, 0)
        policy = payload["policy"]
        policy["approval_event"]["replacement_path"] = "/tmp/wrong"

        integrity = policy_tool.validate_policy_integrity(root, policy)

        self.assertEqual(integrity["status"], "fail")
        self.assertIn("approval_event replacement_path mismatch", integrity["errors"])

    def test_legacy_policy_integrity_warns(self) -> None:
        tmp, root, replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        legacy_policy = {
            "approved_replacement": True,
            "requested_path": str(policy_tool.default_requested_path(root).resolve()),
            "replacement_path": str(replacement.resolve()),
            "replacement_role": "candidate-xr-accel",
            "candidate_audit": policy_tool.DEFAULT_AUDIT_REL,
            "approved_by": "unit-test",
            "approved_at": "2026-06-10T00:00:00Z",
            "reason": "legacy",
        }

        integrity = policy_tool.validate_policy_integrity(root, legacy_policy)

        self.assertEqual(integrity["status"], "legacy-warning")
        self.assertFalse(integrity["has_integrity_fields"])
        self.assertIn("regenerate policy", integrity["warnings"][0])


if __name__ == "__main__":
    unittest.main()
