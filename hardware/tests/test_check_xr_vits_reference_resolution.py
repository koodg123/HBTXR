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

import check_xr_vits_reference_resolution as resolution_tool  # noqa: E402
import create_xr_vits_replacement_policy  # noqa: E402


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_candidate_audit(root: Path, replacement: Path, *, status: str = "candidate-found") -> Path:
    path = root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL
    write_json(
        path,
        {
            "status": status,
            "requested_path": str(create_xr_vits_replacement_policy.default_requested_path(root).resolve()),
            "approved_replacement": False,
            "recommendation": {
                "role": "candidate-xr-accel",
                "path": str(replacement.resolve()),
                "score": 99,
            },
        },
    )
    return path


def write_integrity_policy(root: Path, replacement: Path) -> None:
    audit_path = write_candidate_audit(root, replacement)
    audit = create_xr_vits_replacement_policy.load_candidate_audit(audit_path)
    policy = create_xr_vits_replacement_policy.build_policy(
        requested_path=create_xr_vits_replacement_policy.default_requested_path(root).resolve(),
        replacement_path=replacement.resolve(),
        approved_by="unit-test",
        approved_at="2026-06-10T00:00:00Z",
        reason="unit test approval",
        reason_code=create_xr_vits_replacement_policy.DEFAULT_REASON_CODE,
        candidate_audit_rel=create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
        candidate_audit_path=audit_path.resolve(),
        audit=audit,
    )
    write_json(root / create_xr_vits_replacement_policy.DEFAULT_POLICY_REL, policy)


class CheckXrVitsReferenceResolutionTests(unittest.TestCase):
    def test_valid_candidate_needs_approval_without_policy_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            replacement = Path(tmp) / "XR-VIT" / "XR_Accel"
            replacement.mkdir(parents=True)
            write_candidate_audit(root, replacement)

            result = resolution_tool.build_resolution(
                root,
                root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
            )

            self.assertEqual(result["status"], "candidate-ready-needs-approval")
            self.assertFalse(result["resolution_ready"])
            self.assertTrue(result["approval_required"])
            self.assertEqual(result["candidate"]["status"], "pass")
            self.assertIn(f"--root {root.resolve()}", result["commands"]["replacement_dry_run"])
            self.assertIn(f"--root {root.resolve()}", result["commands"]["replacement_approve"])
            self.assertIn(f"--root {root.resolve()}", result["commands"]["exact_restore"][1])
            self.assertFalse((root / create_xr_vits_replacement_policy.DEFAULT_POLICY_REL).exists())
            self.assertFalse(result["safety"]["creates_xr_vits_policy"])

    def test_exact_path_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            exact = Path(tmp) / "XR-VITs"
            exact.mkdir(parents=True)
            result = resolution_tool.build_resolution(
                root,
                root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
            )

            self.assertEqual(result["status"], "exact-ready")
            self.assertTrue(result["resolution_ready"])
            self.assertFalse(result["approval_required"])

    def test_approved_policy_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            replacement = Path(tmp) / "XR-VIT" / "XR_Accel"
            replacement.mkdir(parents=True)
            write_integrity_policy(root, replacement)

            result = resolution_tool.build_resolution(
                root,
                root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
            )

            self.assertEqual(result["status"], "approved-replacement-ready")
            self.assertTrue(result["resolution_ready"])
            self.assertFalse(result["approval_required"])

    def test_missing_candidate_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            result = resolution_tool.build_resolution(
                root,
                root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
            )

            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["resolution_ready"])
            self.assertEqual(result["candidate"]["status"], "fail")

    def test_cli_writes_outputs_and_returns_zero_for_candidate_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            replacement = Path(tmp) / "XR-VIT" / "XR_Accel"
            replacement.mkdir(parents=True)
            write_candidate_audit(root, replacement)
            json_out = root / "generated" / "resolution.json"
            markdown_out = root / "generated" / "resolution.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = resolution_tool.main(
                    [
                        "--root",
                        str(root),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(json_out.read_text())["status"], "candidate-ready-needs-approval")
            self.assertIn("HGTXR XR-VITs Reference Resolution", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
