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

import audit_reference_inputs as audit_tool  # noqa: E402


class AuditReferenceInputsTests(unittest.TestCase):
    def test_build_audit_reports_candidates_without_approving_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xr_vit = Path(tmp) / "XR-VIT"
            hgpipe = xr_vit / "HGPIPE"
            hgpipe.mkdir(parents=True)
            (hgpipe / "DeiT-Tiny C-Syn Results.png").write_bytes(b"png")
            (xr_vit / "XR_Accel" / "hls").mkdir(parents=True)

            audit = audit_tool.build_audit(xr_vit)

            self.assertEqual(audit["status"], "needs-reference-input")
            self.assertFalse(audit["approved_replacements"])
            self.assertEqual(audit["blocker_count"], 2)
            self.assertTrue(any(candidate["role"] == "candidate-hgpipe" and candidate["exists"] for candidate in audit["image_candidates"]))
            self.assertTrue(any(candidate["role"] == "candidate-xr-accel" and candidate["exists"] for candidate in audit["hls_code_candidates"]))

    def test_build_audit_passes_when_requested_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xr_vit = root / "XR-VIT"
            image = xr_vit / "PAPER_PRJXR" / "05_RESOURCES" / "DeiT-Tiny C-Syn Results.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            (root / "XR-VITs").mkdir()

            audit = audit_tool.build_audit(xr_vit)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["blocker_count"], 0)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xr_vit = Path(tmp) / "XR-VIT"
            (xr_vit / "HGPIPE").mkdir(parents=True)
            json_out = Path(tmp) / "audit.json"
            markdown_out = Path(tmp) / "audit.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = audit_tool.main(
                    [
                        "--xr-vit-root",
                        str(xr_vit),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(json.loads(json_out.read_text())["status"], "needs-reference-input")
            self.assertIn("HGTXR Reference Input Audit", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
