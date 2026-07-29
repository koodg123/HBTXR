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

import write_req9_deit_image_reference_audit as audit_tool  # noqa: E402


PNG_BYTES = bytes.fromhex("89504e470d0a1a0a") + b"\x00req9-test-png"


def make_tree(tmp: str, include_requested: bool = True) -> Path:
    hgtxr = Path(tmp) / "project" / "PRJXR" / "XR-VIT" / "HGTXR"
    hardware = hgtxr / "hardware"
    xr_vit = hgtxr.parent
    requested = xr_vit / "PAPER_PRJXR" / "05_RESOURCES" / audit_tool.IMAGE_NAME
    hgpipe = xr_vit / "HGPIPE" / audit_tool.IMAGE_NAME
    docs_copy = hardware / "docs" / audit_tool.IMAGE_NAME
    requested.parent.mkdir(parents=True)
    hgpipe.parent.mkdir(parents=True)
    docs_copy.parent.mkdir(parents=True)
    if include_requested:
        requested.write_bytes(PNG_BYTES)
    hgpipe.write_bytes(PNG_BYTES)
    docs_copy.write_bytes(PNG_BYTES)
    return hgtxr


class Req9DeitImageReferenceAuditTests(unittest.TestCase):
    def test_passes_when_requested_hgpipe_and_docs_images_match(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            hgtxr = make_tree(tmp)

            audit = audit_tool.build_audit(hgtxr)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["fail_count"], 0)
            self.assertEqual(audit["check_count"], 11)
            self.assertTrue(audit["normalized_requested_path"].endswith("PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png"))

    def test_missing_requested_image_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            hgtxr = make_tree(tmp, include_requested=False)

            audit = audit_tool.build_audit(hgtxr)

            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertEqual(audit["status"], "fail")
            self.assertIn("requested_image_exists", failed)
            self.assertIn("hardware_docs_copy_matches_requested_sha256", failed)

    def test_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            hgtxr = make_tree(tmp)
            json_out = Path(tmp) / "req9.json"
            md_out = Path(tmp) / "req9.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = audit_tool.main(
                    [
                        "--root",
                        str(hgtxr),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(md_out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertIn("Req9 DeiT Image Reference Audit", md_out.read_text())
            self.assertIn("requested_image", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
