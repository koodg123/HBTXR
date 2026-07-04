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

import write_vref_p0_pot_scale_audit as audit_tool  # noqa: E402


class WriteVrefP0PotScaleAuditTests(unittest.TestCase):
    def test_build_audit_passes_on_current_workspace(self) -> None:
        audit = audit_tool.build_audit(ROOT.parent)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["fail_count"], 0)
        macros = audit["observed"]["macros"]
        self.assertEqual(macros["HGTXR_WEIGHT_BIT_WIDTH"], 4)
        self.assertEqual(macros["HGTXR_BIT_WIDTH"], 8)
        self.assertEqual(macros["HGTXR_E2E_ACC_SCALE"], 16)
        self.assertGreaterEqual(len(audit["observed"]["dsp_bind_refs"]), 2)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "pot_scale.json"
            markdown_out = Path(tmp) / "pot_scale.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = audit_tool.main(
                    [
                        "--root",
                        str(ROOT.parent),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertIn("VREF-P0-01 PoT Scale Readiness Audit", markdown_out.read_text())
            self.assertIn("pot_scale_HGTXR_E2E_ACC_SCALE", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
