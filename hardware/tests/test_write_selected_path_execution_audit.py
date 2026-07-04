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

import write_selected_path_execution_audit as audit_tool  # noqa: E402


class WriteSelectedPathExecutionAuditTests(unittest.TestCase):
    def test_build_audit_passes_on_current_workspace(self) -> None:
        audit = audit_tool.build_audit(ROOT.parent)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["fail_count"], 0)
        self.assertEqual(audit["selection"]["path_1"], "A2 then A1")
        self.assertEqual(audit["selection"]["path_2"], "C, with C3b as board-smoke candidate")
        self.assertEqual(audit["selection"]["e"], "pending")
        self.assertEqual(audit["observed"]["a2"]["parallelism"], 8)
        self.assertEqual(audit["observed"]["a1"]["parallelism"], 8)
        self.assertEqual(audit["observed"]["c3b"]["parallelism"], 16)
        self.assertEqual(audit["observed"]["c3b"]["memory_banks"], 16)
        self.assertGreater(audit["observed"]["c3b"]["dsp"], audit["observed"]["a1"]["dsp"])
        self.assertLess(audit["observed"]["c3b"]["latency_cycles"], audit["observed"]["a1"]["latency_cycles"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "selected_path.json"
            markdown_out = Path(tmp) / "selected_path.md"
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
            self.assertGreaterEqual(payload["check_count"], 22)
            self.assertIn("Selected Path Execution Audit", markdown_out.read_text())
            self.assertIn("path2_c3b_recommended", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
