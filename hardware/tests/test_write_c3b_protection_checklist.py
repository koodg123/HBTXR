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

import write_c3b_protection_checklist as checklist_tool  # noqa: E402


class WriteC3bProtectionChecklistTests(unittest.TestCase):
    def test_build_checklist_preserves_current_c3b_baseline(self) -> None:
        checklist = checklist_tool.build_checklist(ROOT.parent)

        self.assertEqual(checklist["status"], "ready-for-board-smoke")
        self.assertEqual(checklist["fail_count"], 0)
        self.assertGreaterEqual(checklist["pending_count"], 1)
        self.assertEqual(checklist["baseline"]["latency_cycles"], 37_508_072)
        self.assertEqual(checklist["baseline"]["wns_ns"], 4.415)
        self.assertEqual(checklist["baseline"]["resources"]["dsp"], 604)
        self.assertEqual(checklist["baseline"]["resources"]["lut"], 126_506)
        self.assertEqual(checklist["baseline"]["resources"]["uram"], 64)
        self.assertEqual(checklist["successor_gate"]["max_latency_cycles"], 37_508_072)
        self.assertEqual(checklist["successor_gate"]["min_wns_ns"], 4.415)
        self.assertEqual(checklist["successor_gate"]["max_dsp"], 604)
        self.assertEqual(checklist["successor_gate"]["max_lut"], 126_506)
        self.assertEqual(checklist["successor_gate"]["max_uram"], 64)

        statuses = {row["name"]: row["status"] for row in checklist["checks"]}
        self.assertEqual(statuses["matrix_policy_no_new_hls_or_vivado_run"], "pass")
        self.assertEqual(statuses["matrix_recommends_c3b"], "pass")
        self.assertEqual(statuses["c3b_hwh_axis_top_marker"], "pass")
        self.assertEqual(statuses["c3b_physical_smoke_json_validated"], "pending")

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "c3b_protection.json"
            markdown_out = Path(tmp) / "c3b_protection.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = checklist_tool.main(
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
            self.assertEqual(payload["status"], "ready-for-board-smoke")
            self.assertEqual(payload["fail_count"], 0)
            self.assertGreaterEqual(payload["pending_count"], 1)
            text = markdown_out.read_text()
            self.assertIn("C3b Baseline Protection Checklist", text)
            self.assertIn("successor_latency_ceiling_is_current_c3b", text)


if __name__ == "__main__":
    unittest.main()
