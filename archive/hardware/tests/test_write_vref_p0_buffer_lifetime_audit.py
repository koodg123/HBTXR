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

import write_vref_p0_buffer_lifetime_audit as audit_tool  # noqa: E402


class WriteVrefP0BufferLifetimeAuditTests(unittest.TestCase):
    def test_build_audit_passes_on_current_workspace(self) -> None:
        audit = audit_tool.build_audit(ROOT.parent)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["fail_count"], 0)
        self.assertEqual(audit["observed"]["macros"]["HGTXR_E2E_FORCE_URAM_BUFFERS"], 1)
        self.assertEqual(audit["observed"]["macros"]["HGTXR_E2E_SMALL_MEM_LUTRAM"], 1)
        self.assertEqual(audit["observed"]["macros"]["HGTXR_E2E_URAM_QKV_WEIGHT_CACHE"], 0)
        self.assertEqual(audit["observed"]["c3b"]["parallelism"], 16)
        self.assertEqual(audit["observed"]["c3b"]["memory_banks"], 16)
        self.assertGreater(audit["observed"]["c3b"]["uram"], 0)
        self.assertTrue(audit["observed"]["rmu_smu_small_refs"]["score"])
        self.assertTrue(audit["observed"]["rmu_smu_small_refs"]["prob"])
        self.assertTrue(audit["observed"]["qkv_uram_refs"]["q_weight_cache"])
        self.assertEqual(audit["observed"]["qkv_uram_successor"]["status"], "pass")
        self.assertEqual(audit["observed"]["qkv_uram_successor"]["delta_vs_dsp_mixed_stream"]["uram"], 8)
        self.assertEqual(audit["check_count"], 51)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "buffer_lifetime.json"
            markdown_out = Path(tmp) / "buffer_lifetime.md"
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
            text = markdown_out.read_text()
            self.assertIn("VREF-P0-02 Buffer Lifetime", text)
            self.assertIn("axis_large_gb.hidden_uram", text)
            self.assertIn("rmu_smu_small_score_lutram", text)
            self.assertIn("qkv_successor_uram_increased_vs_dsp_mixed_stream", text)


if __name__ == "__main__":
    unittest.main()
