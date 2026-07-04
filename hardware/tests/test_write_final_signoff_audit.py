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

import write_final_signoff_audit as audit_tool  # noqa: E402


class WriteFinalSignoffAuditTests(unittest.TestCase):
    def test_build_audit_classifies_known_final_signoff_blockers(self) -> None:
        preflight = {
            "mode": "final-signoff",
            "summary": {"ok": 61, "warn": 4, "fail": 3},
            "checks": [
                {
                    "status": "fail",
                    "name": "C3b AXIS/DMA physical smoke result",
                    "detail": "not captured yet",
                    "path": "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json",
                },
                {"status": "fail", "name": "requested PAPER_PRJXR DeiT image", "detail": "missing"},
                {"status": "fail", "name": "requested XR-VITs sibling", "detail": "missing"},
                {"status": "warn", "name": "tool spec-kit", "detail": "not found on PATH"},
            ],
        }

        audit = audit_tool.build_audit(preflight)

        self.assertEqual(audit["status"], "blocked")
        self.assertEqual(audit["blocker_count"], 3)
        self.assertEqual(audit["warning_count"], 1)
        self.assertEqual(audit["blockers"][0]["kind"], "physical-board-smoke")
        self.assertTrue(any("import_pynq_smoke_result.py" in command for command in audit["next_commands"]))

    def test_build_audit_passes_when_no_failures_exist(self) -> None:
        audit = audit_tool.build_audit(
            {
                "mode": "final-signoff",
                "summary": {"ok": 64, "warn": 0, "fail": 0},
                "checks": [{"status": "ok", "name": "C3b AXIS/DMA physical smoke result", "detail": "validated"}],
            }
        )

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["blocker_count"], 0)
        self.assertEqual(audit["next_commands"], [])

    def test_cli_writes_json_and_markdown_even_when_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preflight = Path(tmp) / "preflight.json"
            json_out = Path(tmp) / "audit.json"
            markdown_out = Path(tmp) / "audit.md"
            preflight.write_text(
                json.dumps(
                    {
                        "mode": "final-signoff",
                        "summary": {"ok": 61, "warn": 4, "fail": 1},
                        "checks": [
                            {
                                "status": "fail",
                                "name": "C3b AXIS/DMA physical smoke result",
                                "detail": "not captured yet",
                            }
                        ],
                    }
                )
            )
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = audit_tool.main(
                    [
                        "--preflight-json",
                        str(preflight),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(json.loads(json_out.read_text())["status"], "blocked")
            self.assertIn("HGTXR Final Signoff Audit", markdown_out.read_text())
            self.assertIn("run_e2e_axis_dma_c3b_mem16_file_smoke.sh", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
