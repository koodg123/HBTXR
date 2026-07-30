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

import write_third_goal_unblock_checklist as checklist_tool  # noqa: E402


class WriteThirdGoalUnblockChecklistTests(unittest.TestCase):
    def final_audit(self, *, blocked: bool) -> dict[str, object]:
        blockers = []
        if blocked:
            blockers = [
                {"name": "requested XR-VITs sibling", "kind": "reference-input"},
                {"name": "C3b AXIS/DMA physical smoke result", "kind": "physical-board-smoke"},
            ]
        return {
            "status": "blocked" if blocked else "pass",
            "blockers": blockers,
        }

    def completion_audit(self, *, blocked: bool) -> dict[str, object]:
        items = []
        if blocked:
            items = [
                {"id": "11", "title": "XR-VITs HLS source requirement", "status": "blocked"},
                {"id": "final", "title": "Final signoff gate", "status": "blocked"},
            ]
        return {
            "status": "blocked" if blocked else "pass",
            "items": items,
        }

    def test_build_checklist_covers_two_known_blockers(self) -> None:
        root = Path("/tmp/XR-VIT/HGTXR")

        checklist = checklist_tool.build_checklist(root, self.final_audit(blocked=True), self.completion_audit(blocked=True))

        self.assertEqual(checklist["status"], "pending-unblock")
        self.assertEqual(checklist["final_blockers"], ["C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"])
        self.assertEqual([step["id"] for step in checklist["steps"]], ["B1", "B2", "B3"])
        self.assertIn("create_xr_vits_replacement_policy.py", checklist["steps"][0]["options"][1]["commands"][0])
        self.assertIn("./run_e2e_axis_dma_c3b_mem16_file_smoke.sh", checklist["steps"][1]["board_commands"])
        self.assertIn("run_third_goal_final_signoff.py", checklist["steps"][1]["host_commands"][0])
        self.assertIn("--dry-run-import-c3b-smoke", checklist["steps"][1]["host_commands"][0])
        self.assertIn("run_third_goal_final_signoff.py", checklist["steps"][1]["host_commands"][1])
        self.assertNotIn("--dry-run-import-c3b-smoke", checklist["steps"][1]["host_commands"][1])
        self.assertIn("generated/signoff/c3b_smoke_import_2026_06_10.json", checklist["steps"][1]["evidence_after"])

    def test_build_checklist_ready_when_no_blockers(self) -> None:
        root = Path("/tmp/XR-VIT/HGTXR")

        checklist = checklist_tool.build_checklist(root, self.final_audit(blocked=False), self.completion_audit(blocked=False))

        self.assertEqual(checklist["status"], "ready-for-final-signoff")
        self.assertEqual(checklist["final_blockers"], [])
        self.assertEqual([step["id"] for step in checklist["steps"]], ["B3"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            docs = root / "docs" / "resources"
            docs.mkdir(parents=True)
            final_audit = docs / "final_signoff_audit_2026_06_10.json"
            completion_audit = docs / "third_goal_completion_audit_2026_06_10.json"
            json_out = root / "generated" / "unblock.json"
            markdown_out = root / "generated" / "unblock.md"
            final_audit.write_text(json.dumps(self.final_audit(blocked=True)))
            completion_audit.write_text(json.dumps(self.completion_audit(blocked=True)))
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = checklist_tool.main(
                    [
                        "--root",
                        str(root),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(json.loads(json_out.read_text())["status"], "pending-unblock")
            text = markdown_out.read_text()
            self.assertIn("HGTXR Third Goal Unblock Checklist", text)
            self.assertIn("Approve XR_Accel as replacement", text)
            self.assertIn("Run and import C3b ZCU104 physical smoke", text)
            self.assertIn("--dry-run-import-c3b-smoke", text)


if __name__ == "__main__":
    unittest.main()
