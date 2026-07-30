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

import write_final_unblock_commands as command_tool  # noqa: E402


class WriteFinalUnblockCommandsTests(unittest.TestCase):
    def final_audit(self, *, blocked: bool) -> dict[str, object]:
        blockers = []
        if blocked:
            blockers = [
                {"name": "requested XR-VITs sibling"},
                {"name": "C3b AXIS/DMA physical smoke result"},
            ]
        return {"status": "blocked" if blocked else "pass", "blockers": blockers}

    def test_build_commands_covers_known_blockers_without_side_effects(self) -> None:
        root = Path("/tmp/XR-VIT/HGTXR")

        card = command_tool.build_commands(
            root=root,
            final_audit=self.final_audit(blocked=True),
            readiness={"status": "ready-for-board"},
            xr_vits_packet={"status": "pending-user-choice"},
            c3b_contract={
                "status": "pass",
                "preset": "axis-c3b-mem16",
                "canonical_result_path": "/tmp/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json",
                "required_fields": {"out_raw": [32, -13, 26, -6, 14, -11]},
                "validation": {
                    "validate_command": "python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
                    "dry_run_import_command": "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --dry-run",
                    "active_import_command": "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
                },
            },
            host_placeholder="<board>",
            user_placeholder="xilinx",
            approver_placeholder="<name>",
        )

        self.assertEqual(card["status"], "pending-unblock")
        self.assertFalse(card["safety"]["creates_board_result"])
        self.assertFalse(card["safety"]["creates_xr_vits_policy"])
        self.assertFalse(card["safety"]["executes_commands"])
        self.assertFalse(card["safety"]["executes_network"])
        self.assertFalse(card["safety"]["writes_canonical_inputs"])
        self.assertEqual([section["id"] for section in card["sections"]], ["U0", "U1", "U2", "U4", "U5", "U3"])
        self.assertIn("check_final_blocker_closure_readiness.py", card["sections"][0]["commands"][0])
        self.assertIn("--json-out", card["sections"][0]["commands"][0])
        self.assertIn("--markdown-out", card["sections"][0]["commands"][0])
        self.assertIn("--c3b-no-require-paths", card["sections"][0]["commands"][0])
        self.assertIn("--xr-vits-mode exact", card["sections"][0]["commands"][1])
        self.assertIn("--json-out", card["sections"][0]["commands"][1])
        self.assertIn("--xr-vits-mode replacement", card["sections"][0]["commands"][2])
        self.assertIn("--markdown-out", card["sections"][0]["commands"][2])
        self.assertIn("--execute", card["sections"][1]["commands"][0])
        self.assertIn("run_third_goal_final_signoff.py", card["sections"][1]["commands"][0])
        self.assertIn("--execute-zcu104-smoke", card["sections"][1]["commands"][0])
        self.assertIn("run_zcu104_c3b_smoke_remote.py", card["sections"][1]["commands"][1])
        self.assertIn("--import-c3b-smoke-json", card["sections"][1]["commands"][2])
        self.assertIn("validate_pynq_smoke_result.py", card["sections"][1]["commands"][3])
        self.assertIn("import_pynq_smoke_result.py", card["sections"][1]["commands"][4])
        self.assertIn("--dry-run", card["sections"][1]["commands"][4])
        self.assertIn("import_pynq_smoke_result.py", card["sections"][1]["commands"][5])
        self.assertEqual(card["sections"][1]["contract"]["status"], "pass")
        self.assertEqual(card["c3b_smoke_contract"]["preset"], "axis-c3b-mem16")
        self.assertTrue(card["xr_vits_policy_integrity"]["required"])
        self.assertFalse(card["xr_vits_policy_integrity"]["legacy_policy_clears_final_signoff"])
        self.assertTrue(
            {
                "candidate_audit_fingerprint",
                "candidate_audit_recommendation_snapshot",
                "candidate_audit_meta",
                "approval_event",
                "policy_fingerprint",
            }.issubset(set(card["xr_vits_policy_integrity"]["required_policy_fields"]))
        )
        self.assertIn("run_third_goal_final_signoff.py", card["sections"][2]["options"][1]["commands"][0])
        self.assertIn("--dry-run-xr-vits-replacement", card["sections"][2]["options"][1]["commands"][0])
        self.assertIn("create_xr_vits_replacement_policy.py", card["sections"][2]["options"][1]["commands"][1])
        self.assertIn("run_third_goal_final_signoff.py", card["sections"][2]["options"][1]["commands"][3])
        self.assertIn("--approve-xr-vits-replacement", card["sections"][2]["options"][1]["commands"][3])
        self.assertTrue(card["sections"][2]["options"][1]["integrity_required"]["required"])
        self.assertEqual(card["sections"][3]["id"], "U4")
        self.assertIn("--import-c3b-smoke-json", card["sections"][3]["options"][0]["commands"][1])
        self.assertIn("test -d /home/kjm26/project/PRJXR/XR-VITs", card["sections"][3]["options"][0]["commands"][0])
        self.assertIn("--dry-run-xr-vits-replacement", card["sections"][3]["options"][1]["commands"][0])
        self.assertIn("--dry-run-import-c3b-smoke", card["sections"][3]["options"][1]["commands"][0])
        self.assertIn("create_xr_vits_replacement_policy.py", card["sections"][3]["options"][1]["commands"][1])
        self.assertIn("create_xr_vits_replacement_policy.py", card["sections"][3]["options"][1]["commands"][2])
        self.assertIn("--approve-xr-vits-replacement", card["sections"][3]["options"][1]["commands"][3])
        self.assertIn("--import-c3b-smoke-json", card["sections"][3]["options"][1]["commands"][3])
        self.assertTrue(card["sections"][3]["options"][1]["integrity_required"]["required"])
        self.assertEqual(card["sections"][4]["id"], "U5")
        self.assertIn("--require-qkv-uram-physical-smoke", card["sections"][4]["commands"][0])
        self.assertIn("--execute-qkv-uram-smoke", card["sections"][4]["commands"][1])
        self.assertIn("axis-vref-p0-softmax-input-x2-qkv-uram", card["sections"][4]["commands"][2])
        self.assertIn("--dry-run-import-qkv-uram-smoke", card["sections"][4]["commands"][3])
        self.assertIn("--import-qkv-uram-smoke-json", card["sections"][4]["commands"][4])

    def test_build_commands_ready_when_no_blockers(self) -> None:
        root = Path("/tmp/XR-VIT/HGTXR")

        card = command_tool.build_commands(
            root=root,
            final_audit=self.final_audit(blocked=False),
            readiness=None,
            xr_vits_packet=None,
            c3b_contract=None,
            host_placeholder="<board>",
            user_placeholder="xilinx",
            approver_placeholder="<name>",
        )

        self.assertEqual(card["status"], "ready-for-final-signoff")
        self.assertEqual(card["remaining_blockers"], [])
        self.assertEqual([section["id"] for section in card["sections"]], ["U3"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            resources = root / "docs" / "resources"
            resources.mkdir(parents=True)
            final_audit = resources / "final_signoff_audit_2026_06_10.json"
            readiness = resources / "c3b_board_smoke_readiness_2026_06_10.json"
            xr_vits = resources / "xr_vits_unblock_packet_2026_06_10.json"
            c3b_contract = resources / "c3b_smoke_result_contract_2026_06_10.json"
            json_out = root / "generated" / "commands.json"
            markdown_out = root / "generated" / "commands.md"
            final_audit.write_text(json.dumps(self.final_audit(blocked=True)) + "\n")
            readiness.write_text(json.dumps({"status": "ready-for-board"}) + "\n")
            xr_vits.write_text(json.dumps({"status": "pending-user-choice"}) + "\n")
            c3b_contract.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "preset": "axis-c3b-mem16",
                        "canonical_result_path": str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                        "required_fields": {"out_raw": [32, -13, 26, -6, 14, -11]},
                        "validation": {
                            "validate_command": "python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
                            "dry_run_import_command": "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --dry-run",
                            "active_import_command": "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
                        },
                    }
                )
                + "\n"
            )
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = command_tool.main(
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
            self.assertIn("HGTXR Final Unblock Commands", text)
            self.assertIn("run_third_goal_final_signoff.py", text)
            self.assertIn("--execute-zcu104-smoke", text)
            self.assertIn("run_zcu104_c3b_smoke_remote.py", text)
            self.assertIn("--import-c3b-smoke-json", text)
            self.assertIn("import_pynq_smoke_result.py", text)
            self.assertIn("c3b_smoke_contract_status", text)
            self.assertIn("expected_out_raw", text)
            self.assertIn("check_final_blocker_closure_readiness.py", text)
            self.assertIn("--approve-xr-vits-replacement", text)
            self.assertIn("--dry-run-xr-vits-replacement", text)
            self.assertIn("--dry-run-import-c3b-smoke", text)
            self.assertIn("Combined one-shot unblock", text)
            self.assertIn("Import C3b result and approve XR_Accel", text)
            self.assertIn("Approve XR_Accel replacement policy", text)
            self.assertIn("xr_vits_policy_integrity_required", text)
            self.assertIn("candidate audit fingerprint", text)
            self.assertIn("Optional QKV URAM successor smoke", text)
            self.assertIn("--execute-qkv-uram-smoke", text)
            self.assertIn("--dry-run-import-qkv-uram-smoke", text)


if __name__ == "__main__":
    unittest.main()
