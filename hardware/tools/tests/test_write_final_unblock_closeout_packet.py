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

import write_final_unblock_closeout_packet as packet_tool  # noqa: E402


class WriteFinalUnblockCloseoutPacketTests(unittest.TestCase):
    def write_json(self, root: Path, rel: str, payload: dict[str, object]) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n")

    def populate(self, root: Path) -> None:
        resources = "docs/resources"
        self.write_json(
            root,
            f"{resources}/final_unblock_commands_2026_06_10.json",
            {
                "status": "pending-unblock",
                "xr_vits_policy_integrity": {
                    "required": True,
                    "policy_path": str(root / "docs/resources/xr_vits_replacement_policy.json"),
                    "candidate_audit": str(root / "docs/resources/xr_vits_candidate_audit_2026_06_10.json"),
                    "generator": "tools/create_xr_vits_replacement_policy.py",
                    "validator": "tools/check_final_blocker_closure_readiness.py",
                    "required_policy_fields": [
                        "candidate_audit_fingerprint",
                        "candidate_audit_recommendation_snapshot",
                        "candidate_audit_meta",
                        "approval_event",
                        "policy_fingerprint",
                    ],
                    "legacy_policy_clears_final_signoff": False,
                },
                "sections": [
                    {
                        "id": "U0",
                        "commands": [
                            "python3 tools/check_final_blocker_closure_readiness.py --root /tmp/HGTXR",
                            "python3 tools/check_final_blocker_closure_readiness.py --root /tmp/HGTXR --xr-vits-mode exact",
                        ],
                    },
                    {
                        "id": "U1",
                        "commands": ["python3 tools/run_zcu104_c3b_smoke_remote.py --execute"],
                    },
                    {
                        "id": "U2",
                        "options": [
                            {"id": "U2a", "commands": ["test -d /tmp/XR-VITs"]},
                            {
                                "id": "U2b",
                                "commands": [
                                    "python3 tools/run_third_goal_final_signoff.py --dry-run-xr-vits-replacement",
                                    "python3 tools/create_xr_vits_replacement_policy.py --dry-run",
                                    "python3 tools/create_xr_vits_replacement_policy.py --approve",
                                    "python3 tools/run_third_goal_final_signoff.py --approve-xr-vits-replacement",
                                ],
                            },
                        ],
                    },
                    {
                        "id": "U4",
                        "options": [
                            {"id": "U4a", "commands": ["combined exact"]},
                            {
                                "id": "U4b",
                                "commands": [
                                    "python3 tools/run_third_goal_final_signoff.py --dry-run-import-c3b-smoke --dry-run-xr-vits-replacement",
                                    "python3 tools/create_xr_vits_replacement_policy.py --dry-run",
                                    "python3 tools/create_xr_vits_replacement_policy.py --approve",
                                    "python3 tools/run_third_goal_final_signoff.py --approve-xr-vits-replacement",
                                ],
                            },
                        ],
                    },
                    {
                        "id": "U5",
                        "commands": [
                            "python3 tools/run_third_goal_final_signoff.py --require-qkv-uram-physical-smoke --allow-blocked",
                            "python3 tools/run_third_goal_final_signoff.py --execute-qkv-uram-smoke --allow-blocked",
                            "python3 tools/run_third_goal_final_signoff.py --import-qkv-uram-smoke-json /tmp/qkv.json --dry-run-import-qkv-uram-smoke --allow-blocked",
                        ],
                    },
                ],
            },
        )
        (root / f"{resources}/final_unblock_commands_2026_06_10.md").write_text("# commands\n")
        self.write_json(root, f"{resources}/c3b_board_smoke_readiness_2026_06_10.json", {"status": "ready-for-board"})
        self.write_json(
            root,
            f"{resources}/c3b_smoke_result_contract_2026_06_10.json",
            {
                "status": "pass",
                "preset": "axis-c3b-mem16",
                "variant": "c3b-mem16",
                "safety": {
                    "executes_commands": False,
                    "executes_network": False,
                    "creates_board_result": False,
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                },
            },
        )
        (root / f"{resources}/c3b_smoke_result_contract_2026_06_10.md").write_text("# contract\n")
        self.write_json(
            root,
            f"{resources}/c3b_smoke_transfer_manifest_2026_06_10.json",
            {
                "preset": "axis-c3b-mem16",
                "tar": {"path": "/tmp/bundle.tar.gz", "sha256": "a" * 64},
                "board_expected_outputs": {"expected_runtime_state": 2, "expected_out_raw": [32, -13, 26, -6, 14, -11]},
            },
        )
        (root / f"{resources}/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256").write_text(
            f"{'a' * 64}  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz\n"
        )
        self.write_json(
            root,
            f"{resources}/xr_vits_reference_resolution_2026_06_10.json",
            {
                "status": "candidate-ready-needs-approval",
                "requested_path": "/tmp/XR-VITs",
                "exact": {"status": "missing"},
                "candidate": {"status": "pass", "replacement_path": "/tmp/XR_Accel", "score": 99},
            },
        )
        self.write_json(
            root,
            f"{resources}/final_blocker_closure_readiness_2026_06_10.json",
            {"status": "blocked", "current_ready": False, "candidate_ready": False, "final_runner_command": ""},
        )
        self.write_json(
            root,
            f"{resources}/final_unblock_intake_2026_06_10.json",
            {
                "status": "blocked",
                "dry_run_final_runner_command": "",
                "active_final_runner_command": "",
                "safety": {
                    "executes_commands": False,
                    "executes_network": False,
                    "creates_board_result": False,
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                },
            },
        )
        (root / f"{resources}/final_unblock_intake_2026_06_10.md").write_text("# intake\n")
        self.write_json(
            root,
            f"{resources}/final_signoff_audit_2026_06_10.json",
            {
                "status": "blocked",
                "blockers": [
                    {"name": "C3b AXIS/DMA physical smoke result"},
                    {"name": "requested XR-VITs sibling"},
                ],
            },
        )
        self.write_json(
            root,
            "hardware/generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.json",
            {
                "candidate": "softmax_input_x2",
                "recommended_resource_variant": "dsp_mixed_stream",
                "recommended_c3b_projection": {
                    "status": "not_promotable",
                    "failures": [],
                    "pending": ["physical_smoke_available"],
                },
                "physical_smoke_result": {
                    "status": "not_captured",
                    "preset": "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                    "result_json": "",
                },
            },
        )
        self.write_json(
            root,
            "hardware/generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.json",
            {
                "status": "pass",
                "candidate": "VREF-P0-02-qkv-weight-cache-uram",
                "macro": "HGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1",
                "csynth": {
                    "latency_cycles": 498485,
                    "resources": {"bram_18k": 114, "dsp": 128, "ff": 19664, "lut": 43236, "uram": 40},
                },
                "overlay": {
                    "timing": {"wns_ns": 4.517},
                    "route_status": {"routing_error_nets": 0},
                },
                "physical_smoke_result": {
                    "status": "not_captured",
                    "preset": "axis-vref-p0-softmax-input-x2-qkv-uram",
                    "result_json": "",
                },
            },
        )

    def test_build_packet_collects_required_artifacts_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate(root)

            packet = packet_tool.build_packet(root)

            self.assertEqual(packet["status"], "ready-for-operator-unblock")
            self.assertEqual(packet["missing_required_artifacts"], [])
            self.assertEqual(packet["board_package"]["preset"], "axis-c3b-mem16")
            self.assertEqual(packet["xr_vits_resolution"]["candidate_score"], 99)
            self.assertTrue(packet["xr_vits_resolution"]["policy_integrity"]["required"])
            self.assertEqual(
                packet["xr_vits_resolution"]["policy_integrity"]["validation"]["status"],
                "pending-policy-creation",
            )
            self.assertIn(
                "policy_fingerprint",
                packet["xr_vits_resolution"]["policy_integrity"]["required_policy_fields"],
            )
            self.assertEqual(packet["unblock_intake"]["status"], "blocked")
            self.assertEqual(packet["vref_successor_gate"]["status"], "ready-for-physical-smoke")
            self.assertFalse(packet["vref_successor_gate"]["required_for_final_signoff"])
            self.assertEqual(packet["vref_successor_gate"]["recommended_resource_variant"], "dsp_mixed_stream")
            self.assertEqual(packet["qkv_uram_successor_gate"]["status"], "ready-for-physical-smoke")
            self.assertFalse(packet["qkv_uram_successor_gate"]["required_for_final_signoff"])
            self.assertEqual(packet["qkv_uram_successor_gate"]["resources"]["uram"], 40)
            self.assertIn("U4b", packet["operator_commands"]["combined_unblock"])
            self.assertIn("--execute-qkv-uram-smoke", " ".join(packet["operator_commands"]["qkv_uram_successor"]))
            self.assertIn("check_final_blocker_closure_readiness.py", packet["operator_commands"]["dry_run_readiness"][0])
            self.assertFalse(packet["safety"]["creates_board_result"])
            self.assertFalse(packet["safety"]["executes_commands"])
            self.assertFalse(packet["safety"]["executes_network"])
            self.assertTrue(all(len(entry["sha256"]) == 64 for entry in packet["required_artifacts"]))

    def test_build_packet_can_require_vref_successor_physical_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate(root)

            packet = packet_tool.build_packet(root, require_vref_successor_physical_smoke=True)

            self.assertTrue(packet["vref_successor_policy"]["require_physical_smoke_for_final_signoff"])
            self.assertTrue(packet["vref_successor_gate"]["required_for_final_signoff"])
            self.assertEqual(packet["vref_successor_gate"]["status"], "blocked")
            self.assertIn("VREF-P0 successor physical smoke result", packet["remaining_blockers"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate(root)
            json_out = root / "generated" / "packet.json"
            markdown_out = root / "generated" / "packet.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = packet_tool.main(
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
            self.assertEqual(json.loads(json_out.read_text())["status"], "ready-for-operator-unblock")
            text = markdown_out.read_text()
            self.assertIn("HGTXR Final Unblock Closeout Packet", text)
            self.assertIn("Combined Unblock", text)
            self.assertIn("XR-VITs Policy Integrity", text)
            self.assertIn("VREF Successor Gate", text)
            self.assertIn("QKV URAM Successor Gate", text)

    def test_cli_can_write_required_vref_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate(root)
            json_out = root / "generated" / "packet.json"
            markdown_out = root / "generated" / "packet.md"

            code = packet_tool.main(
                [
                    "--root",
                    str(root),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                    "--require-vref-successor-physical-smoke",
                ]
            )

            self.assertEqual(code, 0)
            payload = json.loads(json_out.read_text())
            self.assertIn("VREF-P0 successor physical smoke result", payload["remaining_blockers"])
            self.assertIn("require_physical_smoke_for_final_signoff: `True`", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
