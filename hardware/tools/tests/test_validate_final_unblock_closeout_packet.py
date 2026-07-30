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

import validate_final_unblock_closeout_packet as validator  # noqa: E402
import create_xr_vits_replacement_policy as policy_tool  # noqa: E402


def sample_packet() -> dict[str, object]:
    return {
        "status": "ready-for-operator-unblock",
        "root": "/tmp/HGTXR",
        "remaining_blockers": ["requested XR-VITs sibling", "C3b AXIS/DMA physical smoke result"],
        "missing_required_artifacts": [],
        "required_artifacts": [
            {"relative_path": f"artifact_{idx}.json", "status": "pass", "sha256": f"{idx:x}".zfill(64)}
            for idx in range(12)
        ],
        "board_package": {
            "status": "ready-for-board",
            "preset": "axis-c3b-mem16",
            "tar_sha256": validator.EXPECTED_C3B_SHA256,
            "expected_runtime_state": 2,
            "expected_out_raw": [32, -13, 26, -6, 14, -11],
        },
        "c3b_smoke_contract": {
            "status": "pass",
            "preset": "axis-c3b-mem16",
            "variant": "c3b-mem16",
            "canonical_result_path": "/tmp/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json",
            "expected_runtime_state": 2,
            "expected_out_raw": [32, -13, 26, -6, 14, -11],
            "validate_command": "python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
            "dry_run_import_command": "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --dry-run",
            "active_import_command": "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
        },
        "xr_vits_resolution": {
            "status": "candidate-ready-needs-approval",
            "candidate_score": 99,
            "candidate_replacement_path": "/tmp/XR-VIT/XR_Accel",
            "policy_integrity": {
                "required": True,
                "policy_path": "/tmp/HGTXR/docs/resources/xr_vits_replacement_policy.json",
                "candidate_audit": "/tmp/HGTXR/docs/resources/xr_vits_candidate_audit_2026_06_10.json",
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
                "policy_exists": False,
                "validation": {
                    "status": "pending-policy-creation",
                    "policy_exists": False,
                    "candidate_audit_path": "/tmp/HGTXR/docs/resources/xr_vits_candidate_audit_2026_06_10.json",
                    "errors": [],
                },
            },
        },
        "closure_readiness": {"status": "blocked", "current_ready": False},
        "vref_successor_gate": {
            "status": "ready-for-physical-smoke",
            "required_for_final_signoff": False,
            "recommended_resource_variant": "dsp_mixed_stream",
            "projection_status": "not_promotable",
            "projection_pending": ["physical_smoke_available"],
            "physical_smoke_status": "not_captured",
            "physical_smoke_preset": "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
            "expected_result_json": "/tmp/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json",
        },
        "qkv_uram_successor_gate": {
            "status": "ready-for-physical-smoke",
            "required_for_final_signoff": False,
            "csynth_status": "pass",
            "latency_cycles": 498485,
            "resources": {"bram_18k": 114, "dsp": 128, "ff": 19664, "lut": 43236, "uram": 40},
            "routed_wns_ns": 4.517,
            "route_errors": 0,
            "physical_smoke_status": "not_captured",
            "physical_smoke_preset": "axis-vref-p0-softmax-input-x2-qkv-uram",
            "expected_result_json": "/tmp/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json",
        },
        "operator_commands": {
            "dry_run_readiness": [
                "python3 tools/check_final_blocker_closure_readiness.py --root /tmp/HGTXR",
                "python3 tools/check_final_blocker_closure_readiness.py --root /tmp/HGTXR --xr-vits-mode exact",
                "python3 tools/check_final_blocker_closure_readiness.py --root /tmp/HGTXR --xr-vits-mode replacement",
            ],
            "combined_unblock": {
                "U4a": ["combined exact"],
                "U4b": [
                    "python3 tools/run_third_goal_final_signoff.py --dry-run-import-c3b-smoke --dry-run-xr-vits-replacement",
                    "python3 tools/create_xr_vits_replacement_policy.py --dry-run",
                    "python3 tools/create_xr_vits_replacement_policy.py --approve",
                    "python3 tools/run_third_goal_final_signoff.py --approve-xr-vits-replacement",
                ],
            },
            "qkv_uram_successor": [
                "python3 tools/run_third_goal_final_signoff.py --execute-qkv-uram-smoke --allow-blocked",
                "python3 tools/run_third_goal_final_signoff.py --import-qkv-uram-smoke-json /tmp/qkv.json --dry-run-import-qkv-uram-smoke --allow-blocked",
            ],
        },
        "safety": {
            "executes_commands": False,
            "executes_network": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def install_valid_policy(packet: dict[str, object], root: Path) -> Path:
    resources = root / "docs" / "resources"
    resources.mkdir(parents=True)
    audit_path = resources / "xr_vits_candidate_audit_2026_06_10.json"
    policy_path = resources / "xr_vits_replacement_policy.json"
    requested_path = (root.parent.parent / "XR-VITs").resolve()
    replacement_path = (root.parent / "XR_Accel").resolve()
    audit = {
        "requested_path": str(requested_path),
        "recommendation": {
            "path": str(replacement_path),
            "role": "candidate-xr-accel",
            "score": 99,
            "reason": "test candidate",
        },
    }
    audit_path.write_text(json.dumps(audit) + "\n")
    policy = policy_tool.build_policy(
        requested_path=requested_path,
        replacement_path=replacement_path,
        approved_by="tester",
        approved_at="2026-06-16T00:00:00Z",
        reason="test approval",
        reason_code=policy_tool.DEFAULT_REASON_CODE,
        candidate_audit_rel=policy_tool.DEFAULT_AUDIT_REL,
        candidate_audit_path=audit_path,
        audit=audit,
    )
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    packet["root"] = str(root)
    integrity = packet["xr_vits_resolution"]["policy_integrity"]  # type: ignore[index]
    integrity.update({field: policy[field] for field in policy_tool.INTEGRITY_FIELD_NAMES})  # type: ignore[union-attr]
    integrity.update(  # type: ignore[union-attr]
        {
            "policy_path": str(policy_path),
            "candidate_audit": str(audit_path),
            "policy_exists": True,
            "validation": {**policy_tool.validate_policy_integrity(root, policy), "policy_exists": True},
        }
    )
    return policy_path


class ValidateFinalUnblockCloseoutPacketTests(unittest.TestCase):
    def test_valid_packet_passes(self) -> None:
        result = validator.validate_packet(sample_packet())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["fail_count"], 0)
        self.assertGreaterEqual(result["check_count"], 46)
        self.assertFalse(result["safety"]["executes_commands"])

    def test_missing_xr_policy_integrity_fails(self) -> None:
        packet = sample_packet()
        packet["xr_vits_resolution"].pop("policy_integrity")  # type: ignore[index, union-attr]

        result = validator.validate_packet(packet)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("xr_policy_integrity_required", failed)
        self.assertIn("xr_policy_validation_embedded_matches_live", failed)

    def test_legacy_xr_policy_integrity_fails(self) -> None:
        packet = sample_packet()
        integrity = packet["xr_vits_resolution"]["policy_integrity"]  # type: ignore[index]
        integrity["legacy_policy_clears_final_signoff"] = True  # type: ignore[index]
        integrity["validation"]["status"] = "legacy-warning"  # type: ignore[index]

        result = validator.validate_packet(packet)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("xr_legacy_policy_not_accepted", failed)
        self.assertIn("xr_policy_validation_embedded_matches_live", failed)

    def test_existing_fingerprint_bound_policy_passes_and_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            packet = sample_packet()
            policy_path = install_valid_policy(packet, root)

            result = validator.validate_packet(packet)

            self.assertEqual(result["status"], "pass")
            policy = json.loads(policy_path.read_text())
            policy["candidate_audit_fingerprint"] = "0" * 64
            policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")

            drifted = validator.validate_packet(packet)

            self.assertEqual(drifted["status"], "fail")
            failed = [check["name"] for check in drifted["checks"] if check["status"] == "fail"]
            self.assertIn("xr_policy_validation_status", failed)
            self.assertIn("xr_policy_validation_embedded_matches_live", failed)

    def test_qkv_uram_successor_command_group_missing_fails(self) -> None:
        packet = sample_packet()
        packet["operator_commands"]["qkv_uram_successor"] = ["echo nope"]  # type: ignore[index]

        result = validator.validate_packet(packet)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("qkv_uram_successor_command_group_present", failed)

    def test_missing_dry_run_command_fails(self) -> None:
        packet = sample_packet()
        packet["operator_commands"]["dry_run_readiness"] = ["echo nope"]  # type: ignore[index]

        result = validator.validate_packet(packet)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("dry_run_commands_present", failed)
        self.assertIn("dry_run_uses_closure_checker", failed)

    def test_required_vref_successor_blocker_passes_when_listed(self) -> None:
        packet = sample_packet()
        packet["remaining_blockers"] = [  # type: ignore[index]
            "requested XR-VITs sibling",
            "C3b AXIS/DMA physical smoke result",
            "VREF-P0 successor physical smoke result",
        ]
        packet["vref_successor_policy"] = {"require_physical_smoke_for_final_signoff": True}  # type: ignore[index]
        packet["vref_successor_gate"]["required_for_final_signoff"] = True  # type: ignore[index]
        packet["vref_successor_gate"]["status"] = "blocked"  # type: ignore[index]

        result = validator.validate_packet(packet)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["fail_count"], 0)

    def test_required_vref_successor_fails_when_blocker_missing(self) -> None:
        packet = sample_packet()
        packet["vref_successor_policy"] = {"require_physical_smoke_for_final_signoff": True}  # type: ignore[index]
        packet["vref_successor_gate"]["required_for_final_signoff"] = True  # type: ignore[index]
        packet["vref_successor_gate"]["status"] = "blocked"  # type: ignore[index]

        result = validator.validate_packet(packet)

        self.assertEqual(result["status"], "fail")
        failed = [check["name"] for check in result["checks"] if check["status"] == "fail"]
        self.assertIn("remaining_blockers_expected", failed)
        self.assertIn("vref_successor_required_blocker_consistent", failed)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            packet_path = root / "docs" / "resources" / "final_unblock_closeout_packet_2026_06_10.json"
            packet_path.parent.mkdir(parents=True)
            packet_path.write_text(json.dumps(sample_packet()) + "\n")
            json_out = root / "generated" / "validation.json"
            markdown_out = root / "generated" / "validation.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = validator.main(
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
            self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
            self.assertIn("HGTXR Final Unblock Closeout Packet Validation", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
