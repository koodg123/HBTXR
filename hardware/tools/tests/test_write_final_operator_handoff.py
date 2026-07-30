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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import write_final_operator_handoff as handoff_tool  # noqa: E402
from test_write_third_goal_requirements_trace import sample_command_card, sample_candidate_audit, sample_xr_vits_resolution  # noqa: E402


def sample_readiness() -> dict[str, object]:
    return {
        "status": "ready-for-board",
        "ready": True,
        "variant": "c3b-mem16",
        "preset": "axis-c3b-mem16",
        "expected_runtime_state": 2,
        "expected_out_raw": [32, -13, 26, -6, 14, -11],
        "physical_result": {"path": "/tmp/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"},
        "tar": {"path": "/tmp/bundle.tar.gz", "sha256": "abc123"},
    }


def sample_trace() -> dict[str, object]:
    return {
        "status": "blocked",
        "final_signoff": {
            "status": "blocked",
            "summary": {"fail": 2},
            "remaining_blockers": ["requested XR-VITs sibling", "C3b AXIS/DMA physical smoke result"],
        },
        "final_evidence_manifest_contract": {
            "status": "pass",
            "source": "manifest-json",
            "path": "/tmp/HGTXR/docs/resources/final_evidence_manifest_2026_06_10.json",
            "required_count": 68,
            "present_required_count": 68,
            "consistency_count": 83,
            "failed_consistency_checks": [],
            "safety": {
                "executes_commands": False,
                "creates_board_result": False,
                "creates_xr_vits_policy": False,
                "writes_canonical_inputs": False,
            },
        },
    }


class WriteFinalOperatorHandoffTests(unittest.TestCase):
    def test_build_handoff_summarizes_operator_actions_without_side_effects(self) -> None:
        handoff = handoff_tool.build_handoff(
            root=Path("/tmp/HGTXR"),
            command_card=sample_command_card(),
            requirements_trace=sample_trace(),
            readiness=sample_readiness(),
            candidate_audit=sample_candidate_audit(),
            xr_vits_resolution=sample_xr_vits_resolution(),
        )

        self.assertEqual(handoff["status"], "pending-operator-actions")
        self.assertEqual(handoff["board_smoke"]["status"], "ready-for-board")
        self.assertEqual(handoff["board_smoke"]["bundle_sha256"], "abc123")
        self.assertIn("run remote", handoff["board_smoke"]["commands"])
        self.assertIn("test -d XR-VITs", handoff["xr_vits"]["exact_restore_commands"])
        self.assertIn("approve replacement", handoff["xr_vits"]["replacement_approval_commands"])
        self.assertEqual(handoff["xr_vits"]["reference_resolution"]["status"], "candidate-ready-needs-approval")
        self.assertEqual(handoff["xr_vits"]["reference_resolution"]["candidate_score"], 99)
        self.assertFalse(handoff["xr_vits"]["reference_resolution"]["writes_canonical_inputs"])
        integrity = handoff["xr_vits"]["reference_resolution"]["policy_integrity"]
        self.assertTrue(integrity["required"])
        self.assertFalse(integrity["policy_exists"])
        self.assertEqual(integrity["validation"]["status"], "pending-policy-creation")
        self.assertIn("policy_fingerprint", integrity["required_policy_fields"])
        self.assertEqual(handoff["final_evidence_manifest_contract"]["status"], "pass")
        self.assertEqual(handoff["final_evidence_manifest_contract"]["present_required_count"], 68)
        self.assertEqual(handoff["final_evidence_manifest_contract"]["consistency_count"], 83)
        self.assertFalse(handoff["safety"]["creates_board_result"])
        self.assertFalse(handoff["safety"]["creates_xr_vits_policy"])
        self.assertFalse(handoff["safety"]["executes_network"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            resources = root / "docs" / "resources"
            resources.mkdir(parents=True)
            (resources / "final_unblock_commands_2026_06_10.json").write_text(json.dumps(sample_command_card()) + "\n")
            (resources / "third_goal_requirements_trace_2026_06_10.json").write_text(json.dumps(sample_trace()) + "\n")
            (resources / "c3b_board_smoke_readiness_2026_06_10.json").write_text(json.dumps(sample_readiness()) + "\n")
            (resources / "final_unblock_candidate_audit_2026_06_10.json").write_text(json.dumps(sample_candidate_audit()) + "\n")
            (resources / "xr_vits_reference_resolution_2026_06_10.json").write_text(json.dumps(sample_xr_vits_resolution()) + "\n")
            json_out = root / "generated" / "handoff.json"
            markdown_out = root / "generated" / "handoff.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = handoff_tool.main(
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
            self.assertEqual(json.loads(json_out.read_text())["status"], "pending-operator-actions")
            text = markdown_out.read_text()
            self.assertIn("HGTXR Final Operator Handoff", text)
            self.assertIn("Board Smoke", text)
            self.assertIn("XR-VITs Gate", text)
            self.assertIn("Reference Resolution Commands", text)
            self.assertIn("Policy Integrity", text)
            self.assertIn("Final Evidence Manifest Contract", text)


if __name__ == "__main__":
    unittest.main()
