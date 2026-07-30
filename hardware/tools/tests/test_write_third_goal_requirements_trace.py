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

import write_third_goal_requirements_trace as trace_tool  # noqa: E402


def sample_completion() -> dict[str, object]:
    items = []
    for idx in range(12):
        status = "pass"
        gaps: list[str] = []
        if idx == 2:
            status = "partial"
            gaps = ["spec-kit missing"]
        if idx == 11:
            status = "blocked"
            gaps = ["XR-VITs missing"]
        items.append(
            {
                "id": str(idx),
                "title": f"Requirement {idx}",
                "status": status,
                "evidence": [f"evidence {idx}"],
                "gaps": gaps,
            }
        )
    return {"status": "blocked", "pass_count": 10, "partial_count": 1, "blocked_count": 1, "item_count": 12, "items": items}


def sample_matrix() -> dict[str, object]:
    return {
        "summary": {"recommended_board_smoke_variant": "C3b"},
        "rows": [
            {
                "id": "C3b",
                "hls": {
                    "latency_cycles": 37508072,
                    "resources": {"dsp": 604, "lut": 126506, "ff": 59505, "bram_18k": 332, "uram": 64},
                },
                "vivado": {"timing": {"wns_ns": 4.415}, "power": {"total_on_chip_w": 3.476}},
            }
        ],
    }


def sample_command_card() -> dict[str, object]:
    return {
        "xr_vits_policy_integrity": {
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
        },
        "sections": [
            {
                "id": "U1",
                "title": "Run board smoke",
                "status": "pending-board-run",
                "reason": "needs board",
                "commands": ["run remote"],
            },
            {
                "id": "U2",
                "title": "Resolve XR-VITs",
                "status": "pending-user-choice",
                "reason": "needs choice",
                "options": [
                    {"id": "U2a", "commands": ["test -d XR-VITs"]},
                    {"id": "U2b", "commands": ["approve replacement"]},
                ],
            },
            {
                "id": "U4",
                "title": "Combined one-shot unblock",
                "status": "pending-user-choice",
                "reason": "combined",
                "options": [
                    {"id": "U4a", "commands": ["combined exact"]},
                    {"id": "U4b", "commands": ["combined replacement"]},
                ],
            },
        ]
    }


def sample_candidate_audit() -> dict[str, object]:
    return {
        "status": "blocked",
        "would_clear_all": False,
        "remaining_blockers": ["C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"],
        "c3b_smoke": {"status": "missing", "would_clear": False},
        "xr_vits": {"status": "fail", "mode": "exact", "would_clear": False},
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }


def sample_xr_vits_resolution() -> dict[str, object]:
    return {
        "status": "candidate-ready-needs-approval",
        "resolution_ready": False,
        "approval_required": True,
        "requested_path": "/tmp/XR-VITs",
        "exact": {"status": "missing"},
        "active_policy": {"status": "missing"},
        "candidate": {
            "status": "pass",
            "replacement_path": "/tmp/XR-VIT/XR_Accel",
            "score": 99,
        },
        "commands": {
            "replacement_dry_run": (
                "python3 tools/run_third_goal_final_signoff.py --allow-blocked "
                "--approve-xr-vits-replacement --dry-run-xr-vits-replacement "
                "--xr-vits-approved-by <approved-by>"
            ),
            "replacement_approve": (
                "python3 tools/run_third_goal_final_signoff.py --allow-blocked "
                "--approve-xr-vits-replacement --xr-vits-approved-by <approved-by>"
            ),
        },
        "safety": {
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
            "executes_commands": False,
        },
    }


def sample_evidence_manifest() -> dict[str, object]:
    return {
        "status": "pass",
        "required_count": 36,
        "present_required_count": 36,
        "failed_consistency_checks": [],
        "consistency_checks": [
            {"name": "operator_handoff_validation_pass", "status": "pass", "detail": "pass"},
            {"name": "final_bundle_validation_pass", "status": "pass", "detail": "pass"},
            {"name": "xr_vits_resolution_status_known", "status": "pass", "detail": "candidate-ready-needs-approval"},
        ],
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


class WriteThirdGoalRequirementsTraceTests(unittest.TestCase):
    def test_build_trace_links_requirements_resources_and_actions(self) -> None:
        trace = trace_tool.build_trace(
            root=Path("/tmp/HGTXR"),
            completion=sample_completion(),
            matrix=sample_matrix(),
            command_card=sample_command_card(),
            signoff_run={
                "status": "blocked",
                "final_preflight_summary": {"fail": 2},
                "remaining_blockers": ["requested XR-VITs sibling", "C3b AXIS/DMA physical smoke result"],
            },
            candidate_audit=sample_candidate_audit(),
            xr_vits_resolution=sample_xr_vits_resolution(),
            evidence_manifest=sample_evidence_manifest(),
        )

        self.assertEqual(trace["status"], "blocked")
        self.assertIn("11", trace["blocked_requirement_ids"])
        self.assertIn("2", trace["partial_requirement_ids"])
        self.assertEqual(trace["selected_resource_snapshot"]["dsp"], 604)
        self.assertEqual([action["id"] for action in trace["external_actions"]], ["U1", "U2", "U4"])
        self.assertEqual([condition["blocker"] for condition in trace["blocker_resolution_conditions"]], ["requested XR-VITs sibling", "C3b AXIS/DMA physical smoke result"])
        xr_condition = trace["blocker_resolution_conditions"][0]
        self.assertIn("11", xr_condition["clears_requirement_ids"])
        self.assertIn("approve replacement", xr_condition["command_options"]["approve_replacement"])
        self.assertIn("replacement policy validates against current candidate audit; legacy or drifted policy does not clear final signoff", xr_condition["acceptance"])
        self.assertFalse(xr_condition["policy_integrity"]["legacy_policy_clears_final_signoff"])
        self.assertTrue(xr_condition["policy_integrity"]["required_policy_fields_complete"])
        self.assertIn("candidate_audit_fingerprint", xr_condition["policy_integrity"]["required_policy_fields"])
        c3b_condition = trace["blocker_resolution_conditions"][1]
        self.assertIn("3", c3b_condition["clears_requirement_ids"])
        self.assertIn("run remote", c3b_condition["command_options"]["remote_execute"])
        self.assertEqual(trace["candidate_unblock_audit"]["status"], "blocked")
        self.assertEqual(trace["candidate_unblock_audit"]["xr_vits_mode"], "exact")
        self.assertEqual(trace["xr_vits_reference_resolution"]["status"], "candidate-ready-needs-approval")
        self.assertEqual(trace["xr_vits_reference_resolution"]["candidate_score"], 99)
        self.assertFalse(trace["xr_vits_reference_resolution"]["resolution_ready"])
        self.assertFalse(trace["xr_vits_policy_integrity"]["legacy_policy_clears_final_signoff"])
        self.assertTrue(trace["xr_vits_policy_integrity"]["required_policy_fields_complete"])
        self.assertEqual(trace["xr_vits_policy_integrity"]["generator"], "tools/create_xr_vits_replacement_policy.py")
        self.assertEqual(trace["final_evidence_manifest_contract"]["status"], "pass")
        self.assertEqual(trace["final_evidence_manifest_contract"]["present_required_count"], 36)
        self.assertEqual(trace["final_evidence_manifest_contract"]["consistency_count"], 3)
        self.assertFalse(trace["candidate_unblock_audit"]["safety"]["writes_canonical_inputs"])
        self.assertFalse(trace["safety"]["creates_board_result"])

    def test_build_trace_blocks_on_failed_evidence_manifest_contract(self) -> None:
        manifest = sample_evidence_manifest()
        manifest["status"] = "fail"
        manifest["failed_consistency_checks"] = ["final_bundle_validation_pass"]

        trace = trace_tool.build_trace(
            root=Path("/tmp/HGTXR"),
            completion={**sample_completion(), "status": "pass"},
            matrix=sample_matrix(),
            command_card=sample_command_card(),
            signoff_run={"status": "pass", "final_preflight_summary": {"fail": 0}, "remaining_blockers": []},
            candidate_audit=sample_candidate_audit(),
            xr_vits_resolution=sample_xr_vits_resolution(),
            evidence_manifest=manifest,
        )

        self.assertEqual(trace["status"], "blocked")
        self.assertEqual(
            trace["final_evidence_manifest_contract"]["failed_consistency_checks"],
            ["final_bundle_validation_pass"],
        )

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            resources = root / "docs" / "resources"
            resources.mkdir(parents=True)
            (resources / "third_goal_completion_audit_2026_06_10.json").write_text(json.dumps(sample_completion()) + "\n")
            (resources / "e2e_resource_matrix_2026_06_10.json").write_text(json.dumps(sample_matrix()) + "\n")
            (resources / "final_unblock_commands_2026_06_10.json").write_text(json.dumps(sample_command_card()) + "\n")
            (resources / "final_unblock_candidate_audit_2026_06_10.json").write_text(json.dumps(sample_candidate_audit()) + "\n")
            (resources / "xr_vits_reference_resolution_2026_06_10.json").write_text(json.dumps(sample_xr_vits_resolution()) + "\n")
            (resources / "final_evidence_manifest_2026_06_10.json").write_text(json.dumps(sample_evidence_manifest()) + "\n")
            (resources / "third_goal_final_signoff_run_2026_06_10.json").write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "final_preflight_summary": {"fail": 2},
                        "remaining_blockers": ["requested XR-VITs sibling", "C3b AXIS/DMA physical smoke result"],
                    }
                )
                + "\n"
            )
            json_out = root / "generated" / "trace.json"
            markdown_out = root / "generated" / "trace.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = trace_tool.main(
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
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["selected_resource_snapshot"]["selected_variant"], "C3b")
            text = markdown_out.read_text()
            self.assertIn("HGTXR Third Goal Requirements Trace", text)
            self.assertIn("Blocker Resolution Conditions", text)
            self.assertIn("Candidate Unblock Audit", text)
            self.assertIn("XR-VITs Reference Resolution", text)
            self.assertIn("XR-VITs Policy Integrity", text)
            self.assertIn("candidate_audit_fingerprint", text)
            self.assertIn("Final Evidence Manifest Contract", text)
            self.assertIn("C3b AXIS/DMA physical smoke result", text)


if __name__ == "__main__":
    unittest.main()
