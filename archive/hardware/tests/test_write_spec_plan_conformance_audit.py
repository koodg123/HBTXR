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

import write_spec_plan_conformance_audit as audit_tool  # noqa: E402


class WriteSpecPlanConformanceAuditTests(unittest.TestCase):
    def write(self, root: Path, rel: str, text: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_json(self, root: Path, rel: str, payload: dict[str, object]) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n")

    def populate_minimal_root(self, root: Path) -> None:
        live_doc = (
            "spec-kit manual ZCU104 Q4W/Q8A HGTXR_PARALLELISM_FACTOR "
            "HGTXR_BUS_WIDTH HGTXR_FIFO_DEPTH A2 A1 PAR=16 "
            "manifest `48/48` consistency checks `3` required `7` sources `9` "
            "reflected `10`, partial `1`, blocked `1` "
            "operator handoff validation `131/131` bundle validation `152/152`\n"
        )
        self.write(root, "docs/Master-Plan.md", "Path 1 = A2 then A1\nPath 2 = C\nE pending\n" + live_doc)
        self.write(root, "docs/Sub-Plan.md", "task_id: T-033\n" + live_doc)
        self.write(root, "docs/Spec.md", live_doc)
        self.write(root, "docs/Execution.md", "/tools/Xilinx\n")
        self.write(root, "docs/Validation.md", "Selected Path Execution Audit\n" + live_doc)
        self.write(root, "docs/track/PROGRESS.md", "Latest Continuation: Selected Path Execution Audit\n" + live_doc)
        self.write(
            root,
            "docs/track/HANDOVER.md",
            (
                "HGTXR Hardware Handover\n"
                + live_doc
                + "`generated/signoff/final_signoff_bundle_validation_2026_06_10.md`, "
                + "status `pass`, checks `152`, fail `0`.\n"
                + "`generated/signoff/final_operator_handoff_validation_2026_06_10.md`, "
                + "status `pass`, checks `131`, fail `0`.\n"
            ),
        )
        self.write(root, "docs/track/HANDOVER-2026-06-10-E2E.md", "E2E Q4W/Q8A ZCU104 active196_b6_ff768\n")
        self.write(root, "docs/CHOICE.md", "E: pending\nSelected Path Execution Audit\n" + live_doc)
        self.write(root, "docs/track/CHOICE.md", "stale fallback should be ignored when docs/CHOICE.md exists\n")
        self.write(root, "docs/track/log.md", "Selected path execution audit\n" + live_doc)
        requirements = [{"id": str(idx), "status": "pass"} for idx in range(12)]
        self.write_json(
            root,
            "docs/resources/third_goal_requirements_trace_2026_06_10.json",
            {
                "status": "blocked",
                "requirements": requirements,
                "final_evidence_manifest_contract": {"status": "pass"},
            },
        )
        self.write_json(
            root,
            "docs/resources/third_goal_completion_audit_2026_06_10.json",
            {"status": "blocked", "item_count": 14},
        )
        self.write_json(
            root,
            "docs/resources/final_evidence_manifest_2026_06_10.json",
            {
                "status": "pass",
                "present_required_count": 48,
                "required_count": 48,
                "consistency_checks": [{"name": f"check_{idx}", "status": "pass"} for idx in range(3)],
            },
        )
        self.write_json(
            root,
            "docs/resources/third_goal_source_audit_2026_06_16.json",
            {"status": "pass", "required_count": 7, "source_count": 9, "missing_required": []},
        )
        self.write_json(
            root,
            "docs/resources/third_goal_current_audit_2026_06_16.json",
            {"status": "blocked-external", "summary": {"reflected": 10, "partial": 1, "blocked": 1}},
        )
        self.write_json(
            root,
            "docs/resources/selected_path_execution_audit_2026_06_10.json",
            {"status": "pass"},
        )
        self.write_json(
            root,
            "docs/resources/e2e_resource_policy_audit_2026_06_10.json",
            {"status": "pass"},
        )
        self.write_json(
            root,
            "docs/resources/final_operator_handoff_validation_2026_06_10.json",
            {"status": "pass", "check_count": 131, "pass_count": 131, "fail_count": 0},
        )
        self.write_json(
            root,
            "docs/resources/final_signoff_bundle_validation_2026_06_10.json",
            {"status": "pass", "check_count": 152, "pass_count": 152, "fail_count": 0},
        )

    def test_build_audit_passes_on_current_workspace(self) -> None:
        audit = audit_tool.build_audit(ROOT.parent)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["fail_count"], 0)
        self.assertGreaterEqual(audit["check_count"], 32)
        self.assertEqual(audit["observed"]["requirements_count"], 12)
        self.assertEqual(audit["observed"]["selected_path_status"], "pass")
        self.assertEqual(audit["observed"]["evidence_consistency_count"], 347)
        self.assertEqual(audit["observed"]["source_count"], 224)
        self.assertEqual(audit["observed"]["operator_handoff_validation_check_count"], 131)
        self.assertEqual(audit["observed"]["final_signoff_bundle_validation_check_count"], 152)

    def test_minimal_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["fail_count"], 0)

    def test_validation_log_and_choice_are_current_doc_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            self.write(root, "docs/Validation.md", "Selected Path Execution Audit\n")
            self.write(root, "docs/CHOICE.md", "E: pending\nSelected Path Execution Audit\n")
            self.write(root, "docs/track/log.md", "Selected path execution audit\n")

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("validation_records_live_manifest_required", failed)
            self.assertIn("choice_records_live_manifest_required", failed)
            self.assertIn("log_records_live_manifest_required", failed)

    def test_validator_count_freshness_is_current_doc_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            self.write(
                root,
                "docs/track/HANDOVER.md",
                (
                    "HGTXR Hardware Handover\n"
                    "manifest `48/48` consistency checks `3` required `7` sources `9` "
                    "reflected `10`, partial `1`, blocked `1` "
                    "operator handoff validation `72/72` bundle validation `87/87`\n"
                ),
            )

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("current_handover_records_live_operator_handoff_validation", failed)
            self.assertIn("current_handover_records_live_bundle_validation", failed)
            self.assertIn("current_docs_have_no_stale_validator_counts", failed)

    def test_validator_count_freshness_uses_target_counts_during_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            self.write_json(
                root,
                "docs/resources/final_operator_handoff_validation_2026_06_10.json",
                {"status": "fail", "check_count": 131, "pass_count": 130, "fail_count": 1},
            )
            self.write_json(
                root,
                "docs/resources/final_signoff_bundle_validation_2026_06_10.json",
                {"status": "fail", "check_count": 152, "pass_count": 149, "fail_count": 3},
            )

            audit = audit_tool.build_audit(root)

            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertNotIn("master_plan_records_live_operator_handoff_validation", failed)
            self.assertNotIn("master_plan_records_live_bundle_validation", failed)
            self.assertNotIn("current_handover_records_live_operator_handoff_validation", failed)
            self.assertNotIn("current_handover_records_live_bundle_validation", failed)

    def test_handover_raw_validator_summary_counts_are_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            self.write(
                root,
                "docs/track/HANDOVER.md",
                (
                    "HGTXR Hardware Handover\n"
                    "manifest `48/48` consistency checks `3` required `7` sources `9` "
                    "reflected `10`, partial `1`, blocked `1` "
                    "operator handoff validation `131/131` bundle validation `152/152`\n"
                    "`generated/signoff/final_signoff_bundle_validation_2026_06_10.md`, "
                    "status `pass`, checks `150`, fail `0`.\n"
                    "`generated/signoff/final_operator_handoff_validation_2026_06_10.md`, "
                    "status `pass`, checks `129`, fail `0`.\n"
                ),
            )

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("current_handover_records_live_operator_handoff_validation_artifact_summary", failed)
            self.assertIn("current_handover_records_live_bundle_validation_artifact_summary", failed)

    def test_stale_xr_policy_check_count_is_current_doc_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            self.write(
                root,
                "docs/track/HANDOVER.md",
                (
                    "HGTXR Hardware Handover\n"
                    "manifest `48/48` consistency checks `3` required `7` sources `9` "
                    "reflected `10`, partial `1`, blocked `1` "
                    "operator handoff validation `131/131` bundle validation `152/152` "
                    "policy check count `7`\n"
                ),
            )

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("current_docs_have_no_stale_xr_policy_check_count", failed)

    def test_direct_choice_doc_preferred_over_track_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["documents"]["choice"], str(root / "docs" / "CHOICE.md"))

    def test_missing_spec_keyword_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            self.write(root, "docs/Spec.md", "spec-kit manual ZCU104\n")

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("spec_records_param_knobs", failed)

    def test_self_gated_manifest_failures_do_not_deadlock_spec_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            manifest_path = root / "docs/resources/final_evidence_manifest_2026_06_10.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "fail"
            manifest["failed_consistency_checks"] = [
                "spec_plan_current_doc_freshness_contract",
                "third_goal_completion_audit_req12_manifest_pass",
            ]
            self.write_json(root, "docs/resources/final_evidence_manifest_2026_06_10.json", manifest)

            audit = audit_tool.build_audit(root)

            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertNotIn("evidence_manifest_pass", failed)

    def test_external_manifest_failure_blocks_spec_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            manifest_path = root / "docs/resources/final_evidence_manifest_2026_06_10.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "fail"
            manifest["failed_consistency_checks"] = ["final_unblock_closeout_validation_pass"]
            self.write_json(root, "docs/resources/final_evidence_manifest_2026_06_10.json", manifest)

            audit = audit_tool.build_audit(root)

            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("evidence_manifest_pass", failed)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_minimal_root(root)
            json_out = Path(tmp) / "audit.json"
            markdown_out = Path(tmp) / "audit.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = audit_tool.main(
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
            self.assertIn("Spec/Plan Conformance Audit", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
