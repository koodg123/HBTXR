#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_xr_vits_gate_audit as audit_tool  # noqa: E402
import create_xr_vits_replacement_policy as policy_tool  # noqa: E402


class XrVitsGateAuditTests(unittest.TestCase):
    def write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name)
        hgtxr = base / "XR-VIT" / "HGTXR"
        hardware = hgtxr / "hardware"
        replacement = base / "XR-VIT" / "XR_Accel"
        replacement.mkdir(parents=True)
        self.write_json(
            hgtxr / "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
            {
                "status": "candidate-found",
                "requested_path": str(base / "XR-VITs"),
                "approved_replacement": False,
                "recommendation": {
                    "role": "candidate-xr-accel",
                    "path": str(replacement),
                    "score": 99,
                },
                "candidates": [{"path": str(replacement)}],
                "policy": "approval required",
            },
        )
        self.write_json(hgtxr / "docs/resources/xr_vits_replacement_policy.template.json", {"template": True})
        self.write_json(hardware / "generated/signoff/xr_vits_reference_resolution_2026_06_10.json", {"status": "candidate-ready-needs-approval"})
        self.write_json(hardware / "generated/signoff/xr_vits_unblock_packet_2026_06_10.json", {"status": "pending-user-choice"})
        return tmp, hgtxr, hardware

    def test_build_audit_reports_candidate_ready_without_policy(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "blocked")
        self.assertEqual(audit["resolution_mode"], "candidate-ready-needs-approval")
        self.assertFalse(audit["exact"]["exists"])
        self.assertTrue(audit["replacement_candidate"]["exists"])
        self.assertTrue(audit["candidate_audit"]["recommendation_matches"])
        self.assertEqual(audit["active_policy"]["status"], "missing")
        self.assertIn("requested XR-VITs sibling", audit["remaining_blockers"])
        options = {option["id"]: option for option in audit["decision_options"]}
        self.assertIn("X1", options)
        self.assertIn("X2", options)
        self.assertIn("X3", options)
        self.assertIn("--dry-run", options["X2"]["dry_run_command"])
        self.assertTrue(options["X2"]["writes_policy"])

    def test_build_audit_passes_when_exact_path_exists(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / "XR-VITs").mkdir()

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "pass-exact")
        self.assertEqual(audit["resolution_mode"], "exact")
        self.assertEqual(audit["remaining_blockers"], [])

    def test_build_audit_reports_policy_integrity_fingerprint(self) -> None:
        tmp, hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        audit_path = hgtxr / policy_tool.DEFAULT_AUDIT_REL
        candidate_audit = policy_tool.load_candidate_audit(audit_path)
        replacement = Path(tmp.name) / "XR-VIT" / "XR_Accel"
        policy = policy_tool.build_policy(
            requested_path=policy_tool.default_requested_path(hgtxr).resolve(),
            replacement_path=replacement.resolve(),
            approved_by="unit-test",
            approved_at="2026-06-10T00:00:00Z",
            reason="unit test approval",
            reason_code=policy_tool.DEFAULT_REASON_CODE,
            candidate_audit_rel=policy_tool.DEFAULT_AUDIT_REL,
            candidate_audit_path=audit_path.resolve(),
            audit=candidate_audit,
        )
        self.write_json(hgtxr / policy_tool.DEFAULT_POLICY_REL, policy)

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "pass-replacement-policy")
        self.assertEqual(audit["active_policy_integrity"]["status"], "pass")
        self.assertEqual(audit["active_policy_integrity"]["policy_fingerprint"], policy["policy_fingerprint"])
        self.assertEqual(audit["active_policy_summary"]["policy_fingerprint"], policy["policy_fingerprint"])

    def test_write_outputs_creates_json_and_markdown(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        audit = audit_tool.build_audit(hardware)
        json_out = hardware / "generated/signoff/xr.json"
        md_out = hardware / "generated/signoff/xr.md"

        audit_tool.write_outputs(audit, json_out, md_out)

        self.assertEqual(json.loads(json_out.read_text())["status"], "blocked")
        self.assertIn("XR-VITs Gate Audit", md_out.read_text())
        self.assertIn("active_policy_integrity_status", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
