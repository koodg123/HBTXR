#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_req2_spec_subagent_gate_audit as audit_tool  # noqa: E402


class Req2SpecSubagentGateAuditTests(unittest.TestCase):
    def write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_json(self, path: Path, payload: dict[str, object]) -> None:
        self.write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        hgtxr = Path(tmp.name) / "XR-VIT" / "HGTXR"
        hardware = hgtxr / "hardware"
        common = (
            "ZCU104 C3b VREF Q4 Q8 HGTXR_PARALLELISM_FACTOR "
            "HGTXR_BUS_WIDTH HGTXR_FIFO_DEPTH spec-kit specify not available "
            "not found manual manual spec GPT5.3-Codex-Spark Spark GPT5.5 "
            "fallback quota sub-agent Task Cards\n"
        )
        for rel in [
            "docs/Master-Plan.md",
            "docs/Sub-Plan.md",
            "docs/Spec.md",
            "docs/Execution.md",
            "docs/Validation.md",
            "docs/track/PROGRESS.md",
            "docs/track/THIRD_GOAL_REQUIREMENTS_2026_06_16.md",
        ]:
            self.write(hardware / rel, f"# {rel}\n{common}")
        for rel in [
            "docs/Master-Plan.md",
            "docs/Sub-Plan.md",
            "docs/Spec.md",
            "docs/Execution.md",
            "docs/Validation.md",
            "docs/track/PROGRESS.md",
            "docs/track/log.md",
        ]:
            self.write(hgtxr / rel, f"# {rel}\n{common}")
        self.write_json(hardware / "generated/signoff/spec_plan_conformance_audit_2026_06_10.json", {"status": "pass"})
        self.write_json(hardware / "generated/signoff/third_goal_requirements_trace_2026_06_10.json", {"status": "partial"})
        return tmp, hgtxr, hardware

    def test_build_audit_passes_with_manual_spec_and_fallback_evidence(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)

        with patch("write_req2_spec_subagent_gate_audit.shutil.which", return_value=None):
            audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "pass-manual-spec-and-model-fallback")
        self.assertFalse(audit["spec_kit"]["available"])
        self.assertTrue(audit["spec_kit"]["manual_fallback_recorded"])
        self.assertTrue(audit["subagents"]["spark_first_recorded"])
        self.assertTrue(audit["subagents"]["gpt55_fallback_recorded"])
        self.assertTrue(audit["plan_spec"]["plan_spec_covered"])

    def test_build_audit_is_partial_when_manual_fallback_missing(self) -> None:
        tmp, hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        minimal = (
            "ZCU104 C3b VREF Q4 Q8 HGTXR_PARALLELISM_FACTOR "
            "HGTXR_BUS_WIDTH HGTXR_FIFO_DEPTH GPT5.3-Codex-Spark Spark "
            "GPT5.5 fallback quota sub-agent Task Cards\n"
        )
        for rel in [
            "docs/Master-Plan.md",
            "docs/Sub-Plan.md",
            "docs/Spec.md",
            "docs/Execution.md",
            "docs/Validation.md",
            "docs/track/PROGRESS.md",
            "docs/track/THIRD_GOAL_REQUIREMENTS_2026_06_16.md",
        ]:
            self.write(hardware / rel, f"# {rel}\n{minimal}")
        for rel in [
            "docs/Master-Plan.md",
            "docs/Sub-Plan.md",
            "docs/Spec.md",
            "docs/Execution.md",
            "docs/Validation.md",
            "docs/track/PROGRESS.md",
            "docs/track/log.md",
        ]:
            self.write(hgtxr / rel, f"# {rel}\n{minimal}")

        with patch("write_req2_spec_subagent_gate_audit.shutil.which", return_value=None):
            audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "partial")

    def test_write_outputs(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        with patch("write_req2_spec_subagent_gate_audit.shutil.which", return_value=None):
            audit = audit_tool.build_audit(hardware)
        json_out = hardware / "generated/signoff/req2.json"
        md_out = hardware / "generated/signoff/req2.md"

        audit_tool.write_outputs(audit, json_out, md_out)

        self.assertEqual(json.loads(json_out.read_text())["status"], "pass-manual-spec-and-model-fallback")
        self.assertIn("Req2 Spec/Sub-Agent Gate Audit", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
