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

import audit_final_unblock_candidates as audit_tool  # noqa: E402
from test_validate_pynq_smoke_result import c3b_result  # noqa: E402


class AuditFinalUnblockCandidatesTests(unittest.TestCase):
    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name) / "XR-VIT"
        root = base / "HGTXR"
        replacement = base / "XR_Accel"
        audit_path = root / "docs" / "resources" / "xr_vits_candidate_audit_2026_06_10.json"
        replacement.mkdir(parents=True)
        audit_path.parent.mkdir(parents=True)
        audit_path.write_text(
            json.dumps(
                {
                    "requested_path": str(base.parent / "XR-VITs"),
                    "recommendation": {
                        "role": "candidate-xr-accel",
                        "path": str(replacement),
                        "score": 99,
                    },
                }
            )
            + "\n"
        )
        return tmp, root, replacement

    def test_missing_inputs_report_remaining_blockers_without_side_effects(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)

        audit = audit_tool.build_audit(
            root=root,
            c3b_smoke_json=None,
            c3b_require_paths=True,
            xr_vits_mode="replacement",
            approved_by="",
            reason="",
            replacement_path=None,
        )

        self.assertEqual(audit["status"], "blocked")
        self.assertIn("C3b AXIS/DMA physical smoke result", audit["remaining_blockers"])
        self.assertIn("requested XR-VITs sibling", audit["remaining_blockers"])
        self.assertFalse(audit["safety"]["writes_canonical_inputs"])
        self.assertFalse((root / "docs" / "resources" / "xr_vits_replacement_policy.json").exists())

    def test_valid_c3b_and_replacement_approval_would_clear_all(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        smoke = Path(tmp.name) / "board" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
        smoke.parent.mkdir()
        smoke.write_text(json.dumps(c3b_result()) + "\n")

        audit = audit_tool.build_audit(
            root=root,
            c3b_smoke_json=smoke,
            c3b_require_paths=True,
            xr_vits_mode="replacement",
            approved_by="unit-test",
            reason="unit test approval",
            replacement_path=None,
        )

        self.assertEqual(audit["status"], "would-clear")
        self.assertEqual(audit["remaining_blockers"], [])
        self.assertTrue(audit["c3b_smoke"]["would_clear"])
        self.assertTrue(audit["xr_vits"]["would_clear"])
        self.assertFalse((root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json").exists())
        self.assertFalse((root / "docs" / "resources" / "xr_vits_replacement_policy.json").exists())

    def test_exact_xr_vits_mode_uses_requested_directory(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        requested = root.parent.parent / "XR-VITs"
        requested.mkdir()
        smoke = Path(tmp.name) / "smoke.json"
        smoke.write_text(json.dumps(c3b_result()) + "\n")

        audit = audit_tool.build_audit(
            root=root,
            c3b_smoke_json=smoke,
            c3b_require_paths=True,
            xr_vits_mode="exact",
            approved_by="",
            reason="",
            replacement_path=None,
        )

        self.assertEqual(audit["status"], "would-clear")
        self.assertEqual(audit["xr_vits"]["mode"], "exact")
        self.assertEqual(audit["xr_vits"]["requested_path"], str(requested.resolve()))

    def test_cli_writes_json_and_markdown(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        json_out = Path(tmp.name) / "audit.json"
        markdown_out = Path(tmp.name) / "audit.md"
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

        self.assertEqual(code, 1)
        self.assertEqual(json.loads(json_out.read_text())["status"], "blocked")
        self.assertIn("HGTXR Final Unblock Candidate Audit", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
