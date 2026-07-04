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

import create_xr_vits_replacement_policy  # noqa: E402
import write_final_unblock_intake as intake_tool  # noqa: E402
from test_validate_pynq_smoke_result import c3b_result  # noqa: E402


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


class WriteFinalUnblockIntakeTests(unittest.TestCase):
    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name) / "XR-VIT"
        root = base / "HGTXR"
        replacement = base / "XR_Accel"
        replacement.mkdir(parents=True)
        write_json(
            root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
            {
                "requested_path": str(create_xr_vits_replacement_policy.default_requested_path(root).resolve()),
                "recommendation": {
                    "role": "candidate-xr-accel",
                    "path": str(replacement.resolve()),
                    "score": 99,
                },
            },
        )
        return tmp, root, replacement

    def test_missing_inputs_reports_blocked_without_side_effects(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)

        intake = intake_tool.build_intake(
            root=root,
            c3b_smoke_json=None,
            c3b_require_paths=True,
            xr_vits_mode="replacement",
            approved_by="",
            reason="",
            replacement_path=None,
        )

        self.assertEqual(intake["status"], "blocked")
        self.assertFalse(intake["candidate_audit"]["would_clear_all"])
        self.assertEqual(intake["dry_run_final_runner_command"], "")
        self.assertFalse(intake["safety"]["writes_canonical_inputs"])
        self.assertIn("C3b AXIS/DMA physical smoke result", intake["blocker_status"])
        self.assertIn("requested XR-VITs sibling", intake["blocker_status"])
        self.assertEqual(
            intake["blocker_status"]["C3b AXIS/DMA physical smoke result"]["next_input"],
            "board-produced C3b smoke JSON",
        )
        self.assertIn(
            "e2e_axis_dma_c3b_mem16_file_smoke.json",
            intake["blocker_status"]["C3b AXIS/DMA physical smoke result"]["next_input_path"],
        )
        self.assertEqual(len(intake["next_inputs"]), 2)
        self.assertIn("Supply inputs", intake["next_action"])
        self.assertFalse((root / "docs" / "resources" / "xr_vits_replacement_policy.json").exists())

    def test_valid_candidates_emit_dry_run_and_active_commands(self) -> None:
        tmp, root, replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        smoke = Path(tmp.name) / "board" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
        write_json(smoke, c3b_result())

        intake = intake_tool.build_intake(
            root=root,
            c3b_smoke_json=smoke,
            c3b_require_paths=True,
            xr_vits_mode="replacement",
            approved_by="unit-test",
            reason="unit test approval",
            replacement_path=replacement,
        )

        self.assertEqual(intake["status"], "ready-for-active-unblock")
        self.assertTrue(intake["candidate_audit"]["would_clear_all"])
        self.assertIn("--dry-run-import-c3b-smoke", intake["dry_run_final_runner_command"])
        self.assertIn("--dry-run-xr-vits-replacement", intake["dry_run_final_runner_command"])
        self.assertIn("--allow-blocked", intake["dry_run_final_runner_command"])
        self.assertIn("--approve-xr-vits-replacement", intake["active_final_runner_command"])
        self.assertIn("--xr-vits-replacement-path", intake["active_final_runner_command"])
        self.assertIn(str(replacement), intake["active_final_runner_command"])
        self.assertEqual(intake["operator_sequence"][1]["step"], "dry-run-final-runner")
        self.assertFalse(intake["operator_sequence"][1]["side_effects"])
        self.assertTrue(intake["operator_sequence"][2]["side_effects"])
        self.assertTrue(intake["blocker_status"]["C3b AXIS/DMA physical smoke result"]["ready_for_active_unblock"])
        self.assertTrue(intake["blocker_status"]["requested XR-VITs sibling"]["ready_for_active_unblock"])
        self.assertEqual(intake["next_inputs"], [])
        self.assertFalse((root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json").exists())

    def test_cli_writes_json_and_markdown(self) -> None:
        tmp, root, _replacement = self.make_root()
        self.addCleanup(tmp.cleanup)
        json_out = Path(tmp.name) / "intake.json"
        markdown_out = Path(tmp.name) / "intake.md"
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            code = intake_tool.main(
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
        self.assertIn("HGTXR Final Unblock Intake", markdown_out.read_text())
        self.assertIn("Blocker Status", markdown_out.read_text())
        self.assertIn("Next Inputs", markdown_out.read_text())
        self.assertIn("Dry-Run Final Runner", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
