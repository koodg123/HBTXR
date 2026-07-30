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

import write_vref_p0_pot_scale_sweep as sweep_tool  # noqa: E402


class WriteVrefP0PotScaleSweepTests(unittest.TestCase):
    def test_build_sweep_recommends_current_scales(self) -> None:
        sweep = sweep_tool.build_sweep(
            ROOT.parent,
            specs=[
                ROOT / "refs" / "e2e_axis_vector_hgpipe_math_spec.json",
                ROOT / "refs" / "e2e_axis_vector_hgpipe_math_lnq_spec.json",
            ],
        )

        self.assertEqual(sweep["status"], "pass")
        self.assertEqual(sweep["summary"]["total_fail_count"], 0)
        self.assertEqual(sweep["summary"]["specs_recommending_current"], 2)
        self.assertEqual(sweep["summary"]["total_candidate_count"], 30)
        for result in sweep["specs"]:
            self.assertEqual(result["recommended_candidate"], "current")
            current = next(row for row in result["candidates"] if row["candidate"] == "current")
            self.assertTrue(current["exact_raw_match"])
            self.assertEqual(current["raw_l1_delta"], 0)
            self.assertTrue(current["all_scales_power_of_two"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "sweep.json"
            markdown_out = Path(tmp) / "sweep.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = sweep_tool.main(
                    [
                        "--root",
                        str(ROOT.parent),
                        "--spec",
                        str(ROOT / "refs" / "e2e_axis_vector_hgpipe_math_spec.json"),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["spec_count"], 1)
            text = markdown_out.read_text()
            self.assertIn("VREF-P0-01 PoT Scale Candidate Sweep", text)
            self.assertIn("recommended_candidate", text)


if __name__ == "__main__":
    unittest.main()
