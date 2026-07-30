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

import write_e2e_resource_matrix as matrix_tool  # noqa: E402


MINI_XML = """\
<profile>
  <UserAssignments>
    <TopModelName>demo_top</TopModelName>
    <TargetClockPeriod>5.00</TargetClockPeriod>
  </UserAssignments>
  <PerformanceEstimates>
    <SummaryOfTimingAnalysis><EstimatedClockPeriod>3.744</EstimatedClockPeriod></SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency>
      <Best-caseLatency>1000</Best-caseLatency>
      <Best-caseRealTimeLatency>0.005 sec</Best-caseRealTimeLatency>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources><BRAM_18K>10</BRAM_18K><DSP>20</DSP><FF>30</FF><LUT>40</LUT><URAM>5</URAM></Resources>
    <AvailableResources><BRAM_18K>100</BRAM_18K><DSP>200</DSP><FF>300</FF><LUT>400</LUT><URAM>50</URAM></AvailableResources>
  </AreaEstimates>
</profile>
"""

MINI_TIMING = """\
| Design Timing Summary
    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)
      1.926        0.000                      0               124537        0.010        0.000
"""

MINI_POWER = """\
| Total On-Chip Power (W)  | 4.060        |
| Dynamic (W)              | 3.356        |
| Device Static (W)        | 0.703        |
"""


class WriteE2EResourceMatrixTests(unittest.TestCase):
    def test_parse_hls_csynth_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "csynth.xml"
            path.write_text(MINI_XML)

            parsed = matrix_tool.parse_hls_csynth_xml(path)

            self.assertEqual(parsed["top"], "demo_top")
            self.assertEqual(parsed["latency_cycles"], 1000)
            self.assertEqual(parsed["resources"]["dsp"], 20)
            self.assertEqual(parsed["utilization_pct"]["uram"], 10.0)

    def test_parse_timing_and_power_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            timing = Path(tmp) / "timing.rpt"
            power = Path(tmp) / "power.rpt"
            timing.write_text(MINI_TIMING)
            power.write_text(MINI_POWER)

            self.assertEqual(matrix_tool.parse_timing_summary(timing)["wns_ns"], 1.926)
            self.assertEqual(matrix_tool.parse_power(power)["total_on_chip_w"], 4.06)

    def test_cli_writes_current_resource_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "matrix.json"
            markdown_out = Path(tmp) / "matrix.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = matrix_tool.main(
                    [
                        "--root",
                        str(ROOT.parent),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["summary"]["variant_count"], 4)
            self.assertEqual(payload["summary"]["best_latency_variants"], ["C1", "C3b"])
            self.assertEqual(payload["summary"]["highest_dsp_variants"], ["C1", "C3b"])
            self.assertEqual(payload["summary"]["recommended_board_smoke_variant"], "C3b")
            rows = {row["id"]: row for row in payload["rows"]}
            self.assertEqual(rows["A1"]["hls"]["resources"]["dsp"], 332)
            self.assertEqual(rows["C3b"]["hls"]["resources"]["dsp"], 604)
            self.assertIn("HGTXR E2E Resource Matrix", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
