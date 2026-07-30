from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_prefetchall4_300_resource_breakdown as breakdown_tool  # noqa: E402


REPORT = """
1. CLB Logic
------------

+----------------------------+-------+-------+------------+-----------+-------+
|          Site Type         |  Used | Fixed | Prohibited | Available | Util% |
+----------------------------+-------+-------+------------+-----------+-------+
| CLB LUTs*                  | 1,000 |     0 |          0 |    10,000 | 10.00 |
|   LUT as Logic             |   900 |     0 |          0 |    10,000 |  9.00 |
|   LUT as Memory            |   100 |     0 |          0 |     5,000 |  2.00 |
| CLB Registers              | 2,000 |     0 |          0 |    20,000 | 10.00 |
+----------------------------+-------+-------+------------+-----------+-------+

2. BLOCKRAM
-----------

+-------------------+------+-------+------------+-----------+-------+
|     Site Type     | Used | Fixed | Prohibited | Available | Util% |
+-------------------+------+-------+------------+-----------+-------+
| Block RAM Tile    |  4.5 |     0 |          0 |       100 |  4.50 |
|   RAMB36/FIFO*    |    4 |     0 |          0 |       100 |  4.00 |
|   RAMB18          |    1 |     0 |          0 |       200 |  0.50 |
| URAM              |    8 |     0 |          0 |        20 | 40.00 |
+-------------------+------+-------+------------+-----------+-------+

3. ARITHMETIC
-------------

+-----------+------+-------+------------+-----------+-------+
| Site Type | Used | Fixed | Prohibited | Available | Util% |
+-----------+------+-------+------------+-----------+-------+
| DSPs      |   16 |     0 |          0 |       100 | 16.00 |
+-----------+------+-------+------------+-----------+-------+
"""

HIERARCHICAL_REPORT = """
+-----------------------------------------------+--------+------------+------+--------+--------+------+------------+
| Instance                                      | Module | Total LUTs | FFs  | RAMB36 | RAMB18 | URAM | DSP Blocks |
+-----------------------------------------------+--------+------------+------+--------+--------+------+------------+
| hgtxr_e2e_axis_dma_system_i/hgtxr_e2e_axis_top_0 | hls    |        300 |  600 |      4 |      2 |    8 |          5 |
| hgtxr_e2e_axis_dma_system_i/axi_dma_in           | dma    |         20 |   30 |      1 |      0 |    0 |          0 |
+-----------------------------------------------+--------+------------+------+--------+--------+------+------------+
"""


class PrefetchAll4ResourceBreakdownTests(unittest.TestCase):
    def test_parse_resource_table_handles_starred_rows_and_float_bram_tiles(self) -> None:
        rows = breakdown_tool.parse_resource_table(REPORT)

        self.assertEqual(rows["clb_luts"]["used"], 1000.0)
        self.assertEqual(rows["clb_luts"]["available"], 10000.0)
        self.assertEqual(rows["block_ram_tile"]["used"], 4.5)
        self.assertEqual(rows["uram"]["device_util_pct"], 40.0)
        self.assertEqual(rows["dsp"]["used"], 16.0)

    def test_build_breakdown_reports_top_and_ooc_block_shares(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            top = root / "top.rpt"
            block = root / "block.rpt"
            top.write_text(REPORT)
            block.write_text(REPORT.replace("1,000", "250").replace("2,000", "500").replace("|   16 |", "|    4 |"))

            result = breakdown_tool.build_breakdown(
                top_report=top,
                major_reports={"pl_hgtxr_e2e_axis_top": block},
                hierarchical_report=root / "missing_hier.rpt",
            )
            resources = result["major_blocks"]["pl_hgtxr_e2e_axis_top"]["resources"]

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["summary"]["block_count"], 1)
            self.assertEqual(result["summary"]["highest_pressure"]["resource"], "uram")
            self.assertAlmostEqual(resources["clb_luts"]["share_of_placed_top_pct"], 25.0)
            self.assertAlmostEqual(resources["dsp"]["share_of_placed_top_pct"], 25.0)

    def test_build_breakdown_prefers_implemented_hierarchical_report_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            top = root / "top.rpt"
            fallback = root / "fallback.rpt"
            hierarchical = root / "hier.rpt"
            top.write_text(REPORT)
            fallback.write_text(REPORT)
            hierarchical.write_text(HIERARCHICAL_REPORT)

            result = breakdown_tool.build_breakdown(
                top_report=top,
                major_reports={"pl_hgtxr_e2e_axis_top": fallback},
                hierarchical_report=hierarchical,
            )
            block = result["major_blocks"]["pl_hgtxr_e2e_axis_top"]
            resources = block["resources"]

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["hierarchical_report"]["rows_parsed"], 2)
            self.assertEqual(block["evidence_level"], "implemented_hierarchical_utilization")
            self.assertIn("hgtxr_e2e_axis_top_0", block["instance"])
            self.assertEqual(resources["clb_luts"]["used"], 300.0)
            self.assertEqual(resources["clb_registers"]["used"], 600.0)
            self.assertEqual(resources["block_ram_tile"]["used"], 5.0)
            self.assertEqual(resources["uram"]["used"], 8.0)
            self.assertEqual(resources["dsp"]["used"], 5.0)


if __name__ == "__main__":
    unittest.main()
