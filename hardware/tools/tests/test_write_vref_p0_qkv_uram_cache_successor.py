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

import write_vref_p0_qkv_uram_cache_successor as writer  # noqa: E402


def csynth_xml(clock: float, latency: int, bram: int, dsp: int, ff: int, lut: int, uram: int) -> str:
    return f"""<profile>
  <PerformanceEstimates>
    <SummaryOfTimingAnalysis><EstimatedClockPeriod>{clock}</EstimatedClockPeriod></SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency><Average-caseLatency>{latency}</Average-caseLatency></SummaryOfOverallLatency>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources>
      <BRAM_18K>{bram}</BRAM_18K>
      <DSP>{dsp}</DSP>
      <FF>{ff}</FF>
      <LUT>{lut}</LUT>
      <URAM>{uram}</URAM>
    </Resources>
  </AreaEstimates>
</profile>
"""


class VrefP0QkvUramCacheSuccessorTests(unittest.TestCase):
    def write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / "HGTXR"
        hardware = root / "hardware"
        self.write(hardware / "hls/include/hgtxr_e2e_vit.hpp", "#define HGTXR_E2E_URAM_QKV_WEIGHT_CACHE 0\n")
        self.write(
            hardware
            / "generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csim/solution_e2e_q4w8a/csim/report/hgtxr_e2e_axis_top_csim.log",
            "runtime_state=2 count=6 last=1 failures=0\n"
            "E2E AXIS vector comparison passed\n"
            "CSim done with 0 errors\n",
        )
        self.write(
            hardware
            / "generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csynth/solution_e2e_q4w8a/syn/report/csynth.xml",
            csynth_xml(4.069, 498485, 114, 128, 19664, 43236, 40),
        )
        self.write(
            hardware
            / "generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_csynth/solution_e2e_q4w8a/syn/report/csynth.xml",
            csynth_xml(4.069, 498485, 144, 128, 19664, 43236, 32),
        )
        self.write(
            hardware / "generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_csynth/solution_e2e_q4w8a/syn/report/csynth.xml",
            csynth_xml(4.069, 498485, 64, 128, 19664, 43236, 88),
        )
        return tmp, root

    def test_build_audit_passes_with_qkv_uram_resources(self) -> None:
        tmp, root = self.make_root()
        self.addCleanup(tmp.cleanup)

        audit = writer.build_audit(root)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["csim"]["status"], "pass")
        self.assertEqual(audit["csynth"]["resources"]["bram_18k"], 114)
        self.assertEqual(audit["csynth"]["resources"]["uram"], 40)
        self.assertEqual(audit["comparison"]["delta_vs_dsp_mixed_stream"]["bram_18k"], -30)
        self.assertEqual(audit["comparison"]["delta_vs_dsp_mixed_stream"]["uram"], 8)
        self.assertEqual(audit["ip_package"]["status"], "missing")
        self.assertEqual(audit["overlay"]["status"], "missing")
        self.assertEqual(audit["physical_smoke"]["status"], "missing")
        self.assertEqual(audit["pynq_plumbing"]["variant"], "vref-p0-softmax-input-x2-qkv-uram")
        self.assertEqual(audit["pynq_plumbing"]["preset"], "axis-vref-p0-softmax-input-x2-qkv-uram")
        self.assertIn("--variant vref-p0-softmax-input-x2-qkv-uram", audit["pynq_plumbing"]["package_command"])
        self.assertEqual(
            audit["promotion"]["remaining"],
            ["ip_package", "routed_overlay_timing", "physical_smoke_json"],
        )
        failed = [check["name"] for check in audit["checks"] if check["status"] == "fail"]
        self.assertEqual(failed, [])

    def test_optional_route_package_evidence_detected(self) -> None:
        tmp, root = self.make_root()
        self.addCleanup(tmp.cleanup)
        hardware = root / "hardware"
        ip_root = (
            hardware
            / "generated"
            / writer.QKV_IP_PROJECT
            / "solution_e2e_q4w8a"
            / "impl"
        )
        self.write(ip_root / "ip/component.xml", "<component><name>hgtxr_e2e_axis_top</name></component>\n")
        self.write(ip_root / "export.zip", "fake zip marker\n")
        overlay_dir = hardware / "generated/build/vivado/overlay" / writer.QKV_OVERLAY_PROJECT
        impl_dir = (
            hardware
            / "generated/build/vivado"
            / writer.QKV_OVERLAY_PROJECT
            / f"{writer.QKV_OVERLAY_PROJECT}.runs"
            / "impl_1"
        )
        wrapper = f"{writer.QKV_BD_NAME}_wrapper"
        self.write(overlay_dir / f"{writer.QKV_ARTIFACT}.bit", "bit\n")
        self.write(overlay_dir / f"{writer.QKV_ARTIFACT}.hwh", "hwh\n")
        self.write(hardware / "pynq/hgtxr" / f"{writer.QKV_ARTIFACT}.bit", "bit\n")
        self.write(hardware / "pynq/hgtxr" / f"{writer.QKV_ARTIFACT}.hwh", "hwh\n")
        self.write(
            impl_dir / f"{wrapper}_timing_summary_routed.rpt",
            "Design Timing Summary\n"
            "---------------------\n"
            "4.497 0.000 0 0 0.123 0.000 0\n",
        )
        self.write(
            impl_dir / f"{wrapper}_route_status.rpt",
            "# of routable nets: 10\n"
            "# of fully routed nets: 10\n"
            "# of nets with routing errors: 0\n",
        )

        audit = writer.build_audit(root)

        self.assertEqual(audit["ip_package"]["status"], "pass")
        self.assertEqual(audit["overlay"]["status"], "pass")
        self.assertEqual(audit["overlay"]["timing"]["wns_ns"], 4.497)
        self.assertEqual(audit["overlay"]["route_status"]["routing_error_nets"], 0)
        self.assertEqual(audit["promotion"]["remaining"], ["physical_smoke_json"])

    def test_write_outputs(self) -> None:
        tmp, root = self.make_root()
        self.addCleanup(tmp.cleanup)
        json_out = root / "hardware/generated/signoff/qkv.json"
        md_out = root / "hardware/generated/signoff/qkv.md"

        code = writer.main(["--root", str(root), "--json-out", str(json_out), "--markdown-out", str(md_out)])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
        self.assertIn("QKV Weight Cache URAM", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
