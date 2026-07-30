#!/usr/bin/env python3
"""Unit tests for third-goal preflight policy transitions."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_third_goal_preflight as preflight  # noqa: E402
import create_xr_vits_replacement_policy  # noqa: E402


CONFIG_TEXT = """\
#define HGTXR_PARALLELISM_FACTOR 8
#define HGTXR_BUS_WIDTH 256
#define HGTXR_BIT_WIDTH 8
#define HGTXR_WEIGHT_BIT_WIDTH 4
#define HGTXR_BUFFER_SIZE 256
#define HGTXR_FIFO_DEPTH 128
#define HGTXR_E2E_ACTIVE_TOKENS 196
#define HGTXR_E2E_BLOCKS 6
#define HGTXR_E2E_FF_DIM 768
#define HGTXR_E2E_DENSE_PAR 8
"""

E2E_TOP_TEXT = """\
void hgtxr_e2e_axis_top() {
#pragma HLS INTERFACE axis port=axis_in
#pragma HLS INTERFACE axis port=axis_out
}
"""

E2E_M_AXI_TOP_TEXT = """\
void hgtxr_e2e_m_axi_top() {
#pragma HLS INTERFACE m_axi port=frame
#pragma HLS INTERFACE m_axi port=weights
#pragma HLS INTERFACE m_axi port=out_state
}
"""


def set_q4(data: bytearray, elem_offset: int, raw: int) -> None:
    u32_idx = elem_offset // 8
    shift = (elem_offset % 8) * 4
    word = int.from_bytes(data[u32_idx * 4 : u32_idx * 4 + 4], "little")
    word = (word & ~(0xF << shift)) | ((raw & 0xF) << shift)
    data[u32_idx * 4 : u32_idx * 4 + 4] = word.to_bytes(4, "little")


class ThirdGoalPreflightTests(unittest.TestCase):
    @staticmethod
    def parse_stdout_summary(output: str) -> dict[str, int]:
        for line in output.splitlines():
            if line.startswith("[summary] "):
                return {item.split("=")[0]: int(item.split("=")[1]) for item in line.removeprefix("[summary] ").split()}
        raise AssertionError(f"summary line not found in output:\n{output}")

    @staticmethod
    def assert_check_schema(testcase: unittest.TestCase, checks: list[dict[str, object]]) -> None:
        testcase.assertTrue(checks)
        for check in checks:
            testcase.assertIsInstance(check.get("status"), str)
            testcase.assertIn(check.get("status"), {"ok", "warn", "fail"})
            testcase.assertIsInstance(check.get("name"), str)
            testcase.assertIsInstance(check.get("detail"), str)
            if "path" in check:
                testcase.assertIsInstance(check["path"], str)

    def make_root(
        self,
        *,
        old_flow: bool,
        e2e_hwh: bool,
        generated: bool,
        bit: bool = True,
        weights: bool = True,
        par16: bool = False,
        c3b: bool = False,
        c3b_bundle: bool = False,
        c3b_session: bool = False,
        c3b_result: bool = False,
        c3b_bad_result: bool = False,
        paper: bool = False,
        xr_vits_policy: str = "",
    ) -> Path:
        base = Path(self.tmp.name) / "XR-VIT"
        root = base / "HGTXR"
        hardware = root / "hardware"
        (hardware / "configs").mkdir(parents=True)
        (hardware / "hls" / "src").mkdir(parents=True)
        (hardware / "vivado" / "scripts").mkdir(parents=True)
        (hardware / "pynq" / "hgtxr").mkdir(parents=True)
        (hardware / "refs" / "weights").mkdir(parents=True)

        (hardware / "configs" / "zcu104_e2e_q4w8a_defines.h").write_text(CONFIG_TEXT)
        (hardware / "hls" / "src" / "hgtxr_e2e_axis_top.cpp").write_text(E2E_TOP_TEXT)
        (hardware / "hls" / "src" / "hgtxr_e2e_m_axi_top.cpp").write_text(E2E_M_AXI_TOP_TEXT)

        if paper:
            paper_path = base / "PAPER_PRJXR" / "05_RESOURCES" / "DeiT-Tiny C-Syn Results.png"
            paper_path.parent.mkdir(parents=True)
            paper_path.write_bytes(b"paper")

        if xr_vits_policy:
            policy_dir = root / "docs" / "resources"
            policy_dir.mkdir(parents=True)
            replacement = base / "XR_Accel"
            replacement.mkdir(parents=True)
            audit_path = policy_dir / "xr_vits_candidate_audit_2026_06_10.json"
            audit = {
                "status": "candidate-found",
                "requested_path": str(base.parent / "XR-VITs"),
                "approved_replacement": False,
                "recommendation": {
                    "role": "candidate-xr-accel",
                    "path": str(replacement.resolve()),
                    "score": 99,
                    "reason": "unit fixture",
                },
                "candidates": [],
            }
            audit_path.write_text(json.dumps(audit))
            if xr_vits_policy == "bad-path":
                policy = {
                    "approved_replacement": True,
                    "requested_path": str(base.parent / "XR-VITs"),
                    "replacement_path": str(base / "missing"),
                    "replacement_role": "candidate-xr-accel",
                    "candidate_audit": "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
                    "approved_by": "unit-test",
                    "approved_at": "2026-06-10T00:00:00Z",
                    "reason": "unit test fixture",
                }
            elif xr_vits_policy == "approved":
                policy = create_xr_vits_replacement_policy.build_policy(
                    requested_path=(base.parent / "XR-VITs").resolve(),
                    replacement_path=replacement.resolve(),
                    approved_by="unit-test",
                    approved_at="2026-06-10T00:00:00Z",
                    reason="unit test fixture",
                    reason_code=create_xr_vits_replacement_policy.DEFAULT_REASON_CODE,
                    candidate_audit_rel="docs/resources/xr_vits_candidate_audit_2026_06_10.json",
                    candidate_audit_path=audit_path.resolve(),
                    audit=audit,
                )
            else:
                policy = {
                    "approved_replacement": False,
                    "requested_path": str(base.parent / "XR-VITs"),
                    "replacement_path": str(replacement),
                    "replacement_role": "candidate-xr-accel",
                    "candidate_audit": "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
                    "approved_by": "",
                    "approved_at": "",
                    "reason": "unit test fixture",
                }
            (policy_dir / "xr_vits_replacement_policy.json").write_text(json.dumps(policy))

        if old_flow:
            (hardware / "vivado" / "scripts" / "create_hls_project.tcl").write_text("set_top hgtxr_top\n")
            (hardware / "vivado" / "scripts" / "build_bitstream.tcl").write_text(
                "create_bd_cell -type ip -vlnv xilinx.com:hls:hgtxr_top:1.0 hgtxr_top_0\n"
            )
        else:
            (hardware / "vivado" / "scripts" / "create_hls_project.tcl").write_text("set_top hgtxr_e2e_axis_top\n")
            (hardware / "vivado" / "scripts" / "build_bitstream.tcl").write_text(
                "create_bd_cell -type ip -vlnv xilinx.com:hls:hgtxr_e2e_axis_top:1.0 hgtxr_e2e_axis_top_0\n"
            )
            (hardware / "vivado" / "scripts" / "package_e2e_m_axi_ip.tcl").write_text(
                "set_top hgtxr_e2e_m_axi_top\n"
            )
            (hardware / "vivado" / "scripts" / "build_e2e_m_axi_bitstream.tcl").write_text(
                "create_bd_cell -type ip -vlnv xilinx.com:hls:hgtxr_e2e_m_axi_top:1.0 hgtxr_e2e_m_axi_top_0\n"
                "file copy hgtxr_e2e_m_axi.bit\n"
            )

        if bit:
            (hardware / "pynq" / "hgtxr" / "hgtxr.bit").write_bytes(b"bit")
        hwh_text = "xilinx.com:hls:hgtxr_e2e_axis_top:1.0" if e2e_hwh else "xilinx.com:hls:hgtxr_top:1.0 hgtxr_top_0"
        (hardware / "pynq" / "hgtxr" / "hgtxr.hwh").write_text(hwh_text)
        if not old_flow and e2e_hwh:
            (hardware / "pynq" / "hgtxr" / "hgtxr_e2e_m_axi.bit").write_bytes(b"bit")
            (hardware / "pynq" / "hgtxr" / "hgtxr_e2e_m_axi.hwh").write_text(
                "xilinx.com:hls:hgtxr_e2e_m_axi_top:1.0 hgtxr_e2e_m_axi_top_0"
            )
            (hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma.bit").write_bytes(b"axis-bit")
            (hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma.hwh").write_text(
                "xilinx.com:hls:hgtxr_e2e_axis_top:1.0 hgtxr_e2e_axis_top_0"
            )
            if par16:
                (hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_par16.bit").write_bytes(b"axis-par16-bit")
                (hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_par16.hwh").write_text(
                    "xilinx.com:hls:hgtxr_e2e_axis_top:1.0 hgtxr_e2e_axis_top_0"
                )
            if c3b:
                (hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_c3b_mem16.bit").write_bytes(b"axis-c3b-bit")
                (hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_c3b_mem16.hwh").write_text(
                    "xilinx.com:hls:hgtxr_e2e_axis_top:1.0 hgtxr_e2e_axis_top_0"
                )
                if c3b_bundle:
                    bundle_dir = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
                    bundle_dir.mkdir(parents=True)
                    tar_path = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
                    run_script = bundle_dir / "run_e2e_axis_dma_c3b_mem16_file_smoke.sh"
                    run_script.write_text(
                        "#!/usr/bin/env sh\n"
                        "PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke --variant c3b-mem16 "
                        "--weights-mode file --weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin "
                        "--expect-out-raw 32 -13 26 -6 14 -11 "
                        "--json-out e2e_axis_dma_c3b_mem16_file_smoke.json\n"
                    )
                    validate_script = bundle_dir / "validate_e2e_axis_dma_c3b_mem16_file_smoke.sh"
                    validate_script.write_text(
                        "#!/usr/bin/env sh\n"
                        "python3 tools/validate_pynq_smoke_result.py "
                        "e2e_axis_dma_c3b_mem16_file_smoke.json "
                        "--preset axis-c3b-mem16 "
                        "--json-out e2e_axis_dma_c3b_mem16_file_smoke_validation.json\n"
                    )
                    for rel in [
                        "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
                        "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                        "hgtxr/run_e2e_axis_dma_smoke.py",
                        "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
                        "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
                        "tools/validate_pynq_smoke_result.py",
                    ]:
                        path = bundle_dir / rel
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(rel)
                    file_entries = []
                    for rel in [
                        "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
                        "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                        "hgtxr/run_e2e_axis_dma_smoke.py",
                        "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
                        "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
                        "tools/validate_pynq_smoke_result.py",
                        "run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                        "validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                    ]:
                        path = bundle_dir / rel
                        file_entries.append(
                            {
                                "bundle": rel,
                                "path": str(path),
                                "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        )
                    manifest = {
                        "name": "hgtxr_e2e_axis_dma_smoke_bundle",
                        "target": "ZCU104 PYNQ",
                        "variant": "c3b-mem16",
                        "artifact_prefix": "hgtxr_e2e_axis_dma_c3b_mem16",
                        "command": run_script.read_text().splitlines()[-1],
                        "validation_command": validate_script.read_text().splitlines()[-1],
                        "expected_runtime_state": 2,
                        "expected_out_raw": [32, -13, 26, -6, 14, -11],
                        "files": file_entries,
                    }
                    (bundle_dir / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
                    with tarfile.open(tar_path, "w:gz") as tar:
                        tar.add(bundle_dir, arcname=bundle_dir.name)
                    manifest["tar"] = {
                        "path": str(tar_path),
                        "bytes": tar_path.stat().st_size,
                        "sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
                    }
                    (bundle_dir / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
                    if c3b_session:
                        session = {
                            "status": "pass",
                            "target": "ZCU104 PYNQ",
                            "variant": "c3b-mem16",
                            "preset": "axis-c3b-mem16",
                            "bundle_dir": str(bundle_dir),
                            "bundle_root": bundle_dir.name,
                            "tar": {
                                "path": str(tar_path),
                                "name": tar_path.name,
                                "bytes": tar_path.stat().st_size,
                                "sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
                            },
                            "expected_runtime_state": 2,
                            "expected_out_raw": [32, -13, 26, -6, 14, -11],
                            "result_json": "e2e_axis_dma_c3b_mem16_file_smoke.json",
                            "validation_json": "e2e_axis_dma_c3b_mem16_file_smoke_validation.json",
                            "canonical_result_path": str(
                                hardware / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
                            ),
                            "bundle_validation_errors": [],
                            "board_steps": [
                                "tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                                "cd e2e_axis_dma_c3b_mem16_smoke_bundle",
                                "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                                "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                            ],
                            "host_steps": [
                                "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --allow-blocked",
                                "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --allow-blocked",
                                "python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff",
                            ],
                        }
                        session_path = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
                        session_path.write_text(json.dumps(session))
                        (hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.md").write_text(
                            "# HGTXR C3b ZCU104 Smoke Session\n"
                            "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh\n"
                            "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh\n"
                            "--import-c3b-smoke-json\n"
                            "--dry-run-import-c3b-smoke\n"
                        )
                if c3b_result or c3b_bad_result:
                    result = {
                        "status": "pass",
                        "variant": "c3b-mem16",
                        "bitfile": "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
                        "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                        "dma_in_name": "axi_dma_in",
                        "dma_out_name": "axi_dma_out",
                        "frame_pattern": "ramp",
                        "weights_mode": "file",
                        "expected_runtime_state": 2,
                        "runtime_state": 2,
                        "runtime_match": True,
                        "out_state": [2.0, -0.8125, 1.625, -0.375, 0.875, -0.6875],
                        "out_raw": [32, -13, 26, -6, 14, -11],
                        "expected_out_raw": [32, -13, 26, -6, 14, -11],
                        "output_match": True,
                    }
                    if c3b_bad_result:
                        result["status"] = "fail"
                        result["runtime_state"] = 0
                    (hardware / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json").write_text(
                        json.dumps(result)
                    )

        if generated:
            gen = hardware / "generated" / "hgtxr_e2e_hls" / "solution1"
            gen.mkdir(parents=True)
            (gen / "run_csim.log").write_text("csim ok\n")
            (gen / "component.xml").write_text("<component/>\n")
            (gen / "hgtxr_e2e_axis_top_csynth.rpt").write_text("csynth ok\n")
            axis_gen = hardware / "generated" / "hgtxr_e2e_axis_hls" / "solution_e2e_q4w8a" / "impl" / "ip"
            axis_gen.mkdir(parents=True)
            (axis_gen / "component.xml").write_text("<component/>\n")
            if par16:
                par16_gen = hardware / "generated" / "hgtxr_e2e_axis_par16_hls" / "solution_e2e_q4w8a" / "impl" / "ip"
                par16_gen.mkdir(parents=True)
                (par16_gen / "component.xml").write_text("<component/>\n")
            if c3b:
                c3b_gen = (
                    hardware
                    / "generated"
                    / "hgtxr_e2e_axis_par16_c3b_mem16_hls"
                    / "solution_e2e_q4w8a"
                    / "impl"
                    / "ip"
                )
                c3b_gen.mkdir(parents=True)
                (c3b_gen / "component.xml").write_text("<component/>\n")

        if weights:
            capacity_u32 = 3048192
            required_u32 = 338640
            data = bytearray(capacity_u32 * 4)
            patch_base = 0
            patch = 16
            embed = 192
            block_base = 49152
            head_base = 2707968
            set_q4(data, patch_base, 1)
            set_q4(data, patch_base + 2 * patch * patch, -1)
            set_q4(data, head_base + embed + 1, -8)
            set_q4(data, block_base + 4 * embed, 7)
            bin_path = hardware / "refs" / "weights" / "e2e_m_axi_active196_b6_ff768_q4_u32.bin"
            manifest_path = hardware / "refs" / "weights" / "e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json"
            bin_path.write_bytes(data)
            manifest_path.write_text(
                json.dumps(
                    {
                        "binary": str(bin_path),
                        "bytes": len(data),
                        "capacity_u32": capacity_u32,
                        "dtype": "uint32",
                        "expected_raw": [32, -13, 26, -6, 14, -11],
                        "expected_runtime_state": 2,
                        "format": "raw-little-endian-uint32",
                        "known_offset_checks": {
                            "block0_wq_diag0": 7,
                            "head_diag1": -8,
                            "patch_channel0_elem0": 1,
                            "patch_channel2_elem0": -1,
                        },
                        "layout": {
                            "block_weight_elem_base": block_base,
                            "blocks": 6,
                            "bus_width_bits": 256,
                            "e2e_weight_words_256b": 381024,
                            "embed": embed,
                            "ff_dim": 768,
                            "head_weight_elem_base": head_base,
                            "patch": patch,
                            "patch_weight_elem_base": patch_base,
                            "q4_lanes_per_256b_word": 64,
                            "required_weight_words_256b": 42330,
                            "state": 6,
                            "u32_per_256b_word": 8,
                            "weight_bits": 4,
                        },
                        "name": "e2e_m_axi_active196_b6_ff768_q4_u32",
                        "required_u32": required_u32,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "shape": [capacity_u32],
                        "tail_zero_start_u32": required_u32,
                    }
                )
            )

        return root

    def run_checker(self, root: Path, mode: str, json_out: Path | None = None) -> tuple[int, str]:
        real_exists = Path.exists

        def patched_exists(path: Path) -> bool:
            text = str(path)
            if text in {
                "/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
                "/tools/Xilinx/Vivado/2023.2/bin/vivado",
            }:
                return True
            return real_exists(path)

        argv = ["check_third_goal_preflight.py", "--root", str(root), "--mode", mode]
        if json_out is not None:
            argv.extend(["--json-out", str(json_out)])
        stream = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(Path, "exists", patched_exists),
            mock.patch.object(preflight.shutil, "which", return_value=None),
            contextlib.redirect_stdout(stream),
        ):
            code = preflight.main()
        return code, stream.getvalue()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_neutral_allows_choice_gated_old_flow_as_warnings(self) -> None:
        root = self.make_root(old_flow=True, e2e_hwh=False, generated=False)

        code, output = self.run_checker(root, "neutral")

        self.assertEqual(code, 0, output)
        self.assertIn("[warn] board flow top mismatch", output)
        self.assertIn("[warn] legacy PYNQ hwh stale old-flow IP", output)
        self.assertIn("[summary]", output)
        self.assertIn("fail=0", output)

    def test_board_ready_rejects_old_flow_and_missing_e2e_package(self) -> None:
        root = self.make_root(old_flow=True, e2e_hwh=False, generated=False)

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] board flow top mismatch", output)
        self.assertIn("[fail] E2E packaged IP component.xml", output)
        self.assertIn("[fail] PYNQ E2E m_axi bit", output)
        self.assertIn("[fail] PYNQ E2E m_axi hwh", output)
        self.assertIn("[warn] legacy PYNQ hwh stale old-flow IP", output)

    def test_board_ready_accepts_selected_e2e_artifacts_with_only_reference_warnings(self) -> None:
        root = self.make_root(old_flow=False, e2e_hwh=True, generated=True)

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 0, output)
        self.assertIn("[ok] board flow selected E2E m_axi", output)
        self.assertIn("[ok] PYNQ hwh selected E2E IP", output)
        self.assertIn("fail=0", output)

    def test_json_out_records_mode_summary_and_checks(self) -> None:
        root = self.make_root(old_flow=False, e2e_hwh=True, generated=True)
        out = Path(self.tmp.name) / "nested" / "preflight.json"

        code, output = self.run_checker(root, "board-ready", json_out=out)

        self.assertEqual(code, 0, output)
        payload = json.loads(out.read_text())
        self.assertEqual(payload["mode"], "board-ready")
        self.assertEqual(payload["summary"], self.parse_stdout_summary(output))
        self.assertEqual(payload["summary"]["fail"], 0)
        self.assertGreater(payload["summary"]["ok"], 0)
        self.assertIsInstance(payload["checks"], list)
        self.assert_check_schema(self, payload["checks"])
        self.assertTrue(
            any(check["name"] == "PYNQ hwh selected E2E IP" and check["status"] == "ok" for check in payload["checks"])
        )
        self.assertTrue(
            any(check["name"] == "E2E packaged IP component.xml" and check["status"] == "ok" for check in payload["checks"])
        )
        self.assertTrue(
            any(check["name"] == "E2E m_axi weight sha256" and check["status"] == "ok" for check in payload["checks"])
        )

    def test_board_ready_records_optional_c1_par16_artifacts_when_present(self) -> None:
        root = self.make_root(old_flow=False, e2e_hwh=True, generated=True, par16=True)

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 0, output)
        self.assertIn("[ok] PYNQ C1 PAR16 AXIS/DMA bit", output)
        self.assertIn("[ok] PYNQ C1 PAR16 AXIS/DMA hwh", output)
        self.assertIn("[ok] PYNQ C1 PAR16 hwh selected E2E IP", output)
        self.assertIn("fail=0", output)

    def test_board_ready_records_optional_c3b_par16_mem16_artifacts_when_present(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
            c3b_result=True,
        )

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 0, output)
        self.assertIn("[ok] PYNQ C3b PAR16 MEM16 AXIS/DMA bit", output)
        self.assertIn("[ok] PYNQ C3b PAR16 MEM16 AXIS/DMA hwh", output)
        self.assertIn("[ok] PYNQ C3b PAR16 MEM16 hwh selected E2E IP", output)
        self.assertIn("[ok] C3b AXIS/DMA PYNQ smoke bundle manifest", output)
        self.assertIn("[ok] C3b AXIS/DMA PYNQ smoke bundle tar sha256", output)
        self.assertIn("[ok] C3b AXIS/DMA PYNQ smoke bundle validation command", output)
        self.assertIn("[ok] C3b ZCU104 smoke session contract", output)
        self.assertIn("[ok] C3b ZCU104 smoke session markdown contract", output)
        self.assertIn("[ok] C3b AXIS/DMA physical smoke result", output)
        self.assertIn("fail=0", output)

    def test_board_ready_warns_missing_c3b_physical_smoke_result(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
        )

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 0, output)
        self.assertIn("[warn] C3b AXIS/DMA physical smoke result", output)
        self.assertIn("final_unblock_commands_2026_06_10.md", output)
        self.assertIn("--dry-run-import-c3b-smoke", output)
        self.assertIn("fail=0", output)

    def test_final_signoff_rejects_missing_c3b_physical_smoke_result(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
        )

        code, output = self.run_checker(root, "final-signoff")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] C3b AXIS/DMA physical smoke result", output)

    def test_final_signoff_rejects_noncanonical_c3b_physical_smoke_result(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
        )
        hardware = root / "hardware"
        noncanonical = (
            hardware
            / "generated"
            / "pynq"
            / "e2e_axis_dma_c3b_mem16_smoke_bundle"
            / "e2e_axis_dma_c3b_mem16_file_smoke.json"
        )
        noncanonical.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "variant": "c3b-mem16",
                    "bitfile": "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
                    "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                    "dma_in_name": "axi_dma_in",
                    "dma_out_name": "axi_dma_out",
                    "weights_mode": "file",
                    "expected_runtime_state": 2,
                    "runtime_state": 2,
                    "runtime_match": True,
                    "out_state": [2.0, -0.8125, 1.625, -0.375, 0.875, -0.6875],
                    "out_raw": [32, -13, 26, -6, 14, -11],
                    "expected_out_raw": [32, -13, 26, -6, 14, -11],
                    "output_match": True,
                }
            )
        )

        code, output = self.run_checker(root, "final-signoff")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] C3b AXIS/DMA physical smoke result", output)
        self.assertIn("noncanonical result found but not accepted", output)
        self.assertIn("pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json", output)

    def test_final_signoff_requires_reference_inputs_after_c3b_result(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
            c3b_result=True,
        )

        code, output = self.run_checker(root, "final-signoff")

        self.assertEqual(code, 1, output)
        self.assertIn("[ok] C3b AXIS/DMA physical smoke result", output)
        self.assertIn("canonical result", output)
        self.assertIn("[fail] requested PAPER_PRJXR DeiT image", output)
        self.assertIn("[fail] requested XR-VITs sibling", output)

    def test_final_signoff_reports_xr_accel_candidate_without_clearing_gate(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
            c3b_result=True,
            paper=True,
        )
        base = root.parent
        replacement = base / "XR_Accel"
        replacement.mkdir()
        audit_dir = root / "docs" / "resources"
        audit_dir.mkdir(parents=True)
        (audit_dir / "xr_vits_candidate_audit_2026_06_10.json").write_text(
            json.dumps(
                {
                    "requested_path": str(base.parent / "XR-VITs"),
                    "recommendation": {"path": str(replacement), "role": "candidate-xr-accel"},
                }
            )
        )

        code, output = self.run_checker(root, "final-signoff")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] requested XR-VITs sibling", output)
        self.assertIn("XR_Accel candidate exists and candidate audit recommends it", output)
        self.assertIn("approval still required", output)
        self.assertIn("xr_vits_unblock_packet_2026_06_10.md", output)
        self.assertIn("create_xr_vits_replacement_policy.py", output)

    def test_final_signoff_accepts_approved_xr_vits_replacement_policy(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
            c3b_result=True,
            paper=True,
            xr_vits_policy="approved",
        )

        code, output = self.run_checker(root, "final-signoff")

        self.assertEqual(code, 0, output)
        self.assertIn("[ok] requested XR-VITs sibling: approved replacement:", output)
        self.assertIn("policy_fingerprint validated", output)
        self.assertIn("fail=0", output)

    def test_final_signoff_rejects_unapproved_xr_vits_replacement_policy(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
            c3b_result=True,
            paper=True,
            xr_vits_policy="unapproved",
        )

        code, output = self.run_checker(root, "final-signoff")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] requested XR-VITs sibling", output)
        self.assertIn("replacement policy rejected", output)
        self.assertIn("approved_replacement is not true", output)

    def test_final_signoff_rejects_legacy_xr_vits_replacement_policy(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
            c3b_result=True,
            paper=True,
        )
        base = root.parent
        replacement = base / "XR_Accel"
        replacement.mkdir(parents=True, exist_ok=True)
        policy_dir = root / "docs" / "resources"
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "xr_vits_candidate_audit_2026_06_10.json").write_text(
            json.dumps(
                {
                    "status": "candidate-found",
                    "requested_path": str(base.parent / "XR-VITs"),
                    "recommendation": {"role": "candidate-xr-accel", "path": str(replacement.resolve())},
                }
            )
        )
        (policy_dir / "xr_vits_replacement_policy.json").write_text(
            json.dumps(
                {
                    "approved_replacement": True,
                    "requested_path": str(base.parent / "XR-VITs"),
                    "replacement_path": str(replacement.resolve()),
                    "replacement_role": "candidate-xr-accel",
                    "candidate_audit": "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
                    "approved_by": "unit-test",
                    "approved_at": "2026-06-10T00:00:00Z",
                    "reason": "legacy",
                }
            )
        )

        code, output = self.run_checker(root, "final-signoff")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] requested XR-VITs sibling", output)
        self.assertIn("regenerate policy", output)

    def test_final_signoff_rejects_xr_vits_replacement_policy_drift(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_session=True,
            c3b_result=True,
            paper=True,
            xr_vits_policy="approved",
        )
        audit_path = root / "docs" / "resources" / "xr_vits_candidate_audit_2026_06_10.json"
        audit = json.loads(audit_path.read_text())
        audit["recommendation"]["score"] = 1
        audit_path.write_text(json.dumps(audit))

        code, output = self.run_checker(root, "final-signoff")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] requested XR-VITs sibling", output)
        self.assertIn("candidate_audit_fingerprint mismatch", output)

    def test_board_ready_rejects_c3b_bundle_without_smoke_session(self) -> None:
        root = self.make_root(old_flow=False, e2e_hwh=True, generated=True, c3b=True, c3b_bundle=True)

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] C3b ZCU104 smoke session json", output)
        self.assertIn("[fail] C3b ZCU104 smoke session markdown", output)

    def test_board_ready_rejects_c3b_artifacts_without_smoke_bundle(self) -> None:
        root = self.make_root(old_flow=False, e2e_hwh=True, generated=True, c3b=True)

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] C3b AXIS/DMA PYNQ smoke bundle manifest", output)
        self.assertIn("[fail] C3b AXIS/DMA PYNQ smoke bundle tar", output)

    def test_board_ready_rejects_bad_c3b_physical_smoke_result(self) -> None:
        root = self.make_root(
            old_flow=False,
            e2e_hwh=True,
            generated=True,
            c3b=True,
            c3b_bundle=True,
            c3b_bad_result=True,
        )

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] C3b AXIS/DMA physical smoke result", output)

    def test_vref_successor_physical_smoke_result_missing_warns_only(self) -> None:
        hardware = Path(self.tmp.name) / "hardware"
        results: list[dict[str, object]] = []

        preflight.check_vref_successor_physical_smoke_result(results, hardware, "final-signoff")

        self.assertEqual(results[0]["status"], "warn")
        self.assertEqual(results[0]["name"], "VREF-P0 successor physical smoke result")
        self.assertIn("promotion remains gated", results[0]["detail"])

    def test_qkv_uram_successor_smoke_artifacts_validate_bundle_session(self) -> None:
        hardware = Path(self.tmp.name) / "hardware"
        prefix = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram"
        variant = "vref-p0-softmax-input-x2-qkv-uram"
        bundle_dir = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle"
        tar_path = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz"
        bundle_dir.mkdir(parents=True)
        (hardware / "pynq" / "hgtxr").mkdir(parents=True)
        (hardware / "pynq" / "hgtxr" / f"{prefix}.bit").write_bytes(b"bit")
        (hardware / "pynq" / "hgtxr" / f"{prefix}.hwh").write_text("xilinx.com:hls:hgtxr_e2e_axis_top:1.0")

        run_script = bundle_dir / "run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh"
        run_script.write_text(
            "#!/usr/bin/env sh\n"
            "PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke "
            "--variant vref-p0-softmax-input-x2-qkv-uram --weights-mode file "
            "--weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin "
            "--expect-out-raw 58 -51 42 -28 36 -41 "
            "--json-out e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json\n"
        )
        validate_script = bundle_dir / "validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh"
        validate_script.write_text(
            "#!/usr/bin/env sh\n"
            "python3 tools/validate_pynq_smoke_result.py "
            "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json "
            "--preset axis-vref-p0-softmax-input-x2-qkv-uram "
            "--json-out e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke_validation.json\n"
        )
        for rel in [
            f"hgtxr/{prefix}.bit",
            f"hgtxr/{prefix}.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
        ]:
            path = bundle_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rel)
        file_entries = []
        for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(bundle_dir).as_posix()
            file_entries.append(
                {
                    "bundle": rel,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        manifest = {
            "name": "hgtxr_e2e_axis_dma_smoke_bundle",
            "target": "ZCU104 PYNQ",
            "variant": variant,
            "artifact_prefix": prefix,
            "command": run_script.read_text().splitlines()[-1],
            "validation_command": validate_script.read_text().splitlines()[-1],
            "expected_runtime_state": 2,
            "expected_out_raw": [58, -51, 42, -28, 36, -41],
            "files": file_entries,
        }
        (bundle_dir / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=bundle_dir.name)
        manifest["tar"] = {
            "path": str(tar_path),
            "bytes": tar_path.stat().st_size,
            "sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
        }
        (bundle_dir / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
        session = {
            "status": "pass",
            "target": "ZCU104 PYNQ",
            "variant": variant,
            "preset": "axis-vref-p0-softmax-input-x2-qkv-uram",
            "bundle_dir": str(bundle_dir),
            "bundle_root": bundle_dir.name,
            "tar": {
                "path": str(tar_path),
                "name": tar_path.name,
                "bytes": tar_path.stat().st_size,
                "sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
            },
            "expected_runtime_state": 2,
            "expected_out_raw": [58, -51, 42, -28, 36, -41],
            "result_json": "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json",
            "validation_json": "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke_validation.json",
            "canonical_result_path": str(
                hardware / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
            ),
            "bundle_validation_errors": [],
            "board_steps": [
                "tar -xzf e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz",
                "cd e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle",
                "./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
                "./validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            ],
            "host_steps": [
                "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --preset axis-vref-p0-softmax-input-x2-qkv-uram --dry-run",
                "python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff",
            ],
        }
        session_path = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_session.json"
        session_path.write_text(json.dumps(session))
        (hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_session.md").write_text(
            "# HGTXR ZCU104 Smoke Session\n"
            "./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh\n"
            "./validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh\n"
            "--preset axis-vref-p0-softmax-input-x2-qkv-uram\n"
        )
        results: list[dict[str, object]] = []

        preflight.check_qkv_uram_successor_smoke_artifacts(results, hardware, "final-signoff")

        names = {result["name"]: result["status"] for result in results}
        self.assertEqual(names["VREF-P0-02 QKV URAM PYNQ smoke bundle package"], "ok")
        self.assertEqual(names["VREF-P0-02 QKV URAM ZCU104 smoke session contract"], "ok")
        self.assertEqual(names["VREF-P0-02 QKV URAM ZCU104 smoke session markdown contract"], "ok")

    def test_qkv_uram_successor_physical_smoke_missing_warns_only(self) -> None:
        hardware = Path(self.tmp.name) / "hardware"
        results: list[dict[str, object]] = []

        preflight.check_qkv_uram_successor_physical_smoke_result(results, hardware, "final-signoff")

        self.assertEqual(results[0]["status"], "warn")
        self.assertEqual(results[0]["name"], "VREF-P0-02 QKV URAM physical smoke result")
        self.assertIn("promotion remains gated", results[0]["detail"])

    def test_qkv_uram_successor_physical_smoke_result_validates_preset(self) -> None:
        hardware = Path(self.tmp.name) / "hardware"
        result_path = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "variant": "vref-p0-softmax-input-x2-qkv-uram",
                    "bitfile": "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.bit",
                    "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.hwh",
                    "dma_in_name": "axi_dma_in",
                    "dma_out_name": "axi_dma_out",
                    "weights_mode": "file",
                    "expected_runtime_state": 2,
                    "runtime_state": 2,
                    "runtime_match": True,
                    "out_state": [3.625, -3.1875, 2.625, -1.75, 2.25, -2.5625],
                    "out_raw": [58, -51, 42, -28, 36, -41],
                    "expected_out_raw": [58, -51, 42, -28, 36, -41],
                    "output_match": True,
                }
            )
        )
        results: list[dict[str, object]] = []

        preflight.check_qkv_uram_successor_physical_smoke_result(results, hardware, "board-ready")

        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["name"], "VREF-P0-02 QKV URAM physical smoke result")
        self.assertIn("axis-vref-p0-softmax-input-x2-qkv-uram", results[0]["detail"])

    def test_smoke_candidate_discoveries_validate_legacy_and_generic_artifacts(self) -> None:
        hardware = Path(self.tmp.name) / "hardware"
        signoff = hardware / "generated" / "signoff"
        signoff.mkdir(parents=True)
        safety = {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        }
        fixtures = [
            (
                "vref_successor_smoke_candidate_discovery_2026_06_10.json",
                "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
            ),
            ("pynq_smoke_candidate_discovery_c3b_2026_06_16.json", "axis-c3b-mem16"),
            (
                "pynq_smoke_candidate_discovery_vref_p0_2026_06_16.json",
                "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
            ),
        ]
        for name, preset in fixtures:
            (signoff / name).write_text(
                json.dumps(
                    {
                        "status": "missing",
                        "preset": preset,
                        "candidate_count": 0,
                        "pass_count": 0,
                        "safety": safety,
                    }
                )
            )
        results: list[dict[str, object]] = []

        preflight.check_smoke_candidate_discoveries(results, hardware, "board-ready")

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["status"] == "ok" for result in results))
        self.assertEqual(
            {result["name"] for result in results},
            {
                "VREF-P0 successor smoke candidate discovery",
                "C3b generic smoke candidate discovery",
                "VREF-P0 generic smoke candidate discovery",
            },
        )

    def test_vref_successor_physical_smoke_result_validates_preset(self) -> None:
        hardware = Path(self.tmp.name) / "hardware"
        result_path = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "variant": "vref-p0-softmax-input-x2-dsp-mixed-stream",
                    "bitfile": "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit",
                    "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.hwh",
                    "dma_in_name": "axi_dma_in",
                    "dma_out_name": "axi_dma_out",
                    "weights_mode": "file",
                    "expected_runtime_state": 2,
                    "runtime_state": 2,
                    "runtime_match": True,
                    "out_state": [3.625, -3.1875, 2.625, -1.75, 2.25, -2.5625],
                    "out_raw": [58, -51, 42, -28, 36, -41],
                    "expected_out_raw": [58, -51, 42, -28, 36, -41],
                    "output_match": True,
                }
            )
        )
        results: list[dict[str, object]] = []

        preflight.check_vref_successor_physical_smoke_result(results, hardware, "board-ready")

        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["name"], "VREF-P0 successor physical smoke result")
        self.assertIn("axis-vref-p0-softmax-input-x2-dsp-mixed-stream", results[0]["detail"])

    def test_vref_successor_physical_smoke_result_invalid_fails_board_mode(self) -> None:
        hardware = Path(self.tmp.name) / "hardware"
        result_path = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "variant": "vref-p0-softmax-input-x2-dsp-mixed-stream",
                    "bitfile": "bit",
                    "hwhfile": "hwh",
                    "dma_in_name": "axi_dma_in",
                    "dma_out_name": "axi_dma_out",
                    "weights_mode": "file",
                    "expected_runtime_state": 2,
                    "runtime_state": 2,
                    "runtime_match": True,
                    "out_state": [0, 0, 0, 0, 0, 0],
                    "out_raw": [32, -13, 26, -6, 14, -11],
                    "expected_out_raw": [32, -13, 26, -6, 14, -11],
                    "output_match": True,
                }
            )
        )
        results: list[dict[str, object]] = []

        preflight.check_vref_successor_physical_smoke_result(results, hardware, "board-ready")

        self.assertEqual(results[0]["status"], "fail")
        self.assertEqual(results[0]["name"], "VREF-P0 successor physical smoke result")
        self.assertIn("out_raw", results[0]["detail"])

    def test_json_out_is_written_for_board_ready_failures(self) -> None:
        root = self.make_root(old_flow=True, e2e_hwh=False, generated=False)
        out = Path(self.tmp.name) / "failed" / "preflight.json"

        code, output = self.run_checker(root, "board-ready", json_out=out)

        self.assertEqual(code, 1, output)
        payload = json.loads(out.read_text())
        self.assertEqual(payload["mode"], "board-ready")
        self.assertEqual(payload["summary"], self.parse_stdout_summary(output))
        self.assertGreater(payload["summary"]["fail"], 0)
        self.assert_check_schema(self, payload["checks"])
        self.assertTrue(any(check["name"] == "board flow top mismatch" and check["status"] == "fail" for check in payload["checks"]))

    def test_board_ready_rejects_missing_weight_artifacts(self) -> None:
        root = self.make_root(old_flow=False, e2e_hwh=True, generated=True, weights=False)

        code, output = self.run_checker(root, "board-ready")

        self.assertEqual(code, 1, output)
        self.assertIn("[fail] E2E m_axi packed Q4 weight bin", output)
        self.assertIn("[fail] E2E m_axi packed Q4 weight manifest", output)


if __name__ == "__main__":
    unittest.main()
