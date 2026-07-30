#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_zcu104_c3b_smoke_remote as remote_tool  # noqa: E402
from validate_pynq_bundle_package import sha256_file  # noqa: E402


class RunZcu104C3bSmokeRemoteTests(unittest.TestCase):
    def make_inputs(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / "HGTXR"
        bundle = root / "hardware" / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
        bundle.mkdir(parents=True)
        (bundle / "BUNDLE_MANIFEST.json").write_text("{}")
        tar_path = root / "hardware" / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(bundle, arcname=bundle.name)
        sha_path = root / "docs" / "resources" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
        sha_path.parent.mkdir(parents=True)
        sha_path.write_text(f"{sha256_file(tar_path)}  {tar_path.name}\n")
        local_result = root / "hardware" / "generated" / "pynq" / "remote_result.json"
        return tmp, root, tar_path, sha_path, local_result

    def test_dry_run_builds_remote_commands_without_network(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)

        summary = remote_tool.build_summary(
            root=root,
            execute=False,
            host="192.0.2.10",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
        )

        self.assertEqual(summary["status"], "dry-run")
        self.assertEqual(summary["errors"], [])
        self.assertIn("xilinx@192.0.2.10", summary["commands"]["mkdir"])
        self.assertTrue(any("sha256sum -c" in part for part in summary["commands"]["run"]))
        self.assertTrue(any("./run_e2e_axis_dma_c3b_mem16_file_smoke.sh" in part for part in summary["commands"]["run"]))

    def test_dry_run_builds_vref_successor_remote_commands(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)

        summary = remote_tool.build_summary(
            root=root,
            execute=False,
            host="192.0.2.10",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr_vref",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
            profile="vref-p0-softmax-input-x2-dsp-mixed-stream",
            bundle_root="e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle",
            result_json="e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json",
            validation_json="e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke_validation.json",
            run_script="./run_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
            validate_script="./validate_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
            preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
        )

        self.assertEqual(summary["status"], "dry-run")
        self.assertEqual(summary["profile"], "vref-p0-softmax-input-x2-dsp-mixed-stream")
        self.assertEqual(summary["preset"], "axis-vref-p0-softmax-input-x2-dsp-mixed-stream")
        remote_run = summary["commands"]["run"][-1]
        self.assertIn("e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle", remote_run)
        self.assertIn("./run_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh", remote_run)
        self.assertIn("e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json", summary["commands"]["fetch_result"][1])

    def test_dry_run_builds_qkv_uram_remote_commands(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)

        summary = remote_tool.build_summary(
            root=root,
            execute=False,
            host="192.0.2.10",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr_qkv",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
            profile="vref-p0-softmax-input-x2-qkv-uram",
            bundle_root="e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle",
            result_json="e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json",
            validation_json="e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke_validation.json",
            run_script="./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            validate_script="./validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            preset="axis-vref-p0-softmax-input-x2-qkv-uram",
        )

        self.assertEqual(summary["status"], "dry-run")
        self.assertEqual(summary["profile"], "vref-p0-softmax-input-x2-qkv-uram")
        self.assertEqual(summary["preset"], "axis-vref-p0-softmax-input-x2-qkv-uram")
        remote_run = summary["commands"]["run"][-1]
        self.assertIn("e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle", remote_run)
        self.assertIn("./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh", remote_run)
        self.assertIn("e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json", summary["commands"]["fetch_result"][1])

    def test_dry_run_builds_runtime_mode_search_remote_commands(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)

        summary = remote_tool.build_summary(
            root=root,
            execute=False,
            host="192.0.2.10",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr_runtime_search",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
            profile="runtime-mode-par32-search",
            bundle_root="e2e_axis_dma_par32_runtime_mode_search_smoke_bundle",
            result_json="e2e_axis_dma_runtime_mode_par32_search_file_smoke.json",
            validation_json="e2e_axis_dma_runtime_mode_par32_search_file_smoke_validation.json",
            run_script="./run_e2e_axis_dma_runtime_mode_par32_search_file_smoke.sh",
            validate_script="./validate_e2e_axis_dma_runtime_mode_par32_search_file_smoke.sh",
            preset="axis-runtime-mode-par32-search",
        )

        self.assertEqual(summary["status"], "dry-run")
        self.assertEqual(summary["profile"], "runtime-mode-par32-search")
        self.assertEqual(summary["preset"], "axis-runtime-mode-par32-search")
        remote_run = summary["commands"]["run"][-1]
        self.assertIn("e2e_axis_dma_par32_runtime_mode_search_smoke_bundle", remote_run)
        self.assertIn("./run_e2e_axis_dma_runtime_mode_par32_search_file_smoke.sh", remote_run)
        self.assertIn("e2e_axis_dma_runtime_mode_par32_search_file_smoke.json", summary["commands"]["fetch_result"][1])

    def test_dry_run_builds_runtime_mode_track_remote_commands(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)

        summary = remote_tool.build_summary(
            root=root,
            execute=False,
            host="192.0.2.10",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr_runtime_track",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
            profile="runtime-mode-par32-track",
            bundle_root="e2e_axis_dma_par32_runtime_mode_track_smoke_bundle",
            result_json="e2e_axis_dma_runtime_mode_par32_track_file_smoke.json",
            validation_json="e2e_axis_dma_runtime_mode_par32_track_file_smoke_validation.json",
            run_script="./run_e2e_axis_dma_runtime_mode_par32_track_file_smoke.sh",
            validate_script="./validate_e2e_axis_dma_runtime_mode_par32_track_file_smoke.sh",
            preset="axis-runtime-mode-par32-track",
        )

        self.assertEqual(summary["status"], "dry-run")
        self.assertEqual(summary["profile"], "runtime-mode-par32-track")
        self.assertEqual(summary["preset"], "axis-runtime-mode-par32-track")
        remote_run = summary["commands"]["run"][-1]
        self.assertIn("e2e_axis_dma_par32_runtime_mode_track_smoke_bundle", remote_run)
        self.assertIn("./run_e2e_axis_dma_runtime_mode_par32_track_file_smoke.sh", remote_run)
        self.assertIn("e2e_axis_dma_runtime_mode_par32_track_file_smoke.json", summary["commands"]["fetch_result"][1])

    def test_dry_run_builds_par32_prefetchall4_hybrid_remote_commands(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)

        summary = remote_tool.build_summary(
            root=root,
            execute=False,
            host="192.0.2.10",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr_prefetchall4_hybrid",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
            profile="par32-prefetchall4-300-hybrid-10-90",
            bundle_root="e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_smoke_bundle",
            result_json="e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json",
            validation_json="e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke_validation.json",
            run_script="./run_e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.sh",
            validate_script="./validate_e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.sh",
            preset="axis-par32-prefetchall4-300-hybrid-10-90",
        )

        self.assertEqual(summary["status"], "dry-run")
        self.assertEqual(summary["profile"], "par32-prefetchall4-300-hybrid-10-90")
        self.assertEqual(summary["preset"], "axis-par32-prefetchall4-300-hybrid-10-90")
        remote_run = summary["commands"]["run"][-1]
        self.assertIn("e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_smoke_bundle", remote_run)
        self.assertIn("./run_e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.sh", remote_run)
        self.assertIn(
            "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json",
            summary["commands"]["fetch_result"][1],
        )

    def test_blocks_on_bad_sha_file(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)
        sha_path.write_text("bad  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz\n")

        summary = remote_tool.build_summary(
            root=root,
            execute=False,
            host="zcu104.local",
            user=None,
            remote_dir="~/hgtxr",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
        )

        self.assertEqual(summary["status"], "blocked-local-inputs")
        self.assertTrue(any("sha256 line mismatch" in error for error in summary["errors"]))

    def test_tilde_remote_dir_uses_home_in_remote_shell_command(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)

        summary = remote_tool.build_summary(
            root=root,
            execute=False,
            host="zcu104.local",
            user="xilinx",
            remote_dir="~/hgtxr",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
        )

        remote_run = summary["commands"]["run"][-1]
        self.assertIn("cd $HOME/hgtxr", remote_run)
        self.assertNotIn("'~/hgtxr'", remote_run)

    def test_execute_imports_valid_result_with_fake_runner(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)
        calls: list[list[str]] = []

        def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            if cmd[0] == "scp" and cmd[-1] == str(local_result):
                local_result.parent.mkdir(parents=True, exist_ok=True)
                local_result.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "variant": "c3b-mem16",
                            "weights_mode": "file",
                            "expected_runtime_state": 2,
                            "runtime_state": 2,
                            "runtime_match": True,
                            "expected_out_raw": [32, -13, 26, -6, 14, -11],
                            "out_raw": [32, -13, 26, -6, 14, -11],
                            "out_state": [32, -13, 26, -6, 14, -11],
                            "output_match": True,
                            "dma_in_name": "axi_dma_0",
                            "dma_out_name": "axi_dma_1",
                            "bitfile": "hgtxr_e2e_axis_dma_c3b_mem16.bit",
                            "hwhfile": "hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                        }
                    )
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        summary = remote_tool.build_summary(
            root=root,
            execute=True,
            host="zcu104.local",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
            runner=fake_runner,
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(len(calls), 5)
        canonical = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
        self.assertTrue(canonical.exists())
        self.assertEqual(summary["import_result"]["status"], "pass")

    def test_execute_imports_valid_vref_successor_result_with_fake_runner(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)
        calls: list[list[str]] = []

        def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            if cmd[0] == "scp" and cmd[-1] == str(local_result):
                local_result.parent.mkdir(parents=True, exist_ok=True)
                local_result.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "variant": "vref-p0-softmax-input-x2-dsp-mixed-stream",
                            "weights_mode": "file",
                            "expected_runtime_state": 2,
                            "runtime_state": 2,
                            "runtime_match": True,
                            "expected_out_raw": [58, -51, 42, -28, 36, -41],
                            "out_raw": [58, -51, 42, -28, 36, -41],
                            "out_state": [3.625, -3.1875, 2.625, -1.75, 2.25, -2.5625],
                            "output_match": True,
                            "dma_in_name": "axi_dma_0",
                            "dma_out_name": "axi_dma_1",
                            "bitfile": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit",
                            "hwhfile": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.hwh",
                        }
                    )
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        summary = remote_tool.build_summary(
            root=root,
            execute=True,
            host="zcu104.local",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr_vref",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
            profile="vref-p0-softmax-input-x2-dsp-mixed-stream",
            bundle_root="e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle",
            result_json="e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json",
            validation_json="e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke_validation.json",
            run_script="./run_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
            validate_script="./validate_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
            preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
            runner=fake_runner,
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(len(calls), 5)
        canonical = (
            root
            / "hardware"
            / "pynq"
            / "hgtxr"
            / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
        )
        self.assertTrue(canonical.exists())
        self.assertEqual(summary["import_result"]["status"], "pass")

    def test_execute_imports_valid_qkv_uram_result_with_fake_runner(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)
        calls: list[list[str]] = []

        def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            if cmd[0] == "scp" and cmd[-1] == str(local_result):
                local_result.parent.mkdir(parents=True, exist_ok=True)
                local_result.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "variant": "vref-p0-softmax-input-x2-qkv-uram",
                            "weights_mode": "file",
                            "expected_runtime_state": 2,
                            "runtime_state": 2,
                            "runtime_match": True,
                            "expected_out_raw": [58, -51, 42, -28, 36, -41],
                            "out_raw": [58, -51, 42, -28, 36, -41],
                            "out_state": [3.625, -3.1875, 2.625, -1.75, 2.25, -2.5625],
                            "output_match": True,
                            "dma_in_name": "axi_dma_0",
                            "dma_out_name": "axi_dma_1",
                            "bitfile": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.bit",
                            "hwhfile": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.hwh",
                        }
                    )
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        summary = remote_tool.build_summary(
            root=root,
            execute=True,
            host="zcu104.local",
            user="xilinx",
            remote_dir="/home/xilinx/hgtxr_qkv",
            tar_path=tar_path,
            sha256_path=sha_path,
            local_result=local_result,
            identity_file=None,
            port=None,
            profile="vref-p0-softmax-input-x2-qkv-uram",
            bundle_root="e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle",
            result_json="e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json",
            validation_json="e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke_validation.json",
            run_script="./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            validate_script="./validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            preset="axis-vref-p0-softmax-input-x2-qkv-uram",
            runner=fake_runner,
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(len(calls), 5)
        canonical = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
        self.assertTrue(canonical.exists())
        self.assertEqual(summary["import_result"]["status"], "pass")

    def test_cli_writes_dry_run_json_and_markdown(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)
        json_out = root / "hardware" / "generated" / "remote.json"
        markdown_out = root / "hardware" / "generated" / "remote.md"

        code = remote_tool.main(
            [
                "--root",
                str(root),
                "--host",
                "zcu104.local",
                "--tar",
                str(tar_path),
                "--sha256-file",
                str(sha_path),
                "--local-result",
                str(local_result),
                "--json-out",
                str(json_out),
                "--markdown-out",
                str(markdown_out),
            ]
        )

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(json_out.read_text())["status"], "dry-run")
        self.assertIn("HGTXR ZCU104 Remote Smoke Runner", markdown_out.read_text())

    def test_cli_profile_vref_writes_dry_run_json_and_markdown(self) -> None:
        tmp, root, tar_path, sha_path, local_result = self.make_inputs()
        self.addCleanup(tmp.cleanup)
        json_out = root / "hardware" / "generated" / "remote_vref.json"
        markdown_out = root / "hardware" / "generated" / "remote_vref.md"

        code = remote_tool.main(
            [
                "--profile",
                "vref-p0-softmax-input-x2-dsp-mixed-stream",
                "--root",
                str(root),
                "--host",
                "zcu104.local",
                "--tar",
                str(tar_path),
                "--sha256-file",
                str(sha_path),
                "--local-result",
                str(local_result),
                "--json-out",
                str(json_out),
                "--markdown-out",
                str(markdown_out),
            ]
        )

        payload = json.loads(json_out.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")
        self.assertEqual(payload["profile"], "vref-p0-softmax-input-x2-dsp-mixed-stream")
        self.assertIn("axis-vref-p0-softmax-input-x2-dsp-mixed-stream", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
