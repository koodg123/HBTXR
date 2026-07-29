from __future__ import annotations

import json
import socket
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

import run_runtime_mode_board_latency as runner  # noqa: E402
from validate_pynq_bundle_package import sha256_file  # noqa: E402


def resolved(_host: str, _port: int | None) -> list[tuple]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.10", 0))]


def payload(profile: str, latency_ms: float) -> dict:
    runtime_state = 0 if profile == "search" else 1
    return {
        "status": "pass",
        "variant": "runtime-mode-par32",
        "weights_mode": "zero",
        "mode_profile": profile,
        "expected_runtime_state": runtime_state,
        "runtime_state": runtime_state,
        "runtime_match": True,
        "expected_out_raw": None,
        "out_raw": [0, 0, 0, 0, 0, 0],
        "out_state": [0, 0, 0, 0, 0, 0],
        "output_match": True,
        "dma_in_name": "axi_dma_in",
        "dma_out_name": "axi_dma_out",
        "bitfile": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.bit",
        "hwhfile": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.hwh",
        "repeat": 1,
        "latency_ms_samples": [latency_ms],
        "latency_ms_summary": {"count": 1, "max": latency_ms},
        "accelerator_latency_ms_samples": [latency_ms],
        "accelerator_latency_ms_summary": {"count": 1, "max": latency_ms},
    }


def par32_rom_payload(profile: str, latency_ms: float) -> dict:
    runtime_state = 0 if profile == "search" else 1
    expected = [-1169, -1169, -1169, -1169, -1169, -1133] if profile == "search" else [-235, -235, -235, -235, -235, -226]
    return {
        "status": "pass",
        "variant": "par32-rom-compute-300",
        "weights_mode": "zero",
        "mode_profile": profile,
        "expected_runtime_state": runtime_state,
        "runtime_state": runtime_state,
        "runtime_match": True,
        "expected_out_raw": expected,
        "out_raw": expected,
        "out_state": [float(value) / 16.0 for value in expected],
        "output_match": True,
        "dma_in_name": "axi_dma_in",
        "dma_out_name": "axi_dma_out",
        "bitfile": "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.bit",
        "hwhfile": "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.hwh",
        "repeat": 1,
        "latency_ms_samples": [latency_ms],
        "latency_ms_summary": {"count": 1, "max": latency_ms},
        "accelerator_latency_ms_samples": [latency_ms],
        "accelerator_latency_ms_summary": {"count": 1, "max": latency_ms},
    }


class RuntimeModeBoardLatencyRunnerTests(unittest.TestCase):
    def make_profile_configs(self, root: Path) -> dict:
        configs = {}
        for mode in ["search", "track"]:
            profile = f"runtime-mode-par32-{mode}"
            bundle_root = f"e2e_axis_dma_par32_runtime_mode_{mode}_smoke_bundle"
            bundle = root / "hardware" / "generated" / "pynq" / bundle_root
            bundle.mkdir(parents=True)
            (bundle / "BUNDLE_MANIFEST.json").write_text("{}")
            tar_path = root / "hardware" / "generated" / "pynq" / f"{bundle_root}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(bundle, arcname=bundle.name)
            sha_path = tar_path.with_name(tar_path.name + ".sha256")
            sha_path.write_text(f"{sha256_file(tar_path)}  {tar_path.name}\n")
            configs[profile] = {
                "tar": tar_path,
                "sha256": sha_path,
                "local_result": root
                / "hardware"
                / "generated"
                / "pynq"
                / f"e2e_axis_dma_runtime_mode_par32_{mode}_file_smoke.remote.json",
                "remote_dir": f"/home/xilinx/hgtxr_runtime_{mode}",
                "result_json": f"e2e_axis_dma_runtime_mode_par32_{mode}_file_smoke.json",
                "validation_json": f"e2e_axis_dma_runtime_mode_par32_{mode}_file_smoke_validation.json",
                "bundle_root": bundle_root,
                "run_script": f"./run_e2e_axis_dma_runtime_mode_par32_{mode}_file_smoke.sh",
                "validate_script": f"./validate_e2e_axis_dma_runtime_mode_par32_{mode}_file_smoke.sh",
                "preset": f"axis-runtime-mode-par32-{mode}",
            }
        return configs

    def make_par32_rom_compute_configs(self, root: Path) -> dict:
        configs = {}
        for mode in ["search", "track"]:
            profile = f"par32-rom-compute-300-{mode}"
            bundle_root = f"e2e_axis_dma_par32_rom_compute_300_{mode}_smoke_bundle"
            bundle = root / "hardware" / "generated" / "pynq" / bundle_root
            bundle.mkdir(parents=True)
            (bundle / "BUNDLE_MANIFEST.json").write_text("{}")
            tar_path = root / "hardware" / "generated" / "pynq" / f"{bundle_root}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(bundle, arcname=bundle.name)
            sha_path = tar_path.with_name(tar_path.name + ".sha256")
            sha_path.write_text(f"{sha256_file(tar_path)}  {tar_path.name}\n")
            configs[profile] = {
                "tar": tar_path,
                "sha256": sha_path,
                "local_result": root
                / "hardware"
                / "generated"
                / "pynq"
                / f"e2e_axis_dma_par32_rom_compute_300_{mode}_file_smoke.remote.json",
                "remote_dir": f"/home/xilinx/hgtxr_par32_rom_compute_300_{mode}",
                "result_json": f"e2e_axis_dma_par32_rom_compute_300_{mode}_file_smoke.json",
                "validation_json": f"e2e_axis_dma_par32_rom_compute_300_{mode}_file_smoke_validation.json",
                "bundle_root": bundle_root,
                "run_script": f"./run_e2e_axis_dma_par32_rom_compute_300_{mode}_file_smoke.sh",
                "validate_script": f"./validate_e2e_axis_dma_par32_rom_compute_300_{mode}_file_smoke.sh",
                "preset": f"axis-par32-rom-compute-300-{mode}",
            }
        return configs

    def test_dry_run_builds_both_runtime_mode_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            configs = self.make_profile_configs(root)
            summary = runner.build_runtime_mode_summary(
                root=root,
                execute=False,
                host="192.0.2.10",
                user="xilinx",
                identity_file=None,
                port=None,
                profile_configs=configs,
                resolver=resolved,
            )
        self.assertEqual(summary["status"], "dry-run")
        self.assertEqual(summary["remote_statuses"]["runtime-mode-par32-search"], "dry-run")
        self.assertEqual(summary["remote_statuses"]["runtime-mode-par32-track"], "dry-run")
        self.assertEqual(summary["board_latency_gate"]["status"], "missing")

    def test_dry_run_builds_par32_rom_compute_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            configs = self.make_par32_rom_compute_configs(root)
            summary = runner.build_runtime_mode_summary(
                root=root,
                execute=False,
                host="192.0.2.10",
                user="xilinx",
                identity_file=None,
                port=None,
                profile_configs=configs,
                profile_names=runner.PAR32_ROM_COMPUTE_PROFILES,
                profile_set="par32-rom-compute-300",
                resolver=resolved,
            )
        self.assertEqual(summary["status"], "dry-run")
        self.assertEqual(summary["profile_set"], "par32-rom-compute-300")
        self.assertEqual(summary["remote_statuses"]["par32-rom-compute-300-search"], "dry-run")
        self.assertEqual(summary["remote_statuses"]["par32-rom-compute-300-track"], "dry-run")
        self.assertEqual(summary["board_latency_gate"]["status"], "missing")

    def test_execute_imports_both_results_and_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            configs = self.make_profile_configs(root)

            def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
                if cmd[0] == "scp":
                    dst = Path(cmd[-1])
                    if dst.name.endswith(".remote.json"):
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        mode = "search" if "search" in dst.name else "track"
                        dst.write_text(json.dumps(payload(mode, 3.1 if mode == "search" else 0.7)))
                return subprocess.CompletedProcess(cmd, 0, "", "")

            summary = runner.build_runtime_mode_summary(
                root=root,
                execute=True,
                host="192.0.2.10",
                user="xilinx",
                identity_file=None,
                port=None,
                runner=fake_runner,
                profile_configs=configs,
                resolver=resolved,
            )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["board_latency_gate"]["status"], "pass")
        self.assertEqual(summary["board_latency_gate"]["summary"]["cases_pass"], 2)

    def test_execute_imports_par32_rom_compute_results_and_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            configs = self.make_par32_rom_compute_configs(root)

            def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
                if cmd[0] == "scp":
                    dst = Path(cmd[-1])
                    if dst.name.endswith(".remote.json"):
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        mode = "search" if "search" in dst.name else "track"
                        dst.write_text(json.dumps(par32_rom_payload(mode, 3.1 if mode == "search" else 0.7)))
                return subprocess.CompletedProcess(cmd, 0, "", "")

            summary = runner.build_runtime_mode_summary(
                root=root,
                execute=True,
                host="192.0.2.10",
                user="xilinx",
                identity_file=None,
                port=None,
                runner=fake_runner,
                profile_configs=configs,
                profile_names=runner.PAR32_ROM_COMPUTE_PROFILES,
                profile_set="par32-rom-compute-300",
                resolver=resolved,
            )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["board_latency_gate"]["profile_set"], "par32-rom-compute-300")
        self.assertEqual(summary["board_latency_gate"]["status"], "pass")

    def test_execute_stops_before_remote_when_host_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            configs = self.make_profile_configs(root)

            def unresolved(_host: str, _port: int | None) -> list[tuple]:
                raise socket.gaierror("Name or service not known")

            def fail_if_called(cmd: list[str]) -> subprocess.CompletedProcess[str]:
                raise AssertionError(f"unexpected remote command: {cmd}")

            summary = runner.build_runtime_mode_summary(
                root=root,
                execute=True,
                host="zcu104.local",
                user="xilinx",
                identity_file=None,
                port=None,
                runner=fail_if_called,
                profile_configs=configs,
                resolver=unresolved,
            )
        self.assertEqual(summary["status"], "host-unresolved")
        self.assertEqual(summary["host_preflight"]["status"], "fail")
        self.assertIn("host unresolved: zcu104.local", summary["errors"][0])
        self.assertEqual(summary["remote_statuses"]["runtime-mode-par32-search"], "dry-run")
        self.assertEqual(summary["remote_statuses"]["runtime-mode-par32-track"], "dry-run")


if __name__ == "__main__":
    unittest.main()
