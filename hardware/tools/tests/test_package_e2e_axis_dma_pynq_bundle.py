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

import package_e2e_axis_dma_pynq_bundle as bundler  # noqa: E402


class E2EAxisDmaPynqBundleTests(unittest.TestCase):
    def make_fake_hardware_root(self, tmp: Path, prefix: str) -> Path:
        root = tmp / "hardware"
        for rel in bundler.PACKAGE_FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n")
        for rel in bundler.WEIGHT_FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00\x00\x00\x00")
        for rel in bundler.TOOL_FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n")
        for suffix in [".bit", ".hwh"]:
            path = root / "pynq" / "hgtxr" / f"{prefix}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{suffix}\n")
        return root

    def test_build_c3b_bundle_writes_expected_files_manifest_and_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            result = bundler.build_bundle(out_dir, tar_path, "c3b-mem16")
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(result["out_dir"], str(out_dir))
            self.assertEqual(result["manifest"], str(out_dir / "BUNDLE_MANIFEST.json"))
            self.assertTrue(tar_path.exists())
            self.assertEqual(manifest["variant"], "c3b-mem16")
            self.assertEqual(manifest["artifact_prefix"], "hgtxr_e2e_axis_dma_c3b_mem16")
            self.assertEqual(manifest["expected_runtime_state"], 2)
            self.assertEqual(manifest["expected_out_raw"], [32, -13, 26, -6, 14, -11])
            self.assertIn("--variant c3b-mem16", manifest["command"])
            self.assertIn("--weights-mode file", manifest["command"])
            self.assertIn("tools/validate_pynq_smoke_result.py", manifest["validation_command"])
            self.assertIn("--preset axis-c3b-mem16", manifest["validation_command"])

            bundle_files = {entry["bundle"] for entry in manifest["files"]}
            required = {
                "hgtxr/__init__.py",
                "hgtxr/hgtxr_overlay.py",
                "hgtxr/e2e_axis_dma_overlay.py",
                "hgtxr/e2e_m_axi_overlay.py",
                "hgtxr/e2e_m_axi_weights.py",
                "hgtxr/run_e2e_axis_dma_smoke.py",
                "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
                "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
                "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
                "tools/validate_pynq_smoke_result.py",
                "run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                "validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
            }
            self.assertTrue(required.issubset(bundle_files))
            self.assertFalse(any("__pycache__" in path for path in bundle_files))
            self.assertFalse(any(path in {"hgtxr/hgtxr.bit", "hgtxr/hgtxr.hwh"} for path in bundle_files))

            readme = (out_dir / "hgtxr" / "README.md").read_text()
            self.assertIn("--variant c3b-mem16", readme)
            self.assertIn("weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin", readme)
            self.assertIn("--preset axis-c3b-mem16", readme)

    def test_build_vref_successor_bundle_uses_successor_artifacts_and_expected_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "vref-p0-softmax-input-x2-dsp-mixed-stream")
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(manifest["variant"], "vref-p0-softmax-input-x2-dsp-mixed-stream")
            self.assertEqual(
                manifest["artifact_prefix"],
                "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream",
            )
            self.assertEqual(manifest["expected_out_raw"], [58, -51, 42, -28, 36, -41])
            self.assertIn("--variant vref-p0-softmax-input-x2-dsp-mixed-stream", manifest["command"])
            self.assertIn("--preset axis-vref-p0-softmax-input-x2-dsp-mixed-stream", manifest["validation_command"])

            bundle_files = {entry["bundle"] for entry in manifest["files"]}
            required = {
                "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit",
                "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.hwh",
                "run_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
                "validate_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
            }
            self.assertTrue(required.issubset(bundle_files))

    def test_build_qkv_uram_bundle_uses_qkv_artifacts_and_expected_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram"
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"

            bundler.build_bundle(out_dir, tar_path, "vref-p0-softmax-input-x2-qkv-uram")
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(manifest["variant"], "vref-p0-softmax-input-x2-qkv-uram")
            self.assertEqual(manifest["artifact_prefix"], prefix)
            self.assertEqual(manifest["expected_out_raw"], [58, -51, 42, -28, 36, -41])
            self.assertIn("--variant vref-p0-softmax-input-x2-qkv-uram", manifest["command"])
            self.assertIn("--preset axis-vref-p0-softmax-input-x2-qkv-uram", manifest["validation_command"])
            bundle_files = {entry["bundle"] for entry in manifest["files"]}
            self.assertIn(f"hgtxr/{prefix}.bit", bundle_files)
            self.assertIn(f"hgtxr/{prefix}.hwh", bundle_files)
            self.assertIn("run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh", bundle_files)
            self.assertIn("validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh", bundle_files)

    def test_build_runtime_mode_search_bundle_uses_runtime_artifacts_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16"
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"

            bundler.build_bundle(out_dir, tar_path, "runtime-mode-par32-search")
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(manifest["variant"], "runtime-mode-par32-search")
            self.assertEqual(manifest["artifact_prefix"], prefix)
            self.assertEqual(manifest["mode_profile"], "search")
            self.assertEqual(manifest["expected_runtime_state"], 0)
            self.assertIsNone(manifest["expected_out_raw"])
            self.assertIn("--variant runtime-mode-par32", manifest["command"])
            self.assertIn("--mode-profile search", manifest["command"])
            self.assertIn("--preset axis-runtime-mode-par32-search", manifest["validation_command"])

    def test_build_runtime_mode_track_bundle_uses_runtime_artifacts_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16"
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"

            bundler.build_bundle(out_dir, tar_path, "runtime-mode-par32-track")
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(manifest["variant"], "runtime-mode-par32-track")
            self.assertEqual(manifest["artifact_prefix"], prefix)
            self.assertEqual(manifest["mode_profile"], "track")
            self.assertEqual(manifest["expected_runtime_state"], 1)
            self.assertIsNone(manifest["expected_out_raw"])
            self.assertIn("--variant runtime-mode-par32", manifest["command"])
            self.assertIn("--mode-profile track", manifest["command"])
            self.assertIn("--preset axis-runtime-mode-par32-track", manifest["validation_command"])

    def test_build_par32_rom_compute_search_bundle_checks_expected_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = (
                "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
                "corefabric_normuram_compute_300_mem16"
            )
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"

            bundler.build_bundle(out_dir, tar_path, "par32-rom-compute-300-search")
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(manifest["variant"], "par32-rom-compute-300-search")
            self.assertEqual(manifest["artifact_prefix"], prefix)
            self.assertEqual(manifest["mode_profile"], "search")
            self.assertEqual(manifest["expected_runtime_state"], 0)
            self.assertEqual(manifest["expected_out_raw"], [-1169, -1169, -1169, -1169, -1169, -1133])
            self.assertIn("--variant par32-rom-compute-300", manifest["command"])
            self.assertIn("--mode-profile search", manifest["command"])
            self.assertIn("--expect-out-raw -1169 -1169 -1169 -1169 -1169 -1133", manifest["command"])
            self.assertIn("--preset axis-par32-rom-compute-300-search", manifest["validation_command"])

    def test_build_par32_rom_compute_track_bundle_checks_expected_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = (
                "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
                "corefabric_normuram_compute_300_mem16"
            )
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"

            bundler.build_bundle(out_dir, tar_path, "par32-rom-compute-300-track")
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(manifest["variant"], "par32-rom-compute-300-track")
            self.assertEqual(manifest["artifact_prefix"], prefix)
            self.assertEqual(manifest["mode_profile"], "track")
            self.assertEqual(manifest["expected_runtime_state"], 1)
            self.assertEqual(manifest["expected_out_raw"], [-235, -235, -235, -235, -235, -226])
            self.assertIn("--variant par32-rom-compute-300", manifest["command"])
            self.assertIn("--mode-profile track", manifest["command"])
            self.assertIn("--expect-out-raw -235 -235 -235 -235 -235 -226", manifest["command"])
            self.assertIn("--preset axis-par32-rom-compute-300-track", manifest["validation_command"])

    def test_build_par32_prefetchall4_hybrid_bundle_uses_interleaved_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = (
                "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
                "corefabric_normuram_compute_prefetchall4_300_mem16"
            )
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"

            bundler.build_bundle(out_dir, tar_path, "par32-prefetchall4-300-hybrid-10-90")
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(manifest["variant"], "par32-prefetchall4-300-hybrid-10-90")
            self.assertEqual(manifest["artifact_prefix"], prefix)
            self.assertIsNone(manifest["expected_runtime_state"])
            self.assertEqual(
                manifest["expected_out_raw"]["search"],
                [-1169, -1169, -1169, -1169, -1169, -1125],
            )
            self.assertEqual(
                manifest["expected_out_raw"]["track"],
                [-235, -235, -235, -235, -235, -239],
            )
            self.assertIn("hgtxr.run_e2e_axis_dma_hybrid_smoke", manifest["command"])
            self.assertIn("--variant par32-prefetchall4-300", manifest["command"])
            self.assertIn("--repeat-cycles 5", manifest["command"])
            self.assertIn("--preset axis-par32-prefetchall4-300-hybrid-10-90", manifest["validation_command"])
            bundle_files = {entry["bundle"] for entry in manifest["files"]}
            self.assertIn("hgtxr/run_e2e_axis_dma_hybrid_smoke.py", bundle_files)
            self.assertIn("run_e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.sh", bundle_files)


if __name__ == "__main__":
    unittest.main()
