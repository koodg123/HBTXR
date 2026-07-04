from pathlib import Path
import tempfile
import unittest

import numpy as np

from hgpipe_quantization import HgPipeQuantizationPackage, QuantParamStore, TorchIntGraphRunner, evaluate_artifact_image_batch_npy, evaluate_artifact_patch_batch_npy, evaluate_artifact_patch_matrix_manifest, validate_artifact_patch_matrix_manifest
from hgpipe_quantization.graph import ArtifactGraphRunner


class PackageApiTest(unittest.TestCase):
    def setUp(self):
        self.source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"

    def test_top_level_exports_core_package_api(self):
        package = HgPipeQuantizationPackage(self.source_path)

        self.assertIsInstance(package.quant_params, QuantParamStore)
        self.assertEqual(len(package.cases()), 97)
        self.assertIs(TorchIntGraphRunner, __import__("hgpipe_quantization").TorchIntGraphRunner)

    def test_contracts_expose_lut_scalars_tables_and_zero_point_policy(self):
        package = HgPipeQuantizationPackage(self.source_path)

        contracts = package.contracts(kind="requant_table")
        first = next(contract for contract in contracts if contract.name == "attn_0_qq")

        self.assertEqual(len(contracts), 48)
        self.assertIsNotNone(first.params)
        self.assertEqual(len(first.params.scalars), 3)
        self.assertEqual(len(first.params.tables), 1)
        self.assertEqual(first.params.offset, 88)
        self.assertEqual(first.params.shift_scale, 2)
        self.assertEqual(first.params.effective_divisor, 4)
        self.assertEqual(first.params.bound, 63)
        self.assertIsNone(first.params.zero_point)
        self.assertIsNotNone(first.output_dtype)
        self.assertEqual(first.output_dtype.bits, 3)

    def test_export_contracts_returns_json_serializable_contracts(self):
        package = HgPipeQuantizationPackage(self.source_path)

        exported = package.export_contracts(kind="gelu_requant_table")
        first = exported[0]

        self.assertEqual(len(exported), 12)
        self.assertIn("params", first)
        self.assertIn("scalars", first["params"])
        self.assertIn("table_sizes", first["params"])
        self.assertIn("shift_scale", first["params"])
        self.assertIn("effective_divisor", first["params"])
        self.assertIsNone(first["params"]["zero_point"])

    def test_package_compares_fakequant_graph_and_torch_int_outputs(self):
        package = HgPipeQuantizationPackage(self.source_path)

        result = package.compare_graph_runners(topk=3)

        self.assertTrue(result.comparison.passed)
        self.assertEqual(result.comparison.mismatches, 0)
        self.assertTrue(result.comparison.top1_equal)
        self.assertEqual(result.torch_int.numel, 1000)
        self.assertEqual(result.fakequant_graph.numel, 1000)
        self.assertEqual(len(result.torch_int.topk), 3)


    def test_top_level_exports_artifact_patch_matrix_api(self):
        package = HgPipeQuantizationPackage(self.source_path)

        self.assertTrue(hasattr(package, "evaluate_artifact_patch_matrix_manifest"))
        self.assertTrue(hasattr(package, "validate_artifact_patch_matrix_manifest"))
        self.assertIs(evaluate_artifact_patch_matrix_manifest, __import__("hgpipe_quantization").evaluate_artifact_patch_matrix_manifest)
        self.assertIs(validate_artifact_patch_matrix_manifest, __import__("hgpipe_quantization").validate_artifact_patch_matrix_manifest)


    def test_package_evaluates_artifact_patch_batch_npy(self):
        package = HgPipeQuantizationPackage(self.source_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_inputs = Path(tmpdir) / "patch_inputs.npy"
            labels = Path(tmpdir) / "labels.npy"
            patch = ArtifactGraphRunner(self.source_path).array("patch_embed_matmul_input.txt")
            np.save(patch_inputs, patch.reshape(1, -1))
            np.save(labels, np.asarray([0], dtype=np.int64))

            payload = package.evaluate_artifact_patch_batch_npy(
                patch_inputs_npy=patch_inputs,
                labels_npy=labels,
                model="deit_tiny_patch16_224",
                precision="int8",
                paper_equivalent=True,
                topk=2,
            )

            self.assertEqual(payload["samples"], 1)
            self.assertEqual(payload["evaluation_mode"], "hgpipe_artifact_graph")
            self.assertEqual(payload["quantization_flow"], "torch_int")
            self.assertTrue(payload["paper_equivalent"])
            self.assertTrue(payload["runner_comparison_passed"])
            self.assertIs(evaluate_artifact_patch_batch_npy, __import__("hgpipe_quantization").evaluate_artifact_patch_batch_npy)


    def test_package_evaluates_artifact_image_batch_npy(self):
        package = HgPipeQuantizationPackage(self.source_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            images = Path(tmpdir) / "images.npy"
            labels = Path(tmpdir) / "labels.npy"
            np.save(images, np.zeros((1, 3, 224, 224), dtype=np.float32))
            np.save(labels, np.asarray([0], dtype=np.int64))

            payload = package.evaluate_artifact_image_batch_npy(images_npy=images, labels_npy=labels, scale=1.0, topk=2)

            self.assertEqual(payload["samples"], 1)
            self.assertEqual(payload["evaluation_mode"], "hgpipe_artifact_graph_experimental")
            self.assertFalse(payload["paper_equivalent"])
            self.assertTrue(payload["runner_comparison_passed"])
            self.assertIs(evaluate_artifact_image_batch_npy, __import__("hgpipe_quantization").evaluate_artifact_image_batch_npy)


if __name__ == "__main__":
    unittest.main()

class ArtifactPatchMatrixIngestPackageApiAppendTest(unittest.TestCase):
    def test_top_level_exports_artifact_patch_matrix_ingest_api(self):
        import hgpipe_quantization as package_module

        package = HgPipeQuantizationPackage(self.source_path)
        self.assertTrue(hasattr(package, "load_artifact_patch_asset_source_manifest"))
        self.assertTrue(hasattr(package, "ingest_artifact_patch_matrix_assets"))
        self.assertTrue(hasattr(package_module, "load_artifact_patch_asset_source_manifest"))
        self.assertTrue(hasattr(package_module, "ingest_artifact_patch_matrix_assets"))


def _override_artifact_patch_matrix_ingest_package_api_test(self):
    import hgpipe_quantization as package_module

    source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
    package = HgPipeQuantizationPackage(source_path)
    self.assertTrue(hasattr(package, "load_artifact_patch_asset_source_manifest"))
    self.assertTrue(hasattr(package, "ingest_artifact_patch_matrix_assets"))
    self.assertTrue(hasattr(package_module, "load_artifact_patch_asset_source_manifest"))
    self.assertTrue(hasattr(package_module, "ingest_artifact_patch_matrix_assets"))


ArtifactPatchMatrixIngestPackageApiAppendTest.test_top_level_exports_artifact_patch_matrix_ingest_api = _override_artifact_patch_matrix_ingest_package_api_test

class ArtifactPatchMatrixPipelinePackageApiAppendTest(unittest.TestCase):
    def test_top_level_exports_artifact_patch_matrix_pipeline_api(self):
        import hgpipe_quantization as package_module

        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        package = HgPipeQuantizationPackage(source_path)
        self.assertTrue(hasattr(package, "run_artifact_patch_matrix_pipeline"))
        self.assertTrue(hasattr(package_module, "run_artifact_patch_matrix_pipeline"))

class ArtifactPatchSourceManifestPackageApiAppendTest(unittest.TestCase):
    def test_top_level_exports_artifact_patch_source_manifest_api(self):
        import hgpipe_quantization as package_module

        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        package = HgPipeQuantizationPackage(source_path)
        self.assertTrue(hasattr(package, "write_artifact_patch_asset_source_manifest_from_directory"))
        self.assertTrue(hasattr(package_module, "write_artifact_patch_asset_source_manifest_from_directory"))
