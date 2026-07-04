from pathlib import Path
import json
import numpy as np
import tempfile
import unittest

from hgpipe_quantization.cli import build_parser, main
from hgpipe_quantization.graph import ArtifactGraphRunner
from hgpipe_quantization.run_result import make_inference_result


class CliTest(unittest.TestCase):
    def setUp(self):
        self.source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"

    def test_parser_exposes_package_runtime_commands(self):
        parser = build_parser()
        help_text = parser.format_help()

        self.assertIn("audit-completion", help_text)
        self.assertIn("check-paper-equivalence-assets", help_text)
        self.assertIn("validate-artifact-imagenet-report", help_text)
        self.assertIn("export-contracts", help_text)
        self.assertIn("run-int", help_text)
        self.assertIn("run-fakequant-graph", help_text)
        self.assertIn("run-compare", help_text)
        self.assertIn("run-compare-image-npy", help_text)
        self.assertIn("run-artifact-image-batch-npy", help_text)
        self.assertIn("run-artifact-patch-batch-npy", help_text)
        self.assertIn("run-artifact-patch-matrix-npy", help_text)
        self.assertIn("write-artifact-patch-matrix-template", help_text)
        self.assertIn("validate-artifact-patch-matrix-manifest", help_text)
        self.assertIn("estimate-input-scale-npy", help_text)
        self.assertIn("compare-run-results", help_text)

        args = parser.parse_args(["audit-completion", "--refresh", "--device", "cpu"])
        self.assertTrue(args.refresh)
        self.assertEqual(args.device, "cpu")

    def test_export_contracts_cli_writes_scalar_zero_point_lut_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "contracts.json"

            code = main(["--source", str(self.source_path), "export-contracts", "--kind", "requant_table", "--json", str(output)])

            data = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(len(data), 48)
            self.assertIn("scalars", data[0]["params"])
            self.assertIn("shift_scale", data[0]["params"])
            self.assertIn("table_sizes", data[0]["params"])
            self.assertIsNone(data[0]["params"]["zero_point"])

    def test_compare_run_results_cli_reports_exact_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left.json"
            right = Path(tmpdir) / "right.json"
            output = Path(tmpdir) / "comparison.json"
            payload = make_inference_result(runner="x", output_name="logits", values=[1, 3, 2]).to_json()
            left.write_text(json.dumps(payload))
            right.write_text(json.dumps(payload))

            code = main(["compare-run-results", "--left", str(left), "--right", str(right), "--json", str(output)])

            comparison = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertTrue(comparison["passed"])
            self.assertEqual(comparison["mismatches"], 0)
            self.assertTrue(comparison["top1_equal"])

    def test_run_compare_cli_writes_combined_runner_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run_compare.json"
            markdown = Path(tmpdir) / "run_compare.md"

            code = main(["--source", str(self.source_path), "run-compare", "--json", str(output), "--markdown", str(markdown), "--topk", "3"])

            payload = json.loads(output.read_text())
            markdown_text = markdown.read_text()
            self.assertEqual(code, 0)
            self.assertIn("torch_int", payload)
            self.assertIn("fakequant_graph", payload)
            self.assertTrue(payload["comparison"]["passed"])
            self.assertEqual(payload["comparison"]["mismatches"], 0)
            self.assertTrue(payload["comparison"]["top1_equal"])
            self.assertEqual(len(payload["torch_int"]["topk"]), 3)
            self.assertIn("HG-PIPE Runner Comparison", markdown_text)
            self.assertIn("Status: PASS", markdown_text)
            self.assertIn("torch_int", markdown_text)
            self.assertIn("fakequant_graph", markdown_text)

    def test_run_compare_image_npy_cli_uses_explicit_scale_bridge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "image.npy"
            output = Path(tmpdir) / "run_compare_image.json"
            markdown = Path(tmpdir) / "run_compare_image.md"
            np.save(image, np.zeros((3, 224, 224), dtype=np.float32))

            code = main(["--source", str(self.source_path), "run-compare-image-npy", "--image-npy", str(image), "--scale", "1.0", "--json", str(output), "--markdown", str(markdown), "--topk", "2"])

            payload = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertTrue(payload["comparison"]["passed"])
            self.assertFalse(payload["input_bridge"]["paper_equivalent"])
            self.assertEqual(payload["input_bridge"]["scale"], 1.0)
            self.assertEqual(payload["input_bridge"]["image_shape"], [3, 224, 224])
            self.assertIn("Status: PASS", markdown.read_text())

    def test_estimate_input_scale_npy_cli_writes_experimental_scale_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images = Path(tmpdir) / "images.npy"
            output = Path(tmpdir) / "scale.json"
            batch = np.zeros((2, 3, 224, 224), dtype=np.float32)
            batch[0, 0, 0, 0] = 127.0
            np.save(images, batch)

            code = main(["estimate-input-scale-npy", "--images-npy", str(images), "--json", str(output)])

            payload = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(payload["scale"], 1.0)
            self.assertEqual(payload["images"], 2)
            self.assertFalse(payload["paper_equivalent"])
            self.assertEqual(payload["array_shape"], [2, 3, 224, 224])


    def test_run_artifact_image_batch_npy_cli_writes_experimental_accuracy_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images = Path(tmpdir) / "images.npy"
            labels = Path(tmpdir) / "labels.npy"
            output = Path(tmpdir) / "artifact_imagenet_accuracy.json"
            np.save(images, np.zeros((1, 3, 224, 224), dtype=np.float32))
            np.save(labels, np.asarray([0], dtype=np.int64))

            code = main(["--source", str(self.source_path), "run-artifact-image-batch-npy", "--images-npy", str(images), "--labels-npy", str(labels), "--scale", "1.0", "--json", str(output), "--topk", "2"])

            payload = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["samples"], 1)
            self.assertEqual(payload[0]["evaluation_mode"], "hgpipe_artifact_graph_experimental")
            self.assertEqual(payload[0]["quantization_flow"], "input_bridge_explicit_scale")
            self.assertFalse(payload[0]["paper_equivalent"])
            self.assertTrue(payload[0]["runner_comparison_passed"])


    def test_validate_artifact_imagenet_report_cli_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "artifact_imagenet_accuracy.json"
            output = Path(tmpdir) / "artifact_imagenet_validation.json"
            markdown = Path(tmpdir) / "artifact_imagenet_validation.md"
            report.write_text(json.dumps([{"model": "deit_tiny_patch16_224", "precision": "int8", "samples": 1, "top1": 0.0, "top5": 0.0, "evaluation_mode": "hgpipe_artifact_graph_experimental", "quantization_flow": "input_bridge_explicit_scale", "paper_equivalent": False}]))

            code = main(["validate-artifact-imagenet-report", "--report", str(report), "--json", str(output), "--markdown", str(markdown)])
            strict_code = main(["validate-artifact-imagenet-report", "--report", str(report), "--json", str(output), "--markdown", str(markdown), "--strict"])

            payload = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(strict_code, 1)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["expected_rows"], 9)
            self.assertEqual(payload["rows"], 1)
            self.assertEqual(len(payload["missing_pairs"]), 8)
            self.assertIn("paper_equivalent", payload["errors"])
            self.assertIn("Artifact ImageNet Paper-Equivalence Validation", markdown.read_text())


    def test_validate_artifact_patch_matrix_manifest_cli_writes_preflight_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            output = Path(tmpdir) / "manifest_validation.json"
            markdown = Path(tmpdir) / "manifest_validation.md"
            manifest.write_text(json.dumps({"entries": [{"model": "deit_tiny_patch16_224", "precision": "int8", "patch_inputs_npy": "missing_patch.npy", "labels_npy": "missing_labels.npy"}]}))

            code = main(["validate-artifact-patch-matrix-manifest", "--manifest", str(manifest), "--json", str(output), "--markdown", str(markdown)])
            strict_code = main(["validate-artifact-patch-matrix-manifest", "--manifest", str(manifest), "--json", str(output), "--markdown", str(markdown), "--strict"])

            payload = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(strict_code, 1)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["expected_rows"], 9)
            self.assertIn("patch_input_files", payload["errors"])
            self.assertIn("Artifact Patch Matrix Manifest Preflight", markdown.read_text())


    def test_write_artifact_patch_matrix_template_cli_writes_nine_entry_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest.json"

            code = main(["write-artifact-patch-matrix-template", "--json", str(output)])

            payload = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(payload["schema"], "hgpipe_artifact_patch_matrix_v1")
            self.assertEqual(len(payload["entries"]), 9)
            self.assertTrue(all(row["paper_equivalent"] for row in payload["entries"]))


    def test_run_artifact_patch_batch_npy_cli_writes_paper_equivalent_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_inputs = Path(tmpdir) / "patch_inputs.npy"
            labels = Path(tmpdir) / "labels.npy"
            output = Path(tmpdir) / "artifact_imagenet_accuracy.json"
            validation = Path(tmpdir) / "artifact_imagenet_validation.json"
            markdown = Path(tmpdir) / "artifact_imagenet_validation.md"
            patch = ArtifactGraphRunner(self.source_path).array("patch_embed_matmul_input.txt")
            np.save(patch_inputs, patch.reshape(1, -1))
            np.save(labels, np.asarray([0], dtype=np.int64))

            code = main(["--source", str(self.source_path), "run-artifact-patch-batch-npy", "--patch-inputs-npy", str(patch_inputs), "--labels-npy", str(labels), "--model", "deit_tiny_patch16_224", "--precision", "int8", "--paper-equivalent-inputs", "--json", str(output), "--topk", "2"])
            validation_code = main(["validate-artifact-imagenet-report", "--report", str(output), "--json", str(validation), "--markdown", str(markdown)])

            rows = json.loads(output.read_text())
            payload = json.loads(validation.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(validation_code, 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "deit_tiny_patch16_224")
            self.assertEqual(rows[0]["precision"], "int8")
            self.assertEqual(rows[0]["evaluation_mode"], "hgpipe_artifact_graph")
            self.assertEqual(rows[0]["quantization_flow"], "torch_int")
            self.assertTrue(rows[0]["paper_equivalent"])
            self.assertTrue(rows[0]["runner_comparison_passed"])
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["paper_equivalent_rows"], 1)
            self.assertEqual(payload["artifact_rows"], 1)
            self.assertEqual(len(payload["missing_pairs"]), 8)


    def test_check_paper_equivalence_assets_cli_writes_preflight_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "paper_equivalence_assets.json"
            markdown = Path(tmpdir) / "paper_equivalence_assets.md"

            code = main(["--source", str(self.source_path), "check-paper-equivalence-assets", "--json", str(output), "--markdown", str(markdown)])

            payload = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertFalse(payload["paper_equivalent_ready"])
            self.assertEqual(payload["total"], 5)
            original_policy = next(row for row in payload["requirements"] if row["name"] == "original_image_to_patch_quantization_policy")
            self.assertEqual(original_policy["status"], "missing")
            self.assertNotIn("reports/smoke_patch_inputs.npy", original_policy["matches"])
            self.assertIn("Paper-Equivalence Asset Preflight", markdown.read_text())


if __name__ == "__main__":
    unittest.main()

class ArtifactPatchMatrixIngestCliAppendTest(unittest.TestCase):
    def test_ingest_artifact_patch_matrix_assets_cli_smoke(self):
        from hgpipe_quantization.artifact_patch import expected_artifact_patch_matrix_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_entries = []
            for entry in expected_artifact_patch_matrix_manifest()["entries"]:
                source_patch = root / "source_assets" / Path(entry["patch_inputs_npy"]).name
                source_label = root / "source_assets" / Path(entry["labels_npy"]).name
                source_patch.parent.mkdir(parents=True, exist_ok=True)
                np.save(source_patch, np.zeros((1, 196 * 768), dtype=np.int8))
                np.save(source_label, np.array([0], dtype=np.int64))
                source_entries.append({
                    "model": entry["model"],
                    "precision": entry["precision"],
                    "patch_inputs_npy": str(source_patch),
                    "labels_npy": str(source_label),
                    "paper_equivalent": True,
                    "quantization_flow": "torch_int",
                })
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text(json.dumps({"entries": source_entries}))
            template_manifest = root / "configs" / "artifact_patch_matrix_manifest.template.json"
            template_manifest.parent.mkdir(parents=True, exist_ok=True)
            template_manifest.write_text(json.dumps(expected_artifact_patch_matrix_manifest(), indent=2, sort_keys=True))
            output_manifest = root / "configs" / "artifact_patch_matrix_manifest.json"
            report_json = root / "reports" / "artifact_patch_matrix_manifest_validation.json"
            report_markdown = root / "reports" / "artifact_patch_matrix_manifest_validation.md"

            code = main([
                "ingest-artifact-patch-matrix-assets",
                "--source-manifest", str(source_manifest),
                "--template-manifest", str(template_manifest),
                "--output-manifest", str(output_manifest),
                "--report-json", str(report_json),
                "--report-markdown", str(report_markdown),
            ])

            payload = json.loads(report_json.read_text())
            self.assertEqual(code, 0)
            self.assertTrue(output_manifest.exists())
            self.assertTrue(payload["passed"])

class ArtifactPatchMatrixPipelineCliAppendTest(unittest.TestCase):
    def test_run_artifact_patch_matrix_pipeline_cli_help_exposure(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("run-artifact-patch-matrix-pipeline", help_text)

class ArtifactPatchSourceManifestCliAppendTest(unittest.TestCase):
    def test_write_artifact_patch_source_manifest_cli_help_exposure(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("write-artifact-patch-source-manifest", help_text)

    def test_write_artifact_patch_source_manifest_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch_path = root / "deit_tiny_patch16_224_int8_patch_inputs.npy"
            label_path = root / "deit_tiny_patch16_224_int8_labels.npy"
            np.save(patch_path, np.zeros((1, 196 * 768), dtype=np.int8))
            np.save(label_path, np.array([0], dtype=np.int64))
            output_manifest = root / "source_manifest.json"

            code = main([
                "write-artifact-patch-source-manifest",
                "--asset-dir", str(root),
                "--output-manifest", str(output_manifest),
                "--paper-equivalent",
            ])

            payload = json.loads(output_manifest.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(len(payload["entries"]), 1)
            self.assertTrue(payload["entries"][0]["paper_equivalent"])
