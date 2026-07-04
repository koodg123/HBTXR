from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hgpipe_quantization.artifact_patch import (
    evaluate_artifact_patch_batch_npy,
    iter_patch_inputs_from_npy,
    write_artifact_patch_batch_report,
)


class ArtifactPatchTest(unittest.TestCase):
    def _fake_package(self):
        torch_int = SimpleNamespace(topk=[SimpleNamespace(index=1), SimpleNamespace(index=2), SimpleNamespace(index=3), SimpleNamespace(index=4), SimpleNamespace(index=5)])
        fakequant = SimpleNamespace(topk=[SimpleNamespace(index=5), SimpleNamespace(index=4), SimpleNamespace(index=3), SimpleNamespace(index=2), SimpleNamespace(index=1)])
        comparison = SimpleNamespace(passed=True, mismatches=0)
        result = SimpleNamespace(torch_int=torch_int, fakequant_graph=fakequant, comparison=comparison)
        package = SimpleNamespace(compare_graph_runners=lambda input_values=None, topk=5: result)
        return package

    def test_iter_patch_inputs_accepts_vector_and_batch_but_rejects_scalar(self):
        vector = np.arange(4, dtype=np.int64)
        batch = np.arange(8, dtype=np.int64).reshape(2, 4)

        self.assertEqual([x.tolist() for x in iter_patch_inputs_from_npy(vector)], [[0, 1, 2, 3]])
        self.assertEqual([x.tolist() for x in iter_patch_inputs_from_npy(batch)], [[0, 1, 2, 3], [4, 5, 6, 7]])
        with self.assertRaisesRegex(ValueError, "expected 1D patch input"):
            list(iter_patch_inputs_from_npy(np.asarray(1, dtype=np.int64)))

    def test_evaluate_patch_batch_rejects_invalid_flow_and_count_mismatch_before_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_inputs = Path(tmpdir) / "patch.npy"
            labels = Path(tmpdir) / "labels.npy"
            np.save(patch_inputs, np.zeros((1, 4), dtype=np.int64))
            np.save(labels, np.asarray([0, 1], dtype=np.int64))

            with self.assertRaisesRegex(ValueError, "quantization_flow"):
                evaluate_artifact_patch_batch_npy("unused", patch_inputs_npy=patch_inputs, labels_npy=labels, model="m", precision="p", quantization_flow="int2")
            with self.assertRaisesRegex(ValueError, "patch input count 1 does not match label count 2"):
                evaluate_artifact_patch_batch_npy("unused", patch_inputs_npy=patch_inputs, labels_npy=labels, model="m", precision="p")

    def test_evaluate_patch_batch_selects_requested_flow_and_preserves_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_inputs = Path(tmpdir) / "patch.npy"
            labels = Path(tmpdir) / "labels.npy"
            np.save(patch_inputs, np.zeros((1, 4), dtype=np.int64))
            np.save(labels, np.asarray([[5]], dtype=np.int64))

            with patch("hgpipe_quantization.api.HgPipeQuantizationPackage", return_value=self._fake_package()):
                payload = evaluate_artifact_patch_batch_npy(
                    "unused",
                    patch_inputs_npy=patch_inputs,
                    labels_npy=labels,
                    model="deit_tiny_patch16_224",
                    precision="w4a8",
                    quantization_flow="fakequant_graph",
                    paper_equivalent=True,
                    topk=2,
                )

            self.assertEqual(payload["evaluation_mode"], "hgpipe_artifact_graph")
            self.assertEqual(payload["quantization_flow"], "fakequant_graph")
            self.assertTrue(payload["paper_equivalent"])
            self.assertEqual(payload["patch_input_shape"], [1, 4])
            self.assertEqual(payload["labels_shape"], [1, 1])
            self.assertEqual(payload["samples_detail"][0]["selected_top1"], payload["samples_detail"][0]["fakequant_top1"])
            self.assertEqual(payload["top1"], 100.0)
            self.assertEqual(payload["top5"], 100.0)

    def test_write_patch_batch_report_append_replaces_same_model_precision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "report.json"
            first = {"model": "a", "precision": "int8", "samples": 1}
            second = {"model": "b", "precision": "int8", "samples": 2}
            replacement = {"model": "a", "precision": "int8", "samples": 3}

            write_artifact_patch_batch_report(first, report)
            write_artifact_patch_batch_report(second, report, append=True)
            write_artifact_patch_batch_report(replacement, report, append=True)

            rows = json.loads(report.read_text())
            self.assertEqual(len(rows), 2)
            self.assertEqual({(row["model"], row["precision"]): row["samples"] for row in rows}, {("a", "int8"): 3, ("b", "int8"): 2})

            write_artifact_patch_batch_report(second, report, append=False)
            self.assertEqual(json.loads(report.read_text()), [second])


if __name__ == "__main__":
    unittest.main()

from hgpipe_quantization.artifact_patch import (
    evaluate_artifact_patch_matrix_manifest,
    expected_artifact_patch_matrix_manifest,
    load_artifact_patch_matrix_manifest,
    write_artifact_patch_matrix_manifest_template,
)


class ArtifactPatchMatrixManifestTest(unittest.TestCase):
    def _fake_package(self):
        torch_int = SimpleNamespace(topk=[SimpleNamespace(index=1), SimpleNamespace(index=2), SimpleNamespace(index=3), SimpleNamespace(index=4), SimpleNamespace(index=5)])
        fakequant = SimpleNamespace(topk=[SimpleNamespace(index=5), SimpleNamespace(index=4), SimpleNamespace(index=3), SimpleNamespace(index=2), SimpleNamespace(index=1)])
        comparison = SimpleNamespace(passed=True, mismatches=0)
        result = SimpleNamespace(torch_int=torch_int, fakequant_graph=fakequant, comparison=comparison)
        return SimpleNamespace(compare_graph_runners=lambda input_values=None, topk=5: result)

    def test_matrix_manifest_template_has_expected_nine_entries(self):
        manifest = expected_artifact_patch_matrix_manifest()

        self.assertEqual(manifest["schema"], "hgpipe_artifact_patch_matrix_v1")
        self.assertEqual(len(manifest["entries"]), 9)
        self.assertEqual(manifest["entries"][0]["model"], "deit_tiny_patch16_224")
        self.assertEqual(manifest["entries"][0]["precision"], "int8")
        self.assertTrue(all(entry["paper_equivalent"] for entry in manifest["entries"]))

    def test_load_matrix_manifest_resolves_relative_paths_and_validates_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            manifest.write_text(json.dumps({"entries": [{"model": "m", "precision": "p", "patch_inputs_npy": "x.npy", "labels_npy": "y.npy", "quantization_flow": "fakequant_graph", "paper_equivalent": True}]}))

            entries = load_artifact_patch_matrix_manifest(manifest)

            self.assertEqual(entries[0]["patch_inputs_npy"], str(Path(tmpdir) / "x.npy"))
            self.assertEqual(entries[0]["labels_npy"], str(Path(tmpdir) / "y.npy"))
            self.assertEqual(entries[0]["quantization_flow"], "fakequant_graph")
            self.assertTrue(entries[0]["paper_equivalent"])

            manifest.write_text(json.dumps({"entries": [{"model": "m"}]}))
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_artifact_patch_matrix_manifest(manifest)

    def test_evaluate_matrix_manifest_runs_each_entry_and_global_paper_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_inputs = Path(tmpdir) / "patch.npy"
            labels = Path(tmpdir) / "labels.npy"
            manifest = Path(tmpdir) / "manifest.json"
            np.save(patch_inputs, np.zeros((1, 4), dtype=np.int64))
            np.save(labels, np.asarray([5], dtype=np.int64))
            manifest.write_text(json.dumps([
                {"model": "deit_tiny_patch16_224", "precision": "int8", "patch_inputs_npy": "patch.npy", "labels_npy": "labels.npy"},
                {"model": "deit_tiny_patch16_224", "precision": "w4a8", "patch_inputs_npy": "patch.npy", "labels_npy": "labels.npy", "quantization_flow": "fakequant_graph"},
            ]))

            with patch("hgpipe_quantization.api.HgPipeQuantizationPackage", return_value=self._fake_package()):
                rows = evaluate_artifact_patch_matrix_manifest("unused", manifest=manifest, paper_equivalent_inputs=True, topk=2)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["manifest_index"], 0)
            self.assertEqual(rows[1]["quantization_flow"], "fakequant_graph")
            self.assertTrue(all(row["paper_equivalent"] for row in rows))
            self.assertTrue(all(row["runner_comparison_passed"] for row in rows))

    def test_write_matrix_manifest_template_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "template.json"

            write_artifact_patch_matrix_manifest_template(output)

            entries = load_artifact_patch_matrix_manifest(output)
            self.assertEqual(len(entries), 9)

from hgpipe_quantization.artifact_patch import validate_artifact_patch_matrix_manifest


class ArtifactPatchMatrixManifestValidationTest(unittest.TestCase):
    def test_validate_matrix_manifest_reports_missing_files_and_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            manifest.write_text(json.dumps([{"model": "deit_tiny_patch16_224", "precision": "int8", "patch_inputs_npy": "missing_patch.npy", "labels_npy": "missing_labels.npy", "paper_equivalent": False}]))

            payload = validate_artifact_patch_matrix_manifest(manifest)

            self.assertFalse(payload["passed"])
            self.assertEqual(payload["rows"], 1)
            self.assertEqual(payload["expected_rows"], 9)
            self.assertEqual(len(payload["missing_pairs"]), 8)
            self.assertEqual(payload["paper_equivalent_rows"], 0)
            self.assertEqual(payload["existing_patch_input_files"], 0)
            self.assertEqual(payload["existing_label_files"], 0)
            self.assertEqual(len(payload["missing_patch_input_files"]), 1)
            self.assertEqual(len(payload["missing_label_files"]), 1)
            self.assertIn("patch_input_files", payload["errors"])
            self.assertIn("label_files", payload["errors"])

    def test_validate_matrix_manifest_passes_for_complete_existing_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            entries = expected_artifact_patch_matrix_manifest()["entries"]
            for entry in entries:
                patch_path = root / entry["patch_inputs_npy"]
                label_path = root / entry["labels_npy"]
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(patch_path, np.zeros((1, 4), dtype=np.int64))
                np.save(label_path, np.asarray([0], dtype=np.int64))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"entries": entries}))

            payload = validate_artifact_patch_matrix_manifest(manifest)

            self.assertTrue(payload["passed"])
            self.assertEqual(payload["rows"], 9)
            self.assertEqual(payload["paper_equivalent_rows"], 9)
            self.assertEqual(payload["existing_patch_input_files"], 9)
            self.assertEqual(payload["existing_label_files"], 9)
            self.assertEqual(payload["missing_patch_input_files"], [])
            self.assertEqual(payload["missing_label_files"], [])
            self.assertEqual(payload["errors"], [])

import tempfile
from pathlib import Path

from hgpipe_quantization.artifact_patch import expected_artifact_patch_matrix_manifest, validate_artifact_patch_matrix_manifest


class ArtifactPatchManifestIntegrityAppendTest(unittest.TestCase):
    def test_missing_single_row_manifest_reports_zero_loadable_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            manifest.write_text(json.dumps({
                "entries": [
                    {
                        "model": "deit_tiny_patch16_224",
                        "precision": "int8",
                        "patch_inputs_npy": "missing_patch.npy",
                        "labels_npy": "missing_labels.npy",
                        "paper_equivalent": True,
                        "quantization_flow": "torch_int",
                    }
                ]
            }))

            payload = validate_artifact_patch_matrix_manifest(manifest)

            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["loadable_patch_input_files"], 0)
            self.assertEqual(payload["loadable_label_files"], 0)
            self.assertEqual(payload["valid_patch_shape_rows"], 0)
            self.assertEqual(payload["matching_sample_count_rows"], 0)
            self.assertTrue(payload["invalid_entries"])

    def test_complete_nine_row_matrix_reports_all_integrity_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_entries = []
            for entry in expected_artifact_patch_matrix_manifest()["entries"]:
                patch_path = root / entry["patch_inputs_npy"]
                label_path = root / entry["labels_npy"]
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(patch_path, np.zeros((1, 196 * 768), dtype=np.int8))
                np.save(label_path, np.array([0], dtype=np.int64))
                manifest_entries.append({
                    "model": entry["model"],
                    "precision": entry["precision"],
                    "patch_inputs_npy": entry["patch_inputs_npy"],
                    "labels_npy": entry["labels_npy"],
                    "paper_equivalent": True,
                    "quantization_flow": "torch_int",
                })
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"entries": manifest_entries}))

            payload = validate_artifact_patch_matrix_manifest(manifest)

            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["loadable_patch_input_files"], 9)
            self.assertEqual(payload["loadable_label_files"], 9)
            self.assertEqual(payload["valid_patch_shape_rows"], 9)
            self.assertEqual(payload["valid_label_shape_rows"], 9)
            self.assertEqual(payload["integer_patch_dtype_rows"], 9)
            self.assertEqual(payload["integer_label_dtype_rows"], 9)
            self.assertEqual(payload["matching_sample_count_rows"], 9)

    def test_invalid_integrity_reports_shape_dtype_and_sample_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch_path = root / "bad_patch.npy"
            label_path = root / "bad_labels.npy"
            np.save(patch_path, np.zeros((1, 4), dtype=np.float32))
            np.save(label_path, np.array([0, 1], dtype=np.int64))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "entries": [
                    {
                        "model": "deit_tiny_patch16_224",
                        "precision": "int8",
                        "patch_inputs_npy": "bad_patch.npy",
                        "labels_npy": "bad_labels.npy",
                        "paper_equivalent": True,
                        "quantization_flow": "torch_int",
                    }
                ]
            }))

            payload = validate_artifact_patch_matrix_manifest(manifest)

            self.assertEqual(payload["status"], "failed")
            self.assertTrue(payload["invalid_entries"])
            self.assertIn("patch_input_shape", payload["invalid_entries"][0]["errors"])
            self.assertIn("patch_input_dtype", payload["invalid_entries"][0]["errors"])
            self.assertIn("sample_count", payload["invalid_entries"][0]["errors"])


def _override_complete_existing_matrix_test(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        entries = expected_artifact_patch_matrix_manifest()["entries"]
        for entry in entries:
            patch_path = root / entry["patch_inputs_npy"]
            label_path = root / entry["labels_npy"]
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(patch_path, np.zeros((1, 196 * 768), dtype=np.int8))
            np.save(label_path, np.asarray([0], dtype=np.int64))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"entries": entries}))

        payload = validate_artifact_patch_matrix_manifest(manifest)

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["rows"], 9)
        self.assertEqual(payload["paper_equivalent_rows"], 9)
        self.assertEqual(payload["existing_patch_input_files"], 9)
        self.assertEqual(payload["existing_label_files"], 9)
        self.assertEqual(payload["loadable_patch_input_files"], 9)
        self.assertEqual(payload["loadable_label_files"], 9)
        self.assertEqual(payload["valid_patch_shape_rows"], 9)
        self.assertEqual(payload["valid_label_shape_rows"], 9)
        self.assertEqual(payload["integer_patch_dtype_rows"], 9)
        self.assertEqual(payload["integer_label_dtype_rows"], 9)
        self.assertEqual(payload["matching_sample_count_rows"], 9)
        self.assertEqual(payload["patch_input_width"], 196 * 768)
        self.assertEqual(payload["missing_patch_input_files"], [])
        self.assertEqual(payload["missing_label_files"], [])
        self.assertEqual(payload["invalid_entries"], [])
        self.assertEqual(payload["errors"], [])


ArtifactPatchMatrixManifestValidationTest.test_validate_matrix_manifest_passes_for_complete_existing_matrix = _override_complete_existing_matrix_test

from hgpipe_quantization.artifact_patch import ingest_artifact_patch_matrix_assets, load_artifact_patch_asset_source_manifest


class ArtifactPatchManifestIngestAppendTest(unittest.TestCase):
    def test_ingest_copies_full_nine_row_matrix_into_template_destinations(self):
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

            payload = ingest_artifact_patch_matrix_assets(
                source_manifest=source_manifest,
                template_manifest=template_manifest,
                output_manifest=output_manifest,
                copy=True,
                assert_paper_equivalent=False,
                report_json=report_json,
                report_markdown=report_markdown,
            )

            self.assertEqual(payload["matched_rows"], 9)
            self.assertEqual(payload["copied_files"], 18)
            self.assertTrue(payload["validation"]["passed"])
            self.assertEqual(payload["validation"]["status"], "passed")
            self.assertTrue(report_json.exists())
            self.assertTrue(report_markdown.exists())
            for row in load_artifact_patch_matrix_manifest(output_manifest):
                self.assertTrue(Path(row["patch_inputs_npy"]).exists())
                self.assertTrue(Path(row["labels_npy"]).exists())

    def test_ingest_paper_equivalent_flag_requires_source_or_assertion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = expected_artifact_patch_matrix_manifest()["entries"][0]
            source_patch = root / "source_assets" / "patch.npy"
            source_label = root / "source_assets" / "labels.npy"
            source_patch.parent.mkdir(parents=True, exist_ok=True)
            np.save(source_patch, np.zeros((1, 196 * 768), dtype=np.int8))
            np.save(source_label, np.array([0], dtype=np.int64))
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text(json.dumps({"entries": [{
                "model": first["model"],
                "precision": first["precision"],
                "patch_inputs_npy": str(source_patch),
                "labels_npy": str(source_label),
                "paper_equivalent": False,
            }]}))
            template_manifest = root / "configs" / "single.template.json"
            template_manifest.parent.mkdir(parents=True, exist_ok=True)
            template_manifest.write_text(json.dumps({"entries": [{
                "model": first["model"],
                "precision": first["precision"],
                "patch_inputs_npy": "patch_inputs/single_patch.npy",
                "labels_npy": "patch_inputs/single_labels.npy",
                "paper_equivalent": True,
                "quantization_flow": "torch_int",
            }]}, indent=2, sort_keys=True))
            output_false = root / "configs" / "out_false.json"
            output_true = root / "configs" / "out_true.json"

            payload_false = ingest_artifact_patch_matrix_assets(
                source_manifest=source_manifest,
                template_manifest=template_manifest,
                output_manifest=output_false,
                copy=False,
                assert_paper_equivalent=False,
            )
            payload_true = ingest_artifact_patch_matrix_assets(
                source_manifest=source_manifest,
                template_manifest=template_manifest,
                output_manifest=output_true,
                copy=False,
                assert_paper_equivalent=True,
            )

            output_false_rows = json.loads(output_false.read_text())["entries"]
            output_true_rows = json.loads(output_true.read_text())["entries"]
            self.assertFalse(output_false_rows[0]["paper_equivalent"])
            self.assertTrue(output_true_rows[0]["paper_equivalent"])
            self.assertEqual(payload_false["matched_rows"], 1)
            self.assertEqual(payload_true["matched_rows"], 1)

from hgpipe_quantization.artifact_patch import run_artifact_patch_matrix_pipeline


class ArtifactPatchMatrixPipelineAppendTest(unittest.TestCase):
    def _pipeline_rows(self):
        rows = []
        for entry in expected_artifact_patch_matrix_manifest()["entries"]:
            rows.append({
                "model": entry["model"],
                "precision": entry["precision"],
                "samples": 1,
                "top1": 100.0,
                "top5": 100.0,
                "evaluation_mode": "hgpipe_artifact_graph",
                "quantization_flow": "torch_int",
                "paper_equivalent": True,
                "runner_comparison_passed": True,
                "runner_comparison_mismatches": 0,
                "samples_detail": [],
            })
        return rows

    def test_run_artifact_patch_matrix_pipeline_runs_matrix_after_valid_ingest(self):
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
            matrix_report = root / "reports" / "artifact_imagenet_accuracy.json"
            manifest_report_json = root / "reports" / "artifact_patch_matrix_manifest_validation.json"
            manifest_report_markdown = root / "reports" / "artifact_patch_matrix_manifest_validation.md"
            validation_report_json = root / "reports" / "artifact_imagenet_validation.json"
            validation_report_markdown = root / "reports" / "artifact_imagenet_validation.md"

            with patch("hgpipe_quantization.artifact_patch.evaluate_artifact_patch_matrix_manifest", return_value=self._pipeline_rows()) as matrix_eval:
                payload = run_artifact_patch_matrix_pipeline(
                    "unused-source",
                    source_manifest=source_manifest,
                    manifest=output_manifest,
                    template_manifest=template_manifest,
                    output_manifest=output_manifest,
                    copy=True,
                    assert_paper_equivalent=False,
                    matrix_report=matrix_report,
                    manifest_report_json=manifest_report_json,
                    manifest_report_markdown=manifest_report_markdown,
                    validation_report_json=validation_report_json,
                    validation_report_markdown=validation_report_markdown,
                    strict=False,
                )

            self.assertTrue(matrix_eval.called)
            self.assertEqual(payload["status"], "passed")
            self.assertTrue(payload["ran_matrix"])
            self.assertTrue(payload["manifest_validation"]["passed"])
            self.assertTrue(payload["artifact_report_validation"]["passed"])
            self.assertTrue(matrix_report.exists())
            self.assertTrue(validation_report_json.exists())

    def test_run_artifact_patch_matrix_pipeline_skips_matrix_when_manifest_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"entries": [{
                "model": "deit_tiny_patch16_224",
                "precision": "int8",
                "patch_inputs_npy": "missing_patch.npy",
                "labels_npy": "missing_labels.npy",
                "paper_equivalent": True,
                "quantization_flow": "torch_int",
            }]}))
            matrix_report = root / "reports" / "artifact_imagenet_accuracy.json"
            manifest_report_json = root / "reports" / "artifact_patch_matrix_manifest_validation.json"
            manifest_report_markdown = root / "reports" / "artifact_patch_matrix_manifest_validation.md"
            validation_report_json = root / "reports" / "artifact_imagenet_validation.json"
            validation_report_markdown = root / "reports" / "artifact_imagenet_validation.md"

            with patch("hgpipe_quantization.artifact_patch.evaluate_artifact_patch_matrix_manifest") as matrix_eval:
                payload = run_artifact_patch_matrix_pipeline(
                    "unused-source",
                    manifest=manifest,
                    matrix_report=matrix_report,
                    manifest_report_json=manifest_report_json,
                    manifest_report_markdown=manifest_report_markdown,
                    validation_report_json=validation_report_json,
                    validation_report_markdown=validation_report_markdown,
                    strict=False,
                )

            matrix_eval.assert_not_called()
            self.assertEqual(payload["status"], "failed")
            self.assertFalse(payload["ran_matrix"])
            self.assertEqual(payload["skipped_reason"], "manifest_validation_failed")
            self.assertIsNone(payload["artifact_report_validation"])

from hgpipe_quantization.artifact_patch import write_artifact_patch_asset_source_manifest_from_directory


class ArtifactPatchSourceManifestAppendTest(unittest.TestCase):
    def test_write_artifact_patch_source_manifest_finds_flat_and_nested_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            flat_patch = root / "deit_tiny_patch16_224_int8_patch_inputs.npy"
            flat_label = root / "deit_tiny_patch16_224_int8_labels.npy"
            nested_patch = root / "deit_small_patch16_224" / "int4" / "inputs.npy"
            nested_label = root / "deit_small_patch16_224" / "int4" / "targets.npy"
            nested_patch.parent.mkdir(parents=True, exist_ok=True)
            np.save(flat_patch, np.zeros((1, 196 * 768), dtype=np.int8))
            np.save(flat_label, np.array([0], dtype=np.int64))
            np.save(nested_patch, np.zeros((1, 196 * 768), dtype=np.int8))
            np.save(nested_label, np.array([0], dtype=np.int64))
            output_manifest = root / "source_manifest.json"

            payload = write_artifact_patch_asset_source_manifest_from_directory(
                asset_dir=root,
                output_manifest=output_manifest,
                paper_equivalent=False,
                quantization_flow="torch_int",
            )

            manifest_rows = json.loads(output_manifest.read_text())["entries"]
            self.assertEqual(len(manifest_rows), 2)
            self.assertEqual(len(payload["found_pairs"]), 2)
            self.assertGreaterEqual(len(payload["missing_pairs"]), 1)
            self.assertFalse(any(row["paper_equivalent"] for row in manifest_rows))
            self.assertEqual({(row["model"], row["precision"]) for row in manifest_rows}, {("deit_tiny_patch16_224", "int8"), ("deit_small_patch16_224", "int4")})

    def test_write_artifact_patch_source_manifest_sets_paper_equivalent_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch_path = root / "vit_tiny_patch16_224_w4a8_patch_inputs.npy"
            label_path = root / "vit_tiny_patch16_224_w4a8_labels.npy"
            np.save(patch_path, np.zeros((1, 196 * 768), dtype=np.int8))
            np.save(label_path, np.array([0], dtype=np.int64))
            output_false = root / "out_false.json"
            output_true = root / "out_true.json"

            write_artifact_patch_asset_source_manifest_from_directory(root, output_false, paper_equivalent=False)
            write_artifact_patch_asset_source_manifest_from_directory(root, output_true, paper_equivalent=True)

            self.assertFalse(json.loads(output_false.read_text())["entries"][0]["paper_equivalent"])
            self.assertTrue(json.loads(output_true.read_text())["entries"][0]["paper_equivalent"])
