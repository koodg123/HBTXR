from pathlib import Path
import json
import tempfile
import unittest

from hgpipe_quantization import audit_completion
from hgpipe_quantization.cli import main


class CompletionAuditTest(unittest.TestCase):
    def test_audit_completion_summarizes_generated_reports(self):
        project_root = Path(__file__).resolve().parents[1]

        payload = audit_completion(project_root)

        self.assertTrue(payload["passed"])
        self.assertFalse(payload["fully_complete"])
        self.assertEqual(payload["complete"] + payload["partial"], payload["total"])
        self.assertEqual(payload["partial"], 1)
        self.assertEqual(payload["complete"], 14)
        self.assertEqual(payload["incomplete"], 0)
        self.assertEqual(payload["missing"], 0)
        self.assertGreaterEqual(payload["total"], 10)
        self.assertTrue(any(item["requirement"] == "FakeQuant graph" for item in payload["items"]))
        experimental_item = next(item for item in payload["items"] if item["requirement"] == "artifact-backed image-batch report")
        self.assertEqual(experimental_item["status"], "complete")
        self.assertIn("experimental_rows=1", experimental_item["detail"])
        validator_item = next(item for item in payload["items"] if item["requirement"] == "artifact ImageNet paper-equivalence validator report")
        self.assertEqual(validator_item["status"], "complete")
        self.assertIn("status=failed", validator_item["detail"])
        manifest_item = next(item for item in payload["items"] if item["requirement"] == "artifact patch matrix manifest preflight report")
        self.assertEqual(manifest_item["status"], "complete")
        self.assertIn("status=failed", manifest_item["detail"])
        artifact_item = next(item for item in payload["items"] if item["requirement"].startswith("artifact-backed HG-PIPE ImageNet"))
        self.assertEqual(artifact_item["status"], "partial")
        preflight_item = next(item for item in payload["items"] if item["requirement"] == "paper-equivalence asset preflight report")
        self.assertEqual(preflight_item["status"], "complete")
        self.assertIn("ready=False", preflight_item["detail"])
        timm_item = next(item for item in payload["items"] if item["requirement"].startswith("PyTorch timm"))
        self.assertIn("provenance=9/9", timm_item["detail"])

    def test_audit_completion_cli_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "audit.json"
            markdown_path = Path(tmpdir) / "audit.md"

            code = main(["audit-completion", "--json", str(json_path), "--markdown", str(markdown_path)])

            payload = json.loads(json_path.read_text())
            markdown = markdown_path.read_text()
            self.assertEqual(code, 0)
            self.assertTrue(payload["passed"])
            self.assertFalse(payload["fully_complete"])
            self.assertIn("HG-PIPE Quantization Completion Audit", markdown)
            self.assertIn("Status: PARTIAL", markdown)
            self.assertIn("FakeQuant graph", markdown)



    def test_timm_eval_audit_requires_full_model_precision_coverage(self):
        from hgpipe_quantization.completion_audit import _timm_eval_item

        with tempfile.TemporaryDirectory() as tmpdir:
            reports = Path(tmpdir)
            base_row = {
                'model': 'deit_tiny_patch16_224',
                'precision': 'int8',
                'samples': 50000,
                'top1': 1.0,
                'top5': 2.0,
                'elapsed_sec': 1.0,
                'images_per_sec': 50000.0,
                'pretrained': True,
                'device': 'cuda',
            }
            (reports / 'imagenet_accuracy_int8_int4.json').write_text(json.dumps([base_row]))
            (reports / 'imagenet_accuracy_w4a8.json').write_text(json.dumps([]))

            item = _timm_eval_item(reports)

            self.assertEqual(item.status, 'incomplete')
            self.assertIn('missing_pairs', item.detail)
            self.assertIn('expected=9', item.detail)

if __name__ == "__main__":
    unittest.main()
