import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from hgpipe_quantization.eval.model_registry import PAPER_MODELS, create_paper_model


class ModelRegistryTest(unittest.TestCase):
    def test_all_paper_models_forward_cpu(self):
        image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        with torch.no_grad():
            for name in PAPER_MODELS:
                model, metadata = create_paper_model(name)
                output = model(image)
                self.assertEqual(tuple(output.shape), (1, 1000))
                self.assertFalse(metadata["pretrained"])
                self.assertEqual(metadata["model_backend"], "torch_native_vit")
                features = model.forward_features(image)
                self.assertEqual(features.shape[0], 1)

    def test_create_paper_model_does_not_import_timm(self):
        real_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "timm" or name.startswith("timm."):
                raise AssertionError("timm import attempted")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            model, metadata = create_paper_model("deit_tiny_patch16_224")
        self.assertIsNotNone(model)
        self.assertFalse(metadata["pretrained"])

    def test_checkpoint_loading_marks_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "deit_tiny_state.pt"
            model, _ = create_paper_model("deit_tiny_patch16_224")
            torch.save({"state_dict": model.state_dict()}, checkpoint_path)

            loaded_model, metadata = create_paper_model(
                "deit_tiny_patch16_224",
                checkpoint_path=checkpoint_path,
                pretrained=True,
            )

            self.assertTrue(metadata["pretrained"])
            self.assertTrue(metadata["checkpoint_loaded"])
            self.assertEqual(metadata["checkpoint_path"], str(checkpoint_path))
            self.assertTrue(torch.equal(loaded_model.head.weight, model.head.weight))


if __name__ == "__main__":
    unittest.main()
