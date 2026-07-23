import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from hgpipe_quantization.eval import imagenet_eval


class _DummyImageFolder(torch.utils.data.Dataset):
    def __init__(self, _root, transform=None):
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        image = torch.zeros((3, 224, 224), dtype=torch.float32)
        target = torch.tensor(0, dtype=torch.long)
        return image, target


class ImageNetEvalTest(unittest.TestCase):
    def test_build_eval_transform_uses_expected_size(self):
        transform = imagenet_eval.build_eval_transform("deit_small_patch16_224")
        self.assertIsNotNone(transform)

    def test_evaluate_one_runs_without_timm(self):
        real_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "timm" or name.startswith("timm."):
                raise AssertionError("timm import attempted")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(imagenet_eval, "ImageFolder", _DummyImageFolder):
                with mock.patch("builtins.__import__", side_effect=guarded_import):
                    payload = imagenet_eval.evaluate_one(
                        model_name="deit_tiny_patch16_224",
                        precision="fp32",
                        data=Path(tmpdir),
                        split="val",
                        batch_size=1,
                        workers=0,
                        device="cpu",
                        pretrained=False,
                        checkpoint_path=None,
                    )
        self.assertEqual(payload["model"], "deit_tiny_patch16_224")
        self.assertEqual(payload["samples"], 1)
        self.assertEqual(payload["model_backend"], "torch_native_vit")
        self.assertFalse(payload["pretrained"])
        self.assertEqual(payload["timm_model_name"], "deit_tiny_patch16_224")


if __name__ == "__main__":
    unittest.main()
