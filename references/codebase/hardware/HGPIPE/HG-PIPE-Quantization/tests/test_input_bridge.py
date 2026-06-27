import unittest

import numpy as np

from hgpipe_quantization.input_bridge import (
    PatchInputBridgeConfig,
    extract_patch_vectors,
    patch_input_contract,
    quantize_patches_symmetric,
    estimate_scale_from_npy_array,
    iter_images_from_npy,
    to_hgpipe_patch_input,
)


class InputBridgeTest(unittest.TestCase):
    def test_extract_patch_vectors_uses_channel_first_patch_order(self):
        config = PatchInputBridgeConfig(image_size=4, patch_size=2, tokens=4, channels=12, cls_slot=False)
        image = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)

        patches = extract_patch_vectors(image, config)

        self.assertEqual(patches.shape, (4, 12))
        expected_first = image[:, 0:2, 0:2].reshape(-1)
        np.testing.assert_array_equal(patches[0], expected_first)

    def test_to_hgpipe_patch_input_inserts_zero_cls_slot_and_drops_last_patch(self):
        config = PatchInputBridgeConfig(image_size=4, patch_size=2, tokens=4, channels=12, cls_slot=True)
        image = np.ones((3, 4, 4), dtype=np.float32)

        output = to_hgpipe_patch_input(image, scale=1.0, config=config).reshape(4, 12)

        np.testing.assert_array_equal(output[0], np.zeros(12, dtype=np.int64))
        np.testing.assert_array_equal(output[1:], np.ones((3, 12), dtype=np.int64))

    def test_quantize_patches_symmetric_clamps_to_signed_int8(self):
        patches = np.asarray([[-200.0, -1.2, 0.0, 1.2, 200.0]], dtype=np.float32)

        output = quantize_patches_symmetric(patches, scale=1.0)

        np.testing.assert_array_equal(output, np.asarray([[-128, -1, 0, 1, 127]], dtype=np.int64))

    def test_patch_input_contract_exposes_observed_artifact_range(self):
        contract = patch_input_contract()

        self.assertEqual(contract["observed_min"], -102)
        self.assertEqual(contract["observed_max"], 127)
        self.assertTrue(contract["cls_slot"])

    def test_estimate_scale_from_npy_batch_reports_contract(self):
        images = np.zeros((2, 3, 4, 4), dtype=np.float32)
        images[0, 0, 0, 0] = 127.0
        config = PatchInputBridgeConfig(image_size=4, patch_size=2, tokens=4, channels=12, cls_slot=True)

        payload = estimate_scale_from_npy_array(images, config=config)

        self.assertEqual(payload["images"], 2)
        self.assertEqual(payload["scale"], 1.0)
        self.assertFalse(payload["paper_equivalent"])
        self.assertEqual(payload["contract"]["image_size"], 4)

    def test_iter_images_from_npy_accepts_single_image_and_batch(self):
        single = np.zeros((4, 4, 3), dtype=np.float32)
        batch = np.zeros((2, 3, 4, 4), dtype=np.float32)

        self.assertEqual(len(list(iter_images_from_npy(single))), 1)
        self.assertEqual(len(list(iter_images_from_npy(batch))), 2)


if __name__ == "__main__":
    unittest.main()
