from __future__ import annotations

import torch

from swift_hbtxr.model import HBTXRTracker


def test_model_output_abi():
    model = HBTXRTracker(embed_dim=48, depth=2, num_heads=3, patch_size=16, input_size=(256, 256))
    batch = {
        "frame": torch.randn(2, 1, 256, 256),
        "event": torch.randn(2, 2, 256, 256),
        "prev_state": torch.randn(2, 6),
    }
    outputs = model(batch)
    assert outputs["search/eye"].shape == (2, 5)
    assert outputs["search/pupil"].shape == (2, 7)
    assert outputs["event/pupil"].shape == (2, 7)
    assert outputs["track/pupil"].shape == (2, 8)
    assert outputs["search/state"].shape == (2, 6)
    assert outputs["track/state"].shape == (2, 6)
    assert outputs["search/mask_logits"].shape == (2, 1, 256, 256)
