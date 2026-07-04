import torch

from software.dataset import make_synthetic_batch


def test_synthetic_batch_shapes():
    batch = make_synthetic_batch(2)
    assert batch["frame"].shape == (2, 1, 256, 256)
    assert batch["event"].shape == (2, 2, 256, 256)
    assert batch["prev_state"].shape == (2, 6)
    assert torch.isfinite(batch["frame"]).all()

