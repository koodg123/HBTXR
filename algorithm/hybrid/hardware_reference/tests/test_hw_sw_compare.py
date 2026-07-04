import numpy as np

from software.io import ensure_dir


def test_compare_reference_contract(tmp_path):
    root = tmp_path / "refs"
    ensure_dir(root / "outputs")
    np.save(root / "outputs" / "target_state.npy", np.zeros((1, 6), dtype=np.float32))
    assert (root / "outputs" / "target_state.npy").exists()

