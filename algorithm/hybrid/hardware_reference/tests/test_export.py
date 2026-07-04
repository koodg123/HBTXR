from pathlib import Path


def test_hardware_tree_exists():
    root = Path("hardware/hls")
    assert (root / "include" / "common.h").exists()
    assert (root / "src" / "hgtxr_top.cpp").exists()

