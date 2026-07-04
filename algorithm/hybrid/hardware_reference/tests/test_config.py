from software.config import load_config


def test_load_base_config():
    cfg = load_config("software/configs/base.yaml")
    assert cfg["model"]["embed_dim"] == 192
    assert cfg["training"]["stage"] == "stage1"


def test_load_extends_config():
    cfg = load_config("software/configs/stage2_hybrid.yaml")
    assert cfg["training"]["stage"] == "stage2"

