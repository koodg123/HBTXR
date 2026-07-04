from software.config import load_config
from software.trainer import train


def test_train_smoke(tmp_path):
    cfg = load_config("software/configs/base.yaml")
    cfg["model"]["embed_dim"] = 24
    cfg["model"]["depth"] = 1
    cfg["model"]["num_heads"] = 3
    cfg["training"]["batch_size"] = 1
    cfg["experiment"]["output_dir"] = str(tmp_path)
    result = train(cfg, smoke=True)
    assert result["history"]

