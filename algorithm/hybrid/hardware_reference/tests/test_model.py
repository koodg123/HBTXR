from software.config import load_config
from software.dataset import make_synthetic_batch
from software.model import build_model


def test_model_forward_smoke():
    cfg = load_config("software/configs/base.yaml")
    cfg["model"]["embed_dim"] = 24
    cfg["model"]["depth"] = 1
    cfg["model"]["num_heads"] = 3
    model = build_model(cfg)
    batch = make_synthetic_batch(1)
    out = model.forward_train(batch)
    assert out["search/state"].shape == (1, 6)
    assert out["track/state"].shape == (1, 6)

