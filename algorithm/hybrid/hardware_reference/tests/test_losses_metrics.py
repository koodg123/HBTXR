import torch

from hbtxr.loss.common import ellipse_gwd_loss
from software.config import load_config
from software.dataset import make_synthetic_batch
from software.losses import compute_stage1_losses
from software.metrics import compute_metrics
from software.model import build_model


def test_losses_and_metrics():
    cfg = load_config("software/configs/base.yaml")
    cfg["model"]["embed_dim"] = 24
    cfg["model"]["depth"] = 1
    cfg["model"]["num_heads"] = 3
    model = build_model(cfg)
    batch = make_synthetic_batch(1)
    out = model.forward_train(batch)
    losses = compute_stage1_losses(batch, out, cfg["loss"])
    metrics = compute_metrics(batch, out)
    assert "loss/total" in losses
    assert "metric/search_p10" in metrics


def test_ellipse_gwd_backward_is_finite_for_isotropic_covariance():
    pred = torch.tensor([[10.0, 12.0, 8.0, 8.0, 0.0, 1.0]], requires_grad=True)
    target = torch.tensor([[11.0, 13.0, 8.0, 8.0, 0.0, 1.0]])

    loss = ellipse_gwd_loss(pred, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
