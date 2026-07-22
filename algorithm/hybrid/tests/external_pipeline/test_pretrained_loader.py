from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from hybrid.models.hybrid_tracker import HBTXRTracker
from hybrid.models.preload.pretrained_loader import load_pretrained_weights, write_pretrained_report


def test_pretrained_loader_selective_load_and_report(tmp_path: Path):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    checkpoint_path = tmp_path / "pretrained.pt"
    torch.save(
        {
            "state_dict": {
                "module.0.weight": torch.ones_like(model[0].weight),
                "module.0.bias": torch.zeros_like(model[0].bias),
                "module.2.weight": torch.ones_like(model[2].weight),
                "module.2.bias": torch.zeros_like(model[2].bias),
                "module.99.weight": torch.randn(1),
            }
        },
        checkpoint_path,
    )
    report = load_pretrained_weights(model, checkpoint_path=checkpoint_path)
    assert report["loaded_count"] == 4
    assert "99.weight" in report["unmatched_source_keys"]
    report_path = write_pretrained_report(tmp_path / "report.json", report)
    assert report_path.exists()


def _tracker_for_variant(variant: str) -> HBTXRTracker:
    return HBTXRTracker(
        embed_dim=24,
        depth=1,
        num_heads=3,
        patch_size=16,
        input_size=(256, 256),
        patch_embed_cfg={"variant": variant},
    )


def test_pretrained_loader_maps_patch_embed_to_split1(tmp_path: Path):
    model = _tracker_for_variant("split1")
    checkpoint_path = tmp_path / "deit_split1.pt"
    patch_weight = torch.arange(24 * 3 * 16 * 16, dtype=torch.float32).view(24, 3, 16, 16)
    patch_bias = torch.linspace(0.0, 1.0, steps=24)
    torch.save({"model": {"patch_embed.proj.weight": patch_weight, "patch_embed.proj.bias": patch_bias}}, checkpoint_path)

    report = load_pretrained_weights(model, checkpoint_path=checkpoint_path, source="deit")
    assert "patch_frontend.proj.weight" in report["loaded_keys"]
    assert "patch_frontend.proj.bias" in report["loaded_keys"]
    gray = patch_weight.mean(dim=1, keepdim=True)
    loaded = model.state_dict()["patch_frontend.proj.weight"]
    assert torch.allclose(loaded[:, 0:1], gray)
    assert torch.allclose(loaded[:, 1:2], gray)
    assert torch.allclose(loaded[:, 2:3], gray)


def test_pretrained_loader_maps_patch_embed_to_split2(tmp_path: Path):
    model = _tracker_for_variant("split2")
    checkpoint_path = tmp_path / "deit_split2.pt"
    patch_weight = torch.arange(24 * 3 * 16 * 16, dtype=torch.float32).view(24, 3, 16, 16)
    patch_bias = torch.linspace(0.0, 1.0, steps=24)
    torch.save({"model": {"patch_embed.proj.weight": patch_weight, "patch_embed.proj.bias": patch_bias}}, checkpoint_path)

    report = load_pretrained_weights(model, checkpoint_path=checkpoint_path, source="deit")
    assert "patch_frontend.frame_proj.weight" in report["loaded_keys"]
    assert "patch_frontend.frame_proj.bias" in report["loaded_keys"]
    gray = patch_weight.mean(dim=1, keepdim=True)
    loaded = model.state_dict()["patch_frontend.frame_proj.weight"]
    assert torch.allclose(loaded, gray)


def test_pretrained_loader_partially_slices_for_structural_student(tmp_path: Path):
    model = HBTXRTracker(
        embed_dim=12,
        depth=1,
        num_heads=3,
        patch_size=16,
        input_size=(256, 256),
        patch_embed_cfg={"variant": "split2"},
    )
    checkpoint_path = tmp_path / "deit_student.pt"
    patch_weight = torch.arange(24 * 3 * 16 * 16, dtype=torch.float32).view(24, 3, 16, 16)
    patch_bias = torch.linspace(0.0, 1.0, steps=24)
    torch.save({"model": {"patch_embed.proj.weight": patch_weight, "patch_embed.proj.bias": patch_bias}}, checkpoint_path)

    report = load_pretrained_weights(model, checkpoint_path=checkpoint_path, source="deit")
    assert any(item["key"] == "patch_frontend.frame_proj.weight" for item in report["partially_loaded"])
    assert any(item["key"] == "patch_frontend.frame_proj.bias" for item in report["partially_loaded"])
    loaded = model.state_dict()["patch_frontend.frame_proj.weight"]
    expected = patch_weight.mean(dim=1, keepdim=True)[:12]
    assert torch.allclose(loaded, expected)
