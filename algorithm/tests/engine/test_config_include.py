"""Config include/merge (torch-free): experiment configs compose their fragments."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from engine.tools.load_config import load_config

ALGO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def in_algo_root():
    prev = Path.cwd()
    os.chdir(ALGO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.mark.parametrize(
    "name, target",
    [
        ("frame_hbtxr", "models.frame.FrameModel"),
        ("event_hbtxr", "models.event.EventModel"),
        ("hybrid_hbtxr", "models.hybrid.HybridModel"),
        ("mask_unet", "models.mask.MaskModel"),
    ],
)
def test_experiment_config_merges(in_algo_root, name, target):
    cfg = load_config(f"configs/experiment/{name}.yaml")
    assert "include" not in cfg, "include key must not survive the merge"
    assert cfg["model"]["target"] == target
    assert cfg["data"]["mode"] == "mode1"
    assert cfg["training"]["optimizer"]["name"] == "adamw"
    assert cfg["experiment"]["name"] == name


def test_distill_config_carries_teacher(in_algo_root):
    cfg = load_config("configs/experiment/frame_distill.yaml")
    assert cfg["model"]["target"] == "models.frame.FrameModel"          # student, via include
    assert cfg["distill"]["teacher"]["model"]["target"] == "models.frame.FrameModel"
    assert cfg["distill"]["weight"] == 1.0


def test_prune_config_inherits_model(in_algo_root):
    cfg = load_config("configs/experiment/frame_prune.yaml")
    assert cfg["model"]["target"] == "models.frame.FrameModel"          # inherited via include
    assert cfg["pruning"]["amount"] == 0.3
    assert cfg["pruning"]["structured"] is False


def test_local_keys_win_over_includes(in_algo_root):
    # hybrid overrides training.loss_weights; the override is deep-merged on top.
    cfg = load_config("configs/experiment/hybrid_hbtxr.yaml")
    assert cfg["training"]["loss_weights"]["ellipse"] == 1.0
