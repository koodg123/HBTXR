from __future__ import annotations

from pathlib import Path

import torch

from software.config import load_config
from hbtxr.config.runtime_config import build_dataset_kwargs
from hbtxr.reproduction import build_reproduction_manifest, write_reproduction_manifest
from hbtxr.training.model_factory import build_model
from hbtxr.models.hybrid_tracker import HBTXRTracker


def test_paper_reproduce_config_loads_and_exposes_targets():
    cfg = load_config("software/configs/experiments/paper_reproduce.yaml")

    assert cfg["model"]["depth"] == 8
    assert cfg["model"]["track_depth"] == 4
    assert cfg["model"]["event_cut_depth"] == 4
    assert cfg["model"]["frame_input_size"] == [128, 128]
    assert cfg["model"]["event_input_size"] == [64, 64]
    assert cfg["paper"]["targets"]["hybrid"]["pixel_error_px"] == 0.1812
    assert cfg["paper_stages"]["stage1"]["epochs"] == 100
    assert cfg["paper_stages"]["stage2"]["epochs"] == 200


def test_v3_dataset_kwargs_preserve_separate_input_sizes():
    cfg = load_config("software/configs/experiments/paper_reproduce.yaml")
    kwargs = build_dataset_kwargs(cfg["data"])

    assert kwargs["input_size"] == (128, 128)
    assert kwargs["frame_input_size"] == (128, 128)
    assert kwargs["event_input_size"] == (64, 64)
    assert kwargs["data_mode"] == "mode2"


def test_v3_build_model_wires_paper_depth_contract():
    cfg = load_config("software/configs/experiments/paper_reproduce.yaml")
    cfg["model"].update({"embed_dim": 24, "depth": 3, "track_depth": 1, "event_cut_depth": 1, "num_heads": 3})
    cfg["model"]["student"] = {"embed_dim": 24, "depth": 3, "num_heads": 3, "mlp_ratio": 2.0, "patch_size": 16}
    cfg["pruning"] = {"enabled": False}
    model = build_model(cfg)

    assert model.input_size == (128, 128)
    assert model.frame_input_size == (128, 128)
    assert model.event_input_size == (64, 64)
    assert model.event_cut_depth == 1
    assert model.track_depth == 1
    assert model.encoder.event_depth_limit == 1
    assert model.encoder.track_depth_limit == 1


def test_v3_runtime_step_outputs_selected_state_and_reason():
    model = HBTXRTracker(
        embed_dim=24,
        depth=2,
        track_depth=1,
        event_cut_depth=1,
        num_heads=3,
        mlp_ratio=2.0,
        patch_size=16,
        input_size=(64, 64),
        runtime_cfg={
            "search_conf_threshold": 0.0,
            "track_conf_threshold": 0.0,
            "track_quality_threshold": 0.0,
            "similarity_threshold": 0.0,
            "density_threshold": 0.0,
            "max_track_updates": 1,
        },
    )
    frame = torch.randn(1, 1, 64, 64)
    event = torch.randn(1, 2, 64, 64)
    prev_state = torch.zeros(1, 6)

    with torch.no_grad():
        outputs = model.runtime_step(frame=frame, event=event, prev_state=prev_state)

    assert outputs["runtime/state"] in {"search", "track"}
    assert outputs["runtime/reason"] in {"track_ready", "search_keep"}
    assert outputs["runtime/ellipse_state"].shape == (1, 6)
    assert outputs["runtime/search_conf"].shape == (1,)
    assert outputs["runtime/track_conf"].shape == (1,)
    assert outputs["runtime/track_quality"].shape == (1,)


def test_reproduction_manifest_marks_missing_final_artifacts(tmp_path: Path):
    cfg = load_config("software/configs/experiments/paper_reproduce.yaml")
    manifest = build_reproduction_manifest(cfg, project_root=Path.cwd())

    assert manifest["status"] == "structure_reproduction_only"
    assert manifest["exact_metric_reproduction_ready"] is False
    assert manifest["missing_artifacts"]["checkpoint"]["present"] is False
    assert manifest["paper_targets"]["hybrid"]["latency_ms"] == 0.43

    out_path = write_reproduction_manifest(cfg, project_root=Path.cwd(), output_dir=tmp_path)
    assert out_path.name == "reproduction_manifest.json"
    assert out_path.exists()
