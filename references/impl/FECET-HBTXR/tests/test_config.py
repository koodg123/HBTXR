from __future__ import annotations

from pathlib import Path

from _config import apply_config_overrides, load_config, resolve_experiment_name, resolve_training_entry


def test_config_resolution_and_overrides():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "stage1_search.yaml"
    cfg = load_config(config_path)
    resolved_cfg = apply_config_overrides(
        cfg,
        overrides=["training.epochs=3", "data.event_builder.policy=time_bin"],
        device_override="cpu",
        experiment_name_override="unit_stage1",
    )
    assert resolve_experiment_name(resolved_cfg, config_path=config_path) == "unit_stage1"
    assert resolved_cfg["training"]["epochs"] == 3
    assert resolved_cfg["training"]["device"] == "cpu"
    assert resolved_cfg["data"]["event_builder"]["policy"] == "time_bin"

    resolved = resolve_training_entry(
        resolved_cfg,
        config_path=config_path,
        project_root=project_root,
    )
    assert resolved["stage"] == "stage1"
    assert resolved["experiment_name"] == "unit_stage1"
    assert resolved["output_dir"].endswith("runs\\unit_stage1\\stage1") or resolved["output_dir"].endswith("runs/unit_stage1/stage1")
