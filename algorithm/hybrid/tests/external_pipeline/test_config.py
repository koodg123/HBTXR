from __future__ import annotations

import json
import shutil
from pathlib import Path

from hybrid.config.run_contract import collect_reference_only_reports as shared_collect_reference_only_reports
from hybrid.config.run_contract import default_checkpoint_for_stage as shared_default_checkpoint_for_stage
from hybrid.config.run_contract import resolve_experiment_name as shared_resolve_experiment_name
from hybrid.config.run_contract import resolve_manifest_path as shared_resolve_manifest_path
from hybrid.config.run_contract import resolve_resume_root as shared_resolve_resume_root
from hybrid.config.run_contract import resolve_run_contract as shared_resolve_run_contract
from hybrid.config.run_contract import resolve_training_entry as shared_resolve_training_entry
from hybrid.config.run_contract import write_run_artifacts as shared_write_run_artifacts
from hybrid.config.runtime_config import build_dataset_kwargs as shared_build_dataset_kwargs
from hybrid.config.runtime_config import resolve_mode_contract as shared_resolve_mode_contract
from hybrid.data.loader import build_dataset_kwargs as loader_build_dataset_kwargs

from _config import (
    apply_config_overrides,
    build_dataset_kwargs as script_build_dataset_kwargs,
    collect_reference_only_reports,
    default_checkpoint_for_stage,
    latest_run_root,
    load_config,
    materialize_experiment_name,
    resolve_experiment_name,
    resolve_manifest_path,
    resolve_mode_contract,
    resolve_resume_root,
    resolve_run_contract,
    resolve_training_entry,
    write_run_artifacts,
)


def test_config_resolution_and_overrides():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "stage1_train_a.yaml"
    cfg = load_config(config_path)
    resolved_cfg = apply_config_overrides(
        cfg,
        overrides=["training.epochs=3", "data.event_builder.policy=time_bin"],
        device_override="cuda:1",
        experiment_name_override="unit_stage1",
    )
    assert resolve_experiment_name(resolved_cfg, config_path=config_path) == "unit_stage1"
    assert resolved_cfg["training"]["epochs"] == 3
    assert resolved_cfg["training"]["device"] == "cuda:1"
    assert resolved_cfg["data"]["event_builder"]["policy"] == "time_bin"

    resolved = resolve_training_entry(
        resolved_cfg,
        config_path=config_path,
        project_root=project_root,
    )
    assert resolved["stage"] == "stage1"
    assert resolved["mode"] == "mode1"
    assert resolved["manifest_name"] == "manifest1"
    assert resolved["experiment_name"] == "unit_stage1"
    assert "/runs/" in resolved["output_dir"].replace("\\", "/")
    assert resolved["output_dir"].replace("\\", "/").endswith("/train")
    run_contract = resolve_run_contract(resolved_cfg, config_path=config_path, project_root=project_root, action="train")
    assert run_contract["train_dir"].replace("\\", "/").endswith("/train")
    assert run_contract["vis_dir"].replace("\\", "/").endswith("/vis")
    assert materialize_experiment_name("unit_stage1", timestamp="20260325_120000") == "unit_stage1_20260325_120000"


def test_experimental_yaml_alias_and_direct_paths_resolve():
    project_root = Path(__file__).resolve().parents[1]

    alias_mode0 = load_config(project_root / "configs" / "mode0_stage1.yaml")
    direct_mode0 = load_config(project_root / "exps" / "configs" / "mode0_stage1.yaml")
    assert alias_mode0 == direct_mode0
    assert alias_mode0["data"]["mode"] == "mode0"
    assert alias_mode0["training"]["stage"] == "stage1"

    alias_lazy = load_config(project_root / "configs" / "mode2_stage2_lazy_2000fps.yaml")
    direct_lazy = load_config(project_root / "exps" / "configs" / "mode2_stage2_lazy_2000fps.yaml")
    assert alias_lazy == direct_lazy
    assert alias_lazy["data"]["mode2"]["execution"] == "lazy_target_fps"
    assert alias_lazy["data"]["mode2"]["synthetic_event_builder"]["generation_strategy"] == "target_fps_session_store"


def test_manifest_and_resume_helpers(tmp_path: Path):
    project_root = tmp_path
    run_root = project_root / "runs" / "demo_run_20260325_120000"
    run_root.mkdir(parents=True)
    cfg = {
        "experiment": {"name": "demo_run"},
        "data": {"mode": "mode2"},
        "training": {"stage": "stage2"},
    }

    assert latest_run_root(project_root, "demo_run") == run_root
    assert resolve_resume_root(project_root, "demo_run", "auto") == run_root
    assert str(default_checkpoint_for_stage(run_root, "stage2")).replace("\\", "/").endswith("/train/best_track_p10.pt")

    manifest_path = resolve_manifest_path(cfg, project_root=project_root, split="test")
    assert str(manifest_path).replace("\\", "/").endswith("/manifests/manifest2/test_manifest.jsonl")


def test_reference_only_ssl_and_pruning_reports(tmp_path: Path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("experiment:\n  name: demo_run\n", encoding="utf-8")
    cfg = {
        "experiment": {"name": "demo_run"},
        "training": {"stage": "stage1", "device": "cpu"},
        "ssl": {
            "enabled": True,
            "mode": "reference_only",
            "teacher_student": True,
            "feature_similarity": True,
        },
        "pruning": {
            "enabled": False,
            "mode": "reference_only",
            "method": "implicit_width_slicing",
            "width_candidates": [1.0, 0.75],
        },
    }
    reports = collect_reference_only_reports(cfg)
    assert reports["ssl"]["status"] == "enabled"
    assert reports["ssl"]["active_wiring"] is True
    assert reports["pruning"]["implemented"] is True

    run_contract = resolve_run_contract(cfg, config_path=config_path, project_root=tmp_path, action="train")
    write_run_artifacts(
        run_contract=run_contract,
        cfg=cfg,
        config_path=config_path,
        cli_args=["--config", str(config_path)],
        overrides=[],
        device="cpu",
    )

    ssl_report = json.loads((Path(run_contract["hypers_dir"]) / "ssl_reference.json").read_text(encoding="utf-8"))
    pruning_report = json.loads((Path(run_contract["hypers_dir"]) / "pruning_reference.json").read_text(encoding="utf-8"))
    assert ssl_report["feature_name"] == "ssl"
    assert ssl_report["enabled"] is True
    assert pruning_report["feature_name"] == "pruning"
    assert pruning_report["status"] == "available_inactive"


def test_recommended_mode_stage_presets_resolve_to_structural_distillation():
    project_root = Path(__file__).resolve().parents[1]
    cases = [
        ("mode1_stage1.yaml", "mode1", "manifest1", "stage1", "metric_search_p10_pct"),
        ("mode1_stage2.yaml", "mode1", "manifest1", "stage2", "metric_track_p10_pct"),
        ("mode2_stage1.yaml", "mode2", "manifest2", "stage1", "metric_search_p10_pct"),
        ("mode2_stage2.yaml", "mode2", "manifest2", "stage2", "metric_track_p10_pct"),
    ]

    for filename, mode, manifest_name, stage, best_metric in cases:
        cfg = load_config(project_root / "configs" / filename)
        assert cfg["data"]["mode"] == mode
        assert cfg["data"]["manifest_name"] == manifest_name
        assert cfg["training"]["stage"] == stage
        assert cfg["training"]["best_metric_name"] == best_metric
        assert cfg["distillation"]["enabled"] is True
        assert cfg["regularization_ssl"]["enabled"] is False
        assert cfg["pruning"]["enabled"] is True
        assert cfg["pruning"]["scheme"] == "structural_width"
        assert cfg["pruning"]["export"]["enabled"] is True

    mode2_stage1_cfg = load_config(project_root / "configs" / "mode2_stage1.yaml")
    stage1_synth_builder = mode2_stage1_cfg["data"]["mode2"]["synthetic_event_builder"]
    assert stage1_synth_builder["event_count_target"] == 2500
    assert stage1_synth_builder["time_bin_us"] == 2500
    assert stage1_synth_builder["generation_strategy"] == "raw_window"

    mode2_stage2_cfg = load_config(project_root / "configs" / "mode2_stage2.yaml")
    stage2_synth_builder = mode2_stage2_cfg["data"]["mode2"]["synthetic_event_builder"]
    assert stage2_synth_builder["event_count_target"] == 2500
    assert stage2_synth_builder["time_bin_us"] == 2500
    assert stage2_synth_builder["generation_strategy"] == "source_pair_average"
    assert stage2_synth_builder["source_pair_scope"] == "pair_local"
    assert stage2_synth_builder["average_weighting"] == "alpha"


def test_shared_runtime_config_matches_script_and_loader_helpers():
    data_cfg = {
        "mode": "mode2",
        "input_size": [192, 192],
        "components": {
            "reader": {"variant": "default"},
            "roi": {"variant": "default"},
        },
        "mode2": {
            "canonical_name": "canonical2_custom",
            "manifest_name": "manifest2_custom",
            "frame_source": "target_fps_grid",
            "execution": "lazy_target_fps",
            "loader": {"use_cache": False},
            "synthetic_event_builder": {
                "policy": "time_bin",
                "time_bin_us": 2500,
                "generation_strategy": "target_fps_session_store",
            },
            "components": {
                "reader": {"variant": "split_v1"},
                "event_builder": {"variant": "split_v1"},
            },
        },
    }
    cfg = {"data": data_cfg}

    assert shared_build_dataset_kwargs(data_cfg) == loader_build_dataset_kwargs(data_cfg)
    assert shared_build_dataset_kwargs(data_cfg) == script_build_dataset_kwargs(data_cfg)
    assert shared_resolve_mode_contract(data_cfg) == resolve_mode_contract(cfg)


def test_shared_run_contract_helpers_match_script_wrappers(tmp_path: Path):
    config_path = tmp_path / "mode1_stage1.yaml"
    config_path.write_text("experiment:\n  name: demo_run\n", encoding="utf-8")
    cfg = {
        "experiment": {
            "name": "demo_run",
            "output_dir": "runs/manual_demo/train",
        },
        "run": {
            "materialized_experiment_name": "demo_run_20260325_120000",
        },
        "training": {
            "stage": "stage1",
        },
        "data": {
            "mode": "mode1",
        },
        "ssl": {
            "enabled": True,
            "mode": "reference_only",
        },
    }

    script_run_contract = resolve_run_contract(
        cfg,
        config_path=config_path,
        project_root=tmp_path,
        action="train",
    )
    shared_run_contract = shared_resolve_run_contract(
        cfg,
        config_path=config_path,
        project_root=tmp_path,
        action="train",
    )
    assert script_run_contract == shared_run_contract
    assert resolve_experiment_name(cfg, config_path=config_path) == shared_resolve_experiment_name(cfg, config_path=config_path)
    assert resolve_manifest_path(cfg, project_root=tmp_path, split="train") == shared_resolve_manifest_path(cfg, project_root=tmp_path, split="train")
    assert resolve_resume_root(tmp_path, "demo_run", "auto") == shared_resolve_resume_root(tmp_path, "demo_run", "auto")
    assert default_checkpoint_for_stage(tmp_path / "runs" / "demo_run_20260325_120000", "stage1") == shared_default_checkpoint_for_stage(
        tmp_path / "runs" / "demo_run_20260325_120000",
        "stage1",
    )
    assert collect_reference_only_reports(cfg) == shared_collect_reference_only_reports(cfg)

    script_training_entry = resolve_training_entry(
        cfg,
        config_path=config_path,
        project_root=tmp_path,
    )
    shared_training_entry = shared_resolve_training_entry(
        cfg,
        config_path=config_path,
        project_root=tmp_path,
    )
    assert script_training_entry == shared_training_entry

    script_root = Path(script_run_contract["root"])
    shared_root = tmp_path / "runs" / "shared_write_demo_20260325_120000"
    shared_run_contract_for_write = {
        **shared_run_contract,
        "root": str(shared_root),
        "train_dir": str(shared_root / "train"),
        "eval_dir": str(shared_root / "eval"),
        "infer_dir": str(shared_root / "infer"),
        "vis_dir": str(shared_root / "vis"),
        "hypers_dir": str(shared_root / "hypers"),
    }

    write_run_artifacts(
        run_contract=script_run_contract,
        cfg=cfg,
        config_path=config_path,
        cli_args=["--config", str(config_path)],
        overrides=["training.stage=stage1"],
        device="cpu",
    )
    shared_write_run_artifacts(
        run_contract=shared_run_contract_for_write,
        cfg=cfg,
        config_path=config_path,
        cli_args=["--config", str(config_path)],
        overrides=["training.stage=stage1"],
        device="cpu",
    )

    script_hypers = script_root / "hypers"
    shared_hypers = shared_root / "hypers"
    for filename in ("resolved_config.yaml", "resolved_config.json", "cli_args.txt", "overrides.txt", "device.txt", "run_contract.json", "ssl_reference.json"):
        assert (script_hypers / filename).read_text(encoding="utf-8")
        if filename == "run_contract.json":
            script_payload = json.loads((script_hypers / filename).read_text(encoding="utf-8"))
            shared_payload = json.loads((shared_hypers / filename).read_text(encoding="utf-8"))
            script_payload["root"] = "<normalized>"
            script_payload["train_dir"] = "<normalized>"
            script_payload["eval_dir"] = "<normalized>"
            script_payload["infer_dir"] = "<normalized>"
            script_payload["vis_dir"] = "<normalized>"
            script_payload["hypers_dir"] = "<normalized>"
            shared_payload["root"] = "<normalized>"
            shared_payload["train_dir"] = "<normalized>"
            shared_payload["eval_dir"] = "<normalized>"
            shared_payload["infer_dir"] = "<normalized>"
            shared_payload["vis_dir"] = "<normalized>"
            shared_payload["hypers_dir"] = "<normalized>"
            assert script_payload == shared_payload
        else:
            assert (script_hypers / filename).read_text(encoding="utf-8") == (shared_hypers / filename).read_text(encoding="utf-8")

    shutil.rmtree(script_root)
    shutil.rmtree(shared_root)
