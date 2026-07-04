from __future__ import annotations

from pathlib import Path

import json
import torch

from src.preprocess.build_manifests import build_manifests
from src.training.trainer import build_model
from src.training.trainer import train


def _smoke_cfg(stage: str, canonical_root: Path, *, ssl_enabled: bool = False, pruning_enabled: bool = False) -> dict:
    return {
        "seed": 1,
        "model": {
            "embed_dim": 24,
            "depth": 1,
            "num_heads": 3,
            "mlp_ratio": 2.0,
            "patch_size": 16,
            "input_size": [256, 256],
            "aux_classes": 5,
        },
        "data": {
            "canonical_root": str(canonical_root),
            "input_size": [256, 256],
            "resize_policy": "facet_square_direct",
            "use_cache": False,
            "event_builder": {
                "policy": "fixed_count",
                "event_count_target": 4,
                "time_bin_us": 5000,
                "accumulation": "fast_causal_linear",
                "causal_weight_power": 1.0,
                "fast_causal_limit": 25.0,
                "polarity_split": True,
            },
        },
        "training": {
            "stage": stage,
            "batch_size": 1,
            "num_workers": 0,
            "epochs": 1,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "amp": False,
            "device": "cpu",
            "grad_accum_steps": 1,
            "scheduler": {"type": "none"},
            "early_stopping": {"enabled": False},
            "best_metric_name": "metric_search_p10_pct" if stage == "stage1" else "metric_track_p10_pct",
        },
        "loss": {
            "eye_weight": 1.0,
            "eye_conf_weight": 0.1,
            "mask_weight": 0.5,
            "search_xy_weight": 1.0,
            "search_ab_weight": 0.5,
            "search_trig_weight": 1.0,
            "search_geo_weight": 0.2,
            "search_conf_weight": 0.1,
            "event_xy_weight": 1.0,
            "event_ab_weight": 0.5,
            "event_trig_weight": 1.0,
            "event_geo_weight": 0.2,
            "event_conf_weight": 0.1,
            "track_xy_weight": 1.0,
            "track_ab_weight": 0.5,
            "track_trig_weight": 1.0,
            "track_geo_weight": 0.2,
            "track_conf_weight": 0.1,
            "track_quality_weight": 0.1,
            "consistency_weight": 0.1,
            "constraint_center_weight": 0.1,
            "constraint_center_radius": 24.0,
            "aux_weight": 0.0,
        },
        "ssl": {
            "enabled": ssl_enabled,
            "mode": "teacher_student",
            "teacher_student": True,
            "ema_decay": 0.9,
            "feature_similarity": True,
            "feature_weight": 0.05,
            "state_similarity": True,
            "state_weight": 0.05,
            "prediction_similarity": True,
            "prediction_weight": 0.05,
            "cross_modal_contrastive": True,
            "contrastive_weight": 0.05,
            "kd": {"enabled": True, "temperature": 1.0, "weight": 0.05},
            "rkd": {"enabled": True, "distance_weight": 0.01},
        },
        "pruning": {
            "enabled": pruning_enabled,
            "mode": "implicit_width_slicing",
            "method": "implicit_width_slicing",
            "slimmable_backbone": True,
            "width_candidates": [1.0, 0.5],
            "sample_strategy": "min",
            "student_width": 0.5,
            "width_loss_weight": 0.01,
        },
    }


def _structural_cfg(stage: str, canonical_root: Path) -> dict:
    cfg = _smoke_cfg(stage, canonical_root, ssl_enabled=False, pruning_enabled=False)
    cfg["distillation"] = {
        "enabled": True,
        "teacher_student": True,
        "ema_decay": 0.9,
        "feature_similarity": True,
        "feature_weight": 0.05,
        "state_similarity": True,
        "state_weight": 0.05,
        "prediction_similarity": True,
        "prediction_weight": 0.05,
        "mask_similarity": True,
        "mask_weight": 0.05,
        "kd": {"enabled": True, "temperature": 1.0, "weight": 0.05},
        "rkd": {"enabled": True, "distance_weight": 0.01},
    }
    cfg["regularization_ssl"] = {
        "enabled": False,
        "force_with_teacher": False,
        "feature_similarity": False,
        "state_similarity": True,
        "state_weight": 0.05,
        "prediction_similarity": False,
        "cross_modal_contrastive": True,
        "contrastive_weight": 0.05,
    }
    cfg["pruning"] = {
        "enabled": True,
        "scheme": "structural_width",
        "export": {"enabled": True, "filename": "student_export.pt", "report_filename": "student_export_report.json"},
        "student": {"width_ratio": 0.5},
    }
    cfg["model"]["student"] = {
        "embed_dim": 12,
        "depth": 1,
        "num_heads": 3,
        "mlp_ratio": 2.0,
        "patch_size": 16,
        "adapter_hidden_dim": 12,
        "prev_state_hidden_dim": 12,
        "head_hidden_dim": 12,
        "track_head_hidden_dim": 24,
        "mask_hidden_dim": 32,
    }
    return cfg


def test_train_stage1_and_stage2_smoke(synthetic_workspace, tmp_path):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    train_manifest = str(synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl")
    val_manifest = train_manifest

    stage1_out = tmp_path / "stage1"
    result1 = train(
        cfg=_smoke_cfg("stage1", synthetic_workspace["canonical_root"]),
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=stage1_out,
    )
    assert Path(result1["history_path"]).exists()
    assert (stage1_out / "best_search_p10.pt").exists()

    stage2_out = tmp_path / "stage2"
    result2 = train(
        cfg=_smoke_cfg("stage2", synthetic_workspace["canonical_root"]),
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=stage2_out,
        stage1_checkpoint=str(stage1_out / "best_search_p10.pt"),
    )
    assert Path(result2["history_path"]).exists()
    assert (stage2_out / "best_track_p10.pt").exists()


def test_train_smoke_with_ssl_and_pruning(synthetic_workspace, tmp_path):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    train_manifest = str(synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl")

    stage1_out = tmp_path / "stage1_ssl_prune"
    result = train(
        cfg=_smoke_cfg("stage1", synthetic_workspace["canonical_root"], ssl_enabled=True, pruning_enabled=True),
        train_manifest=train_manifest,
        val_manifest=train_manifest,
        output_dir=stage1_out,
    )
    history = json.loads(Path(result["history_path"]).read_text(encoding="utf-8"))
    assert "loss_ssl_total" in history[-1]["train"]
    assert "loss_pruning_total" in history[-1]["train"]
    assert "metric_active_width" in history[-1]["train"]


def test_train_smoke_with_structural_student_distillation_and_export(synthetic_workspace, tmp_path):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    train_manifest = str(synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl")

    stage1_out = tmp_path / "stage1_structural"
    result = train(
        cfg=_structural_cfg("stage1", synthetic_workspace["canonical_root"]),
        train_manifest=train_manifest,
        val_manifest=train_manifest,
        output_dir=stage1_out,
    )
    history = json.loads(Path(result["history_path"]).read_text(encoding="utf-8"))
    assert "loss_distillation_total" in history[-1]["train"]
    export_report = result["export_report"]
    assert export_report is not None
    export_ckpt = stage1_out / "export" / "student_export.pt"
    export_report_path = stage1_out / "export" / "student_export_report.json"
    assert export_ckpt.exists()
    assert export_report_path.exists()

    cfg = _structural_cfg("stage1", synthetic_workspace["canonical_root"])
    student = build_model(cfg, role="student")
    state = torch.load(export_ckpt, map_location="cpu")
    student.load_state_dict(state["model"], strict=False)
