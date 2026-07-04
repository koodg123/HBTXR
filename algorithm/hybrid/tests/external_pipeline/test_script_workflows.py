from __future__ import annotations

from pathlib import Path

from check_dataloader import inspect_dataloader
from export_hbtxr import export_from_checkpoint
from src.preprocess.build_manifests import build_manifests
from src.training.trainer import train


def _structural_cfg(stage: str, canonical_root: Path) -> dict:
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
            "student": {
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
            },
        },
        "data": {
            "mode": "mode1",
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
        "distillation": {
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
        },
        "regularization_ssl": {"enabled": False},
        "pruning": {
            "enabled": True,
            "scheme": "structural_width",
            "export": {"enabled": True, "filename": "student_export.pt", "report_filename": "student_export_report.json"},
            "student": {"width_ratio": 0.5},
        },
    }


def test_check_dataloader_reports_batch_contract(synthetic_workspace):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    manifest_path = synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl"

    summary = inspect_dataloader(
        cfg=_structural_cfg("stage1", synthetic_workspace["canonical_root"]),
        manifest_path=manifest_path,
        split="train",
    )

    assert summary["batch_size"] == 1
    assert "frame" in summary["keys"]
    assert summary["shapes"]["frame"]["shape"] == [1, 1, 256, 256]
    assert summary["shapes"]["event"]["shape"] == [1, 2, 256, 256]


def test_export_hbtxr_exports_student_checkpoint_from_training_checkpoint(synthetic_workspace, tmp_path):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    manifest_path = str(synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl")
    stage1_dir = tmp_path / "stage1"

    train(
        cfg=_structural_cfg("stage1", synthetic_workspace["canonical_root"]),
        train_manifest=manifest_path,
        val_manifest=manifest_path,
        output_dir=stage1_dir,
    )

    report = export_from_checkpoint(
        cfg=_structural_cfg("stage1", synthetic_workspace["canonical_root"]),
        checkpoint_path=stage1_dir / "best_search_p10.pt",
        output_dir=tmp_path / "export_job",
        stage="stage1",
    )

    assert Path(report["checkpoint_path"]).exists()
    assert Path(report["report_path"]).exists()
