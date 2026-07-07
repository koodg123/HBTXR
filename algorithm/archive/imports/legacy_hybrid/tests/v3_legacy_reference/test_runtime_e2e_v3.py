from __future__ import annotations

import json
import sys
from pathlib import Path

from hbtxr.preprocess.build_manifests import build_manifests
from hbtxr.preprocess.canonicalize import canonicalize_dataset
from _config import latest_run_root

from eval_hbtxr import main as eval_main
from infer_hbtxr import main as infer_main
from train_hbtxr import main as train_main
from visualize_inference_results import main as vis_infer_main
from visualize_runtime import main as vis_runtime_main


def _write_config(
    path: Path,
    *,
    stage: str,
    canonical_root: Path,
    experiment_name: str,
    ssl_enabled: bool,
    pruning_enabled: bool,
    mode: str = "mode1",
) -> None:
    mode = str(mode).strip().lower()
    manifest_name = "manifest2" if mode == "mode2" else "manifest1"
    canonical_name = "canonical2" if mode == "mode2" else "canonical1"
    frame_source = "interpolated" if mode == "mode2" else "original"
    payload = {
        "seed": 7,
        "model": {
            "embed_dim": 24,
            "depth": 1,
            "num_heads": 3,
            "mlp_ratio": 2.0,
            "patch_size": 16,
            "input_size": [256, 256],
            "aux_classes": 5,
            "pretrained": {"enabled": False},
        },
        "data": {
            "mode": mode,
            "canonical_name": canonical_name,
            "manifest_name": manifest_name,
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
            mode: {
                "canonical_name": canonical_name,
                "manifest_name": manifest_name,
                "frame_source": frame_source,
                "resize_policy": "facet_square_direct",
                "loader": {"use_cache": False, "per_channel_normalize": True},
                "synthetic_event_builder" if mode == "mode2" else "event_builder": {
                    "policy": "fixed_count",
                    "event_count_target": 4,
                    "time_bin_us": 5000,
                    "accumulation": "fast_causal_linear",
                    "causal_weight_power": 1.0,
                    "fast_causal_limit": 25.0,
                    "polarity_split": True,
                },
            },
        },
        "training": {
            "stage": stage,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
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
        "experiment": {"name": experiment_name},
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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_main(monkeypatch, main_fn, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    main_fn()


def _run_runtime_cycle(
    *,
    monkeypatch,
    tmp_path: Path,
    canonical_root: Path,
    train_manifest: Path,
    mode: str,
) -> None:
    mode = str(mode).strip().lower()
    suffix = f"{mode}_{tmp_path.name}"
    stage1_experiment = f"e2e_stage1_{suffix}"
    stage2_experiment = f"e2e_stage2_{suffix}"
    stage1_cfg = tmp_path / f"{mode}_stage1_e2e.yaml"
    stage2_cfg = tmp_path / f"{mode}_stage2_e2e.yaml"
    _write_config(stage1_cfg, stage="stage1", canonical_root=canonical_root, experiment_name=stage1_experiment, ssl_enabled=True, pruning_enabled=True, mode=mode)
    _write_config(stage2_cfg, stage="stage2", canonical_root=canonical_root, experiment_name=stage2_experiment, ssl_enabled=True, pruning_enabled=True, mode=mode)

    eval_out = tmp_path / "runs" / f"eval_{mode}"
    infer_out = tmp_path / "runs" / f"infer_{mode}"
    vis_infer_out = tmp_path / "runs" / f"vis_infer_{mode}"
    vis_runtime_out = tmp_path / "runs" / f"vis_runtime_{mode}"

    _run_main(
        monkeypatch,
        train_main,
        [
            "train_hbtxr.py",
            "--config",
            str(stage1_cfg),
            "--mode",
            mode,
            "--train-manifest",
            str(train_manifest),
            "--val-manifest",
            str(train_manifest),
            "--device",
            "cpu",
        ],
    )
    stage1_run_root = latest_run_root(Path(__file__).resolve().parents[1], stage1_experiment)
    assert stage1_run_root is not None
    stage1_ckpt = stage1_run_root / "train" / "best_search_p10.pt"
    assert stage1_ckpt.exists()

    _run_main(
        monkeypatch,
        train_main,
        [
            "train_hbtxr.py",
            "--config",
            str(stage2_cfg),
            "--mode",
            mode,
            "--train-manifest",
            str(train_manifest),
            "--val-manifest",
            str(train_manifest),
            "--init-checkpoint",
            str(stage1_ckpt),
            "--device",
            "cpu",
        ],
    )
    stage2_run_root = latest_run_root(Path(__file__).resolve().parents[1], stage2_experiment)
    assert stage2_run_root is not None
    stage2_ckpt = stage2_run_root / "train" / "best_track_p10.pt"
    assert stage2_ckpt.exists()

    _run_main(
        monkeypatch,
        eval_main,
        [
            "eval_hbtxr.py",
            "--config",
            str(stage2_cfg),
            "--mode",
            mode,
            "--manifest",
            str(train_manifest),
            "--checkpoint",
            str(stage2_ckpt),
            "--output",
            str(eval_out),
            "--split",
            "train",
            "--device",
            "cpu",
        ],
    )
    assert (eval_out / "eval_summary.json").exists()

    _run_main(
        monkeypatch,
        infer_main,
        [
            "infer_hbtxr.py",
            "--config",
            str(stage2_cfg),
            "--mode",
            mode,
            "--manifest",
            str(train_manifest),
            "--checkpoint",
            str(stage2_ckpt),
            "--output",
            str(infer_out),
            "--split",
            "train",
            "--device",
            "cpu",
        ],
    )
    infer_rows = infer_out / "infer_rows.jsonl"
    assert infer_rows.exists()

    _run_main(
        monkeypatch,
        vis_infer_main,
        [
            "visualize_inference_results.py",
            "--config",
            str(stage2_cfg),
            "--mode",
            mode,
            "--manifest",
            str(train_manifest),
            "--results",
            str(infer_rows),
            "--output",
            str(vis_infer_out),
            "--split",
            "train",
            "--limit",
            "1",
            "--device",
            "cpu",
        ],
    )
    assert any(vis_infer_out.glob("*.png"))

    _run_main(
        monkeypatch,
        vis_runtime_main,
        [
            "visualize_runtime.py",
            "--results",
            str(infer_rows),
            "--output-dir",
            str(vis_runtime_out),
        ],
    )
    assert (vis_runtime_out / "runtime_trace.txt").exists()

    infer_summary = json.loads((infer_out / "infer_summary.json").read_text(encoding="utf-8"))
    assert infer_summary["mode"] == mode

    with infer_rows.open("r", encoding="utf-8") as handle:
        first_row = json.loads(handle.readline())
    assert first_row["meta"]["data_mode"] == mode
    if mode == "mode2":
        assert first_row["meta"]["synthetic_frame_flag"] is True
        assert first_row["meta"]["frame_source"] == "interpolated"


def test_end_to_end_checkpoint_manifest_runtime_validation(synthetic_workspace, tmp_path, monkeypatch):
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    train_manifest = synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl"
    _run_runtime_cycle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        canonical_root=synthetic_workspace["canonical_root"],
        train_manifest=train_manifest,
        mode="mode1",
    )


def test_mode2_end_to_end_checkpoint_manifest_runtime_validation(synthetic_raw_workspace, tmp_path, monkeypatch):
    canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_alpha=0.5,
        interpolation_model="linear_blend",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    build_manifests(
        canonical_root=synthetic_raw_workspace["canonical_root"],
        manifests_root=synthetic_raw_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    train_manifest = synthetic_raw_workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl"
    _run_runtime_cycle(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        canonical_root=synthetic_raw_workspace["canonical_root"],
        train_manifest=train_manifest,
        mode="mode2",
    )
