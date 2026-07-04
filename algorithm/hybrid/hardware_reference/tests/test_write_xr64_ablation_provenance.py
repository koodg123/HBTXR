from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def base_args(tmp_path: Path, *, lane: str = "a") -> Namespace:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    run_root = tmp_path / "runs" / "XR-64" / "xr64_example_20260621_010101"
    train = run_root / "train"
    train.mkdir(parents=True)
    init_ckpt = project_root / "init.pt"
    teacher_ckpt = project_root / "teacher.pt"
    target_override = project_root / "data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json"
    train_manifest = project_root / "data/_internal/manifests/manifest1/train_manifest.jsonl"
    val_manifest = project_root / "data/_internal/manifests/manifest1/val_manifest.jsonl"
    test_manifest = project_root / "data/_internal/manifests/manifest1/test_manifest.jsonl"
    for path in (init_ckpt, teacher_ckpt, target_override, train_manifest, val_manifest, test_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name + "\n", encoding="utf-8")
    return Namespace(
        project_root=str(project_root),
        run_root=str(run_root),
        lane=lane,
        rule_tag="conservative" if lane == "a" else "threshold",
        experiment_name="xr64_example",
        device="cuda:0",
        init_checkpoint=str(init_ckpt),
        teacher_checkpoint=str(teacher_ckpt),
        target_override_path=str(target_override),
        train_manifest=str(train_manifest),
        val_manifest=str(val_manifest),
        test_manifest=str(test_manifest),
        lr=3e-7,
        epochs=10,
        optimizer_name="adamw",
        optimizer_weight_decay=None,
        best_metric_name="metric_track_p10_pct",
        scheduler_metric_name="metric_track_p10_pct",
        trainable_include="track_center_heatmap_head.*,track_state_aux_head.*,event_adapter.*,patch_frontend.event_embed.proj.*",
        event_count_target=255000,
        min_event_count=160000,
        max_event_count=384000,
        reference_us=4000003,
        scale_power=0.5,
        grid_size=32,
        target_override_weight=0.0006,
        target_override_axis_weight=0.0,
        target_override_angle_weight=0.0,
        center_l2_weight=0.0025,
        heatmap_weight=0.004,
        heatmap_offset_weight=0.0015,
        p10_soft_weight=0.004,
        p5_soft_weight=0.002,
        state_weight=0.00035,
        eval_summary="",
        checkpoint_kind="",
    )


def test_write_train_artifacts_create_required_postrun_files(tmp_path: Path) -> None:
    module = load_module("write_xr64_ablation_provenance", Path("scripts/external/write_xr64_ablation_provenance.py").resolve())
    args = base_args(tmp_path)

    written = module.write_train_artifacts(args)

    names = sorted(path.name for path in written)
    assert names == [
        "ablation_attribution_report.json",
        "ablation_lr_manifest.json",
        "optimizer_config_snapshot.json",
        "teacher_provenance.json",
    ]
    lr_manifest = json.loads((Path(args.run_root) / "train/ablation_lr_manifest.json").read_text(encoding="utf-8"))
    teacher_provenance = json.loads((Path(args.run_root) / "train/teacher_provenance.json").read_text(encoding="utf-8"))
    assert lr_manifest["lane"] == "XR-64A"
    assert lr_manifest["ablation_axis"] == ["head", "loss", "lr", "teacher_model_training"]
    assert set(lr_manifest["ablation_changed_keys"]) == {"head", "loss", "lr", "teacher_model_training"}
    assert lr_manifest["ablation_benchmark_baseline_id"] == "XR-64-prep"
    assert lr_manifest["axis_certified"] is True
    assert teacher_provenance["teacher_checkpoint"]["sha256"]
    assert teacher_provenance["allow_test_target_override"] is False


def test_patch_eval_summary_adds_promotion_schema(tmp_path: Path) -> None:
    module = load_module("write_xr64_ablation_provenance", Path("scripts/external/write_xr64_ablation_provenance.py").resolve())
    args = base_args(tmp_path)
    summary = tmp_path / "eval" / "test" / "eval_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "manifest": str(Path(args.project_root) / "data/_internal/manifests/manifest1/test_manifest.jsonl"),
                "checkpoint": str(Path(args.run_root) / "train/best_track_p10.pt"),
                "mode": "mode1",
                "stage": "stage2",
                "split": "test",
                "metric_track_center_px": 16.0,
                "metric_track_p10_pct": 35.5,
                "metric_track_p5_pct": 12.5,
            }
        ),
        encoding="utf-8",
    )
    args.eval_summary = str(summary)
    args.checkpoint_kind = "best_track_p10"

    module.patch_eval_summary(args)

    patched = json.loads(summary.read_text(encoding="utf-8"))
    assert patched["lane"] == "XR-64A"
    assert patched["ablation_checkpoint_kind"] == "best_track_p10"
    assert patched["ablation_benchmark_baseline_id"] == "XR-64-prep"
    assert patched["axis_certified"] is True
    assert patched["ablation_changed_keys"]["teacher_model_training"]
    assert patched["data_track_target_override_path"] is None
    assert patched["data_allow_test_target_override"] is False
    assert patched["loss_track_target_override_center_l2_weight"] == 0.0
    assert patched["loss_track_state_aux_target_override_center_l2_weight"] == 0.0


def test_patched_eval_summaries_are_accepted_by_promotion_helper(tmp_path: Path) -> None:
    writer = load_module("write_xr64_ablation_provenance", Path("scripts/external/write_xr64_ablation_provenance.py").resolve())
    decision = load_module("decide_xr64_postrun_promotion_for_writer_test", Path("scripts/external/decide_xr64_postrun_promotion.py").resolve())
    specs: list[str] = []
    for lane in ("a", "b"):
        for checkpoint_kind in ("best_metric_track_center_px", "best_track_p10", "best_track_p5"):
            args = base_args(tmp_path / lane / checkpoint_kind, lane=lane)
            lane_id = "XR-64A" if lane == "a" else "XR-64B"
            center, p10, p5 = (16.0, 35.5, 12.5) if (lane, checkpoint_kind) == ("a", "best_track_p10") else (16.6, 35.1, 12.0)
            summary = tmp_path / lane / checkpoint_kind / "eval" / "test" / "eval_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "manifest": str(Path(args.project_root) / "data/_internal/manifests/manifest1/test_manifest.jsonl"),
                        "checkpoint": str(Path(args.run_root) / f"train/{checkpoint_kind}.pt"),
                        "mode": "mode1",
                        "stage": "stage2",
                        "split": "test",
                        "metric_track_center_px": center,
                        "metric_track_p10_pct": p10,
                        "metric_track_p5_pct": p5,
                        "metric_track_p1_pct": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            args.eval_summary = str(summary)
            args.checkpoint_kind = checkpoint_kind
            writer.patch_eval_summary(args)
            specs.append(f"{lane_id}:{checkpoint_kind}:{summary}")

    report = decision.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is True
    assert report["decision_status"] == "promoted"
    assert report["candidate_count"] == 6
    assert report["axis_certified_candidate_count"] == 6
    assert report["override_cleared_candidate_count"] == 6
    assert report["required_candidate_matrix_complete"] is True
    assert report["axis_claims_allowed"] is True
