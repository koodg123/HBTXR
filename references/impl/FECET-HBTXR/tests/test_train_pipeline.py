from __future__ import annotations

from pathlib import Path

from prepare_dataset import build_manifests, build_session_index
from fecet_hbtxr.trainer import make_loader, train
from fecet_hbtxr.runtime import run_runtime_trace
from fecet_hbtxr.trainer import build_model, load_checkpoint, resolve_device_and_wrap


def _smoke_cfg(stage: str, canonical_root: Path) -> dict:
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
                "accumulation": "causal_linear",
                "causal_weight_power": 1.0,
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
        "runtime": {
            "search_conf_threshold": 0.35,
            "track_conf_threshold": 0.45,
            "track_quality_threshold": 0.45,
            "similarity_threshold": 0.5,
            "density_threshold": 0.0,
            "relocalize_cooldown": 2,
        },
    }


def test_train_stage1_and_stage2_smoke(synthetic_workspace, tmp_path):
    build_session_index(canonical_root=synthetic_workspace["canonical_root"], indexes_root=synthetic_workspace["indexes_root"])
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        split_scheme="random",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        event_policy="fixed_count",
        event_count_target=4,
    )
    train_manifest = str(synthetic_workspace["manifests_root"] / "train_manifest.jsonl")
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

    model = build_model(_smoke_cfg("stage2", synthetic_workspace["canonical_root"]))
    model, device = resolve_device_and_wrap(model, "cpu")
    load_checkpoint(model=model, optimizer=None, scheduler=None, scaler=None, checkpoint_path=stage2_out / "best_track_p10.pt", strict=False)
    loader = make_loader(train_manifest, _smoke_cfg("stage2", synthetic_workspace["canonical_root"]), shuffle=False)
    rows = run_runtime_trace(model, loader, device=device)
    assert rows
    assert "runtime_reason" in rows[0]
    assert "ellipse_xywht" in rows[0]
