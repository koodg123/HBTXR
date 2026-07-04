from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_decision_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "decide_xr64_postrun_promotion.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("decide_xr64_postrun_promotion", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(
    path: Path,
    *,
    center: float,
    p10: float,
    p5: float,
    p1: float | None = 1.0,
    split: str = "test",
    axis_certified=True,
    changed_keys=None,
    override_values=None,
) -> None:
    override_defaults = {
        "data_track_target_override_path": None,
        "data_allow_test_target_override": False,
        "loss_track_target_override_center_l2_weight": 0.0,
        "loss_track_state_aux_target_override_center_l2_weight": 0.0,
    }
    if override_values:
        override_defaults.update(override_values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "manifest": "/tmp/project/data/_internal/manifests/manifest1/test_manifest.jsonl",
                "checkpoint": "/tmp/project/runs/XR-64/example/train/best_track_p10.pt",
                "mode": "mode1",
                "split": split,
                "stage": "stage2",
                "metric_track_center_px": center,
                "metric_track_p10_pct": p10,
                "metric_track_p5_pct": p5,
                "ablation_axis": ["head", "loss", "lr", "teacher_model_training"],
                "ablation_changed_keys": changed_keys
                if changed_keys is not None
                else {
                    "head": ["model.track_center_heatmap_head"],
                    "loss": ["loss.track_target_override_center_l2_weight"],
                    "lr": ["optimizer.lr"],
                    "teacher_model_training": ["teacher.primary_checkpoint"],
                },
                "ablation_benchmark_baseline_id": "XR-64-prep",
                "axis_certified": axis_certified,
                **override_defaults,
                **({} if p1 is None else {"metric_track_p1_pct": p1}),
            }
        ),
        encoding="utf-8",
    )


def write_required_matrix(
    root: Path,
    *,
    winning_center: float = 16.0,
    winning_p10: float = 35.5,
    winning_p5: float = 12.5,
    p1_by_candidate: dict[str, float | None] | None = None,
    split_by_candidate: dict[str, str] | None = None,
    axis_certified_by_candidate: dict[str, bool | str | int] | None = None,
    changed_keys_by_candidate: dict[str, object] | None = None,
    override_values_by_candidate: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    specs: list[str] = []
    for lane in ("XR-64A", "XR-64B"):
        for checkpoint_kind in ("best_metric_track_center_px", "best_track_p10", "best_track_p5"):
            key = f"{lane}:{checkpoint_kind}"
            path = root / lane.lower() / checkpoint_kind / "eval_summary.json"
            write_summary(
                path,
                center=winning_center if key == "XR-64A:best_track_p10" else 16.6,
                p10=winning_p10 if key == "XR-64A:best_track_p10" else 35.1,
                p5=winning_p5 if key == "XR-64A:best_track_p10" else 12.0,
                p1=(p1_by_candidate or {}).get(key, 1.0),
                split=(split_by_candidate or {}).get(key, "test"),
                axis_certified=(axis_certified_by_candidate or {}).get(key, True),
                changed_keys=(changed_keys_by_candidate or {}).get(key),
                override_values=(override_values_by_candidate or {}).get(key),
            )
            specs.append(f"{lane}:{checkpoint_kind}:{path}")
    return specs


def test_decision_without_candidates_is_current_missing_state() -> None:
    module = load_decision_module()

    report = module.build_decision([], project_root=Path.cwd())

    assert report["ok"] is True
    assert report["decision_status"] == "missing_postrun_evidence"
    assert report["candidate_count"] == 0
    assert report["future_evidence_metrics"] == ["metric_track_p1_pct"]
    assert report["p1_evidence_candidate_count"] == 0
    assert report["axis_certified_candidate_count"] == 0
    assert report["axis_certification_required"] is True
    assert report["override_cleared_required"] is True
    assert report["override_cleared_candidate_count"] == 0
    assert report["axis_claims_allowed"] is False
    assert report["control_variables_checked"] is False
    assert report["gate_tradeoff_table"] == []
    assert report["bounded_tradeoff_candidate_count"] == 0
    assert report["unbounded_tradeoff_candidate_count"] == 0
    assert report["required_candidate_matrix_complete"] is False
    assert report["missing_required_candidate_count"] == 6
    assert report["rejection_reason"] == "missing_postrun_evidence"
    assert report["software_promotion_allowed"] is False
    assert report["paper_level_completion_allowed"] is False


def test_decision_promotes_candidate_that_beats_gate(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(tmp_path)

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is True
    assert report["decision_status"] == "promoted"
    assert report["candidate_count"] == 6
    assert report["software_promotion_allowed"] is True
    assert report["single_model_sota_claim_allowed"] is True
    assert report["paper_level_completion_allowed"] is False
    assert report["winning_candidate"]["lane"] == "XR-64A"
    assert report["winning_candidate"]["metric_track_p1_pct"] == 1.0
    assert report["p1_evidence_candidate_count"] == 6
    assert report["axis_certified_candidate_count"] == 6
    assert report["override_cleared_candidate_count"] == 6
    assert report["axis_claims_allowed"] is True
    assert report["control_variables_checked"] is True
    assert report["required_candidate_matrix_complete"] is True
    assert report["missing_required_candidate_count"] == 0
    assert report["rejection_reason"] is None
    assert report["gate_tradeoff_table"]
    assert report["winning_candidate"]["tradeoff_classification"] == "clean_multi_gate_improvement"
    assert report["winning_candidate"]["bounded_tradeoff"] is True
    assert report["bounded_tradeoff_candidate_count"] == 6
    assert report["unbounded_tradeoff_candidate_count"] == 0
    assert report["winning_candidate"]["promotion_score"] > 3000
    assert "XR-64A:best_track_p10" in report["axis_claims"]
    assert report["warnings"] == []
    assert report["promotion_summary"]["center_promoted"] is True
    assert report["promotion_summary"]["p10_promoted"] is True
    assert report["promotion_summary"]["p5_promoted"] is True


def test_decision_rejects_partial_candidate_matrix(tmp_path: Path) -> None:
    module = load_decision_module()
    a = tmp_path / "a" / "eval_summary.json"
    b = tmp_path / "b" / "eval_summary.json"
    write_summary(a, center=16.0, p10=35.5, p5=12.5)
    write_summary(b, center=16.6, p10=35.1, p5=12.0)

    report = module.build_decision(
        [
            f"XR-64A:best_track_p10:{a}",
            f"XR-64B:best_track_p10:{b}",
        ],
        project_root=Path.cwd(),
    )

    assert report["ok"] is False
    assert report["decision_status"] == "invalid_candidate_evidence"
    assert report["software_promotion_allowed"] is False
    assert report["single_model_sota_claim_allowed"] is False
    assert report["required_candidate_matrix_complete"] is False
    assert report["missing_required_candidate_count"] == 4
    assert any("missing required candidate matrix entries" in error for error in report["errors"])


def test_decision_rejects_missing_required_lane(tmp_path: Path) -> None:
    module = load_decision_module()
    a = tmp_path / "a" / "eval_summary.json"
    write_summary(a, center=16.0, p10=35.5, p5=12.5)

    report = module.build_decision([f"XR-64A:best_track_p10:{a}"], project_root=Path.cwd())

    assert report["ok"] is False
    assert report["decision_status"] == "invalid_candidate_evidence"
    assert "missing required lanes: ['XR-64B']" in report["errors"]


def test_decision_rejects_non_test_summary(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        split_by_candidate={"XR-64A:best_track_p10": "val"},
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("split must be test" in error for error in report["errors"])


def test_decision_tracks_missing_p1_as_future_evidence_warning(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        p1_by_candidate={
            "XR-64A:best_track_p10": None,
            "XR-64B:best_track_p10": 0.5,
        },
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is True
    assert report["software_promotion_allowed"] is True
    assert report["p1_evidence_candidate_count"] == 5
    assert any("missing future evidence metrics" in warning for warning in report["warnings"])


def test_decision_rejects_missing_axis_certification(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        axis_certified_by_candidate={"XR-64A:best_track_p10": False},
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is False
    assert report["decision_status"] == "invalid_candidate_evidence"
    assert report["software_promotion_allowed"] is False
    assert "XR-64A:best_track_p10: axis_certified must be true" in report["errors"]


def test_decision_rejects_non_cleared_override_fields(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        override_values_by_candidate={
            "XR-64A:best_track_p10": {
                "data_track_target_override_path": "data/_internal/manifests/manifest1/xr64_teacher_targets/test_override.json",
            },
            "XR-64B:best_track_p10": {
                "loss_track_target_override_center_l2_weight": 0.0006,
            },
        },
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is False
    assert report["decision_status"] == "invalid_candidate_evidence"
    assert report["software_promotion_allowed"] is False
    assert report["override_cleared_candidate_count"] == 4
    assert any("data_track_target_override_path must be JSON null" in error for error in report["errors"])
    assert any("loss_track_target_override_center_l2_weight must be JSON numeric 0.0" in error for error in report["errors"])


def test_decision_rejects_missing_override_cleared_fields(tmp_path: Path) -> None:
    module = load_decision_module()
    path = tmp_path / "a" / "eval_summary.json"
    write_summary(path, center=16.0, p10=35.5, p5=12.5)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("data_track_target_override_path")
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = module.build_decision([f"XR-64A:best_track_p10:{path}"], project_root=Path.cwd())

    assert report["ok"] is False
    assert any("data_track_target_override_path must be present" in error for error in report["errors"])


def test_decision_rejects_dotted_override_keys_without_flat_summary_keys(tmp_path: Path) -> None:
    module = load_decision_module()
    path = tmp_path / "a" / "eval_summary.json"
    write_summary(path, center=16.0, p10=35.5, p5=12.5)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "data_track_target_override_path",
        "data_allow_test_target_override",
        "loss_track_target_override_center_l2_weight",
        "loss_track_state_aux_target_override_center_l2_weight",
    ):
        payload.pop(key)
    payload.update(
        {
            "data.track_target_override_path": None,
            "data.allow_test_target_override": False,
            "loss.track_target_override_center_l2_weight": 0.0,
            "loss.track_state_aux_target_override_center_l2_weight": 0.0,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = module.build_decision([f"XR-64A:best_track_p10:{path}"], project_root=Path.cwd())

    assert report["ok"] is False
    assert any("data_track_target_override_path must be present" in error for error in report["errors"])
    assert any("data_allow_test_target_override must be present" in error for error in report["errors"])


def test_decision_rejects_type_ambiguous_override_values(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        override_values_by_candidate={
            "XR-64A:best_track_p10": {
                "data_allow_test_target_override": 0,
            },
            "XR-64B:best_track_p10": {
                "loss_track_target_override_center_l2_weight": False,
                "loss_track_state_aux_target_override_center_l2_weight": "0.0",
            },
        },
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is False
    assert report["decision_status"] == "invalid_candidate_evidence"
    assert report["software_promotion_allowed"] is False
    assert any("data_allow_test_target_override must be JSON boolean false" in error for error in report["errors"])
    assert any("loss_track_target_override_center_l2_weight must be JSON numeric 0.0" in error for error in report["errors"])
    assert any(
        "loss_track_state_aux_target_override_center_l2_weight must be JSON numeric 0.0" in error
        for error in report["errors"]
    )


def test_decision_rejects_override_violation_in_non_winning_required_candidate(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        override_values_by_candidate={
            "XR-64B:best_track_p5": {
                "data_track_target_override_path": "data/_internal/manifests/manifest1/xr64_teacher_targets/test_override.json",
            },
        },
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is False
    assert report["decision_status"] == "invalid_candidate_evidence"
    assert report["software_promotion_allowed"] is False
    assert report["winning_candidate"]["lane"] == "XR-64A"
    assert report["winning_candidate"]["checkpoint_kind"] == "best_track_p10"
    assert any("XR-64B:best_track_p5: override-cleared test eval violation" in error for error in report["errors"])


def test_decision_rejects_non_bool_axis_certification(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        axis_certified_by_candidate={
            "XR-64A:best_track_p10": "true",
            "XR-64B:best_track_p10": 1,
        },
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-64A:best_track_p10: axis_certified must be true" in report["errors"]
    assert "XR-64B:best_track_p10: axis_certified must be true" in report["errors"]


def test_decision_rejects_invalid_changed_keys_shape(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        changed_keys_by_candidate={
            "XR-64A:best_track_p10": ["optimizer.lr"],
            "XR-64B:best_track_p10": {"head": ["model.head"]},
        },
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-64A:best_track_p10: ablation_changed_keys must be an object keyed by ablation axis" in report["errors"]
    assert any("XR-64B:best_track_p10: ablation_changed_keys must cover" in error for error in report["errors"])


def test_decision_blocks_unbounded_tradeoff_from_promotion(tmp_path: Path) -> None:
    module = load_decision_module()
    specs: list[str] = []
    for lane in ("XR-64A", "XR-64B"):
        for checkpoint_kind in ("best_metric_track_center_px", "best_track_p10", "best_track_p5"):
            path = tmp_path / lane.lower() / checkpoint_kind / "eval_summary.json"
            write_summary(path, center=17.0, p10=36.0, p5=10.0)
            specs.append(f"{lane}:{checkpoint_kind}:{path}")

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is True
    assert report["decision_status"] == "no_promotion"
    assert report["software_promotion_allowed"] is False
    assert report["single_model_sota_claim_allowed"] is False
    assert report["winning_candidate"] is None
    assert report["unbounded_tradeoff_candidate_count"] == 6
    assert report["bounded_tradeoff_candidate_count"] == 0
    assert all(
        item["tradeoff_classification"] == "diagnostic_unbounded_tradeoff"
        for item in report["gate_tradeoff_table"]
    )
    assert all(set(item["severe_regressions"]) == {"center", "p5"} for item in report["gate_tradeoff_table"])
    assert report["promotion_summary"]["p10_promoted"] is True


def test_decision_allows_single_gate_bounded_tradeoff(tmp_path: Path) -> None:
    module = load_decision_module()
    specs = write_required_matrix(
        tmp_path,
        winning_center=16.55,
        winning_p10=35.5,
        winning_p5=11.8,
    )

    report = module.build_decision(specs, project_root=Path.cwd())

    assert report["ok"] is True
    assert report["decision_status"] == "promoted"
    assert report["software_promotion_allowed"] is True
    assert report["winning_candidate"]["tradeoff_classification"] == "single_gate_bounded_tradeoff"
    assert report["winning_candidate"]["bounded_tradeoff"] is True
    assert report["bounded_tradeoff_candidate_count"] == 6
    assert report["unbounded_tradeoff_candidate_count"] == 0
