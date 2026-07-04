from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def load_collector_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "collect_xr64_postrun_candidates.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("collect_xr64_postrun_candidates", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(
    path: Path,
    *,
    lane: str = "XR-64A",
    checkpoint_kind: str = "best_track_p10",
    center: float = 16.0,
    p10: float = 35.5,
    p5: float = 12.5,
    patched: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": "/tmp/project/data/_internal/manifests/manifest1/test_manifest.jsonl",
        "checkpoint": f"/tmp/project/runs/XR-64/example/train/{checkpoint_kind}.pt",
        "mode": "mode1",
        "stage": "stage2",
        "split": "test",
        "metric_track_center_px": center,
        "metric_track_p10_pct": p10,
        "metric_track_p5_pct": p5,
        "metric_track_p1_pct": 1.0,
        "data_track_target_override_path": None,
        "data_allow_test_target_override": False,
        "loss_track_target_override_center_l2_weight": 0.0,
        "loss_track_state_aux_target_override_center_l2_weight": 0.0,
    }
    if patched:
        payload.update(
            {
                "lane": lane,
                "ablation_checkpoint_kind": checkpoint_kind,
                "ablation_axis": ["head", "loss", "lr", "teacher_model_training"],
                "ablation_changed_keys": {
                    "head": ["model.heads.track_center_heatmap"],
                    "loss": ["loss.track_target_override_center_l2_weight"],
                    "lr": ["training.lr"],
                    "teacher_model_training": ["distillation.teacher_init_checkpoint"],
                },
                "ablation_benchmark_baseline_id": "XR-64-prep",
                "axis_certified": True,
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_required_matrix(root: Path) -> None:
    for lane in ("XR-64A", "XR-64B"):
        for checkpoint_kind in ("best_metric_track_center_px", "best_track_p10", "best_track_p5"):
            center = 16.0 if (lane, checkpoint_kind) == ("XR-64A", "best_track_p10") else 16.6
            p10 = 35.5 if (lane, checkpoint_kind) == ("XR-64A", "best_track_p10") else 35.1
            p5 = 12.5 if (lane, checkpoint_kind) == ("XR-64A", "best_track_p10") else 12.0
            write_summary(
                root / "runs/XR-64" / lane.lower() / checkpoint_kind / "eval/test/eval_summary.json",
                lane=lane,
                checkpoint_kind=checkpoint_kind,
                center=center,
                p10=p10,
                p5=p5,
            )


def test_collection_without_runs_reports_missing_postrun_evidence(tmp_path: Path) -> None:
    module = load_collector_module()

    report = module.build_collection(
        project_root=Path.cwd(),
        runs_root=tmp_path / "missing_runs",
        lanes={"XR-64A", "XR-64B"},
        checkpoint_kinds={"best_track_p10", "best_track_p5", "best_metric_track_center_px"},
        include_all=False,
    )

    assert report["ok"] is True
    assert report["raw_eval_summary_count"] == 0
    assert report["candidate_count"] == 0
    assert report["decision"]["decision_status"] == "missing_postrun_evidence"


def test_collection_discovers_patched_a_b_candidates(tmp_path: Path) -> None:
    module = load_collector_module()
    write_required_matrix(tmp_path)

    report = module.build_collection(
        project_root=Path.cwd(),
        runs_root=tmp_path / "runs",
        lanes={"XR-64A", "XR-64B"},
        checkpoint_kinds={"best_track_p10", "best_track_p5", "best_metric_track_center_px"},
        include_all=False,
    )

    assert report["ok"] is True
    assert report["candidate_count"] == 6
    assert report["decision"]["decision_status"] == "promoted"
    assert report["decision"]["axis_claims_allowed"] is True
    assert all("eval_summary.json" in spec for spec in report["candidate_specs"])


def test_collection_uses_latest_per_lane_checkpoint_by_default(tmp_path: Path) -> None:
    module = load_collector_module()
    older = tmp_path / "runs/XR-64/old/eval/test/eval_summary.json"
    newer = tmp_path / "runs/XR-64/new/eval/test/eval_summary.json"
    b = tmp_path / "runs/XR-64/b/eval/test/eval_summary.json"
    write_summary(older, lane="XR-64A", checkpoint_kind="best_track_p10", center=16.4)
    write_summary(newer, lane="XR-64A", checkpoint_kind="best_track_p10", center=16.0)
    write_summary(b, lane="XR-64B", checkpoint_kind="best_track_p10", center=16.6, p10=35.1, p5=12.0)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(b, (3, 3))

    report = module.build_collection(
        project_root=Path.cwd(),
        runs_root=tmp_path / "runs",
        lanes={"XR-64A", "XR-64B"},
        checkpoint_kinds={"best_track_p10"},
        include_all=False,
    )

    assert report["discovered_candidate_count"] == 3
    assert report["candidate_count"] == 2
    assert report["ok"] is False
    assert report["decision"]["decision_status"] == "invalid_candidate_evidence"
    assert report["decision"]["missing_required_candidate_count"] == 4
    assert str(newer.resolve()) in report["candidate_specs"][0]
    assert str(older.resolve()) not in "\n".join(report["candidate_specs"])


def test_collection_skips_unpatched_eval_summary(tmp_path: Path) -> None:
    module = load_collector_module()
    summary = tmp_path / "runs/XR-64/unpatched/eval/test/eval_summary.json"
    write_summary(summary, patched=False)

    report = module.build_collection(
        project_root=Path.cwd(),
        runs_root=tmp_path / "runs",
        lanes={"XR-64A", "XR-64B"},
        checkpoint_kinds={"best_track_p10"},
        include_all=False,
    )

    assert report["candidate_count"] == 0
    assert report["skipped_reasons"]["lane_not_selected_or_missing"] == 1


def test_format_commands_outputs_decision_helper_invocation(tmp_path: Path) -> None:
    module = load_collector_module()
    write_required_matrix(tmp_path)
    report = module.build_collection(
        project_root=Path.cwd(),
        runs_root=tmp_path / "runs",
        lanes={"XR-64A", "XR-64B"},
        checkpoint_kinds={"best_track_p10", "best_track_p5", "best_metric_track_center_px"},
        include_all=False,
    )

    command = module.format_commands(report)

    assert command.startswith(".venv/bin/python scripts/external/decide_xr64_postrun_promotion.py")
    assert "--candidate XR-64A:best_track_p10:" in command
    assert "--candidate XR-64B:best_track_p10:" in command
    assert "--candidate XR-64A:best_metric_track_center_px:" in command
    assert "--candidate XR-64B:best_metric_track_center_px:" in command
