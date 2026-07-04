from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_workplan_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_paper_target_bridge_artifact_workplan.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_paper_target_bridge_artifact_workplan", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_workplan() -> dict:
    path = Path("docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_paper_target_bridge_artifact_workplan_passes() -> None:
    module = load_workplan_module()

    report = module.validate_workplan(read_current_workplan(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execute_supported"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["work_package_count"] == 4
    assert report["future_artifact_count"] == 10
    assert report["allowed_while_paused_package_count"] == 2
    assert report["doc_prefill_artifact_count"] == 3
    assert report["current_existing_future_artifact_count"] == 3
    assert report["trained_candidate_artifact_count"] == 2


def test_workplan_rejects_execute_supported_while_paused() -> None:
    module = load_workplan_module()
    workplan = copy.deepcopy(read_current_workplan())
    workplan["execute_supported"] = True

    report = module.validate_workplan(workplan, project_root=Path.cwd())

    assert report["ok"] is False
    assert "execute_supported must be false" in report["errors"]


def test_workplan_rejects_direct_submission_comparison_claim() -> None:
    module = load_workplan_module()
    workplan = copy.deepcopy(read_current_workplan())
    workplan["direct_submission_comparison_allowed"] = True

    report = module.validate_workplan(workplan, project_root=Path.cwd())

    assert report["ok"] is False
    assert "direct_submission_comparison_allowed must be false" in report["errors"]


def test_workplan_rejects_trained_candidate_allowed_while_paused() -> None:
    module = load_workplan_module()
    workplan = copy.deepcopy(read_current_workplan())
    for package in workplan["work_packages"]:
        if package["id"] == "PTB-2-XR64-TRAINED-CANDIDATE":
            package["allowed_while_paused"] = True

    report = module.validate_workplan(workplan, project_root=Path.cwd())

    assert report["ok"] is False
    assert "PTB-2-XR64-TRAINED-CANDIDATE must be blocked while paused and require train/eval" in report["errors"]


def test_workplan_rejects_missing_artifact_mapping() -> None:
    module = load_workplan_module()
    workplan = copy.deepcopy(read_current_workplan())
    workplan["artifact_to_work_package"].pop("hybrid_eval_rows")

    report = module.validate_workplan(workplan, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("artifact_to_work_package ids mismatch" in error for error in report["errors"])


def test_workplan_rejects_train_eval_validation_command() -> None:
    module = load_workplan_module()
    workplan = copy.deepcopy(read_current_workplan())
    workplan["validation_commands"].append(".venv/bin/python eval_hbtxr.py")

    report = module.validate_workplan(workplan, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])
