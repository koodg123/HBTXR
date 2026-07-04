from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_decision_tree_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_post_xr64_decision_tree.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_post_xr64_decision_tree", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_tree() -> dict:
    path = Path("docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_post_xr64_decision_tree_passes() -> None:
    module = load_decision_tree_module()
    report = module.validate_decision_tree(read_current_tree(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["execute_supported"] is False
    assert report["claim_level_count"] == 4
    assert report["decision_input_count"] == 3
    assert report["branch_count"] == 7
    assert report["minimal_next_worker_after_resume"] == "XR-64-prep"
    assert report["current_next_prep_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert report["missing_eval_rows"] == 8
    assert report["missing_overrides"] == 6
    assert report["direct_submission_comparison_allowed"] is False
    assert report["next_experiments"] == ["XR-64C", "XR-65", "XR-66", "XR-67", "XR-68"]


def test_decision_tree_rejects_executable_artifact() -> None:
    module = load_decision_tree_module()
    tree = copy.deepcopy(read_current_tree())
    tree["execute_supported"] = True

    report = module.validate_decision_tree(tree, project_root=Path.cwd())

    assert report["ok"] is False
    assert "execute_supported must be False" in report["errors"]


def test_decision_tree_rejects_bypassing_xr64_prep() -> None:
    module = load_decision_tree_module()
    tree = copy.deepcopy(read_current_tree())
    tree["minimal_next_worker_after_resume"] = "XR-65"

    report = module.validate_decision_tree(tree, project_root=Path.cwd())

    assert report["ok"] is False
    assert "minimal_next_worker_after_resume must remain XR-64-prep" in report["errors"]


def test_decision_tree_rejects_missing_validation_only_guard() -> None:
    module = load_decision_tree_module()
    tree = copy.deepcopy(read_current_tree())
    branch = next(item for item in tree["branches"] if item["result_pattern"] == "validation_gain_without_full_test_gate")
    branch["reject_if"] = ["test-derived selection is used"]

    report = module.validate_decision_tree(tree, project_root=Path.cwd())

    assert report["ok"] is False
    assert "validation_gain_without_full_test_gate must reject validation-only promotion" in report["errors"]


def test_decision_tree_rejects_paper_completion_without_bridge() -> None:
    module = load_decision_tree_module()
    tree = copy.deepcopy(read_current_tree())
    paper_level = next(item for item in tree["claim_levels"] if item["level"] == "paper_level_completion")
    paper_level["allowed_evidence"] = ["required paper-target evidence present"]

    report = module.validate_decision_tree(tree, project_root=Path.cwd())

    assert report["ok"] is False
    assert "paper_level_completion must require metric/protocol bridge completion" in report["errors"]
