from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_followup_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_post_xr64_followup_contract.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_post_xr64_followup_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def experiment(contract: dict, exp_id: str) -> dict:
    return next(item for item in contract["followup_experiments"] if item["id"] == exp_id)


def test_current_followup_contract_passes() -> None:
    module = load_followup_module()
    report = module.validate_contract(read_current_contract(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["execute_supported"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["experiment_count"] == 5
    assert report["experiment_ids"] == ["XR-64C", "XR-65", "XR-66", "XR-67", "XR-68"]
    assert report["blocked_launch_count"] == 5
    assert report["p1_count"] == 3
    assert report["p2_count"] == 2
    assert report["minimal_next_worker_after_resume"] == "XR-64-prep"
    assert report["xr64_postrun_candidate_count"] == 0
    assert report["followup_launch_allowed_now"] is False


def test_followup_contract_rejects_launch_allowed_while_paused() -> None:
    module = load_followup_module()
    contract = copy.deepcopy(read_current_contract())
    experiment(contract, "XR-65")["launch_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-65: launch_allowed must be false" in report["errors"]


def test_followup_contract_rejects_promotion_allowed_without_evidence() -> None:
    module = load_followup_module()
    contract = copy.deepcopy(read_current_contract())
    experiment(contract, "XR-68")["promotion_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-68: promotion_allowed must be false" in report["errors"]


def test_followup_contract_rejects_validation_only_xr64c() -> None:
    module = load_followup_module()
    contract = copy.deepcopy(read_current_contract())
    experiment(contract, "XR-64C")["reject_if"] = ["test-derived target selection"]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-64C must reject validation-only promotion" in report["errors"]


def test_followup_contract_rejects_temporal_without_same_scene_evidence() -> None:
    module = load_followup_module()
    contract = copy.deepcopy(read_current_contract())
    experiment(contract, "XR-65")["required_result_evidence"] = ["full-test center/P10/P5/P1"]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-65 must require same-scene temporal comparison" in report["errors"]


def test_followup_contract_rejects_xr66_training_without_diagnostic() -> None:
    module = load_followup_module()
    contract = copy.deepcopy(read_current_contract())
    experiment(contract, "XR-66")["launch_gate"]["requires_no_train_first"] = False

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-66 must require no-train-first diagnostic" in report["errors"]


def test_followup_contract_rejects_dense_branch_without_dense_segments() -> None:
    module = load_followup_module()
    contract = copy.deepcopy(read_current_contract())
    experiment(contract, "XR-67")["launch_gate"]["requires_dense_or_continuous_segments"] = False

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-67 must require dense or continuous segments" in report["errors"]


def test_followup_contract_rejects_hardware_branch_without_stronger_teacher() -> None:
    module = load_followup_module()
    contract = copy.deepcopy(read_current_contract())
    experiment(contract, "XR-68")["reject_if"] = ["student accuracy tolerance is undefined"]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-68 must reject weak teacher hardware branch" in report["errors"]


def test_followup_contract_rejects_train_command_in_validation_checks() -> None:
    module = load_followup_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"].append("bash scripts/external/run_xr64_teacher_target_construction.sh")

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("required_checks must not execute" in error for error in report["errors"])
