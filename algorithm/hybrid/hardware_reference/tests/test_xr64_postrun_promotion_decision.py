from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_checker_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_xr64_postrun_promotion_decision.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_xr64_postrun_promotion_decision", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_decision() -> dict:
    path = Path("docs/resources/xr64_postrun_promotion_decision_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_xr64_postrun_promotion_decision_artifact_passes() -> None:
    module = load_checker_module()

    report = module.validate_decision(read_current_decision(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["decision_status"] == "missing_postrun_evidence"
    assert report["candidate_count"] == 0
    assert report["software_promotion_allowed"] is False
    assert report["single_model_sota_claim_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["bounded_tradeoff_candidate_count"] == 0
    assert report["unbounded_tradeoff_candidate_count"] == 0
    assert report["missing_required_candidate_count"] == 6


def test_promotion_decision_rejects_premature_software_promotion() -> None:
    module = load_checker_module()
    decision = copy.deepcopy(read_current_decision())
    decision["software_promotion_allowed"] = True
    decision["decision_status"] = "promoted"

    report = module.validate_decision(decision, project_root=Path.cwd())

    assert report["ok"] is False
    assert "decision_status must be missing_postrun_evidence" in report["errors"]
    assert "software_promotion_allowed must be false" in report["errors"]
    assert any("does not match decision helper no-candidate output" in error for error in report["errors"])


def test_promotion_decision_rejects_changed_tradeoff_floors() -> None:
    module = load_checker_module()
    decision = copy.deepcopy(read_current_decision())
    decision["tradeoff_floors"]["center"] = -1.0

    report = module.validate_decision(decision, project_root=Path.cwd())

    assert report["ok"] is False
    assert "tradeoff_floors do not match bounded tradeoff scorecard" in report["errors"]
    assert "tradeoff_floors does not match decision helper no-candidate output" in report["errors"]


def test_promotion_decision_rejects_missing_candidate_matrix_drift() -> None:
    module = load_checker_module()
    decision = copy.deepcopy(read_current_decision())
    decision["missing_required_candidate_count"] = 5
    decision["missing_required_candidates"] = decision["missing_required_candidates"][:-1]

    report = module.validate_decision(decision, project_root=Path.cwd())

    assert report["ok"] is False
    assert "missing_required_candidate_count must be 6" in report["errors"]
    assert "missing_required_candidates must match XR-64A/B x 3 checkpoint matrix" in report["errors"]


def test_promotion_decision_rejects_override_clear_schema_drift() -> None:
    module = load_checker_module()
    decision = copy.deepcopy(read_current_decision())
    field = decision["required_override_cleared_fields"]["data_allow_test_target_override"]
    field["required_value"] = 0
    field["required_json_type"] = "numeric_zero"

    report = module.validate_decision(decision, project_root=Path.cwd())

    assert report["ok"] is False
    assert "required_override_cleared_fields.data_allow_test_target_override.required_value mismatch" in report["errors"]
    assert (
        "required_override_cleared_fields.data_allow_test_target_override.required_json_type mismatch"
        in report["errors"]
    )
