from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_contract_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_xr64_postrun_evidence_contract.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_xr64_postrun_evidence_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/xr64_postrun_evidence_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_xr64_postrun_evidence_contract_passes() -> None:
    module = load_contract_module()

    report = module.validate_contract(read_current_contract(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["evidence_state"] == "missing_postrun_evidence"
    assert report["allowed_to_claim_promotion"] is False
    assert report["allowed_to_mark_second_goal_complete"] is False
    assert report["required_lane_count"] == 2
    assert report["required_checkpoint_kind_count"] == 3
    assert report["required_metric_count"] == 4
    assert report["ablation_axis_count"] == 4
    assert report["ablation_lane_axis_count"] == 2
    assert report["xr64a_eval_summary_count"] == 0
    assert report["xr64b_eval_summary_count"] == 0
    assert report["decision_status"] == "pending_postrun_evidence"


def test_postrun_contract_rejects_premature_promotion() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["allowed_to_claim_promotion"] = True
    contract["decision_template"]["software_promotion_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "allowed_to_claim_promotion must be false" in report["errors"]
    assert "decision_template.software_promotion_allowed must be false" in report["errors"]


def test_postrun_contract_rejects_test_override_path() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["eval_contract"]["required_overrides"]["data.track_target_override_path"] = "data/_internal/manifests/manifest1/xr64_teacher_targets/test_overrides.json"

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "eval_contract.required_overrides do not match override-clear contract" in report["errors"]


def test_postrun_contract_rejects_missing_required_lane() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["required_lanes"] = contract["required_lanes"][:1]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "required_lanes must be exactly XR-64A and XR-64B" in report["errors"]


def test_postrun_contract_rejects_missing_p1_required_metric() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["required_metrics"] = [
        metric for metric in contract["required_metrics"] if metric != "metric_track_p1_pct"
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "required_metrics must be center/P10/P5/P1 track metrics" in report["errors"]


def test_postrun_contract_rejects_missing_ablation_axis_mapping() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["ablation_evidence_contract"]["lane_axis_map"]["XR-64A"].pop("loss")

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "ablation_evidence_contract.lane_axis_map.XR-64A must cover all required axes" in report["errors"]


def test_postrun_contract_rejects_missing_lane_ablation_design() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["required_lanes"][0]["ablation_design"]["axis_tested"].remove("lr")

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-64A.ablation_design.axis_tested must cover required axes" in report["errors"]


def test_postrun_contract_rejects_missing_teacher_provenance_requirement() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["required_lanes"][1]["teacher_provenance_required"]["teacher_checkpoint_sha"] = False

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-64B.teacher_provenance_required.teacher_checkpoint_sha must be true" in report["errors"]
