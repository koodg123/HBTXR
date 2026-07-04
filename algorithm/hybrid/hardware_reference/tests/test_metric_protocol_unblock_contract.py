from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_unblock_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_metric_protocol_unblock_contract.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_metric_protocol_unblock_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/metric_protocol_unblock_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_metric_protocol_unblock_contract_passes() -> None:
    module = load_unblock_module()

    report = module.validate_contract(read_current_contract(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["contract_status"] == "blocked"
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["required_evidence_count"] == 7
    assert "p1_metric_coverage" in report["required_evidence_ids"]
    assert "hybrid_scheduler_eval" in report["required_evidence_ids"]
    assert "coordinate_frame_match" in report["required_evidence_ids"]


def test_unblock_contract_rejects_direct_comparison_claim() -> None:
    module = load_unblock_module()
    contract = copy.deepcopy(read_current_contract())
    contract["direct_submission_comparison_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("direct_submission_comparison_allowed must be false" in error for error in report["errors"])


def test_unblock_contract_rejects_missing_p1_requirement() -> None:
    module = load_unblock_module()
    contract = copy.deepcopy(read_current_contract())
    contract["required_evidence"] = [
        item for item in contract["required_evidence"] if item["id"] != "p1_metric_coverage"
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("required_evidence ids mismatch" in error for error in report["errors"])


def test_unblock_contract_rejects_train_eval_validation_command() -> None:
    module = load_unblock_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0")

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])


def test_unblock_contract_rejects_paper_completion_without_direct_comparison() -> None:
    module = load_unblock_module()
    contract = copy.deepcopy(read_current_contract())
    contract["paper_level_completion_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("paper_level_completion_allowed must be false" in error for error in report["errors"])
