from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_contract_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_paper_target_bridge_decision_contract.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_paper_target_bridge_decision_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/paper_target_bridge_decision_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def copy_source_artifacts(root: Path, contract: dict) -> None:
    for raw_path in contract["source_artifacts"].values():
        source = Path(raw_path)
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def output_path(root: Path, contract: dict) -> Path:
    return root / contract["bridge_decision_output"]["path"]


def valid_blocked_payload() -> dict:
    return {
        "direct_submission_comparison_allowed": False,
        "paper_level_completion_allowed": False,
        "required_evidence_complete": False,
        "metric_protocol_unblocked": False,
        "trained_candidate_valid": False,
        "remaining_blockers": ["paper_target_required_evidence_incomplete"],
        "decision": "blocked",
        "paper_target_manifest_path": "docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.json",
        "metric_protocol_unblock_contract_path": "docs/resources/metric_protocol_unblock_contract_2026_06_21.json",
        "trained_candidate_eval_summary_path": "docs/resources/future/paper_target_bridge/trained_candidate_test_eval_summary.json",
        "trained_candidate_leakage_audit_path": "docs/resources/future/paper_target_bridge/trained_candidate_leakage_audit.json",
        "ptb1_payload_paths": [],
        "source_status_report_path": "docs/resources/future/paper_target_bridge/status_summary.txt",
    }


def test_current_bridge_decision_contract_passes() -> None:
    module = load_contract_module()

    report = module.validate_contract(read_current_contract(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["schema_contract_allowed_while_paused"] is True
    assert report["bridge_decision_payload_allowed_while_paused"] is False
    assert report["execute_supported"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["decision_output_count"] == 1
    assert report["json_output_schema_count"] == 1
    assert report["blocked_output_count"] == 1
    assert report["current_existing_output_count"] == 0
    assert report["signoff_rule_count"] == 6


def test_contract_rejects_direct_comparison_claim() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["direct_submission_comparison_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "direct_submission_comparison_allowed must be false" in report["errors"]


def test_contract_rejects_payload_allowed_while_paused() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["bridge_decision_payload_allowed_while_paused"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "bridge_decision_payload_allowed_while_paused must be false" in report["errors"]


def test_contract_rejects_present_payload_while_paused(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    write_json(output_path(tmp_path, contract), valid_blocked_payload())

    report = module.validate_contract(contract, project_root=tmp_path)

    assert report["ok"] is False
    assert report["current_existing_output_count"] == 1
    assert any("output path must be absent while paused" in error for error in report["errors"])


def test_contract_validates_future_blocked_payload(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    write_json(output_path(tmp_path, contract), valid_blocked_payload())

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_payload_present=True,
    )

    assert report["ok"] is True
    assert report["current_existing_output_count"] == 1


def test_contract_rejects_direct_comparison_with_blockers(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    payload = valid_blocked_payload()
    payload["direct_submission_comparison_allowed"] = True
    payload["required_evidence_complete"] = True
    payload["metric_protocol_unblocked"] = True
    payload["trained_candidate_valid"] = True
    payload["decision"] = "allow"
    write_json(output_path(tmp_path, contract), payload)

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_payload_present=True,
    )

    assert report["ok"] is False
    assert "bridge_decision direct comparison requires empty remaining_blockers" in report["errors"]


def test_contract_rejects_missing_context_field(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    payload = valid_blocked_payload()
    payload.pop("source_status_report_path")
    write_json(output_path(tmp_path, contract), payload)

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_payload_present=True,
    )

    assert report["ok"] is False
    assert any("payload missing context fields" in error for error in report["errors"])
