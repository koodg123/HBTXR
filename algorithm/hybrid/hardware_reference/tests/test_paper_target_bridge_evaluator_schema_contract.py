from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_contract_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_paper_target_bridge_evaluator_schema_contract.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_paper_target_bridge_evaluator_schema_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_evaluator_schema_contract_passes() -> None:
    module = load_contract_module()

    report = module.validate_contract(read_current_contract(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execute_supported"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["evaluator_output_count"] == 5
    assert report["jsonl_output_schema_count"] == 2
    assert report["json_output_schema_count"] == 3
    assert report["blocked_output_count"] == 5
    assert report["current_existing_output_count"] == 0


def test_contract_rejects_direct_submission_comparison_claim() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["direct_submission_comparison_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "direct_submission_comparison_allowed must be false" in report["errors"]


def test_contract_rejects_missing_evaluator_output() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["evaluator_outputs"] = [
        output for output in contract["evaluator_outputs"] if output["id"] != "hybrid_eval_rows"
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("evaluator_outputs ids mismatch" in error for error in report["errors"])


def test_contract_rejects_field_mismatch_against_manifest_schema() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    for output in contract["evaluator_outputs"]:
        if output["id"] == "paper_frame_full_test_p1":
            output["required_fields"].remove("paper_frame_match")

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("paper_frame_full_test_p1.required_fields" in error for error in report["errors"])


def test_contract_rejects_existing_ptb1_output_while_paused(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    project = tmp_path

    # Copy source artifacts referenced by the contract into an isolated project root.
    for source in contract["source_artifacts"].values():
        source_path = Path(source)
        target = project / source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    output_path = project / "docs/resources/future/paper_target_bridge/sensor_space_eval_rows.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('{"sample_id": "s0"}\n', encoding="utf-8")

    report = module.validate_contract(contract, project_root=project)

    assert report["ok"] is False
    assert report["current_existing_output_count"] == 1
    assert any("sensor_space_eval_rows output path must be absent while paused" in error for error in report["errors"])


def test_contract_rejects_train_eval_validation_command() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_commands"].append(".venv/bin/python eval_hbtxr.py")

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])


def test_contract_rejects_missing_payload_checker_command() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_commands"] = [
        command for command in contract["validation_commands"] if "check_paper_target_bridge_payloads.py" not in command
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "validation_commands must include PTB-1 payload checker" in report["errors"]


def test_contract_rejects_missing_payload_checker_test_command() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_commands"] = [
        command for command in contract["validation_commands"] if "test_paper_target_bridge_payloads.py" not in command
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "validation_commands must include PTB-1 payload tests" in report["errors"]
