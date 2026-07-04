from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_runbook_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_paper_target_bridge_ptb1_resume_runbook.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_paper_target_bridge_ptb1_resume_runbook", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_runbook() -> dict:
    path = Path("docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def copy_source_artifacts(runbook: dict, project: Path) -> None:
    for raw_path in runbook["source_artifacts"].values():
        path = Path(raw_path)
        if not path.suffix:
            continue
        source = Path.cwd() / path
        if not source.exists():
            continue
        target = project / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_current_ptb1_resume_runbook_passes() -> None:
    module = load_runbook_module()

    report = module.validate_runbook(read_current_runbook(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execute_supported"] is False
    assert report["allowed_while_paused"] is True
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["sequence_count"] == 5
    assert report["ptb1_output_count"] == 5
    assert report["ptb1_current_existing_output_count"] == 0
    assert report["ptb1_payload_missing_count"] == 5
    assert report["future_payload_validation_requires_allow_present"] is True


def test_runbook_rejects_sequence_reorder() -> None:
    module = load_runbook_module()
    runbook = copy.deepcopy(read_current_runbook())
    runbook["resume_sequence"][0], runbook["resume_sequence"][1] = (
        runbook["resume_sequence"][1],
        runbook["resume_sequence"][0],
    )

    report = module.validate_runbook(runbook, project_root=Path.cwd())

    assert report["ok"] is False
    assert "resume_sequence must preserve required PTB-1 generation order" in report["errors"]


def test_runbook_rejects_paused_payload_creation(tmp_path: Path) -> None:
    module = load_runbook_module()
    runbook = copy.deepcopy(read_current_runbook())
    copy_source_artifacts(runbook, tmp_path)
    payload_path = tmp_path / "docs/resources/future/paper_target_bridge/coordinate_transform_audit.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text("{}", encoding="utf-8")

    report = module.validate_runbook(runbook, project_root=tmp_path)

    assert report["ok"] is False
    assert report["ptb1_current_existing_output_count"] == 1
    assert report["ptb1_payload_missing_count"] == 4
    assert any("coordinate_transform_audit payload must be absent while paused" in error for error in report["errors"])


def test_runbook_rejects_allow_present_missing_from_future_validator() -> None:
    module = load_runbook_module()
    runbook = copy.deepcopy(read_current_runbook())
    runbook["post_resume_validation"]["future_payload_validation"] = (
        ".venv/bin/python scripts/external/check_paper_target_bridge_payloads.py --format summary"
    )

    report = module.validate_runbook(runbook, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("future_payload_validation" in error for error in report["errors"])


def test_runbook_rejects_train_eval_validation_command() -> None:
    module = load_runbook_module()
    runbook = copy.deepcopy(read_current_runbook())
    runbook["validation_commands"].append(".venv/bin/python eval_hbtxr.py")

    report = module.validate_runbook(runbook, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])
