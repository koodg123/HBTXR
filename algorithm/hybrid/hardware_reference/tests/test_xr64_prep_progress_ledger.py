from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_checker_module():
    script_dir = Path("scripts/external").resolve()
    sys.path.insert(0, str(script_dir))
    script = script_dir / "check_xr64_prep_progress_ledger.py"
    spec = importlib.util.spec_from_file_location("check_xr64_prep_progress_ledger", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_ledger() -> dict:
    return json.loads(Path("docs/resources/xr64_prep_progress_ledger_2026_06_21.json").read_text(encoding="utf-8"))


def test_current_xr64_prep_progress_ledger_passes() -> None:
    module = load_checker_module()
    report = module.validate_ledger(read_ledger(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["prep_unit_count"] == 10
    assert report["ready_after_resume_count"] == 8
    assert report["blocked_count"] == 2
    assert report["next_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert report["missing_eval_rows"] == 8
    assert report["missing_overrides"] == 6


def test_ledger_rejects_count_drift() -> None:
    module = load_checker_module()
    ledger = read_ledger()
    ledger["counts"]["ready_after_resume"] = 7

    report = module.validate_ledger(ledger, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("counts.ready_after_resume" in error for error in report["errors"])


def test_ledger_rejects_allowed_to_run_now() -> None:
    module = load_checker_module()
    ledger = read_ledger()
    ledger["allowed_to_run_now"] = True
    ledger["prep_units"][0]["allowed_to_run_now"] = True

    report = module.validate_ledger(ledger, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("allowed_to_run_now" in error for error in report["errors"])


def test_ledger_rejects_unit_status_drift() -> None:
    module = load_checker_module()
    ledger = read_ledger()
    ledger["prep_units"][0]["status"] = "completed"

    report = module.validate_ledger(ledger, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("XR64-EVAL-TRAIN-XR62A.status" in error for error in report["errors"])


def test_ledger_rejects_test_split_unit() -> None:
    module = load_checker_module()
    ledger = copy.deepcopy(read_ledger())
    ledger["prep_units"][0]["split"] = "test"

    report = module.validate_ledger(ledger, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("split must not be test" in error for error in report["errors"])


def test_ledger_rejects_launch_validation_command() -> None:
    module = load_checker_module()
    ledger = read_ledger()
    ledger["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0")

    report = module.validate_ledger(ledger, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("training runner" in error for error in report["errors"])
