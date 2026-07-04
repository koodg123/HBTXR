from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_contract_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_xr64_experiment_design_contract.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_xr64_experiment_design_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/xr64_experiment_design_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_xr64_experiment_design_contract_passes() -> None:
    module = load_contract_module()
    report = module.validate_contract(read_current_contract())

    assert report["ok"] is True
    assert report["lane_count"] == 4
    assert report["lane_ids"] == ["XR-64-prep", "XR-64A", "XR-64B", "XR-64C"]
    assert report["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert report["allowed_while_paused"] is True
    assert report["execute_supported"] is False
    assert report["current_ready_to_train"] is False
    assert report["current_missing_eval_rows"] == 8
    assert report["current_missing_overrides"] == 6


def test_contract_rejects_executable_state() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["execute_supported"] = True
    contract["lanes"][1]["launch_allowed_now"] = True

    report = module.validate_contract(contract)

    assert report["ok"] is False
    assert "execute_supported must be false" in report["errors"]
    assert "XR-64A: launch_allowed_now must be false" in report["errors"]


def test_contract_rejects_gpu_or_lr_drift() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["lanes"][2]["gpu"] = "cuda:0"
    contract["lanes"][2]["hyperparameters"]["lr"] = "1e-4"

    report = module.validate_contract(contract)

    assert report["ok"] is False
    assert "XR-64B: gpu must be cuda:1" in report["errors"]
    assert "XR-64B: lr must be 5e-7" in report["errors"]


def test_contract_rejects_missing_prep_artifact_counts() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["global_controls"]["expected_eval_row_files"] = 7
    contract["lanes"][0]["expected_override_json_files"] = 5

    report = module.validate_contract(contract)

    assert report["ok"] is False
    assert "expected_eval_row_files must be 8" in report["errors"]
    assert "XR-64-prep expected_override_json_files must be 6" in report["errors"]
