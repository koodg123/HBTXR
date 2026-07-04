from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_synthesis_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_current_result_synthesis.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_current_result_synthesis", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_synthesis() -> dict:
    path = Path("docs/resources/second_goal_current_result_synthesis_2026_06_20.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_result_synthesis_passes() -> None:
    module = load_synthesis_module()
    report = module.validate_synthesis(read_current_synthesis(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["active_goal_complete"] is False
    assert report["center_best"] == 16.468481131962367
    assert report["p10_best"] == 35.02295998845781
    assert report["p5_best"] == 12.133503770828247
    assert report["oracle_promotable"] is False
    assert report["direct_submission_comparison_valid"] is False
    assert report["xr64_current_status"] == "generated_incomplete"
    assert report["xr64_ready_to_train"] is False
    assert report["xr64_missing_eval_rows"] == 8
    assert report["xr64_missing_overrides"] == 6


def test_synthesis_rejects_oracle_promotion() -> None:
    module = load_synthesis_module()
    synthesis = copy.deepcopy(read_current_synthesis())
    synthesis["oracle_diagnostic"]["promotable_as_result"] = True

    report = module.validate_synthesis(synthesis, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-63 oracle must not be promotable as a trained result" in report["errors"]


def test_synthesis_rejects_direct_submission_comparison() -> None:
    module = load_synthesis_module()
    synthesis = copy.deepcopy(read_current_synthesis())
    synthesis["submission_target_relation"]["direct_comparison_valid"] = True

    report = module.validate_synthesis(synthesis, project_root=Path.cwd())

    assert report["ok"] is False
    assert "direct comparison with submission target must remain false" in report["errors"]
