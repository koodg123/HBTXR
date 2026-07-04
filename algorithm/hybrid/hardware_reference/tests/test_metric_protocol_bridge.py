from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_bridge_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_metric_protocol_bridge.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_metric_protocol_bridge", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_bridge() -> dict:
    path = Path("docs/resources/second_goal_metric_protocol_bridge_2026_06_20.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_metric_protocol_bridge_passes() -> None:
    module = load_bridge_module()

    report = module.validate_bridge(read_current_bridge(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["bridge_status"] == "blocked_until_metric_frame_and_protocol_match"
    assert report["direct_submission_comparison_allowed"] is False
    assert report["metric_frame"] == "post-transform input coordinate frame"
    assert report["directly_sensor_px"] is False
    assert report["hybrid_match_status"] == "not_proven"
    assert report["current_p1_metric_available"] is True
    assert report["current_p1_full_test_evidence_available"] is False
    assert report["xr64_current_ready_to_train"] is False


def test_metric_protocol_bridge_rejects_direct_comparison() -> None:
    module = load_bridge_module()
    bridge = copy.deepcopy(read_current_bridge())
    bridge["direct_submission_comparison_allowed"] = True

    report = module.validate_bridge(bridge, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("direct_submission_comparison_allowed must be false" in error for error in report["errors"])


def test_metric_protocol_bridge_rejects_hybrid_match_without_evidence() -> None:
    module = load_bridge_module()
    bridge = copy.deepcopy(read_current_bridge())
    bridge["hybrid_protocol_match"]["match_status"] = "matched"
    bridge["hybrid_protocol_match"]["current_p1_full_test_evidence_available"] = True

    report = module.validate_bridge(bridge, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("hybrid match status must remain not_proven" in error for error in report["errors"])
    assert any("current_p1_full_test_evidence_available must be false" in error for error in report["errors"])
