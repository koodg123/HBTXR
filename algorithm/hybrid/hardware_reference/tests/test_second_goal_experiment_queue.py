from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_queue_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_experiment_queue.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_experiment_queue", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_queue() -> dict:
    path = Path("docs/resources/second_goal_experiment_queue_2026_06_20.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_second_goal_queue_passes() -> None:
    module = load_queue_module()
    report = module.validate_queue(read_current_queue())

    assert report["ok"] is True
    assert report["experiment_count"] == 8
    assert report["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert report["paused"] is True
    assert report["do_not_execute_until_user_resumes"] is True
    assert "teacher_model_training" in report["covered_axes"]


def test_queue_checker_rejects_unpaused_execution() -> None:
    module = load_queue_module()
    queue = copy.deepcopy(read_current_queue())
    queue["do_not_execute_until_user_resumes"] = False

    report = module.validate_queue(queue)

    assert report["ok"] is False
    assert "do_not_execute_until_user_resumes must be true" in report["errors"]


def test_queue_checker_rejects_stale_p0_item() -> None:
    module = load_queue_module()
    queue = copy.deepcopy(read_current_queue())
    queue["queue"][0]["id"] = "XR-60"
    queue["queue"][0]["objective"] = "candidate-head-only replay"

    report = module.validate_queue(queue)

    assert report["ok"] is False
    assert any("experiment order" in error for error in report["errors"])
    assert any("stale marker" in error for error in report["errors"])
