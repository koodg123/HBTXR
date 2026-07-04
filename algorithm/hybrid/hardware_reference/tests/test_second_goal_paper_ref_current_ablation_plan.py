from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_plan_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_paper_ref_current_ablation_plan.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_paper_ref_current_ablation_plan", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def current_plan_text() -> str:
    return Path("docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md").read_text(
        encoding="utf-8"
    )


def test_current_paper_ref_ablation_plan_passes() -> None:
    module = load_plan_module()
    report = module.validate_plan_text(current_plan_text())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["execute_supported"] is False
    assert report["allowed_while_paused"] is True
    assert report["experiment_count"] == 8
    assert report["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert report["paper_signal_count"] == 18
    assert report["axis_count"] == 8
    assert report["task_card_count"] >= 2
    assert report["completion_allowed"] is False
    assert report["direct_submission_comparison_allowed"] is False


def test_plan_checker_rejects_missing_skill_prompt_pack() -> None:
    module = load_plan_module()
    text = current_plan_text().replace("IDEA-Gen phase: P1", "IDEA-Gen phase: P2")

    report = module.validate_plan_text(text)

    assert report["ok"] is False
    assert "prompt pack must state IDEA-Gen phase P1" in report["errors"]


def test_plan_checker_rejects_missing_xr64_p0_lane() -> None:
    module = load_plan_module()
    text = current_plan_text().replace("| XR-64B | P0 |", "| XR-64B | P1 |", 1)

    report = module.validate_plan_text(text)

    assert report["ok"] is False
    assert "ablation matrix P0 IDs must be ['XR-64-prep', 'XR-64A', 'XR-64B']" in report["errors"]
