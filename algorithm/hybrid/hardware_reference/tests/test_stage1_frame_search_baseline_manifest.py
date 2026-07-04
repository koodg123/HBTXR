from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "write_stage1_frame_search_baseline_manifest.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("write_stage1_frame_search_baseline_manifest", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_run(root: Path, *, epochs: int, p10: float, p5: float, p1: float, center: float) -> None:
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)
    history = [
        {
            "epoch": epoch,
            "val": {
                "metric_search_p10_pct": p10,
                "metric_search_p5_pct": p5,
                "metric_search_p1_pct": p1,
                "metric_search_center_px": center,
            },
        }
        for epoch in range(1, epochs + 1)
    ]
    (train / "history.json").write_text(__import__("json").dumps(history), encoding="utf-8")
    (train / "best_search_p10.pt").write_text("checkpoint\n", encoding="utf-8")


def test_manifest_keeps_baseline_before_min_epoch(tmp_path: Path) -> None:
    module = load_module()
    baseline = tmp_path / "baseline"
    followup = tmp_path / "followup"
    write_run(baseline, epochs=300, p10=28.0, p5=9.0, p1=1.0, center=17.3)
    write_run(followup, epochs=10, p10=29.0, p5=10.0, p1=1.0, center=17.0)

    payload = module.build_manifest(
        baseline=baseline,
        followups=[followup],
        min_epochs=50,
        target_epochs=80,
    )

    assert payload["state"] == "pending_min_epoch"
    assert payload["promotion"]["ready"] is False
    assert payload["active_baseline"]["source"] == "baseline"
    assert payload["active_baseline"]["checkpoint"].endswith("baseline/train/best_search_p10.pt")


def test_manifest_promotes_followup_after_min_epoch(tmp_path: Path) -> None:
    module = load_module()
    baseline = tmp_path / "baseline"
    followup = tmp_path / "followup"
    write_run(baseline, epochs=300, p10=28.0, p5=9.0, p1=1.0, center=17.3)
    write_run(followup, epochs=50, p10=29.0, p5=10.0, p1=1.0, center=17.0)

    payload = module.build_manifest(
        baseline=baseline,
        followups=[followup],
        min_epochs=50,
        target_epochs=80,
    )

    assert payload["state"] == "promoted_followup"
    assert payload["promotion"]["ready"] is True
    assert payload["active_baseline"]["source"] == "followup"
    assert payload["active_baseline"]["run"].endswith("followup")
    assert payload["active_baseline"]["checkpoint"].endswith("followup/train/best_search_p10.pt")


def test_manifest_prefers_center_safe_baseline_candidate(tmp_path: Path) -> None:
    module = load_module()
    baseline = tmp_path / "baseline"
    high_p10_center_regressed = tmp_path / "high_p10_center_regressed"
    lower_p10_center_safe = tmp_path / "lower_p10_center_safe"
    write_run(baseline, epochs=300, p10=28.0, p5=9.0, p1=1.0, center=17.3)
    write_run(high_p10_center_regressed, epochs=10, p10=29.0, p5=9.5, p1=1.0, center=17.5)
    write_run(lower_p10_center_safe, epochs=10, p10=28.5, p5=10.0, p1=1.0, center=17.0)

    payload = module.build_manifest(
        baseline=baseline,
        followups=[high_p10_center_regressed, lower_p10_center_safe],
        min_epochs=50,
        target_epochs=80,
    )

    assert payload["leading_ranked_candidate"]["run"].endswith("high_p10_center_regressed")
    assert payload["leading_baseline_candidate"]["run"].endswith("lower_p10_center_safe")
