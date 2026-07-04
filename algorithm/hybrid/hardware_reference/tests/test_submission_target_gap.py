from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_gap_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_submission_target_gap.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_submission_target_gap", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_source_tex(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Experimental results show 0.1812 pixel error and 0.43 ms latency.",
                "Mode & P10 & P5 & P1",
                "hybrid & 99.97 & 99.72 & 99.61",
                "Stage~1 is trained for 100 epochs.",
                "Stage~2 is trained for 200 epochs.",
            ]
        ),
        encoding="utf-8",
    )


def base_audit(tmp_path: Path) -> dict:
    source = tmp_path / "submission/main.tex"
    write_source_tex(source)
    return {
        "execution_state": "paused_by_user_directive",
        "source_file": str(source),
        "submission_claims": {
            "hybrid_pixel_error_px": 0.1812,
            "hybrid_latency_ms": 0.43,
            "hybrid_p10_pct": 99.97,
            "hybrid_p5_pct": 99.72,
            "hybrid_p1_pct": 99.61,
            "stage1_epochs": 100,
            "stage1_optimizer": "AdamW",
            "stage1_lr": 0.001,
            "stage2_epochs": 200,
            "stage2_optimizer": "AdamW",
            "stage2_lr": 0.0001,
            "stage3": "model slimming via self-supervised distillation",
        },
        "current_software_gates": {
            "center_lt": 16.468481131962367,
            "p10_gt": 35.02295998845781,
            "p5_gt": 12.133503770828247,
        },
        "direct_numeric_gap_if_same_metric_frame": {
            "center_error_ratio_current_over_submission": 90.88565746116096,
            "center_error_absolute_gap_px": 16.287281131962366,
            "p10_gap_pct_points": 64.94704001154219,
            "p5_gap_pct_points": 87.58649622917176,
        },
        "direct_comparison_valid": False,
        "direct_comparison_blocker": "metric_track_center_px coordinate frame is not matched to 0.1812 px",
        "required_alignment_checks": [
            "Verify coordinate frame and scale.",
            "Verify split protocol.",
            "Verify hybrid scheduler mode.",
            "Verify P10/P5/P1 target points.",
            "Separate latency and hardware claims.",
        ],
        "experiment_implications": [
            "XR-64 remains immediate P0.",
            "XR-65 follows XR-64 non-collapse.",
            "XR-66 follows diagnostics.",
            "XR-68 is blocked until teacher improves.",
        ],
        "completion_status": "incomplete",
    }


def test_submission_target_gap_current_file_passes() -> None:
    module = load_gap_module()
    audit = json.loads(Path("docs/resources/second_goal_submission_target_gap_audit_2026_06_20.json").read_text())

    report = module.validate_audit(audit, project_root=Path.cwd())

    assert report["ok"] is True
    assert report["source_file_exists"] is True
    assert report["completion_status"] == "incomplete"
    assert report["direct_comparison_valid"] is False
    assert report["submission_center_target_px"] == 0.1812


def test_submission_target_gap_fixture_passes(tmp_path: Path) -> None:
    module = load_gap_module()

    report = module.validate_audit(base_audit(tmp_path), project_root=tmp_path)

    assert report["ok"] is True
    assert report["alignment_check_count"] == 5
    assert report["experiment_implication_count"] == 4


def test_submission_target_gap_rejects_direct_comparison_claim(tmp_path: Path) -> None:
    module = load_gap_module()
    audit = copy.deepcopy(base_audit(tmp_path))
    audit["direct_comparison_valid"] = True

    report = module.validate_audit(audit, project_root=tmp_path)

    assert report["ok"] is False
    assert any("direct_comparison_valid must remain false" in error for error in report["errors"])
