from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_trace_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_submission_target_source_trace.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_submission_target_source_trace", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_trace() -> dict:
    path = Path("docs/resources/submission_target_source_trace_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_submission_target_source_trace_passes() -> None:
    module = load_trace_module()
    report = module.validate_trace(read_current_trace(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["source_file_exists"] is True
    assert report["direct_comparison_valid"] is False
    assert report["target_definition_present"] is True
    assert report["mode_hybrid_pixel_error_px"] == 0.1812
    assert report["mode_hybrid_p10_pct"] == 99.97
    assert report["mode_hybrid_p5_pct"] == 99.72
    assert report["mode_hybrid_p1_pct"] == 99.61
    assert report["mode_hybrid_latency_ms"] == 0.43
    assert report["option_a_power_w"] == 1.34
    assert report["accelerator_range_is_single_target"] is False
    assert report["alignment_requirement_count"] == 6


def test_source_trace_rejects_direct_comparison_claim() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["current_software_target_relation"]["direct_comparison_valid"] = True

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_software_target_relation.direct_comparison_valid must be false" in report["errors"]


def test_source_trace_rejects_table_iv_range_as_single_target() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["source_targets"]["accelerator_table_range"]["is_single_accuracy_target"] = True

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert "accelerator_table_range.is_single_accuracy_target must be false" in report["errors"]


def test_source_trace_rejects_changed_hybrid_claim_value() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["source_targets"]["mode_table_hybrid"]["p1_pct"] = 99.0

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert "source_targets.mode_table_hybrid.p1_pct must be 99.61" in report["errors"]


def test_algorithm_table_extraction_is_scoped_to_tab_algo() -> None:
    module = load_trace_module()
    text = r"""
P10 (\%) & -- & \textbf{0.00} \\
P5 (\%) & -- & \textbf{0.00} \\
P1 (\%) & -- & \textbf{0.00} \\
Pixel Error (px) & -- & \textbf{99.0000} \\
\begin{table*}
\caption{Comparison with representative eye-tracking algorithms.}
\label{tab:algo}
\begin{tabular}{l c}
P10 (\%)       & \textbf{99.97} \\
P5 (\%)        & \textbf{99.72} \\
P1 (\%)        & \textbf{99.61} \\
Pixel Error (px) & \textbf{0.1812} \\
\end{tabular}
\end{table*}
"""

    extracted = module.extract_algo_hbtxr(text)

    assert extracted == {
        "p10_pct": 99.97,
        "p5_pct": 99.72,
        "p1_pct": 99.61,
        "pixel_error_px": 0.1812,
    }


def test_source_trace_rejects_missing_target_definition() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["source_targets"]["target_definition"]["track_residual_definition"] = ""

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert "source_targets.target_definition.track_residual_definition must mention Delta s" in report["errors"]


def test_source_trace_rejects_train_command_validation() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["validation_commands"].append(".venv/bin/python train_hbtxr.py")

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation_commands must not execute" in error for error in report["errors"])
