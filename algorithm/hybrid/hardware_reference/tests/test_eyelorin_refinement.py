from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v3" / "eval_eyelorin_refinement.py"


def _load_module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("eval_eyelorin_refinement", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    return [
        {
            "sample_id": "user01__left__session_001__000000_100",
            "raw_state": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "target_state": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "weight": 0.0,
            "meta": {"sample_timestamp_us": 100},
        },
        {
            "sample_id": "user01__left__session_001__000001_200",
            "raw_state": [100.0, 100.0, 1.0, 1.0, 0.0, 1.0],
            "target_state": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            "weight": 1.0,
            "meta": {"sample_timestamp_us": 200},
        },
        {
            "sample_id": "user01__left__session_001__000002_300",
            "raw_state": [2.0, 2.0, 1.0, 1.0, 0.0, 1.0],
            "target_state": [2.0, 2.0, 1.0, 1.0, 0.0, 1.0],
            "weight": 0.0,
            "meta": {"sample_timestamp_us": 300},
        },
    ]


def test_parse_windows_normalizes_to_odd_unique_values():
    module = _load_module()

    assert module.parse_windows("2,3,4,3") == [3, 5]


def test_median_refinement_reduces_center_error_for_spike():
    module = _load_module()
    rows = _rows()
    raw = module._weighted_metrics(rows, state_key="raw_state")
    refined_rows = module.refine_median(rows, state_key="raw_state", window=3, output_key="refined")
    refined = module._weighted_metrics(refined_rows, state_key="refined")

    assert refined["metric_track_center_px"] < raw["metric_track_center_px"]
    assert refined_rows[1]["refined"][:2] == [2.0, 2.0]


def test_adaptive_refinement_preserves_row_count_and_adds_state():
    module = _load_module()
    rows = _rows()
    refined_rows = module.refine_adaptive_m2f(
        rows,
        state_key="raw_state",
        min_window=3,
        max_window=5,
        output_key="adaptive",
    )

    assert len(refined_rows) == len(rows)
    assert all("adaptive" in row for row in refined_rows)
