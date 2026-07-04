from __future__ import annotations

from pathlib import Path


def test_stage2_checkpoint_specs_include_center_checkpoint() -> None:
    source = Path("src/hbtxr/training/trainer.py").read_text(encoding="utf-8")

    assert '"metric": "metric_track_p10_pct", "filename": "best_track_p10.pt"' in source
    assert '"metric": "metric_track_p5_pct", "filename": "best_track_p5.pt"' in source
    assert '"metric": "metric_track_center_px", "filename": "best_metric_track_center_px.pt"' in source
