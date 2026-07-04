from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from src.preprocess import event_generation as event_generation_module
from src.preprocess.event_generation import _V2eRuntime, _ensure_v2e_optional_imports
from src.preprocess.v2e_experiment import compare_target_fps_roots
from src.utils.io import write_jsonl


def _write_fake_target_session(
    *,
    target_root: Path,
    target_fps: float,
    session_code: str,
    event_generation_backend: str,
    event_counts: list[int],
    event_p: np.ndarray,
) -> Path:
    fps_tag = str(target_fps).replace(".0", "")
    session_dir = target_root / f"fps_{fps_tag}" / "sessions" / "user01" / "left" / f"session_{session_code}"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_store_path = session_dir / "session_arrays.npz"
    frame_event_ranges = []
    cursor = 0
    for count in event_counts:
        frame_event_ranges.append([cursor, cursor + int(count)])
        cursor += int(count)
    np.savez_compressed(
        session_store_path,
        source_frame_images=np.zeros((2, 4, 4), dtype=np.uint8),
        source_frame_timestamps_us=np.asarray([1000, 2000], dtype=np.int64),
        frame_images=np.zeros((len(event_counts), 4, 4), dtype=np.uint8),
        frame_timestamps_us=np.asarray([1000 + 1000 * idx for idx in range(len(event_counts))], dtype=np.int64),
        frame_event_ranges=np.asarray(frame_event_ranges, dtype=np.int64),
        event_t=np.arange(int(cursor), dtype=np.int64),
        event_x=np.zeros((int(cursor),), dtype=np.int16),
        event_y=np.zeros((int(cursor),), dtype=np.int16),
        event_p=np.asarray(event_p, dtype=np.int8),
    )
    rows = [
        {"frame_index": idx, "event_count": int(count), "event_index_range": frame_event_ranges[idx]}
        for idx, count in enumerate(event_counts)
    ]
    write_jsonl(rows, session_dir / "frame_labels.jsonl")
    session_meta = {
        "session_key": f"user01/left/{session_code}",
        "session_store_path": str(session_store_path),
        "session_store_format": "npz",
        "event_generation_backend": str(event_generation_backend),
        "interpolation_backend": "linear_blend",
        "n_target_frames": int(len(event_counts)),
        "n_rebinned_events": int(cursor),
    }
    (session_dir / "session_meta.json").write_text(json.dumps(session_meta, indent=2), encoding="utf-8")
    return session_dir


def test_compare_target_fps_roots_summarizes_fake_session(tmp_path: Path):
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    _write_fake_target_session(
        target_root=baseline_root,
        target_fps=50.0,
        session_code="101",
        event_generation_backend="none",
        event_counts=[1, 2, 1],
        event_p=np.asarray([1, -1, 1, -1], dtype=np.int8),
    )
    _write_fake_target_session(
        target_root=candidate_root,
        target_fps=50.0,
        session_code="101",
        event_generation_backend="v2e",
        event_counts=[3, 4, 5],
        event_p=np.asarray([1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1], dtype=np.int8),
    )

    summary = compare_target_fps_roots(
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        target_fps=50.0,
        output_dir=tmp_path / "report",
    )

    assert summary["n_shared_sessions"] == 1
    assert summary["baseline_total_events"] == 4
    assert summary["candidate_total_events"] == 12
    assert summary["event_ratio_candidate_vs_baseline"] == 3.0
    assert summary["sessions"][0]["event_count_curve_mae"] == 8.0 / 3.0
    assert Path(tmp_path / "report" / "summary.json").exists()


def test_ensure_v2e_optional_imports_installs_stub_modules(monkeypatch):
    for name in ("dv_processing", "easygui", "screeninfo"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, package=None):
        if name in {"dv_processing", "easygui", "screeninfo"}:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(event_generation_module.importlib.util, "find_spec", _fake_find_spec)

    _ensure_v2e_optional_imports()

    assert "dv_processing" in sys.modules
    assert "easygui" in sys.modules
    assert "screeninfo" in sys.modules
    assert callable(sys.modules["screeninfo"].get_monitors)


def test_v2e_runtime_normalizes_timestamps_before_generation():
    captured_timestamps = []

    class _FakeEmulator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate_events(self, frame, timestamp_s):
            captured_timestamps.append(float(timestamp_s))
            if len(captured_timestamps) == 1:
                return np.asarray([[0.0, 1, 2, 1]], dtype=np.float32)
            return np.asarray([[0.02, 3, 4, -1]], dtype=np.float32)

    runtime = _V2eRuntime.__new__(_V2eRuntime)
    runtime.emulator_cls = _FakeEmulator
    runtime.device = "cpu"
    runtime.v2e_kwargs = {}

    events, packets = runtime.generate_sequence_events(
        frames=np.zeros((2, 4, 4), dtype=np.uint8),
        frame_timestamps_us=np.asarray([1_657_711_084_457_716, 1_657_711_084_477_716], dtype=np.int64),
    )

    assert captured_timestamps == [0.0, 0.02]
    assert events["t"].tolist() == [1_657_711_084_457_716, 1_657_711_084_477_716]
    assert packets.tolist() == [[0, 1], [1, 2]]
