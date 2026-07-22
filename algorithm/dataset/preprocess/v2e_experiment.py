from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset.preprocess.target_fps_build import resolve_target_fps_root
from utils.io import read_json, read_jsonl


@dataclass(frozen=True)
class TargetSessionStats:
    session_key: str
    relative_session_dir: str
    session_dir: Path
    session_store_path: Path
    event_generation_backend: str
    interpolation_backend: str
    n_target_frames: int
    n_rebinned_events: int
    event_counts: np.ndarray
    pos_count: int
    neg_count: int


def _session_dirs(target_root: str | Path, target_fps: float) -> list[Path]:
    fps_root = resolve_target_fps_root(target_root, target_fps)
    sessions_root = fps_root / "sessions"
    if not sessions_root.exists():
        return []
    return sorted(path for path in sessions_root.glob("user*/left/session_*")) + sorted(
        path for path in sessions_root.glob("user*/right/session_*")
    )


def _load_event_polarity_counts(session_store_path: Path, store_format: str) -> tuple[int, int]:
    text = str(store_format).strip().lower()
    if text == "h5":
        import h5py

        with h5py.File(session_store_path, "r") as handle:
            p = np.asarray(handle["events"]["p"], dtype=np.int8)
    elif text == "npz":
        payload = np.load(session_store_path, allow_pickle=False)
        p = np.asarray(payload["event_p"], dtype=np.int8)
    else:
        raise ValueError(f"Unsupported session_store_format: {store_format!r}")
    pos_count = int(np.count_nonzero(p > 0))
    neg_count = int(np.count_nonzero(p <= 0))
    return pos_count, neg_count


def load_target_session_stats(session_dir: str | Path) -> TargetSessionStats:
    session_dir = Path(session_dir).resolve()
    session_meta = read_json(session_dir / "session_meta.json")
    frame_labels = read_jsonl(session_dir / "frame_labels.jsonl")
    event_counts = np.asarray([int(row.get("event_count", 0)) for row in frame_labels], dtype=np.int64)
    session_store_path = Path(session_meta["session_store_path"]).resolve()
    pos_count, neg_count = _load_event_polarity_counts(session_store_path, str(session_meta["session_store_format"]))
    relative_session_dir = str(session_dir.relative_to(session_dir.parents[3]))
    return TargetSessionStats(
        session_key=str(session_meta["session_key"]),
        relative_session_dir=relative_session_dir,
        session_dir=session_dir,
        session_store_path=session_store_path,
        event_generation_backend=str(session_meta.get("event_generation_backend", "none")),
        interpolation_backend=str(session_meta.get("interpolation_backend", "linear_blend")),
        n_target_frames=int(session_meta.get("n_target_frames", len(frame_labels))),
        n_rebinned_events=int(session_meta.get("n_rebinned_events", int(event_counts.sum()))),
        event_counts=event_counts,
        pos_count=pos_count,
        neg_count=neg_count,
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _save_event_count_plot(*, baseline: TargetSessionStats, candidate: TargetSessionStats, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(int(max(len(baseline.event_counts), len(candidate.event_counts))), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs[: len(baseline.event_counts)], baseline.event_counts, label=f"{baseline.event_generation_backend}", linewidth=1.5)
    ax.plot(xs[: len(candidate.event_counts)], candidate.event_counts, label=f"{candidate.event_generation_backend}", linewidth=1.5)
    ax.set_title(f"{baseline.session_key} event count per target frame")
    ax.set_xlabel("target frame index")
    ax.set_ylabel("event count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compare_target_fps_roots(
    *,
    baseline_root: str | Path,
    candidate_root: str | Path,
    target_fps: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    baseline_dirs = _session_dirs(baseline_root, target_fps)
    candidate_dirs = _session_dirs(candidate_root, target_fps)
    baseline_map = {str(path.relative_to(path.parents[3])): path for path in baseline_dirs}
    candidate_map = {str(path.relative_to(path.parents[3])): path for path in candidate_dirs}
    shared_keys = sorted(set(baseline_map) & set(candidate_map))

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions: list[dict[str, Any]] = []
    total_base_events = 0
    total_candidate_events = 0
    total_base_pos = 0
    total_base_neg = 0
    total_candidate_pos = 0
    total_candidate_neg = 0

    for key in shared_keys:
        baseline_stats = load_target_session_stats(baseline_map[key])
        candidate_stats = load_target_session_stats(candidate_map[key])
        total_base_events += int(baseline_stats.n_rebinned_events)
        total_candidate_events += int(candidate_stats.n_rebinned_events)
        total_base_pos += int(baseline_stats.pos_count)
        total_base_neg += int(baseline_stats.neg_count)
        total_candidate_pos += int(candidate_stats.pos_count)
        total_candidate_neg += int(candidate_stats.neg_count)

        plot_path = output_dir / f"{baseline_stats.session_key}_event_count_plot.png"
        _save_event_count_plot(baseline=baseline_stats, candidate=candidate_stats, output_path=plot_path)

        session_summary = {
            "session_key": baseline_stats.session_key,
            "relative_session_dir": key,
            "baseline_backend": baseline_stats.event_generation_backend,
            "candidate_backend": candidate_stats.event_generation_backend,
            "interpolation_backend": candidate_stats.interpolation_backend,
            "n_target_frames": int(candidate_stats.n_target_frames),
            "baseline_total_events": int(baseline_stats.n_rebinned_events),
            "candidate_total_events": int(candidate_stats.n_rebinned_events),
            "event_ratio_candidate_vs_baseline": _safe_ratio(candidate_stats.n_rebinned_events, baseline_stats.n_rebinned_events),
            "baseline_mean_events_per_frame": float(np.mean(baseline_stats.event_counts)) if len(baseline_stats.event_counts) else 0.0,
            "candidate_mean_events_per_frame": float(np.mean(candidate_stats.event_counts)) if len(candidate_stats.event_counts) else 0.0,
            "baseline_max_events_per_frame": int(np.max(baseline_stats.event_counts)) if len(baseline_stats.event_counts) else 0,
            "candidate_max_events_per_frame": int(np.max(candidate_stats.event_counts)) if len(candidate_stats.event_counts) else 0,
            "baseline_pos_fraction": _safe_ratio(baseline_stats.pos_count, baseline_stats.pos_count + baseline_stats.neg_count),
            "candidate_pos_fraction": _safe_ratio(candidate_stats.pos_count, candidate_stats.pos_count + candidate_stats.neg_count),
            "event_count_curve_mae": float(np.mean(np.abs(candidate_stats.event_counts - baseline_stats.event_counts)))
            if len(candidate_stats.event_counts) and len(candidate_stats.event_counts) == len(baseline_stats.event_counts)
            else None,
            "plot_path": str(plot_path),
        }
        sessions.append(session_summary)

    summary = {
        "target_fps": float(target_fps),
        "n_shared_sessions": int(len(shared_keys)),
        "baseline_root": str(Path(baseline_root).resolve()),
        "candidate_root": str(Path(candidate_root).resolve()),
        "baseline_total_events": int(total_base_events),
        "candidate_total_events": int(total_candidate_events),
        "event_ratio_candidate_vs_baseline": _safe_ratio(total_candidate_events, total_base_events),
        "baseline_pos_fraction": _safe_ratio(total_base_pos, total_base_pos + total_base_neg),
        "candidate_pos_fraction": _safe_ratio(total_candidate_pos, total_candidate_pos + total_candidate_neg),
        "sessions": sessions,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = [
    "TargetSessionStats",
    "compare_target_fps_roots",
    "load_target_session_stats",
]
