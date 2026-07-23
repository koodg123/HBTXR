from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from swift_hbtxr.io import ensure_dir, normalize_user_path, read_jsonl, resolve_stored_path


@dataclass
class TimeLensConfig:
    timelens_root: Path
    checkpoint_file: Path
    python_bin: str = sys.executable
    frames_to_insert: int = 199
    frames_to_skip: int = 0


@dataclass
class TimeLensPrepConfig:
    canonical_root: Path
    prepared_root: Path
    indexes_root: Path | None = None
    frame_source: str = "auto"
    link_mode: str = "auto"
    start_index: int = 0
    frame_step: int = 1
    max_frames: int | None = None
    overwrite: bool = False


class TimeLensRunner:
    def __init__(self, config: TimeLensConfig) -> None:
        self.config = config

    @property
    def runner_script(self) -> Path:
        return Path(__file__).resolve().parents[1] / "tools" / "run_timelens_attention.py"

    def validate(self) -> None:
        if not self.config.timelens_root.exists():
            raise FileNotFoundError(f"TimeLens root not found: {self.config.timelens_root}")
        if not self.runner_script.exists():
            raise FileNotFoundError(f"TimeLens runner not found: {self.runner_script}")
        if not self.config.checkpoint_file.exists():
            raise FileNotFoundError(f"TimeLens checkpoint not found: {self.config.checkpoint_file}")

    def build_command(self, *, image_root: str | Path, event_root: str | Path, output_root: str | Path) -> list[str]:
        return [
            self.config.python_bin,
            str(self.runner_script),
            "--timelens-root",
            str(self.config.timelens_root),
            "--checkpoint-file",
            str(self.config.checkpoint_file),
            "--root-image-folder",
            str(Path(image_root).resolve()),
            "--root-event-folder",
            str(Path(event_root).resolve()),
            "--root-output-folder",
            str(Path(output_root).resolve()),
            "--number-of-frames-to-skip",
            str(int(self.config.frames_to_skip)),
            "--number-of-frames-to-insert",
            str(int(self.config.frames_to_insert)),
        ]

    def run(
        self,
        *,
        image_root: str | Path,
        event_root: str | Path,
        output_root: str | Path,
        extra_args: Sequence[str] | None = None,
        check: bool = True,
    ) -> dict[str, str | int | list[str]]:
        self.validate()
        command = self.build_command(image_root=image_root, event_root=event_root, output_root=output_root)
        if extra_args:
            command.extend([str(item) for item in extra_args])
        completed = subprocess.run(command, check=check, capture_output=True, text=True)
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "output_root": str(Path(output_root).resolve()),
        }

    def write_summary(self, summary: dict, path: str | Path) -> None:
        dst = Path(path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_session_index(*, canonical_root: str | Path, indexes_root: str | Path | None = None) -> dict[str, dict]:
    canonical_path = Path(canonical_root).resolve()
    index_root = canonical_path / "indexes" if indexes_root is None else Path(indexes_root).resolve()
    index_path = index_root / "sessions.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Session index not found: {index_path}")
    rows = read_jsonl(index_path)
    return {str(row["session_key"]): row for row in rows if row.get("session_key")}


def _session_relpath(session_key: str) -> Path:
    return Path(*[part for part in str(session_key).split("/") if part])


def _can_read_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _sorted_png_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.png") if path.is_file())


def _select_frame_source(session_row: dict, *, config: TimeLensPrepConfig) -> tuple[str, Path, list[Path]]:
    session_key = str(session_row["session_key"])
    canonical_frames_dir: Path | None = None
    raw_frames_dir: Path | None = None

    frames_dir = session_row.get("frames_dir") or str(Path("sessions") / _session_relpath(session_key) / "frames")
    if frames_dir:
        candidate = resolve_stored_path(config.canonical_root, frames_dir)
        if candidate.exists():
            canonical_frames_dir = candidate
    raw_session_dir = session_row.get("raw_session_dir")
    if raw_session_dir:
        candidate = normalize_user_path(raw_session_dir) / "frames"
        if candidate.exists():
            raw_frames_dir = candidate

    def _probe(path: Path | None) -> tuple[Path | None, list[Path], bool]:
        if path is None:
            return None, [], False
        files = _sorted_png_files(path)
        readable = any(_can_read_file(item) for item in files[: min(len(files), 4)])
        return path, files, readable

    canonical_dir, canonical_files, canonical_readable = _probe(canonical_frames_dir)
    raw_dir, raw_files, raw_readable = _probe(raw_frames_dir)
    requested = str(config.frame_source).strip().lower()

    if requested not in {"auto", "canonical", "raw"}:
        raise ValueError(f"Unsupported frame_source: {config.frame_source}")

    if requested == "canonical":
        if not canonical_files:
            raise FileNotFoundError(f"No canonical frames found for session: {session_key}")
        if not canonical_readable:
            raise OSError(f"Canonical frames are not readable for session: {session_key}")
        return "canonical", canonical_dir, canonical_files

    if requested == "raw":
        if not raw_files:
            raise FileNotFoundError(f"No raw frames found for session: {session_key}")
        if not raw_readable:
            raise OSError(f"Raw frames are not readable for session: {session_key}")
        return "raw", raw_dir, raw_files

    if canonical_files and canonical_readable:
        return "canonical", canonical_dir, canonical_files
    if raw_files and raw_readable:
        return "raw", raw_dir, raw_files
    if canonical_files:
        raise OSError(f"Canonical frames exist but are not readable, and no raw fallback exists for session: {session_key}")
    if raw_files:
        raise OSError(f"Raw frames exist but are not readable for session: {session_key}")
    raise FileNotFoundError(f"No frame source found for session: {session_key}")


def _slice_frame_files(frame_files: Sequence[Path], *, start_index: int, frame_step: int, max_frames: int | None) -> list[Path]:
    if frame_step <= 0:
        raise ValueError("frame_step must be >= 1")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    selected = list(frame_files[start_index::frame_step])
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("max_frames must be >= 2 when provided")
        selected = selected[:max_frames]
    if len(selected) < 2:
        raise ValueError("TimeLens preparation requires at least two frames")
    return selected


def _parse_frame_timestamp(path: Path) -> int:
    stem = path.stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Could not infer timestamp from frame filename: {path.name}")


def _materialize_file(src: Path, dst: Path, *, link_mode: str) -> str:
    mode = str(link_mode).strip().lower()
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if mode == "symlink":
        dst.symlink_to(src)
        return "symlink"
    if mode != "auto":
        raise ValueError(f"Unsupported link_mode: {link_mode}")

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def prepare_timelens_session_inputs(*, session_key: str, config: TimeLensPrepConfig) -> dict[str, str | int | list[int]]:
    session_index = _load_session_index(canonical_root=config.canonical_root, indexes_root=config.indexes_root)
    if session_key not in session_index:
        raise KeyError(f"Unknown session_key: {session_key}")

    session_row = session_index[session_key]
    source_kind, source_frames_dir, frame_files = _select_frame_source(session_row, config=config)
    selected_frames = _slice_frame_files(
        frame_files,
        start_index=int(config.start_index),
        frame_step=int(config.frame_step),
        max_frames=config.max_frames,
    )
    timestamps = [_parse_frame_timestamp(path) for path in selected_frames]

    session_rel = _session_relpath(session_key)
    image_dir = config.prepared_root / "images" / session_rel
    event_dir = config.prepared_root / "events" / session_rel
    if config.overwrite:
        shutil.rmtree(image_dir, ignore_errors=True)
        shutil.rmtree(event_dir, ignore_errors=True)
    ensure_dir(image_dir)
    ensure_dir(event_dir)

    effective_link_mode = "copy"
    for frame_path in selected_frames:
        materialized = _materialize_file(frame_path, image_dir / frame_path.name, link_mode=config.link_mode)
        if effective_link_mode == "copy" and materialized != "copy":
            effective_link_mode = materialized

    timestamp_path = image_dir / "timestamp.txt"
    timestamp_path.write_text("\n".join(str(value) for value in timestamps) + "\n", encoding="utf-8")

    events_src = resolve_stored_path(config.canonical_root, session_row["events_npz"])
    events_dst = event_dir / "0000001.npz"
    event_materialized = _materialize_file(events_src, events_dst, link_mode=config.link_mode)
    if effective_link_mode == "copy" and event_materialized != "copy":
        effective_link_mode = event_materialized

    return {
        "session_key": session_key,
        "frame_source": source_kind,
        "frame_count": len(selected_frames),
        "start_timestamp_us": int(timestamps[0]),
        "end_timestamp_us": int(timestamps[-1]),
        "sensor_size_wh": list(session_row.get("sensor_size_wh", [346, 240])),
        "image_dir": str(image_dir.resolve()),
        "event_dir": str(event_dir.resolve()),
        "timestamp_path": str(timestamp_path.resolve()),
        "events_file": str(events_dst.resolve()),
        "source_frames_dir": str(source_frames_dir.resolve()),
        "source_events_file": str(events_src.resolve()),
        "materialize_mode": effective_link_mode,
    }


def prepare_timelens_inputs(*, session_keys: Sequence[str], config: TimeLensPrepConfig) -> dict[str, object]:
    unique_session_keys: list[str] = []
    seen: set[str] = set()
    for session_key in session_keys:
        normalized = str(session_key).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_session_keys.append(normalized)
    if not unique_session_keys:
        raise ValueError("At least one session_key is required")

    sessions = [prepare_timelens_session_inputs(session_key=session_key, config=config) for session_key in unique_session_keys]
    return {
        "prepared_root": str(config.prepared_root.resolve()),
        "image_root": str((config.prepared_root / "images").resolve()),
        "event_root": str((config.prepared_root / "events").resolve()),
        "session_count": len(sessions),
        "sessions": sessions,
    }
