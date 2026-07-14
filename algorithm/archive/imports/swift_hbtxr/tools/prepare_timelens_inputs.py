from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from _config import load_config, resolve_project_path

from swift_hbtxr.interpolation import TimeLensPrepConfig, prepare_timelens_inputs
from swift_hbtxr.io import read_jsonl


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TimeLens-ready session folders from SWIFT-HBTXR canonical data")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--canonical-root", type=str, default=None)
    parser.add_argument("--indexes-root", type=str, default=None)
    parser.add_argument("--prepared-root", type=str, default=None)
    parser.add_argument("--summary", type=str, default=None)
    parser.add_argument("--session-key", action="append", default=[])
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--frame-source", choices=("auto", "canonical", "raw"), default=None)
    parser.add_argument("--link-mode", choices=("auto", "copy", "hardlink", "symlink"), default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _required_path(path_value: str | Path | None, *, label: str) -> Path:
    if path_value is None:
        raise ValueError(f"Missing required path for {label}")
    return Path(path_value).resolve()


def _resolve_session_keys(*, manifest_paths: list[str], explicit_session_keys: list[str]) -> list[str]:
    session_keys: list[str] = []
    seen: set[str] = set()

    def _push(value: str | None) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            session_keys.append(text)

    for item in explicit_session_keys:
        _push(item)
    for manifest_path in manifest_paths:
        for row in read_jsonl(manifest_path):
            _push(row.get("session_key"))
    return session_keys


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    interpolation_cfg = cfg.get("interpolation") or {}
    data_cfg = cfg.get("data") or {}

    canonical_root = _required_path(
        resolve_project_path(args.canonical_root or data_cfg.get("canonical_root"), project_root=PROJECT_ROOT),
        label="canonical_root",
    )
    indexes_root = resolve_project_path(args.indexes_root, project_root=PROJECT_ROOT)
    prepared_root = _required_path(
        resolve_project_path(
            args.prepared_root or interpolation_cfg.get("prepared_root") or "data/_internal/timelens_ready",
            project_root=PROJECT_ROOT,
        ),
        label="prepared_root",
    )
    summary_path = resolve_project_path(
        args.summary or interpolation_cfg.get("prepared_summary_path") or prepared_root / "summary.json",
        project_root=PROJECT_ROOT,
    )

    manifest_paths = [
        str(_required_path(resolve_project_path(item, project_root=PROJECT_ROOT), label="manifest"))
        for item in args.manifest
    ]
    session_keys = _resolve_session_keys(manifest_paths=manifest_paths, explicit_session_keys=args.session_key)
    if not session_keys:
        raise ValueError("Provide at least one --session-key or --manifest")

    summary = prepare_timelens_inputs(
        session_keys=session_keys,
        config=TimeLensPrepConfig(
            canonical_root=canonical_root,
            prepared_root=prepared_root,
            indexes_root=indexes_root,
            frame_source=str(args.frame_source or interpolation_cfg.get("frame_source") or "auto"),
            link_mode=str(args.link_mode or interpolation_cfg.get("link_mode") or "auto"),
            start_index=int(args.start_index),
            frame_step=int(args.frame_step),
            max_frames=args.max_frames,
            overwrite=bool(args.overwrite),
        ),
    )
    summary["config"] = str(Path(args.config).resolve())
    summary["canonical_root"] = str(canonical_root)
    summary["indexes_root"] = None if indexes_root is None else str(Path(indexes_root).resolve())
    summary["manifest_paths"] = manifest_paths
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["summary_path"] = str(Path(summary_path).resolve())
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
