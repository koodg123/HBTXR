from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from _config import apply_config_overrides, load_config, resolve_project_path

from swift_hbtxr.interpolation import TimeLensConfig, TimeLensRunner


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TimeLens interpolation through the SWIFT-HBTXR wrapper")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--timelens-root", type=str, default=None)
    parser.add_argument("--timelens-checkpoint", type=str, default=None)
    parser.add_argument("--timelens-python", type=str, default=None)
    parser.add_argument("--frames-to-insert", type=int, default=None)
    parser.add_argument("--frames-to-skip", type=int, default=None)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--event-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--summary", type=str, default=None)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--override", action="append", default=[])
    return parser


def _required_path(path_value: str | Path | None, *, label: str) -> Path:
    if path_value is None:
        raise ValueError(f"Missing required path for {label}")
    return Path(path_value).resolve()


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    cfg = apply_config_overrides(cfg, overrides=args.override)
    interpolation_cfg = cfg.get("interpolation") or {}

    timelens_root = _required_path(
        resolve_project_path(args.timelens_root or interpolation_cfg.get("timelens_root"), project_root=PROJECT_ROOT),
        label="timelens_root",
    )
    checkpoint_file = _required_path(
        resolve_project_path(args.timelens_checkpoint or interpolation_cfg.get("checkpoint_file"), project_root=PROJECT_ROOT),
        label="checkpoint_file",
    )
    image_root = _required_path(resolve_project_path(args.image_root, project_root=PROJECT_ROOT), label="image_root")
    event_root = _required_path(resolve_project_path(args.event_root, project_root=PROJECT_ROOT), label="event_root")
    output_root = _required_path(
        resolve_project_path(args.output_root or interpolation_cfg.get("output_root"), project_root=PROJECT_ROOT),
        label="output_root",
    )
    summary_path = resolve_project_path(args.summary or interpolation_cfg.get("summary_path"), project_root=PROJECT_ROOT)

    runner = TimeLensRunner(
        TimeLensConfig(
            timelens_root=timelens_root,
            checkpoint_file=checkpoint_file,
            python_bin=str(args.timelens_python or interpolation_cfg.get("python_bin") or "python"),
            frames_to_insert=int(args.frames_to_insert if args.frames_to_insert is not None else interpolation_cfg.get("frames_to_insert", 199)),
            frames_to_skip=int(args.frames_to_skip if args.frames_to_skip is not None else interpolation_cfg.get("frames_to_skip", 0)),
        )
    )
    summary = runner.run(
        image_root=image_root,
        event_root=event_root,
        output_root=output_root,
        extra_args=args.extra_arg,
    )
    summary["config"] = str(Path(args.config).resolve())
    if summary_path is not None:
        runner.write_summary(summary, summary_path)
        summary["summary_path"] = str(Path(summary_path).resolve())
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
