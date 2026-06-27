from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT
from _config import load_config, resolve_project_path

from swift_hbtxr.antiblink import AntiBlinkConfig, AntiBlinkDetector
from swift_hbtxr.compat import import_swift_eye_antiblink_weights


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import the Swift-Eye anti-blink UNet weights into SWIFT-HBTXR format")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-checkpoint", type=str, default=str(PROJECT_ROOT / "runs" / "import_swift_eye" / "antiblink_detector.pt"))
    parser.add_argument("--output-report", type=str, default=str(PROJECT_ROOT / "runs" / "import_swift_eye" / "import_report.json"))
    return parser


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    antiblink_cfg = cfg.get("antiblink") or {}
    detector = AntiBlinkDetector(
        config=AntiBlinkConfig(
            closed_threshold=float(antiblink_cfg.get("closed_threshold", 0.2)),
            hold_threshold=float(antiblink_cfg.get("hold_threshold", 0.35)),
            detection_threshold=float(antiblink_cfg.get("detection_threshold", 0.75)),
            template_update_threshold=float(antiblink_cfg.get("template_update_threshold", 0.95)),
        )
    )
    output_checkpoint = Path(resolve_project_path(args.output_checkpoint, project_root=PROJECT_ROOT)).resolve()
    output_report = Path(resolve_project_path(args.output_report, project_root=PROJECT_ROOT)).resolve()
    report = import_swift_eye_antiblink_weights(
        checkpoint_path=args.checkpoint,
        detector=detector,
        report_path=output_report,
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": detector.model.state_dict(),
            "config": vars(detector.config),
            "report": report.to_dict(),
        },
        output_checkpoint,
    )
    return {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_checkpoint": str(output_checkpoint),
        "output_report": str(output_report),
        "imported_keys": len(report.imported_keys),
        "skipped_keys": len(report.skipped_keys),
    }


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
