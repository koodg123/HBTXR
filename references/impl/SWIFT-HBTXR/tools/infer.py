from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT
from _config import apply_config_overrides, load_config, resolve_project_path, resolve_training_entry

from swift_hbtxr.antiblink import AntiBlinkConfig, AntiBlinkDetector
from swift_hbtxr.compat import import_swift_eye_antiblink_weights
from swift_hbtxr.io import write_json, write_jsonl
from swift_hbtxr.runtime import run_runtime_trace
from swift_hbtxr.trainer import build_model, load_checkpoint, make_loader, resolve_device_and_wrap


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWIFT-HBTXR inference with Search/Track FSM and optional anti-blink hold-last runtime")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stage2_hybrid.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-jsonl", type=str, default=str(PROJECT_ROOT / "runs" / "inference" / "runtime_trace.jsonl"))
    parser.add_argument("--output-summary", type=str, default=str(PROJECT_ROOT / "runs" / "inference" / "runtime_summary.json"))
    parser.add_argument("--antiblink-checkpoint", type=str, default=None)
    parser.add_argument("--antiblink-report", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser


def build_antiblink_detector(
    cfg: dict,
    *,
    project_root: Path,
    checkpoint_override: str | None = None,
    report_override: str | None = None,
) -> tuple[AntiBlinkDetector | None, dict | None]:
    antiblink_cfg = cfg.get("antiblink") or {}
    checkpoint_path = resolve_project_path(
        checkpoint_override if checkpoint_override is not None else antiblink_cfg.get("checkpoint"),
        project_root=project_root,
    )
    enabled = bool(antiblink_cfg.get("enabled", False)) or checkpoint_path is not None
    if not enabled:
        return None, None

    detector = AntiBlinkDetector(
        config=AntiBlinkConfig(
            closed_threshold=float(antiblink_cfg.get("closed_threshold", 0.2)),
            hold_threshold=float(antiblink_cfg.get("hold_threshold", 0.35)),
            detection_threshold=float(antiblink_cfg.get("detection_threshold", 0.75)),
            template_update_threshold=float(antiblink_cfg.get("template_update_threshold", 0.95)),
        )
    )
    report = None
    if checkpoint_path is not None:
        report_path = resolve_project_path(
            report_override if report_override is not None else antiblink_cfg.get("report_path"),
            project_root=project_root,
        )
        payload = torch.load(checkpoint_path, map_location="cpu")
        direct_model = payload.get("model") if isinstance(payload, dict) else None
        detector_state = detector.model.state_dict()
        if isinstance(direct_model, dict) and direct_model and set(direct_model.keys()).issubset(set(detector_state.keys())):
            detector.model.load_state_dict(direct_model, strict=False)
            stored_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
            for field in ("closed_threshold", "hold_threshold", "detection_threshold", "template_update_threshold"):
                if field in stored_config:
                    setattr(detector.config, field, float(stored_config[field]))
            report = payload.get("report") if isinstance(payload.get("report"), dict) else {
                "imported_keys": sorted(direct_model.keys()),
                "skipped_keys": [],
                "missing_keys": sorted(set(detector_state.keys()) - set(direct_model.keys())),
                "thresholds": vars(detector.config),
            }
            if report_path is not None:
                write_json(report, report_path)
        else:
            imported = import_swift_eye_antiblink_weights(
                checkpoint_path=checkpoint_path,
                detector=detector,
                report_path=report_path,
            )
            report = imported.to_dict()
    return detector, report


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    cfg = apply_config_overrides(cfg, overrides=args.override, device_override=args.device)
    entry = resolve_training_entry(cfg, config_path=args.config, project_root=PROJECT_ROOT)
    manifest = args.manifest or entry["val_manifest"] or entry["train_manifest"]
    loader = make_loader(manifest, cfg, shuffle=False)

    model = build_model(cfg)
    model, device = resolve_device_and_wrap(model, str((cfg.get("training") or {}).get("device", "cpu")))
    load_checkpoint(model=model, optimizer=None, scheduler=None, scaler=None, checkpoint_path=args.checkpoint, strict=False)
    antiblink_detector, antiblink_report = build_antiblink_detector(
        cfg,
        project_root=PROJECT_ROOT,
        checkpoint_override=args.antiblink_checkpoint,
        report_override=args.antiblink_report,
    )
    hold_last_on_blink = bool((cfg.get("runtime") or {}).get("hold_last_on_blink", True))
    rows = run_runtime_trace(
        model,
        loader,
        device=device,
        antiblink_detector=antiblink_detector,
        hold_last_on_blink=hold_last_on_blink,
    )

    jsonl_path = Path(args.output_jsonl).resolve()
    summary_path = Path(args.output_summary).resolve()
    write_jsonl(rows, jsonl_path)
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(manifest).resolve()),
        "output_jsonl": str(jsonl_path),
        "row_count": len(rows),
        "antiblink_enabled": antiblink_detector is not None,
        "hold_last_on_blink": hold_last_on_blink,
        "antiblink_report": antiblink_report,
    }
    write_json(summary, summary_path)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
