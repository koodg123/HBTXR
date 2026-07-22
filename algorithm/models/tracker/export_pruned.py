from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .pruning import build_export_report_payload, normalize_compression_cfg, resolve_model_role_cfg


def _student_export_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_compression_cfg(cfg)
    export_cfg = deepcopy(normalized)
    student_role_cfg = resolve_model_role_cfg(normalized, role="student")
    export_cfg["model"] = {**student_role_cfg, "student": deepcopy((normalized.get("model") or {}).get("student") or {})}
    export_heads = deepcopy(((export_cfg.get("model") or {}).get("heads") or {}))
    export_heads["event"] = False
    export_heads["mask"] = False
    export_cfg["model"]["heads"] = export_heads
    export_cfg.setdefault("distillation", {})
    export_cfg["distillation"]["enabled"] = False
    export_cfg.setdefault("regularization_ssl", {})
    export_cfg["regularization_ssl"]["enabled"] = False
    export_cfg.setdefault("pruning", {})
    export_cfg["pruning"]["enabled"] = False
    export_cfg["pruning"]["scheme"] = "exported_structural_student"
    export_cfg["pruning"]["exported"] = True
    return export_cfg


def _strip_deployment_only_heads(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remove_prefixes = ("event_head.", "mask_head.")
    return {key: value for key, value in state_dict.items() if not key.startswith(remove_prefixes)}


def export_structural_student(
    *,
    model: nn.Module,
    cfg: dict[str, Any],
    output_dir: str | Path,
    stage: str,
) -> dict[str, Any]:
    export_dir = Path(output_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_compression_cfg(cfg)
    pruning_cfg = normalized.get("pruning") or {}
    export_cfg = pruning_cfg.get("export") or {}
    checkpoint_path = export_dir / str(export_cfg.get("filename", "student_export.pt"))
    report_path = export_dir / str(export_cfg.get("report_filename", "student_export_report.json"))
    config_path = export_dir / "student_export_config.json"

    student_model = model.module if isinstance(model, nn.DataParallel) else model
    payload_cfg = _student_export_cfg(normalized)
    payload = {
        "model": _strip_deployment_only_heads(student_model.state_dict()),
        "cfg": payload_cfg,
        "stage": str(stage),
    }
    torch.save(payload, checkpoint_path)
    config_path.write_text(json.dumps(payload_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    report = build_export_report_payload(
        cfg=normalized,
        stage=stage,
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
    )
    report["config_path"] = str(config_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


__all__ = ["export_structural_student"]
