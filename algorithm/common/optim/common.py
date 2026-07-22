from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZER_DOCS_ROOT = PROJECT_ROOT / "docs" / "hbtxr" / "optimizers"


def resolve_optimizer_cfg(training_cfg: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = dict(training_cfg.get("optimizer") or {})
    name = str(optimizer_cfg.get("name", "adamw")).strip().lower()
    lr = float(optimizer_cfg.get("lr", training_cfg.get("lr", 3.0e-4)))
    weight_decay = float(optimizer_cfg.get("weight_decay", training_cfg.get("weight_decay", 1.0e-4)))
    betas_raw = optimizer_cfg.get("betas", [0.9, 0.999])
    betas = (float(betas_raw[0]), float(betas_raw[1]))
    eps = float(optimizer_cfg.get("eps", 1.0e-8))
    kwargs = dict(optimizer_cfg.get("kwargs") or {})
    return {
        "name": name,
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": eps,
        "kwargs": kwargs,
    }


def resolve_optimizer_modifiers(training_cfg: dict[str, Any]) -> dict[str, bool]:
    modifiers = dict(training_cfg.get("optimizer_modifiers") or {})
    return {
        "cautious": bool(modifiers.get("cautious", False)),
        "schedule_free": bool(modifiers.get("schedule_free", False)),
    }


def optimizer_hypers_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if output_path.name == "train":
        return output_path.parent / "hypers"
    return output_path / "hypers"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def estimate_optimizer_state_size_bytes(optimizer: Any) -> int:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict):
        return 0
    total = 0
    for value in state.values():
        if not isinstance(value, dict):
            continue
        for item in value.values():
            if torch.is_tensor(item):
                total += int(item.numel() * item.element_size())
    return total


def named_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    base = model.module if isinstance(model, nn.DataParallel) else model
    return [(name, param) for name, param in base.named_parameters() if param.requires_grad]


def split_params_auto_2d_plus(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    muon_params: list[nn.Parameter] = []
    sgd_params: list[nn.Parameter] = []
    for _, param in named_trainable_parameters(model):
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            sgd_params.append(param)
    return muon_params, sgd_params


def diff_doc_path(name: str) -> Path:
    return OPTIMIZER_DOCS_ROOT / f"{name}_diff.md"


def source_header(name: str, source: str, upstream: str) -> str:
    return "\n".join(
        [
            f"# Source: {source}",
            f"# Upstream: {upstream}",
            f"# Diff: see docs/hbtxr/optimizers/{name}_diff.md",
        ]
    )


def optimizer_summary_payload(
    *,
    resolved_cfg: dict[str, Any],
    modifiers: dict[str, bool],
    metadata: dict[str, Any],
    optimizer: Any,
) -> dict[str, Any]:
    return {
        "name": resolved_cfg["name"],
        "lr": float(resolved_cfg["lr"]),
        "weight_decay": float(resolved_cfg["weight_decay"]),
        "betas": [float(v) for v in resolved_cfg["betas"]],
        "eps": float(resolved_cfg["eps"]),
        "kwargs": resolved_cfg["kwargs"],
        "modifiers": modifiers,
        "implemented": bool(metadata.get("implemented", False)),
        "source": metadata.get("source"),
        "upstream": metadata.get("upstream"),
        "diff_doc": metadata.get("diff_doc"),
        "algorithmic_diff": bool(metadata.get("algorithmic_diff", False)),
        "external_scheduler_allowed": bool(metadata.get("external_scheduler_allowed", True)),
        "state_size_bytes": estimate_optimizer_state_size_bytes(optimizer),
        "param_group_count": len(getattr(optimizer, "param_groups", [])),
    }
