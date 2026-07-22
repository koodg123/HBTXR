from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import amp, nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: amp.GradScaler | None,
    epoch: int,
    stage: str,
    cfg: dict[str, Any],
    history: list[dict[str, Any]],
    best_metrics: dict[str, dict[str, float | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": int(epoch),
        "stage": stage,
        "cfg": cfg,
        "history": history,
        "best_metrics": best_metrics,
    }
    torch.save(payload, path)


def save_teacher_checkpoint(
    path: Path,
    *,
    teacher_model: nn.Module,
    epoch: int,
    stage: str,
    cfg: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": teacher_model.state_dict(),
        "epoch": int(epoch),
        "stage": stage,
        "cfg": cfg,
    }
    torch.save(payload, path)


def flatten_best_checkpoints_from_state(state: dict[str, Any] | None) -> dict[str, float | None]:
    best = (state or {}).get("best_metrics") or {}
    flat: dict[str, float | None] = {}
    for key, value in best.items():
        flat[f"{key}_value"] = None if value.get("value") is None else float(value["value"])
        flat[f"{key}_epoch"] = None if value.get("epoch") is None else float(value["epoch"])
    return flat


def checkpoint_summary_metadata(state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    return {
        "epoch": int(state.get("epoch", -1)),
        "stage": state.get("stage"),
        **flatten_best_checkpoints_from_state(state),
    }


def slice_tensor_to_shape(tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor | None:
    if tensor.ndim != len(target_shape):
        return None
    if any(source_dim < target_dim for source_dim, target_dim in zip(tensor.shape, target_shape)):
        return None
    slices = tuple(slice(0, int(dim)) for dim in target_shape)
    return tensor[slices].clone()


def load_model_state(
    *,
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool,
) -> None:
    target = unwrap_model(model)
    target_state = target.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key not in target_state:
            continue
        if tuple(target_state[key].shape) == tuple(tensor.shape):
            loadable[key] = tensor
            continue
        if strict:
            continue
        sliced = slice_tensor_to_shape(tensor, tuple(target_state[key].shape))
        if sliced is not None:
            loadable[key] = sliced
    target.load_state_dict(loadable, strict=False)


def load_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: amp.GradScaler | None,
    checkpoint_path: str | Path,
    strict: bool,
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu")
    load_model_state(model=model, state_dict=state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state


__all__ = [
    "checkpoint_summary_metadata",
    "flatten_best_checkpoints_from_state",
    "load_checkpoint",
    "load_model_state",
    "save_checkpoint",
    "save_teacher_checkpoint",
    "unwrap_model",
]
