from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SwiftHBTXRDataset
from .losses import compute_stage1_losses, compute_stage2_losses
from .metrics import compute_metrics
from .model import HBTXRTracker
from .scheduler import build_lr_scheduler, current_lr, is_better, metric_mode, warmup_lr


TENSOR_KEYS = {
    "frame",
    "event",
    "mask_target",
    "eye_target",
    "prev_state",
    "cur_state",
    "pupil_search_target",
    "pupil_track_target",
    "constraint_center",
    "annotation_quality",
    "similarity_target",
    "event_density",
    "closed_eye_flag",
    "open_extent",
    "mask_valid",
    "valid_track",
    "aux_target",
    "ellipse_xywht",
    "state6",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]
        if torch.is_tensor(first):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = move_to_device(value, device)
        else:
            moved[key] = value
    return moved


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def serialize_stats(stats: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in stats.items()}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_best_metric(stage: str) -> str:
    return "metric_search_p10_pct" if stage == "stage1" else "metric_track_p10_pct"


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler | None,
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


def make_loader(manifest_path: str, cfg: dict, shuffle: bool) -> DataLoader:
    data_cfg = cfg.get("data") or {}
    event_builder = data_cfg.get("event_builder") or {}
    dataset_kwargs = {
        "input_size": tuple(data_cfg.get("input_size", [256, 256])),
        "resize_policy": str(data_cfg.get("resize_policy", "facet_square_direct")),
        "canonical_root": data_cfg.get("canonical_root"),
        "cache_root": data_cfg.get("cache_root"),
        "use_cache": bool(data_cfg.get("use_cache", True)),
        "per_channel_normalize": bool(data_cfg.get("per_channel_normalize", True)),
        "event_builder": {
            "policy": str(event_builder.get("policy", "fixed_count")),
            "time_bin_us": int(event_builder.get("time_bin_us", 5000)),
            "event_count_target": int(event_builder.get("event_count_target", 5000)),
            "accumulation": str(event_builder.get("accumulation", "causal_linear")),
            "causal_weight_power": float(event_builder.get("causal_weight_power", 1.0)),
            "polarity_split": bool(event_builder.get("polarity_split", True)),
        },
    }
    dataset = SwiftHBTXRDataset(manifest_path, **dataset_kwargs)
    training_cfg = cfg.get("training") or {}
    device_spec = str(training_cfg.get("device", "cpu")).strip().lower()
    pin_memory = torch.cuda.is_available() and (device_spec.startswith("cuda") or device_spec == "multi-gpu" or "," in device_spec)
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=bool(shuffle),
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def build_model(cfg: dict) -> HBTXRTracker:
    model_cfg = cfg.get("model") or {}
    runtime_cfg = cfg.get("runtime") or {}
    return HBTXRTracker(
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        patch_size=int(model_cfg.get("patch_size", 16)),
        input_size=tuple(model_cfg.get("input_size", cfg.get("data", {}).get("input_size", [256, 256]))),
        dropout=float(model_cfg.get("dropout", 0.0)),
        aux_classes=int(model_cfg.get("aux_classes", 5)),
        runtime_cfg=runtime_cfg,
    )


def resolve_device_and_wrap(model: nn.Module, device_spec: str) -> tuple[nn.Module, torch.device]:
    spec = str(device_spec or "cpu").strip()
    if spec == "multi-gpu":
        if not torch.cuda.is_available():
            return model.to(torch.device("cpu")), torch.device("cpu")
        device = torch.device("cuda:0")
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model, device
    if "," in spec:
        if not torch.cuda.is_available():
            return model.to(torch.device("cpu")), torch.device("cpu")
        ids = []
        for item in spec.split(","):
            item = item.strip()
            if item.startswith("cuda:"):
                ids.append(int(item.split(":", 1)[1]))
        device = torch.device(f"cuda:{ids[0]}" if ids else "cuda:0")
        model = model.to(device)
        if len(ids) > 1:
            model = nn.DataParallel(model, device_ids=ids)
        return model, device
    device = torch.device(spec)
    return model.to(device), device


def load_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: GradScaler | None,
    checkpoint_path: str | Path,
    strict: bool,
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu")
    unwrap_model(model).load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state


def build_grad_scaler(*, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(*, enabled: bool, device_type: str):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def epoch_loop(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
    stage: str,
    loss_cfg: dict[str, Any],
    amp_enabled: bool,
    grad_accum_steps: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        with autocast_context(enabled=amp_enabled, device_type=device.type):
            outputs = model(batch)
            if stage == "stage1":
                losses = compute_stage1_losses(batch, outputs, loss_cfg)
            else:
                losses = compute_stage2_losses(batch, outputs, loss_cfg)
            metrics = compute_metrics(batch, outputs)
            loss_total = losses["loss_total"]

        if training:
            scaled_loss = loss_total / max(1, grad_accum_steps)
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            if step % max(1, grad_accum_steps) == 0:
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)

        combined = {**losses, **metrics}
        for key, value in combined.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
        count += 1

    if training and count % max(1, grad_accum_steps) != 0:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {key: value / max(1, count) for key, value in totals.items()}


def train(
    *,
    cfg: dict,
    train_manifest: str,
    val_manifest: str | None,
    output_dir: str | Path,
    stage1_checkpoint: str | None = None,
    resume_checkpoint: str | None = None,
) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    training_cfg = cfg.get("training") or {}
    loss_cfg = cfg.get("loss") or {}
    stage = str(training_cfg.get("stage", "stage1")).strip().lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    history_jsonl_path = output_dir / "history.jsonl"
    train_log_path = output_dir / "train_log.txt"

    train_loader = make_loader(train_manifest, cfg, shuffle=True)
    val_loader = make_loader(val_manifest, cfg, shuffle=False) if val_manifest else None

    model = build_model(cfg)
    model, device = resolve_device_and_wrap(model, str(training_cfg.get("device", "cpu")))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 3.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
    )
    scheduler = build_lr_scheduler(optimizer, training_cfg, int(training_cfg.get("epochs", 1)))
    scaler = build_grad_scaler(enabled=bool(training_cfg.get("amp", False)) and device.type == "cuda")
    if not scaler.is_enabled():
        scaler = None

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, dict[str, float | str]] = {}

    if stage == "stage2" and stage1_checkpoint:
        load_checkpoint(model=model, optimizer=None, scheduler=None, scaler=None, checkpoint_path=stage1_checkpoint, strict=False)

    if resume_checkpoint:
        state = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_path=resume_checkpoint,
            strict=False,
        )
        start_epoch = int(state.get("epoch", 0)) + 1
        history = list(state.get("history") or [])
        best_metrics = dict(state.get("best_metrics") or {})

    total_epochs = int(training_cfg.get("epochs", 1))
    best_metric_name = str(training_cfg.get("best_metric_name") or default_best_metric(stage))
    best_metric_control_mode = metric_mode(best_metric_name)
    early_cfg = training_cfg.get("early_stopping") or {}
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 10))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_start_epoch = int(early_cfg.get("start_epoch", 1))
    early_counter = 0
    best_control_value = None if best_metrics.get(best_metric_name) is None else float(best_metrics[best_metric_name]["value"])

    checkpoint_specs = (
        [{"metric": "metric_search_p10_pct", "filename": "best_search_p10.pt"}, {"metric": "metric_search_p5_pct", "filename": "best_search_p5.pt"}]
        if stage == "stage1"
        else [{"metric": "metric_track_p10_pct", "filename": "best_track_p10.pt"}, {"metric": "metric_track_p5_pct", "filename": "best_track_p5.pt"}]
    )

    started_at = time.time()
    base_lr = float(training_cfg.get("lr", 3.0e-4))
    warmup_epochs = int((training_cfg.get("scheduler") or {}).get("warmup_epochs", 0))

    for epoch in range(start_epoch, total_epochs + 1):
        warmup_lr(optimizer, base_lr, epoch=epoch, warmup_epochs=warmup_epochs)
        train_stats = epoch_loop(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            stage=stage,
            loss_cfg=loss_cfg,
            amp_enabled=bool(training_cfg.get("amp", False)) and device.type == "cuda",
            grad_accum_steps=int(training_cfg.get("grad_accum_steps", 1)),
        )
        if val_loader is not None:
            with torch.no_grad():
                val_stats = epoch_loop(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    scaler=None,
                    device=device,
                    stage=stage,
                    loss_cfg=loss_cfg,
                    amp_enabled=bool(training_cfg.get("amp", False)) and device.type == "cuda",
                    grad_accum_steps=1,
                )
        else:
            val_stats = None

        ref_stats = val_stats or train_stats
        control_value = float(ref_stats.get(best_metric_name, ref_stats.get("loss_total", 0.0)))
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(control_value)
            elif epoch > warmup_epochs:
                scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "stage": stage,
            "lr": current_lr(optimizer),
            "elapsed_sec": time.time() - started_at,
            "train": serialize_stats(train_stats),
            "val": None if val_stats is None else serialize_stats(val_stats),
        }
        history.append(epoch_row)
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        append_jsonl(history_jsonl_path, epoch_row)
        with train_log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(json.dumps(epoch_row, ensure_ascii=False) + "\n")

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            stage=stage,
            cfg=cfg,
            history=history,
            best_metrics=best_metrics,
        )

        for spec in checkpoint_specs:
            metric_name = spec["metric"]
            if metric_name not in ref_stats:
                continue
            metric_value = float(ref_stats[metric_name])
            prior = best_metrics.get(metric_name)
            if is_better(metric_value, None if prior is None else float(prior["value"]), mode=metric_mode(metric_name)):
                best_metrics[metric_name] = {"value": metric_value, "epoch": epoch, "path": spec["filename"]}
                save_checkpoint(
                    output_dir / spec["filename"],
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    stage=stage,
                    cfg=cfg,
                    history=history,
                    best_metrics=best_metrics,
                )
                if metric_name == best_metric_name:
                    save_checkpoint(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        stage=stage,
                        cfg=cfg,
                        history=history,
                        best_metrics=best_metrics,
                    )

        if is_better(control_value, best_control_value, mode=best_metric_control_mode, min_delta=early_min_delta):
            best_control_value = control_value
            early_counter = 0
        elif early_enabled and epoch >= early_start_epoch:
            early_counter += 1
            if early_counter >= early_patience:
                break

    return {
        "output_dir": str(output_dir),
        "history_path": str(history_path),
        "best_metrics": best_metrics,
    }
