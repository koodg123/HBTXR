from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SchedulerDecision:
    state: str
    changed: bool
    reason: str


class TrackSearchSchedulerFSM:
    def __init__(
        self,
        *,
        search_conf_threshold: float = 0.35,
        track_conf_threshold: float = 0.45,
        track_quality_threshold: float = 0.45,
        similarity_threshold: float = 0.5,
        density_threshold: float = 0.002,
        relocalize_cooldown: int = 2,
    ) -> None:
        self.search_conf_threshold = float(search_conf_threshold)
        self.track_conf_threshold = float(track_conf_threshold)
        self.track_quality_threshold = float(track_quality_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.density_threshold = float(density_threshold)
        self.relocalize_cooldown = int(relocalize_cooldown)
        self.state = "search"
        self.cooldown = 0

    def reset(self) -> None:
        self.state = "search"
        self.cooldown = 0

    def step(
        self,
        *,
        search_conf: float,
        track_conf: float,
        track_quality: float,
        event_density: float,
        similarity: float,
        closed_eye_flag: bool,
    ) -> SchedulerDecision:
        if self.cooldown > 0:
            self.cooldown -= 1

        if closed_eye_flag:
            changed = self.state != "search"
            self.state = "search"
            self.cooldown = self.relocalize_cooldown
            return SchedulerDecision(state=self.state, changed=changed, reason="closed_eye")

        if self.state == "track":
            if (
                track_conf < self.track_conf_threshold
                or track_quality < self.track_quality_threshold
                or similarity < self.similarity_threshold
                or event_density < self.density_threshold
            ):
                self.state = "search"
                self.cooldown = self.relocalize_cooldown
                return SchedulerDecision(state=self.state, changed=True, reason="track_degraded")
            return SchedulerDecision(state=self.state, changed=False, reason="track_keep")

        if self.cooldown > 0:
            return SchedulerDecision(state=self.state, changed=False, reason="search_cooldown")

        if (
            search_conf >= self.search_conf_threshold
            and track_conf >= self.track_conf_threshold
            and track_quality >= self.track_quality_threshold
            and similarity >= self.similarity_threshold
            and event_density >= self.density_threshold
        ):
            self.state = "track"
            return SchedulerDecision(state=self.state, changed=True, reason="track_ready")
        return SchedulerDecision(state=self.state, changed=False, reason="search_keep")


def metric_mode(metric_name: str) -> str:
    lowered = str(metric_name).lower()
    if lowered.endswith("_pct") or lowered.endswith("_acc") or lowered.endswith("_score"):
        return "max"
    return "min"


def is_better(candidate: float, current: float | None, *, mode: str, min_delta: float = 0.0) -> bool:
    if current is None:
        return True
    if mode == "max":
        return candidate > current + min_delta
    return candidate < current - min_delta


def build_lr_scheduler(optimizer: torch.optim.Optimizer, training_cfg: dict[str, Any], total_epochs: int):
    scheduler_cfg = training_cfg.get("scheduler") or {}
    scheduler_type = str(scheduler_cfg.get("type", "none")).strip().lower()
    if scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_epochs)),
            eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
        )
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 10)),
            gamma=float(scheduler_cfg.get("gamma", 0.5)),
        )
    if scheduler_type == "plateau":
        metric_name = str(scheduler_cfg.get("metric_name") or training_cfg.get("best_metric_name") or "")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=metric_mode(metric_name),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
            threshold=float(scheduler_cfg.get("threshold", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, *, epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    scale = max(epoch, 1) / float(warmup_epochs)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0
