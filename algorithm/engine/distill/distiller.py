"""Knowledge distillation for the reimplemented HBTXR models (P5.8).

The existing ``common.losses.distillation.compute_distillation_losses`` is keyed to
the OLD model output names ("search/pooled", "search/pupil", ...) and is inert on
the new contract, so this module provides a contract-native distiller: the student
is matched to a frozen teacher on the shared output heads (box / state / residual /
eye_box / mask / reliability) while still learning the task loss from the GT batch.

- ``distillation_loss``: teacher-student agreement on shared output tensors
  (smooth L1 for the geometric heads + mask, MSE for the reliability head).
- ``DistillTrainer``: per step, forward the frozen teacher (no_grad) and the student,
  then ``total = task_loss + distill_weight * distill_loss`` and update the student.
  Modality-aware exactly like engine.train.Trainer (direct vs hybrid branches).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from engine.train.losses import compute_losses
from engine.train.trainer import TrainConfig

_REGRESSION_KEYS = ("box", "state", "eye_box", "residual")

_DEFAULT_DISTILL_WEIGHTS: dict[str, float] = {
    "box": 1.0,
    "state": 1.0,
    "eye_box": 0.5,
    "residual": 1.0,
    "mask": 1.0,
    "reliability": 0.5,
}


def distillation_loss(
    student_out: dict[str, torch.Tensor],
    teacher_out: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Student<-teacher agreement on shared heads; teacher tensors are detached."""
    w = {**_DEFAULT_DISTILL_WEIGHTS, **(weights or {})}
    losses: dict[str, torch.Tensor] = {}
    for key in _REGRESSION_KEYS:
        if key in student_out and key in teacher_out:
            losses[key] = F.smooth_l1_loss(student_out[key], teacher_out[key].detach())
    if "mask" in student_out and "mask" in teacher_out:
        losses["mask"] = F.smooth_l1_loss(student_out["mask"], teacher_out["mask"].detach())
    if "reliability" in student_out and "reliability" in teacher_out:
        losses["reliability"] = F.mse_loss(student_out["reliability"], teacher_out["reliability"].detach())
    if losses:
        total = sum(w.get(name, 1.0) * value for name, value in losses.items())
    else:
        total = torch.zeros((), requires_grad=True)
    losses["total"] = total
    return losses


@dataclass
class DistillConfig(TrainConfig):
    distill_weight: float = 1.0
    distill_weights: dict[str, float] | None = None


class DistillTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: DistillConfig | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.config = config or DistillConfig()
        self.student = student.to(self.config.device)
        self.teacher = teacher.to(self.config.device).eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_hybrid = self.config.modality.lower().startswith("hybrid")

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        dev = self.config.device
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _direct(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        student_out = self.student(batch["image"])
        if torch.is_tensor(student_out):
            student_out = {"mask": student_out}
        with torch.no_grad():
            teacher_out = self.teacher(batch["image"])
            if torch.is_tensor(teacher_out):
                teacher_out = {"mask": teacher_out}
        task = compute_losses(student_out, batch, self.config.loss_weights)
        distill = distillation_loss(student_out, teacher_out, self.config.distill_weights)
        return self._combine(task, distill)

    def _hybrid(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        task_total = torch.zeros((), device=self.config.device)
        distill_total = torch.zeros((), device=self.config.device)
        merged: dict[str, torch.Tensor] = {}
        if "frame" in batch:
            s_out = self.student.search_step(batch["frame"])
            with torch.no_grad():
                t_out = self.teacher.search_step(batch["frame"])
            task_total = task_total + compute_losses(s_out, batch, self.config.loss_weights)["total"]
            distill_total = distill_total + distillation_loss(s_out, t_out, self.config.distill_weights)["total"]
        if "event" in batch:
            anchor = batch.get("anchor_state")
            if anchor is None and "state" in batch:
                anchor = batch["state"]
            s_out = self.student.track_step(batch["event"], anchor)
            with torch.no_grad():
                t_out = self.teacher.track_step(batch["event"], anchor)
            task_total = task_total + compute_losses(s_out, batch, self.config.loss_weights)["total"]
            distill_total = distill_total + distillation_loss(s_out, t_out, self.config.distill_weights)["total"]
        merged["task"] = task_total
        merged["distill"] = distill_total
        merged["total"] = task_total + self.config.distill_weight * distill_total
        return merged

    def _combine(self, task: dict[str, torch.Tensor], distill: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        merged = {f"task_{k}": v for k, v in task.items() if k != "total"}
        merged.update({f"distill_{k}": v for k, v in distill.items() if k != "total"})
        merged["task"] = task["total"]
        merged["distill"] = distill["total"]
        merged["total"] = task["total"] + self.config.distill_weight * distill["total"]
        return merged

    def training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch = self._to_device(batch)
        losses = self._hybrid(batch) if self.is_hybrid else self._direct(batch)
        self.optimizer.zero_grad()
        losses["total"].backward()
        self.optimizer.step()
        return losses

    def fit(self, dataloader) -> None:
        for _ in range(self.config.epochs):
            self.student.train()
            for batch in dataloader:
                self.training_step(batch)
            if self.scheduler is not None:
                self.scheduler.step()


__all__ = ["distillation_loss", "DistillConfig", "DistillTrainer"]
