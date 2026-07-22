from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import amp, nn
from torch.nn.utils import clip_grad_norm_

from hybrid.training.losses import (
    compute_distillation_losses,
    compute_metrics,
    compute_regularization_ssl_losses,
    compute_stage1_losses,
    compute_stage2_losses,
)


class ForwardAndLossRunner:
    def __init__(
        self,
        *,
        model: nn.Module,
        teacher_model: nn.Module | None,
        device: torch.device,
        stage: str,
        loss_cfg: dict[str, Any],
        distillation_cfg: dict[str, Any],
        regularization_ssl_cfg: dict[str, Any],
        pruning_cfg: dict[str, Any],
        amp_enabled: bool,
        active_head: str,
        pruning_regularizer_fn,
    ) -> None:
        self.model = model
        self.teacher_model = teacher_model
        self.device = device
        self.stage = stage
        self.loss_cfg = loss_cfg
        self.distillation_cfg = distillation_cfg
        self.regularization_ssl_cfg = regularization_ssl_cfg
        self.pruning_cfg = pruning_cfg
        self.amp_enabled = amp_enabled
        self.active_head = active_head
        self.pruning_regularizer_fn = pruning_regularizer_fn

    def __call__(
        self,
        batch: dict[str, Any],
        width_ratio: float,
        *,
        training: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        teacher_outputs = None
        if training and self.teacher_model is not None:
            with torch.no_grad():
                with amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    teacher_outputs = self.teacher_model(batch, width_ratio=1.0)
        with amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            outputs = self.model(batch, width_ratio=width_ratio)
            if self.stage == "stage1":
                losses = compute_stage1_losses(batch, outputs, self.loss_cfg, active_head=self.active_head)
            else:
                losses = compute_stage2_losses(batch, outputs, self.loss_cfg, active_head=self.active_head)
            distillation_losses = (
                compute_distillation_losses(batch, outputs, teacher_outputs, self.distillation_cfg, stage=self.stage)
                if training and teacher_outputs is not None
                else {}
            )
            regularization_ssl_losses = (
                compute_regularization_ssl_losses(
                    batch,
                    outputs,
                    self.regularization_ssl_cfg,
                    teacher_outputs=teacher_outputs,
                    stage=self.stage,
                )
                if training
                else {}
            )
            pruning_losses = self.pruning_regularizer_fn(width_ratio, {"pruning": self.pruning_cfg}, self.device) if training else {}
            if distillation_losses:
                losses.update(distillation_losses)
            if regularization_ssl_losses:
                losses.update(regularization_ssl_losses)
            if pruning_losses:
                losses.update(pruning_losses)
            if "loss_ssl_total" not in losses:
                ssl_total = torch.zeros((), device=self.device)
                if "loss_distillation_total" in losses:
                    ssl_total = ssl_total + losses["loss_distillation_total"]
                if "loss_regularization_ssl_total" in losses:
                    ssl_total = ssl_total + losses["loss_regularization_ssl_total"]
                if ssl_total.detach().abs().item() > 0.0:
                    losses["loss_ssl_total"] = ssl_total
            if "loss_distillation_total" in losses:
                losses["loss_total"] = losses["loss_total"] + losses["loss_distillation_total"]
            if "loss_regularization_ssl_total" in losses:
                losses["loss_total"] = losses["loss_total"] + losses["loss_regularization_ssl_total"]
            if (
                "loss_ssl_total" in losses
                and "loss_distillation_total" not in losses
                and "loss_regularization_ssl_total" not in losses
            ):
                losses["loss_total"] = losses["loss_total"] + losses["loss_ssl_total"]
            if "loss_pruning_total" in losses:
                losses["loss_total"] = losses["loss_total"] + losses["loss_pruning_total"]
            metrics = compute_metrics(batch, outputs)
            metrics["metric_active_width"] = torch.tensor(float(width_ratio), device=self.device)
        return losses, metrics, losses["loss_total"]


@dataclass
class EpochRunResult:
    totals: dict[str, float]
    count: int

    def averages(self) -> dict[str, float]:
        return {key: value / max(1, self.count) for key, value in self.totals.items()}


class _BaseStepRunner:
    def __init__(
        self,
        *,
        model: nn.Module,
        loader,
        optimizer,
        scaler,
        teacher_model: nn.Module | None,
        grad_accum_steps: int,
        grad_clip_norm: float,
        stage: str,
        pruning_cfg: dict[str, Any],
        forward_once,
        console_logger,
        epoch: int | None,
        total_epochs: int | None,
        total_steps: int,
        phase: str,
        best_metric_name: str | None,
        best_metric_value: float | None,
        current_lr_fn,
        maybe_run_hessian_refresh_fn,
        optimizer_step_kwargs_fn,
        optimizer_step_index_fn,
        filter_stage_stats_fn,
        ema_update_fn,
        unwrap_model_fn,
        distillation_cfg: dict[str, Any],
    ) -> None:
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.scaler = scaler
        self.teacher_model = teacher_model
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.grad_clip_norm = float(grad_clip_norm)
        self.stage = stage
        self.pruning_cfg = pruning_cfg
        self.forward_once = forward_once
        self.console_logger = console_logger
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.phase = phase
        self.best_metric_name = best_metric_name
        self.best_metric_value = best_metric_value
        self.current_lr_fn = current_lr_fn
        self.maybe_run_hessian_refresh_fn = maybe_run_hessian_refresh_fn
        self.optimizer_step_kwargs_fn = optimizer_step_kwargs_fn
        self.optimizer_step_index_fn = optimizer_step_index_fn
        self.filter_stage_stats_fn = filter_stage_stats_fn
        self.ema_update_fn = ema_update_fn
        self.unwrap_model_fn = unwrap_model_fn
        self.distillation_cfg = distillation_cfg
        self.totals: dict[str, float] = {}
        self.count = 0

    def _record_totals(self, combined: dict[str, Any], *, sampled: bool = False, increment: int = 1) -> None:
        for key, value in combined.items():
            scalar = float(value) if sampled else float(value.detach().cpu().item())
            self.totals[key] = self.totals.get(key, 0.0) + scalar
        self.count += int(increment)

    def _log_progress(self, *, step: int) -> None:
        if self.console_logger is None or self.epoch is None or self.total_epochs is None:
            return
        running_stats = {key: value / max(1, self.count) for key, value in self.totals.items()}
        self.console_logger.step(
            phase=self.phase,
            epoch=self.epoch,
            total_epochs=self.total_epochs,
            step=step,
            total_steps=self.total_steps,
            running_stats=running_stats,
            lr=None if self.optimizer is None else self.current_lr_fn(self.optimizer),
            best_metric_name=self.best_metric_name,
            best_metric_value=self.best_metric_value,
        )

    def _ema_update_teacher(self) -> None:
        if self.teacher_model is None:
            return
        if not bool(self.distillation_cfg.get("ema_update_enabled", True)):
            return
        if bool(getattr(self.teacher_model, "is_fixed_ensemble_teacher", False)):
            return
        self.ema_update_fn(
            self.teacher_model,
            self.unwrap_model_fn(self.model),
            float(self.distillation_cfg.get("ema_decay", 0.996)),
        )

    def _maybe_run_hessian_refresh(self, *, batch: dict[str, Any], width_ratio: float) -> None:
        if self.optimizer is None:
            return
        opt_step_index = self.optimizer_step_index_fn(self.optimizer)
        self.maybe_run_hessian_refresh_fn(
            optimizer=self.optimizer,
            scaler=self.scaler,
            batch=batch,
            width_ratio=width_ratio,
            optimizer_step_index=opt_step_index,
            forward_once=self.forward_once,
        )


class ExactMarsStepRunner(_BaseStepRunner):
    def __init__(self, *, load_batch_group_fn, manual_unscale_optimizer_grads_fn, device, training: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_batch_group_fn = load_batch_group_fn
        self.manual_unscale_optimizer_grads_fn = manual_unscale_optimizer_grads_fn
        self.device = device
        self.training = training

    def run(self) -> EpochRunResult:
        loader_iter = iter(self.loader)
        current_group = self.load_batch_group_fn(
            loader_iter,
            group_size=self.grad_accum_steps,
            device=self.device,
            pruning_cfg=self.pruning_cfg,
            training=self.training,
        )
        processed_steps = 0
        while current_group:
            next_group = self.load_batch_group_fn(
                loader_iter,
                group_size=self.grad_accum_steps,
                device=self.device,
                pruning_cfg=self.pruning_cfg,
                training=self.training,
            )
            if next_group:
                self.optimizer.zero_grad(set_to_none=True)
                for future_batch, future_width_ratio in next_group:
                    _, _, future_loss_total = self.forward_once(future_batch, future_width_ratio)
                    future_scaled_loss = future_loss_total / max(1, self.grad_accum_steps)
                    if self.scaler is None:
                        future_scaled_loss.backward()
                    else:
                        self.scaler.scale(future_scaled_loss).backward()
                if self.scaler is not None:
                    self.manual_unscale_optimizer_grads_fn(self.optimizer, self.scaler)
                if self.grad_clip_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.update_previous_grad()
                self.optimizer.zero_grad(set_to_none=True)

            current_totals: dict[str, float] = {}
            for batch, width_ratio in current_group:
                losses, metrics, loss_total = self.forward_once(batch, width_ratio)
                scaled_loss = loss_total / max(1, self.grad_accum_steps)
                if self.scaler is None:
                    scaled_loss.backward()
                else:
                    self.scaler.scale(scaled_loss).backward()
                for key, value in self.filter_stage_stats_fn(self.stage, {**losses, **metrics}).items():
                    current_totals[key] = current_totals.get(key, 0.0) + float(value.detach().cpu().item())

            step_kwargs = self.optimizer_step_kwargs_fn(
                self.optimizer,
                batch=current_group[-1][0],
                grad_accum_steps=self.grad_accum_steps,
            )
            if self.scaler is None:
                if self.grad_clip_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step(**step_kwargs)
            else:
                if step_kwargs:
                    raise NotImplementedError("Optimizers with step kwargs are not yet supported with AMP/GradScaler in HBTXR.")
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self._ema_update_teacher()
            self.optimizer.zero_grad(set_to_none=True)
            if next_group:
                self.optimizer.update_last_grad()

            for key, value in current_totals.items():
                self.totals[key] = self.totals.get(key, 0.0) + value
            self.count += len(current_group)
            processed_steps += len(current_group)
            self._log_progress(step=processed_steps)
            current_group = next_group
        return EpochRunResult(self.totals, self.count)


class SampledParamStepRunner(_BaseStepRunner):
    def __init__(
        self,
        *,
        move_to_device_fn,
        resolve_pruning_width_fn,
        enabled_heads: dict[str, bool],
        active_head: str,
        training: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.move_to_device_fn = move_to_device_fn
        self.resolve_pruning_width_fn = resolve_pruning_width_fn
        self.enabled_heads = enabled_heads
        self.active_head = active_head
        self.training = training

    def run(self) -> EpochRunResult:
        if self.scaler is not None:
            raise NotImplementedError("Sampled-parameter optimizers are not yet supported with AMP/GradScaler in HBTXR.")
        last_batch = None
        last_width_ratio = None
        for step, batch in enumerate(self.loader, start=1):
            batch = self.move_to_device_fn(batch)
            width_ratio = self.resolve_pruning_width_fn({"pruning": self.pruning_cfg}, training=self.training)
            sample_totals: dict[str, float] = {}
            mc_samples = max(1, int(getattr(self.optimizer, "mc_samples", 1)))
            for _ in range(mc_samples):
                self.optimizer.zero_grad(set_to_none=True)
                with self.optimizer.sampled_params(train=True):
                    losses, metrics, loss_total = self.forward_once(batch, width_ratio)
                    sampled_loss = loss_total / max(1, self.grad_accum_steps)
                    sampled_loss.backward()
                    if self.grad_clip_norm > 0:
                        clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                combined = self.filter_stage_stats_fn(
                    self.stage,
                    {**losses, **metrics},
                    enabled_heads=self.enabled_heads,
                    active_head=self.active_head,
                )
                for key, value in combined.items():
                    sample_totals[key] = sample_totals.get(key, 0.0) + float(value.detach().cpu().item())
            combined = {key: value / float(mc_samples) for key, value in sample_totals.items()}
            if step % max(1, self.grad_accum_steps) == 0:
                self.optimizer.step()
                self._maybe_run_hessian_refresh(batch=batch, width_ratio=width_ratio)
                self._ema_update_teacher()
                self.optimizer.zero_grad(set_to_none=True)
            self._record_totals(combined, sampled=True, increment=1)
            self._log_progress(step=step)
            last_batch = batch
            last_width_ratio = width_ratio

        if self.training and self.count % max(1, self.grad_accum_steps) != 0 and last_batch is not None and last_width_ratio is not None:
            self.optimizer.step()
            self._maybe_run_hessian_refresh(batch=last_batch, width_ratio=last_width_ratio)
            self._ema_update_teacher()
            self.optimizer.zero_grad(set_to_none=True)
        return EpochRunResult(self.totals, self.count)


class StandardStepRunner(_BaseStepRunner):
    def __init__(
        self,
        *,
        move_to_device_fn,
        resolve_pruning_width_fn,
        enabled_heads: dict[str, bool],
        active_head: str,
        training: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.move_to_device_fn = move_to_device_fn
        self.resolve_pruning_width_fn = resolve_pruning_width_fn
        self.enabled_heads = enabled_heads
        self.active_head = active_head
        self.training = training

    def _step_optimizer(self, *, batch: dict[str, Any], width_ratio: float) -> None:
        step_kwargs = self.optimizer_step_kwargs_fn(self.optimizer, batch=batch, grad_accum_steps=self.grad_accum_steps)
        if self.scaler is None:
            if self.grad_clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step(**step_kwargs)
            self._maybe_run_hessian_refresh(batch=batch, width_ratio=width_ratio)
        else:
            if step_kwargs:
                raise NotImplementedError("Optimizers with step kwargs are not yet supported with AMP/GradScaler in HBTXR.")
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._maybe_run_hessian_refresh(batch=batch, width_ratio=width_ratio)
        self._ema_update_teacher()
        self.optimizer.zero_grad(set_to_none=True)

    def run(self) -> EpochRunResult:
        last_batch = None
        last_width_ratio = None
        for step, batch in enumerate(self.loader, start=1):
            batch = self.move_to_device_fn(batch)
            width_ratio = self.resolve_pruning_width_fn({"pruning": self.pruning_cfg}, training=self.training)
            losses, metrics, loss_total = self.forward_once(batch, width_ratio)
            if self.training:
                scaled_loss = loss_total / max(1, self.grad_accum_steps)
                if self.scaler is None:
                    scaled_loss.backward()
                else:
                    self.scaler.scale(scaled_loss).backward()
                if step % max(1, self.grad_accum_steps) == 0:
                    self._step_optimizer(batch=batch, width_ratio=width_ratio)
            combined = self.filter_stage_stats_fn(
                self.stage,
                {**losses, **metrics},
                enabled_heads=self.enabled_heads,
                active_head=self.active_head,
            )
            self._record_totals(combined, sampled=False, increment=1)
            self._log_progress(step=step)
            last_batch = batch
            last_width_ratio = width_ratio

        if self.training and self.count % max(1, self.grad_accum_steps) != 0 and last_batch is not None and last_width_ratio is not None:
            self._step_optimizer(batch=last_batch, width_ratio=last_width_ratio)
        return EpochRunResult(self.totals, self.count)


__all__ = [
    "EpochRunResult",
    "ExactMarsStepRunner",
    "ForwardAndLossRunner",
    "SampledParamStepRunner",
    "StandardStepRunner",
]
