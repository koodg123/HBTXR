from __future__ import annotations

import json
import itertools
import inspect
import math
import random
import sys
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import amp, nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - plain-text fallback is covered instead
    tqdm = None

from hybrid.data.loader import make_loader_by_mode
from hybrid.optim.common import optimizer_hypers_dir, write_json
from hybrid.optim.registry import build_optimizer
from hybrid.optim.lr_schedulers import build_lr_scheduler
from hybrid.models.export_pruned import export_structural_student
from hybrid.models.pruning import (
    is_legacy_masking_enabled,
    is_structural_pruning_enabled,
    normalize_compression_cfg,
    resolve_student_model_spec,
)
from hybrid.models.preload.pretrained_loader import load_pretrained_weights, write_pretrained_report
from .checkpoints import (
    checkpoint_summary_metadata,
    flatten_best_checkpoints_from_state,
    load_checkpoint as _load_checkpoint,
    load_model_state as _load_model_state,
    save_checkpoint as _save_checkpoint,
    save_teacher_checkpoint as _save_teacher_checkpoint,
    unwrap_model as _unwrap_model,
)
from .model_factory import build_model
from .model_factory import create_teacher_model as _create_teacher_model
from .step_runner import (
    ExactMarsStepRunner,
    ForwardAndLossRunner,
    SampledParamStepRunner,
    StandardStepRunner,
)
from .teacher import ema_update as _ema_update


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
    "mask_valid",
    "valid_track",
    "aux_target",
}


class _ConsoleLogger:
    def __init__(self, training_cfg: dict[str, Any]) -> None:
        console_cfg = dict(training_cfg.get("console_log") or {})
        self.enabled = bool(console_cfg.get("enabled", True))
        self.style = str(console_cfg.get("style", "tqdm")).strip().lower()
        step_interval = console_cfg.get("step_interval")
        self.step_interval = max(1, int(training_cfg.get("log_every", 10))) if step_interval in (None, "") else max(1, int(step_interval))
        self.epoch_interval = max(1, int(training_cfg.get("log_epoch_every", 1)))
        self.show_lr = bool(console_cfg.get("show_lr", True))
        self.show_best_metric = bool(console_cfg.get("show_best_metric", True))
        self.show_width = bool(console_cfg.get("show_width", True))
        self.stream = sys.stdout
        self._progress = None
        self._warned_missing_tqdm = False

    @property
    def _use_tqdm(self) -> bool:
        return self.enabled and self.style == "tqdm" and tqdm is not None

    def write(self, message: str) -> None:
        if not self.enabled:
            return
        if self._progress is not None and self._use_tqdm:
            tqdm.write(message, file=self.stream)
            return
        print(message, file=self.stream, flush=True)

    def maybe_warn_missing_tqdm(self) -> None:
        if not self.enabled or self.style != "tqdm" or tqdm is not None or self._warned_missing_tqdm:
            return
        self._warned_missing_tqdm = True
        self.write("[console-log] tqdm is not installed; using plain-text progress fallback")

    def start_run(
        self,
        *,
        stage: str,
        mode: str,
        experiment_name: str,
        requested_device: str,
        resolved_device: str,
        resolved_device_ids: list[int],
        batch_size: int,
        num_workers: int,
        train_samples: int,
        val_samples: int,
    ) -> None:
        if not self.enabled:
            return
        self.maybe_warn_missing_tqdm()
        self.write(
            "[train-start] "
            f"stage={stage} mode={mode} experiment={experiment_name} "
            f"requested_device={requested_device} resolved_device={resolved_device} "
            f"resolved_device_ids={resolved_device_ids} batch_size={batch_size} "
            f"num_workers={num_workers} train_samples={train_samples} val_samples={val_samples}"
        )

    def start_epoch(self, *, phase: str, epoch: int, total_epochs: int, total_steps: int) -> None:
        if not self.enabled:
            return
        desc = f"{phase} {epoch}/{total_epochs}"
        if self._use_tqdm:
            self._progress = tqdm(total=max(1, total_steps), desc=desc, file=self.stream, dynamic_ncols=True, leave=True)
            return
        self.write(f"[epoch-start] phase={phase} epoch={epoch}/{total_epochs} steps={total_steps}")

    def step(
        self,
        *,
        phase: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        running_stats: dict[str, float],
        lr: float | None,
        best_metric_name: str | None,
        best_metric_value: float | None,
    ) -> None:
        if not self.enabled:
            return
        should_log = step == 1 or step == total_steps or step % self.step_interval == 0
        if self._use_tqdm:
            if self._progress is None:
                self.start_epoch(phase=phase, epoch=epoch, total_epochs=total_epochs, total_steps=total_steps)
            if self._progress is not None:
                self._progress.update(1)
                if should_log:
                    postfix: dict[str, str] = {}
                    if "loss_total" in running_stats:
                        postfix["loss"] = f"{running_stats['loss_total']:.4f}"
                    if self.show_lr and lr is not None:
                        postfix["lr"] = f"{lr:.2e}"
                    if self.show_width and "metric_active_width" in running_stats:
                        postfix["w"] = f"{running_stats['metric_active_width']:.2f}"
                    if self.show_best_metric and best_metric_name and best_metric_value is not None:
                        postfix["best"] = f"{best_metric_name}={best_metric_value:.4f}"
                    self._progress.set_postfix(postfix, refresh=True)
            return
        if should_log:
            parts = [f"[step] phase={phase} epoch={epoch}/{total_epochs} step={step}/{total_steps}"]
            if "loss_total" in running_stats:
                parts.append(f"loss={running_stats['loss_total']:.4f}")
            if self.show_lr and lr is not None:
                parts.append(f"lr={lr:.2e}")
            if self.show_width and "metric_active_width" in running_stats:
                parts.append(f"active_width={running_stats['metric_active_width']:.2f}")
            if self.show_best_metric and best_metric_name and best_metric_value is not None:
                parts.append(f"best_{best_metric_name}={best_metric_value:.4f}")
            self.write(" ".join(parts))

    def close_epoch(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None

    def epoch_summary(
        self,
        *,
        epoch: int,
        total_epochs: int,
        lr: float,
        train_stats: dict[str, float],
        val_stats: dict[str, float] | None,
        best_metric_name: str,
        best_metric_value: float | None,
        saved_checkpoints: list[str],
        early_counter: int,
        early_patience: int,
    ) -> None:
        if not self.enabled or epoch % self.epoch_interval != 0:
            return
        parts = [
            f"[epoch-summary] epoch={epoch}/{total_epochs}",
            f"lr={lr:.2e}",
        ]
        if "loss_total" in train_stats:
            parts.append(f"train_loss={train_stats['loss_total']:.4f}")
        if val_stats and "loss_total" in val_stats:
            parts.append(f"val_loss={val_stats['loss_total']:.4f}")
        if best_metric_value is not None:
            parts.append(f"best_{best_metric_name}={best_metric_value:.4f}")
        if saved_checkpoints:
            parts.append(f"saved={','.join(saved_checkpoints)}")
        if early_patience > 0:
            parts.append(f"early_counter={early_counter}/{early_patience}")
        self.write(" ".join(parts))


def _enabled_heads_from_model(model: nn.Module) -> dict[str, bool]:
    unwrapped = _unwrap_model(model)
    return {
        "eye": getattr(unwrapped, "eye_head", None) is not None,
        "search": (
            getattr(unwrapped, "search_head", None) is not None
            or getattr(unwrapped, "search_bbox_aux_head", None) is not None
            or getattr(unwrapped, "search_obb_aux_head", None) is not None
        ),
        "event": (
            getattr(unwrapped, "event_head", None) is not None
            or getattr(unwrapped, "event_bbox_aux_head", None) is not None
            or getattr(unwrapped, "event_obb_aux_head", None) is not None
        ),
        "track": getattr(unwrapped, "track_head", None) is not None,
        "mask": getattr(unwrapped, "mask_head", None) is not None,
        "aux": getattr(unwrapped, "aux_head", None) is not None,
    }


def _filter_stage_stats(
    stage: str,
    stats: dict[str, float | torch.Tensor],
    *,
    enabled_heads: dict[str, bool] | None = None,
    active_head: str = "all",
) -> dict[str, float | torch.Tensor]:
    enabled_heads = dict(enabled_heads or {})
    active_head = str(active_head or "all").strip().lower()
    filtered: dict[str, float | torch.Tensor] = {}
    for key, value in stats.items():
        if not enabled_heads.get("eye", False) and key.startswith("loss_eye"):
            continue
        if not enabled_heads.get("mask", False) and key.startswith("loss_mask"):
            continue
        if not enabled_heads.get("search", False) and (key.startswith("loss_search_") or key.startswith("loss_loss_search_") or key.startswith("metric_search_")):
            continue
        if not enabled_heads.get("event", False) and (key.startswith("loss_event_") or key.startswith("metric_event_")):
            continue
        if not enabled_heads.get("track", False) and (key.startswith("loss_track_") or key.startswith("metric_track_")):
            continue
        if not enabled_heads.get("aux", False) and key == "loss_aux":
            continue
        if active_head == "all":
            if str(stage).strip().lower() == "stage1" and (key.startswith("loss_event_") or key.startswith("metric_event_") or key.startswith("loss_track_") or key.startswith("metric_track_") or key == "loss_consistency"):
                continue
            if str(stage).strip().lower() == "stage2" and (key.startswith("loss_eye") or key.startswith("loss_mask")):
                continue
        if (not enabled_heads.get("search", False) or not enabled_heads.get("track", False)) and key == "loss_consistency":
            continue
        if not enabled_heads.get("search", False) and not enabled_heads.get("track", False) and key == "loss_constraint_center":
            continue
        filtered[key] = value
    return filtered


def _safe_cuda_available() -> tuple[bool, str | None]:
    try:
        return bool(torch.cuda.is_available()), None
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{type(exc).__name__}: {exc}"


def _safe_cuda_device_count() -> tuple[int, str | None]:
    try:
        return int(torch.cuda.device_count()), None
    except Exception as exc:  # pragma: no cover - defensive
        return 0, f"{type(exc).__name__}: {exc}"


def _parse_requested_device_spec(device_spec: str) -> dict[str, Any]:
    spec = str(device_spec or "cpu").strip()
    lowered = spec.lower()
    if lowered == "cpu":
        return {"requested": spec, "kind": "cpu", "device_ids": []}
    if lowered == "multi-gpu":
        return {"requested": spec, "kind": "cuda", "device_ids": None}
    if "," in spec:
        ids: list[int] = []
        for item in spec.split(","):
            token = item.strip().lower()
            if not token:
                continue
            if token.startswith("cuda:"):
                token = token.split(":", 1)[1].strip()
            if token.isdigit():
                ids.append(int(token))
                continue
            raise ValueError(
                f"Unsupported CUDA device token: {item!r}. "
                "Examples: cpu, cuda:0, cuda:0,cuda:1, cuda:0,1, 0,1, multi-gpu"
            )
        if not ids:
            raise ValueError(
                f"Unsupported device specification: {spec!r}. "
                "Examples: cpu, cuda:0, cuda:0,cuda:1, cuda:0,1, 0,1, multi-gpu"
            )
        return {"requested": spec, "kind": "cuda", "device_ids": ids}
    if lowered.startswith("cuda:"):
        token = lowered.split(":", 1)[1].strip()
        if token.isdigit():
            return {"requested": spec, "kind": "cuda", "device_ids": [int(token)]}
    raise ValueError(
        f"Unsupported device specification: {spec!r}. "
        "Examples: cpu, cuda:0, cuda:0,cuda:1, cuda:0,1, 0,1, multi-gpu"
    )


def _cuda_request_error(parsed: dict[str, Any], *, available: bool, available_error: str | None, device_count: int, count_error: str | None) -> RuntimeError:
    requested = str(parsed.get("requested", ""))
    ids = parsed.get("device_ids")
    return RuntimeError(
        "CUDA device request could not be satisfied. "
        f"requested_device={requested!r} parsed_device_ids={ids} "
        f"cuda_is_available={available} cuda_available_error={available_error!r} "
        f"cuda_device_count={device_count} cuda_count_error={count_error!r}. "
        "Use --device cpu for CPU execution or a valid CUDA spec such as --device cuda:0,cuda:1."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _serialize_stats(stats: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in stats.items()}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metric_mode(metric_name: str) -> str:
    lowered = metric_name.lower()
    if lowered.endswith("_pct") or lowered.endswith("_acc") or lowered.endswith("_score"):
        return "max"
    return "min"


def _is_better(candidate: float, current: float | None, *, mode: str, min_delta: float = 0.0) -> bool:
    if current is None:
        return True
    if mode == "max":
        return candidate > current + min_delta
    return candidate < current - min_delta


def _default_best_metric(stage: str) -> str:
    return "metric_search_p10_pct" if stage == "stage1" else "metric_track_p10_pct"


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    total_epochs: int,
    *,
    optimizer_meta: dict[str, Any] | None = None,
):
    return build_lr_scheduler(optimizer, training_cfg, total_epochs, optimizer_meta=optimizer_meta)


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0


def _set_optimizer_mode(optimizer: torch.optim.Optimizer | None, *, training: bool) -> None:
    if optimizer is None:
        return
    if training:
        fn = getattr(optimizer, "train", None)
    else:
        fn = getattr(optimizer, "eval", None)
    if callable(fn):
        fn()


def _write_optimizer_reports(
    *,
    output_dir: Path,
    optimizer_summary: dict[str, Any],
    optimizer_meta: dict[str, Any],
) -> None:
    hypers_dir = optimizer_hypers_dir(output_dir)
    hypers_dir.mkdir(parents=True, exist_ok=True)
    write_json(hypers_dir / "optimizer_resolved.json", optimizer_summary)
    diff_summary = {
        "name": optimizer_summary.get("name"),
        "implemented": bool(optimizer_meta.get("implemented", False)),
        "source": optimizer_meta.get("source"),
        "upstream": optimizer_meta.get("upstream"),
        "diff_doc": optimizer_meta.get("diff_doc"),
        "algorithmic_diff": bool(optimizer_meta.get("algorithmic_diff", False)),
        "modifiers": optimizer_meta.get("modifiers") or {},
    }
    write_json(hypers_dir / "optimizer_diff_summary.json", diff_summary)


def _pattern_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _name_matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch(name, pattern) or pattern in name for pattern in patterns)


def _apply_trainable_filter(
    model: nn.Module,
    training_cfg: dict[str, Any],
    *,
    output_dir: Path,
    console_logger: _ConsoleLogger,
) -> dict[str, Any] | None:
    trainable_cfg = training_cfg.get("trainable") or {}
    include = _pattern_list(trainable_cfg.get("include") or training_cfg.get("trainable_include"))
    exclude = _pattern_list(trainable_cfg.get("exclude") or training_cfg.get("trainable_exclude"))
    if not include and not exclude:
        return None

    base = model.module if isinstance(model, nn.DataParallel) else model
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    total_params = 0
    trainable_params = 0
    for name, param in base.named_parameters():
        total_params += int(param.numel())
        should_train = True if not include else _name_matches_any(name, include)
        if exclude and _name_matches_any(name, exclude):
            should_train = False
        param.requires_grad_(should_train)
        if should_train:
            trainable_names.append(name)
            trainable_params += int(param.numel())
        else:
            frozen_names.append(name)

    if not trainable_names:
        raise ValueError(f"training.trainable filter left no trainable parameters. include={include} exclude={exclude}")

    report = {
        "include": include,
        "exclude": exclude,
        "trainable_param_count": trainable_params,
        "total_param_count": total_params,
        "frozen_param_count": total_params - trainable_params,
        "trainable_tensor_count": len(trainable_names),
        "frozen_tensor_count": len(frozen_names),
        "trainable_names": trainable_names,
        "frozen_name_sample": frozen_names[:64],
    }
    write_json(optimizer_hypers_dir(output_dir) / "trainable_filter.json", report)
    console_logger.write(
        "[trainable-filter] "
        f"trainable_tensors={len(trainable_names)} frozen_tensors={len(frozen_names)} "
        f"trainable_params={trainable_params}/{total_params}"
    )
    return report


def make_loader(manifest_path: str, cfg: dict, shuffle: bool):
    return make_loader_by_mode(manifest_path, cfg, shuffle)


def _ssl_enabled(cfg: dict[str, Any]) -> bool:
    normalized_cfg = normalize_compression_cfg(cfg)
    return bool((normalized_cfg.get("regularization_ssl") or {}).get("enabled", False))


def _pruning_enabled(cfg: dict[str, Any]) -> bool:
    return bool((normalize_compression_cfg(cfg).get("pruning") or {}).get("enabled", False))


def _resolve_pruning_width(cfg: dict[str, Any], *, training: bool) -> float:
    normalized_cfg = normalize_compression_cfg(cfg)
    if is_structural_pruning_enabled(normalized_cfg):
        return resolve_student_model_spec(normalized_cfg).structural_width_ratio
    pruning_cfg = normalized_cfg.get("pruning") or {}
    if not is_legacy_masking_enabled(normalized_cfg):
        return 1.0
    candidates = [float(v) for v in pruning_cfg.get("width_candidates", [1.0]) if float(v) > 0.0]
    if training and candidates:
        strategy = str(pruning_cfg.get("sample_strategy", "random")).strip().lower()
        if strategy == "min":
            return min(candidates)
        if strategy == "max":
            return max(candidates)
        return random.choice(candidates)
    return float(pruning_cfg.get("student_width", 1.0))


def _pruning_regularizer(width_ratio: float, cfg: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    normalized_cfg = normalize_compression_cfg(cfg)
    pruning_cfg = normalized_cfg.get("pruning") or {}
    if not is_legacy_masking_enabled(normalized_cfg):
        return {}
    weight = float(pruning_cfg.get("width_loss_weight", 0.0))
    if weight <= 0.0:
        return {}
    loss = torch.tensor(float(width_ratio), device=device) * weight
    return {"loss_pruning_width": loss, "loss_pruning_total": loss}


def _resolve_device_and_wrap(model: nn.Module, device_spec: str) -> tuple[nn.Module, torch.device, dict[str, Any]]:
    parsed = _parse_requested_device_spec(device_spec)
    if parsed["kind"] == "cpu":
        device = torch.device("cpu")
        resolved = {
            "requested_device": parsed["requested"],
            "resolved_device": str(device),
            "resolved_device_ids": [],
            "data_parallel": False,
            "cuda_available": False,
            "cuda_device_count": 0,
        }
        return model.to(device), device, resolved

    cuda_available, available_error = _safe_cuda_available()
    device_count, count_error = _safe_cuda_device_count()
    if not cuda_available:
        raise _cuda_request_error(
            parsed,
            available=cuda_available,
            available_error=available_error,
            device_count=device_count,
            count_error=count_error,
        )

    requested_ids = parsed["device_ids"]
    resolved_ids = list(range(device_count)) if requested_ids is None else [int(value) for value in requested_ids]
    if not resolved_ids:
        raise _cuda_request_error(
            parsed,
            available=cuda_available,
            available_error=available_error,
            device_count=device_count,
            count_error=count_error,
        )
    invalid_ids = [idx for idx in resolved_ids if idx < 0 or idx >= device_count]
    if invalid_ids:
        raise RuntimeError(
            "CUDA device ids are out of range. "
            f"requested_device={parsed['requested']!r} resolved_device_ids={resolved_ids} invalid_ids={invalid_ids} "
            f"cuda_device_count={device_count}. Use a valid CUDA spec such as --device cuda:0,cuda:1."
        )

    device = torch.device(f"cuda:{resolved_ids[0]}")
    model = model.to(device)
    if len(resolved_ids) > 1:
        model = nn.DataParallel(model, device_ids=resolved_ids)
    resolved = {
        "requested_device": parsed["requested"],
        "resolved_device": str(device),
        "resolved_device_ids": resolved_ids,
        "data_parallel": len(resolved_ids) > 1,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
    }
    return model, device, resolved


def _maybe_apply_pretrained(
    *,
    model: nn.Module,
    cfg: dict[str, Any],
    stage: str,
    output_dir: Path,
    resume_checkpoint: str | None,
) -> dict[str, Any] | None:
    pretrained_cfg = ((cfg.get("model") or {}).get("pretrained") or {})
    if resume_checkpoint:
        return None
    if not bool(pretrained_cfg.get("enabled", False)):
        return None
    load_stage = str(pretrained_cfg.get("load_stage", "stage1")).strip().lower()
    if load_stage not in {stage, "all", "any"}:
        return None
    checkpoint_path = pretrained_cfg.get("path")
    if not checkpoint_path:
        return None
    report = load_pretrained_weights(
        _unwrap_model(model),
        checkpoint_path=checkpoint_path,
        source=str(pretrained_cfg.get("source", "auto")),
        strict_shape=bool(pretrained_cfg.get("strict_shape", False)),
    )
    report_path = output_dir.parent / "hypers" / ("pretrained_report_stage1.json" if stage == "stage1" else "pretrained_report.json")
    write_pretrained_report(report_path, report)
    return report


def _warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, *, epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    scale = max(epoch, 1) / float(warmup_epochs)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def _infer_batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, dict):
            nested = _infer_batch_size(value)
            if nested > 0:
                return nested
    return 1


def _manual_unscale_optimizer_grads(optimizer: torch.optim.Optimizer, scaler: amp.GradScaler | None) -> None:
    if scaler is None:
        return
    scale = float(scaler.get_scale())
    if scale == 0.0:
        return
    inv_scale = 1.0 / scale
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.detach().mul_(inv_scale)


def _load_batch_group(
    loader_iter,
    *,
    group_size: int,
    device: torch.device,
    pruning_cfg: dict[str, Any],
    training: bool,
) -> list[tuple[dict[str, Any], float]]:
    group: list[tuple[dict[str, Any], float]] = []
    for _ in range(max(1, group_size)):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        group.append(
            (
                move_to_device(batch, device),
                _resolve_pruning_width({"pruning": pruning_cfg}, training=training),
            )
        )
    return group


def _optimizer_step_kwargs(optimizer: torch.optim.Optimizer, *, batch: dict[str, Any], grad_accum_steps: int) -> dict[str, Any]:
    try:
        step_signature = inspect.signature(optimizer.step)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {}
    if "bs" not in step_signature.parameters:
        return {}
    batch_scale = float(getattr(optimizer, "hbtxr_step_bs_scale", 1.0))
    effective_batch = max(1, int(round(_infer_batch_size(batch) * max(1, grad_accum_steps) * batch_scale)))
    return {"bs": effective_batch}


def _optimizer_step_index(optimizer: torch.optim.Optimizer) -> int:
    current_step = getattr(optimizer, "current_step", None)
    if current_step is not None:
        try:
            return int(current_step)
        except Exception:  # pragma: no cover - defensive
            pass
    for state in getattr(optimizer, "state", {}).values():
        if not isinstance(state, dict) or "step" not in state:
            continue
        step = state["step"]
        if torch.is_tensor(step):
            return int(step.detach().cpu().item())
        try:
            return int(step)
        except Exception:  # pragma: no cover - defensive
            continue
    return 0


def _uses_exact_mars_optimizer(optimizer: torch.optim.Optimizer | None) -> bool:
    if optimizer is None:
        return False
    return bool(getattr(optimizer, "hbtxr_exact_multi_pass", False))


def _maybe_run_hessian_refresh(
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: amp.GradScaler | None,
    batch: dict[str, Any],
    width_ratio: float,
    optimizer_step_index: int,
    forward_once,
) -> None:
    if optimizer is None or not hasattr(optimizer, "update_hessian"):
        return
    interval = int(getattr(optimizer, "hbtxr_hess_interval", 0))
    if interval <= 0 or optimizer_step_index <= 0 or optimizer_step_index % interval != 0:
        return
    optimizer.zero_grad(set_to_none=True)
    _, _, hessian_loss = forward_once(batch, width_ratio)
    if scaler is None:
        hessian_loss.backward()
    else:
        scaler.scale(hessian_loss).backward()
        _manual_unscale_optimizer_grads(optimizer, scaler)
    optimizer.update_hessian()
    optimizer.zero_grad(set_to_none=True)


def _epoch_loop(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: amp.GradScaler | None,
    device: torch.device,
    stage: str,
    loss_cfg: dict[str, Any],
    distillation_cfg: dict[str, Any],
    regularization_ssl_cfg: dict[str, Any],
    pruning_cfg: dict[str, Any],
    teacher_model: nn.Module | None,
    amp_enabled: bool,
    grad_accum_steps: int,
    grad_clip_norm: float,
    epoch: int | None = None,
    total_epochs: int | None = None,
    phase: str = "train",
    console_logger: _ConsoleLogger | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
    active_head: str = "all",
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    uses_sampled_params = training and hasattr(optimizer, "sampled_params")
    uses_exact_mars = _uses_exact_mars_optimizer(optimizer)
    target_model = _unwrap_model(model)
    if hasattr(target_model, "set_epoch_context"):
        target_model.set_epoch_context(epoch)
    if teacher_model is not None and hasattr(teacher_model, "set_epoch_context"):
        teacher_model.set_epoch_context(epoch)
    model.train(training)
    if teacher_model is not None:
        teacher_model.eval()
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_steps = len(loader)
    if max_batches is not None and int(max_batches) > 0:
        total_steps = min(total_steps, int(max_batches))
        loader = itertools.islice(loader, int(max_batches))
    enabled_heads = _enabled_heads_from_model(model)
    if console_logger is not None and epoch is not None and total_epochs is not None:
        console_logger.start_epoch(phase=phase, epoch=epoch, total_epochs=total_epochs, total_steps=total_steps)

    forward_runner = ForwardAndLossRunner(
        model=model,
        teacher_model=teacher_model,
        device=device,
        stage=stage,
        loss_cfg=loss_cfg,
        distillation_cfg=distillation_cfg,
        regularization_ssl_cfg=regularization_ssl_cfg,
        pruning_cfg=pruning_cfg,
        amp_enabled=amp_enabled,
        active_head=active_head,
        pruning_regularizer_fn=_pruning_regularizer,
    )

    def _forward_once(batch: dict[str, Any], width_ratio: float):
        return forward_runner(batch, width_ratio, training=training)

    common_runner_kwargs = {
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "scaler": scaler,
        "teacher_model": teacher_model,
        "grad_accum_steps": grad_accum_steps,
        "grad_clip_norm": grad_clip_norm,
        "stage": stage,
        "pruning_cfg": pruning_cfg,
        "forward_once": _forward_once,
        "console_logger": console_logger,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "total_steps": total_steps,
        "phase": phase,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "current_lr_fn": _current_lr,
        "maybe_run_hessian_refresh_fn": _maybe_run_hessian_refresh,
        "optimizer_step_kwargs_fn": _optimizer_step_kwargs,
        "optimizer_step_index_fn": _optimizer_step_index,
        "filter_stage_stats_fn": _filter_stage_stats,
        "ema_update_fn": _ema_update,
        "unwrap_model_fn": _unwrap_model,
        "distillation_cfg": distillation_cfg,
    }

    if uses_exact_mars:
        epoch_runner = ExactMarsStepRunner(
            load_batch_group_fn=_load_batch_group,
            manual_unscale_optimizer_grads_fn=_manual_unscale_optimizer_grads,
            device=device,
            training=training,
            **common_runner_kwargs,
        )
    elif uses_sampled_params:
        epoch_runner = SampledParamStepRunner(
            move_to_device_fn=lambda batch: move_to_device(batch, device),
            resolve_pruning_width_fn=_resolve_pruning_width,
            enabled_heads=enabled_heads,
            active_head=active_head,
            training=training,
            **common_runner_kwargs,
        )
    else:
        epoch_runner = StandardStepRunner(
            move_to_device_fn=lambda batch: move_to_device(batch, device),
            resolve_pruning_width_fn=_resolve_pruning_width,
            enabled_heads=enabled_heads,
            active_head=active_head,
            training=training,
            **common_runner_kwargs,
        )

    try:
        result = epoch_runner.run()
    finally:
        if console_logger is not None:
            console_logger.close_epoch()

    return result.averages()


class TrainingSession:
    def __init__(
        self,
        *,
        cfg: dict,
        train_manifest: str,
        val_manifest: str | None,
        output_dir: str | Path,
        stage1_checkpoint: str | None = None,
        resume_checkpoint: str | None = None,
    ) -> None:
        self.cfg = normalize_compression_cfg(cfg)
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.output_dir = Path(output_dir)
        self.stage1_checkpoint = stage1_checkpoint
        self.resume_checkpoint = resume_checkpoint

    def _prepare_runtime(self) -> None:
        set_seed(int(self.cfg.get("seed", 42)))
        self.training_cfg = self.cfg.get("training") or {}
        self.loss_cfg = self.cfg.get("loss") or {}
        self.active_head = str((((self.cfg.get("model") or {}).get("heads") or {}).get("active", "all"))).strip().lower()
        self.stage = str(self.training_cfg.get("stage", "stage1")).strip().lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "history.json"
        self.history_jsonl_path = self.output_dir / "history.jsonl"
        self.train_log_path = self.output_dir / "train_log.txt"
        self.console_logger = _ConsoleLogger(self.training_cfg)

        self.train_loader = make_loader(self.train_manifest, self.cfg, shuffle=True)
        self.val_loader = make_loader(self.val_manifest, self.cfg, shuffle=False) if self.val_manifest else None

        optimizer_cfg = dict(self.training_cfg.get("optimizer") or {})
        optimizer_name = str(optimizer_cfg.get("name", "adamw")).strip().lower()
        if optimizer_name == "ivon":
            optimizer_kwargs = dict(optimizer_cfg.get("kwargs") or {})
            optimizer_kwargs.setdefault("ess", float(max(1, len(getattr(self.train_loader, "dataset", [])))))
            optimizer_cfg["kwargs"] = optimizer_kwargs
            self.training_cfg["optimizer"] = optimizer_cfg

        self.model = build_model(self.cfg, role="student")
        self.model, self.device, self.resolved_device = _resolve_device_and_wrap(self.model, str(self.training_cfg.get("device", "cpu")))
        self.trainable_filter_report = _apply_trainable_filter(
            self.model,
            self.training_cfg,
            output_dir=self.output_dir,
            console_logger=self.console_logger,
        )
        self.optimizer, self.optimizer_resolved, self.optimizer_meta, optimizer_summary = build_optimizer(self.model, self.cfg)
        self.scheduler = _build_scheduler(
            self.optimizer,
            self.training_cfg,
            int(self.training_cfg.get("epochs", 1)),
            optimizer_meta=self.optimizer_meta,
        )
        optimizer_summary["scheduler_enabled"] = self.scheduler is not None
        _write_optimizer_reports(
            output_dir=self.output_dir,
            optimizer_summary=optimizer_summary,
            optimizer_meta=self.optimizer_meta,
        )
        self.amp_enabled = bool(self.training_cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = amp.GradScaler("cuda", enabled=self.amp_enabled) if self.device.type == "cuda" else None
        if self.scaler is not None and not self.scaler.is_enabled():
            self.scaler = None

        self.start_epoch = 1
        self.history: list[dict[str, Any]] = []
        self.best_metrics: dict[str, dict[str, float | str]] = {}
        self.pretrained_report = None
        self.distillation_cfg = dict(self.cfg.get("distillation") or {})
        self.regularization_ssl_cfg = dict(self.cfg.get("regularization_ssl") or {})
        self.pruning_cfg = dict(self.cfg.get("pruning") or {})

    def _initialize_model_state(self) -> None:
        if self.stage == "stage2" and self.stage1_checkpoint:
            _load_checkpoint(model=self.model, optimizer=None, scheduler=None, scaler=None, checkpoint_path=self.stage1_checkpoint, strict=False)

        self.pretrained_report = _maybe_apply_pretrained(
            model=self.model,
            cfg=self.cfg,
            stage=self.stage,
            output_dir=self.output_dir,
            resume_checkpoint=self.resume_checkpoint,
        )
        if self.pretrained_report is not None:
            self.console_logger.write(
                "[pretrained] "
                f"loaded_count={self.pretrained_report.get('loaded_count', 0)} "
                f"partial={len(self.pretrained_report.get('partially_loaded', []))} "
                f"skipped={len(self.pretrained_report.get('skipped', []))}"
            )

        if self.resume_checkpoint:
            state = _load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                checkpoint_path=self.resume_checkpoint,
                strict=False,
            )
            self.start_epoch = int(state.get("epoch", 0)) + 1
            self.history = list(state.get("history") or [])
            self.best_metrics = dict(state.get("best_metrics") or {})
            self.console_logger.write(f"[resume] checkpoint={self.resume_checkpoint} start_epoch={self.start_epoch}")

        self.teacher_model = _create_teacher_model(self.model, self.cfg, self.device)
        if self.teacher_model is None:
            return

        teacher_checkpoint = self.distillation_cfg.get("teacher_init_checkpoint")
        if self.resume_checkpoint:
            teacher_resume = Path(self.resume_checkpoint).with_name("teacher_last.pt")
            if teacher_resume.exists():
                _load_checkpoint(model=self.teacher_model, optimizer=None, scheduler=None, scaler=None, checkpoint_path=teacher_resume, strict=False)
        elif teacher_checkpoint:
            _load_checkpoint(model=self.teacher_model, optimizer=None, scheduler=None, scaler=None, checkpoint_path=teacher_checkpoint, strict=False)
        elif self.stage == "stage2" and self.stage1_checkpoint:
            _load_checkpoint(model=self.teacher_model, optimizer=None, scheduler=None, scaler=None, checkpoint_path=self.stage1_checkpoint, strict=False)
        elif self.pretrained_report is not None:
            pretrained_cfg = ((self.cfg.get("model") or {}).get("pretrained") or {})
            checkpoint_path = pretrained_cfg.get("path")
            if checkpoint_path:
                load_pretrained_weights(
                    self.teacher_model,
                    checkpoint_path=checkpoint_path,
                    source=str(pretrained_cfg.get("source", "auto")),
                    strict_shape=bool(pretrained_cfg.get("strict_shape", False)),
                )
        elif bool(self.distillation_cfg.get("force_teacher_from_student", True)):
            _load_model_state(model=self.teacher_model, state_dict=_unwrap_model(self.model).state_dict(), strict=False)

    def _initialize_epoch_control(self) -> None:
        self.total_epochs = int(self.training_cfg.get("epochs", 1))
        self.best_metric_name = str(self.training_cfg.get("best_metric_name") or _default_best_metric(self.stage))
        self.best_metric_mode = _metric_mode(self.best_metric_name)
        early_cfg = self.training_cfg.get("early_stopping") or {}
        self.early_enabled = bool(early_cfg.get("enabled", False))
        self.early_patience = int(early_cfg.get("patience", 10))
        self.early_min_delta = float(early_cfg.get("min_delta", 0.0))
        self.early_start_epoch = int(early_cfg.get("start_epoch", 1))
        self.early_counter = 0
        self.best_control_value = None if self.best_metrics.get(self.best_metric_name) is None else float(self.best_metrics[self.best_metric_name]["value"])

        self.checkpoint_specs = (
            [{"metric": "metric_search_p10_pct", "filename": "best_search_p10.pt"}, {"metric": "metric_search_p5_pct", "filename": "best_search_p5.pt"}]
            if self.stage == "stage1"
            else [
                {"metric": "metric_track_p10_pct", "filename": "best_track_p10.pt"},
                {"metric": "metric_track_p5_pct", "filename": "best_track_p5.pt"},
                {"metric": "metric_track_center_px", "filename": "best_metric_track_center_px.pt"},
            ]
        )
        if all(spec["metric"] != self.best_metric_name for spec in self.checkpoint_specs):
            safe_metric_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in self.best_metric_name)
            self.checkpoint_specs.append({"metric": self.best_metric_name, "filename": f"best_{safe_metric_name}.pt"})

        self.started_at = time.time()
        self.base_lr = float(self.optimizer_resolved["lr"])
        self.warmup_epochs = (
            0
            if not bool(self.optimizer_meta.get("external_scheduler_allowed", True))
            else int((self.training_cfg.get("scheduler") or {}).get("warmup_epochs", 0))
        )
        mode = str((self.cfg.get("data") or {}).get("mode", "mode1")).strip().lower()
        experiment_name = str(((self.cfg.get("run") or {}).get("materialized_experiment_name")) or ((self.cfg.get("experiment") or {}).get("name") or self.output_dir.parent.name))
        self.console_logger.start_run(
            stage=self.stage,
            mode=mode,
            experiment_name=experiment_name,
            requested_device=str(self.training_cfg.get("device", "cpu")),
            resolved_device=str(self.resolved_device["resolved_device"]),
            resolved_device_ids=list(self.resolved_device["resolved_device_ids"]),
            batch_size=int(self.training_cfg.get("batch_size", 0)),
            num_workers=int(self.training_cfg.get("num_workers", 0)),
            train_samples=len(getattr(self.train_loader, "dataset", [])),
            val_samples=0 if self.val_loader is None else len(getattr(self.val_loader, "dataset", [])),
        )

    def _run_epoch(self, *, epoch: int, training: bool) -> dict[str, float]:
        loader = self.train_loader if training else self.val_loader
        if loader is None:
            return {}
        _set_optimizer_mode(self.optimizer, training=training)
        if training:
            _warmup_lr(self.optimizer, self.base_lr, epoch=epoch, warmup_epochs=self.warmup_epochs)
        grad_accum_steps = int(self.training_cfg.get("grad_accum_steps", 1)) if training else 1
        grad_clip_norm = float(self.training_cfg.get("grad_clip_norm", 0.0)) if training else 0.0
        optimizer = self.optimizer if training else None
        scaler = self.scaler if training else None
        phase = "train" if training else "val"
        if training:
            max_batches = self.training_cfg.get("max_train_batches")
            return _epoch_loop(
                model=self.model,
                loader=loader,
                optimizer=optimizer,
                scaler=scaler,
                device=self.device,
                stage=self.stage,
                loss_cfg=self.loss_cfg,
                distillation_cfg=self.distillation_cfg,
                regularization_ssl_cfg=self.regularization_ssl_cfg,
                pruning_cfg=self.pruning_cfg,
                teacher_model=self.teacher_model,
                amp_enabled=self.amp_enabled,
                grad_accum_steps=grad_accum_steps,
                grad_clip_norm=grad_clip_norm,
                epoch=epoch,
                total_epochs=self.total_epochs,
                phase=phase,
                console_logger=self.console_logger,
                best_metric_name=self.best_metric_name,
                best_metric_value=self.best_control_value,
                active_head=self.active_head,
                max_batches=None if max_batches in (None, "") else int(max_batches),
            )
        with torch.no_grad():
            max_batches = self.training_cfg.get("max_val_batches")
            return _epoch_loop(
                model=self.model,
                loader=loader,
                optimizer=optimizer,
                scaler=scaler,
                device=self.device,
                stage=self.stage,
                loss_cfg=self.loss_cfg,
                distillation_cfg=self.distillation_cfg,
                regularization_ssl_cfg=self.regularization_ssl_cfg,
                pruning_cfg=self.pruning_cfg,
                teacher_model=self.teacher_model,
                amp_enabled=self.amp_enabled,
                grad_accum_steps=grad_accum_steps,
                grad_clip_norm=grad_clip_norm,
                epoch=epoch,
                total_epochs=self.total_epochs,
                phase=phase,
                console_logger=self.console_logger,
                best_metric_name=self.best_metric_name,
                best_metric_value=self.best_control_value,
                active_head=self.active_head,
                max_batches=None if max_batches in (None, "") else int(max_batches),
            )

    def _save_epoch_artifacts(self, *, epoch: int, train_stats: dict[str, float], val_stats: dict[str, float] | None) -> None:
        epoch_row = {
            "epoch": epoch,
            "stage": self.stage,
            "lr": _current_lr(self.optimizer),
            "elapsed_sec": time.time() - self.started_at,
            "train": _serialize_stats(train_stats),
            "val": None if val_stats is None else _serialize_stats(val_stats),
        }
        self.history.append(epoch_row)
        self.history_path.write_text(json.dumps(self.history, indent=2, ensure_ascii=False), encoding="utf-8")
        _append_jsonl(self.history_jsonl_path, epoch_row)
        with self.train_log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(json.dumps(epoch_row, ensure_ascii=False) + "\n")

        _save_checkpoint(
            self.output_dir / "last.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            stage=self.stage,
            cfg=self.cfg,
            history=self.history,
            best_metrics=self.best_metrics,
        )
        if self.teacher_model is not None:
            _save_teacher_checkpoint(
                self.output_dir / "teacher_last.pt",
                teacher_model=self.teacher_model,
                epoch=epoch,
                stage=self.stage,
                cfg=self.cfg,
            )

    def _update_best_checkpoints(self, *, epoch: int, ref_stats: dict[str, float]) -> list[str]:
        saved_checkpoints: list[str] = []
        for spec in self.checkpoint_specs:
            metric_name = spec["metric"]
            if metric_name not in ref_stats:
                continue
            metric_value = float(ref_stats[metric_name])
            prior = self.best_metrics.get(metric_name)
            if _is_better(metric_value, None if prior is None else float(prior["value"]), mode=_metric_mode(metric_name)):
                self.best_metrics[metric_name] = {"value": metric_value, "epoch": epoch, "path": spec["filename"]}
                _save_checkpoint(
                    self.output_dir / spec["filename"],
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    stage=self.stage,
                    cfg=self.cfg,
                    history=self.history,
                    best_metrics=self.best_metrics,
                )
                saved_checkpoints.append(spec["filename"])
                self.console_logger.write(
                    f"[checkpoint] saved={spec['filename']} metric={metric_name} value={metric_value:.4f} epoch={epoch}"
                )
                if metric_name == self.best_metric_name:
                    _save_checkpoint(
                        self.output_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        stage=self.stage,
                        cfg=self.cfg,
                        history=self.history,
                        best_metrics=self.best_metrics,
                    )
                    if self.teacher_model is not None:
                        _save_teacher_checkpoint(
                            self.output_dir / "teacher_best.pt",
                            teacher_model=self.teacher_model,
                            epoch=epoch,
                            stage=self.stage,
                            cfg=self.cfg,
                        )
                    saved_checkpoints.append("best.pt")
        return saved_checkpoints

    def _update_early_stopping(self, *, epoch: int, control_value: float) -> bool:
        if _is_better(control_value, self.best_control_value, mode=self.best_metric_mode, min_delta=self.early_min_delta):
            self.best_control_value = control_value
            self.early_counter = 0
        elif self.early_enabled and epoch >= self.early_start_epoch:
            self.early_counter += 1

        if self.early_enabled and epoch >= self.early_start_epoch and self.early_counter >= self.early_patience:
            self.console_logger.write(
                f"[early-stop] stopping at epoch={epoch} best_{self.best_metric_name}={self.best_control_value:.4f}"
            )
            return True
        return False

    def _run_epochs(self) -> None:
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            train_stats = self._run_epoch(epoch=epoch, training=True)
            val_stats = self._run_epoch(epoch=epoch, training=False) if self.val_loader is not None else None
            if self.val_loader is None:
                _set_optimizer_mode(self.optimizer, training=False)

            ref_stats = val_stats or train_stats
            control_value = float(ref_stats.get(self.best_metric_name, ref_stats.get("loss_total", 0.0)))
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(control_value)
                elif epoch > self.warmup_epochs:
                    self.scheduler.step()

            self._save_epoch_artifacts(epoch=epoch, train_stats=train_stats, val_stats=val_stats)
            saved_checkpoints = self._update_best_checkpoints(epoch=epoch, ref_stats=ref_stats)
            should_stop = self._update_early_stopping(epoch=epoch, control_value=control_value)
            self.console_logger.epoch_summary(
                epoch=epoch,
                total_epochs=self.total_epochs,
                lr=_current_lr(self.optimizer),
                train_stats=train_stats,
                val_stats=val_stats,
                best_metric_name=self.best_metric_name,
                best_metric_value=self.best_control_value,
                saved_checkpoints=saved_checkpoints,
                early_counter=self.early_counter,
                early_patience=self.early_patience if self.early_enabled else 0,
            )
            if should_stop:
                break

    def _maybe_export(self) -> dict[str, Any] | None:
        export_cfg = self.pruning_cfg.get("export") or {}
        if not is_structural_pruning_enabled(self.cfg) or not bool(export_cfg.get("enabled", True)):
            return None
        export_report = export_structural_student(
            model=_unwrap_model(self.model),
            cfg=self.cfg,
            output_dir=self.output_dir,
            stage=self.stage,
        )
        self.console_logger.write(
            "[export] "
            f"checkpoint={export_report.get('checkpoint_path')} "
            f"report={export_report.get('report_path')}"
        )
        return export_report

    def run(self) -> dict[str, Any]:
        self._prepare_runtime()
        self._initialize_model_state()
        self._initialize_epoch_control()
        self._run_epochs()
        export_report = self._maybe_export()
        return {
            "output_dir": str(self.output_dir),
            "history_path": str(self.history_path),
            "best_metrics": self.best_metrics,
            "pretrained_report": self.pretrained_report,
            "export_report": export_report,
        }


def train(
    *,
    cfg: dict,
    train_manifest: str,
    val_manifest: str | None,
    output_dir: str | Path,
    stage1_checkpoint: str | None = None,
    resume_checkpoint: str | None = None,
) -> dict[str, Any]:
    session = TrainingSession(
        cfg=cfg,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        stage1_checkpoint=stage1_checkpoint,
        resume_checkpoint=resume_checkpoint,
    )
    return session.run()
