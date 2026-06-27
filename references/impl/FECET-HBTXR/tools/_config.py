from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(config_path.parent / str(parent))
        cfg = deep_update(base, cfg)
    return cfg


def resolve_project_path(path_value: str | Path | None, *, project_root: str | Path) -> Path | None:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (Path(project_root) / path).resolve()


def parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = cfg
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def apply_config_overrides(
    cfg: dict,
    *,
    overrides: list[str] | tuple[str, ...] | None = None,
    device_override: str | None = None,
    experiment_name_override: str | None = None,
    output_dir_override: str | Path | None = None,
    train_manifest_override: str | Path | None = None,
    val_manifest_override: str | Path | None = None,
    checkpoint_override: str | Path | None = None,
) -> dict:
    resolved = deepcopy(cfg)
    if device_override:
        set_dotted(resolved, "training.device", str(device_override))
    if experiment_name_override:
        set_dotted(resolved, "experiment.name", str(experiment_name_override))
    if output_dir_override is not None:
        set_dotted(resolved, "experiment.output_dir", str(output_dir_override))
    if train_manifest_override is not None:
        set_dotted(resolved, "experiment.train_manifest", str(train_manifest_override))
    if val_manifest_override is not None:
        set_dotted(resolved, "experiment.val_manifest", str(val_manifest_override))
    if checkpoint_override is not None:
        set_dotted(resolved, "experiment.init_checkpoint", str(checkpoint_override))

    for item in overrides or ():
        text = str(item).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Override must be KEY=VALUE, got: {text}")
        key, raw_value = text.split("=", 1)
        set_dotted(resolved, key.strip(), parse_override_value(raw_value))
    return resolved


def resolve_experiment_name(cfg: dict, *, config_path: str | Path) -> str:
    experiment_cfg = cfg.get("experiment") or {}
    raw = str(experiment_cfg.get("name", "")).strip()
    if raw:
        return raw
    return Path(config_path).stem


def default_manifest_path(*, project_root: str | Path, split: str) -> Path:
    return Path(project_root) / "data" / "_internal" / "manifests" / f"{split}_manifest.jsonl"


def default_output_dir(*, project_root: str | Path, stage: str, experiment_name: str) -> Path:
    run_root = Path(project_root) / "runs" / experiment_name
    stage_name = "stage1" if stage == "stage1" else "stage2"
    return run_root / stage_name


def resolve_training_entry(
    cfg: dict,
    *,
    config_path: str | Path,
    project_root: str | Path,
    train_manifest_override: str | Path | None = None,
    val_manifest_override: str | Path | None = None,
    output_override: str | Path | None = None,
    init_checkpoint_override: str | Path | None = None,
) -> dict:
    training_cfg = cfg.get("training") or {}
    experiment_cfg = cfg.get("experiment") or {}
    stage = str(training_cfg.get("stage", "stage1")).strip().lower()
    experiment_name = resolve_experiment_name(cfg, config_path=config_path)

    train_manifest = resolve_project_path(
        train_manifest_override if train_manifest_override is not None else experiment_cfg.get("train_manifest"),
        project_root=project_root,
    )
    if train_manifest is None:
        train_manifest = default_manifest_path(project_root=project_root, split="train")

    val_manifest = resolve_project_path(
        val_manifest_override if val_manifest_override is not None else experiment_cfg.get("val_manifest"),
        project_root=project_root,
    )
    if val_manifest is None:
        fallback = default_manifest_path(project_root=project_root, split="val")
        val_manifest = fallback if fallback.exists() else None

    output_dir = resolve_project_path(
        output_override if output_override is not None else experiment_cfg.get("output_dir"),
        project_root=project_root,
    )
    if output_dir is None:
        output_dir = default_output_dir(project_root=project_root, stage=stage, experiment_name=experiment_name)

    init_checkpoint = resolve_project_path(
        init_checkpoint_override if init_checkpoint_override is not None else experiment_cfg.get("init_checkpoint"),
        project_root=project_root,
    )

    return {
        "stage": stage,
        "experiment_name": experiment_name,
        "train_manifest": str(train_manifest),
        "val_manifest": None if val_manifest is None else str(val_manifest),
        "output_dir": str(output_dir),
        "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
    }


def build_dataset_kwargs(data_cfg: dict) -> dict:
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    event_builder = data_cfg.get("event_builder") or {}
    return {
        "input_size": input_size,
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
