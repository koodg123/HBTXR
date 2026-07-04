from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.config.run_contract import (
    collect_reference_only_reports as _shared_collect_reference_only_reports,
    default_checkpoint_for_stage as _shared_default_checkpoint_for_stage,
    latest_run_root as _shared_latest_run_root,
    materialize_experiment_name as _shared_materialize_experiment_name,
    resolve_experiment_name as _shared_resolve_experiment_name,
    resolve_manifest_path as _shared_resolve_manifest_path,
    resolve_project_path as _shared_resolve_project_path,
    resolve_resume_root as _shared_resolve_resume_root,
    resolve_run_contract as _shared_resolve_run_contract,
    resolve_training_entry as _shared_resolve_training_entry,
    write_run_artifacts as _shared_write_run_artifacts,
)
from src.config.runtime_config import (
    build_dataset_kwargs as _shared_build_dataset_kwargs,
    mode_defaults as _shared_mode_defaults,
    resolve_data_mode as _shared_resolve_data_mode,
    resolve_mode_block as _shared_resolve_mode_block,
    resolve_mode_contract as _shared_resolve_mode_contract,
)


_WARNED_EXPERIMENTAL_ALIASES: set[str] = set()


def _deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_experimental_config_alias(config_path: Path) -> Path:
    if config_path.exists():
        return config_path
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        return config_path

    project_root = _project_root()
    relative: Path | None = None
    if config_path.is_absolute():
        try:
            relative = config_path.relative_to(project_root / "configs")
        except ValueError:
            relative = None
    else:
        parts = config_path.parts
        if parts and parts[0] == "configs":
            relative = Path(*parts[1:])

    if relative is None:
        return config_path

    alias_path = project_root / "exps" / "configs" / relative
    if alias_path.exists():
        alias_key = str(alias_path)
        if alias_key not in _WARNED_EXPERIMENTAL_ALIASES:
            _WARNED_EXPERIMENTAL_ALIASES.add(alias_key)
            print(
                f"[DEPRECATED] Config path '{config_path}' moved to '{alias_path.relative_to(project_root)}'. "
                "Please update your command to use the exps path.",
            )
        return alias_path
    return config_path


def load_config(path: str | Path) -> dict:
    config_path = _resolve_experimental_config_alias(Path(path))
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(config_path.parent / str(parent))
        cfg = _deep_update(base, cfg)
    return cfg


def resolve_project_path(path_value: str | Path | None, *, project_root: str | Path) -> Path | None:
    return _shared_resolve_project_path(path_value, project_root=project_root)


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
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
        _set_dotted(resolved, "training.device", str(device_override))
    if experiment_name_override:
        _set_dotted(resolved, "experiment.name", str(experiment_name_override))
    if output_dir_override is not None:
        _set_dotted(resolved, "experiment.output_dir", str(output_dir_override))
    if train_manifest_override is not None:
        _set_dotted(resolved, "experiment.train_manifest", str(train_manifest_override))
    if val_manifest_override is not None:
        _set_dotted(resolved, "experiment.val_manifest", str(val_manifest_override))
    if checkpoint_override is not None:
        _set_dotted(resolved, "experiment.init_checkpoint", str(checkpoint_override))

    for item in overrides or ():
        text = str(item).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Override must be KEY=VALUE, got: {text}")
        key, raw_value = text.split("=", 1)
        _set_dotted(resolved, key.strip(), _parse_override_value(raw_value))
    return resolved


def resolve_experiment_name(cfg: dict, *, config_path: str | Path) -> str:
    return _shared_resolve_experiment_name(cfg, config_path=config_path)


def _mode_defaults(mode: str) -> dict[str, str]:
    return _shared_mode_defaults(mode)


def resolve_data_mode(cfg: dict) -> str:
    return _shared_resolve_data_mode(cfg.get("data") or {})


def resolve_mode_block(cfg: dict) -> dict[str, Any]:
    return _shared_resolve_mode_block(cfg.get("data") or {})


def resolve_mode_contract(cfg: dict) -> dict[str, str]:
    return _shared_resolve_mode_contract(cfg.get("data") or {})


def materialize_experiment_name(experiment_name: str, *, timestamp: str | None = None, resume: bool = False) -> str:
    return _shared_materialize_experiment_name(experiment_name, timestamp=timestamp, resume=resume)


def latest_run_root(project_root: str | Path, experiment_name: str) -> Path | None:
    return _shared_latest_run_root(project_root, experiment_name)


def resolve_resume_root(project_root: str | Path, experiment_name: str, resume_arg: str | None) -> Path | None:
    return _shared_resolve_resume_root(project_root, experiment_name, resume_arg)


def default_checkpoint_for_stage(run_root: str | Path, stage: str) -> Path:
    return _shared_default_checkpoint_for_stage(run_root, stage)


def resolve_manifest_path(
    cfg: dict,
    *,
    project_root: str | Path,
    split: str,
    manifest_override: str | Path | None = None,
) -> Path:
    return _shared_resolve_manifest_path(
        cfg,
        project_root=project_root,
        split=split,
        manifest_override=manifest_override,
    )


def collect_reference_only_reports(cfg: dict) -> dict[str, dict[str, Any]]:
    return _shared_collect_reference_only_reports(cfg)


def resolve_run_contract(
    cfg: dict,
    *,
    config_path: str | Path,
    project_root: str | Path,
    action: str,
    resume_root: str | Path | None = None,
) -> dict[str, str]:
    return _shared_resolve_run_contract(
        cfg,
        config_path=config_path,
        project_root=project_root,
        action=action,
        resume_root=resume_root,
    )


def write_run_artifacts(
    *,
    run_contract: dict[str, str],
    cfg: dict,
    config_path: str | Path,
    cli_args: list[str],
    overrides: list[str],
    device: str | None,
    pretrained_report_name: str | None = None,
) -> None:
    _shared_write_run_artifacts(
        run_contract=run_contract,
        cfg=cfg,
        config_path=config_path,
        cli_args=cli_args,
        overrides=overrides,
        device=device,
        pretrained_report_name=pretrained_report_name,
    )


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
    return _shared_resolve_training_entry(
        cfg,
        config_path=config_path,
        project_root=project_root,
        train_manifest_override=train_manifest_override,
        val_manifest_override=val_manifest_override,
        output_override=output_override,
        init_checkpoint_override=init_checkpoint_override,
    )


def build_dataset_kwargs(data_cfg: dict) -> dict:
    return _shared_build_dataset_kwargs(data_cfg)
