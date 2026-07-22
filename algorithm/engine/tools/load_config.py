"""Config loader (migrated from apps): YAML/JSON -> dict, with ``include`` merge.

A config may carry a top-level ``include:`` (a path or list of paths). Each is loaded
(recursively, so includes may nest) and deep-merged left-to-right; the current
document is merged on top last, so local keys win. This lets experiment configs
compose the modality model fragment (configs/modality/*.yaml) with shared base
fragments (configs/base/*.yaml) without duplication, while the loader still returns
a single flat dict for the entrypoints (no include key survives).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _load_raw(path: Path) -> Any:
    with path.open(encoding="utf-8") as config_file:
        if str(path).endswith((".yaml", ".yml")):
            return yaml.safe_load(config_file)
        return json.load(config_file)


def _resolve_path(name: str, config_dir_path: Path) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return config_dir_path / name


def _resolve_include(include: str, *, base_dir: Path, config_dir_path: Path) -> Path:
    candidate = Path(include)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    sibling = base_dir / include
    if sibling.exists():
        return sibling
    return config_dir_path / include


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_file_name: str, config_dir_path: Path = Path("configs")) -> dict[str, Any]:
    """Load a YAML/JSON config into a dict, resolving any ``include`` merges.

    An absolute path, or any path that already resolves from the current working
    directory, is used verbatim; a bare file name resolves under ``config_dir_path``.
    """
    path = _resolve_path(str(config_file_name), config_dir_path)
    raw = _load_raw(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} is not a dictionary")

    includes = raw.pop("include", None) or []
    if isinstance(includes, str):
        includes = [includes]

    merged: dict[str, Any] = {}
    for include in includes:
        include_path = _resolve_include(str(include), base_dir=path.parent, config_dir_path=config_dir_path)
        merged = _deep_merge(merged, load_config(str(include_path), config_dir_path))
    return _deep_merge(merged, raw)


__all__ = ["load_config"]
