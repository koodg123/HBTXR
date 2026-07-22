"""Config loader (migrated from apps: YAML/JSON -> dict)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_file_name: str, config_dir_path: Path = Path("configs")) -> dict[str, Any]:
    """Load a YAML/JSON config into a dict.

    An absolute path, or any path that already resolves from the current working
    directory, is used verbatim. A bare file name keeps the historical behaviour of
    resolving under ``config_dir_path`` so launchers can accept explicit config
    paths without depending on where they are invoked from.
    """
    candidate = Path(config_file_name)
    if candidate.is_absolute() or candidate.exists():
        config_path = candidate
    else:
        config_path = config_dir_path / config_file_name
    with config_path.open(encoding="utf-8") as config_file:
        if str(config_path).endswith(("yaml", "yml")):
            config = yaml.safe_load(config_file)
        else:
            config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} is not a dictionary")
    return config


__all__ = ["load_config"]
