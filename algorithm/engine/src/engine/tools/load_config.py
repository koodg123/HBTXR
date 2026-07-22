from typing import Any
from pathlib import Path
import json
import yaml


def load_config(
    config_file_name: str, config_dir_path: Path = Path("configs")
) -> dict[str, Any]:
    """Load a YAML/JSON config.

    An absolute path, or any path that already resolves from the current working
    directory, is used verbatim. A bare file name keeps the historical behaviour
    of resolving under ``config_dir_path``. This lets launchers accept explicit
    config paths without depending on where they are invoked from.
    """
    candidate = Path(config_file_name)
    if candidate.is_absolute() or candidate.exists():
        config_path = candidate
    else:
        config_path = config_dir_path / config_file_name
    with config_path.open() as config_file:
        if str(config_path).endswith("yaml"):
            config = yaml.safe_load(config_file)
        else:
            config = json.load(config_file)
    assert isinstance(
        config, dict
    ), f"Config file {str(config_path)} is not a dictionary"
    return config
