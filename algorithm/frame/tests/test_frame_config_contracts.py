from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eveye.dataset.dataset_factory import DATASET_CLASSES
from eveye.engine.model_factory import MODEL_CLASSES

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _config_paths() -> list[Path]:
    return sorted(CONFIG_DIR.rglob("*.yaml"))


def _dataset_types(config: dict) -> list[str]:
    found = []
    dataloader = config.get("dataloader") or {}
    for split_cfg in dataloader.values():
        if isinstance(split_cfg, dict):
            dataset_cfg = split_cfg.get("dataset") or {}
            if isinstance(dataset_cfg, dict) and "type" in dataset_cfg:
                found.append(dataset_cfg["type"])
    top_level = config.get("dataset") or {}
    if isinstance(top_level, dict) and "type" in top_level:
        found.append(top_level["type"])
    return found


def test_every_config_is_present_and_parses() -> None:
    paths = _config_paths()
    assert paths, f"no configs discovered under {CONFIG_DIR}"
    for path in paths:
        loaded = yaml.safe_load(path.read_text())
        assert isinstance(loaded, dict), f"{path} is not a mapping"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.name)
def test_model_type_is_registered(path: Path) -> None:
    config = yaml.safe_load(path.read_text())
    model_cfg = config.get("model")
    assert isinstance(model_cfg, dict), f"{path} has no model section"
    assert model_cfg["type"] in MODEL_CLASSES, f"{path} uses unregistered model {model_cfg[type]}"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.name)
def test_dataset_types_are_registered(path: Path) -> None:
    config = yaml.safe_load(path.read_text())
    types = _dataset_types(config)
    assert types, f"{path} declares no dataset type"
    for dataset_type in types:
        assert dataset_type in DATASET_CLASSES, f"{path} uses unregistered dataset {dataset_type}"


def test_frame_owner_declares_no_python_package() -> None:
    """AM-080 forbids creating an empty frame/src package."""
    assert not (Path(__file__).resolve().parents[1] / "src").exists()
