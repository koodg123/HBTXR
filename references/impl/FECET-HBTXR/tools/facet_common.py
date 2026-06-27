from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import PROJECT_ROOT


DEFAULT_FACET_REPO_ROOT = (PROJECT_ROOT.parent.parent / "references" / "FACET-main" / "FACET-main").resolve()


def resolve_facet_repo_root(path: str | Path | None = None) -> Path:
    root = DEFAULT_FACET_REPO_ROOT if path is None else Path(path)
    return root.resolve()


def ensure_facet_repo_importable(facet_repo_root: str | Path) -> Path:
    root = resolve_facet_repo_root(facet_repo_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def import_facet_module(facet_repo_root: str | Path, module_name: str):
    ensure_facet_repo_importable(facet_repo_root)
    return importlib.import_module(module_name)


def resolve_facet_config_path(config: str | Path, *, facet_repo_root: str | Path) -> Path:
    candidate = Path(config)
    if candidate.is_absolute():
        return candidate.resolve()
    facet_repo_root = resolve_facet_repo_root(facet_repo_root)
    direct = (facet_repo_root / candidate).resolve()
    if direct.exists():
        return direct
    return (facet_repo_root / "configs" / candidate).resolve()


def load_facet_config(config: str | Path, *, facet_repo_root: str | Path) -> dict[str, Any]:
    config_path = resolve_facet_config_path(config, facet_repo_root=facet_repo_root)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"FACET config must be a mapping: {config_path}")
    return loaded


def patch_facet_config(
    cfg: dict[str, Any],
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    experiment_name: str,
    sensor_size_whc: tuple[int, int, int],
    default_resolution: tuple[int, int],
    batch_size: int | None,
    num_workers: int | None,
    max_epochs: int | None,
    checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    patched = copy.deepcopy(cfg)
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()

    for split in ("train", "val"):
        dataloader_cfg = ((patched.get("dataloader") or {}).get(split) or {})
        dataset_cfg = dataloader_cfg.get("dataset") or {}
        if dataset_cfg:
            dataset_cfg["root_path"] = str(dataset_root)
            dataset_cfg["sensor_size"] = [int(v) for v in sensor_size_whc]
            dataset_cfg["default_resolution"] = [int(v) for v in default_resolution]
        if batch_size is not None:
            dataloader_cfg["batch_size"] = int(batch_size)
        if num_workers is not None:
            dataloader_cfg["num_workers"] = int(num_workers)

    logger_cfg = patched.setdefault("logger", {})
    logger_cfg["save_dir"] = str((output_root / "logs").resolve())
    logger_cfg["name"] = str(experiment_name)

    checkpoint_dir = (output_root / experiment_name / "checkpoints").resolve()
    for callback_cfg in patched.get("callback") or []:
        if isinstance(callback_cfg, dict) and callback_cfg.get("type") == "ModelCheckpoint":
            callback_cfg["dirpath"] = str(checkpoint_dir)

    train_cfg = patched.setdefault("train", {})
    if max_epochs is not None:
        train_cfg["max_epochs"] = int(max_epochs)
    if checkpoint_path is not None:
        checkpoint_str = str(Path(checkpoint_path).resolve())
        train_cfg["ckpt_path"] = checkpoint_str
        patched.setdefault("val", {})["ckpt_path"] = checkpoint_str

    return patched


def trainer_device_kwargs(device: str | None) -> dict[str, Any]:
    import torch

    raw = str(device or "cpu").strip().lower()
    if raw in {"cpu", "auto-cpu"}:
        return {"accelerator": "cpu", "devices": 1}
    if raw in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        return {"accelerator": "gpu", "devices": 1}
    if raw.startswith("cuda:") or raw.startswith("gpu:"):
        _, _, suffix = raw.partition(":")
        device_ids = [int(item.strip()) for item in suffix.split(",") if item.strip()]
        if not device_ids:
            raise ValueError(f"Invalid FACET device specification: {device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        return {"accelerator": "gpu", "devices": device_ids}
    if raw == "auto":
        return {"accelerator": "auto", "devices": 1}
    raise ValueError(f"Unsupported FACET device string: {device}")


def torch_device_from_arg(device: str | None):
    import torch

    raw = str(device or "cpu").strip().lower()
    if raw in {"cuda", "gpu"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def collect_model_checkpoint_paths(callbacks: list[Any]) -> list[str]:
    paths: list[str] = []
    for callback in callbacks:
        for value in (getattr(callback, "best_model_path", ""), getattr(callback, "last_model_path", "")):
            text = str(value).strip()
            if text and text not in paths:
                paths.append(text)
    return paths
