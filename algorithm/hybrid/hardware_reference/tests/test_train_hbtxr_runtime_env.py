from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = SOFTWARE_ROOT / "scripts" / "v3" / "train_hbtxr.py"
EVAL_SCRIPT = SOFTWARE_ROOT / "scripts" / "v3" / "eval_hbtxr.py"


def _load_script_module(path: Path, name: str):
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disable_cudnn_env_flag_toggles_torch_backend():
    import torch

    module = _load_script_module(TRAIN_SCRIPT, "train_hbtxr")
    original_enabled = torch.backends.cudnn.enabled
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        module._apply_torch_runtime_env({"HBTXR_DISABLE_CUDNN": "1"})

        assert torch.backends.cudnn.enabled is False
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.enabled = original_enabled
        torch.backends.cudnn.benchmark = original_benchmark


def test_eval_disable_cudnn_env_flag_toggles_torch_backend():
    import torch

    module = _load_script_module(EVAL_SCRIPT, "eval_hbtxr")
    original_enabled = torch.backends.cudnn.enabled
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        module._apply_torch_runtime_env({"HBTXR_DISABLE_CUDNN": "1"})

        assert torch.backends.cudnn.enabled is False
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.enabled = original_enabled
        torch.backends.cudnn.benchmark = original_benchmark
