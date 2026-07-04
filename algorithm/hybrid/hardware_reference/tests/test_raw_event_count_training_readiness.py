from __future__ import annotations

import importlib.util
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
READINESS = SOFTWARE_ROOT / "scripts" / "v3" / "check_raw_event_count_training_readiness.py"


def _load_readiness_module():
    spec = importlib.util.spec_from_file_location("check_raw_event_count_training_readiness", READINESS)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_ready_requires_torch_cuda_and_nvidia_smi_success():
    module = _load_readiness_module()

    assert module._cuda_ready(
        {
            "torch_cuda_available": True,
            "torch_cuda_device_count": 1,
            "nvidia_smi_returncode": 0,
        }
    )
    assert not module._cuda_ready(
        {
            "torch_cuda_available": True,
            "torch_cuda_device_count": 1,
            "nvidia_smi_returncode": 9,
        }
    )
    assert not module._cuda_ready(
        {
            "torch_cuda_available": False,
            "torch_cuda_device_count": 0,
            "nvidia_smi_returncode": 0,
        }
    )


def test_latest_run_root_ignores_prefix_smoke_names(tmp_path: Path):
    module = _load_readiness_module()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    experiment_name = "raw_mode1_stage2_event_count"
    full_run = runs_root / f"{experiment_name}_20260610_193719"
    smoke_run = runs_root / f"{experiment_name}_config_smoke_20260610_092747"
    full_run.mkdir()
    smoke_run.mkdir()

    assert module._latest_run_root(runs_root, experiment_name) == full_run


def test_latest_run_root_finds_organized_non_xr_runs(tmp_path: Path):
    module = _load_readiness_module()
    runs_root = tmp_path / "runs"
    experiment_name = "raw_mode1_stage2_event_count"
    organized = runs_root / "NON_XR" / "raw" / f"{experiment_name}_20260610_193719"
    smoke_run = runs_root / "NON_XR" / "raw" / f"{experiment_name}_config_smoke_20260610_092747"
    organized.mkdir(parents=True)
    smoke_run.mkdir()

    assert module._latest_run_root(runs_root, experiment_name) == organized
