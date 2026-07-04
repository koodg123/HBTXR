from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
CHECKER = SOFTWARE_ROOT / "scripts" / "v3" / "check_raw_event_count_training_result.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location("check_raw_event_count_training_result", CHECKER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(root: Path, *, stage: str, init_checkpoint: str | None = None, limited: bool = False) -> Path:
    run_root = root / f"raw_{stage}_20260610_000000"
    train_dir = run_root / "train"
    hypers_dir = run_root / "hypers"
    train_dir.mkdir(parents=True)
    hypers_dir.mkdir(parents=True)
    ckpt_name = "best_search_p10.pt" if stage == "stage1" else "best_track_p10.pt"
    (train_dir / ckpt_name).write_bytes(b"checkpoint")
    _write_json(
        train_dir / "history.json",
        [
            {
                "epoch": 1,
                "stage": stage,
                "train": {"loss_total": 1.0},
                "val": {"loss_total": 1.5},
            }
        ],
    )
    cfg = {
        "model": {
            "heads": {
                "event": stage == "stage2",
                "track": stage == "stage2",
                "mask": stage == "stage1",
            }
        },
        "data": {
            "mode": "mode1",
            "mode1": {
                "event_builder": {
                    "policy": "fixed_count",
                    "event_count_target": 5000,
                    "accumulation": "fast_causal_linear",
                }
            },
        },
        "training": {"stage": stage},
        "experiment": {"init_checkpoint": init_checkpoint},
    }
    if limited:
        cfg["training"]["max_train_batches"] = 1
        cfg["training"]["max_val_batches"] = 1
    _write_json(hypers_dir / "resolved_config.json", cfg)
    (hypers_dir / "cli_args.txt").write_text(str(init_checkpoint or ""), encoding="utf-8")
    return run_root


def test_training_result_checker_accepts_limited_smoke_with_flag(tmp_path: Path):
    stage1 = _write_run(tmp_path, stage="stage1", limited=True)
    stage1_ckpt = stage1 / "train" / "best_search_p10.pt"
    stage2 = _write_run(tmp_path, stage="stage2", init_checkpoint=str(stage1_ckpt), limited=True)

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--project-root",
            str(tmp_path),
            "--stage1-run",
            str(stage1),
            "--stage2-run",
            str(stage2),
            "--allow-limited",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert '"ok": true' in result.stdout


def test_training_result_checker_rejects_limited_without_flag(tmp_path: Path):
    stage1 = _write_run(tmp_path, stage="stage1", limited=True)
    stage1_ckpt = stage1 / "train" / "best_search_p10.pt"
    stage2 = _write_run(tmp_path, stage="stage2", init_checkpoint=str(stage1_ckpt), limited=True)

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--project-root",
            str(tmp_path),
            "--stage1-run",
            str(stage1),
            "--stage2-run",
            str(stage2),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert "limited training keys present" in result.stdout


def test_training_result_checker_derives_stage1_run_from_checkpoint(tmp_path: Path):
    stage1 = _write_run(tmp_path, stage="stage1", limited=True)
    stage1_ckpt = stage1 / "train" / "best_search_p10.pt"
    stage2 = _write_run(tmp_path, stage="stage2", init_checkpoint=str(stage1_ckpt), limited=True)

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--project-root",
            str(tmp_path),
            "--stage1-checkpoint",
            str(stage1_ckpt),
            "--stage2-run",
            str(stage2),
            "--allow-limited",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert '"ok": true' in result.stdout
    assert str(stage1_ckpt) in result.stdout


def test_latest_run_root_ignores_prefix_smoke_names(tmp_path: Path):
    module = _load_checker_module()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    experiment_name = "raw_mode1_stage2_event_count"
    full_run = runs_root / f"{experiment_name}_20260610_193719"
    smoke_run = runs_root / f"{experiment_name}_config_smoke_20260610_092747"
    full_run.mkdir()
    smoke_run.mkdir()

    assert module._latest_run_root(runs_root, experiment_name) == full_run


def test_latest_run_root_finds_organized_non_xr_runs(tmp_path: Path):
    module = _load_checker_module()
    runs_root = tmp_path / "runs"
    experiment_name = "raw_mode1_stage2_event_count"
    organized = runs_root / "NON_XR" / "raw" / f"{experiment_name}_20260610_193719"
    smoke_run = runs_root / "NON_XR" / "raw" / f"{experiment_name}_config_smoke_20260610_092747"
    organized.mkdir(parents=True)
    smoke_run.mkdir()

    assert module._latest_run_root(runs_root, experiment_name) == organized
