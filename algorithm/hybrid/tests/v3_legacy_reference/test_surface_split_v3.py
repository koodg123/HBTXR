from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_help(*argv: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{pythonpath}" if pythonpath else str(PROJECT_ROOT / "src")
    return subprocess.run(
        list(argv),
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_main_surface_help_smoke():
    cases = [
        ("sh", "scripts/run_train.sh", "--help"),
        ("sh", "scripts/run_prepare_and_train.sh", "--help"),
        ("sh", "scripts/run_mode2_end_to_end.sh", "--help"),
        ("sh", "scripts/hbtxr_mode_pipeline.sh", "--help"),
    ]
    for case in cases:
        result = _run_help(*case)
        assert result.returncode == 0, result.stderr or result.stdout


def test_experimental_surface_help_smoke():
    py_result = _run_help(sys.executable, "exps/scripts/run_roi_crop_prompted_preview_batch.py", "--help")
    assert py_result.returncode == 0, py_result.stderr or py_result.stdout

    sh_result = _run_help("sh", "exps/scripts/run_mode0_train_stage1.sh", "--help")
    assert sh_result.returncode == 0, sh_result.stderr or sh_result.stdout

    export_result = _run_help(sys.executable, "exps/scripts/export_timelens_xl_finetune_dataset.py", "--help")
    assert export_result.returncode == 0, export_result.stderr or export_result.stdout

    finetune_result = _run_help(sys.executable, "exps/scripts/finetune_timelens_xl.py", "--help")
    assert finetune_result.returncode == 0, finetune_result.stderr or finetune_result.stdout

    v2e_compare_result = _run_help(sys.executable, "exps/scripts/compare_v2e_event_generation.py", "--help")
    assert v2e_compare_result.returncode == 0, v2e_compare_result.stderr or v2e_compare_result.stdout

    v2e_experiment_result = _run_help(sys.executable, "exps/scripts/run_v2e_event_experiment.py", "--help")
    assert v2e_experiment_result.returncode == 0, v2e_experiment_result.stderr or v2e_experiment_result.stdout

    v2e_sweep_result = _run_help(sys.executable, "exps/scripts/run_v2e_event_sweep.py", "--help")
    assert v2e_sweep_result.returncode == 0, v2e_sweep_result.stderr or v2e_sweep_result.stdout


def test_external_backend_selector_help_smoke():
    annotate = _run_help(sys.executable, "scripts/annotate_groundedsam_ev_eye.py", "--help")
    assert annotate.returncode == 0, annotate.stderr or annotate.stdout
    assert "--annotation-backend" in annotate.stdout

    build_dataset = _run_help(sys.executable, "scripts/build_groundedsam_dataset.py", "--help")
    assert build_dataset.returncode == 0, build_dataset.stderr or build_dataset.stdout
    assert "--annotation-backend" in build_dataset.stdout

    prepare = _run_help(sys.executable, "scripts/prepare_ev_eye.py", "--help")
    assert prepare.returncode == 0, prepare.stderr or prepare.stdout
    assert "--interp-backend" in prepare.stdout
    assert "--event-generation-backend" in prepare.stdout

    canonicalize = _run_help(sys.executable, "scripts/canonicalize_hbtxr.py", "--help")
    assert canonicalize.returncode == 0, canonicalize.stderr or canonicalize.stdout
    assert "--event-generation-backend" in canonicalize.stdout
