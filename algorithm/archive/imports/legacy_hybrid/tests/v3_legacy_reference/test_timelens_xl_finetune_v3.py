from __future__ import annotations

import numpy as np
from pathlib import Path

from hbtxr.preprocess.target_fps_build import build_target_fps_dataset
from hbtxr.preprocess.timelens_xl_finetune import build_timelens_xl_run_argv, export_timelens_xl_finetune_dataset

from tests.test_target_fps_build_v3 import _build_target_fps_raw_workspace


def _build_materialized_target_fps_workspace(tmp_path: Path) -> dict[str, Path]:
    workspace = _build_target_fps_raw_workspace(tmp_path)
    build_target_fps_dataset(
        raw_root=workspace["raw_root"],
        target_root=workspace["target_root"],
        target_fps=2000.0,
        include_nonstandard_sessions=True,
        execute=True,
        session_store_format="npz",
        overwrite=True,
    )
    return workspace


def test_export_timelens_xl_finetune_dataset_writes_bsergb_like_surface(tmp_path: Path):
    workspace = _build_materialized_target_fps_workspace(tmp_path)
    output_root = tmp_path / "timelens_xl_finetune_dataset"

    summary = export_timelens_xl_finetune_dataset(
        target_root=workspace["target_root"],
        target_fps=2000.0,
        output_root=output_root,
        overwrite=True,
        val_every_nth_sequence=0,
    )

    sequence_root = output_root / "train" / "user01_left_session_101"
    images_dir = sequence_root / "images"
    events_dir = sequence_root / "events"
    first_packet = np.load(events_dir / "000000.npz")

    assert summary["n_sequences_exported"] == 1
    assert summary["n_sequences_train"] == 1
    assert summary["n_sequences_val"] == 0
    assert len(list(images_dir.glob("*.png"))) == 81
    assert len(list(events_dir.glob("*.npz"))) == 80
    assert first_packet["data"].shape == (240, 346)
    assert float(np.abs(first_packet["data"]).sum()) > 0.0
    assert (sequence_root / "sequence_meta.json").exists()
    assert (output_root / "export_summary.json").exists()


def test_build_timelens_xl_run_argv_matches_native_entry_contract():
    argv = build_timelens_xl_run_argv(
        param_name="hbtxr_timelens_xl_tuning",
        model_name="TimeLens",
        model_pretrained="/tmp/checkpoint.pth",
        init_step=12,
        skip_training=True,
        clear_previous=True,
        extension="_debug",
    )

    assert argv[:5] == ["run_network.py", "--param_name", "hbtxr_timelens_xl_tuning", "--model_name", "TimeLens"]
    assert "--model_pretrained" in argv
    assert "--skip_training" in argv
    assert "--clear_previous" in argv
    assert "--extension" in argv
