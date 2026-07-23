from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from prepare_dataset import build_session_index
from swift_hbtxr.antiblink import AntiBlinkDetector
from swift_hbtxr.compat import import_swift_eye_antiblink_weights
from swift_hbtxr.geometry import compute_open_extent_from_binary_mask
from swift_hbtxr.interpolation import TimeLensConfig, TimeLensPrepConfig, TimeLensRunner, prepare_timelens_inputs


def test_compute_open_extent_from_binary_mask() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    extent = compute_open_extent_from_binary_mask(mask, [16.0, 16.0, 12.0, 12.0, 0.0])
    assert 0.5 < extent <= 1.0


def test_timelens_runner_build_command(tmp_path: Path) -> None:
    timelens_root = tmp_path / "timelens"
    runner_path = timelens_root / "tests"
    runner_path.mkdir(parents=True, exist_ok=True)
    (runner_path / "run_attention.py").write_text("print('ok')\n", encoding="utf-8")
    checkpoint = tmp_path / "attention_average_network.pt"
    checkpoint.write_bytes(b"checkpoint")
    runner = TimeLensRunner(
        TimeLensConfig(
            timelens_root=timelens_root,
            checkpoint_file=checkpoint,
            python_bin="python",
            frames_to_insert=3,
            frames_to_skip=1,
        )
    )
    command = runner.build_command(
        image_root=tmp_path / "images",
        event_root=tmp_path / "events",
        output_root=tmp_path / "output",
    )
    assert command[0] == "python"
    assert "--checkpoint-file" in command
    assert "--number-of-frames-to-insert" in command
    assert "3" in command


def test_import_swift_eye_antiblink_weights_reports_skips(tmp_path: Path) -> None:
    detector = AntiBlinkDetector()
    state = detector.model.state_dict()
    first_key = next(iter(state))
    checkpoint = tmp_path / "swift_eye_antiblink.pt"
    torch.save(
        {
            "state_dict": {
                f"unet.{first_key}": torch.ones_like(state[first_key]),
                "backbone.block.weight": torch.randn(2, 2),
            }
        },
        checkpoint,
    )
    report_path = tmp_path / "report.json"
    report = import_swift_eye_antiblink_weights(
        checkpoint_path=checkpoint,
        detector=detector,
        report_path=report_path,
    )
    assert report.imported_keys
    assert "backbone.block.weight" in report.skipped_keys
    assert report_path.exists()


def test_prepare_timelens_inputs_from_session_index(synthetic_workspace) -> None:
    build_session_index(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
    )
    prepared_root = synthetic_workspace["project_root"] / "timelens_ready"
    summary = prepare_timelens_inputs(
        session_keys=["user01/left/session_101"],
        config=TimeLensPrepConfig(
            canonical_root=synthetic_workspace["canonical_root"],
            indexes_root=synthetic_workspace["indexes_root"],
            prepared_root=prepared_root,
            overwrite=True,
        ),
    )
    assert summary["session_count"] == 1
    session = summary["sessions"][0]
    image_dir = Path(str(session["image_dir"]))
    event_dir = Path(str(session["event_dir"]))
    assert image_dir.exists()
    assert event_dir.exists()
    assert len(list(image_dir.glob("*.png"))) == 2
    assert (event_dir / "0000001.npz").exists()
    timestamps = (image_dir / "timestamp.txt").read_text(encoding="utf-8").strip().splitlines()
    assert timestamps == ["1000", "2000"]
