from __future__ import annotations

import re
from pathlib import Path


RUNNER = Path("scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh")
EXPECTED_CHECKPOINT_KINDS = {"best_track_p10", "best_track_p5", "best_metric_track_center_px"}


def read_runner() -> str:
    return RUNNER.read_text(encoding="utf-8")


def parse_runner_checkpoint_kinds(text: str) -> set[str]:
    match = re.search(r"REQUIRED_CHECKPOINT_KINDS=\(([^)]*)\)", text)
    assert match is not None
    return {part.strip() for part in match.group(1).split() if part.strip()}


def test_xr65_runner_requires_all_checkpoint_kinds_by_default() -> None:
    text = read_runner()

    assert 'XR65_REQUIRE_ALL_CHECKPOINTS="${XR65_REQUIRE_ALL_CHECKPOINTS:-1}"' in text
    assert parse_runner_checkpoint_kinds(text) == EXPECTED_CHECKPOINT_KINDS
    assert 'for CKPT_KIND in "${REQUIRED_CHECKPOINT_KINDS[@]}"; do' in text
    assert "XR65_REQUIRED_CHECKPOINT_MISSING" in text
    assert "XR65_REQUIRED_EVAL_COUNT_MISMATCH" in text


def test_xr65_runner_uses_train_only_override_and_clears_test_override() -> None:
    text = read_runner()

    train_override = '--stage2-arg --override=data.track_target_override_path="$TARGET_OVERRIDE_PATH"'
    eval_override = "--override data.track_target_override_path=null"
    eval_override_loss = "--override loss.track_target_override_center_l2_weight=0.0"
    train_call = "sh scripts/external/run_prepare_and_train.sh"
    eval_call = ".venv/bin/python scripts/external/eval_hbtxr.py"

    assert train_override in text
    assert "--stage2-arg --override=data.allow_test_target_override=false" in text
    assert eval_override in text
    assert "--override data.allow_test_target_override=false" in text
    assert eval_override_loss in text
    assert "--override loss.track_state_aux_target_override_center_l2_weight=0.0" in text
    assert text.index(train_call) < text.index(eval_call)
    assert text.index(train_override) < text.index(train_call)
    assert text.index(eval_override) < text.index(eval_call)


def test_xr65_runner_defaults_from_xr64c_center_improved_checkpoint() -> None:
    text = read_runner()

    assert "XR64C_BEST_P10" in text
    assert "xr64_xr62a_minerror_teacher_target_c255000" in text
    assert "train/best_track_p10.pt" in text
    assert "center<16.464661524977004" in text
    assert "p10>35.02295998845781" in text
    assert "p5>12.133503770828247" in text
