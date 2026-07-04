from __future__ import annotations

import json
import re
from pathlib import Path


RUNNER = Path("scripts/external/run_xr64_teacher_target_construction.sh")
CONTRACT = Path("docs/resources/xr64_postrun_evidence_contract_2026_06_21.json")
WRITER = Path("scripts/external/write_xr64_ablation_provenance.py")
EXPECTED_CHECKPOINT_KINDS = {"best_track_p10", "best_track_p5", "best_metric_track_center_px"}


def read_runner() -> str:
    return RUNNER.read_text(encoding="utf-8")


def read_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def parse_runner_checkpoint_kinds(text: str) -> set[str]:
    match = re.search(r"REQUIRED_CHECKPOINT_KINDS=\(([^)]*)\)", text)
    assert match is not None
    return {part.strip() for part in match.group(1).split() if part.strip()}


def parse_writer_expected_checkpoint_kinds() -> set[str]:
    text = WRITER.read_text(encoding="utf-8")
    match = re.search(r'"checkpoint_kinds": \[(.*?)\]', text, flags=re.S)
    assert match is not None
    return set(re.findall(r'"(best_[^"]+)"', match.group(1)))


def test_runner_requires_all_postrun_checkpoint_kinds_by_default() -> None:
    text = read_runner()
    contract = read_contract()

    assert 'XR64_REQUIRE_ALL_CHECKPOINTS="${XR64_REQUIRE_ALL_CHECKPOINTS:-1}"' in text
    assert "REQUIRED_CHECKPOINT_KINDS=(best_track_p10 best_track_p5 best_metric_track_center_px)" in text
    assert parse_runner_checkpoint_kinds(text) == EXPECTED_CHECKPOINT_KINDS
    assert set(contract["required_checkpoint_kinds"]) == EXPECTED_CHECKPOINT_KINDS
    assert parse_runner_checkpoint_kinds(text) == set(contract["required_checkpoint_kinds"])
    assert 'for CKPT_KIND in "${REQUIRED_CHECKPOINT_KINDS[@]}"; do' in text
    assert "XR64_REQUIRED_CHECKPOINT_MISSING" in text
    assert 'if [ "$XR64_REQUIRE_ALL_CHECKPOINTS" = "1" ]; then' in text
    assert "XR64_EVAL_SKIP" in text


def test_runner_contract_and_writer_checkpoint_kind_sets_match() -> None:
    text = read_runner()
    contract = read_contract()

    assert parse_runner_checkpoint_kinds(text) == EXPECTED_CHECKPOINT_KINDS
    assert set(contract["required_checkpoint_kinds"]) == EXPECTED_CHECKPOINT_KINDS
    assert parse_writer_expected_checkpoint_kinds() == EXPECTED_CHECKPOINT_KINDS


def test_runner_fails_when_eval_outputs_required_by_contract_are_missing() -> None:
    text = read_runner()

    assert 'EVAL_SUMMARY="${EVAL_ROOT}/eval/test/eval_summary.json"' in text
    assert 'EVAL_ROWS="${EVAL_ROOT}/eval/test/eval_rows.json"' in text
    assert "XR64_EVAL_SUMMARY_MISSING" in text
    assert "XR64_EVAL_ROWS_MISSING" in text
    assert "XR64_REQUIRED_EVAL_COUNT_MISMATCH" in text


def test_runner_patches_eval_summary_after_output_presence_checks() -> None:
    text = read_runner()

    summary_check = text.index("XR64_EVAL_SUMMARY_MISSING")
    rows_check = text.index("XR64_EVAL_ROWS_MISSING")
    patch_call = text.index("--patch-eval-summary")
    assert summary_check < patch_call
    assert rows_check < patch_call
