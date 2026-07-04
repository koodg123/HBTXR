from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_selector_module():
    script_dir = Path("scripts/external").resolve()
    sys.path.insert(0, str(script_dir))
    script = script_dir / "emit_xr64_next_prep_command.py"
    spec = importlib.util.spec_from_file_location("emit_xr64_next_prep_command", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_manifest() -> dict:
    return json.loads(Path("docs/resources/xr64_resume_command_manifest_2026_06_20.json").read_text(encoding="utf-8"))


def minimal_prep_runner_text() -> str:
    return "\n".join(
        [
            "set -Eeuo pipefail",
            "XR64 prep refuses test split",
            'SPLIT_CSV="${XR64_SPLITS:-train,val}"',
            'TEACHER_CSV="${XR64_TEACHERS:-xr62a,xr39,xr56b,xr58a}"',
            'ACTION_CSV="${XR64_ACTIONS:-eval,build}"',
            "--override data.track_target_override_path=null",
            "--override data.allow_test_target_override=false",
            "XR64_EVAL_START",
            "XR64_EVAL_DONE",
            "XR64_OVERRIDE_START",
            "XR64_OVERRIDE_DONE",
            "validate_json_rows",
            "json_row_count",
            '2>&1 | tee -a "$LOG"',
            "eval_rows:${split}:${teacher}:existing",
            "override:${split}:${rule}:existing",
            "eval_rows:${split}:${teacher}:build-input",
        ]
    )


def split_manifest_text(raw_path: str, manifest: dict) -> str | None:
    rows = (manifest.get("expected_generated_counts") or {}).get("split_manifest_rows") or {}
    if raw_path.endswith("train_manifest.jsonl"):
        return "\n".join(f'{{"sample_id":"train_{idx}"}}' for idx in range(int(rows["train"]))) + "\n"
    if raw_path.endswith("val_manifest.jsonl"):
        return "\n".join(f'{{"sample_id":"val_{idx}"}}' for idx in range(int(rows["val"]))) + "\n"
    return None


def create_fake_project(root: Path, manifest: dict) -> None:
    for raw_path in (manifest.get("source_artifacts") or {}).values():
        path = root / raw_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path == "scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh":
            path.write_text(minimal_prep_runner_text(), encoding="utf-8")
        else:
            path.write_text("ok\n", encoding="utf-8")
    for command in manifest["prep_commands"]:
        for raw_path in command.get("required_inputs") or []:
            path = root / raw_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if raw_path.endswith(".json"):
                continue
            text = split_manifest_text(raw_path, manifest) or "ok\n"
            path.write_text(text, encoding="utf-8")


def write_eval_rows(root: Path, raw_path: str) -> None:
    path = root / raw_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"sample_id": "s0", "track_state": [1, 2, 3, 4, 5, 6]}]), encoding="utf-8")


def test_current_project_next_command_is_empty_after_prep_complete() -> None:
    module = load_selector_module()
    selection = module.build_selection(read_manifest(), project_root=Path.cwd(), selection="next")

    assert selection["ok"] is True
    assert selection["execute_supported"] is False
    assert selection["counts"]["ready_after_resume"] == 0
    assert selection["counts"]["blocked"] == 0
    assert selection["counts"]["completed"] == 10
    assert selection["next_command"] is None


def test_fake_completed_first_eval_advances_next_command(tmp_path: Path) -> None:
    module = load_selector_module()
    manifest = read_manifest()
    create_fake_project(tmp_path, manifest)
    write_eval_rows(tmp_path, manifest["prep_commands"][0]["expected_artifact"])

    selection = module.build_selection(manifest, project_root=tmp_path, selection="next")

    assert selection["ok"] is True
    assert selection["counts"]["completed"] == 1
    assert selection["counts"]["ready_after_resume"] == 7
    assert selection["next_command"]["id"] == "XR64-EVAL-TRAIN-XR39"


def test_fake_train_eval_rows_make_train_build_ready(tmp_path: Path) -> None:
    module = load_selector_module()
    manifest = read_manifest()
    create_fake_project(tmp_path, manifest)
    for command in manifest["prep_commands"]:
        if command.get("action") == "eval" and command.get("split") == "train":
            write_eval_rows(tmp_path, command["expected_artifact"])

    selection = module.build_selection(manifest, project_root=tmp_path, selection="all")
    by_id = {item["id"]: item for item in selection["commands"]}

    assert by_id["XR64-BUILD-TRAIN"]["status"] == "ready_after_resume"
    assert by_id["XR64-BUILD-VAL"]["status"] == "blocked_missing_inputs"
    assert selection["counts"]["completed"] == 4
    assert selection["counts"]["ready_after_resume"] == 5
    assert selection["counts"]["blocked"] == 1


def test_commands_output_is_review_only() -> None:
    module = load_selector_module()
    selection = module.build_selection(read_manifest(), project_root=Path.cwd(), selection="next")
    text = module.format_commands(selection)

    assert "execute_supported=false" in text
    assert "Command lines below are commented intentionally" in text
    assert "\nXR64_ACTIONS=eval" not in text
    assert "# command:" not in text
