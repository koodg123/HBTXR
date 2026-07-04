from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_emitter_module():
    script_dir = Path("scripts/external").resolve()
    sys.path.insert(0, str(script_dir))
    script = script_dir / "emit_xr64_resume_commands.py"
    spec = importlib.util.spec_from_file_location("emit_xr64_resume_commands", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_manifest() -> dict:
    return json.loads(Path("docs/resources/xr64_resume_command_manifest_2026_06_20.json").read_text(encoding="utf-8"))


def test_build_emission_summary_counts() -> None:
    module = load_emitter_module()
    emission = module.build_emission(read_manifest(), section="summary")

    assert emission["execute_supported"] is False
    assert emission["execution_state"] == "paused_by_user_directive"
    assert emission["counts"]["prep"] == 10
    assert emission["counts"]["eval_commands"] == 8
    assert emission["counts"]["build_commands"] == 2
    assert emission["counts"]["postrun"] == 2
    assert emission["counts"]["expected_eval_rows"] == 8
    assert emission["counts"]["expected_overrides"] == 6
    assert emission["commands"] == []


def test_build_emission_prep_commands_only() -> None:
    module = load_emitter_module()
    emission = module.build_emission(read_manifest(), section="prep")

    assert len(emission["commands"]) == 10
    assert all(item["kind"] == "prep" for item in emission["commands"])
    assert all(item["allowed_to_run"] is False for item in emission["commands"])
    assert any("XR64_SPLITS=train" in item["command"] for item in emission["commands"])
    assert not any("XR64_SPLITS=test" in item["command"] for item in emission["commands"])


def test_build_emission_postrun_commands_only() -> None:
    module = load_emitter_module()
    emission = module.build_emission(read_manifest(), section="postrun")

    assert len(emission["commands"]) == 2
    assert all(item["kind"] == "postrun" for item in emission["commands"])
    assert all(item["allowed_to_run"] is False for item in emission["commands"])
    assert any("--format summary" in item["command"] for item in emission["commands"])
    assert any("--format commands" in item["command"] for item in emission["commands"])
    assert all("collect_xr64_postrun_candidates.py" in item["command"] for item in emission["commands"])
    assert not any("run_xr64_teacher_target_construction.sh" in item["command"] for item in emission["commands"])


def test_format_script_comments_out_commands() -> None:
    module = load_emitter_module()
    emission = module.build_emission(read_manifest(), section="launch")
    text = module.format_script(emission)

    assert "intentionally commented out" in text
    assert "# bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0" in text
    assert "\nbash scripts/external/run_xr64_teacher_target_construction.sh" not in text


def test_format_commands_is_review_only() -> None:
    module = load_emitter_module()
    emission = module.build_emission(read_manifest(), section="prep")
    text = module.format_commands(emission)

    assert "execute_supported=false" in text
    assert "Command lines below are commented intentionally" in text
    assert "# command: XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr62a" in text
    assert "\nXR64_ACTIONS=eval" not in text
    assert "\nXR64_ACTIONS=build" not in text
