from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_manifest_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_xr64_resume_command_manifest.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_xr64_resume_command_manifest", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_manifest() -> dict:
    path = Path("docs/resources/xr64_resume_command_manifest_2026_06_20.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_xr64_resume_command_manifest_passes() -> None:
    module = load_manifest_module()
    report = module.validate_manifest(read_current_manifest(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["do_not_execute_until_user_resumes"] is True
    assert report["default_allowed_to_run"] is False
    assert report["prep_command_count"] == 10
    assert report["eval_command_count"] == 8
    assert report["build_command_count"] == 2
    assert report["expected_eval_rows"] == 8
    assert report["expected_overrides"] == 6
    assert report["expected_split_train_rows"] == 5929
    assert report["expected_split_val_rows"] == 844
    assert report["expected_teacher_count"] == 4
    assert report["expected_override_rule_count"] == 3
    assert report["expected_total_eval_row_records"] == 27092
    assert report["expected_total_override_records"] == 20319
    assert report["eval_required_input_count"] == 6
    assert report["existing_eval_required_input_count"] == 6
    assert report["missing_eval_required_input_count"] == 0
    assert report["eval_inputs_ready"] is True
    assert report["build_required_input_count"] == 8
    assert report["existing_build_required_input_count"] == 0
    assert report["missing_build_required_input_count"] == 8
    assert report["build_inputs_ready"] is False
    assert report["launch_lanes"] == ["XR-64A", "XR-64B"]
    assert report["postrun_command_count"] == 2
    assert report["prep_runner_contract_ok"] is True
    assert report["prep_runner_contract_check_count"] == 17


def test_manifest_rejects_test_split_command() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["prep_commands"][0]["split"] = "test"
    manifest["prep_commands"][0]["env"]["XR64_SPLITS"] = "test"
    manifest["prep_commands"][0]["command"] = manifest["prep_commands"][0]["command"].replace("XR64_SPLITS=train", "XR64_SPLITS=test")

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("split must be train or val" in error for error in report["errors"])
    assert any("forbidden test" in error for error in report["errors"])


def test_manifest_rejects_allowed_launch_while_paused() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["launch_commands_after_strict_ready"][0]["allowed_to_run"] = True

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("launch allowed_to_run must be false" in error for error in report["errors"])


def test_manifest_rejects_allowed_postrun_while_paused() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["postrun_commands_after_launch"][0]["allowed_to_run"] = True

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("postrun allowed_to_run must be false" in error for error in report["errors"])


def test_manifest_rejects_postrun_train_helper() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["postrun_commands_after_launch"][0]["command"] = "bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0"

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("postrun command must use collect_xr64_postrun_candidates.py" in error for error in report["errors"])
    assert any("postrun command must not run train/eval helpers" in error for error in report["errors"])


def test_manifest_rejects_build_required_input_mismatch() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    build = next(item for item in manifest["prep_commands"] if item["id"] == "XR64-BUILD-TRAIN")
    build["required_inputs"].pop()

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("required_inputs must match all train eval row artifacts" in error for error in report["errors"])


def test_manifest_rejects_eval_required_input_mismatch() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    eval_item = next(item for item in manifest["prep_commands"] if item["id"] == "XR64-EVAL-TRAIN-XR62A")
    eval_item["required_inputs"] = [eval_item["required_inputs"][0]]

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("eval required_inputs must contain split manifest and one checkpoint" in error for error in report["errors"])


def test_manifest_rejects_missing_full_coverage_contract() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest.pop("strict_generated_artifact_contract")

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("strict_generated_artifact_contract.eval_rows_full_manifest_coverage_required" in error for error in report["errors"])


def test_manifest_rejects_subset_ready_contract_drift() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["strict_generated_artifact_contract"]["subset_artifacts_satisfy_ready_to_train"] = True

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("strict_generated_artifact_contract.subset_artifacts_satisfy_ready_to_train" in error for error in report["errors"])


def test_manifest_rejects_expected_split_count_drift() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["expected_generated_counts"]["split_manifest_rows"]["train"] = 8

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("expected_generated_counts.split_manifest_rows" in error for error in report["errors"])


def test_manifest_rejects_expected_total_eval_record_drift() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["expected_generated_counts"]["total_eval_row_records"] = 8

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("expected_generated_counts.total_eval_row_records" in error for error in report["errors"])


def test_manifest_rejects_expected_total_override_record_drift() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["expected_generated_counts"]["total_override_records"] = 6

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("expected_generated_counts.total_override_records" in error for error in report["errors"])


def test_prep_runner_contract_rejects_missing_test_refusal() -> None:
    module = load_manifest_module()
    text = Path("scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh").read_text(encoding="utf-8")
    broken = text.replace("XR64 prep refuses test split", "XR64 prep allows test split")

    errors = module.validate_prep_runner_script_text(broken)

    assert any("prep runner missing test split refusal" in error for error in errors)


def test_prep_runner_contract_rejects_forbidden_test_override() -> None:
    module = load_manifest_module()
    text = Path("scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh").read_text(encoding="utf-8")
    broken = text + "\n# bad future edit: allow_test_target_override=true\n"

    errors = module.validate_prep_runner_script_text(broken)

    assert any("prep runner contains forbidden marker: allow_test_target_override=true" in error for error in errors)
