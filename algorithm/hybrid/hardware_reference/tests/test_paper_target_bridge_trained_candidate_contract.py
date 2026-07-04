from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_contract_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_paper_target_bridge_trained_candidate_contract.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_paper_target_bridge_trained_candidate_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def output_paths(root: Path, contract: dict) -> dict[str, Path]:
    return {item["id"]: root / item["path"] for item in contract["trained_candidate_outputs"]}


def copy_source_artifacts(root: Path, contract: dict) -> None:
    for raw_path in contract["source_artifacts"].values():
        source = Path(raw_path)
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def valid_eval_summary() -> dict:
    return {
        "run_id": "xr64a-demo",
        "split": "test",
        "checkpoint": "runs/XR-64/demo/train/best_track_p10.pt",
        "target_override_path": None,
        "metric_track_center_px": 16.0,
        "metric_track_p10_pct": 36.0,
        "metric_track_p5_pct": 13.0,
        "metric_track_p1_pct": 1.0,
        "lane_id": "XR-64A",
        "checkpoint_kind": "best_track_p10",
        "sample_count": 2238,
        "eval_rows_path": "runs/XR-64/demo/eval_rows.json",
        "resolved_config_path": "runs/XR-64/demo/hypers/resolved_config.json",
        "teacher_provenance_path": "runs/XR-64/demo/train/teacher_provenance.json",
        "ablation_attribution_report_path": "runs/XR-64/demo/train/ablation_attribution_report.json",
        "allow_test_target_override": False,
    }


def valid_leakage_audit() -> dict:
    return {
        "run_id": "xr64a-demo",
        "test_derived_targets_used": False,
        "train_override_paths": ["train_overrides.json"],
        "val_override_paths": ["val_overrides.json"],
        "test_override_paths": [],
        "leakage_risk": "none",
        "decision": "valid",
        "eval_summary_path": "docs/resources/future/paper_target_bridge/trained_candidate_test_eval_summary.json",
        "checked_train_inputs": ["train_manifest.jsonl"],
        "checked_eval_overrides": ["data.track_target_override_path=null"],
        "source_manifest": "data/_internal/manifests/manifest1/test_manifest.jsonl",
        "teacher_checkpoint_id": "xr39_p10",
        "teacher_checkpoint_sha": "abc123",
    }


def write_valid_payloads(root: Path, contract: dict) -> None:
    paths = output_paths(root, contract)
    write_json(paths["trained_candidate_eval_summary"], valid_eval_summary())
    write_json(paths["trained_candidate_leakage_audit"], valid_leakage_audit())


def test_current_trained_candidate_contract_passes() -> None:
    module = load_contract_module()

    report = module.validate_contract(read_current_contract(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["schema_contract_allowed_while_paused"] is True
    assert report["trained_candidate_payload_allowed_while_paused"] is False
    assert report["execute_supported"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["trained_candidate_output_count"] == 2
    assert report["json_output_schema_count"] == 2
    assert report["blocked_output_count"] == 2
    assert report["current_existing_output_count"] == 0


def test_contract_rejects_payload_allowed_while_paused() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["trained_candidate_payload_allowed_while_paused"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "trained_candidate_payload_allowed_while_paused must be false" in report["errors"]


def test_contract_rejects_direct_comparison_claim() -> None:
    module = load_contract_module()
    contract = copy.deepcopy(read_current_contract())
    contract["direct_submission_comparison_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "direct_submission_comparison_allowed must be false" in report["errors"]


def test_contract_rejects_present_payload_while_paused(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    paths = output_paths(tmp_path, contract)
    write_json(paths["trained_candidate_eval_summary"], valid_eval_summary())

    report = module.validate_contract(contract, project_root=tmp_path)

    assert report["ok"] is False
    assert report["current_existing_output_count"] == 1
    assert any("output path must be absent while paused" in error for error in report["errors"])


def test_contract_validates_future_payload_pair(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    write_valid_payloads(tmp_path, contract)

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_all_present=True,
    )

    assert report["ok"] is True
    assert report["current_existing_output_count"] == 2


def test_contract_rejects_eval_summary_without_leakage_audit(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    paths = output_paths(tmp_path, contract)
    write_json(paths["trained_candidate_eval_summary"], valid_eval_summary())

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_all_present=False,
    )

    assert report["ok"] is False
    assert "trained-candidate eval summary and leakage audit must be present together" in report["errors"]


def test_contract_rejects_non_null_target_override(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    write_valid_payloads(tmp_path, contract)
    paths = output_paths(tmp_path, contract)
    payload = valid_eval_summary()
    payload["target_override_path"] = "test_overrides.json"
    write_json(paths["trained_candidate_eval_summary"], payload)

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_all_present=True,
    )

    assert report["ok"] is False
    assert "trained_candidate_eval_summary.target_override_path must be null" in report["errors"]


def test_contract_rejects_allow_test_target_override(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    write_valid_payloads(tmp_path, contract)
    paths = output_paths(tmp_path, contract)
    payload = valid_eval_summary()
    payload["allow_test_target_override"] = True
    write_json(paths["trained_candidate_eval_summary"], payload)

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_all_present=True,
    )

    assert report["ok"] is False
    assert "trained_candidate_eval_summary.allow_test_target_override must be false" in report["errors"]


def test_contract_rejects_leakage_risk(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    write_valid_payloads(tmp_path, contract)
    paths = output_paths(tmp_path, contract)
    payload = valid_leakage_audit()
    payload["leakage_risk"] = "high"
    write_json(paths["trained_candidate_leakage_audit"], payload)

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_all_present=True,
    )

    assert report["ok"] is False
    assert "trained_candidate_leakage_audit.leakage_risk must be none" in report["errors"]


def test_contract_rejects_run_id_mismatch(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    write_valid_payloads(tmp_path, contract)
    paths = output_paths(tmp_path, contract)
    payload = valid_leakage_audit()
    payload["run_id"] = "different-run"
    write_json(paths["trained_candidate_leakage_audit"], payload)

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_all_present=True,
    )

    assert report["ok"] is False
    assert "trained candidate eval summary and leakage audit run_id must match" in report["errors"]


def test_contract_rejects_missing_context_field(tmp_path: Path) -> None:
    module = load_contract_module()
    contract = read_current_contract()
    copy_source_artifacts(tmp_path, contract)
    write_valid_payloads(tmp_path, contract)
    paths = output_paths(tmp_path, contract)
    payload = valid_eval_summary()
    payload.pop("teacher_provenance_path")
    write_json(paths["trained_candidate_eval_summary"], payload)

    report = module.validate_contract(
        contract,
        project_root=tmp_path,
        allow_payload_present=True,
        require_all_present=True,
    )

    assert report["ok"] is False
    assert any("payload missing context fields" in error for error in report["errors"])
