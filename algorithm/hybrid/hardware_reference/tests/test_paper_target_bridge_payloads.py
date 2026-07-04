from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_payload_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_paper_target_bridge_payloads.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_paper_target_bridge_payloads", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_valid_payloads(root: Path, contract: dict) -> None:
    outputs = {item["id"]: item for item in contract["evaluator_outputs"]}
    write_jsonl(
        root / outputs["sensor_space_eval_rows"]["path"],
        [
            {
                "sample_id": "s0",
                "split": "test",
                "pred_center_transform_xy": [1.0, 2.0],
                "target_center_transform_xy": [1.5, 2.5],
                "pred_center_sensor_xy": [10.0, 20.0],
                "target_center_sensor_xy": [11.0, 21.0],
                "center_error_sensor_px": 1.41421356,
            }
        ],
    )
    write_json(
        root / outputs["coordinate_transform_audit"]["path"],
        {
            "metric_frame": "sensor_px",
            "transform_policy": "inverse_manifest_transform",
            "directly_sensor_px": True,
            "sample_count": 1,
            "decision": "valid",
        },
    )
    write_jsonl(
        root / outputs["hybrid_eval_rows"]["path"],
        [
            {
                "sample_id": "s0",
                "split": "test",
                "chosen_mode": "track",
                "chosen_state": {"x": 10.0, "y": 20.0},
                "center_error": 1.0,
                "p10_hit": True,
                "p5_hit": True,
                "p1_hit": False,
                "target_definition": "cur_state",
            }
        ],
    )
    write_json(
        root / outputs["hybrid_metric_report"]["path"],
        {
            "metric_frame": "sensor_px",
            "sample_count": 1,
            "metric_hybrid_center_px": 1.0,
            "metric_hybrid_p10_pct": 100.0,
            "metric_hybrid_p5_pct": 100.0,
            "metric_hybrid_p1_pct": 0.0,
        },
    )
    write_json(
        root / outputs["paper_frame_full_test_p1"]["path"],
        {
            "metric_frame": "sensor_px",
            "split": "test",
            "sample_count": 1,
            "metric_track_p1_pct": 0.0,
            "paper_frame_match": True,
        },
    )


def test_current_payload_check_passes_absent_paused_state() -> None:
    module = load_payload_module()

    report = module.validate_payloads(
        read_current_contract(),
        project_root=Path.cwd(),
        allow_present=False,
        require_all_present=False,
    )

    assert report["ok"] is True
    assert report["execute_supported"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["payload_present_count"] == 0
    assert report["payload_missing_count"] == 5
    assert report["validated_payload_count"] == 0


def test_payload_check_rejects_present_output_while_paused(tmp_path: Path) -> None:
    module = load_payload_module()
    contract = read_current_contract()
    output = next(item for item in contract["evaluator_outputs"] if item["id"] == "sensor_space_eval_rows")
    write_jsonl(tmp_path / output["path"], [{"sample_id": "s0"}])

    report = module.validate_payloads(
        contract,
        project_root=tmp_path,
        allow_present=False,
        require_all_present=False,
    )

    assert report["ok"] is False
    assert report["payload_present_count"] == 1
    assert any("sensor_space_eval_rows output path must be absent while paused" in error for error in report["errors"])


def test_payload_check_validates_future_complete_payloads(tmp_path: Path) -> None:
    module = load_payload_module()
    contract = read_current_contract()
    write_valid_payloads(tmp_path, contract)

    report = module.validate_payloads(
        contract,
        project_root=tmp_path,
        allow_present=True,
        require_all_present=True,
    )

    assert report["ok"] is True
    assert report["payload_present_count"] == 5
    assert report["payload_missing_count"] == 0
    assert report["validated_payload_count"] == 5


def test_payload_check_requires_all_future_payloads_when_requested(tmp_path: Path) -> None:
    module = load_payload_module()
    contract = read_current_contract()

    report = module.validate_payloads(
        contract,
        project_root=tmp_path,
        allow_present=True,
        require_all_present=True,
    )

    assert report["ok"] is False
    assert report["payload_missing_count"] == 5
    assert any("missing required PTB-1 payloads" in error for error in report["errors"])


def test_payload_check_rejects_missing_required_row_field(tmp_path: Path) -> None:
    module = load_payload_module()
    contract = read_current_contract()
    bad_contract = copy.deepcopy(contract)
    output = next(item for item in bad_contract["evaluator_outputs"] if item["id"] == "hybrid_eval_rows")
    write_jsonl(tmp_path / output["path"], [{"sample_id": "s0", "split": "test"}])

    report = module.validate_payloads(
        bad_contract,
        project_root=tmp_path,
        allow_present=True,
        require_all_present=False,
    )

    assert report["ok"] is False
    assert any("hybrid_eval_rows row 1 missing fields" in error for error in report["errors"])


def test_payload_check_rejects_hybrid_metric_count_mismatch(tmp_path: Path) -> None:
    module = load_payload_module()
    contract = read_current_contract()
    write_valid_payloads(tmp_path, contract)
    outputs = {item["id"]: item for item in contract["evaluator_outputs"]}
    metric_path = tmp_path / outputs["hybrid_metric_report"]["path"]
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    metric["sample_count"] = 2
    write_json(metric_path, metric)

    report = module.validate_payloads(
        contract,
        project_root=tmp_path,
        allow_present=True,
        require_all_present=True,
    )

    assert report["ok"] is False
    assert "hybrid_metric_report.sample_count must match hybrid_eval_rows row count" in report["errors"]


def test_payload_check_rejects_invalid_p1_split(tmp_path: Path) -> None:
    module = load_payload_module()
    contract = read_current_contract()
    write_valid_payloads(tmp_path, contract)
    outputs = {item["id"]: item for item in contract["evaluator_outputs"]}
    p1_path = tmp_path / outputs["paper_frame_full_test_p1"]["path"]
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    p1["split"] = "val"
    write_json(p1_path, p1)

    report = module.validate_payloads(
        contract,
        project_root=tmp_path,
        allow_present=True,
        require_all_present=True,
    )

    assert report["ok"] is False
    assert "paper_frame_full_test_p1.split must be test" in report["errors"]


def test_payload_check_rejects_direct_comparison_claim() -> None:
    module = load_payload_module()
    contract = read_current_contract()
    contract["direct_submission_comparison_allowed"] = True

    report = module.validate_payloads(
        contract,
        project_root=Path.cwd(),
        allow_present=False,
        require_all_present=False,
    )

    assert report["ok"] is False
    assert "contract direct_submission_comparison_allowed must be false" in report["errors"]


def test_payload_check_rejects_train_eval_validation_command() -> None:
    module = load_payload_module()
    contract = read_current_contract()
    contract["validation_commands"].append(".venv/bin/python eval_hbtxr.py")

    report = module.validate_payloads(
        contract,
        project_root=Path.cwd(),
        allow_present=False,
        require_all_present=False,
    )

    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])


def test_payload_check_rejects_empty_jsonl(tmp_path: Path) -> None:
    module = load_payload_module()
    contract = read_current_contract()
    output = next(item for item in contract["evaluator_outputs"] if item["id"] == "sensor_space_eval_rows")
    path = tmp_path / output["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    report = module.validate_payloads(
        contract,
        project_root=tmp_path,
        allow_present=True,
        require_all_present=False,
    )

    assert report["ok"] is False
    assert any("JSONL payload must contain at least one row" in error for error in report["errors"])
