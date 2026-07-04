from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_dag_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_xr64_resume_execution_dag.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_xr64_resume_execution_dag", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_dag() -> dict:
    path = Path("docs/resources/xr64_resume_execution_dag_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_xr64_resume_execution_dag_passes() -> None:
    module = load_dag_module()
    report = module.validate_dag(read_current_dag(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execute_supported"] is False
    assert report["allowed_while_paused"] is True
    assert report["node_count"] == 17
    assert report["phase_count"] == 6
    assert report["prep_nodes"] == 10
    assert report["eval_nodes"] == 8
    assert report["build_nodes"] == 2
    assert report["launch_nodes"] == 2
    assert report["postrun_nodes"] == 2
    assert report["ready_to_train"] is False
    assert report["missing_eval_rows"] == 8
    assert report["missing_overrides"] == 6
    assert report["current_next_node"] == "XR64-EVAL-TRAIN-XR62A"
    assert report["manifest_ok"] is True
    assert report["design_contract_ok"] is True


def test_dag_rejects_unpaused_execution_flag() -> None:
    module = load_dag_module()
    dag = copy.deepcopy(read_current_dag())
    dag["execute_supported"] = True
    dag["nodes"][2]["allowed_to_run_now"] = True

    report = module.validate_dag(dag, project_root=Path.cwd())

    assert report["ok"] is False
    assert "execute_supported must be false" in report["errors"]
    assert "XR64-EVAL-TRAIN-XR62A: allowed_to_run_now must be false" in report["errors"]


def test_dag_rejects_test_split_node() -> None:
    module = load_dag_module()
    dag = copy.deepcopy(read_current_dag())
    dag["nodes"][2]["split"] = "test"

    report = module.validate_dag(dag, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR64-EVAL-TRAIN-XR62A: test split is forbidden" in report["errors"]


def test_dag_rejects_launch_without_strict_gate() -> None:
    module = load_dag_module()
    dag = copy.deepcopy(read_current_dag())
    for node in dag["nodes"]:
        if node["id"] == "XR64A-LAUNCH":
            node["depends_on"] = ["XR64-BUILD-TRAIN"]

    report = module.validate_dag(dag, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR64A-LAUNCH: must depend on XR64-STRICT-READY" in report["errors"]
