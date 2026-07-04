"""Completion audit over generated HG-PIPE quantization reports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditItem:
    requirement: str
    status: str
    evidence: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == "complete"

    def to_json(self) -> dict[str, str]:
        return {
            "requirement": self.requirement,
            "status": self.status,
            "evidence": self.evidence,
            "detail": self.detail,
        }


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _list_report_item(requirement: str, reports: Path, filename: str, expected_count: int) -> AuditItem:
    path = reports / filename
    data = _read_json(path)
    if not isinstance(data, list):
        return AuditItem(requirement, "missing", str(path), "report is missing or is not a list")
    passed = sum(1 for row in data if row.get("passed") is True)
    mismatches = sum(int(row.get("mismatches", 0)) for row in data)
    status = "complete" if len(data) == expected_count and passed == expected_count and mismatches == 0 else "incomplete"
    detail = f"count={len(data)} expected={expected_count} passed={passed} mismatches={mismatches}"
    return AuditItem(requirement, status, str(path), detail)


def _contract_item(reports: Path) -> AuditItem:
    path = reports / "quant_contracts.json"
    data = _read_json(path)
    if not isinstance(data, list):
        return AuditItem("scale/scalar, dtype, range, zero-point policy contract export", "missing", str(path), "report is missing or is not a list")
    has_required = False
    for row in data:
        params = row.get("params") or {}
        observed = row.get("observed_range") or {}
        if {"scalars", "shift_scale", "effective_divisor", "bound", "zero_point", "table_sizes"}.issubset(params) and observed:
            has_required = True
            break
    status = "complete" if len(data) == 97 and has_required else "incomplete"
    return AuditItem(
        "scale/scalar, dtype, range, zero-point policy contract export",
        status,
        str(path),
        f"contracts={len(data)} has_required_fields={has_required}",
    )


def _runner_compare_item(reports: Path) -> AuditItem:
    path = reports / "run_compare_result.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return AuditItem("torch.int versus FakeQuant graph final-output comparison", "missing", str(path), "report is missing or is not an object")
    comparison = data.get("comparison") or {}
    status = "complete" if comparison.get("passed") is True and comparison.get("mismatches") == 0 and comparison.get("top1_equal") is True else "incomplete"
    detail = "passed={} mismatches={} top1_equal={}".format(comparison.get("passed"), comparison.get("mismatches"), comparison.get("top1_equal"))
    return AuditItem("torch.int versus FakeQuant graph final-output comparison", status, str(path), detail)


def _image_bridge_item(reports: Path) -> AuditItem:
    scale_path = reports / "input_scale_estimate.json"
    image_path = reports / "run_compare_image_result.json"
    scale = _read_json(scale_path)
    image = _read_json(image_path)
    if not isinstance(scale, dict) or not isinstance(image, dict):
        return AuditItem("experimental explicit-scale image bridge", "missing", f"{scale_path}, {image_path}", "one or both reports are missing")
    comparison = image.get("comparison") or {}
    bridge = image.get("input_bridge") or {}
    ok = (
        scale.get("paper_equivalent") is False
        and bridge.get("paper_equivalent") is False
        and comparison.get("passed") is True
        and comparison.get("mismatches") == 0
    )
    status = "complete" if ok else "incomplete"
    detail = (
        "scale={} scale_paper_equivalent={} comparison_passed={} bridge_paper_equivalent={}".format(
            scale.get("scale"),
            scale.get("paper_equivalent"),
            comparison.get("passed"),
            bridge.get("paper_equivalent"),
        )
    )
    return AuditItem("experimental explicit-scale image bridge", status, f"{scale_path}, {image_path}", detail)


def _timm_eval_item(reports: Path) -> AuditItem:
    paths = [reports / 'imagenet_accuracy_int8_int4.json', reports / 'imagenet_accuracy_w4a8.json']
    datasets = [_read_json(path) for path in paths]
    if not all(isinstance(data, list) for data in datasets):
        detail = ', '.join('{}={}'.format(path.name, len(data) if isinstance(data, list) else 'missing') for path, data in zip(paths, datasets))
        return AuditItem('PyTorch timm ImageNet fake-quant evaluation reports', 'missing', ', '.join(str(path) for path in paths), detail)

    rows = [row for data in datasets for row in data]
    expected_models = {'deit_tiny_patch16_224', 'deit_small_patch16_224', 'vit_tiny_patch16_224'}
    expected_precisions = {'int8', 'int4', 'w4a8'}
    observed_pairs = {(row.get('model'), row.get('precision')) for row in rows}
    expected_pairs = {(model, precision) for model in expected_models for precision in expected_precisions}
    required_fields = {"model", "precision", "samples", "top1", "top5", "elapsed_sec", "images_per_sec", "pretrained", "device", "paper_model", "timm_model_name", "evaluation_mode", "quantization_flow", "paper_equivalent", "dataset_path", "dataset_split", "eval_script", "command"}

    rows_with_required_fields = sum(1 for row in rows if required_fields.issubset(row))
    full_val_rows = sum(1 for row in rows if isinstance(row.get('samples'), int) and row.get('samples') >= 50000)
    cuda_rows = sum(1 for row in rows if row.get('device') == 'cuda')
    pretrained_rows = sum(1 for row in rows if row.get('pretrained') is True)
    provenance_rows = sum(1 for row in rows if row.get("evaluation_mode") == "timm_fake_quant" and row.get("quantization_flow") == "fake_quant" and row.get("paper_equivalent") is False and row.get("dataset_split") == "val")
    valid_metric_rows = sum(
        1
        for row in rows
        if isinstance(row.get('top1'), (int, float))
        and isinstance(row.get('top5'), (int, float))
        and 0.0 <= float(row.get('top1')) <= 100.0
        and 0.0 <= float(row.get('top5')) <= 100.0
        and float(row.get('top1')) <= float(row.get('top5'))
    )
    missing_pairs = sorted(expected_pairs - observed_pairs)
    row_count_ok = len(rows) == len(expected_pairs)
    ok = (
        row_count_ok
        and not missing_pairs
        and rows_with_required_fields == len(rows)
        and full_val_rows == len(rows)
        and cuda_rows == len(rows)
        and pretrained_rows == len(rows)
        and provenance_rows == len(rows)
        and valid_metric_rows == len(rows)
    )
    detail = (
        'rows={} expected={} required_fields={}/{} full_val={}/{} cuda={}/{} pretrained={}/{} provenance={}/{} valid_metrics={}/{} missing_pairs={}'.format(
            len(rows),
            len(expected_pairs),
            rows_with_required_fields,
            len(rows),
            full_val_rows,
            len(rows),
            cuda_rows,
            len(rows),
            pretrained_rows,
            len(rows),
            provenance_rows,
            len(rows),
            valid_metric_rows,
            len(rows),
            missing_pairs,
        )
    )
    return AuditItem('PyTorch timm ImageNet fake-quant evaluation reports', 'complete' if ok else 'incomplete', ', '.join(str(path) for path in paths), detail)

def _paper_equivalence_assets_item(reports: Path) -> AuditItem:
    path = reports / "paper_equivalence_assets.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return AuditItem("paper-equivalence asset preflight report", "missing", str(path), "report is missing or is not an object")
    requirements = data.get("requirements")
    ok = isinstance(requirements, list) and len(requirements) == 5 and "paper_equivalent_ready" in data and "missing" in data
    detail = "status={} present={}/{} missing={} ready={}".format(data.get("status"), data.get("present"), data.get("total"), data.get("missing"), data.get("paper_equivalent_ready"))
    return AuditItem("paper-equivalence asset preflight report", "complete" if ok else "incomplete", str(path), detail)

def _experimental_artifact_imagenet_item(reports: Path) -> AuditItem:
    path = reports / "artifact_imagenet_accuracy.json"
    data = _read_json(path)
    if not isinstance(data, list):
        return AuditItem("artifact-backed image-batch report", "missing", str(path), "report is missing or is not a list")
    required_fields = {"model", "precision", "samples", "top1", "top5", "evaluation_mode", "quantization_flow", "paper_equivalent", "runner_comparison_passed", "runner_comparison_mismatches", "samples_detail"}
    rows_with_required_fields = sum(1 for row in data if required_fields.issubset(row))
    experimental_rows = sum(
        1
        for row in data
        if row.get("evaluation_mode") == "hgpipe_artifact_graph_experimental"
        and row.get("quantization_flow") == "input_bridge_explicit_scale"
        and row.get("paper_equivalent") is False
        and row.get("runner_comparison_passed") is True
        and row.get("runner_comparison_mismatches") == 0
        and isinstance(row.get("samples"), int)
        and row.get("samples") > 0
    )
    artifact_patch_rows = sum(
        1
        for row in data
        if row.get("evaluation_mode") == "hgpipe_artifact_graph"
        and row.get("quantization_flow") in {"torch_int", "fakequant_graph"}
        and row.get("runner_comparison_passed") is True
        and row.get("runner_comparison_mismatches") == 0
        and isinstance(row.get("samples"), int)
        and row.get("samples") > 0
    )
    valid_metric_rows = sum(
        1
        for row in data
        if isinstance(row.get("top1"), (int, float))
        and isinstance(row.get("top5"), (int, float))
        and 0.0 <= float(row.get("top1")) <= 100.0
        and 0.0 <= float(row.get("top5")) <= 100.0
        and float(row.get("top1")) <= float(row.get("top5"))
    )
    covered_rows = experimental_rows + artifact_patch_rows
    ok = len(data) > 0 and rows_with_required_fields == len(data) and covered_rows == len(data) and valid_metric_rows == len(data)
    detail = "rows={} required_fields={}/{} experimental_rows={} artifact_patch_rows={} covered_rows={}/{} valid_metrics={}/{}".format(len(data), rows_with_required_fields, len(data), experimental_rows, artifact_patch_rows, covered_rows, len(data), valid_metric_rows, len(data))
    return AuditItem("artifact-backed image-batch report", "complete" if ok else "incomplete", str(path), detail)


def _artifact_patch_matrix_manifest_validation_item(reports: Path) -> AuditItem:
    path = reports / "artifact_patch_matrix_manifest_validation.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return AuditItem("artifact patch matrix manifest preflight report", "missing", str(path), "report is missing or is not an object")
    ok = isinstance(data.get("passed"), bool) and data.get("expected_rows") == 9 and data.get("status") in {"passed", "failed"}
    detail = "status={} rows={}/{} missing_pairs={} existing_patch_inputs={} existing_labels={} errors={}".format(
        data.get("status"),
        data.get("rows"),
        data.get("expected_rows"),
        len(data.get("missing_pairs", [])) if isinstance(data.get("missing_pairs"), list) else "unknown",
        data.get("existing_patch_input_files"),
        data.get("existing_label_files"),
        data.get("errors"),
    )
    return AuditItem("artifact patch matrix manifest preflight report", "complete" if ok else "incomplete", str(path), detail)

def _artifact_imagenet_validation_item(reports: Path) -> AuditItem:
    path = reports / "artifact_imagenet_validation.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return AuditItem("artifact ImageNet paper-equivalence validator report", "missing", str(path), "report is missing or is not an object")
    ok = isinstance(data.get("passed"), bool) and data.get("expected_rows") == 9 and data.get("status") in {"passed", "failed"}
    detail = "status={} rows={}/{} missing_pairs={} paper_equivalent_rows={} errors={}".format(
        data.get("status"),
        data.get("rows"),
        data.get("expected_rows"),
        len(data.get("missing_pairs", [])) if isinstance(data.get("missing_pairs"), list) else "unknown",
        data.get("paper_equivalent_rows"),
        data.get("errors"),
    )
    return AuditItem("artifact ImageNet paper-equivalence validator report", "complete" if ok else "incomplete", str(path), detail)


def _artifact_imagenet_item(reports: Path) -> AuditItem:
    path = reports / "artifact_imagenet_accuracy.json"
    data = _read_json(path)
    if not isinstance(data, list):
        return AuditItem(
            "artifact-backed HG-PIPE ImageNet paper-equivalence report",
            "partial",
            str(path),
            "report is absent; artifact graph is verified on saved patch input, but arbitrary ImageNet paper-equivalent accuracy still lacks original calibration/QAT/export flow",
        )
    required_fields = {"model", "precision", "samples", "top1", "top5", "evaluation_mode", "quantization_flow", "paper_equivalent"}
    expected_models = {"deit_tiny_patch16_224", "deit_small_patch16_224", "vit_tiny_patch16_224"}
    expected_precisions = {"int8", "int4", "w4a8"}
    observed_pairs = {(row.get("model"), row.get("precision")) for row in data}
    expected_pairs = {(model, precision) for model in expected_models for precision in expected_precisions}
    complete_rows = sum(
        1
        for row in data
        if required_fields.issubset(row)
        and row.get("evaluation_mode") == "hgpipe_artifact_graph"
        and row.get("quantization_flow") in {"torch_int", "fakequant_graph"}
        and row.get("paper_equivalent") is True
    )
    missing_pairs = sorted(expected_pairs - observed_pairs)
    ok = len(data) == len(expected_pairs) and complete_rows == len(data) and not missing_pairs
    status = "complete" if ok else "partial"
    detail = "rows={} expected={} artifact_rows={} missing_pairs={}".format(len(data), len(expected_pairs), complete_rows, missing_pairs)
    return AuditItem("artifact-backed HG-PIPE ImageNet paper-equivalence report", status, str(path), detail)

def audit_completion(project_root: str | Path = Path(".")) -> dict[str, Any]:
    root = Path(project_root)
    reports = root / "reports"
    items = [
        _contract_item(reports),
        _list_report_item("core quantization kernels", reports, "verification.json", 97),
        _list_report_item("artifact component graph", reports, "graph_verification.json", 268),
        _list_report_item("single-input end-to-end graph", reports, "e2e_graph_verification.json", 293),
        _list_report_item("torch.int end-to-end graph", reports, "torch_int_verification.json", 293),
        _list_report_item("FakeQuant LUT cases", reports, "fakequant_verification.json", 60),
        _list_report_item("FakeQuant graph", reports, "fakequant_graph_verification.json", 293),
        _runner_compare_item(reports),
        _image_bridge_item(reports),
        _timm_eval_item(reports),
        _paper_equivalence_assets_item(reports),
        _experimental_artifact_imagenet_item(reports),
        _artifact_imagenet_validation_item(reports),
        _artifact_patch_matrix_manifest_validation_item(reports),
        _artifact_imagenet_item(reports),
    ]
    complete = sum(1 for item in items if item.status == "complete")
    partial = sum(1 for item in items if item.status == "partial")
    missing = sum(1 for item in items if item.status == "missing")
    incomplete = sum(1 for item in items if item.status == "incomplete")
    residual_limits = [
        "Source-backed HG-PIPE LUT contracts expose zero_point=None because affine zero-point artifacts are absent.",
        "Artifact-backed arbitrary ImageNet accuracy is not paper-equivalent without the original preprocessing/calibration/QAT export flow.",
    ]
    return {
        "complete": complete,
        "partial": partial,
        "missing": missing,
        "incomplete": incomplete,
        "total": len(items),
        "passed": incomplete == 0 and missing == 0,
        "fully_complete": partial == 0 and incomplete == 0 and missing == 0,
        "items": [item.to_json() for item in items],
        "residual_limits": residual_limits,
    }


def write_completion_audit_markdown(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    status = "PASS" if payload.get("fully_complete") else ("PARTIAL" if payload["passed"] else "FAIL")
    lines = [
        "# HG-PIPE Quantization Completion Audit",
        "",
        "## Summary",
        "",
        "- Status: {}".format(status),
        "- Complete: {}/{}".format(payload["complete"], payload["total"]),
        "- Incomplete: {}".format(payload["incomplete"]),
        "- Partial: {}".format(payload["partial"]),
        "- Missing: {}".format(payload["missing"]),
        "",
        "## Requirement Matrix",
        "",
        "| Requirement | Status | Evidence | Detail |",
        "|---|---|---|---|",
    ]
    for item in payload["items"]:
        lines.append("| {} | {} | {} | {} |".format(item["requirement"], item["status"], item["evidence"], item["detail"]))
    refresh = payload.get("refresh")
    if isinstance(refresh, dict):
        lines.extend(["", "## Refresh", ""])
        lines.append("- Reports dir: {}".format(refresh.get("reports_dir")))
        lines.append("- Device: {}".format(refresh.get("device")))
        lines.append("- Files:")
        for file_name in refresh.get("files", []):
            lines.append("- {}".format(file_name))
    lines.extend(["", "## Residual Limits", ""])
    for limit in payload["residual_limits"]:
        lines.append("- {}".format(limit))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
