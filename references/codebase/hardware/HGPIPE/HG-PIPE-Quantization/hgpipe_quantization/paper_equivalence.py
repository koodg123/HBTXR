from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIREMENTS = [
    {"name": "original_image_to_patch_quantization_policy", "patterns": ["preprocess", "image_to_patch", "patch_input", "calibration", "calib"], "description": "Original preprocessing or image-to-int8 patch input quantization policy.", "required_for": "Converting ImageNet images into paper-equivalent HG-PIPE patch input tensors."},
    {"name": "qat_or_calibration_flow", "patterns": ["qat", "calibration", "calib", "train", "finetune"], "description": "Original QAT or calibration scripts and configs.", "required_for": "Reproducing paper-equivalent activation scales and tables beyond saved reference inputs."},
    {"name": "quantized_model_checkpoints", "patterns": [".pt", ".pth", ".ckpt", ".onnx", "checkpoint", "weights"], "description": "Quantized model checkpoints or exported model state for all paper models and precisions.", "required_for": "Evaluating DeiT-tiny, DeiT-small, and ViT-tiny under int8, int4, and W4A8 without synthetic weights."},
    {"name": "model_export_or_ref_generation_flow", "patterns": ["export", "generate", "gen_ref", "ref_gen", "convert"], "description": "Scripted flow that exports weights, scales, tables, and ImageNet-ready refs.", "required_for": "Regenerating HG-PIPE artifacts from model checkpoints instead of only replaying checked-in refs."},
    {"name": "full_artifact_imagenet_accuracy_matrix", "patterns": ["artifact_imagenet_accuracy"], "description": "Artifact-backed ImageNet report covering three models by int8, int4, and W4A8 with paper_equivalent true.", "required_for": "Closing the remaining completion-audit partial item."},
]

PAPER_EQUIVALENT_MODELS = ("deit_tiny_patch16_224", "deit_small_patch16_224", "vit_tiny_patch16_224")
PAPER_EQUIVALENT_PRECISIONS = ("int8", "int4", "w4a8")
ARTIFACT_IMAGENET_REQUIRED_FIELDS = {"model", "precision", "samples", "top1", "top5", "evaluation_mode", "quantization_flow", "paper_equivalent"}

def expected_artifact_imagenet_pairs() -> list[dict[str, str]]:
    return [
        {"model": model, "precision": precision}
        for model in PAPER_EQUIVALENT_MODELS
        for precision in PAPER_EQUIVALENT_PRECISIONS
    ]

def _pair_key(row: dict[str, Any]) -> tuple[str | None, str | None]:
    model = row.get("model")
    precision = row.get("precision")
    return (model if isinstance(model, str) else None, precision if isinstance(precision, str) else None)

def _is_valid_metric(row: dict[str, Any]) -> bool:
    top1 = row.get("top1")
    top5 = row.get("top5")
    if not isinstance(top1, (int, float)) or not isinstance(top5, (int, float)):
        return False
    return 0.0 <= float(top1) <= 100.0 and 0.0 <= float(top5) <= 100.0 and float(top1) <= float(top5)

def validate_artifact_imagenet_report(report: Any = Path("reports/artifact_imagenet_accuracy.json")) -> dict[str, object]:
    report = Path(report)
    expected_pairs = {(row["model"], row["precision"]) for row in expected_artifact_imagenet_pairs()}
    errors: list[str] = []
    if not report.exists():
        rows: Any = []
        errors.append("report_missing")
    else:
        try:
            rows = json.loads(report.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            rows = []
            errors.append("report_unreadable:{}".format(type(exc).__name__))
    if not isinstance(rows, list):
        errors.append("report_not_list")
        rows = []
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if len(dict_rows) != len(rows):
        errors.append("non_object_rows={}".format(len(rows) - len(dict_rows)))
    observed_pairs = {_pair_key(row) for row in dict_rows}
    present_pairs = sorted(pair for pair in observed_pairs if pair in expected_pairs)
    missing_pairs = sorted(expected_pairs - observed_pairs)
    unexpected_pairs = sorted(pair for pair in observed_pairs if pair not in expected_pairs)
    rows_with_required_fields = sum(1 for row in dict_rows if ARTIFACT_IMAGENET_REQUIRED_FIELDS.issubset(row))
    paper_equivalent_rows = sum(1 for row in dict_rows if row.get("paper_equivalent") is True)
    artifact_rows = sum(1 for row in dict_rows if row.get("evaluation_mode") == "hgpipe_artifact_graph")
    quant_flow_rows = sum(1 for row in dict_rows if row.get("quantization_flow") in {"torch_int", "fakequant_graph"})
    valid_metric_rows = sum(1 for row in dict_rows if _is_valid_metric(row))
    valid_sample_rows = sum(1 for row in dict_rows if isinstance(row.get("samples"), int) and row.get("samples") > 0)
    if len(dict_rows) != len(expected_pairs):
        errors.append("row_count")
    if missing_pairs:
        errors.append("missing_pairs")
    if unexpected_pairs:
        errors.append("unexpected_pairs")
    if rows_with_required_fields != len(dict_rows):
        errors.append("required_fields")
    if paper_equivalent_rows != len(dict_rows):
        errors.append("paper_equivalent")
    if artifact_rows != len(dict_rows):
        errors.append("evaluation_mode")
    if quant_flow_rows != len(dict_rows):
        errors.append("quantization_flow")
    if valid_metric_rows != len(dict_rows):
        errors.append("metrics")
    if valid_sample_rows != len(dict_rows):
        errors.append("samples")
    passed = not errors and len(dict_rows) == len(expected_pairs)
    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "report": str(report),
        "rows": len(dict_rows),
        "expected_rows": len(expected_pairs),
        "required_fields": sorted(ARTIFACT_IMAGENET_REQUIRED_FIELDS),
        "present_pairs": [{"model": model, "precision": precision} for model, precision in present_pairs],
        "missing_pairs": [{"model": model, "precision": precision} for model, precision in missing_pairs],
        "unexpected_pairs": [{"model": model, "precision": precision} for model, precision in unexpected_pairs],
        "rows_with_required_fields": rows_with_required_fields,
        "paper_equivalent_rows": paper_equivalent_rows,
        "artifact_rows": artifact_rows,
        "quant_flow_rows": quant_flow_rows,
        "valid_metric_rows": valid_metric_rows,
        "valid_sample_rows": valid_sample_rows,
        "errors": sorted(set(errors)),
    }

def write_artifact_imagenet_validation_json(payload: dict[str, object], path: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

def write_artifact_imagenet_validation_markdown(payload: dict[str, object], path: Any) -> None:
    path = Path(path)
    lines = [
        "# Artifact ImageNet Paper-Equivalence Validation",
        "",
        "## Summary",
        "",
        "- Status: {}".format(payload["status"]),
        "- Passed: {}".format(payload["passed"]),
        "- Rows: {}/{}".format(payload["rows"], payload["expected_rows"]),
        "- Paper-equivalent rows: {}".format(payload["paper_equivalent_rows"]),
        "- Artifact graph rows: {}".format(payload["artifact_rows"]),
        "- Quant-flow rows: {}".format(payload["quant_flow_rows"]),
        "- Valid metric rows: {}".format(payload["valid_metric_rows"]),
        "- Valid sample rows: {}".format(payload["valid_sample_rows"]),
        "",
        "## Missing Pairs",
        "",
    ]
    missing_pairs = payload.get("missing_pairs", [])
    if missing_pairs:
        for row in missing_pairs:
            lines.append("- {} {}".format(row["model"], row["precision"]))
    else:
        lines.append("- none")
    lines.extend(["", "## Unexpected Pairs", ""])
    unexpected_pairs = payload.get("unexpected_pairs", [])
    if unexpected_pairs:
        for row in unexpected_pairs:
            lines.append("- {} {}".format(row["model"], row["precision"]))
    else:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    errors = payload.get("errors", [])
    if errors:
        for error in errors:
            lines.append("- {}".format(error))
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10))

def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    ignored = {".venv", "__pycache__", "reports"}
    return [path for path in root.rglob("*") if path.is_file() and not any(part in ignored for part in path.parts)]

def _relative(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)

def _full_matrix_ready(report: Path) -> bool:
    if not report.exists():
        return False
    try:
        rows = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(rows, list):
        return False
    models = {"deit_tiny_patch16_224", "deit_small_patch16_224", "vit_tiny_patch16_224"}
    precisions = {"int8", "int4", "w4a8"}
    expected = {(model, precision) for model in models for precision in precisions}
    observed = {(row.get("model"), row.get("precision")) for row in rows if isinstance(row, dict)}
    return len(rows) == len(expected) and observed == expected and all(isinstance(row, dict) and row.get("paper_equivalent") is True for row in rows)

def scan_paper_equivalence_assets(source_root: Any, project_root: Any = Path(".")) -> dict[str, object]:
    source_root = Path(source_root)
    project_root = Path(project_root)
    roots = [source_root, project_root]
    candidates = []
    seen = set()
    for root in roots:
        for path in _iter_files(root):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
    rows = []
    for requirement in REQUIREMENTS:
        matches = [_relative(path, roots) for path in candidates if any(pattern.lower() in str(path).lower() for pattern in requirement["patterns"])]
        if requirement["name"] == "full_artifact_imagenet_accuracy_matrix":
            status = "present" if _full_matrix_ready(project_root / "reports" / "artifact_imagenet_accuracy.json") else "missing"
        else:
            status = "present" if matches else "missing"
        rows.append({**requirement, "status": status, "matches": matches[:50], "match_count": len(matches)})
    missing = sum(1 for row in rows if row["status"] == "missing")
    present = sum(1 for row in rows if row["status"] == "present")
    return {"source_root": str(source_root), "project_root": str(project_root), "status": "complete" if missing == 0 else "incomplete", "present": present, "missing": missing, "total": len(rows), "paper_equivalent_ready": missing == 0, "requirements": rows}

def write_paper_equivalence_assets_json(payload: dict[str, object], path: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

def write_paper_equivalence_assets_markdown(payload: dict[str, object], path: Any) -> None:
    path = Path(path)
    lines = ["# HG-PIPE Paper-Equivalence Asset Preflight", "", "## Summary", "", "- Status: {}".format(payload["status"]), "- Paper-equivalent ready: {}".format(payload["paper_equivalent_ready"]), "- Present: {}/{}".format(payload["present"], payload["total"]), "- Missing: {}".format(payload["missing"]), "", "## Requirements", ""]
    for row in payload["requirements"]:
        lines.append("- {}: status={}, match_count={}, required_for={}".format(row["name"], row["status"], row["match_count"], row["required_for"]))
    lines.extend(["", "## Missing Items", ""])
    for row in payload["requirements"]:
        if row["status"] == "missing":
            lines.append("- {}: {}".format(row["name"], row["description"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10))
