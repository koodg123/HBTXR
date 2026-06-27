"""Artifact-backed evaluation from already-quantized HG-PIPE patch inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .input_bridge import patch_input_contract

VALID_QUANT_FLOWS = {"torch_int", "fakequant_graph"}


def iter_patch_inputs_from_npy(array: np.ndarray):
    """Yield flattened patch-input tensors from one vector or a batch array."""

    arr = np.asarray(array, dtype=np.int64)
    if arr.ndim == 1:
        yield arr.reshape(-1)
        return
    if arr.ndim < 2:
        raise ValueError("expected 1D patch input or batched patch inputs, got shape {}".format(arr.shape))
    for sample in arr:
        yield np.asarray(sample, dtype=np.int64).reshape(-1)


def _topk_hit(topk_entries, label: int, k: int) -> bool:
    return any(int(entry.index) == int(label) for entry in topk_entries[:k])


def _select_result(result, quantization_flow: str):
    if quantization_flow == "torch_int":
        return result.torch_int
    if quantization_flow == "fakequant_graph":
        return result.fakequant_graph
    raise ValueError("unsupported quantization_flow: {}".format(quantization_flow))


def evaluate_artifact_patch_batch_npy(
    source,
    *,
    patch_inputs_npy: str | Path,
    labels_npy: str | Path,
    model: str,
    precision: str,
    quantization_flow: str = "torch_int",
    paper_equivalent: bool = False,
    device: str | None = None,
    topk: int = 5,
) -> dict[str, object]:
    if quantization_flow not in VALID_QUANT_FLOWS:
        raise ValueError("quantization_flow must be one of {}".format(sorted(VALID_QUANT_FLOWS)))
    patch_inputs_npy = Path(patch_inputs_npy)
    labels_npy = Path(labels_npy)
    patch_array = np.load(patch_inputs_npy)
    labels_array = np.asarray(np.load(labels_npy))
    labels = labels_array.reshape(-1)
    patch_inputs = list(iter_patch_inputs_from_npy(patch_array))
    if len(patch_inputs) != len(labels):
        raise ValueError("patch input count {} does not match label count {}".format(len(patch_inputs), len(labels)))

    from .api import HgPipeQuantizationPackage

    package = HgPipeQuantizationPackage(source, device=device)
    samples: list[dict[str, object]] = []
    top1_hits = 0
    top5_hits = 0
    comparison_mismatches = 0
    for index, (patch_input, label) in enumerate(zip(patch_inputs, labels)):
        result = package.compare_graph_runners(input_values=patch_input, topk=max(topk, 5))
        selected = _select_result(result, quantization_flow)
        top1_hit = _topk_hit(selected.topk, int(label), 1)
        top5_hit = _topk_hit(selected.topk, int(label), 5)
        top1_hits += int(top1_hit)
        top5_hits += int(top5_hit)
        comparison_mismatches += int(result.comparison.mismatches)
        samples.append(
            {
                "index": index,
                "label": int(label),
                "torch_int_top1": result.torch_int.topk[0].index if result.torch_int.topk else None,
                "fakequant_top1": result.fakequant_graph.topk[0].index if result.fakequant_graph.topk else None,
                "selected_top1": selected.topk[0].index if selected.topk else None,
                "top1_hit": top1_hit,
                "top5_hit": top5_hit,
                "comparison_passed": result.comparison.passed,
                "comparison_mismatches": result.comparison.mismatches,
            }
        )

    count = len(patch_inputs)
    return {
        "model": model,
        "precision": precision,
        "samples": count,
        "top1": (100.0 * top1_hits / count) if count else 0.0,
        "top5": (100.0 * top5_hits / count) if count else 0.0,
        "evaluation_mode": "hgpipe_artifact_graph",
        "quantization_flow": quantization_flow,
        "paper_equivalent": bool(paper_equivalent),
        "device": device or "default",
        "patch_inputs_npy": str(patch_inputs_npy),
        "labels_npy": str(labels_npy),
        "patch_input_shape": list(patch_array.shape),
        "labels_shape": list(labels_array.shape),
        "contract": patch_input_contract(),
        "runner_comparison_passed": comparison_mismatches == 0,
        "runner_comparison_mismatches": comparison_mismatches,
        "samples_detail": samples,
        "provenance_note": "Already-quantized HG-PIPE patch inputs; paper_equivalent reflects the caller assertion from --paper-equivalent-inputs.",
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("existing report is not a list: {}".format(path))
    return [row for row in data if isinstance(row, dict)]


def write_artifact_patch_batch_report(payload: dict[str, object], path: str | Path, *, append: bool = False) -> None:
    path = Path(path)
    rows = _read_rows(path) if append else []
    key = (payload.get("model"), payload.get("precision"))
    rows = [row for row in rows if (row.get("model"), row.get("precision")) != key]
    rows.append(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True))


def _resolve_manifest_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_artifact_patch_matrix_manifest(manifest: str | Path) -> list[dict[str, Any]]:
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("artifact patch matrix manifest must be a list or an object with an entries list")
    rows: list[dict[str, Any]] = []
    required = {"model", "precision", "patch_inputs_npy", "labels_npy"}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("manifest entry {} is not an object".format(index))
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError("manifest entry {} missing required fields: {}".format(index, missing))
        quantization_flow = entry.get("quantization_flow", "torch_int")
        if quantization_flow not in VALID_QUANT_FLOWS:
            raise ValueError("manifest entry {} has unsupported quantization_flow: {}".format(index, quantization_flow))
        rows.append(
            {
                "model": str(entry["model"]),
                "precision": str(entry["precision"]),
                "patch_inputs_npy": str(_resolve_manifest_path(entry["patch_inputs_npy"], manifest.parent)),
                "labels_npy": str(_resolve_manifest_path(entry["labels_npy"], manifest.parent)),
                "quantization_flow": quantization_flow,
                "paper_equivalent": bool(entry.get("paper_equivalent", False)),
            }
        )
    return rows


def expected_artifact_patch_matrix_manifest() -> dict[str, object]:
    models = ["deit_tiny_patch16_224", "deit_small_patch16_224", "vit_tiny_patch16_224"]
    precisions = ["int8", "int4", "w4a8"]
    entries = []
    for model in models:
        for precision in precisions:
            stem = "{}_{}".format(model, precision)
            entries.append(
                {
                    "model": model,
                    "precision": precision,
                    "patch_inputs_npy": "patch_inputs/{}_patch_inputs.npy".format(stem),
                    "labels_npy": "patch_inputs/{}_labels.npy".format(stem),
                    "quantization_flow": "torch_int",
                    "paper_equivalent": True,
                }
            )
    return {
        "schema": "hgpipe_artifact_patch_matrix_v1",
        "description": "Fill these paths with already-quantized HG-PIPE patch input batches. paper_equivalent=true is a caller assertion that the tensors came from the original or accepted HG-PIPE preprocessing/calibration flow.",
        "entries": entries,
    }


def write_artifact_patch_matrix_manifest_template(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected_artifact_patch_matrix_manifest(), indent=2, sort_keys=True))


def evaluate_artifact_patch_matrix_manifest(
    source,
    *,
    manifest: str | Path,
    paper_equivalent_inputs: bool = False,
    device: str | None = None,
    topk: int = 5,
) -> list[dict[str, object]]:
    manifest = Path(manifest)
    entries = load_artifact_patch_matrix_manifest(manifest)
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        row = evaluate_artifact_patch_batch_npy(
            source,
            patch_inputs_npy=entry["patch_inputs_npy"],
            labels_npy=entry["labels_npy"],
            model=entry["model"],
            precision=entry["precision"],
            quantization_flow=entry["quantization_flow"],
            paper_equivalent=bool(paper_equivalent_inputs or entry["paper_equivalent"]),
            device=device,
            topk=topk,
        )
        row["manifest"] = str(manifest)
        row["manifest_index"] = index
        rows.append(row)
    return rows


def write_artifact_patch_matrix_report(rows: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True))


def _expected_matrix_pairs() -> set[tuple[str, str]]:
    manifest = expected_artifact_patch_matrix_manifest()
    return {(entry["model"], entry["precision"]) for entry in manifest["entries"]}


def validate_artifact_patch_matrix_manifest(manifest: str | Path) -> dict[str, object]:
    manifest = Path(manifest)
    expected_pairs = _expected_matrix_pairs()
    errors: list[str] = []
    schema = None
    try:
        raw_manifest = json.loads(manifest.read_text())
        schema = raw_manifest.get("schema") if isinstance(raw_manifest, dict) else "unversioned_list"
    except Exception:
        schema = None
    try:
        entries = load_artifact_patch_matrix_manifest(manifest)
    except Exception as exc:
        return {
            "passed": False,
            "status": "failed",
            "schema": schema,
            "manifest": str(manifest),
            "rows": 0,
            "expected_rows": len(expected_pairs),
            "present_pairs": [],
            "missing_pairs": [{"model": model, "precision": precision} for model, precision in sorted(expected_pairs)],
            "unexpected_pairs": [],
            "duplicate_pairs": [],
            "paper_equivalent_rows": 0,
            "valid_flow_rows": 0,
            "existing_patch_input_files": 0,
            "existing_label_files": 0,
            "missing_patch_input_files": [],
            "missing_label_files": [],
            "errors": ["manifest_unreadable:{}".format(type(exc).__name__)],
        }

    observed_pairs = [(entry["model"], entry["precision"]) for entry in entries]
    observed_set = set(observed_pairs)
    duplicate_pairs = sorted({pair for pair in observed_pairs if observed_pairs.count(pair) > 1})
    present_pairs = sorted(pair for pair in observed_set if pair in expected_pairs)
    missing_pairs = sorted(expected_pairs - observed_set)
    unexpected_pairs = sorted(pair for pair in observed_set if pair not in expected_pairs)
    paper_equivalent_rows = sum(1 for entry in entries if entry.get("paper_equivalent") is True)
    valid_flow_rows = sum(1 for entry in entries if entry.get("quantization_flow") in VALID_QUANT_FLOWS)
    missing_patch_input_files = [entry["patch_inputs_npy"] for entry in entries if not Path(entry["patch_inputs_npy"]).exists()]
    missing_label_files = [entry["labels_npy"] for entry in entries if not Path(entry["labels_npy"]).exists()]
    existing_patch_input_files = len(entries) - len(missing_patch_input_files)
    existing_label_files = len(entries) - len(missing_label_files)
    if schema not in {None, "unversioned_list", "hgpipe_artifact_patch_matrix_v1"}:
        errors.append("schema")
    if len(entries) != len(expected_pairs):
        errors.append("row_count")
    if missing_pairs:
        errors.append("missing_pairs")
    if unexpected_pairs:
        errors.append("unexpected_pairs")
    if duplicate_pairs:
        errors.append("duplicate_pairs")
    if paper_equivalent_rows != len(entries):
        errors.append("paper_equivalent")
    if valid_flow_rows != len(entries):
        errors.append("quantization_flow")
    if existing_patch_input_files != len(entries):
        errors.append("patch_input_files")
    if existing_label_files != len(entries):
        errors.append("label_files")
    passed = not errors and len(entries) == len(expected_pairs)
    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "schema": schema,
        "manifest": str(manifest),
        "rows": len(entries),
        "expected_rows": len(expected_pairs),
        "present_pairs": [{"model": model, "precision": precision} for model, precision in present_pairs],
        "missing_pairs": [{"model": model, "precision": precision} for model, precision in missing_pairs],
        "unexpected_pairs": [{"model": model, "precision": precision} for model, precision in unexpected_pairs],
        "duplicate_pairs": [{"model": model, "precision": precision} for model, precision in duplicate_pairs],
        "paper_equivalent_rows": paper_equivalent_rows,
        "valid_flow_rows": valid_flow_rows,
        "existing_patch_input_files": existing_patch_input_files,
        "existing_label_files": existing_label_files,
        "missing_patch_input_files": missing_patch_input_files,
        "missing_label_files": missing_label_files,
        "errors": sorted(set(errors)),
    }


def write_artifact_patch_matrix_manifest_validation_json(payload: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_artifact_patch_matrix_manifest_validation_markdown(payload: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    lines = [
        "# Artifact Patch Matrix Manifest Preflight",
        "",
        "## Summary",
        "",
        "- Status: {}".format(payload["status"]),
        "- Passed: {}".format(payload["passed"]),
        "- Rows: {}/{}".format(payload["rows"], payload["expected_rows"]),
        "- Paper-equivalent rows: {}".format(payload["paper_equivalent_rows"]),
        "- Valid-flow rows: {}".format(payload["valid_flow_rows"]),
        "- Existing patch input files: {}".format(payload["existing_patch_input_files"]),
        "- Existing label files: {}".format(payload["existing_label_files"]),
        "",
        "## Missing Files",
        "",
    ]
    missing_patch_files = payload.get("missing_patch_input_files", [])
    missing_label_files = payload.get("missing_label_files", [])
    if missing_patch_files:
        lines.append("Patch input files:")
        for file_name in missing_patch_files:
            lines.append("- {}".format(file_name))
    else:
        lines.append("Patch input files: none")
    if missing_label_files:
        lines.append("Label files:")
        for file_name in missing_label_files:
            lines.append("- {}".format(file_name))
    else:
        lines.append("Label files: none")
    lines.extend([
        "",
        "## Missing Pairs",
        "",
    ])
    missing_pairs = payload.get("missing_pairs", [])
    if missing_pairs:
        for row in missing_pairs:
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

# Backward-compatible override: strengthen manifest preflight with read-only .npy integrity checks.
from pathlib import Path as _ArtifactPatchPath
import numpy as np

_ARTIFACT_PATCH_INPUT_WIDTH = 196 * 768


def _artifact_patch_npy_metadata(path):
    metadata = {
        "loadable": False,
        "error": None,
        "shape": None,
        "dtype": None,
        "samples": None,
        "valid_shape": False,
        "integer_dtype": False,
    }
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        metadata["error"] = "{}:{}".format(type(exc).__name__, exc)
        return metadata
    metadata["loadable"] = True
    metadata["shape"] = [int(dim) for dim in array.shape]
    metadata["dtype"] = str(array.dtype)
    metadata["integer_dtype"] = bool(np.issubdtype(array.dtype, np.integer))
    if array.ndim == 1 and int(array.size) == _ARTIFACT_PATCH_INPUT_WIDTH:
        metadata["valid_shape"] = True
        metadata["samples"] = 1
    elif array.ndim >= 2 and int(np.prod(array.shape[1:], dtype=np.int64)) == _ARTIFACT_PATCH_INPUT_WIDTH:
        metadata["valid_shape"] = True
        metadata["samples"] = int(array.shape[0])
    return metadata


def _artifact_label_npy_metadata(path):
    metadata = {
        "loadable": False,
        "error": None,
        "shape": None,
        "dtype": None,
        "samples": None,
        "valid_shape": False,
        "integer_dtype": False,
    }
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        metadata["error"] = "{}:{}".format(type(exc).__name__, exc)
        return metadata
    metadata["loadable"] = True
    metadata["shape"] = [int(dim) for dim in array.shape]
    metadata["dtype"] = str(array.dtype)
    metadata["integer_dtype"] = bool(np.issubdtype(array.dtype, np.integer))
    try:
        metadata["samples"] = int(array.reshape(-1).shape[0])
        metadata["valid_shape"] = True
    except Exception as exc:
        metadata["error"] = "{}:{}".format(type(exc).__name__, exc)
    return metadata


def validate_artifact_patch_matrix_manifest(manifest_path: str | Path) -> dict[str, object]:
    manifest = _ArtifactPatchPath(manifest_path)
    expected_entries = expected_artifact_patch_matrix_manifest()["entries"]
    expected_pairs = {(entry["model"], entry["precision"]) for entry in expected_entries}
    errors: list[str] = []
    schema = None
    try:
        raw_manifest = json.loads(manifest.read_text())
        schema = raw_manifest.get("schema") if isinstance(raw_manifest, dict) else "unversioned_list"
    except Exception:
        schema = None
    try:
        entries = load_artifact_patch_matrix_manifest(manifest)
    except Exception as exc:
        return {
            "passed": False,
            "status": "failed",
            "schema": schema,
            "manifest": str(manifest),
            "rows": 0,
            "expected_rows": len(expected_pairs),
            "present_pairs": [],
            "missing_pairs": [{"model": model, "precision": precision} for model, precision in sorted(expected_pairs)],
            "unexpected_pairs": [],
            "duplicate_pairs": [],
            "paper_equivalent_rows": 0,
            "valid_flow_rows": 0,
            "existing_patch_input_files": 0,
            "existing_label_files": 0,
            "loadable_patch_input_files": 0,
            "loadable_label_files": 0,
            "valid_patch_shape_rows": 0,
            "valid_label_shape_rows": 0,
            "integer_patch_dtype_rows": 0,
            "integer_label_dtype_rows": 0,
            "matching_sample_count_rows": 0,
            "patch_input_width": _ARTIFACT_PATCH_INPUT_WIDTH,
            "missing_patch_input_files": [],
            "missing_label_files": [],
            "invalid_entries": [],
            "errors": ["manifest_unreadable:{}".format(type(exc).__name__)],
        }

    observed_pairs = [(entry["model"], entry["precision"]) for entry in entries]
    observed_set = set(observed_pairs)
    duplicate_pairs = sorted({pair for pair in observed_pairs if observed_pairs.count(pair) > 1})
    present_pairs = sorted(pair for pair in observed_set if pair in expected_pairs)
    missing_pairs = sorted(expected_pairs - observed_set)
    unexpected_pairs = sorted(pair for pair in observed_set if pair not in expected_pairs)
    paper_equivalent_rows = sum(1 for entry in entries if entry.get("paper_equivalent") is True)
    valid_flow_rows = sum(1 for entry in entries if entry.get("quantization_flow") in VALID_QUANT_FLOWS)

    rows = len(entries)
    existing_patch_input_files = 0
    existing_label_files = 0
    loadable_patch_input_files = 0
    loadable_label_files = 0
    valid_patch_shape_rows = 0
    valid_label_shape_rows = 0
    integer_patch_dtype_rows = 0
    integer_label_dtype_rows = 0
    matching_sample_count_rows = 0
    missing_patch_input_files: list[str] = []
    missing_label_files: list[str] = []
    invalid_entries: list[dict[str, object]] = []

    for index, entry in enumerate(entries):
        patch_path = _ArtifactPatchPath(entry["patch_inputs_npy"])
        label_path = _ArtifactPatchPath(entry["labels_npy"])
        if not patch_path.is_absolute():
            patch_path = manifest.parent / patch_path
        if not label_path.is_absolute():
            label_path = manifest.parent / label_path

        row_errors: list[str] = []
        if patch_path.exists():
            existing_patch_input_files += 1
            patch_meta = _artifact_patch_npy_metadata(patch_path)
        else:
            patch_meta = {
                "loadable": False,
                "error": "missing",
                "shape": None,
                "dtype": None,
                "samples": None,
                "valid_shape": False,
                "integer_dtype": False,
            }
            row_errors.append("patch_input_missing")
            missing_patch_input_files.append(str(patch_path))

        if label_path.exists():
            existing_label_files += 1
            label_meta = _artifact_label_npy_metadata(label_path)
        else:
            label_meta = {
                "loadable": False,
                "error": "missing",
                "shape": None,
                "dtype": None,
                "samples": None,
                "valid_shape": False,
                "integer_dtype": False,
            }
            row_errors.append("label_missing")
            missing_label_files.append(str(label_path))

        if patch_meta["loadable"]:
            loadable_patch_input_files += 1
        elif "patch_input_missing" not in row_errors:
            row_errors.append("patch_input_load")
        if label_meta["loadable"]:
            loadable_label_files += 1
        elif "label_missing" not in row_errors:
            row_errors.append("label_load")

        if patch_meta["valid_shape"]:
            valid_patch_shape_rows += 1
        else:
            row_errors.append("patch_input_shape")
        if label_meta["valid_shape"]:
            valid_label_shape_rows += 1
        else:
            row_errors.append("label_shape")

        if patch_meta["integer_dtype"]:
            integer_patch_dtype_rows += 1
        else:
            row_errors.append("patch_input_dtype")
        if label_meta["integer_dtype"]:
            integer_label_dtype_rows += 1
        else:
            row_errors.append("label_dtype")

        if patch_meta["samples"] is not None and patch_meta["samples"] == label_meta["samples"]:
            matching_sample_count_rows += 1
        else:
            row_errors.append("sample_count")

        if row_errors:
            invalid_entries.append(
                {
                    "manifest_index": index,
                    "model": entry["model"],
                    "precision": entry["precision"],
                    "patch_inputs_npy": str(patch_path),
                    "labels_npy": str(label_path),
                    "patch_input": patch_meta,
                    "labels": label_meta,
                    "errors": row_errors,
                }
            )

    if schema not in {None, "unversioned_list", "hgpipe_artifact_patch_matrix_v1"}:
        errors.append("schema")
    if rows != len(expected_pairs):
        errors.append("row_count")
    if missing_pairs:
        errors.append("missing_pairs")
    if unexpected_pairs:
        errors.append("unexpected_pairs")
    if duplicate_pairs:
        errors.append("duplicate_pairs")
    if paper_equivalent_rows != rows:
        errors.append("paper_equivalent")
    if valid_flow_rows != rows:
        errors.append("quantization_flow")
    if existing_patch_input_files != rows:
        errors.append("patch_input_files")
    if existing_label_files != rows:
        errors.append("label_files")
    if loadable_patch_input_files != rows:
        errors.append("loadable_patch_input_files")
    if loadable_label_files != rows:
        errors.append("loadable_label_files")
    if valid_patch_shape_rows != rows:
        errors.append("valid_patch_shape_rows")
    if valid_label_shape_rows != rows:
        errors.append("valid_label_shape_rows")
    if integer_patch_dtype_rows != rows:
        errors.append("integer_patch_dtype_rows")
    if integer_label_dtype_rows != rows:
        errors.append("integer_label_dtype_rows")
    if matching_sample_count_rows != rows:
        errors.append("matching_sample_count_rows")

    passed = not errors
    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "schema": schema,
        "manifest": str(manifest),
        "rows": rows,
        "expected_rows": len(expected_pairs),
        "present_pairs": [{"model": model, "precision": precision} for model, precision in present_pairs],
        "missing_pairs": [{"model": model, "precision": precision} for model, precision in missing_pairs],
        "unexpected_pairs": [{"model": model, "precision": precision} for model, precision in unexpected_pairs],
        "duplicate_pairs": [{"model": model, "precision": precision} for model, precision in duplicate_pairs],
        "paper_equivalent_rows": paper_equivalent_rows,
        "valid_flow_rows": valid_flow_rows,
        "existing_patch_input_files": existing_patch_input_files,
        "existing_label_files": existing_label_files,
        "loadable_patch_input_files": loadable_patch_input_files,
        "loadable_label_files": loadable_label_files,
        "valid_patch_shape_rows": valid_patch_shape_rows,
        "valid_label_shape_rows": valid_label_shape_rows,
        "integer_patch_dtype_rows": integer_patch_dtype_rows,
        "integer_label_dtype_rows": integer_label_dtype_rows,
        "matching_sample_count_rows": matching_sample_count_rows,
        "patch_input_width": _ARTIFACT_PATCH_INPUT_WIDTH,
        "missing_patch_input_files": missing_patch_input_files,
        "missing_label_files": missing_label_files,
        "invalid_entries": invalid_entries,
        "errors": sorted(set(errors)),
    }


def write_artifact_patch_matrix_manifest_validation_markdown(report: dict[str, object], output_path: str | Path) -> None:
    output_path = _ArtifactPatchPath(output_path)
    invalid_entries = report.get("invalid_entries", [])[:50]
    lines = [
        "# Artifact Patch Matrix Manifest Preflight",
        "",
        "## Summary",
        "",
        "- Status: {}".format(report["status"]),
        "- Passed: {}".format(report["passed"]),
        "- Rows: {}/{}".format(report["rows"], report["expected_rows"]),
        "- Paper-equivalent rows: {}".format(report["paper_equivalent_rows"]),
        "- Valid-flow rows: {}".format(report["valid_flow_rows"]),
        "- Existing patch input files: {}".format(report["existing_patch_input_files"]),
        "- Existing label files: {}".format(report["existing_label_files"]),
        "- Loadable patch input files: {}".format(report.get("loadable_patch_input_files", 0)),
        "- Loadable label files: {}".format(report.get("loadable_label_files", 0)),
        "- Valid patch-shape rows: {}".format(report.get("valid_patch_shape_rows", 0)),
        "- Valid label-shape rows: {}".format(report.get("valid_label_shape_rows", 0)),
        "- Integer patch dtype rows: {}".format(report.get("integer_patch_dtype_rows", 0)),
        "- Integer label dtype rows: {}".format(report.get("integer_label_dtype_rows", 0)),
        "- Matching sample-count rows: {}".format(report.get("matching_sample_count_rows", 0)),
        "- Patch input width: {}".format(report.get("patch_input_width", _ARTIFACT_PATCH_INPUT_WIDTH)),
        "",
        "## Missing Files",
        "",
    ]
    missing_patch_files = report.get("missing_patch_input_files", [])
    missing_label_files = report.get("missing_label_files", [])
    if missing_patch_files:
        lines.append("Patch input files:")
        for file_name in missing_patch_files:
            lines.append("- {}".format(file_name))
    else:
        lines.append("Patch input files: none")
    if missing_label_files:
        lines.append("Label files:")
        for file_name in missing_label_files:
            lines.append("- {}".format(file_name))
    else:
        lines.append("Label files: none")
    lines.extend([
        "",
        "## Invalid Entries",
        "",
    ])
    if invalid_entries:
        for row in invalid_entries:
            lines.append(
                "- [{}] {} {}: {}".format(
                    row.get("manifest_index"),
                    row.get("model"),
                    row.get("precision"),
                    ", ".join(row.get("errors", [])),
                )
            )
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Missing Pairs",
        "",
    ])
    missing_pairs = report.get("missing_pairs", [])
    if missing_pairs:
        for row in missing_pairs:
            lines.append("- {} {}".format(row["model"], row["precision"]))
    else:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    if report.get("errors"):
        for error in report["errors"]:
            lines.append("- {}".format(error))
    else:
        lines.append("- none")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(chr(10).join(lines) + chr(10))

# Canonical artifact patch matrix ingest/readiness gate.
import shutil as _artifact_patch_shutil


def load_artifact_patch_asset_source_manifest(source_manifest: str | Path) -> list[dict[str, object]]:
    source_manifest = _ArtifactPatchPath(source_manifest)
    data = json.loads(source_manifest.read_text())
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("artifact patch asset source manifest must be a list or an object with an entries list")
    rows: list[dict[str, object]] = []
    required = {"model", "precision", "patch_inputs_npy", "labels_npy"}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("source manifest entry {} is not an object".format(index))
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError("source manifest entry {} missing required fields: {}".format(index, missing))
        quantization_flow = entry.get("quantization_flow", "torch_int")
        if quantization_flow not in VALID_QUANT_FLOWS:
            raise ValueError("source manifest entry {} has unsupported quantization_flow: {}".format(index, quantization_flow))
        rows.append(
            {
                "model": str(entry["model"]),
                "precision": str(entry["precision"]),
                "patch_inputs_npy": str(_resolve_manifest_path(entry["patch_inputs_npy"], source_manifest.parent)),
                "labels_npy": str(_resolve_manifest_path(entry["labels_npy"], source_manifest.parent)),
                "paper_equivalent": bool(entry.get("paper_equivalent", False)),
                "quantization_flow": quantization_flow,
            }
        )
    return rows


def ingest_artifact_patch_matrix_assets(
    source_manifest: str | Path,
    template_manifest: str | Path = "configs/artifact_patch_matrix_manifest.template.json",
    output_manifest: str | Path = "configs/artifact_patch_matrix_manifest.json",
    copy: bool = True,
    assert_paper_equivalent: bool = False,
    report_json: str | Path | None = None,
    report_markdown: str | Path | None = None,
) -> dict[str, object]:
    source_manifest = _ArtifactPatchPath(source_manifest)
    template_manifest = _ArtifactPatchPath(template_manifest)
    output_manifest = _ArtifactPatchPath(output_manifest)

    source_rows = load_artifact_patch_asset_source_manifest(source_manifest)
    template_payload = json.loads(template_manifest.read_text())
    template_entries_raw = template_payload.get("entries") if isinstance(template_payload, dict) else template_payload
    if not isinstance(template_entries_raw, list):
        raise ValueError("template manifest must be a list or an object with an entries list")
    template_rows = load_artifact_patch_matrix_manifest(template_manifest)

    source_map: dict[tuple[str, str], dict[str, object]] = {}
    source_pairs: list[tuple[str, str]] = []
    for row in source_rows:
        pair = (str(row["model"]), str(row["precision"]))
        source_pairs.append(pair)
        source_map[pair] = row

    template_pairs = {(str(row["model"]), str(row["precision"])) for row in template_rows}
    matched_rows = 0
    copied_files = 0
    output_entries: list[dict[str, object]] = []

    for raw_entry, template_row in zip(template_entries_raw, template_rows):
        pair = (str(template_row["model"]), str(template_row["precision"]))
        source_row = source_map.get(pair)
        if source_row is None:
            continue
        matched_rows += 1
        patch_destination = _ArtifactPatchPath(template_row["patch_inputs_npy"])
        label_destination = _ArtifactPatchPath(template_row["labels_npy"])
        if copy:
            patch_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            _artifact_patch_shutil.copy2(str(source_row["patch_inputs_npy"]), str(patch_destination))
            _artifact_patch_shutil.copy2(str(source_row["labels_npy"]), str(label_destination))
            copied_files += 2
        output_entries.append(
            {
                "model": str(template_row["model"]),
                "precision": str(template_row["precision"]),
                "patch_inputs_npy": str(raw_entry["patch_inputs_npy"]),
                "labels_npy": str(raw_entry["labels_npy"]),
                "paper_equivalent": bool(source_row.get("paper_equivalent")) or bool(assert_paper_equivalent),
                "quantization_flow": str(source_row.get("quantization_flow", "torch_int")),
            }
        )

    missing_source_pairs = [
        {"model": model, "precision": precision}
        for model, precision in sorted(template_pairs - set(source_pairs))
    ]
    unexpected_source_pairs = [
        {"model": model, "precision": precision}
        for model, precision in sorted(set(source_pairs) - template_pairs)
    ]

    output_payload = {
        "schema": template_payload.get("schema", "hgpipe_artifact_patch_matrix_v1") if isinstance(template_payload, dict) else "hgpipe_artifact_patch_matrix_v1",
        "description": template_payload.get("description") if isinstance(template_payload, dict) else None,
        "entries": output_entries,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(output_payload, indent=2, sort_keys=True))

    validation = validate_artifact_patch_matrix_manifest(output_manifest)
    if report_json is not None:
        write_artifact_patch_matrix_manifest_validation_json(validation, report_json)
    if report_markdown is not None:
        write_artifact_patch_matrix_manifest_validation_markdown(validation, report_markdown)

    return {
        "status": validation["status"],
        "source_manifest": str(source_manifest),
        "template_manifest": str(template_manifest),
        "output_manifest": str(output_manifest),
        "copied_files": copied_files,
        "matched_rows": matched_rows,
        "missing_source_pairs": missing_source_pairs,
        "unexpected_source_pairs": unexpected_source_pairs,
        "validation": validation,
    }

from .paper_equivalence import validate_artifact_imagenet_report as _artifact_patch_validate_artifact_imagenet_report
from .paper_equivalence import write_artifact_imagenet_validation_json as _artifact_patch_write_artifact_imagenet_validation_json
from .paper_equivalence import write_artifact_imagenet_validation_markdown as _artifact_patch_write_artifact_imagenet_validation_markdown


def run_artifact_patch_matrix_pipeline(
    source,
    *,
    source_manifest: str | Path | None = None,
    manifest: str | Path = "configs/artifact_patch_matrix_manifest.json",
    template_manifest: str | Path = "configs/artifact_patch_matrix_manifest.template.json",
    output_manifest: str | Path | None = None,
    copy: bool = True,
    assert_paper_equivalent: bool = False,
    matrix_report: str | Path = "reports/artifact_imagenet_accuracy.json",
    manifest_report_json: str | Path = "reports/artifact_patch_matrix_manifest_validation.json",
    manifest_report_markdown: str | Path = "reports/artifact_patch_matrix_manifest_validation.md",
    validation_report_json: str | Path = "reports/artifact_imagenet_validation.json",
    validation_report_markdown: str | Path = "reports/artifact_imagenet_validation.md",
    strict: bool = False,
) -> dict[str, object]:
    manifest = Path(manifest)
    template_manifest = Path(template_manifest)
    matrix_report = Path(matrix_report)
    manifest_report_json = Path(manifest_report_json)
    manifest_report_markdown = Path(manifest_report_markdown)
    validation_report_json = Path(validation_report_json)
    validation_report_markdown = Path(validation_report_markdown)

    ingest_payload = None
    selected_manifest = manifest
    if source_manifest is not None:
        selected_manifest = Path(output_manifest) if output_manifest is not None else manifest
        ingest_payload = ingest_artifact_patch_matrix_assets(
            source_manifest=source_manifest,
            template_manifest=template_manifest,
            output_manifest=selected_manifest,
            copy=copy,
            assert_paper_equivalent=assert_paper_equivalent,
            report_json=None,
            report_markdown=None,
        )
    elif output_manifest is not None:
        selected_manifest = Path(output_manifest)

    manifest_validation = validate_artifact_patch_matrix_manifest(selected_manifest)
    write_artifact_patch_matrix_manifest_validation_json(manifest_validation, manifest_report_json)
    write_artifact_patch_matrix_manifest_validation_markdown(manifest_validation, manifest_report_markdown)

    if not manifest_validation.get("passed"):
        skipped_reason = "manifest_validation_failed"
        return {
            "status": "failed",
            "ingest": ingest_payload,
            "manifest_validation": manifest_validation,
            "matrix_report": {"path": str(matrix_report), "rows": 0},
            "artifact_report_validation": None,
            "ran_matrix": False,
            "skipped_reason": skipped_reason,
        }

    rows = evaluate_artifact_patch_matrix_manifest(
        source,
        manifest=selected_manifest,
        paper_equivalent_inputs=assert_paper_equivalent,
    )
    write_artifact_patch_matrix_report(rows, matrix_report)
    artifact_report_validation = _artifact_patch_validate_artifact_imagenet_report(matrix_report)
    _artifact_patch_write_artifact_imagenet_validation_json(artifact_report_validation, validation_report_json)
    _artifact_patch_write_artifact_imagenet_validation_markdown(artifact_report_validation, validation_report_markdown)
    pipeline_status = "passed" if artifact_report_validation.get("passed") else "failed"
    return {
        "status": pipeline_status,
        "ingest": ingest_payload,
        "manifest_validation": manifest_validation,
        "matrix_report": {"path": str(matrix_report), "rows": len(rows)},
        "artifact_report_validation": artifact_report_validation,
        "ran_matrix": True,
        "skipped_reason": None,
    }


def write_artifact_patch_asset_source_manifest_from_directory(
    asset_dir: str | Path,
    output_manifest: str | Path,
    models: list[str] | None = None,
    precisions: list[str] | None = None,
    paper_equivalent: bool = False,
    quantization_flow: str = "torch_int",
) -> dict[str, object]:
    asset_dir = Path(asset_dir)
    output_manifest = Path(output_manifest)
    if quantization_flow not in VALID_QUANT_FLOWS:
        raise ValueError("quantization_flow must be one of {}".format(sorted(VALID_QUANT_FLOWS)))

    expected_rows = expected_artifact_patch_matrix_manifest()["entries"]
    if models is not None:
        model_set = {str(value) for value in models}
        expected_rows = [row for row in expected_rows if row["model"] in model_set]
    if precisions is not None:
        precision_set = {str(value) for value in precisions}
        expected_rows = [row for row in expected_rows if row["precision"] in precision_set]

    npy_files = [path for path in asset_dir.rglob("*.npy") if path.is_file()]

    def _match_path(model: str, precision: str, kind: str) -> Path | None:
        flat_names = {
            "patch": {"{}_{}_patch_inputs.npy".format(model, precision)},
            "label": {"{}_{}_labels.npy".format(model, precision)},
        }
        tail_names = {
            "patch": {"patch_inputs.npy", "inputs.npy"},
            "label": {"labels.npy", "targets.npy"},
        }
        for candidate in npy_files:
            rel = candidate.relative_to(asset_dir)
            if candidate.name in flat_names[kind]:
                return candidate
            if len(rel.parts) >= 3 and rel.parts[-3] == model and rel.parts[-2] == precision and rel.parts[-1] in tail_names[kind]:
                return candidate
        return None

    entries: list[dict[str, object]] = []
    found_pairs: list[dict[str, str]] = []
    missing_pairs: list[dict[str, str]] = []
    for row in expected_rows:
        patch_path = _match_path(row["model"], row["precision"], "patch")
        label_path = _match_path(row["model"], row["precision"], "label")
        if patch_path is None or label_path is None:
            missing_pairs.append({"model": row["model"], "precision": row["precision"]})
            continue
        entries.append(
            {
                "model": row["model"],
                "precision": row["precision"],
                "patch_inputs_npy": str(patch_path),
                "labels_npy": str(label_path),
                "paper_equivalent": bool(paper_equivalent),
                "quantization_flow": quantization_flow,
            }
        )
        found_pairs.append({"model": row["model"], "precision": row["precision"]})

    payload = {
        "entries": entries,
        "asset_dir": str(asset_dir),
        "found_pairs": found_pairs,
        "missing_pairs": missing_pairs,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {
        "asset_dir": str(asset_dir),
        "output_manifest": str(output_manifest),
        "rows": len(entries),
        "found_pairs": found_pairs,
        "missing_pairs": missing_pairs,
        "paper_equivalent": bool(paper_equivalent),
        "quantization_flow": quantization_flow,
    }
