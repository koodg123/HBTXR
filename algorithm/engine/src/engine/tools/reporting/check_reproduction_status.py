import argparse
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path("/home/kjm26/project/PRJXR/HBTXR")
FACET_ROOT = ROOT / "references/codebase/software/FACET"
REPORT_ROOT = ROOT / "references/report/FACET"
RAW_ROOT = Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data")
ARTIFACT_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.(?:json|md)$")
EXPECTED_COMPARISON_ROWS = {
    "P10": "higher",
    "P5": "higher",
    "P3": "higher",
    "P1": "higher",
    "mean pixel error": "lower",
    "IoU": "higher",
    "AP": "higher",
    "params M": "lower",
    "trainable params M": "lower",
    "FLOPs G": "lower",
    "latency ms": "lower",
}
EXPECTED_SUMMARY_RESULT_LABELS = {
    "EPNet_full_unet",
    "HBTXR_full_unet",
    "EPNet_fpn_dw_full_unet",
    "HBTXR_full_unet_effbs32",
}
EXPECTED_SUMMARY_COMPARISON_LABELS = {
    "EPNet_vs_HBTXR",
    "EPNet_vs_HBTXR_effbs32",
}
EXPECTED_SUMMARY_RESULT_PATH_PATTERN_BY_LABEL = {
    "EPNet_full_unet": "FACET_reproduction_results_*.json",
    "HBTXR_full_unet": "FACET_hbtxr_reproduction_results_*.json",
    "EPNet_fpn_dw_full_unet": "FACET_epnet_fpn_dw_reproduction_results_*.json",
    "HBTXR_full_unet_effbs32": "FACET_hbtxr_effbs32_reproduction_results_*.json",
}
EXPECTED_SUMMARY_COMPARISON_PATH_PATTERN_BY_LABEL = {
    "EPNet_vs_HBTXR": "FACET_epnet_vs_hbtxr_comparison_*.json",
    "EPNet_vs_HBTXR_effbs32": "FACET_epnet_vs_hbtxr_effbs32_comparison_*.json",
}
EXPECTED_EVAL_CONTEXT_BY_PATTERN = {
    "FACET_reproduction_results_*.json": {
        "config": "DavisEyeEllipse_EPNet_full_unet.yaml",
        "model_type": "EPNet",
    },
    "FACET_hbtxr_reproduction_results_*.json": {
        "config": "DavisEyeEllipse_HBTXR_full_unet.yaml",
        "model_type": "HBTXR",
    },
    "FACET_epnet_fpn_dw_reproduction_results_*.json": {
        "config": "DavisEyeEllipse_EPNet_fpn_dw_full_unet.yaml",
        "model_type": "EPNet",
    },
    "FACET_hbtxr_effbs32_reproduction_results_*.json": {
        "config": "DavisEyeEllipse_HBTXR_full_unet_effbs32.yaml",
        "model_type": "HBTXR",
    },
}
EXPECTED_COMPARISON_CONTEXT_BY_PATTERN = {
    "FACET_epnet_vs_hbtxr_comparison_*.json": {
        "left_label": "EPNet_full_unet",
        "right_label": "HBTXR_full_unet",
    },
    "FACET_epnet_vs_hbtxr_effbs32_comparison_*.json": {
        "left_label": "EPNet_full_unet",
        "right_label": "HBTXR_full_unet_effbs32",
    },
}
EXPECTED_MARKDOWN_TERMS_BY_PATTERN = {
    "FACET_reproduction_results_*.md": [
        "# FACET Reproduction Results",
        "## Evaluation Artifacts",
        "## Model Metrics",
        "## Pairwise Comparisons",
    ],
    "FACET_table2_comparison_*.md": [
        "Evaluation Result",
        "| Metric | Current | Paper Table II reference | Delta |",
        "mean pixel error",
        "FLOPs G",
        "latency ms",
    ],
    "FACET_hbtxr_reproduction_results_*.md": [
        "Evaluation Result",
        "| Metric | Current | Paper Table II reference | Delta |",
        "mean pixel error",
        "FLOPs G",
        "latency ms",
    ],
    "FACET_epnet_vs_hbtxr_comparison_*.md": [
        "Evaluation Comparison",
        "Left checkpoint:",
        "Right checkpoint:",
        "Right - left",
        "Winner",
    ],
    "FACET_epnet_fpn_dw_table2_comparison_*.md": [
        "Evaluation Result",
        "| Metric | Current | Paper Table II reference | Delta |",
        "mean pixel error",
        "FLOPs G",
        "latency ms",
    ],
    "FACET_hbtxr_effbs32_reproduction_results_*.md": [
        "Evaluation Result",
        "| Metric | Current | Paper Table II reference | Delta |",
        "mean pixel error",
        "FLOPs G",
        "latency ms",
    ],
    "FACET_epnet_vs_hbtxr_effbs32_comparison_*.md": [
        "Evaluation Comparison",
        "Left checkpoint:",
        "Right checkpoint:",
        "Right - left",
        "Winner",
    ],
}


def exists(path: Path) -> bool:
    return path.exists()


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def glob_count(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def latest_matches(pattern: str) -> list[Path]:
    return sorted(REPORT_ROOT.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)


def artifact_date(path: Path) -> str | None:
    match = ARTIFACT_DATE_RE.search(path.name)
    if not match:
        return None
    return match.group(1)


def training_complete(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    completion_patterns = [
        r"`max_epochs=70` reached",
        r"\bmax_epochs=70\b reached",
        r"Trainer\.fit stopped:.*max_epochs=70.*reached",
    ]
    return any(re.search(pattern, text) for pattern in completion_patterns)


def run_command(command: list[str], cwd: Path | None = None, timeout: int = 10):
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def status_item(name, state, evidence=None, missing=None, note=None):
    return {
        "name": name,
        "state": state,
        "evidence": evidence or [],
        "missing": missing or [],
        "note": note or "",
    }


def is_finite_number(value):
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def validate_checkpoint_path(value, label: str, issues: list[str]):
    if not value:
        issues.append(f"{label} checkpoint path is missing")
        return
    checkpoint_path = Path(value)
    if not checkpoint_path.is_file():
        issues.append(f"{label} checkpoint path does not exist: {value}")


def validate_expected_eval_context(data: dict, expected: dict, issues: list[str]):
    for key, expected_value in expected.items():
        actual_value = data.get(key)
        if actual_value != expected_value:
            issues.append(f"{key} is {actual_value!r}, expected {expected_value!r}")


def validate_expected_comparison_context(data: dict, expected: dict, issues: list[str]):
    for key, expected_value in expected.items():
        actual_value = data.get(key)
        if actual_value != expected_value:
            issues.append(f"{key} is {actual_value!r}, expected {expected_value!r}")


def validate_markdown_artifact(path: Path, required_terms: list[str]):
    if not path.exists():
        return False, ["file is missing"], {"path": str(path)}
    text = path.read_text(encoding="utf-8", errors="ignore")
    issues = []
    if not text.strip():
        issues.append("markdown artifact is empty")
    missing_terms = [term for term in required_terms if term not in text]
    if missing_terms:
        issues.append("missing markdown terms: " + ", ".join(missing_terms))
    evidence = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "required_terms": required_terms,
    }
    return not issues, issues, evidence


def validate_summary_entry_paths(
    items: list,
    expected_patterns_by_label: dict[str, str],
    entry_kind: str,
    issues: list[str],
):
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        expected_pattern = expected_patterns_by_label.get(label)
        if expected_pattern is None:
            continue
        item_path_value = item.get("path")
        if not item_path_value:
            issues.append(f"{entry_kind} {label} path is missing")
            continue
        item_path = Path(item_path_value)
        if not item_path.is_file():
            issues.append(f"{entry_kind} {label} path does not exist: {item_path_value}")
            continue
        if not item_path.match(expected_pattern):
            issues.append(
                f"{entry_kind} {label} path {item_path.name!r} does not match {expected_pattern!r}"
            )


def check_preflight(python_path: Path):
    import_check = run_command(
        [
            str(python_path),
            "-c",
            (
                "import torch, lightning, cv2, albumentations, timm, tonic, h5py; "
                "print(torch.__version__); print(torch.cuda.is_available())"
            ),
        ],
        cwd=FACET_ROOT,
        timeout=30,
    )
    nvidia = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader",
        ],
        cwd=FACET_ROOT,
        timeout=10,
    )
    evidence = [
        {"python_import_check": import_check},
        {"nvidia_smi": nvidia},
    ]
    if import_check["returncode"] != 0:
        return status_item(
            "Gate 0 preflight",
            "missing",
            evidence=evidence,
            missing=["Python runtime imports"],
        )
    cuda_available = import_check["stdout"].splitlines()[-1:] == ["True"]
    if nvidia["returncode"] != 0 or not cuda_available:
        return status_item(
            "Gate 0 preflight",
            "blocked",
            evidence=evidence,
            missing=["usable CUDA GPU / NVIDIA driver"],
            note="CPU smoke is possible, but full training and full expansion are GPU-gated.",
        )
    return status_item("Gate 0 preflight", "passed", evidence=evidence)


def check_dataset_manifests():
    dean_manifest_path = RAW_ROOT / "DeanDataset/manifest.json"
    unet_manifest_path = RAW_ROOT / "DavisWithMaskDataset_labelled_subset/manifest.json"
    full_manifest_path = RAW_ROOT / "DeanDataset_full_unet/manifest.json"
    dean = read_json(dean_manifest_path)
    unet = read_json(unet_manifest_path)
    full = read_json(full_manifest_path)

    items = []
    if dean and dean.get("num_samples") == 8911:
        items.append(
            status_item(
                "Phase 1 subset DeanDataset",
                "passed",
                evidence=[{"manifest": str(dean_manifest_path), "num_samples": dean.get("num_samples")}],
            )
        )
    else:
        items.append(
            status_item(
                "Phase 1 subset DeanDataset",
                "missing",
                missing=[str(dean_manifest_path)],
            )
        )

    if unet and unet.get("num_samples") == 9011:
        items.append(
            status_item(
                "Phase 2 U-Net labelled PNG dataset",
                "passed",
                evidence=[{"manifest": str(unet_manifest_path), "num_samples": unet.get("num_samples")}],
            )
        )
    else:
        items.append(
            status_item(
                "Phase 2 U-Net labelled PNG dataset",
                "missing",
                missing=[str(unet_manifest_path)],
            )
        )

    if full and full.get("valid_ellipse_count", 0) > 0:
        items.append(
            status_item(
                "Phase 3 full DeanDataset_full_unet",
                "passed",
                evidence=[
                    {
                        "manifest": str(full_manifest_path),
                        "valid_ellipse_count": full.get("valid_ellipse_count"),
                    }
                ],
            )
        )
    else:
        items.append(
            status_item(
                "Phase 3 full DeanDataset_full_unet",
                "missing",
                missing=[str(full_manifest_path)],
                note="Required before full EPNet reproduction.",
            )
        )

    return items


def check_reports_and_samples():
    required_reports = [
        "FACET_reproduction_plan_2026-06-25.md",
        "FACET_phase1_subset_smoke_2026-06-25.md",
        "FACET_phase2_unet_dataset_prep_2026-06-25.md",
        "FACET_phase3_full_expansion_prep_2026-06-25.md",
        "FACET_phase4_evaluation_prep_2026-06-25.md",
        "FACET_reproduction_execution_runbook_2026-06-25.md",
    ]
    missing = [name for name in required_reports if not (REPORT_ROOT / name).exists()]
    items = [
        status_item(
            "Report artifacts",
            "passed" if not missing else "missing",
            evidence=[str(REPORT_ROOT / name) for name in required_reports if (REPORT_ROOT / name).exists()],
            missing=[str(REPORT_ROOT / name) for name in missing],
        )
    ]

    sample_manifest = REPORT_ROOT / "unet_dataset_samples/manifest.json"
    sample = read_json(sample_manifest)
    if sample and sample.get("num_records") == 10:
        items.append(
            status_item(
                "U-Net labelled subset visual samples",
                "passed",
                evidence=[{"manifest": str(sample_manifest), "num_records": sample.get("num_records")}],
            )
        )
    else:
        items.append(
            status_item(
                "U-Net labelled subset visual samples",
                "missing",
                missing=[str(sample_manifest)],
            )
        )
    return items


def check_checkpoints():
    epnet_smoke = FACET_ROOT / "runs/logs/EPNet_local_train_smoke"
    unet_smoke = FACET_ROOT / "runs/logs/RGBUNet_local_train_smoke"
    unet_full = FACET_ROOT / "runs/logs/RGBUNet_local_subset"
    epnet_full = FACET_ROOT / "runs/logs/EPNet_full_unet"
    hbtxr_full = FACET_ROOT / "runs/logs/HBTXR_full_unet"
    epnet_fpn_dw_full = FACET_ROOT / "runs/logs/EPNet_fpn_dw_full_unet"
    hbtxr_effbs32_full = FACET_ROOT / "runs/logs/HBTXR_full_unet_effbs32"
    epnet_train_log = REPORT_ROOT / "EPNet_full_unet_gpu0_train_2026-06-26.log"
    hbtxr_train_log = REPORT_ROOT / "HBTXR_full_unet_gpu1_train_2026-06-26.log"
    epnet_fpn_dw_train_log = REPORT_ROOT / "EPNet_fpn_dw_full_unet_gpu0_train_2026-06-26.log"
    hbtxr_effbs32_train_log = REPORT_ROOT / "HBTXR_full_unet_effbs32_gpu1_train_2026-06-26.log"

    items = []
    epnet_smoke_count = glob_count(epnet_smoke, "version_*/checkpoints/*.ckpt")
    items.append(
        status_item(
            "Phase 1 EPNet smoke checkpoint",
            "passed" if epnet_smoke_count else "missing",
            evidence=[{"checkpoint_count": epnet_smoke_count, "root": str(epnet_smoke)}],
        )
    )
    unet_smoke_count = glob_count(unet_smoke, "version_*/checkpoints/*.ckpt")
    items.append(
        status_item(
            "Phase 2 U-Net smoke checkpoint",
            "passed" if unet_smoke_count else "missing",
            evidence=[{"checkpoint_count": unet_smoke_count, "root": str(unet_smoke)}],
        )
    )
    unet_full_count = glob_count(unet_full, "version_*/checkpoints/*.ckpt")
    items.append(
        status_item(
            "Phase 2 full U-Net checkpoint",
            "passed" if unet_full_count else "missing",
            evidence=[{"checkpoint_count": unet_full_count, "root": str(unet_full)}],
            missing=[] if unet_full_count else ["full U-Net training output"],
        )
    )
    epnet_full_count = glob_count(epnet_full, "version_*/checkpoints/*.ckpt")
    items.append(
        status_item(
            "Phase 4 full EPNet checkpoint",
            "passed" if epnet_full_count else "missing",
            evidence=[{"checkpoint_count": epnet_full_count, "root": str(epnet_full)}],
            missing=[] if epnet_full_count else ["full EPNet training output"],
        )
    )
    epnet_complete = training_complete(epnet_train_log)
    items.append(
        status_item(
            "Phase 4 full EPNet training completion",
            "passed" if epnet_complete else "missing",
            evidence=[{"log": str(epnet_train_log), "completion_marker_found": epnet_complete}],
            missing=[] if epnet_complete else ["EPNet max_epochs=70 completion log"],
            note="A checkpoint alone is not sufficient; this gate prevents intermediate epoch checkpoints from being treated as full reproduction completion.",
        )
    )
    hbtxr_full_count = glob_count(hbtxr_full, "version_*/checkpoints/*.ckpt")
    items.append(
        status_item(
            "Phase 4B full HBTXR checkpoint",
            "passed" if hbtxr_full_count else "missing",
            evidence=[{"checkpoint_count": hbtxr_full_count, "root": str(hbtxr_full)}],
            missing=[] if hbtxr_full_count else ["full HBTXR training output"],
        )
    )
    hbtxr_complete = training_complete(hbtxr_train_log)
    items.append(
        status_item(
            "Phase 4B full HBTXR training completion",
            "passed" if hbtxr_complete else "missing",
            evidence=[{"log": str(hbtxr_train_log), "completion_marker_found": hbtxr_complete}],
            missing=[] if hbtxr_complete else ["HBTXR max_epochs=70 completion log"],
            note="A checkpoint alone is not sufficient; this gate prevents intermediate epoch checkpoints from being treated as full reproduction completion.",
        )
    )
    epnet_fpn_dw_full_count = glob_count(epnet_fpn_dw_full, "version_*/checkpoints/*.ckpt")
    items.append(
        status_item(
            "Phase 4 EPNet fpn_dw ablation checkpoint",
            "passed" if epnet_fpn_dw_full_count else "missing",
            evidence=[{"checkpoint_count": epnet_fpn_dw_full_count, "root": str(epnet_fpn_dw_full)}],
            missing=[] if epnet_fpn_dw_full_count else ["EPNet fpn_dw ablation training output"],
            note="The reproduction plan calls for fpn_2d baseline plus fpn_dw ablation for paper correspondence.",
        )
    )
    epnet_fpn_dw_complete = training_complete(epnet_fpn_dw_train_log)
    items.append(
        status_item(
            "Phase 4 EPNet fpn_dw ablation completion",
            "passed" if epnet_fpn_dw_complete else "missing",
            evidence=[
                {
                    "log": str(epnet_fpn_dw_train_log),
                    "completion_marker_found": epnet_fpn_dw_complete,
                }
            ],
            missing=[] if epnet_fpn_dw_complete else ["EPNet fpn_dw max_epochs=70 completion log"],
            note="A checkpoint alone is not sufficient for the planned fpn_dw ablation.",
        )
    )
    hbtxr_effbs32_full_count = glob_count(hbtxr_effbs32_full, "version_*/checkpoints/*.ckpt")
    items.append(
        status_item(
            "Phase 4B HBTXR effective-batch-32 checkpoint",
            "passed" if hbtxr_effbs32_full_count else "missing",
            evidence=[
                {
                    "checkpoint_count": hbtxr_effbs32_full_count,
                    "root": str(hbtxr_effbs32_full),
                }
            ],
            missing=[] if hbtxr_effbs32_full_count else ["HBTXR effective-batch-32 training output"],
            note="This stricter comparison run matches EPNet's effective batch size of 32.",
        )
    )
    hbtxr_effbs32_complete = training_complete(hbtxr_effbs32_train_log)
    items.append(
        status_item(
            "Phase 4B HBTXR effective-batch-32 completion",
            "passed" if hbtxr_effbs32_complete else "missing",
            evidence=[
                {
                    "log": str(hbtxr_effbs32_train_log),
                    "completion_marker_found": hbtxr_effbs32_complete,
                }
            ],
            missing=[] if hbtxr_effbs32_complete else ["HBTXR effective-batch-32 max_epochs=70 completion log"],
            note="A checkpoint alone is not sufficient for the planned fair effective-batch comparison.",
        )
    )
    return items


def validate_eval_result_json(path: Path):
    data = read_json(path)
    if data is None:
        return False, ["file is missing or not readable"], {}

    issues = []
    metrics = data.get("metrics")
    required_metrics = [
        "val_p10_acc",
        "val_p5_acc",
        "val_p3_acc",
        "val_p1_acc",
        "val_mean_distance",
        "val_IoU",
        "val_AP",
    ]
    if not isinstance(metrics, dict):
        issues.append("metrics is missing or not an object")
    else:
        missing_metrics = [key for key in required_metrics if key not in metrics]
        if missing_metrics:
            issues.append("missing metrics: " + ", ".join(missing_metrics))
        non_numeric_metrics = [
            key for key in required_metrics if key in metrics and not is_finite_number(metrics.get(key))
        ]
        if non_numeric_metrics:
            issues.append("non-finite or non-numeric metrics: " + ", ".join(non_numeric_metrics))

    required_numeric_fields = [
        "params_m",
        "trainable_params_m",
        "flops_g",
        "latency_ms",
    ]
    non_numeric_fields = [
        key for key in required_numeric_fields if not is_finite_number(data.get(key))
    ]
    if non_numeric_fields:
        issues.append("non-finite or non-numeric fields: " + ", ".join(non_numeric_fields))

    if data.get("max_batches") not in {0, None}:
        issues.append(f"max_batches is {data.get('max_batches')}, expected 0 for full validation")

    evaluated_batches = data.get("evaluated_batches")
    if not isinstance(evaluated_batches, int) or evaluated_batches <= 0:
        issues.append(f"evaluated_batches is {evaluated_batches!r}, expected a positive integer")

    expected_root = str(RAW_ROOT / "DeanDataset_full_unet")
    if data.get("dataset_root") != expected_root:
        issues.append(f"dataset_root is {data.get('dataset_root')!r}, expected {expected_root!r}")

    validate_checkpoint_path(data.get("checkpoint"), "evaluation", issues)

    evidence = {
        "path": str(path),
        "config": data.get("config"),
        "checkpoint": data.get("checkpoint"),
        "dataset_root": data.get("dataset_root"),
        "max_batches": data.get("max_batches"),
        "evaluated_batches": evaluated_batches,
    }
    return not issues, issues, evidence


def validate_comparison_json(path: Path):
    data = read_json(path)
    if data is None:
        return False, ["file is missing or not readable"], {}

    issues = []
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        issues.append("comparison rows are missing or empty")
    else:
        row_by_metric = {
            row.get("metric"): row
            for row in rows
            if isinstance(row, dict)
        }
        missing_rows = [
            metric for metric in EXPECTED_COMPARISON_ROWS if metric not in row_by_metric
        ]
        if missing_rows:
            issues.append("missing comparison metric rows: " + ", ".join(missing_rows))
        unexpected_rows = [
            metric for metric in row_by_metric if metric not in EXPECTED_COMPARISON_ROWS
        ]
        if unexpected_rows:
            issues.append("unexpected comparison metric rows: " + ", ".join(unexpected_rows))
        for metric, expected_direction in EXPECTED_COMPARISON_ROWS.items():
            row = row_by_metric.get(metric)
            if row is None:
                continue
            if row.get("preferred_direction") != expected_direction:
                issues.append(
                    f"{metric}.preferred_direction is {row.get('preferred_direction')!r}, "
                    f"expected {expected_direction!r}"
                )
            for field in ("left", "right", "right_minus_left"):
                if not is_finite_number(row.get(field)):
                    issues.append(f"{metric}.{field} is not a finite number")
            if not row.get("winner"):
                issues.append(f"{metric}.winner is missing")

    for side in ("left", "right"):
        side_data = data.get(side)
        if not isinstance(side_data, dict):
            issues.append(f"{side} summary is missing or not an object")
            continue
        evaluated_batches = side_data.get("evaluated_batches")
        if not isinstance(evaluated_batches, int) or evaluated_batches <= 0:
            issues.append(f"{side}.evaluated_batches is {evaluated_batches!r}, expected a positive integer")
        expected_root = str(RAW_ROOT / "DeanDataset_full_unet")
        if side_data.get("dataset_root") != expected_root:
            issues.append(f"{side}.dataset_root is {side_data.get('dataset_root')!r}, expected {expected_root!r}")
        validate_checkpoint_path(side_data.get("checkpoint"), side, issues)

    evidence = {
        "path": str(path),
        "left_label": data.get("left_label"),
        "right_label": data.get("right_label"),
        "row_count": 0 if not isinstance(rows, list) else len(rows),
    }
    return not issues, issues, evidence


def validate_summary_json(path: Path):
    data = read_json(path)
    if data is None:
        return False, ["file is missing or not readable"], {}

    issues = []
    if data.get("summary_state") != "complete":
        issues.append(f"summary_state is {data.get('summary_state')!r}, expected 'complete'")
    if data.get("missing_artifacts"):
        issues.append("missing_artifacts is not empty")

    results = data.get("results")
    if not isinstance(results, list) or not results:
        issues.append("results are missing or empty")
    else:
        result_labels = {
            item.get("label")
            for item in results
            if isinstance(item, dict)
        }
        missing_result_labels = [
            label for label in EXPECTED_SUMMARY_RESULT_LABELS if label not in result_labels
        ]
        if missing_result_labels:
            issues.append("missing summary result labels: " + ", ".join(missing_result_labels))
        unexpected_result_labels = [
            label for label in result_labels if label not in EXPECTED_SUMMARY_RESULT_LABELS
        ]
        if unexpected_result_labels:
            issues.append("unexpected summary result labels: " + ", ".join(unexpected_result_labels))
        unavailable = [item.get("label", "<unknown>") for item in results if item.get("state") != "available"]
        if unavailable:
            issues.append("unavailable results: " + ", ".join(unavailable))
        validate_summary_entry_paths(
            results,
            EXPECTED_SUMMARY_RESULT_PATH_PATTERN_BY_LABEL,
            "result",
            issues,
        )

    comparisons = data.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        issues.append("comparisons are missing or empty")
    else:
        comparison_labels = {
            item.get("label")
            for item in comparisons
            if isinstance(item, dict)
        }
        missing_comparison_labels = [
            label for label in EXPECTED_SUMMARY_COMPARISON_LABELS if label not in comparison_labels
        ]
        if missing_comparison_labels:
            issues.append("missing summary comparison labels: " + ", ".join(missing_comparison_labels))
        unexpected_comparison_labels = [
            label for label in comparison_labels if label not in EXPECTED_SUMMARY_COMPARISON_LABELS
        ]
        if unexpected_comparison_labels:
            issues.append("unexpected summary comparison labels: " + ", ".join(unexpected_comparison_labels))
        unavailable = [item.get("label", "<unknown>") for item in comparisons if item.get("state") != "available"]
        if unavailable:
            issues.append("unavailable comparisons: " + ", ".join(unavailable))
        validate_summary_entry_paths(
            comparisons,
            EXPECTED_SUMMARY_COMPARISON_PATH_PATTERN_BY_LABEL,
            "comparison",
            issues,
        )

    evidence = {
        "path": str(path),
        "summary_state": data.get("summary_state"),
        "missing_artifacts": data.get("missing_artifacts"),
        "result_count": 0 if not isinstance(results, list) else len(results),
        "comparison_count": 0 if not isinstance(comparisons, list) else len(comparisons),
    }
    return not issues, issues, evidence


def check_final_evaluation():
    required_patterns = {
        "FACET_reproduction_results_*.json": {"type": "eval_json"},
        "FACET_reproduction_results_*.md": {"type": "markdown"},
        "FACET_reproduction_summary_*.json": {"type": "summary_json"},
        "FACET_table2_comparison_*.md": {"type": "markdown"},
        "FACET_hbtxr_reproduction_results_*.json": {"type": "eval_json"},
        "FACET_hbtxr_reproduction_results_*.md": {"type": "markdown"},
        "FACET_epnet_vs_hbtxr_comparison_*.json": {"type": "comparison_json"},
        "FACET_epnet_vs_hbtxr_comparison_*.md": {"type": "markdown"},
        "FACET_epnet_fpn_dw_reproduction_results_*.json": {"type": "eval_json"},
        "FACET_epnet_fpn_dw_table2_comparison_*.md": {"type": "markdown"},
        "FACET_hbtxr_effbs32_reproduction_results_*.json": {"type": "eval_json"},
        "FACET_hbtxr_effbs32_reproduction_results_*.md": {"type": "markdown"},
        "FACET_epnet_vs_hbtxr_effbs32_comparison_*.json": {"type": "comparison_json"},
        "FACET_epnet_vs_hbtxr_effbs32_comparison_*.md": {"type": "markdown"},
    }
    evidence = []
    missing = []
    invalid = []
    selected_artifact_dates = {}
    for pattern, validator_meta in required_patterns.items():
        matches = latest_matches(pattern)
        if not matches:
            missing.append(pattern)
            continue
        path = matches[0]
        selected_artifact_dates[pattern] = {
            "path": str(path),
            "date": artifact_date(path),
        }
        validator_type = validator_meta["type"]
        if validator_type == "eval_json":
            ok, issues, validation_evidence = validate_eval_result_json(path)
            expected_context = EXPECTED_EVAL_CONTEXT_BY_PATTERN.get(pattern)
            if expected_context:
                data = read_json(path) or {}
                validate_expected_eval_context(data, expected_context, issues)
                ok = ok and not issues
                validation_evidence["expected_context"] = expected_context
        elif validator_type == "comparison_json":
            ok, issues, validation_evidence = validate_comparison_json(path)
            expected_context = EXPECTED_COMPARISON_CONTEXT_BY_PATTERN.get(pattern)
            if expected_context:
                data = read_json(path) or {}
                validate_expected_comparison_context(data, expected_context, issues)
                ok = ok and not issues
                validation_evidence["expected_context"] = expected_context
        elif validator_type == "summary_json":
            ok, issues, validation_evidence = validate_summary_json(path)
        else:
            ok, issues, validation_evidence = validate_markdown_artifact(
                path,
                EXPECTED_MARKDOWN_TERMS_BY_PATTERN.get(pattern, []),
            )
        evidence.append(validation_evidence)
        if not ok:
            invalid.append({"pattern": pattern, "path": str(path), "issues": issues})
    present_dates = {
        item["date"]
        for item in selected_artifact_dates.values()
        if item["date"] is not None
    }
    missing_dates = [
        {"pattern": pattern, **item}
        for pattern, item in selected_artifact_dates.items()
        if item["date"] is None
    ]
    if missing_dates:
        invalid.append(
            {
                "pattern": "artifact_date_suffix",
                "path": "",
                "issues": ["missing artifact date suffix"],
                "artifacts": missing_dates,
            }
        )
    if len(present_dates) > 1:
        invalid.append(
            {
                "pattern": "artifact_date_consistency",
                "path": "",
                "issues": ["final artifacts have mixed date suffixes"],
                "artifacts": selected_artifact_dates,
            }
        )
    if selected_artifact_dates:
        evidence.append(
            {
                "artifact_date_consistency": selected_artifact_dates,
                "unique_dates": sorted(present_dates),
            }
        )
    if not missing:
        if not invalid:
            return [
                status_item(
                    "Phase 4 final evaluation artifacts",
                    "passed",
                    evidence=evidence,
                )
            ]
        return [
            status_item(
                "Phase 4 final evaluation artifacts",
                "partial",
                evidence=evidence,
                missing=invalid,
                note="Final artifact files exist, but at least one JSON artifact is stale, partial, smoke-only, or not from the full DeanDataset_full_unet validation split.",
            )
        ]
    return [
        status_item(
            "Phase 4 final evaluation artifacts",
            "missing",
            evidence=evidence,
            missing=[*missing, *invalid],
            note="Final reproduction requires both EPNet/FACET paper comparison and HBTXR-vs-EPNet comparison artifacts.",
        )
    ]


def summarize(items):
    if any(item["state"] == "blocked" for item in items):
        overall = "blocked"
    elif any(item["state"] in {"missing", "partial"} for item in items):
        overall = "incomplete"
    else:
        overall = "passed"
    counts = {}
    for item in items:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    return overall, counts


def make_markdown(result):
    lines = [
        "# FACET Reproduction Status",
        "",
        f"Overall status: `{result['overall_status']}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in sorted(result["counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Items", ""])
    for item in result["items"]:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- state: `{item['state']}`",
            ]
        )
        if item["note"]:
            lines.append(f"- note: {item['note']}")
        if item["evidence"]:
            lines.append("- evidence:")
            for evidence in item["evidence"]:
                lines.append(f"  - `{evidence}`")
        if item["missing"]:
            lines.append("- missing:")
            for missing in item["missing"]:
                lines.append(f"  - `{missing}`")
        lines.append("")
    lines.extend(
        [
            "## Completion Requirement",
            "",
            "Do not mark the FACET reproduction complete until all items are `passed` and the final evaluation artifacts are produced from the full checkpoint and full validation split.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check FACET reproduction progress gates.")
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".facet-train-venv/bin/python",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    items = []
    items.append(check_preflight(args.python))
    items.extend(check_dataset_manifests())
    items.extend(check_reports_and_samples())
    items.extend(check_checkpoints())
    items.extend(check_final_evaluation())
    overall, counts = summarize(items)
    result = {
        "overall_status": overall,
        "counts": counts,
        "items": items,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
