#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
PY="$ROOT/.facet-train-venv/bin/python"
OUT_JSON="/tmp/facet_artifact_validation_summary_smoke.json"
OUT_MD="/tmp/facet_artifact_validation_summary_smoke.md"
NULL_METRIC_JSON="/tmp/facet_artifact_validation_null_metric.json"
MISSING_CKPT_JSON="/tmp/facet_artifact_validation_missing_checkpoint.json"
PARTIAL_COMPARISON_JSON="/tmp/facet_artifact_validation_partial_comparison.json"
PARTIAL_SUMMARY_JSON="/tmp/facet_artifact_validation_partial_summary.json"
BAD_PATH_SUMMARY_JSON="/tmp/facet_artifact_validation_bad_path_summary.json"
PARTIAL_MARKDOWN="/tmp/facet_artifact_validation_partial.md"
EXISTING_CKPT="/tmp/facet_artifact_validation_existing.ckpt"

export PYTHONPATH="$FACET_ROOT"
export PYTHONPYCACHEPREFIX=/tmp/facet_artifact_validation_smoke_pycache

SMOKE_JSON="$REPORT_ROOT/FACET_phase4_epnet_eval_smoke_2026-06-25.json"
printf 'placeholder checkpoint for validation smoke\n' >"$EXISTING_CKPT"

if "$PY" "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
  --type eval \
  --path "$SMOKE_JSON" \
  >/tmp/facet_artifact_validation_eval_smoke_stdout.json 2>/tmp/facet_artifact_validation_eval_smoke_stderr.txt; then
  echo "expected smoke evaluation artifact to be rejected, but validator accepted it" >&2
  exit 1
fi

"$PY" - "$NULL_METRIC_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "model_type": "EPNet",
            "config": "DavisEyeEllipse_EPNet_full_unet.yaml",
            "checkpoint": "/tmp/facet_artifact_validation_existing.ckpt",
            "device": "cuda:0",
            "dataset_root": "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet",
            "max_batches": 0,
            "evaluated_batches": 1,
            "metrics": {
                "val_p10_acc": 1.0,
                "val_p5_acc": 1.0,
                "val_p3_acc": None,
                "val_p1_acc": 1.0,
                "val_mean_distance": 0.1,
                "val_IoU": 1.0,
                "val_AP": 1.0,
            },
            "params_m": 3.9,
            "trainable_params_m": 3.9,
            "flops_g": 3.4,
            "latency_ms": 0.5,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

if "$PY" "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
  --type eval \
  --path "$NULL_METRIC_JSON" \
  >/tmp/facet_artifact_validation_null_metric_stdout.json 2>/tmp/facet_artifact_validation_null_metric_stderr.txt; then
  echo "expected null-metric evaluation artifact to be rejected, but validator accepted it" >&2
  exit 1
fi

"$PY" - "$MISSING_CKPT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "model_type": "EPNet",
            "config": "DavisEyeEllipse_EPNet_full_unet.yaml",
            "checkpoint": "/tmp/facet_artifact_validation_missing.ckpt",
            "device": "cuda:0",
            "dataset_root": "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet",
            "max_batches": 0,
            "evaluated_batches": 1,
            "metrics": {
                "val_p10_acc": 1.0,
                "val_p5_acc": 1.0,
                "val_p3_acc": 1.0,
                "val_p1_acc": 1.0,
                "val_mean_distance": 0.1,
                "val_IoU": 1.0,
                "val_AP": 1.0,
            },
            "params_m": 3.9,
            "trainable_params_m": 3.9,
            "flops_g": 3.4,
            "latency_ms": 0.5,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

if "$PY" "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
  --type eval \
  --path "$MISSING_CKPT_JSON" \
  >/tmp/facet_artifact_validation_missing_checkpoint_stdout.json 2>/tmp/facet_artifact_validation_missing_checkpoint_stderr.txt; then
  echo "expected missing-checkpoint evaluation artifact to be rejected, but validator accepted it" >&2
  exit 1
fi

"$PY" - "$PARTIAL_COMPARISON_JSON" <<'PY'
import json
import sys
from pathlib import Path

side = {
    "model_type": "EPNet",
    "config": "DavisEyeEllipse_EPNet_full_unet.yaml",
    "checkpoint": "/tmp/facet_artifact_validation_existing.ckpt",
    "device": "cuda:0",
    "evaluated_batches": 1,
    "dataset_root": "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet",
}
path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "left_label": "left",
            "right_label": "right",
            "left": side,
            "right": side,
            "rows": [
                {
                    "metric": "P10",
                    "left": 1.0,
                    "right": 1.0,
                    "right_minus_left": 0.0,
                    "preferred_direction": "higher",
                    "winner": "tie",
                }
            ],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

if "$PY" "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
  --type comparison \
  --path "$PARTIAL_COMPARISON_JSON" \
  >/tmp/facet_artifact_validation_partial_comparison_stdout.json 2>/tmp/facet_artifact_validation_partial_comparison_stderr.txt; then
  echo "expected partial comparison artifact to be rejected, but validator accepted it" >&2
  exit 1
fi

"$PY" - "$PARTIAL_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "summary_state": "complete",
            "missing_artifacts": [],
            "results": [
                {"label": "EPNet_full_unet", "state": "available"}
            ],
            "comparisons": [
                {"label": "EPNet_vs_HBTXR", "state": "available"},
                {"label": "EPNet_vs_HBTXR_effbs32", "state": "available"},
            ],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

if "$PY" "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
  --type summary \
  --path "$PARTIAL_SUMMARY_JSON" \
  >/tmp/facet_artifact_validation_partial_summary_stdout.json 2>/tmp/facet_artifact_validation_partial_summary_stderr.txt; then
  echo "expected partial summary artifact to be rejected, but validator accepted it" >&2
  exit 1
fi

"$PY" - "$BAD_PATH_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

report_root = Path("/tmp/facet_artifact_validation_summary_paths")
report_root.mkdir(parents=True, exist_ok=True)
bad_epnet_path = report_root / "FACET_wrong_results_2026-06-26.json"
for name in [
    "FACET_hbtxr_reproduction_results_2026-06-26.json",
    "FACET_epnet_fpn_dw_reproduction_results_2026-06-26.json",
    "FACET_hbtxr_effbs32_reproduction_results_2026-06-26.json",
    "FACET_epnet_vs_hbtxr_comparison_2026-06-26.json",
    "FACET_epnet_vs_hbtxr_effbs32_comparison_2026-06-26.json",
]:
    (report_root / name).write_text("{}\n", encoding="utf-8")
bad_epnet_path.write_text("{}\n", encoding="utf-8")

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "summary_state": "complete",
            "missing_artifacts": [],
            "results": [
                {"label": "EPNet_full_unet", "state": "available", "path": str(bad_epnet_path)},
                {
                    "label": "HBTXR_full_unet",
                    "state": "available",
                    "path": str(report_root / "FACET_hbtxr_reproduction_results_2026-06-26.json"),
                },
                {
                    "label": "EPNet_fpn_dw_full_unet",
                    "state": "available",
                    "path": str(report_root / "FACET_epnet_fpn_dw_reproduction_results_2026-06-26.json"),
                },
                {
                    "label": "HBTXR_full_unet_effbs32",
                    "state": "available",
                    "path": str(report_root / "FACET_hbtxr_effbs32_reproduction_results_2026-06-26.json"),
                },
            ],
            "comparisons": [
                {
                    "label": "EPNet_vs_HBTXR",
                    "state": "available",
                    "path": str(report_root / "FACET_epnet_vs_hbtxr_comparison_2026-06-26.json"),
                },
                {
                    "label": "EPNet_vs_HBTXR_effbs32",
                    "state": "available",
                    "path": str(report_root / "FACET_epnet_vs_hbtxr_effbs32_comparison_2026-06-26.json"),
                },
            ],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

if "$PY" "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
  --type summary \
  --path "$BAD_PATH_SUMMARY_JSON" \
  >/tmp/facet_artifact_validation_bad_path_summary_stdout.json 2>/tmp/facet_artifact_validation_bad_path_summary_stderr.txt; then
  echo "expected bad-path summary artifact to be rejected, but validator accepted it" >&2
  exit 1
fi

"$PY" <<'PY'
from EvEye.utils.scripts.check_reproduction_status import (
    artifact_date,
    validate_expected_comparison_context,
    validate_expected_eval_context,
    validate_markdown_artifact,
)
from pathlib import Path

issues = []
validate_expected_eval_context(
    {"config": "wrong.yaml", "model_type": "EPNet"},
    {"config": "DavisEyeEllipse_EPNet_full_unet.yaml", "model_type": "EPNet"},
    issues,
)
if not any("config is 'wrong.yaml'" in issue for issue in issues):
    raise SystemExit("expected eval config mismatch validation issue")

issues = []
validate_expected_comparison_context(
    {"left_label": "EPNet_full_unet", "right_label": "wrong"},
    {"left_label": "EPNet_full_unet", "right_label": "HBTXR_full_unet"},
    issues,
)
if not any("right_label is 'wrong'" in issue for issue in issues):
    raise SystemExit("expected comparison right_label mismatch validation issue")

partial_markdown = Path("/tmp/facet_artifact_validation_partial.md")
partial_markdown.write_text("# Wrong Report\n\nNo metric table here.\n", encoding="utf-8")
ok, issues, _ = validate_markdown_artifact(
    partial_markdown,
    [
        "# FACET Reproduction Results",
        "## Evaluation Artifacts",
        "## Model Metrics",
    ],
)
if ok:
    raise SystemExit("expected partial markdown artifact to be rejected")
if not any("missing markdown terms" in issue for issue in issues):
    raise SystemExit("expected missing markdown terms issue")

if artifact_date(Path("/tmp/FACET_reproduction_results_2026-06-26.md")) != "2026-06-26":
    raise SystemExit("expected artifact date suffix to be parsed")
if artifact_date(Path("/tmp/FACET_reproduction_results_latest.md")) is not None:
    raise SystemExit("expected non-date artifact suffix to be rejected")
dates = {
    artifact_date(Path("/tmp/FACET_reproduction_results_2026-06-26.md")),
    artifact_date(Path("/tmp/FACET_hbtxr_reproduction_results_2026-06-27.md")),
}
if len(dates) != 2:
    raise SystemExit("expected mixed artifact dates to be detectable")
PY

"$PY" "$FACET_ROOT/EvEye/utils/scripts/build_reproduction_summary.py" \
  --result "EPNet_smoke:$SMOKE_JSON" \
  --output-json "$OUT_JSON" \
  --output-md "$OUT_MD" \
  >/tmp/facet_artifact_validation_summary_stdout.json

"$PY" - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("summary_state") != "partial":
    raise SystemExit(f"expected summary_state=partial, got {data.get('summary_state')!r}")
results = data.get("results") or []
if len(results) != 1:
    raise SystemExit(f"expected one result, got {len(results)}")
result = results[0]
if result.get("state") != "invalid":
    raise SystemExit(f"expected EPNet_smoke state=invalid, got {result.get('state')!r}")
issues = result.get("validation_issues") or []
if not any("max_batches is 2" in issue for issue in issues):
    raise SystemExit("expected max_batches validation issue")
if not any("DeanDataset_full_unet" in issue for issue in issues):
    raise SystemExit("expected DeanDataset_full_unet validation issue")
print("artifact validation smoke passed")
PY
