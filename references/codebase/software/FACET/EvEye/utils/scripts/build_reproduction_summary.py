import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


FACET_ROOT = Path(__file__).resolve().parents[3]
if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))

from EvEye.utils.scripts.check_reproduction_status import (  # noqa: E402
    validate_comparison_json,
    validate_eval_result_json,
)


METRIC_ROWS = [
    ("P10", ("metrics", "val_p10_acc")),
    ("P5", ("metrics", "val_p5_acc")),
    ("P3", ("metrics", "val_p3_acc")),
    ("P1", ("metrics", "val_p1_acc")),
    ("mean pixel error", ("metrics", "val_mean_distance")),
    ("IoU", ("metrics", "val_IoU")),
    ("AP", ("metrics", "val_AP")),
    ("params M", ("params_m",)),
    ("trainable params M", ("trainable_params_m",)),
    ("FLOPs G", ("flops_g",)),
    ("latency ms", ("latency_ms",)),
]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nested_get(data, keys):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def parse_labeled_path(value: str):
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected LABEL:PATH")
    label, path = value.split(":", 1)
    if not label:
        raise argparse.ArgumentTypeError("label must be non-empty")
    return label, Path(path)


def summarize_result(label: str, path: Path):
    data = load_json(path)
    if data is None:
        return {
            "label": label,
            "path": str(path),
            "state": "missing",
            "data": None,
            "validation_issues": ["file is missing or not readable"],
        }
    valid, issues, _ = validate_eval_result_json(path)
    return {
        "label": label,
        "path": str(path),
        "state": "available" if valid else "invalid",
        "validation_issues": issues,
        "model_type": data.get("model_type"),
        "config": data.get("config"),
        "checkpoint": data.get("checkpoint"),
        "device": data.get("device"),
        "dataset_root": data.get("dataset_root"),
        "evaluated_batches": data.get("evaluated_batches"),
        "metrics": {
            name: nested_get(data, keys)
            for name, keys in METRIC_ROWS
        },
        "paper_table2_reference": data.get("paper_table2_reference"),
    }


def summarize_comparison(label: str, path: Path):
    data = load_json(path)
    if data is None:
        return {
            "label": label,
            "path": str(path),
            "state": "missing",
            "rows": [],
            "validation_issues": ["file is missing or not readable"],
        }
    valid, issues, _ = validate_comparison_json(path)
    return {
        "label": label,
        "path": str(path),
        "state": "available" if valid else "invalid",
        "validation_issues": issues,
        "left_label": data.get("left_label"),
        "right_label": data.get("right_label"),
        "left": data.get("left"),
        "right": data.get("right"),
        "rows": data.get("rows", []),
    }


def build_summary(args):
    results = [
        summarize_result(label, path)
        for label, path in args.result
    ]
    comparisons = [
        summarize_comparison(label, path)
        for label, path in args.comparison
    ]
    status = load_json(args.status_json) if args.status_json else None
    missing = [
        item["label"]
        for item in [*results, *comparisons]
        if item["state"] != "available"
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_json": str(args.status_json) if args.status_json else None,
        "status_overall": None if status is None else status.get("overall_status"),
        "status_counts": None if status is None else status.get("counts"),
        "summary_state": "complete" if not missing else "partial",
        "missing_artifacts": missing,
        "results": results,
        "comparisons": comparisons,
        "notes": [
            "This summary is regenerated from evaluation JSON artifacts.",
            "A partial state is expected while long-running full training or follow-up ablations are still incomplete.",
            "The report satisfies the reproduction-plan Markdown summary path only after the referenced final evaluation JSON artifacts exist.",
        ],
    }


def make_markdown(summary):
    lines = [
        "# FACET Reproduction Results",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Summary state: `{summary['summary_state']}`",
    ]
    if summary["status_json"]:
        lines.append(f"Status JSON: `{summary['status_json']}`")
    if summary["status_overall"]:
        lines.append(f"Status checker overall: `{summary['status_overall']}`")
    if summary["status_counts"]:
        lines.append(f"Status checker counts: `{summary['status_counts']}`")

    lines.extend(["", "## Evaluation Artifacts", ""])
    lines.append("| Label | State | Path |")
    lines.append("|---|---|---|")
    for item in summary["results"]:
        lines.append(f"| {item['label']} | {item['state']} | `{item['path']}` |")
    for item in summary["comparisons"]:
        lines.append(f"| {item['label']} | {item['state']} | `{item['path']}` |")

    lines.extend(["", "## Model Metrics", ""])
    header = "| Metric | " + " | ".join(item["label"] for item in summary["results"]) + " |"
    separator = "|---" + "|---:" * len(summary["results"]) + "|"
    lines.append(header)
    lines.append(separator)
    for metric_name, _ in METRIC_ROWS:
        row = [metric_name]
        for item in summary["results"]:
            metrics = item.get("metrics") or {}
            row.append(fmt(metrics.get(metric_name)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Run Context", ""])
    lines.append("| Label | Config | Checkpoint | Dataset | Batches | Device |")
    lines.append("|---|---|---|---|---:|---|")
    for item in summary["results"]:
        lines.append(
            "| {label} | `{config}` | `{checkpoint}` | `{dataset}` | {batches} | `{device}` |".format(
                label=item["label"],
                config=fmt(item.get("config")),
                checkpoint=fmt(item.get("checkpoint")),
                dataset=fmt(item.get("dataset_root")),
                batches=fmt(item.get("evaluated_batches")),
                device=fmt(item.get("device")),
            )
        )

    lines.extend(["", "## Pairwise Comparisons", ""])
    for comparison in summary["comparisons"]:
        lines.extend(
            [
                f"### {comparison['label']}",
                "",
                f"- state: `{comparison['state']}`",
                f"- path: `{comparison['path']}`",
            ]
        )
        if comparison["state"] != "available":
            if comparison.get("validation_issues"):
                lines.append("- validation issues:")
                for issue in comparison["validation_issues"]:
                    lines.append(f"  - {issue}")
            lines.append("")
            continue
        lines.append("")
        lines.append("| Metric | Left | Right | Right - left | Preferred | Winner |")
        lines.append("|---|---:|---:|---:|---|---|")
        for row in comparison["rows"]:
            lines.append(
                "| {metric} | {left} | {right} | {delta} | {direction} | {winner} |".format(
                    metric=row.get("metric"),
                    left=fmt(row.get("left")),
                    right=fmt(row.get("right")),
                    delta=fmt(row.get("right_minus_left")),
                    direction=fmt(row.get("preferred_direction")),
                    winner=fmt(row.get("winner")),
                )
            )
        lines.append("")

    lines.extend(["## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    if summary["missing_artifacts"]:
        lines.append(
            "- Missing or invalid artifacts: "
            + ", ".join(f"`{name}`" for name in summary["missing_artifacts"])
        )
    invalid_items = [
        item
        for item in [*summary["results"], *summary["comparisons"]]
        if item.get("state") == "invalid"
    ]
    if invalid_items:
        lines.extend(["", "## Validation Issues", ""])
        for item in invalid_items:
            lines.append(f"### {item['label']}")
            for issue in item.get("validation_issues", []):
                lines.append(f"- {issue}")
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Build a FACET reproduction summary from evaluation and comparison JSON artifacts."
    )
    parser.add_argument(
        "--result",
        type=parse_labeled_path,
        action="append",
        default=[],
        help="Evaluation result in LABEL:PATH form. Can be repeated.",
    )
    parser.add_argument(
        "--comparison",
        type=parse_labeled_path,
        action="append",
        default=[],
        help="Pairwise comparison in LABEL:PATH form. Can be repeated.",
    )
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(make_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
