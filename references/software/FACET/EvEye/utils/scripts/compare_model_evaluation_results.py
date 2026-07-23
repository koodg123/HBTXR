import argparse
import json
import sys
from pathlib import Path


FACET_ROOT = Path(__file__).resolve().parents[3]
if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))

from EvEye.utils.scripts.check_reproduction_status import validate_eval_result_json  # noqa: E402


METRIC_ROWS = [
    ("P10", ("metrics", "val_p10_acc"), "higher"),
    ("P5", ("metrics", "val_p5_acc"), "higher"),
    ("P3", ("metrics", "val_p3_acc"), "higher"),
    ("P1", ("metrics", "val_p1_acc"), "higher"),
    ("mean pixel error", ("metrics", "val_mean_distance"), "lower"),
    ("IoU", ("metrics", "val_IoU"), "higher"),
    ("AP", ("metrics", "val_AP"), "higher"),
    ("params M", ("params_m",), "lower"),
    ("trainable params M", ("trainable_params_m",), "lower"),
    ("FLOPs G", ("flops_g",), "lower"),
    ("latency ms", ("latency_ms",), "lower"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_valid_eval_json(path: Path, label: str):
    ok, issues, _ = validate_eval_result_json(path)
    if ok:
        return
    joined = "; ".join(issues)
    raise SystemExit(f"{label} evaluation JSON is not valid for full reproduction: {joined}")


def nested_get(data, keys):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def as_float(value):
    if value is None:
        return None
    return float(value)


def winner(left, right, direction):
    if left is None or right is None:
        return "n/a"
    if left == right:
        return "tie"
    if direction == "higher":
        return "left" if left > right else "right"
    return "left" if left < right else "right"


def fmt(value):
    return "n/a" if value is None else f"{value:.6g}"


def build_rows(left, right, left_label, right_label):
    rows = []
    for name, keys, direction in METRIC_ROWS:
        left_value = as_float(nested_get(left, keys))
        right_value = as_float(nested_get(right, keys))
        delta = None if left_value is None or right_value is None else right_value - left_value
        row_winner = winner(left_value, right_value, direction)
        if row_winner == "left":
            winner_label = left_label
        elif row_winner == "right":
            winner_label = right_label
        else:
            winner_label = row_winner
        rows.append(
            {
                "metric": name,
                "left": left_value,
                "right": right_value,
                "right_minus_left": delta,
                "preferred_direction": direction,
                "winner": winner_label,
            }
        )
    return rows


def make_markdown(result):
    left_label = result["left_label"]
    right_label = result["right_label"]
    lines = [
        f"# {left_label} vs {right_label} Evaluation Comparison",
        "",
        f"Left: `{left_label}`",
        f"Right: `{right_label}`",
        f"Left checkpoint: `{result['left']['checkpoint']}`",
        f"Right checkpoint: `{result['right']['checkpoint']}`",
        f"Left evaluated batches: `{result['left']['evaluated_batches']}`",
        f"Right evaluated batches: `{result['right']['evaluated_batches']}`",
        "",
        "| Metric | "
        + left_label
        + " | "
        + right_label
        + " | Right - left | Preferred | Winner |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in result["rows"]:
        lines.append(
            "| {metric} | {left} | {right} | {delta} | {direction} | {winner} |".format(
                metric=row["metric"],
                left=fmt(row["left"]),
                right=fmt(row["right"]),
                delta=fmt(row["right_minus_left"]),
                direction=row["preferred_direction"],
                winner=row["winner"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a pairwise comparison of locally evaluated checkpoints on the configured validation split.",
            "- Positive `Right - left` is better only for metrics whose preferred direction is `higher`.",
            "- Latency is measured in the current Python runtime and is not directly equivalent to the paper's optimized deployment latency.",
            "- If the two runs used different effective batch sizes, interpret training dynamics separately from validation-set metric comparability.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two FACET evaluation JSON artifacts and emit pairwise JSON/Markdown."
    )
    parser.add_argument("--left-json", type=Path, required=True)
    parser.add_argument("--right-json", type=Path, required=True)
    parser.add_argument("--left-label", default="EPNet")
    parser.add_argument("--right-label", default="HBTXR")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--allow-invalid-inputs",
        action="store_true",
        help="Allow smoke or partial evaluation JSON inputs. Intended only for debugging.",
    )
    args = parser.parse_args()

    if not args.allow_invalid_inputs:
        require_valid_eval_json(args.left_json, args.left_label)
        require_valid_eval_json(args.right_json, args.right_label)

    left = load_json(args.left_json)
    right = load_json(args.right_json)
    result = {
        "left_label": args.left_label,
        "right_label": args.right_label,
        "left_json": str(args.left_json),
        "right_json": str(args.right_json),
        "left": {
            "model_type": left.get("model_type"),
            "config": left.get("config"),
            "checkpoint": left.get("checkpoint"),
            "device": left.get("device"),
            "evaluated_batches": left.get("evaluated_batches"),
            "dataset_root": left.get("dataset_root"),
        },
        "right": {
            "model_type": right.get("model_type"),
            "config": right.get("config"),
            "checkpoint": right.get("checkpoint"),
            "device": right.get("device"),
            "evaluated_batches": right.get("evaluated_batches"),
            "dataset_root": right.get("dataset_root"),
        },
        "rows": build_rows(left, right, args.left_label, args.right_label),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
