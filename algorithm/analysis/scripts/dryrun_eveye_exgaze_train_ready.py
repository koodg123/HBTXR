from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from smoke_eveye_exgaze_hybrid_loaders import run_loader_smoke


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    ROOT / "analysis/configs/EV_Eye_Hybrid_frame128_event64_subject_independent_train_ready.json",
    ROOT / "analysis/configs/EX_Gaze_Hybrid_frame128_event64_subject_independent_train_ready.json",
]


def count_rows(path: Path) -> int:
    with path.open() as f:
        return max(sum(1 for _ in f) - 1, 0)


def validate_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    source_root = Path(cfg["dataset"]["source_root"])
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    split_results = {}
    for split, spec in cfg["dataset"]["splits"].items():
        manifest = dataset_root / spec["manifest"]
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        rows = count_rows(manifest)
        if rows != int(spec["samples"]):
            raise AssertionError(f"{manifest}: expected {spec['samples']}, got {rows}")
        split_results[split] = {
            "manifest": str(manifest),
            "rows": rows,
            "expected": int(spec["samples"]),
            "subjects": spec["subjects"],
        }
    training = cfg["training"]
    required_training = {
        "epochs": 70,
        "batch_size": 32,
        "num_workers": 4,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
    }
    for key, expected in required_training.items():
        if training[key] != expected:
            raise AssertionError(f"{path}: training.{key} expected {expected}, got {training[key]}")
    scheduler = training["scheduler"]
    if scheduler["type"] != "StepLR" or scheduler["step_size"] != 10 or scheduler["gamma"] != 0.7:
        raise AssertionError(f"{path}: scheduler mismatch: {scheduler}")
    return {
        "config": str(path),
        "experiment_name": cfg["experiment_name"],
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "splits": split_results,
        "training": training,
        "gpu_plan": cfg["gpu_plan"],
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# EV-Eye and EX-Gaze Train-Ready Dry Run",
        "",
        "Date: 2026-07-04",
        "",
        "## Config Validation",
        "",
        "| Experiment | GPU | Train | Val | Test | Batch | Workers | Epochs | Optimizer | Scheduler |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in result["configs"]:
        tr = item["training"]
        lines.append(
            f"| {item['experiment_name']} | {item['gpu_plan']['intended_gpu']} | "
            f"{item['splits']['train']['rows']} | {item['splits']['val']['rows']} | "
            f"{item['splits']['test']['rows']} | {tr['batch_size']} | {tr['num_workers']} | "
            f"{tr['epochs']} | {tr['optimizer']} lr={tr['learning_rate']} wd={tr['weight_decay']} | "
            f"{tr['scheduler']['type']} step={tr['scheduler']['step_size']} gamma={tr['scheduler']['gamma']} |"
        )
    live = result["live_loader"]
    lines.extend(
        [
            "",
            "## Live Loader Check",
            "",
            f"- Tag: `{live['tag']}`",
            f"- Batch size: `{live['batch_size']}`",
            f"- Live mode: `{live['live']}`",
            "",
            "| Split | EV len | EX len | EV frame | EV event | EX patches |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for split, data in live["splits"].items():
        lines.append(
            f"| {split} | {data['ev_len']} | {data['ex_len']} | "
            f"`{data['ev_frame']['shape']}` | `{data['ev_event']['shape']}` | "
            f"`{data['ex_event_patches']['shape']}` |"
        )
    lines.extend(
        [
            "",
            "## Judgment",
            "",
            "- Train-ready compact manifests exist for EV-Eye and EX-Gaze.",
            "- The split counts match the HBTXR subject-independent split.",
            "- Training hyperparameters are aligned to HBTXR: batch 32, workers 4, epochs 70, Adam, lr 1e-3, weight decay 1e-5, StepLR step 10 gamma 0.7.",
            "- Live loader first-batch checks pass for train/val/test.",
            "- Training has not been launched.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="train_ready")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "references/report/EV-Eye/EV_Eye_EX_Gaze_hybrid_train_ready_dryrun_2026-07-04.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "references/report/EV-Eye/EV_Eye_EX_Gaze_hybrid_train_ready_dryrun_2026-07-04.md",
    )
    parser.add_argument(
        "--mirror-markdown-output",
        type=Path,
        default=ROOT / "references/report/EX-Gaze/EV_Eye_EX_Gaze_hybrid_train_ready_dryrun_2026-07-04.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [validate_config(path) for path in CONFIGS]
    ev_root = Path(configs[0]["dataset_root"])
    target_base = ev_root.parents[1]
    source_root = Path(configs[0]["source_root"])
    live_loader = run_loader_smoke(args.tag, args.batch_size, target_base, source_root, live=True)
    result = {"configs": configs, "live_loader": live_loader}
    markdown = render_markdown(result)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown + "\n")
    if args.mirror_markdown_output:
        args.mirror_markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.mirror_markdown_output.write_text(markdown + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
