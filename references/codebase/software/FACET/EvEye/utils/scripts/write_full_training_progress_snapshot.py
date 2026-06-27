import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PROGRESS_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+):\s+"
    r"(?P<percent>\d+)%\|.*?\|\s+"
    r"(?P<step>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[^<\]]+)<(?P<remaining>[^,\]]+),\s+"
    r"(?P<rate>[0-9.]+)it/s"
)


def parse_duration_seconds(value: str) -> int | None:
    parts = value.strip().split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return hours * 3600 + minutes * 60 + seconds
    return None


def parse_latest_progress(log_path: Path) -> dict:
    if not log_path.exists():
        return {
            "log": str(log_path),
            "exists": False,
            "progress_found": False,
        }

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(PROGRESS_RE.finditer(text.replace("\r", "\n")))
    stat = log_path.stat()
    if not matches:
        return {
            "log": str(log_path),
            "exists": True,
            "progress_found": False,
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }

    match = matches[-1]
    step = int(match.group("step"))
    total = int(match.group("total"))
    percent_precise = step / total * 100.0 if total else None
    rate = float(match.group("rate"))
    remaining_seconds = parse_duration_seconds(match.group("remaining"))
    elapsed_seconds = parse_duration_seconds(match.group("elapsed"))
    return {
        "log": str(log_path),
        "exists": True,
        "progress_found": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "epoch": int(match.group("epoch")),
        "percent_text": int(match.group("percent")),
        "percent_precise": percent_precise,
        "step": step,
        "total_steps": total,
        "steps_remaining": max(total - step, 0),
        "elapsed_text": match.group("elapsed"),
        "elapsed_seconds": elapsed_seconds,
        "remaining_text": match.group("remaining"),
        "remaining_seconds": remaining_seconds,
        "rate_it_s": rate,
    }


def count_checkpoints(root: Path) -> int:
    if not root.exists():
        return 0
    return len(list(root.glob("version_*/checkpoints/*.ckpt")))


def make_markdown(snapshot: dict) -> str:
    lines = [
        "# FACET Full Training Progress Snapshot",
        "",
        f"Generated UTC: `{snapshot['generated_utc']}`",
        "",
        "| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in snapshot["models"]:
        if not model["progress"].get("progress_found"):
            lines.append(
                f"| {model['name']} | n/a | n/a | n/a | n/a | n/a | {model['checkpoint_count']} |"
            )
            continue
        progress = model["progress"]
        percent = progress.get("percent_precise")
        percent_text = "n/a" if percent is None else f"{percent:.2f}%"
        step_text = f"{progress['step']} / {progress['total_steps']}"
        rate_text = f"{progress['rate_it_s']:.2f} it/s"
        lines.append(
            "| "
            f"{model['name']} | "
            f"{progress['epoch']} | "
            f"{step_text} | "
            f"{percent_text} | "
            f"{rate_text} | "
            f"{progress['remaining_text']} | "
            f"{model['checkpoint_count']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This snapshot is parsed from the current training logs and is not a final result.",
            "- Full reproduction still requires final full-training checkpoints and evaluation artifacts.",
            "- Checkpoint counts include all `.ckpt` files under each full training run root.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Write a progress snapshot for FACET EPNet/HBTXR full training logs."
    )
    parser.add_argument("--epnet-log", type=Path, required=True)
    parser.add_argument("--hbtxr-log", type=Path, required=True)
    parser.add_argument("--epnet-run-root", type=Path, required=True)
    parser.add_argument("--hbtxr-run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    snapshot = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "models": [
            {
                "name": "EPNet_full_unet",
                "progress": parse_latest_progress(args.epnet_log),
                "run_root": str(args.epnet_run_root),
                "checkpoint_count": count_checkpoints(args.epnet_run_root),
            },
            {
                "name": "HBTXR_full_unet",
                "progress": parse_latest_progress(args.hbtxr_log),
                "run_root": str(args.hbtxr_run_root),
                "checkpoint_count": count_checkpoints(args.hbtxr_run_root),
            },
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(make_markdown(snapshot), encoding="utf-8")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
