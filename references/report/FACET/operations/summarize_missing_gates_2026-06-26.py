#!/usr/bin/env python3
"""Summarize FACET missing gates without reading training logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/kjm26/project/PRJXR/HBTXR")
REPORT_ROOT = ROOT / "references/report/FACET"
OPERATIONS_ROOT = REPORT_ROOT / "operations"
ANALYSIS_ROOT = REPORT_ROOT / "analysis"
TRAINING_ROOT = REPORT_ROOT / "training"
AUDIT_JSON = OPERATIONS_ROOT / "FACET_reproduction_completion_audit_2026-06-26.json"
PROGRESS_JSON = TRAINING_ROOT / "FACET_full_training_progress_snapshot_2026-06-26.json"
STATUS_JSON = OPERATIONS_ROOT / "FACET_reproduction_status_2026-06-26.json"
STATUS_MD = ANALYSIS_ROOT / "FACET_reproduction_status_2026-06-26.md"
PROGRESS_MD = TRAINING_ROOT / "FACET_full_training_progress_snapshot_2026-06-26.md"
EXPECTED_SESSIONS = [
    "facet_epnet_full_gpu0",
    "facet_hbtxr_full_gpu1",
    "facet_full_training_watchdog",
    "facet_full_eval_watcher",
    "facet_epnet_fpn_dw_gpu0_waiter",
    "facet_epnet_fpn_dw_eval_watcher",
    "facet_hbtxr_effbs32_gpu1_waiter",
    "facet_hbtxr_effbs32_eval_watcher",
    "facet_followup_training_watchdog",
    "facet_next_hourly_refresh_once",
]
EXPECTED_PROCESSES = [
    "tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml",
    "tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml",
    "watch_full_training_jobs_2026-06-26.sh",
    "watch_followup_training_jobs_2026-06-26.sh",
    "watch_full_checkpoints_and_evaluate_2026-06-26.sh",
    "watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh",
    "watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh",
    "run_next_hourly_refresh_once_2026-06-26.sh",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_local(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def latest_artifact_mtime(paths: list[Path]) -> tuple[float, Path | None]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return 0.0, None
    latest = max(existing, key=lambda path: path.stat().st_mtime)
    return latest.stat().st_mtime, latest


def command_result(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, check=False, text=True, capture_output=True)
    return result


def tmux_session_names() -> tuple[set[str], str | None]:
    result = command_result(["tmux", "ls"])
    if result.returncode != 0:
        reason = (result.stderr or "tmux ls failed").strip()
        return set(), reason
    names: set[str] = set()
    for line in result.stdout.splitlines():
        if ":" in line:
            names.add(line.split(":", 1)[0])
    return names, None


def process_lines() -> tuple[list[str], str | None]:
    result = command_result(["ps", "-eo", "pid=,args="])
    if result.returncode != 0:
        reason = (result.stderr or "ps failed").strip()
        return [], reason
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    # Codex sandboxed subprocesses run in a separate PID namespace. In that
    # case, host training processes are not observable and "missing" would be
    # misleading.
    if any("bwrap --new-session" in line or "codex-linux-sandbox" in line for line in lines):
        return [], "host process namespace unavailable from this subprocess"
    return lines, None


def append_runtime_snapshot(lines: list[str]) -> None:
    sessions, tmux_error = tmux_session_names()
    processes, process_error = process_lines()

    lines.append("")
    lines.append("## Runtime Snapshot")
    lines.append("")
    lines.append("This section reads tmux and process listings only. It does not scan training logs.")
    lines.append("")
    lines.append("| Expected session | State |")
    lines.append("|---|---|")
    for session in EXPECTED_SESSIONS:
        if tmux_error is not None:
            state = "unavailable"
        else:
            state = "alive" if session in sessions else "missing"
        lines.append(f"| `{session}` | `{state}` |")
    if tmux_error is not None:
        lines.append("")
        lines.append(f"tmux_status_note: `{tmux_error}`")

    lines.append("")
    lines.append("| Expected process pattern | State |")
    lines.append("|---|---|")
    for pattern in EXPECTED_PROCESSES:
        if process_error is not None:
            state = "unavailable"
        else:
            state = "present" if any(pattern in line for line in processes) else "missing"
        lines.append(f"| `{pattern}` | `{state}` |")
    if process_error is not None:
        lines.append("")
        lines.append(f"process_status_note: `{process_error}`")


def build_summary(include_runtime: bool = False) -> str:
    audit = load_json(AUDIT_JSON)
    progress = load_json(PROGRESS_JSON)

    min_interval = int(os.environ.get("FACET_STATUS_REFRESH_MIN_INTERVAL_SECONDS", "3600"))
    latest_mtime, latest_path = latest_artifact_mtime(
        [STATUS_JSON, STATUS_MD, PROGRESS_JSON, PROGRESS_MD, AUDIT_JSON]
    )
    now = time.time()
    age = int(now - latest_mtime) if latest_mtime else 0
    next_due_in = max(0, min_interval - age) if latest_mtime else 0
    refresh_state = "due" if latest_mtime and age >= min_interval else "fresh"

    lines: list[str] = []
    lines.append("# FACET Missing Gate Summary")
    lines.append("")
    lines.append(f"generated_local: `{fmt_local(now)}`")
    if latest_path is not None:
        lines.append(f"latest_artifact: `{latest_path}`")
        lines.append(f"latest_artifact_mtime: `{fmt_local(latest_mtime)}`")
    lines.append(f"refresh_min_interval_seconds: `{min_interval}`")
    lines.append(f"latest_artifact_age_seconds: `{age}`")
    lines.append(f"refresh_next_due_in_seconds: `{next_due_in}`")
    lines.append(f"refresh_state: `{refresh_state}`")
    lines.append("")
    lines.append("## Completion")
    lines.append("")
    lines.append(f"overall: `{audit.get('status_overall')}`")
    lines.append(f"counts: `{audit.get('status_counts')}`")
    lines.append(f"can_mark_goal_complete: `{audit.get('can_mark_goal_complete')}`")
    lines.append(f"completion_decision: `{audit.get('completion_decision')}`")
    lines.append("")
    lines.append("## Missing Gates")
    lines.append("")
    lines.append("| # | Gate | Missing evidence |")
    lines.append("|---:|---|---|")
    for idx, item in enumerate(audit.get("non_passed_status_items", []), start=1):
        missing = ", ".join(item.get("missing") or [])
        lines.append(f"| {idx} | {item.get('name', '')} | {missing} |")
    lines.append("")
    lines.append("## Progress Snapshot")
    lines.append("")
    lines.append("| Model | Epoch | Step | Progress | Rate | Remaining | Checkpoints |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model in progress.get("models", []):
        prog = model.get("progress", {})
        step_text = f"{prog.get('step', 'n/a')} / {prog.get('total_steps', 'n/a')}"
        percent = prog.get("percent_precise")
        percent_text = "n/a" if percent is None else f"{percent:.2f}%"
        lines.append(
            "| {name} | {epoch} | {step} | {percent} | {rate} it/s | {remaining} | {ckpts} |".format(
                name=model.get("name", ""),
                epoch=prog.get("epoch", "n/a"),
                step=step_text,
                percent=percent_text,
                rate=prog.get("rate_it_s", "n/a"),
                remaining=prog.get("remaining_text", "n/a"),
                ckpts=model.get("checkpoint_count", "n/a"),
            )
        )
    lines.append("")
    lines.append("This summary only reads status/progress/audit JSON artifacts. It does not scan training logs.")
    if include_runtime:
        append_runtime_snapshot(lines)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", type=Path, help="Optional Markdown output path.")
    parser.add_argument(
        "--include-runtime",
        action="store_true",
        help="Include tmux/process status without reading training logs.",
    )
    args = parser.parse_args()

    summary = build_summary(include_runtime=args.include_runtime)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
