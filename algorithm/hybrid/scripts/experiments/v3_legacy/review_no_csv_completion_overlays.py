#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.utils.io import read_json, read_jsonl, write_json


FULLFRAME_SIZE = (346, 240)
LABEL_H = 54
COLUMNS = 4
ROWS = 4


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paginated overlay preview sheets for sampled no-CSV completion frames.",
    )
    parser.add_argument(
        "--completed-root",
        type=str,
        default="workspace_session_samples_11/all48_dataset_construction_same8samples_no_csv_completion_v10",
        help="Completed sampled dataset root created by no-CSV completion experiment.",
    )
    parser.add_argument(
        "--source-batch-root",
        type=str,
        default="workspace_session_samples_7/roi_crop_prompted_all48_v10_native_then_crop128_rescue_same8samples",
        help="ROI-crop preview batch root that holds overlay images.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="workspace_session_samples_11/all48_dataset_construction_same8samples_no_csv_completion_v10_overlay_preview_live",
        help="Output directory for paginated preview sheets.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _abs_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _load_source_frame_index(source_batch_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    frame_index: dict[tuple[str, str], dict[str, Any]] = {}
    for summary_path in sorted((source_batch_root / "sessions").rglob("summary.json")):
        summary = read_json(summary_path)
        session_key = str(summary["session_key"])
        for frame in summary.get("frames", []):
            frame_index[(session_key, str(frame["frame_filename"]))] = dict(frame)
    if not frame_index:
        raise FileNotFoundError(f"No summary.json frames found under {source_batch_root}")
    return frame_index


def _compose_panel(record: dict[str, Any]) -> Image.Image:
    image = Image.open(record["full_overlay_path"]).convert("RGB").resize(FULLFRAME_SIZE)
    panel = Image.new("RGB", (FULLFRAME_SIZE[0], FULLFRAME_SIZE[1] + LABEL_H), (8, 8, 8))
    panel.paste(image, (0, LABEL_H))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), str(record["session_key"]), fill=(255, 255, 255))
    draw.text((6, 18), str(record["frame_filename"]), fill=(255, 220, 120))
    draw.text((6, 32), f"stage={record['predicted_pupil_stage_name']} rank={record['predicted_candidate_rank']}", fill=(160, 255, 180))
    draw.text((6, 44), f"fail_like={int(bool(record['fail_like_union']))} conf={record['predicted_box_confidence']:.3f}", fill=(255, 170, 170))
    return panel


def _write_pages(records: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return []
    panel_w = FULLFRAME_SIZE[0]
    panel_h = FULLFRAME_SIZE[1] + LABEL_H
    page_size = COLUMNS * ROWS
    page_paths: list[str] = []
    for page_index in range(0, len(records), page_size):
        chunk = records[page_index : page_index + page_size]
        n_rows = math.ceil(len(chunk) / COLUMNS)
        canvas = Image.new("RGB", (panel_w * COLUMNS, panel_h * n_rows), (0, 0, 0))
        for idx, record in enumerate(chunk):
            panel = _compose_panel(record)
            x = (idx % COLUMNS) * panel_w
            y = (idx // COLUMNS) * panel_h
            canvas.paste(panel, (x, y))
        page_path = output_dir / f"page_{page_index // page_size + 1:03d}.png"
        canvas.save(page_path)
        page_paths.append(str(page_path.resolve()))
    return page_paths


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# No-CSV Completion Overlay Preview",
        "",
        "## Aggregate",
        f"- completed_root: {summary['completed_root']}",
        f"- source_batch_root: {summary['source_batch_root']}",
        f"- records: {summary['n_records']}",
        f"- fail_like_records: {summary['n_fail_like_records']}",
        f"- all_pages: {len(summary['all_pages'])}",
        f"- fail_like_pages: {len(summary['fail_like_pages'])}",
        "",
        "## Stage Counts",
    ]
    for key, value in summary.get("stage_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Fail-Like Sessions"])
    for row in summary.get("top_fail_like_sessions", []):
        lines.append(f"- {row['session_key']}: fail_like={row['n_fail_like']}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    completed_root = _abs_path(args.completed_root)
    source_batch_root = _abs_path(args.source_batch_root)
    output_root = _abs_path(args.output_root)
    summary_path = output_root / "overlay_preview_summary.json"
    if summary_path.exists() and not args.overwrite:
        summary = read_json(summary_path)
        print(
            f"[DONE] reused_existing_summary records={summary['n_records']} "
            f"all_pages={len(summary['all_pages'])} fail_like_pages={len(summary['fail_like_pages'])} "
            f"summary={summary_path}"
        )
        return

    completion_records = read_jsonl(completed_root / "analysis" / "no_csv_completion_records.jsonl")
    source_frame_index = _load_source_frame_index(source_batch_root)
    records: list[dict[str, Any]] = []
    fail_like_records: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    fail_like_session_counts: Counter[str] = Counter()

    for record in completion_records:
        source = source_frame_index[(str(record["session_key"]), str(record["frame_filename"]))]
        preview_row = {
            "session_key": str(record["session_key"]),
            "frame_filename": str(record["frame_filename"]),
            "frame_idx": int(record["frame_idx"]),
            "full_overlay_path": str(source["full_overlay_path"]),
            "crop_overlay_path": str(source.get("crop_overlay_path") or ""),
            "fail_like_union": bool(record.get("fail_like_union", False)),
            "predicted_pupil_stage_name": str(record.get("predicted_pupil_stage_name") or "none"),
            "predicted_candidate_rank": int(source.get("predicted_candidate_rank") or 0),
            "predicted_box_confidence": float(record.get("predicted_box_confidence") or 0.0),
        }
        records.append(preview_row)
        stage_counts[preview_row["predicted_pupil_stage_name"]] += 1
        if preview_row["fail_like_union"]:
            fail_like_records.append(dict(preview_row))
            fail_like_session_counts[preview_row["session_key"]] += 1

    records = sorted(records, key=lambda row: (row["session_key"], row["frame_idx"], row["frame_filename"]))
    fail_like_records = sorted(fail_like_records, key=lambda row: (row["session_key"], row["frame_idx"], row["frame_filename"]))

    all_pages = _write_pages(records, output_root / "all_pages")
    fail_like_pages = _write_pages(fail_like_records, output_root / "fail_like_pages")
    records_path = output_root / "overlay_preview_records.jsonl"
    fail_like_records_path = output_root / "overlay_preview_fail_like_records.jsonl"
    from hbtxr.utils.io import write_jsonl

    write_jsonl(records, records_path)
    write_jsonl(fail_like_records, fail_like_records_path)

    summary = {
        "completed_root": str(completed_root),
        "source_batch_root": str(source_batch_root),
        "output_root": str(output_root),
        "n_records": int(len(records)),
        "n_fail_like_records": int(len(fail_like_records)),
        "all_pages": all_pages,
        "fail_like_pages": fail_like_pages,
        "records_path": str(records_path.resolve()),
        "fail_like_records_path": str(fail_like_records_path.resolve()),
        "stage_counts": {key: int(stage_counts[key]) for key in sorted(stage_counts.keys())},
        "top_fail_like_sessions": [
            {"session_key": session_key, "n_fail_like": int(count)}
            for session_key, count in fail_like_session_counts.most_common(20)
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, summary_path)
    _write_markdown(summary, output_root / "overlay_preview_summary.md")

    print(
        f"[DONE] records={summary['n_records']} fail_like_records={summary['n_fail_like_records']} "
        f"all_pages={len(all_pages)} fail_like_pages={len(fail_like_pages)} summary={summary_path}"
    )


if __name__ == "__main__":
    main()
