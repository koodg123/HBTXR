#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import EllipseAnnotation, discover_session_layout, parse_via_csv_with_report
from hbtxr.preprocess.raw_ellipse_blink import build_raw_ellipse_blink_metadata
from hbtxr.utils.io import read_json, write_json


SENSOR_WIDTH = 346
SENSOR_HEIGHT = 240
FULLFRAME_SIZE = (SENSOR_WIDTH, SENSOR_HEIGHT)
PAIR_GAP_PX = 8
LABEL_H = 42
SINGLE_COLUMNS = 4
SINGLE_ROWS = 4
COMPARE_COLUMNS = 2
COMPARE_ROWS = 4


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a ROI-crop prompted batch run into grouped review artifacts: "
            "session index, raw-CSV comparison sheets, no-raw-CSV preview sheets, and blink-labeled preview sheets."
        ),
    )
    parser.add_argument("--batch-root", type=str, required=True, help="Workspace root created by run_roi_crop_prompted_preview_batch.py")
    parser.add_argument("--output-root", type=str, required=True, help="Directory for grouped review outputs")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _fit_image_to_box(image: Image.Image, size_wh: tuple[int, int], *, background_rgb: tuple[int, int, int] = (16, 16, 16)) -> Image.Image:
    target_w, target_h = [int(v) for v in size_wh]
    canvas = Image.new("RGB", (target_w, target_h), background_rgb)
    src = image.copy()
    src.thumbnail((target_w, target_h))
    offset = ((target_w - src.width) // 2, (target_h - src.height) // 2)
    canvas.paste(src, offset)
    return canvas


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _draw_rotated_ellipse(
    draw: ImageDraw.ImageDraw,
    ellipse_xywht: list[float],
    *,
    outline: str,
    width: int = 2,
    steps: int = 72,
) -> None:
    cx, cy, ew, eh, theta = [float(v) for v in ellipse_xywht]
    a = max(1.0, ew / 2.0)
    b = max(1.0, eh / 2.0)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    points: list[tuple[float, float]] = []
    for index in range(int(max(12, steps))):
        angle = 2.0 * math.pi * float(index) / float(max(12, steps))
        ex = a * math.cos(angle)
        ey = b * math.sin(angle)
        px = cx + ex * cos_t - ey * sin_t
        py = cy + ex * sin_t + ey * cos_t
        points.append((px, py))
    if points:
        points.append(points[0])
        draw.line(points, fill=outline, width=width)


def _ellipse_bbox_xywh(ellipse_xywht: list[float]) -> list[float]:
    cx, cy, w, h, _theta = [float(v) for v in ellipse_xywht]
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def _render_raw_csv_overlay(frame_path: Path, annotation: EllipseAnnotation | None, *, caption: str) -> Image.Image:
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if annotation is not None:
        ellipse_xywht = [float(v) for v in annotation.ellipse_xywht]
        bbox_xywh = _ellipse_bbox_xywh(ellipse_xywht)
        _draw_xywh(draw, bbox_xywh, outline="#4da6ff", width=2)
        _draw_rotated_ellipse(draw, ellipse_xywht, outline="#00e0ff", width=2)
    draw.text((6, 6), caption, fill=(255, 255, 255))
    return image


def _compose_single_panel(record: dict[str, Any]) -> Image.Image:
    image = Image.open(record["full_overlay_path"]).convert("RGB").resize(FULLFRAME_SIZE)
    panel = Image.new("RGB", (FULLFRAME_SIZE[0], FULLFRAME_SIZE[1] + LABEL_H), (8, 8, 8))
    panel.paste(image, (0, LABEL_H))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), record["session_key"], fill=(255, 255, 255))
    draw.text((6, 16), record["frame_filename"], fill=(255, 220, 120))
    status_line = f"status={record['status']} csv={record['csv_status']}"
    blink_line = f"blink={record['blink_label_source']} fail_like={int(bool(record.get('fail_like_union')))}"
    draw.text((6, 28), status_line, fill=(160, 255, 180))
    draw.text((170, 28), blink_line, fill=(255, 160, 160))
    return panel


def _compose_compare_panel(record: dict[str, Any]) -> Image.Image:
    v2_image = Image.open(record["full_overlay_path"]).convert("RGB").resize(FULLFRAME_SIZE)
    raw_csv_image = _fit_image_to_box(record["raw_csv_overlay_image"], FULLFRAME_SIZE)
    panel_w = FULLFRAME_SIZE[0] * 2 + PAIR_GAP_PX
    panel = Image.new("RGB", (panel_w, FULLFRAME_SIZE[1] + LABEL_H), (8, 8, 8))
    panel.paste(v2_image, (0, LABEL_H))
    panel.paste(raw_csv_image, (FULLFRAME_SIZE[0] + PAIR_GAP_PX, LABEL_H))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), record["session_key"], fill=(255, 255, 255))
    draw.text((6, 16), record["frame_filename"], fill=(255, 220, 120))
    draw.text((6, 28), f"v2={record['status']} class={record.get('predicted_class_name')}", fill=(160, 255, 180))
    draw.text((FULLFRAME_SIZE[0] + PAIR_GAP_PX + 6, 28), f"raw_csv rows={record['csv_rows_found']}", fill=(120, 200, 255))
    return panel


def _write_single_pages(records: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return []
    page_paths: list[str] = []
    page_size = SINGLE_COLUMNS * SINGLE_ROWS
    panel_w = FULLFRAME_SIZE[0]
    panel_h = FULLFRAME_SIZE[1] + LABEL_H
    for page_index in range(0, len(records), page_size):
        page_records = records[page_index : page_index + page_size]
        rows = math.ceil(len(page_records) / SINGLE_COLUMNS)
        canvas = Image.new("RGB", (panel_w * SINGLE_COLUMNS, panel_h * rows), (0, 0, 0))
        for idx, record in enumerate(page_records):
            panel = _compose_single_panel(record)
            x = (idx % SINGLE_COLUMNS) * panel_w
            y = (idx // SINGLE_COLUMNS) * panel_h
            canvas.paste(panel, (x, y))
        page_path = output_dir / f"page_{page_index // page_size + 1:03d}.png"
        canvas.save(page_path)
        page_paths.append(str(page_path.resolve()))
    return page_paths


def _write_compare_pages(records: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return []
    page_paths: list[str] = []
    page_size = COMPARE_COLUMNS * COMPARE_ROWS
    panel_w = FULLFRAME_SIZE[0] * 2 + PAIR_GAP_PX
    panel_h = FULLFRAME_SIZE[1] + LABEL_H
    for page_index in range(0, len(records), page_size):
        page_records = records[page_index : page_index + page_size]
        rows = math.ceil(len(page_records) / COMPARE_COLUMNS)
        canvas = Image.new("RGB", (panel_w * COMPARE_COLUMNS, panel_h * rows), (0, 0, 0))
        for idx, record in enumerate(page_records):
            panel = _compose_compare_panel(record)
            x = (idx % COMPARE_COLUMNS) * panel_w
            y = (idx // COMPARE_COLUMNS) * panel_h
            canvas.paste(panel, (x, y))
        page_path = output_dir / f"page_{page_index // page_size + 1:03d}.png"
        canvas.save(page_path)
        page_paths.append(str(page_path.resolve()))
    return page_paths


def _load_csv_rows_by_filename(csv_path: Path | None) -> dict[str, list[dict[str, str]]]:
    rows_by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
    if csv_path is None or not csv_path.exists():
        return rows_by_filename
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_filename[str(row.get("filename", "")).strip()].append(dict(row))
    return rows_by_filename


def _int_from_row(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _csv_status(rows: list[dict[str, str]], csv_exists: bool) -> tuple[str, int | None]:
    if not csv_exists:
        return "csv_file_missing", None
    if not rows:
        return "csv_row_missing", None
    max_region = max((_int_from_row(row, "region_count", 0) for row in rows), default=0)
    if max_region > 0:
        return "csv_positive", int(max_region)
    return "csv_region_zero", 0


def _session_summary_records(batch_root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((batch_root / "sessions").rglob("summary.json"))]


def _write_markdown_report(summary: dict[str, Any], dst: Path) -> None:
    lines = [
        f"# ROI-Crop Prompted Batch Review ({summary['batch_root']})",
        "",
        "## Aggregate",
        f"- sessions: {summary['n_sessions']}",
        f"- sampled_frames: {summary['n_sampled_frames']}",
        f"- completed: {summary['n_completed']}",
        f"- eye_failed: {summary['n_eye_failed']}",
        f"- pupil_failed: {summary['n_pupil_failed']}",
        f"- fail_like: {summary['n_fail_like']}",
        f"- csv_positive: {summary['n_csv_positive']}",
        f"- no_raw_csv_supervision: {summary['n_no_raw_csv_supervision']}",
        f"- blink_labeled: {summary['n_blink_labeled']}",
        "",
        "## Output Pages",
        f"- csv_positive_compare_pages: {len(summary['csv_positive_compare_pages'])}",
        f"- no_raw_csv_pages: {len(summary['no_raw_csv_pages'])}",
        f"- blink_labeled_pages: {len(summary['blink_labeled_pages'])}",
        f"- pupil_failed_pages: {len(summary['pupil_failed_pages'])}",
        f"- fail_like_pages: {len(summary['fail_like_pages'])}",
        "",
        "## Top Sessions By Pupil Failed",
    ]
    for row in summary.get("top_sessions_by_pupil_failed", []):
        lines.append(
            f"- {row['session_key']}: pupil_failed={row['n_pupil_failed']} fail_like={row['n_fail_like_union']} completed={row['n_completed']}/{row['n_sampled']}"
        )
    lines.append("")
    lines.append("## Top Sessions By Fail Like")
    for row in summary.get("top_sessions_by_fail_like", []):
        lines.append(
            f"- {row['session_key']}: fail_like={row['n_fail_like_union']} pupil_failed={row['n_pupil_failed']} completed={row['n_completed']}/{row['n_sampled']}"
        )
    lines.append("")
    lines.append("## Top Sessions By No Raw CSV Supervision")
    for row in summary.get("top_sessions_by_no_raw_csv", []):
        lines.append(
            f"- {row['session_key']}: no_raw_csv={row['n_no_raw_csv_supervision']} blink_labeled={row['n_blink_labeled']}"
        )
    dst.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    batch_root = (PROJECT_ROOT / args.batch_root).resolve() if not Path(args.batch_root).is_absolute() else Path(args.batch_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    if output_root.exists() and not args.overwrite:
        existing = output_root / "review_summary.json"
        if existing.exists():
            cached = read_json(existing)
            print(f"[REUSED] {existing}")
            print(json.dumps(cached, indent=2, ensure_ascii=False))
            return

    output_root.mkdir(parents=True, exist_ok=True)
    session_summaries = _session_summary_records(batch_root)
    session_index: list[dict[str, Any]] = []
    csv_positive_compare_records: list[dict[str, Any]] = []
    no_raw_csv_records: list[dict[str, Any]] = []
    blink_labeled_records: list[dict[str, Any]] = []
    pupil_failed_records: list[dict[str, Any]] = []
    fail_like_records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    csv_status_counts: Counter[str] = Counter()
    blink_source_counts: Counter[str] = Counter()

    for session in session_summaries:
        raw_session_dir = Path(session["raw_session_dir"])
        layout = discover_session_layout(raw_session_dir, user_id=int(session["user_id"]))
        rows_by_filename = _load_csv_rows_by_filename(layout.annotation_csv)
        annotations, _parse_report = parse_via_csv_with_report(layout.annotation_csv) if layout.annotation_csv else ([], {})
        annotation_by_filename = {str(ann.frame_filename): ann for ann in annotations}
        blink_metadata_by_frame, _blink_report = build_raw_ellipse_blink_metadata(annotations) if annotations else ({}, {})

        n_no_raw_csv_supervision = 0
        n_blink_labeled = 0
        for frame in session.get("frames", []):
            frame_filename = str(frame["frame_filename"])
            csv_rows = rows_by_filename.get(frame_filename, [])
            csv_status, csv_max_region_count = _csv_status(csv_rows, csv_exists=(layout.annotation_csv is not None and layout.annotation_csv.exists()))
            csv_rows_found = len(csv_rows)
            csv_positive = bool(csv_status == "csv_positive")
            no_raw_csv_supervision = not csv_positive
            blink_meta = blink_metadata_by_frame.get(frame_filename, {})
            blink_from_heuristic = bool(blink_meta.get("closed_eye_flag", False))
            blink_labeled = bool(blink_from_heuristic or csv_status == "csv_region_zero")
            if blink_from_heuristic:
                blink_label_source = str(blink_meta.get("blink_candidate_source") or "raw_ellipse_heuristic")
            elif csv_status == "csv_region_zero":
                blink_label_source = "csv_region_zero_proxy"
            else:
                blink_label_source = "none"

            record = {
                "session_key": str(session["session_key"]),
                "user_id": int(session["user_id"]),
                "eye": str(session["eye"]),
                "session_code": str(session["session_code"]),
                "frame_filename": frame_filename,
                "frame_idx": int(frame["frame_idx"]),
                "timestamp_us": int(frame["timestamp_us"]),
                "status": str(frame["status"]),
                "predicted_class_name": frame.get("predicted_class_name"),
                "fail_like_union": bool(frame.get("fail_like_union", False)),
                "full_overlay_path": str(frame["full_overlay_path"]),
                "raw_frame_path": str(frame["raw_frame_path"]),
                "csv_status": csv_status,
                "csv_rows_found": int(csv_rows_found),
                "csv_max_region_count": csv_max_region_count,
                "no_raw_csv_supervision": bool(no_raw_csv_supervision),
                "blink_labeled": bool(blink_labeled),
                "blink_label_source": blink_label_source,
                "blink_candidate_score": blink_meta.get("blink_candidate_score"),
                "closed_eye_flag": bool(blink_from_heuristic),
            }
            all_records.append(record)
            csv_status_counts[csv_status] += 1
            blink_source_counts[blink_label_source] += 1
            if record["status"] == "pupil_failed":
                pupil_failed_records.append(record)
            if record["fail_like_union"]:
                fail_like_records.append(record)
            if no_raw_csv_supervision:
                no_raw_csv_records.append(record)
                n_no_raw_csv_supervision += 1
            if blink_labeled:
                blink_labeled_records.append(record)
                n_blink_labeled += 1
            if csv_positive:
                compare_record = dict(record)
                compare_record["raw_csv_overlay_image"] = _render_raw_csv_overlay(
                    Path(frame["raw_frame_path"]),
                    annotation_by_filename.get(frame_filename),
                    caption=f"{record['session_key']} | {frame_filename} | raw-csv",
                )
                csv_positive_compare_records.append(compare_record)

        session_index.append(
            {
                "session_key": str(session["session_key"]),
                "user_id": int(session["user_id"]),
                "eye": str(session["eye"]),
                "session_code": str(session["session_code"]),
                "contact_sheet_pairs": session.get("contact_sheet_pairs"),
                "summary_path": str((batch_root / "sessions" / f"user{int(session['user_id']):02d}" / str(session["eye"]) / f"session_{session['session_code']}" / "summary.json").resolve()),
                "n_sampled": int(session["n_sampled"]),
                "n_completed": int(session["n_completed"]),
                "n_eye_failed": int(session["n_eye_failed"]),
                "n_pupil_failed": int(session["n_pupil_failed"]),
                "n_fail_like_union": int(session["n_fail_like_union"]),
                "n_no_raw_csv_supervision": int(n_no_raw_csv_supervision),
                "n_blink_labeled": int(n_blink_labeled),
            }
        )

    session_index.sort(key=lambda row: row["session_key"])
    top_sessions_by_pupil_failed = sorted(session_index, key=lambda row: (-row["n_pupil_failed"], -row["n_fail_like_union"], row["session_key"]))[:20]
    top_sessions_by_fail_like = sorted(session_index, key=lambda row: (-row["n_fail_like_union"], -row["n_pupil_failed"], row["session_key"]))[:20]
    top_sessions_by_no_raw_csv = sorted(session_index, key=lambda row: (-row["n_no_raw_csv_supervision"], -row["n_blink_labeled"], row["session_key"]))[:20]

    csv_positive_pages = _write_compare_pages(csv_positive_compare_records, output_root / "csv_positive_compare" / "pages")
    no_raw_csv_pages = _write_single_pages(no_raw_csv_records, output_root / "no_raw_csv_overlays" / "pages")
    blink_labeled_pages = _write_single_pages(blink_labeled_records, output_root / "blink_labeled_overlays" / "pages")
    pupil_failed_pages = _write_single_pages(pupil_failed_records, output_root / "pupil_failed_overlays" / "pages")
    fail_like_pages = _write_single_pages(fail_like_records, output_root / "fail_like_overlays" / "pages")

    summary = {
        "experiment": "roi_crop_prompted_batch_review",
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "n_sessions": len(session_index),
        "n_sampled_frames": len(all_records),
        "n_completed": sum(1 for row in all_records if row["status"] == "completed"),
        "n_eye_failed": sum(1 for row in all_records if row["status"] == "eye_failed"),
        "n_pupil_failed": sum(1 for row in all_records if row["status"] == "pupil_failed"),
        "n_fail_like": sum(1 for row in all_records if row["fail_like_union"]),
        "n_csv_positive": sum(1 for row in all_records if row["csv_status"] == "csv_positive"),
        "n_no_raw_csv_supervision": len(no_raw_csv_records),
        "n_blink_labeled": len(blink_labeled_records),
        "csv_status_counts": dict(sorted(csv_status_counts.items())),
        "blink_label_source_counts": dict(sorted(blink_source_counts.items())),
        "session_index_path": str((output_root / "session_index.json").resolve()),
        "review_records_path": str((output_root / "review_records.json").resolve()),
        "csv_positive_compare_pages": csv_positive_pages,
        "no_raw_csv_pages": no_raw_csv_pages,
        "blink_labeled_pages": blink_labeled_pages,
        "pupil_failed_pages": pupil_failed_pages,
        "fail_like_pages": fail_like_pages,
        "top_sessions_by_pupil_failed": top_sessions_by_pupil_failed,
        "top_sessions_by_fail_like": top_sessions_by_fail_like,
        "top_sessions_by_no_raw_csv": top_sessions_by_no_raw_csv,
    }

    write_json(session_index, output_root / "session_index.json")
    write_json(all_records, output_root / "review_records.json")
    write_json(summary, output_root / "review_summary.json")
    _write_markdown_report(summary, output_root / "review_summary.md")
    print(f"[DONE] frames={summary['n_sampled_frames']} csv_positive={summary['n_csv_positive']} no_raw_csv={summary['n_no_raw_csv_supervision']} blink={summary['n_blink_labeled']} summary={output_root / 'review_summary.json'}")


if __name__ == "__main__":
    main()
