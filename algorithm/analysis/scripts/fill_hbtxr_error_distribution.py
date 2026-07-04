#!/usr/bin/env python3
"""Fill JETCAS Error-Distributions table from HBTXR test inference.

The workbook is edited through the XLSX zip/XML files so this script does not
depend on openpyxl. Blink is intentionally excluded from the error table.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FACET_ROOT = ROOT / "references/codebase/software/FACET"
EVAL_SCRIPT = FACET_ROOT / "EvEye/utils/scripts/evaluate_hbtxr_val_motion.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "references/codebase/software/FACET/runs/logs/HBTXR_subject_independent_img64_patch4/version_0/checkpoints/epoch=66-val_mean_distance=0.5401.ckpt",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT / "analysis/RESULTS/JETCAS_REPLY_TABLES (Error-Distributions).xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/RESULTS/HBTXR_subject37_48_error_distribution",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def load_eval_module():
    spec = importlib.util.spec_from_file_location("hbtxr_eval", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hbtxr_eval"] = module
    spec.loader.exec_module(module)
    return module


def describe(values: pd.Series) -> dict[str, float | int]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(arr) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p95": np.nan, "p99": np.nan}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def build_or_load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_mod = load_eval_module()
    config = eval_mod.load_yaml(args.config)
    prefix = eval_mod.output_prefix("HBTXR_subject_independent_img64_patch4", "test")
    dataset_root = Path(config["dataloader"]["val"]["dataset"]["root_path"])
    metadata = eval_mod.build_metadata(dataset_root, args.output_dir, "test", prefix)
    if args.skip_inference:
        pred = pd.read_csv(args.output_dir / f"{prefix}_sample_predictions.csv")
    else:
        pred = eval_mod.run_inference(
            config=config,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=0,
            split="test",
            prefix=prefix,
        )
    merged = metadata.merge(pred, on="sample_idx", how="inner", validate="one_to_one")
    merged.to_csv(args.output_dir / "HBTXR_subject37_48_test_predictions_with_metadata.csv", index=False)
    return merged


def join_motion_labels(pred: pd.DataFrame, labels_path: Path, output_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    labels = labels.rename(columns={"subject": "user"})
    labels["session_code"] = labels["session_code"].astype(str)
    labels["frame_idx"] = labels["frame_idx"].astype(int)
    labels["user"] = labels["user"].astype(int)
    key = ["user", "eye", "session_code", "frame_idx"]
    dup = labels.duplicated(key, keep=False)
    if dup.any():
        labels.loc[dup].to_csv(output_dir / "duplicate_motion_labels.csv", index=False)
        raise RuntimeError(f"Duplicate motion-label join keys: {int(dup.sum())}")

    pred = pred.copy()
    pred["session_code"] = pred["session_code"].astype(str)
    pred["frame_idx"] = pred["frame_idx"].astype(int)
    pred["user"] = pred["user"].astype(int)
    joined = pred.merge(
        labels[key + ["speed_pxps", "motion_state"]],
        on=key,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = joined[joined["_merge"] != "both"]
    if len(missing):
        # These are expected when Blink is intentionally excluded from the
        # error-distribution table. Keep them for audit, then aggregate only
        # the NoBlink matched rows.
        missing.to_csv(output_dir / "HBTXR_subject37_48_dropped_blink_predictions.csv", index=False)
    joined = joined[joined["_merge"] == "both"].copy()
    if "motion_state_y" in joined.columns:
        joined["motion_state"] = joined["motion_state_y"]
        joined["motion_label_speed_pxps"] = joined["speed_pxps_y"]
        drop_cols = [c for c in ["motion_state_x", "motion_state_y", "speed_pxps_x", "speed_pxps_y"] if c in joined.columns]
        joined = joined.drop(columns=drop_cols)
    joined = joined.drop(columns=["_merge"])
    joined.to_csv(output_dir / "HBTXR_subject37_48_test_joined_motion_error.csv", index=False)
    return joined


def aggregate(joined: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for (user, state), group in joined.groupby(["user", "motion_state"], sort=True):
        stats = describe(group["error_input64_px"])
        rows.append(
            {
                "Subject": int(user),
                "Split": "Test",
                "Motion": state,
                "N": stats["n"],
                "Mean": stats["mean"],
                "Median": stats["median"],
                "P95": stats["p95"],
                "P99": stats["p99"],
                "IoU_Mean": float(pd.to_numeric(group["iou_input64"], errors="coerce").mean()),
                "IoU_Median": float(pd.to_numeric(group["iou_input64"], errors="coerce").median()),
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / "HBTXR_subject37_48_error_distribution_by_subject_motion.csv", index=False)

    counts = joined.groupby(["user", "motion_state"]).size().unstack(fill_value=0)
    counts.to_csv(output_dir / "HBTXR_subject37_48_joined_motion_counts.csv")
    return stats_df


XLS_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("", XLS_NS)


def col_to_idx(col: str) -> int:
    out = 0
    for ch in col:
        out = out * 26 + ord(ch.upper()) - ord("A") + 1
    return out


def cell_ref(col: str, row: int) -> str:
    return f"{col}{row}"


def ensure_row(sheet_data: ET.Element, row_idx: int) -> ET.Element:
    ns = {"x": XLS_NS}
    for row in sheet_data.findall("x:row", ns):
        if int(row.attrib["r"]) == row_idx:
            return row
    row = ET.Element(f"{{{XLS_NS}}}row", {"r": str(row_idx)})
    inserted = False
    for pos, old in enumerate(list(sheet_data)):
        if old.tag.endswith("row") and int(old.attrib["r"]) > row_idx:
            sheet_data.insert(pos, row)
            inserted = True
            break
    if not inserted:
        sheet_data.append(row)
    return row


def set_cell(row: ET.Element, col: str, row_idx: int, value: float | str) -> None:
    ref = cell_ref(col, row_idx)
    ns = {"x": XLS_NS}
    cell = None
    for c in row.findall("x:c", ns):
        if c.attrib.get("r") == ref:
            cell = c
            break
    if cell is None:
        cell = ET.Element(f"{{{XLS_NS}}}c", {"r": ref})
        target = col_to_idx(col)
        inserted = False
        for pos, old in enumerate(list(row)):
            old_ref = old.attrib.get("r", "")
            old_col = "".join(ch for ch in old_ref if ch.isalpha())
            if old_col and col_to_idx(old_col) > target:
                row.insert(pos, cell)
                inserted = True
                break
        if not inserted:
            row.append(cell)

    for child in list(cell):
        cell.remove(child)
    if isinstance(value, str):
        cell.attrib["t"] = "inlineStr"
        is_el = ET.SubElement(cell, f"{{{XLS_NS}}}is")
        t_el = ET.SubElement(is_el, f"{{{XLS_NS}}}t")
        t_el.text = value
    else:
        cell.attrib.pop("t", None)
        v_el = ET.SubElement(cell, f"{{{XLS_NS}}}v")
        v_el.text = "" if pd.isna(value) else f"{float(value):.6f}"


def workbook_sheet_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    ns = {"x": XLS_NS, "r": REL_NS, "pr": PKG_REL_NS}
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    target_rid = None
    for sheet in wb.findall(".//x:sheet", ns):
        if sheet.attrib.get("name") == sheet_name:
            target_rid = sheet.attrib[f"{{{REL_NS}}}id"]
            break
    if target_rid is None:
        raise RuntimeError(f"Workbook sheet not found: {sheet_name}")
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    target = None
    for rel in rels.findall("pr:Relationship", ns):
        if rel.attrib.get("Id") == target_rid:
            target = rel.attrib["Target"]
            break
    if target is None:
        raise RuntimeError(f"Sheet relationship not found: {target_rid}")
    target = target.lstrip("/")
    if not target.startswith("xl/"):
        target = "xl/" + target
    return target


def update_workbook(workbook: Path, stats_df: pd.DataFrame, output_dir: Path) -> Path:
    backup = workbook.with_suffix(".backup_before_hbtxr_fill.xlsx")
    if not backup.exists():
        shutil.copy2(workbook, backup)

    by_key = {(int(r.Subject), str(r.Motion)): r for r in stats_df.itertuples(index=False)}
    columns = {
        "Fixation": ("D", "E", "F", "G"),
        "Smooth": ("H", "I", "J", "K"),
        "Saccade": ("L", "M", "N", "O"),
        "Blink": ("P", "Q", "R", "S"),
    }
    metrics = ["Mean", "Median", "P95", "P99"]

    tmp = output_dir / f"{workbook.stem}.tmp.xlsx"
    with zipfile.ZipFile(workbook, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        sheet_path = workbook_sheet_path(zin, "Error Distribution")
        sheet_xml = ET.fromstring(zin.read(sheet_path))
        ns = {"x": XLS_NS}
        sheet_data = sheet_xml.find("x:sheetData", ns)
        if sheet_data is None:
            raise RuntimeError("sheetData not found")
        for row_idx, subject in enumerate(range(37, 49), start=6):
            row = ensure_row(sheet_data, row_idx)
            for state in ("Fixation", "Smooth", "Saccade"):
                rec = by_key.get((subject, state))
                if rec is None:
                    values = [np.nan, np.nan, np.nan, np.nan]
                else:
                    values = [getattr(rec, m) for m in metrics]
                for col, value in zip(columns[state], values):
                    set_cell(row, col, row_idx, value)
            for col in columns["Blink"]:
                set_cell(row, col, row_idx, "n/a")

        updated_xml = ET.tostring(sheet_xml, encoding="utf-8", xml_declaration=True)
        for item in zin.infolist():
            data = updated_xml if item.filename == sheet_path else zin.read(item.filename)
            zout.writestr(item, data)

    shutil.move(tmp, workbook)
    return backup


def main() -> None:
    args = parse_args()
    pred = build_or_load_predictions(args)
    joined = join_motion_labels(pred, args.labels, args.output_dir)
    stats_df = aggregate(joined, args.output_dir)
    backup = update_workbook(args.workbook, stats_df, args.output_dir)
    print(f"joined_rows={len(joined)}")
    print(f"stats_rows={len(stats_df)}")
    print(f"updated_workbook={args.workbook}")
    print(f"backup={backup}")


if __name__ == "__main__":
    main()
