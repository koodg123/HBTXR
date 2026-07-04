#!/usr/bin/env python3
"""Build subject-independent test error tables for center-sequence models.

The evaluated models predict pupil centers only. For IoU reporting, this script
uses the ground-truth ellipse axes/angle and shifts that ellipse to the
predicted center. Therefore IoU is a center-localization proxy, not a predicted
ellipse-shape metric.
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

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
FACET_ROOT = ROOT / "references/codebase/software/FACET"
TENN_ROOT = ROOT / "references/codebase/software/ais2024/eye_track_spatiotemporal"
HBTXR_EVAL_SCRIPT = FACET_ROOT / "EvEye/utils/scripts/evaluate_hbtxr_val_motion.py"

if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("TennSt", "TENNs-Eye"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="FACET TennSt YAML or TENNs-Eye run config YAML.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv",
    )
    parser.add_argument(
        "--workbook-template",
        type=Path,
        default=ROOT / "analysis/RESULTS/JETCAS_REPLY_TABLES (Error-Distributions).xlsx",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def load_hbtxr_eval_module():
    spec = importlib.util.spec_from_file_location("hbtxr_eval_for_sequence", HBTXR_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {HBTXR_EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hbtxr_eval_for_sequence"] = module
    spec.loader.exec_module(module)
    return module


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_config(model_name: str) -> Path:
    if model_name == "TennSt":
        return FACET_ROOT / "configs/DavisEyeEllipse_TennSt_subject_independent_img64.yaml"
    return TENN_ROOT / "runs/TENNs_Eye_subject_independent_img64_20260630_010754/config.yaml"


def dataset_cfg_from_config(model_name: str, config: dict) -> dict:
    if model_name == "TennSt":
        cfg = dict(config["dataloader"]["val"]["dataset"])
        cfg.pop("type", None)
        cfg["split"] = "test"
        cfg["temporal_transform"] = False
        return cfg

    ds = config["dataset"]
    return {
        "root_path": Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent"),
        "split": "test",
        "frames_per_segment": int(ds.get("frames_per_segment", 50)),
        "stride": int(ds.get("frames_per_segment", 50)),
        "sensor_size": (346, 260, 2),
        "events_interpolation": ds.get("events_interpolation", "causal_linear_ori"),
        "pupil_area": 200,
        "default_resolution": tuple(ds.get("sensor_size", [64, 64])),
        "temporal_transform": False,
    }


def model_cfg_from_config(model_name: str, config: dict) -> dict:
    cfg = dict(config["model"])
    cfg.pop("type", None)
    if model_name == "TENNs-Eye":
        cfg.pop("activity_regularization", None)
    return cfg


def load_model(model_name: str, config: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if model_name == "TennSt":
        from EvEye.model.DavisEyeCenter.TennSt import TennSt

        model = TennSt(**model_cfg_from_config(model_name, config))
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    else:
        if str(TENN_ROOT) not in sys.path:
            sys.path.insert(0, str(TENN_ROOT))
        from tenn_model import TennSt

        model = TennSt(**model_cfg_from_config(model_name, config))
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def process_prediction(model_name: str, pred: torch.Tensor) -> torch.Tensor:
    if model_name == "TennSt":
        from EvEye.dataset.DavisEyeCenter.losses import process_detector_prediction

        return process_detector_prediction(pred)

    if str(TENN_ROOT) not in sys.path:
        sys.path.insert(0, str(TENN_ROOT))
    from losses import process_detector_prediction

    return process_detector_prediction(pred)


def ellipse_iou_center_proxy(pred_x64: float, pred_y64: float, gt_row_heatmap: np.ndarray) -> float:
    gt = gt_row_heatmap.astype(float).copy()
    if not np.isfinite(gt).all() or gt[2] <= 0 or gt[3] <= 0:
        return float("nan")
    gt[:4] *= 4.0
    pred = gt.copy()
    pred[0] = pred_x64
    pred[1] = pred_y64

    canvas_p = np.zeros((64, 64), dtype=np.uint8)
    canvas_g = np.zeros((64, 64), dtype=np.uint8)
    for canvas, row in ((canvas_p, pred), (canvas_g, gt)):
        x, y, a, b, angle = row
        if not np.isfinite(row).all() or a <= 0 or b <= 0:
            continue
        center = (int(round(np.clip(x, 0, 63))), int(round(np.clip(y, 0, 63))))
        axes = (
            max(1, int(round(np.clip(a / 2.0, 1, 64)))),
            max(1, int(round(np.clip(b / 2.0, 1, 64)))),
        )
        cv2.ellipse(canvas, center, axes, float(angle), 0, 360, 1, -1)
    union = np.logical_or(canvas_p, canvas_g).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(canvas_p, canvas_g).sum() / union)


def build_metadata(dataset_root: Path, output_dir: Path, prefix: str) -> pd.DataFrame:
    eval_mod = load_hbtxr_eval_module()
    return eval_mod.build_metadata(dataset_root, output_dir, "test", prefix)


def run_or_load_predictions(args: argparse.Namespace, config: dict, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = dataset_cfg_from_config(args.model, config)
    dataset_root = Path(cfg["root_path"])
    metadata = build_metadata(dataset_root, args.output_dir, prefix)
    pred_path = args.output_dir / f"{prefix}_sample_predictions.csv"
    if args.skip_inference and pred_path.exists():
        return metadata, pd.read_csv(pred_path)

    from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseCenterSequenceDataset import (
        DavisEyeEllipseCenterSequenceDataset,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cudnn.enabled = False

    dataset = DavisEyeEllipseCenterSequenceDataset(**cfg)
    if args.max_segments > 0:
        dataset.segments = dataset.segments[: args.max_segments]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_model(args.model, config, args.checkpoint, device)

    rows: list[dict] = []
    with torch.no_grad():
        for batch_idx, (event, center_norm, openness) in enumerate(
            tqdm(loader, desc=f"{args.model} test inference")
        ):
            event = event.to(device, non_blocking=True).float()
            center_norm = center_norm.cpu().numpy()
            openness_np = openness.cpu().numpy()
            pred = process_prediction(args.model, model(event)).detach().cpu().numpy()

            batch_size = pred.shape[0]
            for b in range(batch_size):
                start, end = dataset.segments[batch_idx * args.batch_size + b]
                for t, sample_idx in enumerate(range(start, end)):
                    gt_x64 = float(center_norm[b, 0, t] * 64.0)
                    gt_y64 = float(center_norm[b, 1, t] * 64.0)
                    pred_x64 = float(pred[b, 0, t] * 64.0)
                    pred_y64 = float(pred[b, 1, t] * 64.0)
                    valid = int(openness_np[b, t] == 1.0 and gt_x64 > 0 and gt_y64 > 0)
                    err = float(np.hypot(pred_x64 - gt_x64, pred_y64 - gt_y64)) if valid else np.nan
                    rows.append(
                        {
                            "sample_idx": int(sample_idx),
                            "segment_start": int(start),
                            "segment_frame_idx": int(t),
                            "valid": valid,
                            "gt_x_input64": gt_x64,
                            "gt_y_input64": gt_y64,
                            "pred_x_input64": pred_x64,
                            "pred_y_input64": pred_y64,
                            "error_input64_px": err,
                        }
                    )

    pred_df = pd.DataFrame(rows)
    # Convert original sensor-space pseudo-label ellipse to 64x64 input space,
    # then to heatmap scale to match the HBTXR helper convention.
    sensor_w, sensor_h = cfg.get("sensor_size", (346, 260, 2))[:2]
    res_h, res_w = cfg.get("default_resolution", (64, 64))
    gt_ellipse_heat = metadata.set_index("sample_idx")[["gt_x_orig", "gt_y_orig", "gt_a_orig", "gt_b_orig", "gt_ang"]].copy()
    gt_ellipse_heat["gt_x_orig"] = gt_ellipse_heat["gt_x_orig"] * (float(res_w) / float(sensor_w)) / 4.0
    gt_ellipse_heat["gt_y_orig"] = gt_ellipse_heat["gt_y_orig"] * (float(res_h) / float(sensor_h)) / 4.0
    gt_ellipse_heat["gt_a_orig"] = gt_ellipse_heat["gt_a_orig"] * (float(res_w) / float(sensor_w)) / 4.0
    gt_ellipse_heat["gt_b_orig"] = gt_ellipse_heat["gt_b_orig"] * (float(res_h) / float(sensor_h)) / 4.0
    ious = []
    for r in pred_df.itertuples(index=False):
        if not r.valid or r.sample_idx not in gt_ellipse_heat.index:
            ious.append(np.nan)
            continue
        gt = gt_ellipse_heat.loc[r.sample_idx].to_numpy(float)
        ious.append(ellipse_iou_center_proxy(r.pred_x_input64, r.pred_y_input64, gt))
    pred_df["iou_input64"] = ious
    pred_df["iou_note"] = "center_proxy_gt_axes_angle"
    pred_df.to_csv(pred_path, index=False)
    return metadata, pred_df


def join_motion_labels(pred_meta: pd.DataFrame, labels_path: Path, output_dir: Path, model_name: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_path).rename(columns={"subject": "user"})
    labels["session_code"] = labels["session_code"].astype(str)
    labels["frame_idx"] = labels["frame_idx"].astype(int)
    labels["user"] = labels["user"].astype(int)
    key = ["user", "eye", "session_code", "frame_idx"]
    dup = labels.duplicated(key, keep=False)
    if dup.any():
        labels.loc[dup].to_csv(output_dir / "duplicate_motion_labels.csv", index=False)
        raise RuntimeError(f"Duplicate motion-label join keys: {int(dup.sum())}")

    pred_meta = pred_meta.copy()
    pred_meta["session_code"] = pred_meta["session_code"].astype(str)
    pred_meta["frame_idx"] = pred_meta["frame_idx"].astype(int)
    pred_meta["user"] = pred_meta["user"].astype(int)
    joined = pred_meta.merge(
        labels[key + ["speed_pxps", "motion_state"]],
        on=key,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing = joined[joined["_merge"] != "both"]
    if len(missing):
        missing.to_csv(output_dir / f"{model_name}_subject37_48_dropped_blink_predictions.csv", index=False)
    joined = joined[joined["_merge"] == "both"].copy()
    if "motion_state_y" in joined.columns:
        joined["motion_state"] = joined["motion_state_y"]
        joined["motion_label_speed_pxps"] = joined["speed_pxps_y"]
        drop_cols = [c for c in ["motion_state_x", "motion_state_y", "speed_pxps_x", "speed_pxps_y"] if c in joined.columns]
        joined = joined.drop(columns=drop_cols)
    joined = joined.drop(columns=["_merge"])
    joined.to_csv(output_dir / f"{model_name}_subject37_48_test_joined_motion_error.csv", index=False)
    return joined


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


def aggregate(joined: pd.DataFrame, output_dir: Path, model_name: str) -> pd.DataFrame:
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
                "IoU_Note": "center_proxy_gt_axes_angle",
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / f"{model_name}_subject37_48_error_distribution_by_subject_motion.csv", index=False)
    joined.groupby(["user", "motion_state"]).size().unstack(fill_value=0).to_csv(
        output_dir / f"{model_name}_subject37_48_joined_motion_counts.csv"
    )
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


def ensure_row(sheet_data: ET.Element, row_idx: int) -> ET.Element:
    ns = {"x": XLS_NS}
    for row in sheet_data.findall("x:row", ns):
        if int(row.attrib["r"]) == row_idx:
            return row
    row = ET.Element(f"{{{XLS_NS}}}row", {"r": str(row_idx)})
    sheet_data.append(row)
    return row


def set_cell(row: ET.Element, col: str, row_idx: int, value: float | str) -> None:
    ref = f"{col}{row_idx}"
    ns = {"x": XLS_NS}
    cell = next((c for c in row.findall("x:c", ns) if c.attrib.get("r") == ref), None)
    if cell is None:
        cell = ET.Element(f"{{{XLS_NS}}}c", {"r": ref})
        target = col_to_idx(col)
        inserted = False
        for pos, old in enumerate(list(row)):
            old_col = "".join(ch for ch in old.attrib.get("r", "") if ch.isalpha())
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
    return target if target.startswith("xl/") else "xl/" + target


def update_workbook(template: Path, stats_df: pd.DataFrame, output_dir: Path, model_name: str) -> Path:
    workbook = output_dir / f"JETCAS_REPLY_TABLES (Error-Distributions)_{model_name}.xlsx"
    shutil.copy2(template, workbook)
    by_key = {(int(r.Subject), str(r.Motion)): r for r in stats_df.itertuples(index=False)}
    columns = {
        "Fixation": ("D", "E", "F", "G"),
        "Smooth": ("H", "I", "J", "K"),
        "Saccade": ("L", "M", "N", "O"),
        "Blink": ("P", "Q", "R", "S"),
    }
    metrics = ["Mean", "Median", "P95", "P99"]
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".xlsx", delete=False) as tmp_file:
        tmp = Path(tmp_file.name)
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
                values = [np.nan, np.nan, np.nan, np.nan] if rec is None else [getattr(rec, m) for m in metrics]
                for col, value in zip(columns[state], values):
                    set_cell(row, col, row_idx, value)
            for col in columns["Blink"]:
                set_cell(row, col, row_idx, "n/a")
        updated_xml = ET.tostring(sheet_xml, encoding="utf-8", xml_declaration=True)
        for item in zin.infolist():
            data = updated_xml if item.filename == sheet_path else zin.read(item.filename)
            zout.writestr(item, data)
    shutil.move(tmp, workbook)
    return workbook


def write_report(args: argparse.Namespace, config_path: Path, stats_df: pd.DataFrame, joined: pd.DataFrame, workbook: Path) -> Path:
    report = args.output_dir / f"{args.model}_subject37_48_error_distribution_report.md"
    total_valid = int(pd.to_numeric(stats_df["N"], errors="coerce").sum())
    overall_mean = float(np.average(stats_df["Mean"], weights=stats_df["N"]))
    overall_iou = float(np.average(stats_df["IoU_Mean"], weights=stats_df["N"]))
    lines = [
        f"# {args.model} Subject-Independent Test Error Distribution",
        "",
        "## Scope",
        "",
        f"- Model: `{args.model}`",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "- Split: `test`, subjects `37-48`.",
        f"- Device requested for inference: `{args.device}`.",
        "- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.",
        "- Pixel error is in 64x64 input coordinates.",
        "- IoU is `center_proxy_gt_axes_angle`: GT ellipse axes/angle shifted to the predicted center.",
        "",
        "## Summary",
        "",
        f"- Joined non-Blink rows: {len(joined):,}",
        f"- Valid error rows used in statistics: {total_valid:,}",
        f"- Weighted mean pixel error: {overall_mean:.4f}",
        f"- Weighted mean IoU proxy: {overall_iou:.4f}",
        f"- Workbook: `{workbook.name}`",
        "",
        "## Files",
        "",
        f"- `{args.model}_subject37_48_error_distribution_by_subject_motion.csv`",
        f"- `{args.model}_subject37_48_test_joined_motion_error.csv`",
        f"- `{args.model}_subject37_48_joined_motion_counts.csv`",
        f"- `{args.model}_subject37_48_dropped_blink_predictions.csv`",
        f"- `{workbook.name}`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    config_path = args.config or default_config(args.model)
    config = load_yaml(config_path)
    prefix = f"{args.model}_subject_independent_img64_test"
    metadata, pred = run_or_load_predictions(args, config, prefix)
    pred_meta = metadata.merge(pred, on="sample_idx", how="inner", validate="one_to_one")
    pred_meta.to_csv(args.output_dir / f"{args.model}_subject37_48_test_predictions_with_metadata.csv", index=False)
    joined = join_motion_labels(pred_meta, args.labels, args.output_dir, args.model)
    stats_df = aggregate(joined, args.output_dir, args.model)
    workbook = update_workbook(args.workbook_template, stats_df, args.output_dir, args.model)
    report = write_report(args, config_path, stats_df, joined, workbook)
    print(f"model={args.model}")
    print(f"joined_rows={len(joined)}")
    print(f"stats_rows={len(stats_df)}")
    print(f"workbook={workbook}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
