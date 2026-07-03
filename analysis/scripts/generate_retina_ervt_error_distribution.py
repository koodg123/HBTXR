#!/usr/bin/env python3
"""Generate subject37-48 pixel-error distribution packages for Retina and ERVT."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
FACET_ROOT = REPO_ROOT / "references" / "codebase" / "software" / "FACET"
RETINA_ROOT = REPO_ROOT / "references" / "codebase" / "software" / "retina"
ERVT_ROOT = REPO_ROOT / "references" / "codebase" / "software" / "ais2024" / "ERVT"
DATASET_ROOT = Path("/mnt/d/dataset/EV_Eye/target_data/DeanDataset_full_unet_subject_independent")
HBTXR_PACKAGE = REPO_ROOT / "analysis" / "results" / "HBTXR"
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "hbtxr_matplotlib"))
DEFAULT_MOTION_LABEL_MAP = HBTXR_PACKAGE / "HBTXR_subject37_48_test_joined_motion_error.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=("Retina", "ERVT"), default=["Retina", "ERVT"])
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "analysis" / "results")
    parser.add_argument("--metadata", type=Path, default=HBTXR_PACKAGE / "HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv")
    parser.add_argument("--motion-label-map", type=Path, default=DEFAULT_MOTION_LABEL_MAP)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--retina-ckpt", type=Path, default=REPO_ROOT / "analysis" / "results" / "Retina" / "checkpoints" / "epoch=66-val_loss=2.8817.ckpt")
    parser.add_argument("--retina-config", type=Path, default=RETINA_ROOT / "configs" / "hbtxr_subject_independent_img64_patch4.yaml")
    parser.add_argument("--ervt-ckpt", type=Path, default=REPO_ROOT / "analysis" / "results" / "ERVT" / "checkpoints" / "best_epoch012_val_distance_8.5869.pth")
    parser.add_argument("--ervt-config", type=Path, default=ERVT_ROOT / "configs" / "hbtxr_subject_independent_img64.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def install_retina_optional_dependency_stubs() -> None:
    def _stub_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return module

    if "sinabs" not in sys.modules:
        sinabs = _stub_module("sinabs")
        activation = _stub_module("sinabs.activation")
        layers = _stub_module("sinabs.layers")
        exodus = _stub_module("sinabs.exodus")

        class _DummyAnalyzer:
            def __init__(self, *args, **kwargs):
                pass

            def get_model_statistics(self):
                return {}

            def get_layer_statistics(self):
                return {"parameter": {}, "spiking": {}}

        class _UnavailableLayer(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError("sinabs layers are unavailable for retina_ann evaluation")

        sinabs.SNNAnalyzer = _DummyAnalyzer
        activation.MultiSpike = object
        activation.SingleSpike = object
        activation.MembraneReset = object
        activation.MembraneSubtract = object
        activation.Heaviside = object
        activation.PeriodicExponential = object
        activation.SingleExponential = object
        layers.SumPool2d = _UnavailableLayer
        layers.IAFSqueeze = _UnavailableLayer
        sinabs.activation = activation
        sinabs.layers = layers
        sinabs.exodus = exodus
        sys.modules["sinabs"] = sinabs
        sys.modules["sinabs.activation"] = activation
        sys.modules["sinabs.layers"] = layers
        sys.modules["sinabs.exodus"] = exodus

    if "onnx" not in sys.modules:
        onnx = _stub_module("onnx")
        onnx.version_converter = _stub_module("onnx.version_converter")
        sys.modules["onnx"] = onnx
        sys.modules["onnx.version_converter"] = onnx.version_converter

    if "onnxruntime" not in sys.modules:
        onnxruntime = _stub_module("onnxruntime")

        class _UnavailableInferenceSession:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("onnxruntime is unavailable for Retina ANN evaluation")

        onnxruntime.InferenceSession = _UnavailableInferenceSession
        sys.modules["onnxruntime"] = onnxruntime


def base_prediction_row(sample_idx: int, valid: bool, gt_x64: float, gt_y64: float, pred_x64: float, pred_y64: float, pred_score: float = math.nan) -> dict:
    if valid:
        err64 = float(math.hypot(pred_x64 - gt_x64, pred_y64 - gt_y64))
        err_heat = err64 / 4.0
    else:
        err64 = math.nan
        err_heat = math.nan

    return {
        "sample_idx": int(sample_idx),
        "valid": int(valid),
        "gt_x_heatmap": float(gt_x64 / 4.0),
        "gt_y_heatmap": float(gt_y64 / 4.0),
        "pred_x_heatmap": float(pred_x64 / 4.0),
        "pred_y_heatmap": float(pred_y64 / 4.0),
        "error_heatmap_px": err_heat,
        "error_input64_px": err64,
        "pred_score": pred_score,
        "gt_a_heatmap": math.nan,
        "gt_b_heatmap": math.nan,
        "gt_ang": math.nan,
        "pred_a_heatmap": math.nan,
        "pred_b_heatmap": math.nan,
        "pred_ang": math.nan,
        "iou_input64": math.nan,
    }


def evaluate_retina(args: argparse.Namespace, metadata: pd.DataFrame) -> pd.DataFrame:
    install_retina_optional_dependency_stubs()
    sys.path.insert(0, str(RETINA_ROOT))
    sys.path.insert(0, str(FACET_ROOT))

    from data.datasets.hbtxr_dean.hbtxr_dean_dataset import HBTXRDeanDataset  # noqa: PLC0415
    from engine.models.retina.helper import get_retina_model_configs  # noqa: PLC0415
    from engine.models.retina.retina import Retina  # noqa: PLC0415
    from engine.module import EyeTrackingModelModule  # noqa: PLC0415

    cfg = read_yaml(args.retina_config)
    training_params = dict(cfg["training_params"])
    dataset_params = dict(cfg["dataset_params"])
    quant_params = dict(cfg["quant_params"])
    training_params["batch_size"] = args.batch_size
    training_params["num_workers"] = args.num_workers
    training_params["out_dir"] = str(args.output_root / "Retina")

    layers_config = get_retina_model_configs(dataset_params, training_params, quant_params)
    model = Retina(dataset_params, training_params, layers_config)
    module = EyeTrackingModelModule(model, dataset_params, training_params)
    ckpt = torch.load(args.retina_ckpt, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"])

    device = resolve_device(args.device)
    module.to(device)
    module.eval()

    dataset = HBTXRDeanDataset("test", training_params, dataset_params)
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    rows = []
    offset = 0
    width = float(dataset_params["img_width"])
    height = float(dataset_params["img_height"])
    with torch.no_grad():
        for events, labels, _avg_dt, _exp_id in tqdm(loader, desc="Retina test inference"):
            events = events.to(device, non_blocking=True).float()
            labels = labels.cpu()
            outputs = module.forward(events).detach().cpu().clamp(0.0, 1.0)
            bs = outputs.shape[0]
            pred_x64 = ((outputs[:, 0] + outputs[:, 2]) * 0.5 * width).numpy()
            pred_y64 = ((outputs[:, 1] + outputs[:, 3]) * 0.5 * height).numpy()
            gt_x64 = ((labels[:, 0] + labels[:, 2]) * 0.5 * width).numpy()
            gt_y64 = ((labels[:, 1] + labels[:, 3]) * 0.5 * height).numpy()
            for i in range(bs):
                sample_idx = offset + i
                meta_valid = True
                if sample_idx < len(metadata):
                    meta_valid = float(metadata.iloc[sample_idx]["gt_x_orig"]) > 0 or float(metadata.iloc[sample_idx]["gt_y_orig"]) > 0
                valid = bool(meta_valid and np.isfinite(gt_x64[i]) and np.isfinite(gt_y64[i]))
                rows.append(base_prediction_row(sample_idx, valid, gt_x64[i], gt_y64[i], pred_x64[i], pred_y64[i]))
            offset += bs

    return pd.DataFrame(rows)


def evaluate_ervt(args: argparse.Namespace, metadata: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(ERVT_ROOT))
    sys.path.insert(0, str(FACET_ROOT))

    from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseCenterSequenceDataset import DavisEyeEllipseCenterSequenceDataset  # noqa: PLC0415
    from model.RVT import RVT  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    cfg = read_json(args.ervt_config)
    cfg["in_channels"] = 2
    cfg["device"] = args.device
    device = resolve_device(args.device)
    model = RVT(SimpleNamespace(**cfg)).to(device)
    ckpt = torch.load(args.ervt_ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    dataset = DavisEyeEllipseCenterSequenceDataset(
        root_path=DATASET_ROOT,
        split="test",
        frames_per_segment=int(cfg.get("test_length", cfg["val_length"])),
        stride=int(cfg.get("test_stride", cfg["val_stride"])),
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear_ori",
        pupil_area=200,
        default_resolution=(int(cfg["sensor_width"]), int(cfg["sensor_height"])),
        temporal_transform=False,
    )
    if args.max_samples > 0:
        max_segments = max(1, math.ceil(args.max_samples / int(cfg.get("test_length", cfg["val_length"]))))
        dataset = torch.utils.data.Subset(dataset, range(min(max_segments, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    rows = []
    segment_cursor = 0
    width = float(cfg["sensor_width"])
    height = float(cfg["sensor_height"])
    with torch.no_grad():
        for events, centers, _openness in tqdm(loader, desc="ERVT test inference"):
            events = events.moveaxis(1, 2).to(device, non_blocking=True).float()
            targets = centers.moveaxis(1, 2).cpu().float()
            outputs, _hidden = model(events)
            outputs = outputs.detach().cpu()
            bs, seq_len = outputs.shape[:2]
            for b in range(bs):
                if isinstance(dataset, torch.utils.data.Subset):
                    start, _end = dataset.dataset.segments[dataset.indices[segment_cursor + b]]
                else:
                    start, _end = dataset.segments[segment_cursor + b]
                for t in range(seq_len):
                    sample_idx = start + t
                    if args.max_samples > 0 and sample_idx >= args.max_samples:
                        continue
                    gt_x64 = float(targets[b, t, 0] * width)
                    gt_y64 = float(targets[b, t, 1] * height)
                    pred_x64 = float(outputs[b, t, 0] * width)
                    pred_y64 = float(outputs[b, t, 1] * height)
                    meta_valid = True
                    if sample_idx < len(metadata):
                        meta_valid = float(metadata.iloc[sample_idx]["gt_x_orig"]) > 0 or float(metadata.iloc[sample_idx]["gt_y_orig"]) > 0
                    valid = bool(meta_valid and np.isfinite(gt_x64) and np.isfinite(gt_y64))
                    rows.append(base_prediction_row(sample_idx, valid, gt_x64, gt_y64, pred_x64, pred_y64))
            segment_cursor += bs

    df = pd.DataFrame(rows)
    return df.drop_duplicates("sample_idx", keep="first").sort_values("sample_idx").reset_index(drop=True)


def describe(values: pd.Series) -> dict:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return {"N": 0, "Mean": math.nan, "Median": math.nan, "P95": math.nan, "P99": math.nan}
    return {
        "N": int(len(vals)),
        "Mean": float(np.mean(vals)),
        "Median": float(np.median(vals)),
        "P95": float(np.percentile(vals, 95)),
        "P99": float(np.percentile(vals, 99)),
    }


def read_motion_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, usecols=["sample_idx", "motion_state", "motion_label_speed_pxps"])
    return labels.drop_duplicates("sample_idx", keep="first")


def write_distribution_package(
    model_name: str,
    out_dir: Path,
    metadata: pd.DataFrame,
    pred: pd.DataFrame,
    motion_labels: pd.DataFrame,
    motion_label_path: Path,
    overwrite: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_subject37_48"
    pred_path = out_dir / f"{prefix}_test_sample_predictions.csv"
    if pred_path.exists() and not overwrite:
        raise FileExistsError(f"{pred_path} exists; pass --overwrite")
    pred.to_csv(pred_path, index=False)

    metadata = metadata.copy()
    if "gt_ang" in metadata.columns:
        metadata = metadata.rename(columns={"gt_ang": "gt_ang_meta"})
    merged = metadata.merge(pred, on="sample_idx", how="inner")
    merged.to_csv(out_dir / f"{prefix}_test_predictions_with_metadata.csv", index=False)

    joined_base = merged.drop(columns=[c for c in ("motion_state", "speed_pxps") if c in merged.columns])
    joined = joined_base.merge(motion_labels, on="sample_idx", how="inner")
    joined.to_csv(out_dir / f"{prefix}_test_joined_motion_error.csv", index=False)

    full_join = joined_base.merge(motion_labels, on="sample_idx", how="outer", indicator=True)
    full_join.loc[full_join["_merge"] != "both"].to_csv(out_dir / f"{prefix}_unmatched_predictions.csv", index=False)
    full_join.loc[full_join["_merge"] != "both"].to_csv(
        out_dir / f"{prefix}_dropped_blink_predictions.csv",
        index=False,
    )

    counts = joined.groupby(["user", "motion_state"]).size().unstack(fill_value=0).reset_index()
    for col in ("Fixation", "Saccade", "Smooth"):
        if col not in counts:
            counts[col] = 0
    counts[["user", "Fixation", "Saccade", "Smooth"]].to_csv(out_dir / f"{prefix}_joined_motion_counts.csv", index=False)

    rows = []
    for (subject, motion), group in joined.groupby(["user", "motion_state"], dropna=False):
        stats = describe(group["error_input64_px"])
        rows.append(
            {
                "Subject": int(subject),
                "Split": "Test",
                "Motion": motion,
                **stats,
                "IoU_Mean": float(group["iou_input64"].mean()) if group["iou_input64"].notna().any() else math.nan,
                "IoU_Median": float(group["iou_input64"].median()) if group["iou_input64"].notna().any() else math.nan,
            }
        )
    dist = pd.DataFrame(rows).sort_values(["Subject", "Motion"])
    dist.to_csv(out_dir / f"{prefix}_error_distribution_by_subject_motion.csv", index=False)

    status = [
        f"# {model_name} Subject37-48 Pixel Error Distribution",
        "",
        "Status: generated.",
        "",
        f"- Motion label map: {motion_label_path.relative_to(REPO_ROOT) if motion_label_path.is_relative_to(REPO_ROOT) else motion_label_path}",
        f"- Prediction rows: {len(pred):,}",
        f"- Joined metadata rows: {len(joined):,}",
        f"- Unmatched rows: {int((full_join['_merge'] != 'both').sum()):,}",
        f"- Valid joined error rows: {int(joined['error_input64_px'].notna().sum()):,}",
    ]
    overall = describe(joined["error_input64_px"])
    status.extend(
        [
            f"- Mean error: {overall['Mean']:.6f} input64 px",
            f"- Median error: {overall['Median']:.6f} input64 px",
            f"- P95 error: {overall['P95']:.6f} input64 px",
            f"- P99 error: {overall['P99']:.6f} input64 px",
            "",
            "Files follow the HBTXR subject37-48 package naming convention.",
            "",
        ]
    )
    (out_dir / "STATUS.md").write_text("\n".join(status), encoding="utf-8")


def main() -> int:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    motion_labels = read_motion_labels(args.motion_label_map)
    if args.max_samples > 0:
        metadata = metadata.iloc[: args.max_samples].copy()
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    for model_name in args.models:
        out_dir = args.output_root / model_name
        if model_name == "Retina":
            pred = evaluate_retina(args, metadata)
        elif model_name == "ERVT":
            pred = evaluate_ervt(args, metadata)
        else:
            raise AssertionError(model_name)
        write_distribution_package(model_name, out_dir, metadata, pred, motion_labels, args.motion_label_map, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
