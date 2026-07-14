from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from _bootstrap import PROJECT_ROOT
from facet_common import import_facet_module, load_facet_config, patch_facet_config, resolve_facet_repo_root, torch_device_from_arg

from fecet_hbtxr.io import write_json, write_jsonl


def _flatten_sensor_ellipse(ellipse: tuple[tuple[float, float], tuple[float, float], float]) -> list[float]:
    return [float(ellipse[0][0]), float(ellipse[0][1]), float(ellipse[1][0]), float(ellipse[1][1]), float(ellipse[2])]


def _load_checkpoint_state(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return dict(payload["state_dict"])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Unsupported FACET checkpoint format: {path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FACET EPNet inference and export per-sample predictions")
    parser.add_argument("--facet-root", type=str, default=str(resolve_facet_repo_root()))
    parser.add_argument("--config", type=str, default="DavisEyeEllipse_EPNet.yaml")
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "data" / "facet_reference"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-jsonl", type=str, default=str(PROJECT_ROOT / "runs" / "comparison" / "facet" / "facet_infer_predictions.jsonl"))
    parser.add_argument("--output-summary", type=str, default=str(PROJECT_ROOT / "runs" / "comparison" / "facet" / "facet_infer_summary.json"))
    parser.add_argument("--sensor-width", type=int, default=346)
    parser.add_argument("--sensor-height", type=int, default=240)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--input-height", type=int, default=256)
    return parser


def run(args: argparse.Namespace) -> dict:
    facet_root = resolve_facet_repo_root(args.facet_root)
    cfg = load_facet_config(args.config, facet_repo_root=facet_root)
    cfg = patch_facet_config(
        cfg,
        dataset_root=args.dataset_root,
        output_root=Path(args.output_summary).resolve().parent,
        experiment_name="facet_infer",
        sensor_size_whc=(int(args.sensor_width), int(args.sensor_height), 2),
        default_resolution=(int(args.input_width), int(args.input_height)),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_epochs=None,
        checkpoint_path=args.checkpoint,
    )

    dataset_factory = import_facet_module(facet_root, "EvEye.dataset.dataset_factory")
    model_factory = import_facet_module(facet_root, "EvEye.model.model_factory")
    predict_module = import_facet_module(facet_root, "EvEye.model.DavisEyeEllipse.EPNet.Predict")

    dataset_cfg = dict((cfg.get("dataloader") or {}).get("val", {}).get("dataset", {}))
    dataset_cfg["root_path"] = str(Path(args.dataset_root).resolve())
    dataset_cfg["split"] = str(args.split)
    dataset = dataset_factory.make_dataset(dataset_cfg)
    loader = DataLoader(dataset=dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    model = model_factory.make_model(dict(cfg["model"]))
    device = torch_device_from_arg(args.device)
    model.load_state_dict(_load_checkpoint_state(args.checkpoint, device), strict=False)
    model.eval()
    model.to(device)

    rows: list[dict[str, Any]] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            outputs = model(inputs)
            dets = predict_module.post_process(outputs)
            batch_size = int(inputs.shape[0])
            for item_index in range(batch_size):
                if args.limit is not None and seen >= int(args.limit):
                    break
                angle_deg = float(dets["ang"][item_index, 0].detach().cpu().item()) - 90.0
                pred_ellipse_64 = [
                    float(dets["xs"][item_index, 0].detach().cpu().item()),
                    float(dets["ys"][item_index, 0].detach().cpu().item()),
                    float(dets["ab"][item_index, 0].detach().cpu().item()),
                    float(dets["ab"][item_index, 1].detach().cpu().item()),
                    angle_deg,
                ]
                pred_sensor = predict_module.transform_ellipse(
                    ((pred_ellipse_64[0], pred_ellipse_64[1]), (pred_ellipse_64[2], pred_ellipse_64[3]), pred_ellipse_64[4]),
                    orig_size=(64, 64),
                    target_size=(int(args.sensor_height), int(args.sensor_width)),
                )
                target_ellipse = batch["ellipse"][item_index].detach().cpu().tolist()
                target_center = batch["center"][item_index].detach().cpu().tolist()
                rows.append(
                    {
                        "sample_index": seen,
                        "split": str(args.split),
                        "pred_score": float(dets["scores"][item_index, 0].detach().cpu().item()),
                        "pred_center_64": [pred_ellipse_64[0], pred_ellipse_64[1]],
                        "pred_ellipse_64_xyabang": pred_ellipse_64,
                        "pred_ellipse_sensor_xyabang": _flatten_sensor_ellipse(pred_sensor),
                        "target_center_64": [float(target_center[0]), float(target_center[1])],
                        "target_ellipse_64_xyabang": [float(value) for value in target_ellipse],
                        "close_flag": int(batch["close"][item_index].detach().cpu().item()),
                    }
                )
                seen += 1
            if args.limit is not None and seen >= int(args.limit):
                break

    write_jsonl(rows, args.output_jsonl)
    summary = {
        "facet_root": str(facet_root),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": str(args.split),
        "row_count": len(rows),
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
    }
    write_json(summary, args.output_summary)
    summary["output_summary"] = str(Path(args.output_summary).resolve())
    return summary


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
