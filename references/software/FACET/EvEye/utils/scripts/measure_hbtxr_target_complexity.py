#!/usr/bin/env python3
"""Measure parameters and MACs for HBTXR target-training models.

The script profiles the HBTXR subject-independent 64x64 comparison setting.
It uses CPU dummy inputs and isolates each target in a subprocess so that
same-named modules from different upstream repositories do not conflict.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch
import yaml
from thop import profile


REPO_ROOT = Path(__file__).resolve().parents[7]
SOFTWARE_ROOT = REPO_ROOT / "references" / "codebase" / "software"
FACET_ROOT = SOFTWARE_ROOT / "FACET"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_model(model: torch.nn.Module, sample: torch.Tensor) -> tuple[int, int]:
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(sample,), verbose=False)
    return int(macs), int(params)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def build_facet_model(config_name: str) -> tuple[torch.nn.Module, torch.Tensor]:
    sys.path.insert(0, str(FACET_ROOT))
    from EvEye.model.model_factory import make_model

    cfg = read_yaml(FACET_ROOT / "configs" / config_name)
    model_cfg = dict(cfg["model"])
    model = make_model(model_cfg)
    if cfg["model"]["type"] == "TennSt":
        dataset_cfg = cfg["dataloader"]["train"]["dataset"]
        frames = int(dataset_cfg["frames_per_segment"])
        height, width = dataset_cfg["default_resolution"]
        sample = torch.randn(1, 2, frames, height, width)
    else:
        height, width = cfg["dataloader"]["train"]["dataset"]["default_resolution"]
        sample = torch.randn(1, int(cfg["model"].get("input_channels", 2)), height, width)
    return model, sample


def build_retina() -> tuple[torch.nn.Module, torch.Tensor]:
    root = SOFTWARE_ROOT / "retina"
    sys.path.insert(0, str(root))
    if "sinabs" not in sys.modules:
        sinabs = types.ModuleType("sinabs")
        activation = types.ModuleType("sinabs.activation")
        layers = types.ModuleType("sinabs.layers")

        class _UnavailableSinabsLayer(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError("sinabs is unavailable; only retina_ann can be profiled")

        activation.MultiSpike = object
        activation.SingleSpike = object
        activation.MembraneReset = object
        activation.MembraneSubtract = object
        activation.Heaviside = object
        activation.PeriodicExponential = object
        activation.SingleExponential = object
        layers.SumPool2d = _UnavailableSinabsLayer
        layers.IAFSqueeze = _UnavailableSinabsLayer
        sinabs.activation = activation
        sinabs.layers = layers
        sys.modules["sinabs"] = sinabs
        sys.modules["sinabs.activation"] = activation
        sys.modules["sinabs.layers"] = layers
    if "onnx" not in sys.modules:
        onnx = types.ModuleType("onnx")
        onnx.version_converter = types.ModuleType("onnx.version_converter")
        sys.modules["onnx"] = onnx
        sys.modules["onnx.version_converter"] = onnx.version_converter
    if "onnxruntime" not in sys.modules:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

    from engine.models.retina.helper import get_retina_model_configs
    from engine.models.retina.retina import Retina

    cfg = read_yaml(root / "configs" / "hbtxr_subject_independent_img64_patch4.yaml")
    training_params = cfg["training_params"]
    dataset_params = cfg["dataset_params"]
    quant_params = cfg["quant_params"]
    layers_config = get_retina_model_configs(dataset_params, training_params, quant_params)
    model = Retina(dataset_params, training_params, layers_config)
    sample = torch.randn(
        1,
        int(dataset_params["input_channel"]),
        int(dataset_params["img_height"]),
        int(dataset_params["img_width"]),
    )
    return model, sample


def build_tdtracker() -> tuple[torch.nn.Module, torch.Tensor]:
    root = SOFTWARE_ROOT / "ais2025" / "tdtracker"
    sys.path.insert(0, str(root))
    from argparse import Namespace
    from models.TDTracker import Model

    args = Namespace(sensor_width=64, sensor_height=64, spatial_factor=1.0)
    model = Model(args)
    sample = torch.randn(1, 100, 2, 64, 64)
    return model, sample


def build_ervt() -> tuple[torch.nn.Module, torch.Tensor]:
    root = SOFTWARE_ROOT / "ais2024" / "ERVT"
    sys.path.insert(0, str(root))
    from argparse import Namespace
    from model.RVT import RVT

    cfg = read_json(root / "configs" / "hbtxr_subject_independent_img64.json")
    args = Namespace(**cfg)
    model = RVT(args)
    sample = torch.randn(1, int(args.train_length), int(args.n_time_bins), 64, 64)
    return model, sample


def build_tenns_eye() -> tuple[torch.nn.Module, torch.Tensor]:
    root = SOFTWARE_ROOT / "ais2024" / "eye_track_spatiotemporal"
    sys.path.insert(0, str(root))
    from tenn_model import TennSt

    cfg = read_yaml(root / "config_hbtxr_subject_independent_img64.yaml")
    model = TennSt(**cfg["model"])
    frames = int(cfg["dataset"]["frames_per_segment"])
    height, width = cfg["dataset"]["sensor_size"]
    sample = torch.randn(1, 2, frames, height, width)
    return model, sample


def build_brat() -> tuple[torch.nn.Module, torch.Tensor]:
    root = SOFTWARE_ROOT / "ais2025" / "Event-based-Eye-Tracking-Challenge-Solution"
    sys.path.insert(0, str(root))
    from argparse import Namespace
    from model.CNN_GRU_base import Model

    cfg = read_json(root / "configs" / "hbtxr_subject_independent_img64.json")
    args = Namespace(**cfg)
    model = Model(args)
    sample = torch.randn(1, int(args.train_length), 2, 64, 64)
    return model, sample


BUILDERS = {
    "HBTXR": lambda: build_facet_model("DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml"),
    "EPNet_FECET": lambda: build_facet_model("DavisEyeEllipse_EPNet_subject_independent_img64.yaml"),
    "FACET_TennSt": lambda: build_facet_model("DavisEyeEllipse_TennSt_subject_independent_img64.yaml"),
    "Retina": build_retina,
    "TDTracker": build_tdtracker,
    "ERVT": build_ervt,
    "TENNs_Eye": build_tenns_eye,
    "BRAT": build_brat,
}


def measure_one(target: str) -> dict[str, str]:
    model, sample = BUILDERS[target]()
    total, trainable = count_params(model)
    row = {
        "target": target,
        "input_shape": "x".join(str(x) for x in sample.shape),
        "params": str(total),
        "trainable_params": str(trainable),
        "macs": "",
        "flops": "",
        "status": "ok",
        "note": "",
    }
    try:
        macs, _ = profile_model(model, sample)
        row["macs"] = str(macs)
        row["flops"] = str(macs * 2)
    except Exception as exc:  # pragma: no cover - diagnostic path
        row["status"] = "params_only"
        row["note"] = f"{type(exc).__name__}: {exc}"
    return row


def write_outputs(rows: list[dict[str, str]], output_csv: Path, output_md: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["target", "input_shape", "params", "trainable_params", "macs", "flops", "status", "note"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# HBTXR Target Model Complexity",
        "",
        "Scope: HBTXR subject-independent 64x64 comparison setting.",
        "",
        "| Target | Input Shape | Params | Trainable Params | MACs | FLOPs | Status | Note |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {input_shape} | {params} | {trainable_params} | {macs} | {flops} | {status} | {note} |".format(
                **{k: row.get(k, "") for k in fields}
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- MACs are measured with `thop.profile` on CPU dummy inputs.",
            "- FLOPs are reported as `2 * MACs` for multiply-add operations.",
            "- `EPNet_FECET` is used because no separate `FECET` model path was found in `references/codebase/software`.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")


def run_all(args: argparse.Namespace) -> int:
    rows = []
    for target in BUILDERS:
        cmd = [sys.executable, __file__, "--target", target]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            rows.append(
                {
                    "target": target,
                    "input_shape": "",
                    "params": "",
                    "trainable_params": "",
                    "macs": "",
                    "flops": "",
                    "status": "failed",
                    "note": completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown error",
                }
            )
            continue
        json_line = completed.stdout.strip().splitlines()[-1]
        rows.append(json.loads(json_line))
    write_outputs(rows, Path(args.output_csv), Path(args.output_md))
    print(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(BUILDERS))
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "references" / "report" / "HBTXR_target_model_complexity_2026-06-30.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "references" / "report" / "HBTXR_target_model_complexity_2026-06-30.md"),
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.target:
        print(json.dumps(measure_one(args.target)))
        return 0
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
