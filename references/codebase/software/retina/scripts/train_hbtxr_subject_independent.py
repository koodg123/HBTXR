#!/usr/bin/env python3
"""Train Retina ANN on the HBTXR subject-independent 64x64 dataset.

This runner avoids the upstream Retina script's optional wandb and ONNX export
paths so the HBTXR comparison can run in the existing FACET training venv.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def install_optional_dependency_stubs() -> None:
    """Provide minimal stubs for optional Retina ANN-unused dependencies."""

    if "sinabs" not in sys.modules:
        sinabs = types.ModuleType("sinabs")
        activation = types.ModuleType("sinabs.activation")
        layers = types.ModuleType("sinabs.layers")
        exodus = types.ModuleType("sinabs.exodus")

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
                raise RuntimeError("sinabs layers are unavailable in retina_ann runner")

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
        onnx = types.ModuleType("onnx")
        onnx.version_converter = types.ModuleType("onnx.version_converter")
        sys.modules["onnx"] = onnx
        sys.modules["onnx.version_converter"] = onnx.version_converter

    if "onnxruntime" not in sys.modules:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")


def read_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[5]
    retina_root = repo_root / "references" / "codebase" / "software" / "retina"
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--retina-root", type=Path, default=retina_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=retina_root / "configs" / "hbtxr_subject_independent_img64_patch4.yaml",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--run-name", default="Retina_subject_independent_img64")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=retina_root / "runs",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def main() -> int:
    install_optional_dependency_stubs()
    args = build_parser().parse_args()

    sys.path.insert(0, str(args.retina_root))
    sys.path.insert(0, str(args.repo_root / "references" / "codebase" / "software" / "FACET"))

    from data.datasets.hbtxr_dean.hbtxr_dean_dataset import HBTXRDeanDataset
    from engine.models.retina.helper import get_retina_model_configs
    from engine.models.retina.retina import Retina
    from engine.module import EyeTrackingModelModule

    cfg = read_yaml(args.config)
    training_params = dict(cfg["training_params"])
    dataset_params = dict(cfg["dataset_params"])
    quant_params = dict(cfg["quant_params"])

    if training_params["arch_name"] != "retina_ann":
        raise ValueError("This HBTXR runner supports only retina_ann")

    # The local Torch/CUDA stack can report a cuDNN sublibrary mismatch on
    # this small ANN model. Disable cuDNN and use native CUDA kernels.
    torch.backends.cudnn.enabled = False

    if args.max_epochs is not None:
        training_params["num_epochs"] = args.max_epochs
    if args.batch_size is not None:
        training_params["batch_size"] = args.batch_size
    if args.num_workers is not None:
        training_params["num_workers"] = args.num_workers

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{args.run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    training_params["out_dir"] = str(out_dir)

    with (out_dir / "training_params.yaml").open("w") as f:
        yaml.safe_dump(training_params, f, sort_keys=False)
    with (out_dir / "dataset_params.yaml").open("w") as f:
        yaml.safe_dump(dataset_params, f, sort_keys=False)
    with (out_dir / "quant_params.yaml").open("w") as f:
        yaml.safe_dump(quant_params, f, sort_keys=False)

    layers_config = get_retina_model_configs(dataset_params, training_params, quant_params)
    with (out_dir / "layer_configs.yaml").open("w") as f:
        yaml.safe_dump(layers_config, f, sort_keys=False)

    torch.set_float32_matmul_precision("medium")
    model = Retina(dataset_params, training_params, layers_config)
    module = EyeTrackingModelModule(model, dataset_params, training_params)

    train_dataset = HBTXRDeanDataset("train", training_params, dataset_params)
    val_dataset = HBTXRDeanDataset("val", training_params, dataset_params)
    num_workers = int(training_params.get("num_workers", 4))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(training_params["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(training_params["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    checkpoint = ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="{epoch:02d}-{val_loss:.4f}",
    )
    callbacks = [checkpoint]

    trainer = pl.Trainer(
        max_epochs=int(training_params["num_epochs"]),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.device] if torch.cuda.is_available() else "auto",
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=False,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=50,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(module, dataloaders=val_loader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
