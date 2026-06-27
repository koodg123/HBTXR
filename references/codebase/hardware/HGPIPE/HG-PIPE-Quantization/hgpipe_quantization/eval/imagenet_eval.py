"""ImageNet validation entrypoint for HG-PIPE paper models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, CenterCrop, InterpolationMode, Normalize, Resize, ToTensor
from tqdm import tqdm

from .fake_quant import activation_fake_quant_hooks, precision_to_config, quantize_model_weights
from .model_registry import create_paper_model, resolve_models, resolve_model


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(float(correct_k.mul_(100.0 / target.size(0)).item()))
    return results


def imagenet_root(data: Path, split: str) -> Path:
    candidate = data / split
    return candidate if candidate.exists() else data


def build_eval_transform(model_name: str):
    spec = resolve_model(model_name)
    interpolation = InterpolationMode.BICUBIC if spec.interpolation == "bicubic" else InterpolationMode.BILINEAR
    resize_size = int(round(spec.input_size[1] / spec.crop_pct))
    return Compose(
        [
            Resize(resize_size, interpolation=interpolation),
            CenterCrop(spec.input_size[1]),
            ToTensor(),
            Normalize(mean=spec.mean, std=spec.std),
        ]
    )


def evaluate_one(
    *,
    model_name: str,
    precision: str,
    data: Path,
    split: str,
    batch_size: int,
    workers: int,
    device: str,
    pretrained: bool,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    model, model_metadata = create_paper_model(model_name, checkpoint_path=checkpoint_path, pretrained=pretrained)
    transform = build_eval_transform(model_name)
    dataset = ImageFolder(str(imagenet_root(data, split)), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device == "cuda")

    model.to(device)
    fq_config = precision_to_config(precision)
    if fq_config is not None:
        quantize_model_weights(model, fq_config)

    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    start = perf_counter()
    with torch.inference_mode():
        if fq_config is None:
            for images, target in tqdm(loader, desc=f"{model_name}:{precision}", leave=False):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(images)
                acc1, acc5 = accuracy(output, target)
                batch = target.size(0)
                total += batch
                top1_sum += acc1 * batch
                top5_sum += acc5 * batch
        else:
            with activation_fake_quant_hooks(model, fq_config):
                for images, target in tqdm(loader, desc=f"{model_name}:{precision}", leave=False):
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    output = model(images)
                    acc1, acc5 = accuracy(output, target)
                    batch = target.size(0)
                    total += batch
                    top1_sum += acc1 * batch
                    top5_sum += acc5 * batch

    elapsed = perf_counter() - start
    return {
        "model": model_name,
        "precision": precision,
        "samples": total,
        "top1": top1_sum / total if total else 0.0,
        "top5": top5_sum / total if total else 0.0,
        "elapsed_sec": elapsed,
        "images_per_sec": total / elapsed if elapsed > 0 else 0.0,
        "pretrained": bool(model_metadata["pretrained"]),
        "requested_pretrained": bool(pretrained),
        "checkpoint_loaded": bool(model_metadata["checkpoint_loaded"]),
        "checkpoint_path": model_metadata["checkpoint_path"],
        "device": device,
        "paper_model": model_name,
        "timm_model_name": model_name,
        "evaluation_mode": "timm_fake_quant",
        "model_backend": "torch_native_vit",
        "quantization_flow": "fp32_baseline" if fq_config is None else "fake_quant",
        "paper_equivalent": False,
        "dataset_path": str(data),
        "dataset_split": split,
        "dataset_root": str(imagenet_root(data, split)),
        "eval_script": "hgpipe_quantization.eval.imagenet_eval",
        "provenance_note": "Pure torch.nn ViT/DeiT fake-quant sanity result using legacy timm-compatible report fields; not artifact-backed HG-PIPE paper-equivalent ImageNet validation.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate HG-PIPE paper models on ImageNet.")
    parser.add_argument("--data", type=Path, required=True, help="ImageNet root containing val/ or class folders.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--precisions", nargs="+", default=["fp32", "int8", "int4", "w4a8"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional torch state_dict checkpoint for a single selected model.")
    parser.add_argument("--output", type=Path, default=Path("reports/imagenet_accuracy.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    models = resolve_models(args.models)
    if args.checkpoint is not None and len(models) != 1:
        raise ValueError("--checkpoint requires exactly one selected model")

    results: list[dict[str, Any]] = []
    for model in models:
        checkpoint_path = args.checkpoint if args.checkpoint is not None else None
        for precision in args.precisions:
            results.append(
                evaluate_one(
                    model_name=model.name,
                    precision=precision,
                    data=args.data,
                    split=args.split,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=args.device,
                    pretrained=args.pretrained,
                    checkpoint_path=checkpoint_path,
                )
            )

    command = "python -m hgpipe_quantization.eval.imagenet_eval " + " ".join(argv if argv is not None else sys.argv[1:])
    for row in results:
        row["command"] = command
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
