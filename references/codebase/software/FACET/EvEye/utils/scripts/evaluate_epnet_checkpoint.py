import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model
from EvEye.utils.scripts.load_config import load_config


PAPER_TABLE2 = {
    "paper_p1_acc_percent": 99.59,
    "paper_mean_pixel_error": 0.2030,
    "paper_params_m": 3.92,
    "paper_flops_g": 3.44,
    "paper_latency_ms": 0.5302,
}


def move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def load_checkpoint(model, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


def tensor_to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def evaluate_metrics(model, dataloader, device, max_batches: int):
    metric_rows = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            batch = move_to_device(batch, device)
            metrics = model.validation_step(batch, batch_idx)
            if metrics is None:
                continue
            row = {key: tensor_to_float(value) for key, value in metrics.items()}
            metric_rows.append(row)

    if not metric_rows:
        return {}, 0

    keys = metric_rows[0].keys()
    averaged = {
        key: sum(row[key] for row in metric_rows) / len(metric_rows) for key in keys
    }
    return averaged, len(metric_rows)


def count_params(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def measure_flops(model, sample_input):
    try:
        from thop import profile
    except ImportError:
        return None

    model.eval()
    with torch.no_grad():
        flops, _ = profile(model, inputs=(sample_input,), verbose=False)
    return float(flops)


def measure_latency_ms(model, sample_input, warmup: int, iterations: int):
    model.eval()
    device = sample_input.device
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(iterations, 1)


def make_markdown(result):
    metrics = result["metrics"]
    paper = result["paper_table2_reference"]
    model_type = result.get("model_type", "model")
    rows = [
        ("P10", metrics.get("val_p10_acc"), None),
        ("P5", metrics.get("val_p5_acc"), None),
        ("P3", metrics.get("val_p3_acc"), None),
        ("P1", metrics.get("val_p1_acc"), paper["paper_p1_acc_percent"] / 100.0),
        ("mean pixel error", metrics.get("val_mean_distance"), paper["paper_mean_pixel_error"]),
        ("IoU", metrics.get("val_IoU"), None),
        ("AP", metrics.get("val_AP"), None),
        ("params M", result.get("params_m"), paper["paper_params_m"]),
        ("FLOPs G", result.get("flops_g"), paper["paper_flops_g"]),
        ("latency ms", result.get("latency_ms"), paper["paper_latency_ms"]),
    ]

    lines = [
        f"# FACET {model_type} Evaluation Result",
        "",
        f"Model type: `{model_type}`",
        f"Config: `{result['config']}`",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Device: `{result['device']}`",
        f"Evaluated batches: `{result['evaluated_batches']}`",
        "",
        "| Metric | Current | Paper Table II reference | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name, current, reference in rows:
        current_text = "n/a" if current is None else f"{current:.6g}"
        reference_text = "n/a" if reference is None else f"{reference:.6g}"
        if current is None or reference is None:
            delta_text = "n/a"
        else:
            delta_text = f"{current - reference:.6g}"
        lines.append(f"| {name} | {current_text} | {reference_text} | {delta_text} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Paper Table II reference values available in the local analysis are P1, mean pixel error, params, FLOPs, and latency.",
            "- P10/P5/P3/IoU/AP paper reference values were not recovered from the inspected local report text.",
            "- Latency is measured in the current runtime and is not TensorRT latency unless the runtime is explicitly TensorRT-backed.",
            "- This report is a metric pipeline artifact; it is only a reproduction result when run on the final full checkpoint and full validation split.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a FACET DavisEyeEllipse checkpoint and emit JSON/Markdown comparison artifacts."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-iterations", type=int, default=20)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    disable_cudnn = os.environ.get("FACET_DISABLE_CUDNN", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if disable_cudnn:
        torch.backends.cudnn.enabled = False

    cfg = load_config(args.config)
    val_cfg = copy.deepcopy(cfg["dataloader"]["val"])
    val_cfg["shuffle"] = False
    if val_cfg.get("num_workers", 0) is None:
        val_cfg["num_workers"] = 0
    dataloader = make_dataloader(val_cfg)

    model = make_model(copy.deepcopy(cfg["model"])).to(device)
    load_checkpoint(model, args.checkpoint, device)

    metrics, evaluated_batches = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
    )

    sample_batch = next(iter(dataloader))
    sample_input = sample_batch["input"].to(device)
    total_params, trainable_params = count_params(model)
    flops = measure_flops(model, sample_input[:1])
    latency_ms = measure_latency_ms(
        model=model,
        sample_input=sample_input[:1],
        warmup=args.latency_warmup,
        iterations=args.latency_iterations,
    )

    result = {
        "model_type": cfg["model"].get("type", "unknown"),
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "dataset_root": cfg["dataloader"]["val"]["dataset"]["root_path"],
        "max_batches": args.max_batches,
        "evaluated_batches": evaluated_batches,
        "metrics": metrics,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "flops": flops,
        "flops_g": None if flops is None else flops / 1e9,
        "latency_ms": latency_ms,
        "latency_warmup": args.latency_warmup,
        "latency_iterations": args.latency_iterations,
        "paper_table2_reference": PAPER_TABLE2,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(make_markdown(result), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
