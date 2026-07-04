import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from EvEye.dataset.dataset_factory import make_dataset
from EvEye.model.model_factory import make_model
from EvEye.utils.scripts.load_config import load_config


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def build_loader(config: dict, batch_size: int, num_workers: int) -> DataLoader:
    dataset_cfg = copy.deepcopy(config["dataloader"]["train"]["dataset"])
    dataset = make_dataset(dataset_cfg)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )


def probe_batch_size(
    config: dict,
    batch_size: int,
    device: torch.device,
    steps: int,
    num_workers: int,
    precision: str,
) -> dict:
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    loader = build_loader(config, batch_size=batch_size, num_workers=num_workers)
    model_cfg = copy.deepcopy(config["model"])
    model = make_model(model_cfg).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)

    durations = []
    completed_steps = 0
    try:
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            batch = move_batch_to_device(batch, device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            if precision == "bf16-mixed":
                autocast_dtype = torch.bfloat16
            elif precision == "16-mixed":
                autocast_dtype = torch.float16
            else:
                autocast_dtype = None

            if autocast_dtype is None:
                pred = model(batch["input"])
                loss, _ = model.criterion(pred, batch)
            else:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    pred = model(batch["input"])
                    loss, _ = model.criterion(pred, batch)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
            completed_steps += 1
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "batch_size": batch_size,
            "precision": precision,
            "ok": False,
            "error": "CUDA out of memory",
            "detail": str(exc).splitlines()[0],
            "completed_steps": completed_steps,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        }
    finally:
        del model
        del optimizer
        del loader
        torch.cuda.empty_cache()

    avg_step_seconds = sum(durations) / len(durations) if durations else None
    return {
        "batch_size": batch_size,
        "precision": precision,
        "ok": True,
        "completed_steps": completed_steps,
        "avg_step_seconds": avg_step_seconds,
        "samples_per_second": (
            batch_size / avg_step_seconds if avg_step_seconds and avg_step_seconds > 0 else None
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe feasible HBTXR full-training batch sizes on one GPU."
    )
    parser.add_argument("--config", default="DavisEyeEllipse_HBTXR_full_unet.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="2,4,6,8")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16-mixed", "16-mixed"),
        default="fp32",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("medium")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This probe is intended for CUDA devices.")
    torch.cuda.set_device(device)

    config = load_config(args.config)
    batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",") if value.strip()]
    results = {
        "config": args.config,
        "device": str(device),
        "steps": args.steps,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "batch_sizes": batch_sizes,
        "results": [],
    }

    for batch_size in batch_sizes:
        print(f"probing batch_size={batch_size}", flush=True)
        result = probe_batch_size(
            config=config,
            batch_size=batch_size,
            device=device,
            steps=args.steps,
            num_workers=args.num_workers,
            precision=args.precision,
        )
        print(json.dumps(result, indent=2), flush=True)
        results["results"].append(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
