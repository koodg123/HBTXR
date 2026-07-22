import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.DavisEyeEllipse.DavisEyeEllipseFrameDataset import (
    DavisEyeEllipseFrameDataset,
)
from models.detectors.hbtxr import HBTXR


def make_loader(dataset, batch_size: int, workers: int, prefetch: int):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
    return DataLoader(**kwargs)


def benchmark_candidate(dataset, batch_size: int, workers: int, prefetch: int, steps: int, warmup: int):
    loader = make_loader(dataset, batch_size, workers, prefetch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HBTXR(input_channels=1, img_size=128, patch_size=4, pretrained=False).to(device)
    model.train()
    model.set_optimizer_config(learning_rate=1e-3, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    measured = 0
    start = None
    last_loss = None
    iterator = iter(loader)
    for step in range(warmup + steps):
        batch = next(iterator)
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["input"])
        loss, _ = model.criterion(pred, batch)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step == warmup - 1:
            start = time.perf_counter()
        elif step >= warmup:
            measured += 1
    elapsed = time.perf_counter() - start
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
    )
    del loader, model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "workers": workers,
        "prefetch_factor": prefetch,
        "steps": measured,
        "elapsed_sec": elapsed,
        "iter_per_sec": measured / elapsed,
        "samples_per_sec": measured * batch_size / elapsed,
        "peak_memory_mb": peak_mb,
        "last_loss": last_loss,
    }


def write_best_config(base_config: Path, out_config: Path, best: dict):
    import yaml

    with base_config.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    for split in ("train", "val"):
        config["dataloader"][split]["batch_size"] = int(best["batch_size"])
        config["dataloader"][split]["num_workers"] = int(best["workers"])
        config["dataloader"][split]["prefetch_factor"] = int(best["prefetch_factor"])
        config["dataloader"][split]["persistent_workers"] = int(best["workers"]) > 0
    with out_config.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Benchmark HBTXR frame dataloader candidates.")
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--prefetch", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--best-config", type=Path)
    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dataset = DavisEyeEllipseFrameDataset(
        root_path=args.root_path,
        split="train",
        default_resolution=[128, 128],
        channels=1,
        use_cached_frame=True,
    )
    results = []
    for workers in args.workers:
        for prefetch in args.prefetch:
            candidate = benchmark_candidate(
                dataset=dataset,
                batch_size=args.batch_size,
                workers=workers,
                prefetch=prefetch,
                steps=args.steps,
                warmup=args.warmup,
            )
            print(json.dumps(candidate, indent=2))
            results.append(candidate)

    best = max(results, key=lambda item: item["samples_per_sec"])
    payload = {"best": best, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    if args.base_config and args.best_config:
        write_best_config(args.base_config, args.best_config, best)
    print("BEST", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
