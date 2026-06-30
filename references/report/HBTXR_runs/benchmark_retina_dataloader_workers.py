#!/usr/bin/env python3
"""Benchmark Retina HBTXR DataLoader throughput for worker count comparison."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

torch.multiprocessing.set_sharing_strategy("file_system")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=300)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--raw-list-collate",
        action="store_true",
        help="Return a list of samples from each worker to avoid torch shared-memory IPC.",
    )
    parser.add_argument(
        "--count-only-collate",
        action="store_true",
        help="Fetch samples in workers but return only the batch size to avoid tensor IPC.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = args.repo_root
    retina_root = repo / "references" / "codebase" / "software" / "retina"
    facet_root = repo / "references" / "codebase" / "software" / "FACET"
    sys.path.insert(0, str(retina_root))
    sys.path.insert(0, str(facet_root))

    from data.datasets.hbtxr_dean.hbtxr_dean_dataset import HBTXRDeanDataset

    cfg_path = retina_root / "configs" / "hbtxr_subject_independent_img64_patch4.yaml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    training_params = dict(cfg["training_params"])
    dataset_params = dict(cfg["dataset_params"])
    training_params["batch_size"] = args.batch_size

    print(
        "benchmark_config",
        "split=train",
        f"batch_size={args.batch_size}",
        f"max_batches={args.max_batches}",
        f"warmup_batches={args.warmup_batches}",
        f"shuffle={args.shuffle}",
        f"pin_memory={args.pin_memory}",
        flush=True,
    )
    print(f"root_path={dataset_params['root_path']}", flush=True)

    for num_workers in args.workers:
        training_params["num_workers"] = num_workers
        dataset = HBTXRDeanDataset("train", training_params, dataset_params)
        if args.count_only_collate:
            collate_fn = lambda samples: len(samples)
        elif args.raw_list_collate:
            collate_fn = lambda samples: samples
        else:
            collate_fn = None

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=True,
            collate_fn=collate_fn,
        )

        batch_times: list[float] = []
        sample_count = 0
        ok_batches = 0
        t0 = time.perf_counter()
        prev = t0
        for batch in loader:
            now = time.perf_counter()
            batch_times.append(now - prev)
            prev = now
            ok_batches += 1
            if args.count_only_collate:
                sample_count += int(batch)
            elif args.raw_list_collate:
                sample_count += len(batch)
            else:
                sample_count += int(batch[0].shape[0])
            if ok_batches >= args.max_batches:
                break

        total = time.perf_counter() - t0
        steady_times = (
            batch_times[args.warmup_batches :]
            if len(batch_times) > args.warmup_batches
            else batch_times
        )
        steady = sum(steady_times)
        steady_batches = len(steady_times)
        print(
            "RESULT",
            f"num_workers={num_workers}",
            f"batches={ok_batches}",
            f"samples={sample_count}",
            f"total_seconds={total:.3f}",
            f"total_batches_per_sec={ok_batches / total:.3f}",
            f"total_samples_per_sec={sample_count / total:.1f}",
            f"steady_batches={steady_batches}",
            f"steady_seconds={steady:.3f}",
            (
                f"steady_batches_per_sec={steady_batches / steady:.3f}"
                if steady > 0
                else "steady_batches_per_sec=nan"
            ),
            (
                f"steady_samples_per_sec={(steady_batches * args.batch_size) / steady:.1f}"
                if steady > 0
                else "steady_samples_per_sec=nan"
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
