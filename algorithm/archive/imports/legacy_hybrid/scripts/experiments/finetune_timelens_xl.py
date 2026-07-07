#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.preprocess.timelens_xl_finetune import (
    DEFAULT_TIMELENS_XL_DATALOADER,
    DEFAULT_TIMELENS_XL_MODEL_NAME,
    DEFAULT_TIMELENS_XL_PARAM_NAME,
    default_timelens_xl_dataset_root,
    default_timelens_xl_runs_root,
    launch_timelens_xl_finetune,
)


def _comma_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch native TimeLens-XL fine-tuning from an HBTXR-exported dataset surface")
    add_common_path_args(parser, need_raw=False, need_canonical=False)
    parser.add_argument("--dataset-root", type=str, default=None, help="Exported TimeLens-XL dataset root. Defaults to <project_root>/workspace/timelens_xl_finetune_dataset")
    parser.add_argument("--output-root", type=str, default=None, help="Run output root. Defaults to <project_root>/workspace/timelens_xl_finetune_runs")
    parser.add_argument("--param-name", type=str, default=DEFAULT_TIMELENS_XL_PARAM_NAME)
    parser.add_argument("--model-name", type=str, default=DEFAULT_TIMELENS_XL_MODEL_NAME)
    parser.add_argument("--dataloader", type=str, default=DEFAULT_TIMELENS_XL_DATALOADER, choices=["loader_timelens", "loader_timelens_mix"])
    parser.add_argument("--model-pretrained", type=str, default=None, help="Optional native TimeLens-XL checkpoint/pretrained model path")
    parser.add_argument("--init-step", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=128, help="Set 0 to disable random crop")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--interp-ratio", type=int, default=2)
    parser.add_argument("--rgb-sampling-ratio", type=int, default=1)
    parser.add_argument("--random-t", action="store_true", default=True)
    parser.add_argument("--no-random-t", action="store_false", dest="random_t")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--milestones", type=str, default="12,24", help="Comma-separated epoch milestones for MultiLR")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--max-epoch", type=int, default=27)
    parser.add_argument("--training-stage", type=str, default="tuning")
    parser.add_argument("--enable-perceptual-metrics", action="store_true", help="Enable LPIPS/DISTS perceptual losses and metrics inside the native trainer")
    parser.add_argument("--skip-training", action="store_true", help="Run validation-only inside the native trainer")
    parser.add_argument("--clear-previous", action="store_true")
    parser.add_argument("--extension", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved native run payload without launching training")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=False, need_canonical=False)
    if paths.timelens_xl_root is None:
        raise ValueError("timelens_xl_root is required. Pass --timelens-xl-root, --paths-config, or vendor TimeLens-XL under Third/FI")

    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else default_timelens_xl_dataset_root(paths)
    output_root = Path(args.output_root).resolve() if args.output_root else default_timelens_xl_runs_root(paths)
    summary = launch_timelens_xl_finetune(
        repo_root=paths.timelens_xl_root,
        dataset_root=dataset_root,
        output_root=output_root,
        param_name=str(args.param_name),
        model_name=str(args.model_name),
        dataloader=str(args.dataloader),
        crop_size=int(args.crop_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        interp_ratio=int(args.interp_ratio),
        rgb_sampling_ratio=int(args.rgb_sampling_ratio),
        random_t=bool(args.random_t),
        learning_rate=float(args.lr),
        milestones=_comma_ints(args.milestones),
        gamma=float(args.gamma),
        max_epoch=int(args.max_epoch),
        training_stage=str(args.training_stage),
        enable_perceptual_metrics=bool(args.enable_perceptual_metrics),
        model_pretrained=None if args.model_pretrained is None else str(Path(args.model_pretrained).resolve()),
        init_step=args.init_step,
        skip_training=bool(args.skip_training),
        clear_previous=bool(args.clear_previous),
        extension=str(args.extension),
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(
            f"[DONE] stage={summary['stage']} model={summary['model_name']} "
            f"dataset_root={summary['dataset_root']} output_root={summary['output_root']}"
        )


if __name__ == "__main__":
    main()
