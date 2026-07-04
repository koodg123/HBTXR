from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.preprocess.timelens_xl_finetune import (
    _ensure_timelens_xl_repo_importable,
    register_hbtxr_timelens_xl_param,
)


def _parse_milestones(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal HBTXR entrypoint for native TimeLens-XL training")
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--param-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataloader", type=str, required=True)
    parser.add_argument("--crop-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--interp-ratio", type=int, required=True)
    parser.add_argument("--rgb-sampling-ratio", type=int, required=True)
    parser.add_argument("--random-t", type=int, choices=[0, 1], required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--milestones", type=str, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--training-stage", type=str, required=True)
    parser.add_argument("--enable-perceptual-metrics", type=int, choices=[0, 1], required=True)
    parser.add_argument("--skip-training", type=int, choices=[0, 1], required=True)
    parser.add_argument("--clear-previous", type=int, choices=[0, 1], required=True)
    parser.add_argument("--extension", type=str, required=True)
    parser.add_argument("--child-argv-json", type=str, required=True)
    parser.add_argument("--model-pretrained", type=str, default="")
    parser.add_argument("--init-step", type=str, default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    _ensure_timelens_xl_repo_importable(repo_root)
    register_hbtxr_timelens_xl_param(
        repo_root=repo_root,
        dataset_root=Path(args.dataset_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        param_name=str(args.param_name),
        model_name=str(args.model_name),
        dataloader=str(args.dataloader),
        crop_size=int(args.crop_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        interp_ratio=int(args.interp_ratio),
        rgb_sampling_ratio=int(args.rgb_sampling_ratio),
        random_t=bool(int(args.random_t)),
        learning_rate=float(args.learning_rate),
        milestones=_parse_milestones(args.milestones),
        gamma=float(args.gamma),
        max_epoch=int(args.max_epoch),
        training_stage=str(args.training_stage),
        enable_perceptual_metrics=bool(int(args.enable_perceptual_metrics)),
    )
    sys.argv = json.loads(args.child_argv_json)
    import run_network

    run_network.main()


if __name__ == "__main__":
    main()
