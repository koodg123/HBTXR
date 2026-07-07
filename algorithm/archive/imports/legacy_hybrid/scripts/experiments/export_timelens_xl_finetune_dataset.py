#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.path_utils import add_common_path_args, resolve_paths
from hbtxr.preprocess.timelens_xl_finetune import default_timelens_xl_dataset_root, export_timelens_xl_finetune_dataset


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a TimeLens-XL fine-tuning surface from a materialized HBTXR target-FPS dataset"
    )
    add_common_path_args(parser, need_raw=False, need_canonical=False)
    parser.add_argument("--target-root", type=str, default=None, help="Target dataset workspace root. Defaults to <project_root>/workspace/target_data")
    parser.add_argument("--target-fps", type=float, required=True, help="Target FPS tag used under <target_root>/fps_<tag>")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Export root. Defaults to <project_root>/workspace/timelens_xl_finetune_dataset",
    )
    parser.add_argument(
        "--split-map",
        type=str,
        default=None,
        help="Optional JSON split map with train/val session_key lists",
    )
    parser.add_argument(
        "--val-every-nth-sequence",
        type=int,
        default=5,
        help="Without --split-map, send every Nth sequence to val. Use 0 to keep all sequences in train.",
    )
    parser.add_argument("--export-device", type=str, default="cpu", help="Device for heavy event-packet accumulation during export. Use cuda when available")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_paths(args, need_raw=False, need_canonical=False)
    target_root = Path(args.target_root).resolve() if args.target_root else (paths.project_root / "workspace" / "target_data").resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else default_timelens_xl_dataset_root(paths)
    summary = export_timelens_xl_finetune_dataset(
        target_root=target_root,
        target_fps=float(args.target_fps),
        output_root=output_root,
        overwrite=bool(args.overwrite),
        split_map_path=args.split_map,
        val_every_nth_sequence=int(args.val_every_nth_sequence),
        export_device=str(args.export_device),
    )
    print(
        f"[DONE] stage={summary['stage']} exported={summary['n_sequences_exported']} "
        f"train={summary['n_sequences_train']} val={summary['n_sequences_val']} root={summary['output_root']}"
    )


if __name__ == "__main__":
    main()
