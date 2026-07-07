#!/usr/bin/env python3
from __future__ import annotations

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.relocate_dataset import build_argparser, run


if __name__ == "__main__":
    summary = run(build_argparser().parse_args())
    print(f"[DONE] relocated {summary['n_sessions']} sessions under {summary['canonical_root']}")
