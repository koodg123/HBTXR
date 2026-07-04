#!/usr/bin/env python3
from __future__ import annotations

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from src.preprocess.canonicalize import build_argparser, run


if __name__ == "__main__":
    summary = run(build_argparser().parse_args())
    print(f"[DONE] canonicalized {summary['n_sessions_ok']} sessions -> {summary['canonical_root']}")
