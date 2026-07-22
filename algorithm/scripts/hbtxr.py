#!/usr/bin/env python3
"""Unified HBTXR CLI dispatcher (run from the ``algorithm/`` root).

Thin front-end over the engine entrypoints; forwards the remaining args verbatim::

    python scripts/hbtxr.py train   -c configs/experiment/frame_hbtxr.yaml
    python scripts/hbtxr.py eval     -c configs/experiment/frame_hbtxr.yaml --ckpt runs/.../final.pt
    python scripts/hbtxr.py infer    -c configs/experiment/frame_hbtxr.yaml -o preds.csv
    python scripts/hbtxr.py distill  -c configs/experiment/frame_distill.yaml
    python scripts/hbtxr.py prune    -c configs/experiment/frame_prune.yaml

Each subcommand maps to ``python -m engine.<area>.entrypoint`` and accepts the same
options; see that module's ``--help``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Pin algorithm/ as the import root for the flat, non-installed layout.
_ALGORITHM_ROOT = Path(__file__).resolve().parent.parent
if str(_ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGORITHM_ROOT))

_COMMANDS = {
    "train": "engine.train.entrypoint",
    "eval": "engine.eval.entrypoint",
    "infer": "engine.infer.entrypoint",
    "predict": "engine.infer.entrypoint",
    "distill": "engine.distill.entrypoint",
    "prune": "engine.compress.entrypoint",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: hbtxr {%s} [options]" % ",".join(_COMMANDS))
        return
    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        raise SystemExit(f"unknown command {command!r}; choose from {', '.join(_COMMANDS)}")
    import importlib

    module = importlib.import_module(_COMMANDS[command])
    module.main(rest)


if __name__ == "__main__":
    main()
