#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.utils.io import write_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create numbered top-level folder aliases inside tsgss according to directory "
            "creation order, without renaming the original folders."
        ),
    )
    parser.add_argument(
        "--tsgss-root",
        type=str,
        default="tsgss",
        help="Root directory whose top-level folders will receive numbered aliases",
    )
    parser.add_argument(
        "--prefix-width",
        type=int,
        default=2,
        help="Zero-pad width for numeric prefixes",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden top-level directories such as .mplconfig",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _dir_birth_mtime(path: Path) -> tuple[float, float]:
    stat_result = path.stat()
    birth = float(getattr(stat_result, "st_birthtime", -1.0))
    if birth <= 0:
        birth = float(stat_result.st_ctime)
    mtime = float(stat_result.st_mtime)
    return birth, mtime


def _alias_name(index: int, name: str, width: int) -> str:
    normalized = str(name)
    if normalized.startswith("."):
        normalized = normalized[1:]
    return f"{index:0{width}d}_{normalized}"


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def main() -> None:
    args = build_argparser().parse_args()
    tsgss_root = _resolve_path(args.tsgss_root)
    if not tsgss_root.exists():
        raise FileNotFoundError(f"Missing tsgss root: {tsgss_root}")

    entries: list[Path] = []
    for path in sorted(tsgss_root.iterdir()):
        if not path.is_dir():
            continue
        if not args.include_hidden and path.name.startswith("."):
            continue
        if path.name.startswith(tuple(f"{i:0{int(args.prefix_width)}d}_" for i in range(1, 100))):
            continue
        entries.append(path)

    ordered = sorted(
        entries,
        key=lambda path: (
            _dir_birth_mtime(path)[0],
            _dir_birth_mtime(path)[1],
            path.name,
        ),
    )

    aliases: list[dict[str, Any]] = []
    alias_paths: list[Path] = []
    for index, path in enumerate(ordered, start=1):
        alias = tsgss_root / _alias_name(index, path.name, int(args.prefix_width))
        alias_paths.append(alias)
        aliases.append(
            {
                "index": index,
                "alias_name": alias.name,
                "original_name": path.name,
                "alias_path": str(alias),
                "original_path": str(path.resolve()),
                "created_at_epoch": _dir_birth_mtime(path)[0],
                "modified_at_epoch": _dir_birth_mtime(path)[1],
            },
        )

    if args.overwrite:
        existing = [
            path for path in tsgss_root.iterdir()
            if path.is_symlink() and any(path.name == alias_path.name for alias_path in alias_paths)
        ]
        for path in existing:
            _safe_unlink(path)

    for alias_info in aliases:
        alias_path = Path(str(alias_info["alias_path"]))
        target_path = Path(str(alias_info["original_path"]))
        if alias_path.exists() or alias_path.is_symlink():
            if not args.overwrite:
                raise FileExistsError(f"Alias already exists: {alias_path}")
            _safe_unlink(alias_path)
        relative_target = os.path.relpath(target_path, alias_path.parent)
        alias_path.symlink_to(relative_target, target_is_directory=True)

    summary = {
        "experiment": "tsgss_folder_order_aliases",
        "tsgss_root": str(tsgss_root),
        "include_hidden": bool(args.include_hidden),
        "prefix_width": int(args.prefix_width),
        "n_aliases": len(aliases),
        "aliases": aliases,
    }
    write_json(summary, tsgss_root / "folder_order_aliases.json")

    lines = [
        "# TSGSS Folder Order Aliases",
        "",
        "Generated numbered aliases for top-level folders without renaming the originals.",
        "",
        "## Aliases",
    ]
    for alias_info in aliases:
        lines.append(f"- `{alias_info['alias_name']}` -> `{alias_info['original_name']}`")
    (tsgss_root / "folder_order_aliases.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(
        f"[DONE] aliases={len(aliases)} root={tsgss_root} "
        f"hidden_included={bool(args.include_hidden)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
