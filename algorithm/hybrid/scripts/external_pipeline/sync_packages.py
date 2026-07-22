#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hybrid.utils.external_packages import (
    ExternalPackageSpec,
    build_clone_command,
    build_update_commands,
    builtin_package_specs,
    default_packages_root,
    filter_package_specs,
    load_package_manifest,
    merge_package_specs,
    parse_extra_repo,
)


def _default_manifest_path() -> Path:
    return PROJECT_ROOT / "packages" / "repositories.json"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clone or update external package repositories under packages/")
    parser.add_argument("--packages-root", type=str, default=None, help="Target packages root. Defaults to <project_root>/packages")
    parser.add_argument("--manifest", type=str, default=None, help="Package manifest JSON. Defaults to packages/repositories.json")
    parser.add_argument("--only", nargs="*", default=None, help="Optional package names to include")
    parser.add_argument("--exclude", nargs="*", default=None, help="Optional package names to skip")
    parser.add_argument(
        "--category",
        nargs="*",
        default=None,
        help="Optional package categories to include. Known values: annotation, frame_interpolation, event_generation, reference, inactive_optional",
    )
    parser.add_argument("--extra-repo", action="append", default=None, help="Add an extra repo as NAME=URL")
    parser.add_argument("--update-existing", action="store_true", help="Update already-cloned repositories instead of skipping them")
    parser.add_argument("--shallow", action="store_true", help="Use shallow clone where possible")
    parser.add_argument("--dry-run", action="store_true", help="Print planned git commands without executing them")
    parser.add_argument("--print-manifest", action="store_true", help="Print the resolved package manifest as JSON")
    return parser


def _load_specs(manifest_path: Path | None, extra_repos: list[str] | None) -> list[ExternalPackageSpec]:
    if manifest_path is not None and manifest_path.exists():
        base_specs = load_package_manifest(manifest_path)
    else:
        base_specs = builtin_package_specs()
    extra_specs = [parse_extra_repo(item) for item in (extra_repos or [])]
    return merge_package_specs(base_specs, extra_specs)


def _run_command(command: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("[DRY-RUN]", " ".join(command))
        return
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> dict:
    packages_root = (Path(args.packages_root).resolve() if args.packages_root else default_packages_root(PROJECT_ROOT))
    manifest_path = Path(args.manifest).resolve() if args.manifest else _default_manifest_path()
    packages_root.mkdir(parents=True, exist_ok=True)
    specs = _load_specs(manifest_path if manifest_path.exists() else None, args.extra_repo)
    specs = filter_package_specs(specs, only=args.only, exclude=args.exclude, categories=args.category)

    if args.print_manifest:
        payload = {
            "packages_root": str(packages_root),
            "manifest": str(manifest_path),
            "packages": [
                {
                    "name": spec.name,
                    "repo_url": spec.repo_url,
                    "directory": spec.directory,
                    "description": spec.description,
                    "branch": spec.branch,
                    "checkout": spec.checkout,
                    "category": spec.category,
                    "runtime_key": spec.runtime_key,
                    "default_enabled": spec.default_enabled,
                    "used_by": list(spec.used_by),
                }
                for spec in specs
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    results = []
    for spec in specs:
        target_dir = packages_root / spec.directory
        if not target_dir.exists():
            command = build_clone_command(spec, target_dir=target_dir, shallow=bool(args.shallow))
            _run_command(command, dry_run=bool(args.dry_run))
            results.append({"name": spec.name, "status": "cloned", "path": str(target_dir), "repo_url": spec.repo_url})
            continue

        git_dir = target_dir / ".git"
        if not git_dir.exists():
            raise RuntimeError(f"Target path exists but is not a git repository: {target_dir}")

        if not args.update_existing:
            results.append({"name": spec.name, "status": "skipped_existing", "path": str(target_dir), "repo_url": spec.repo_url})
            continue

        for command in build_update_commands(spec, target_dir=target_dir):
            _run_command(command, dry_run=bool(args.dry_run))
        results.append({"name": spec.name, "status": "updated", "path": str(target_dir), "repo_url": spec.repo_url})

    summary = {
        "packages_root": str(packages_root),
        "manifest": str(manifest_path),
        "count": len(results),
        "results": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run(build_argparser().parse_args())
