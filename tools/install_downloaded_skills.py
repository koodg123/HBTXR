#!/usr/bin/env python3
import argparse
import filecmp
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def is_system_skill(skill_dir: Path, source_root: Path) -> bool:
    return ".system" in skill_dir.relative_to(source_root).parts


def discover_skill_dirs(source_root: Path) -> list[Path]:
    return sorted(path.parent for path in source_root.rglob("SKILL.md"))


def destination_for(skill_dir: Path, source_root: Path, dest_root: Path) -> Path:
    name = skill_dir.name
    if is_system_skill(skill_dir, source_root):
        return dest_root / ".system" / name
    return dest_root / name


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def copy_tree_merge(source: Path, dest: Path, dest_root: Path, backup_root: Path, dry_run: bool):
    counts = {
        "created_files": 0,
        "updated_files": 0,
        "identical_files": 0,
        "backup_files": 0,
        "created_dirs": 0,
    }
    actions = []

    if not dest.exists():
        counts["created_dirs"] += 1
        actions.append({"action": "create_dir", "path": str(dest)})
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)

    for src_file in iter_files(source):
        rel = src_file.relative_to(source)
        dst_file = dest / rel
        if not dst_file.parent.exists():
            counts["created_dirs"] += 1
            actions.append({"action": "create_dir", "path": str(dst_file.parent)})
            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)

        if not dst_file.exists():
            counts["created_files"] += 1
            actions.append({"action": "create_file", "source": str(src_file), "dest": str(dst_file)})
            if not dry_run:
                shutil.copy2(src_file, dst_file)
            continue

        if filecmp.cmp(src_file, dst_file, shallow=False):
            counts["identical_files"] += 1
            continue

        backup_file = backup_root / dest.relative_to(dest_root) / rel
        counts["backup_files"] += 1
        counts["updated_files"] += 1
        actions.append(
            {
                "action": "update_file_with_backup",
                "source": str(src_file),
                "dest": str(dst_file),
                "backup": str(backup_file),
            }
        )
        if not dry_run:
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst_file, backup_file)
            shutil.copy2(src_file, dst_file)

    return counts, actions


def add_counts(total: dict, counts: dict):
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/mnt/c/Users/JM/Downloads/skills")
    parser.add_argument("--dest", default="/home/user/.codex/skills")
    parser.add_argument("--report", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    dest_root = Path(args.dest).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not dest_root.exists():
        raise FileNotFoundError(dest_root)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = dest_root.parent / "skill-install-backups" / stamp

    skill_dirs = discover_skill_dirs(source_root)
    destinations = {}
    total_counts = {
        "created_files": 0,
        "updated_files": 0,
        "identical_files": 0,
        "backup_files": 0,
        "created_dirs": 0,
    }
    installed = []

    for skill_dir in skill_dirs:
        dest = destination_for(skill_dir, source_root, dest_root)
        existed = dest.exists()
        counts, actions = copy_tree_merge(skill_dir, dest, dest_root, backup_root, args.dry_run)
        add_counts(total_counts, counts)
        rel_dest = str(dest.relative_to(dest_root))
        destinations.setdefault(rel_dest, []).append(str(skill_dir))
        installed.append(
            {
                "name": skill_dir.name,
                "source": str(skill_dir),
                "dest": str(dest),
                "dest_relative": rel_dest,
                "system": is_system_skill(skill_dir, source_root),
                "preexisting": existed,
                "counts": counts,
                "actions": actions[:50],
                "actions_truncated": len(actions) > 50,
            }
        )

    duplicate_destinations = {
        dest: sources for dest, sources in destinations.items() if len(sources) > 1
    }
    installed_skill_md_count = sum(1 for _ in dest_root.glob("*/SKILL.md"))
    installed_system_skill_md_count = sum(1 for _ in (dest_root / ".system").glob("*/SKILL.md"))

    report = {
        "dry_run": args.dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "source_skill_dirs": len(skill_dirs),
        "destination_count": len(destinations),
        "duplicate_destinations": duplicate_destinations,
        "backup_root": str(backup_root),
        "total_counts": total_counts,
        "installed_skill_md_count": installed_skill_md_count,
        "installed_system_skill_md_count": installed_system_skill_md_count,
        "installed": installed,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in [
        "dry_run",
        "source_skill_dirs",
        "destination_count",
        "duplicate_destinations",
        "backup_root",
        "total_counts",
        "installed_skill_md_count",
        "installed_system_skill_md_count",
    ]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
