from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

from dataset.preprocess.io_utils import discover_frames_dir, maybe_link_or_copy, read_json, read_jsonl, write_json, write_jsonl
from dataset.preprocess.path_utils import add_common_path_args, expand_path, relativize_to, resolve_paths, resolve_stored_path


def _rewrite_value(value: str, *, old_root: Path, new_root: Path, relative: bool) -> str:
    original = Path(str(value))
    absolute = original if original.is_absolute() else (old_root / original).resolve()
    try:
        rel = absolute.relative_to(old_root.resolve())
        rewritten = (new_root / rel).resolve()
    except ValueError:
        rewritten = absolute
    return relativize_to(new_root, rewritten) if relative else str(rewritten)


def _rewrite_meta(
    meta: Dict,
    *,
    old_canonical_root: Path,
    new_canonical_root: Path,
    old_raw_root: Path | None,
    new_raw_root: Path | None,
    relative: bool,
) -> Dict:
    out = dict(meta)
    for key in ("annotation_store_path", "events_npz", "frames_dir", "session_package_path", "frame_index_path"):
        if out.get(key):
            out[key] = _rewrite_value(out[key], old_root=old_canonical_root, new_root=new_canonical_root, relative=relative)
    if old_raw_root and new_raw_root:
        for key in ("raw_session_dir", "annotation_csv"):
            if out.get(key):
                out[key] = _rewrite_value(out[key], old_root=old_raw_root, new_root=new_raw_root, relative=False)
    layout = dict(out.get("layout", {}))
    if layout and old_raw_root and new_raw_root:
        for key in ("raw_session_dir", "frames_dir", "events_dir", "event_file", "annotation_csv"):
            if layout.get(key):
                layout[key] = _rewrite_value(layout[key], old_root=old_raw_root, new_root=new_raw_root, relative=False)
        out["layout"] = layout
    return out


def _rewrite_ann_row(row: Dict, *, old_canonical_root: Path, new_canonical_root: Path, relative: bool) -> Dict:
    out = dict(row)
    for key in ("frame_path", "mask_path"):
        if out.get(key):
            out[key] = _rewrite_value(out[key], old_root=old_canonical_root, new_root=new_canonical_root, relative=relative)
    return out


def _rewrite_session_package(payload: Dict, *, old_canonical_root: Path, new_canonical_root: Path, relative: bool) -> Dict:
    out = dict(payload)
    for key in ("events_npz", "frame_index_path", "annotation_store_path", "session_package_path"):
        if out.get(key):
            out[key] = _rewrite_value(out[key], old_root=old_canonical_root, new_root=new_canonical_root, relative=relative)
    return out


def _rewrite_manifest_row(row: Dict, *, old_canonical_root: Path, new_canonical_root: Path, relative: bool) -> Dict:
    out = dict(row)
    for key in ("events_npz", "frame_path", "annotation_store_path", "session_package_path"):
        if out.get(key):
            out[key] = _rewrite_value(out[key], old_root=old_canonical_root, new_root=new_canonical_root, relative=relative)
    out["canonical_root"] = str(new_canonical_root)

    frame_branch = dict(out.get("frame_branch", {}))
    frame_target = dict(frame_branch.get("target", {}))
    if frame_target.get("mask_path"):
        frame_target["mask_path"] = _rewrite_value(frame_target["mask_path"], old_root=old_canonical_root, new_root=new_canonical_root, relative=relative)
    frame_branch["target"] = frame_target
    out["frame_branch"] = frame_branch

    event_branch = dict(out.get("event_branch", {}))
    for ref_key in ("pre_ann_ref", "cur_ann_ref"):
        ref = dict(event_branch.get(ref_key, {}))
        if ref.get("annotation_store_path"):
            ref["annotation_store_path"] = _rewrite_value(ref["annotation_store_path"], old_root=old_canonical_root, new_root=new_canonical_root, relative=relative)
        event_branch[ref_key] = ref
    out["event_branch"] = event_branch
    return out


def _iter_manifest_paths(manifests_root: Path) -> Iterable[Path]:
    if not manifests_root.exists():
        return []
    return sorted(manifests_root.rglob("*.jsonl"))


def _relink_session_frames(
    meta: Dict,
    *,
    canonical_root: Path,
    link_mode: str,
    overwrite_links: bool,
) -> int:
    raw_session_dir = expand_path(meta.get("raw_session_dir"))
    if raw_session_dir is None or not raw_session_dir.exists():
        return 0
    raw_frames_dir = discover_frames_dir(raw_session_dir)
    if raw_frames_dir is None:
        return 0
    dst_frames_dir = resolve_stored_path(canonical_root, meta["frames_dir"])
    dst_frames_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame_path in sorted(raw_frames_dir.glob("*.png")):
        maybe_link_or_copy(frame_path, dst_frames_dir / frame_path.name, mode=link_mode, overwrite=overwrite_links)
        count += 1
    return count


def relocate_dataset(
    *,
    project_root: Path | None = None,
    canonical_root: Path,
    old_canonical_root: Path,
    new_canonical_root: Path,
    old_raw_root: Path | None = None,
    new_raw_root: Path | None = None,
    indexes_root: Path | None = None,
    manifests_root: Path | None = None,
    relative: bool = True,
    recreate_symlinks: bool = False,
    link_mode: str = "symlink",
    overwrite_links: bool = False,
    dry_run: bool = False,
) -> Dict:
    project_root = project_root.resolve() if project_root is not None else None
    canonical_root = canonical_root.resolve()
    indexes_root = (indexes_root or (canonical_root / "indexes")).resolve()
    default_manifests_root = (project_root / "manifests") if project_root is not None else (canonical_root / "manifests")
    manifests_root = (manifests_root or default_manifests_root).resolve()

    session_index_path = indexes_root / "sessions.jsonl"
    session_rows = read_jsonl(session_index_path)
    rewritten_sessions = [
        _rewrite_meta(
            row,
            old_canonical_root=old_canonical_root,
            new_canonical_root=new_canonical_root,
            old_raw_root=old_raw_root,
            new_raw_root=new_raw_root,
            relative=relative,
        )
        for row in session_rows
    ]

    relinked_frames = 0
    for row in rewritten_sessions:
        if row.get("skipped"):
            continue
        session_dir = canonical_root / "sessions" / f"user{int(row['user_id']):02d}" / row["eye"] / f"session_{row['session_code']}"
        meta_path = session_dir / "meta.json"
        eye_region_path = session_dir / "labels" / "eye_region.json"
        ann_path = session_dir / "labels" / "frame_annotations.jsonl"
        session_package_path = session_dir / "labels" / "session_package.json"

        if meta_path.exists():
            updated_meta = _rewrite_meta(
                read_json(meta_path),
                old_canonical_root=old_canonical_root,
                new_canonical_root=new_canonical_root,
                old_raw_root=old_raw_root,
                new_raw_root=new_raw_root,
                relative=relative,
            )
            if not dry_run:
                write_json(updated_meta, meta_path)
        else:
            updated_meta = row

        if eye_region_path.exists():
            eye_region = _rewrite_meta(
                read_json(eye_region_path),
                old_canonical_root=old_canonical_root,
                new_canonical_root=new_canonical_root,
                old_raw_root=old_raw_root,
                new_raw_root=new_raw_root,
                relative=relative,
            )
            if not dry_run:
                write_json(eye_region, eye_region_path)

        if ann_path.exists():
            ann_rows = read_jsonl(ann_path)
            updated_ann_rows = [
                _rewrite_ann_row(ann_row, old_canonical_root=old_canonical_root, new_canonical_root=new_canonical_root, relative=relative)
                for ann_row in ann_rows
            ]
            if not dry_run:
                write_jsonl(updated_ann_rows, ann_path)

        if session_package_path.exists():
            session_package = _rewrite_session_package(
                read_json(session_package_path),
                old_canonical_root=old_canonical_root,
                new_canonical_root=new_canonical_root,
                relative=relative,
            )
            if not dry_run:
                write_json(session_package, session_package_path)

        if recreate_symlinks and not dry_run:
            relinked_frames += _relink_session_frames(
                updated_meta,
                canonical_root=canonical_root,
                link_mode=link_mode,
                overwrite_links=overwrite_links,
            )

    if not dry_run:
        write_jsonl(rewritten_sessions, session_index_path)

    for manifest_path in _iter_manifest_paths(manifests_root):
        manifest_rows = read_jsonl(manifest_path)
        updated_rows = [
            _rewrite_manifest_row(row, old_canonical_root=old_canonical_root, new_canonical_root=new_canonical_root, relative=relative)
            for row in manifest_rows
        ]
        if not dry_run:
            write_jsonl(updated_rows, manifest_path)

    summary = {
        "project_root": str(project_root) if project_root is not None else None,
        "canonical_root": str(canonical_root),
        "indexes_root": str(indexes_root),
        "manifests_root": str(manifests_root),
        "n_sessions": len(rewritten_sessions),
        "relinked_frames": relinked_frames,
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        write_json(summary, indexes_root / "relocate_summary.json")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite canonical/manifests after raw/canonical root changes and optionally recreate frame symlinks")
    add_common_path_args(parser, need_canonical=True)
    parser.add_argument("--old-canonical-root", type=str, required=True)
    parser.add_argument("--new-canonical-root", type=str, default=None)
    parser.add_argument("--old-raw-root", type=str, default=None)
    parser.add_argument("--new-raw-root", type=str, default=None)
    parser.add_argument("--absolute-paths", action="store_true", help="Store absolute canonical paths instead of canonical-root-relative paths")
    parser.add_argument("--recreate-symlinks", action="store_true", help="Recreate frame symlinks after path rewrite")
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--overwrite-links", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Dict:
    paths = resolve_paths(args, need_canonical=True)
    old_canonical_root = expand_path(args.old_canonical_root)
    new_canonical_root = expand_path(args.new_canonical_root) if args.new_canonical_root else paths.canonical_root
    old_raw_root = expand_path(args.old_raw_root) if args.old_raw_root else None
    new_raw_root = expand_path(args.new_raw_root) if args.new_raw_root else None
    if old_canonical_root is None or new_canonical_root is None:
        raise ValueError("old/new canonical roots must resolve to concrete paths")
    return relocate_dataset(
        project_root=paths.project_root,
        canonical_root=paths.canonical_root,
        old_canonical_root=old_canonical_root,
        new_canonical_root=new_canonical_root,
        old_raw_root=old_raw_root,
        new_raw_root=new_raw_root,
        indexes_root=paths.indexes_root,
        manifests_root=paths.manifests_root,
        relative=not bool(args.absolute_paths),
        recreate_symlinks=bool(args.recreate_symlinks),
        link_mode=str(args.link_mode),
        overwrite_links=bool(args.overwrite_links),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    summary = run(build_argparser().parse_args())
    print(f"[DONE] relocated {summary['n_sessions']} sessions under {summary['canonical_root']}")
