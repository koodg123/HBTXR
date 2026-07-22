from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.external_packages import default_external_repo_roots
from utils.paths import normalize_user_path, resolve_stored_path as _resolve_stored_path

ENV_KEYS = {
    "raw_root": ["EV_EYE_RAW_ROOT", "EVEYE_RAW_ROOT"],
    "canonical_root": ["EV_EYE_CANONICAL_ROOT", "EVEYE_CANONICAL_ROOT"],
    "indexes_root": ["EV_EYE_INDEXES_ROOT", "EVEYE_INDEXES_ROOT"],
    "manifests_root": ["EV_EYE_MANIFESTS_ROOT", "EVEYE_MANIFESTS_ROOT"],
    "annotation_root": ["EV_EYE_ANNOTATION_ROOT", "EVEYE_ANNOTATION_ROOT"],
    "groundedsam_root": ["GROUNDEDSAM_ROOT", "EV_EYE_GROUNDEDSAM_ROOT", "EVEYE_GROUNDEDSAM_ROOT"],
    "ultralytics_root": ["ULTRALYTICS_ROOT", "EV_EYE_ULTRALYTICS_ROOT", "EVEYE_ULTRALYTICS_ROOT"],
    "groundedsam2_root": ["GROUNDEDSAM2_ROOT", "EV_EYE_GROUNDEDSAM2_ROOT", "EVEYE_GROUNDEDSAM2_ROOT"],
    "timelens_root": ["TIMELENS_ROOT", "EV_EYE_TIMELENS_ROOT", "EVEYE_TIMELENS_ROOT"],
    "timelens_xl_root": ["TIMELENS_XL_ROOT", "EV_EYE_TIMELENS_XL_ROOT", "EVEYE_TIMELENS_XL_ROOT"],
    "v2e_root": ["V2E_ROOT", "EV_EYE_V2E_ROOT", "EVEYE_V2E_ROOT"],
    "preview_root": ["EV_EYE_PREVIEW_ROOT", "EVEYE_PREVIEW_ROOT"],
    "paths_config": ["EV_EYE_PATHS_CONFIG", "EVEYE_PATHS_CONFIG"],
}


def _first_env(keys: list[str]) -> Optional[str]:
    for key in keys:
        val = os.environ.get(key)
        if val:
            return val
    return None


@dataclass
class ResolvedPaths:
    project_root: Path
    raw_root: Optional[Path]
    canonical_root: Optional[Path]
    indexes_root: Optional[Path]
    manifests_root: Optional[Path]
    annotation_root: Optional[Path]
    groundedsam_root: Optional[Path]
    ultralytics_root: Optional[Path]
    groundedsam2_root: Optional[Path]
    timelens_root: Optional[Path]
    timelens_xl_root: Optional[Path]
    v2e_root: Optional[Path]
    preview_root: Optional[Path]
    config_path: Optional[Path]


def expand_path(path: Optional[str | Path], base_dir: Optional[str | Path] = None) -> Optional[Path]:
    if path is None:
        return None
    return normalize_user_path(path, base=base_dir)


def load_paths_config(config_path: str | Path) -> dict[str, str]:
    path = expand_path(config_path)
    if path is None:
        raise ValueError("config_path is required")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid paths config: {path}")
    return data


def save_example_paths_config(path: str | Path) -> Path:
    dst = expand_path(path)
    if dst is None:
        raise ValueError("path is required")
    dst.parent.mkdir(parents=True, exist_ok=True)
    project_root = dst.parent.parent.resolve()
    example = {
        "project_root": str(project_root),
        "raw_root": str(project_root.parent / "raw_data" / "Data_davis"),
        "groundedsam_root": str(project_root / "packages" / "Grounded-Segment-Anything"),
        "ultralytics_root": str(project_root / "packages" / "ultralytics"),
        "groundedsam2_root": str(project_root / "packages" / "Grounded-SAM-2"),
        "timelens_root": str(project_root / "packages" / "timelens"),
        "timelens_xl_root": str(project_root / "Third" / "FI"),
        "v2e_root": str(project_root / "Third" / "EI"),
        "annotation_root": str(project_root / "workspace" / "groundedsam_annotations"),
        "canonical_root": str(project_root / "workspace"),
        "indexes_root": None,
        "manifests_root": str(project_root / "manifests"),
        "preview_root": str(project_root / "workspace" / "previews"),
    }
    dst.write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8")
    return dst


def add_common_path_args(
    parser: argparse.ArgumentParser,
    *,
    need_raw: bool = False,
    need_canonical: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument("--paths-config", type=str, default=None, help="JSON file with raw_root/canonical_root/indexes_root/manifests_root and external package roots")
    parser.add_argument("--project-root", type=str, default=None, help="Project root used to resolve relative paths")
    parser.add_argument("--raw-root", type=str, required=False if not need_raw else False, help="Raw EV-Eye Data_davis root")
    parser.add_argument("--canonical-root", type=str, required=False if not need_canonical else False, help="Canonical workspace root or dataset root")
    parser.add_argument("--indexes-root", type=str, default=None, help="Optional explicit indexes root. If omitted, mode-aware canonical indexes are resolved under <canonical_root>/<canonical_name>/indexes when available")
    parser.add_argument("--manifests-root", type=str, default=None, help="Manifests root. Defaults to <project_root>/manifests")
    parser.add_argument("--annotation-root", type=str, default=None, help="Writable Grounded-SAM annotation export root")
    parser.add_argument("--groundedsam-root", type=str, default=None, help="Read-only Grounded-SAM repository root")
    parser.add_argument("--ultralytics-root", type=str, default=None, help="Read-only Ultralytics repository root used for the SAM3 annotation backend")
    parser.add_argument("--groundedsam2-root", type=str, default=None, help="Read-only Grounded-SAM-2 repository root")
    parser.add_argument("--timelens-root", type=str, default=None, help="Read-only TimeLens repository root")
    parser.add_argument("--timelens-xl-root", type=str, default=None, help="Read-only TimeLens-XL repository root. Defaults to Third/FI when present")
    parser.add_argument("--v2e-root", type=str, default=None, help="Read-only v2e repository root. Defaults to Third/EI when present")
    parser.add_argument("--preview-root", type=str, default=None, help="Visualization preview output root")
    return parser


def resolve_paths(
    args: argparse.Namespace,
    *,
    need_raw: bool = False,
    need_canonical: bool = False,
) -> ResolvedPaths:
    explicit_project_root = getattr(args, "project_root", None)
    project_root = expand_path(explicit_project_root) if explicit_project_root else Path.cwd().resolve()
    config_path = getattr(args, "paths_config", None) or _first_env(ENV_KEYS["paths_config"])
    config: dict[str, str] = {}
    config_base = project_root
    if config_path:
        config_path = expand_path(config_path, project_root)
        if config_path is None:
            raise ValueError("paths_config could not be resolved")
        config = load_paths_config(config_path)
        config_base = config_path.parent
        if config.get("project_root"):
            project_root = expand_path(config["project_root"], config_base) or project_root

    def _pick(name: str) -> Optional[Path]:
        cli_val = getattr(args, name, None)
        if cli_val:
            return expand_path(cli_val, project_root)
        if config.get(name):
            return expand_path(config[name], config_base)
        env_val = _first_env(ENV_KEYS.get(name, []))
        if env_val:
            return expand_path(env_val, project_root)
        return None

    raw_root = _pick("raw_root")
    canonical_root = _pick("canonical_root")
    indexes_root = _pick("indexes_root") or ((canonical_root / "indexes").resolve() if canonical_root else None)
    manifests_root = _pick("manifests_root") or (project_root / "manifests").resolve()
    annotation_root = _pick("annotation_root")
    external_repo_roots = default_external_repo_roots(project_root)
    groundedsam_root = _pick("groundedsam_root") or external_repo_roots.get("groundedsam")
    ultralytics_root = _pick("ultralytics_root") or external_repo_roots.get("ultralytics_sam3")
    groundedsam2_root = _pick("groundedsam2_root") or external_repo_roots.get("groundedsam2")
    timelens_root = _pick("timelens_root") or external_repo_roots.get("timelens")
    timelens_xl_root = _pick("timelens_xl_root") or external_repo_roots.get("timelens_xl")
    v2e_root = _pick("v2e_root") or external_repo_roots.get("v2e")
    preview_root = _pick("preview_root") or (project_root / "workspace" / "previews").resolve()

    if need_raw and raw_root is None:
        raise ValueError("raw_root is required. Pass --raw-root or --paths-config.")
    if need_canonical and canonical_root is None:
        raise ValueError("canonical_root is required. Pass --canonical-root or --paths-config.")

    return ResolvedPaths(
        project_root=project_root,
        raw_root=raw_root,
        canonical_root=canonical_root,
        indexes_root=indexes_root,
        manifests_root=manifests_root,
        annotation_root=annotation_root,
        groundedsam_root=groundedsam_root,
        ultralytics_root=ultralytics_root,
        groundedsam2_root=groundedsam2_root,
        timelens_root=timelens_root,
        timelens_xl_root=timelens_xl_root,
        v2e_root=v2e_root,
        preview_root=preview_root,
        config_path=config_path,
    )


def relativize_to(root: str | Path, path: str | Path) -> str:
    root_path = expand_path(root)
    target_path = expand_path(path)
    if root_path is None or target_path is None:
        raise ValueError("root and path are required")
    try:
        return str(target_path.relative_to(root_path))
    except ValueError:
        return str(target_path)


def resolve_stored_path(root: str | Path, stored_path: str | Path) -> Path:
    return _resolve_stored_path(root, stored_path)
