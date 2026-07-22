from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExternalPackageSpec:
    name: str
    repo_url: str
    directory: str
    description: str = ""
    branch: str | None = None
    checkout: str | None = None
    category: str = "reference"
    runtime_key: str | None = None
    default_enabled: bool = True
    used_by: tuple[str, ...] = ()


def default_packages_root(project_root: str | Path) -> Path:
    return (Path(project_root).resolve() / "packages").resolve()


def default_third_root(project_root: str | Path) -> Path:
    return (Path(project_root).resolve() / "Third").resolve()


def _candidate_packages_roots(project_root: str | Path) -> list[Path]:
    root = Path(project_root).resolve()
    candidates = [default_packages_root(root)]
    parent_packages = (root.parent / "packages").resolve()
    if parent_packages not in candidates:
        candidates.append(parent_packages)
    return candidates


def _candidate_integrated_dirs(project_root: str | Path, *, spec_name: str) -> list[Path]:
    root = Path(project_root).resolve()
    third_root = default_third_root(root)
    mapping = {
        "timelens_xl": (third_root / "FI",),
        "v2e": (third_root / "EI",),
    }
    return [candidate.resolve() for candidate in mapping.get(str(spec_name), ()) if candidate.exists()]


def builtin_package_specs() -> list[ExternalPackageSpec]:
    return [
        ExternalPackageSpec(
            name="groundedsam",
            repo_url="https://github.com/IDEA-Research/Grounded-Segment-Anything",
            directory="Grounded-Segment-Anything",
            description="Grounded-SAM annotation runtime and synthetic mode2 re-annotation backend",
            category="annotation",
            runtime_key="groundedsam",
            used_by=("annotation", "canonicalize", "mode2_synthetic_rerun"),
        ),
        ExternalPackageSpec(
            name="timelens",
            repo_url="https://github.com/ztysdu/timelens",
            directory="timelens",
            description="TimeLens interpolation backend used by mode2 synthetic-frame generation",
            category="frame_interpolation",
            runtime_key="timelens",
            used_by=("mode2_interpolation",),
        ),
        ExternalPackageSpec(
            name="ultralytics-sam3",
            repo_url="https://github.com/ultralytics/ultralytics",
            directory="ultralytics",
            description="Ultralytics SAM3 annotation backend integrated as a parallel eye-annotation runtime",
            category="annotation",
            runtime_key="ultralytics_sam3",
            used_by=("annotation",),
        ),
        ExternalPackageSpec(
            name="groundedsam2",
            repo_url="https://github.com/IDEA-Research/Grounded-SAM-2",
            directory="Grounded-SAM-2",
            description="Grounded-SAM-2 annotation backend integrated as a parallel image/video-aware runtime",
            category="annotation",
            runtime_key="groundedsam2",
            used_by=("annotation",),
        ),
        ExternalPackageSpec(
            name="timelens-xl",
            repo_url="https://github.com/OpenImagingLab/TimeLens-XL",
            directory="TimeLens-XL",
            description="TimeLens-XL frame interpolation backend wired alongside the original TimeLens path",
            category="frame_interpolation",
            runtime_key="timelens_xl",
            used_by=("mode2_interpolation",),
        ),
        ExternalPackageSpec(
            name="v2e",
            repo_url="https://github.com/SensorsINI/v2e",
            directory="v2e",
            description="v2e event-generation backend used after target-frame interpolation when synthetic events are requested",
            category="event_generation",
            runtime_key="v2e",
            used_by=("synthetic_event_generation",),
        ),
        ExternalPackageSpec(
            name="swift-eye",
            repo_url="https://github.com/ztysdu/Swift-Eye",
            directory="Swift-Eye",
            description="Swift-Eye reference repository for comparison and archive provenance",
            category="reference",
            runtime_key="swift-eye",
            default_enabled=False,
            used_by=("reference",),
        ),
    ]


def _candidate_package_dirs(project_root: str | Path, *, spec_name: str) -> list[Path]:
    package_roots = _candidate_packages_roots(project_root)
    if spec_name == "groundedsam":
        names = ("Grounded-Segment-Anything", "Grounded-Segment-Anything-main")
    elif spec_name == "ultralytics_sam3":
        names = ("ultralytics",)
    elif spec_name == "groundedsam2":
        names = ("Grounded-SAM-2",)
    elif spec_name == "timelens":
        names = ("timelens", "TimeLens")
    elif spec_name == "timelens_xl":
        names = ("TimeLens-XL",)
    elif spec_name == "v2e":
        names = ("v2e",)
    elif spec_name == "swift-eye":
        names = ("Swift-Eye",)
    else:
        names = (spec_name,)
    candidates: list[Path] = []
    candidates.extend(_candidate_integrated_dirs(project_root, spec_name=spec_name))
    for root in package_roots:
        for name in names:
            candidates.append(root / name)
    return candidates


def default_external_repo_roots(project_root: str | Path) -> dict[str, Path | None]:
    roots: dict[str, Path | None] = {}
    for spec in builtin_package_specs():
        if spec.runtime_key is None:
            continue
        resolved = None
        for candidate in _candidate_package_dirs(project_root, spec_name=spec.runtime_key):
            if candidate.exists():
                resolved = candidate.resolve()
                break
        roots[str(spec.runtime_key)] = resolved
    return roots


def _normalize_package_spec(item: dict[str, Any]) -> ExternalPackageSpec:
    name = str(item.get("name", "")).strip()
    repo_url = str(item.get("repo_url", item.get("url", ""))).strip()
    directory = str(item.get("directory", item.get("dir", name))).strip()
    if not name:
        raise ValueError(f"Package entry is missing name: {item}")
    if not repo_url:
        raise ValueError(f"Package entry is missing repo_url: {item}")
    if not directory:
        raise ValueError(f"Package entry is missing directory: {item}")
    return ExternalPackageSpec(
        name=name,
        repo_url=repo_url,
        directory=directory,
        description=str(item.get("description", "")).strip(),
        branch=None if item.get("branch") in {None, ""} else str(item.get("branch")).strip(),
        checkout=None if item.get("checkout") in {None, ""} else str(item.get("checkout")).strip(),
        category=str(item.get("category", "reference")).strip() or "reference",
        runtime_key=None if item.get("runtime_key") in {None, ""} else str(item.get("runtime_key")).strip(),
        default_enabled=bool(item.get("default_enabled", True)),
        used_by=tuple(str(value).strip() for value in item.get("used_by", []) if str(value).strip()),
    )


def load_package_manifest(path: str | Path) -> list[ExternalPackageSpec]:
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_items: Iterable[dict[str, Any]]
    if isinstance(data, dict):
        raw_items = data.get("packages", [])
    elif isinstance(data, list):
        raw_items = data
    else:
        raise ValueError(f"Invalid package manifest format: {manifest_path}")
    return [_normalize_package_spec(item) for item in raw_items]


def parse_extra_repo(text: str) -> ExternalPackageSpec:
    raw = str(text).strip()
    if "=" not in raw:
        raise ValueError(f"extra repo must be NAME=URL, got: {raw}")
    name, repo_url = raw.split("=", 1)
    normalized_name = str(name).strip()
    normalized_url = str(repo_url).strip()
    if not normalized_name or not normalized_url:
        raise ValueError(f"extra repo must be NAME=URL, got: {raw}")
    return ExternalPackageSpec(
        name=normalized_name,
        repo_url=normalized_url,
        directory=normalized_name,
        description="user-specified extra external package",
        category="reference",
        runtime_key=normalized_name,
        default_enabled=True,
        used_by=("extra_repo",),
    )


def merge_package_specs(
    base_specs: Iterable[ExternalPackageSpec],
    extra_specs: Iterable[ExternalPackageSpec] | None = None,
) -> list[ExternalPackageSpec]:
    merged: dict[str, ExternalPackageSpec] = {spec.name: spec for spec in base_specs}
    for spec in extra_specs or ():
        merged[spec.name] = spec
    return list(merged.values())


def filter_package_specs(
    specs: Iterable[ExternalPackageSpec],
    *,
    only: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
) -> list[ExternalPackageSpec]:
    only_set = {str(item).strip() for item in only or () if str(item).strip()}
    exclude_set = {str(item).strip() for item in exclude or () if str(item).strip()}
    category_set = {str(item).strip() for item in categories or () if str(item).strip()}
    filtered = []
    for spec in specs:
        if only_set and spec.name not in only_set:
            continue
        if spec.name in exclude_set:
            continue
        if category_set and spec.category not in category_set:
            continue
        filtered.append(spec)
    return filtered


def build_clone_command(spec: ExternalPackageSpec, *, target_dir: str | Path, shallow: bool) -> list[str]:
    command = ["git", "clone"]
    if shallow and not spec.checkout:
        command.extend(["--depth", "1"])
    if spec.branch:
        command.extend(["--branch", spec.branch])
    command.extend([spec.repo_url, str(Path(target_dir))])
    return command


def build_update_commands(spec: ExternalPackageSpec, *, target_dir: str | Path) -> list[list[str]]:
    target = str(Path(target_dir))
    commands: list[list[str]] = [["git", "-C", target, "fetch", "--all", "--tags"]]
    if spec.branch:
        commands.append(["git", "-C", target, "checkout", spec.branch])
        commands.append(["git", "-C", target, "pull", "--ff-only", "origin", spec.branch])
    elif spec.checkout:
        commands.append(["git", "-C", target, "checkout", spec.checkout])
    else:
        commands.append(["git", "-C", target, "pull", "--ff-only"])
    return commands
