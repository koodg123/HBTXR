from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from hbtxr.preprocess.path_utils import resolve_paths
from hbtxr.utils.external_packages import (
    build_clone_command,
    builtin_package_specs,
    default_external_repo_roots,
    load_package_manifest,
    parse_extra_repo,
)


def test_builtin_package_specs_include_groundedsam_and_timelens():
    specs = builtin_package_specs()
    names = [spec.name for spec in specs]

    assert "groundedsam" in names
    assert "timelens" in names
    assert "ultralytics-sam3" in names
    assert "groundedsam2" in names
    assert "timelens-xl" in names
    assert "v2e" in names


def test_default_external_repo_roots_detects_packages_layout(tmp_path):
    project_root = tmp_path / "proj"
    (project_root / "packages" / "Grounded-Segment-Anything").mkdir(parents=True)
    (project_root / "packages" / "timelens").mkdir(parents=True)
    (project_root / "packages" / "ultralytics").mkdir(parents=True)
    (project_root / "packages" / "Grounded-SAM-2").mkdir(parents=True)
    (project_root / "packages" / "TimeLens-XL").mkdir(parents=True)
    (project_root / "packages" / "v2e").mkdir(parents=True)

    roots = default_external_repo_roots(project_root)

    assert roots["groundedsam"] == (project_root / "packages" / "Grounded-Segment-Anything").resolve()
    assert roots["timelens"] == (project_root / "packages" / "timelens").resolve()
    assert roots["ultralytics_sam3"] == (project_root / "packages" / "ultralytics").resolve()
    assert roots["groundedsam2"] == (project_root / "packages" / "Grounded-SAM-2").resolve()
    assert roots["timelens_xl"] == (project_root / "packages" / "TimeLens-XL").resolve()
    assert roots["v2e"] == (project_root / "packages" / "v2e").resolve()


def test_default_external_repo_roots_prefers_third_vendor_roots(tmp_path):
    project_root = tmp_path / "proj"
    (project_root / "Third" / "FI").mkdir(parents=True)
    (project_root / "Third" / "EI").mkdir(parents=True)
    (project_root / "packages" / "TimeLens-XL").mkdir(parents=True)
    (project_root / "packages" / "v2e").mkdir(parents=True)

    roots = default_external_repo_roots(project_root)

    assert roots["timelens_xl"] == (project_root / "Third" / "FI").resolve()
    assert roots["v2e"] == (project_root / "Third" / "EI").resolve()


def test_default_external_repo_roots_falls_back_to_parent_packages_root(tmp_path):
    merge_root = tmp_path / "merge"
    project_root = merge_root / "HBTXR_v3_0"
    project_root.mkdir(parents=True)
    (merge_root / "packages" / "v2e").mkdir(parents=True)
    (merge_root / "packages" / "TimeLens-XL").mkdir(parents=True)

    roots = default_external_repo_roots(project_root)

    assert roots["v2e"] == (merge_root / "packages" / "v2e").resolve()
    assert roots["timelens_xl"] == (merge_root / "packages" / "TimeLens-XL").resolve()


def test_load_package_manifest_reads_default_shape(tmp_path):
    manifest_path = tmp_path / "repositories.json"
    manifest_path.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "groundedsam",
                        "repo_url": "https://example.com/groundedsam.git",
                        "directory": "Grounded-Segment-Anything",
                        "category": "annotation",
                        "runtime_key": "groundedsam",
                        "default_enabled": True,
                        "used_by": ["canonicalize", "annotation"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_package_manifest(manifest_path)

    assert len(specs) == 1
    assert specs[0].name == "groundedsam"
    assert specs[0].directory == "Grounded-Segment-Anything"
    assert specs[0].category == "annotation"
    assert specs[0].runtime_key == "groundedsam"
    assert specs[0].default_enabled is True
    assert specs[0].used_by == ("canonicalize", "annotation")


def test_parse_extra_repo_supports_name_equals_url():
    spec = parse_extra_repo("swift-eye=https://example.com/swift-eye.git")

    assert spec.name == "swift-eye"
    assert spec.repo_url == "https://example.com/swift-eye.git"
    assert spec.directory == "swift-eye"


def test_build_clone_command_supports_branch_and_shallow():
    spec = parse_extra_repo("swift-eye=https://example.com/swift-eye.git")
    command = build_clone_command(spec, target_dir="packages/swift-eye", shallow=True)

    assert command[:2] == ["git", "clone"]
    assert "https://example.com/swift-eye.git" in command
    assert Path(command[-1]) == Path("packages/swift-eye")


def test_resolve_paths_auto_detects_managed_package_roots(tmp_path):
    project_root = tmp_path / "proj"
    raw_root = project_root / "raw"
    canonical_root = project_root / "workspace"
    (project_root / "packages" / "Grounded-Segment-Anything").mkdir(parents=True)
    (project_root / "packages" / "timelens").mkdir(parents=True)
    (project_root / "packages" / "ultralytics").mkdir(parents=True)
    (project_root / "packages" / "Grounded-SAM-2").mkdir(parents=True)
    (project_root / "packages" / "TimeLens-XL").mkdir(parents=True)
    (project_root / "packages" / "v2e").mkdir(parents=True)

    args = argparse.Namespace(
        paths_config=None,
        project_root=str(project_root),
        raw_root=str(raw_root),
        canonical_root=str(canonical_root),
        indexes_root=None,
        manifests_root=None,
        annotation_root=None,
        groundedsam_root=None,
        timelens_root=None,
        ultralytics_root=None,
        groundedsam2_root=None,
        timelens_xl_root=None,
        v2e_root=None,
        preview_root=None,
    )

    paths = resolve_paths(args, need_raw=False, need_canonical=False)

    assert paths.groundedsam_root == (project_root / "packages" / "Grounded-Segment-Anything").resolve()
    assert paths.timelens_root == (project_root / "packages" / "timelens").resolve()
    assert paths.ultralytics_root == (project_root / "packages" / "ultralytics").resolve()
    assert paths.groundedsam2_root == (project_root / "packages" / "Grounded-SAM-2").resolve()
    assert paths.timelens_xl_root == (project_root / "packages" / "TimeLens-XL").resolve()
    assert paths.v2e_root == (project_root / "packages" / "v2e").resolve()


def test_resolve_paths_auto_detects_third_vendor_roots(tmp_path):
    project_root = tmp_path / "proj"
    raw_root = project_root / "raw"
    canonical_root = project_root / "workspace"
    (project_root / "Third" / "FI").mkdir(parents=True)
    (project_root / "Third" / "EI").mkdir(parents=True)

    args = argparse.Namespace(
        paths_config=None,
        project_root=str(project_root),
        raw_root=str(raw_root),
        canonical_root=str(canonical_root),
        indexes_root=None,
        manifests_root=None,
        annotation_root=None,
        groundedsam_root=None,
        timelens_root=None,
        ultralytics_root=None,
        groundedsam2_root=None,
        timelens_xl_root=None,
        v2e_root=None,
        preview_root=None,
    )

    paths = resolve_paths(args, need_raw=False, need_canonical=False)

    assert paths.timelens_xl_root == (project_root / "Third" / "FI").resolve()
    assert paths.v2e_root == (project_root / "Third" / "EI").resolve()


def test_sync_packages_run_supports_category_filter(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sync_packages.py"
    spec = importlib.util.spec_from_file_location("hbtxr_sync_packages", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest_path = tmp_path / "repositories.json"
    manifest_path.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "groundedsam",
                        "repo_url": "https://example.com/groundedsam.git",
                        "directory": "Grounded-Segment-Anything",
                        "category": "annotation",
                        "runtime_key": "groundedsam",
                    },
                    {
                        "name": "timelens",
                        "repo_url": "https://example.com/timelens.git",
                        "directory": "timelens",
                        "category": "frame_interpolation",
                        "runtime_key": "timelens",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        packages_root=str(tmp_path / "packages"),
        manifest=str(manifest_path),
        only=None,
        exclude=None,
        category=["annotation"],
        extra_repo=None,
        update_existing=False,
        shallow=False,
        dry_run=True,
        print_manifest=False,
    )
    summary = module.run(args)
    assert summary["count"] == 1
    assert summary["results"][0]["name"] == "groundedsam"


def test_third_manifest_declares_fi_and_ei_vendor_roots():
    manifest_path = Path(__file__).resolve().parents[1] / "Third" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = {item["runtime_key"]: item["vendored_path"] for item in payload["integrated_roots"]}

    assert roots["timelens_xl"] == "Third/FI"
    assert roots["v2e"] == "Third/EI"
