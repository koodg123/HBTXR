from __future__ import annotations

from importlib.machinery import PathFinder
from pathlib import Path

from setuptools.config.pyprojecttoml import load_file

ALGORITHM_ROOT = Path(__file__).resolve().parents[2]

# After the eveye namespace was removed, every owner is a bare top-level package
# discovered from its own src root. This is deliberate: the project accepts the
# top-level-name collision risk in exchange for shorter imports.
EXPECTED_PACKAGE_DIR = {
    "common": "common/src/common",
    "dataset": "dataset/src/dataset",
    "utils": "utils/src/utils",
    "engine": "engine/src/engine",
    "event": "event/src/event",
    "hybrid": "hybrid/src/hybrid",
    "apps": "apps/src/apps",
}
EXPECTED_OWNER_PATHS = {
    name: ALGORITHM_ROOT / Path(rel) for name, rel in EXPECTED_PACKAGE_DIR.items()
}
OWNER_SOURCE_ROOTS = tuple(
    ALGORITHM_ROOT / owner / "src" for owner in EXPECTED_PACKAGE_DIR
)
EXPECTED_DISCOVERY = {
    "where": [owner + "/src" for owner in EXPECTED_PACKAGE_DIR],
    "include": [
        entry
        for owner in EXPECTED_PACKAGE_DIR
        for entry in (owner, owner + ".*")
    ],
    "namespaces": True,
}


def test_setuptools_configuration_maps_all_owner_packages() -> None:
    setuptools_config = load_file(ALGORITHM_ROOT / "pyproject.toml")["tool"][
        "setuptools"
    ]

    assert setuptools_config["package-dir"] == EXPECTED_PACKAGE_DIR
    assert setuptools_config["packages"]["find"] == EXPECTED_DISCOVERY
    assert {
        package: (ALGORITHM_ROOT / relative_path).resolve()
        for package, relative_path in setuptools_config["package-dir"].items()
    } == {
        package: path.resolve() for package, path in EXPECTED_OWNER_PATHS.items()
    }


def test_each_owner_resolves_as_a_bare_top_level_package() -> None:
    for name, path in EXPECTED_OWNER_PATHS.items():
        spec = PathFinder.find_spec(name, [str(path.parent)])
        assert spec is not None, name + " must resolve from its own src root"
        # Owners are a mix of regular packages (with __init__.py, so spec.origin
        # points at it) and namespace packages (origin is None, the directory is
        # in submodule_search_locations). Both must resolve to the owner dir.
        if spec.origin is not None:
            assert Path(spec.origin).resolve() == (path / "__init__.py").resolve()
        else:
            locations = [Path(p).resolve() for p in (spec.submodule_search_locations or ())]
            assert path.resolve() in locations


def test_eveye_namespace_no_longer_resolves() -> None:
    for source_root in OWNER_SOURCE_ROOTS:
        assert PathFinder.find_spec("eveye", [str(source_root)]) is None
    assert not any((root / "eveye").exists() for root in OWNER_SOURCE_ROOTS)


def test_legacy_eveye_owner_is_retired() -> None:
    assert not (ALGORITHM_ROOT / "common" / "src" / "EvEye").exists()
