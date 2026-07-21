from __future__ import annotations

from importlib.machinery import PathFinder
from pathlib import Path

from setuptools.config.pyprojecttoml import load_file

ALGORITHM_ROOT = Path(__file__).resolve().parents[2]
COMMON_SOURCE_ROOT = ALGORITHM_ROOT / "common" / "src"
EXPECTED_PACKAGE_DIR = {
    "eveye.common": "common/src/eveye/common",
    "eveye.dataset": "dataset/src/eveye/dataset",
    "eveye.utils": "utils/src/eveye/utils",
    "eveye.engine": "engine/src/eveye/engine",
    "eveye.event": "event/src/eveye/event",
}
EXPECTED_OWNER_PATHS = {
    "eveye.common": ALGORITHM_ROOT / "common" / "src" / "eveye" / "common",
    "eveye.dataset": ALGORITHM_ROOT / "dataset" / "src" / "eveye" / "dataset",
    "eveye.utils": ALGORITHM_ROOT / "utils" / "src" / "eveye" / "utils",
    "eveye.engine": ALGORITHM_ROOT / "engine" / "src" / "eveye" / "engine",
    "eveye.event": ALGORITHM_ROOT / "event" / "src" / "eveye" / "event",
}
OWNER_SOURCE_ROOTS = tuple(
    ALGORITHM_ROOT / owner / "src"
    for owner in ("common", "dataset", "utils", "engine", "event")
)
EXPECTED_DISCOVERY = {
    "where": [
        "common/src",
        "dataset/src",
        "utils/src",
        "engine/src",
        "event/src",
    ],
    "include": [
        "eveye.common",
        "eveye.common.*",
        "eveye.dataset",
        "eveye.dataset.*",
        "eveye.utils",
        "eveye.utils.*",
        "eveye.engine",
        "eveye.engine.*",
        "eveye.event",
        "eveye.event.*",
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
        package: path.resolve()
        for package, path in EXPECTED_OWNER_PATHS.items()
    }


def test_legacy_eveye_owner_is_retired() -> None:
    """AM-060: EvEye is neither configured nor resolvable from any owner root."""
    assert not (COMMON_SOURCE_ROOT / "EvEye").exists()
    for source_root in OWNER_SOURCE_ROOTS:
        assert PathFinder.find_spec("EvEye", [str(source_root)]) is None


def test_current_owner_roots_do_not_resolve_bare_packages() -> None:
    for bare_name in ("dataset", "utils", "engine"):
        for source_root in OWNER_SOURCE_ROOTS:
            assert PathFinder.find_spec(bare_name, [str(source_root)]) is None, (
                f"{bare_name!r} unexpectedly resolves from {source_root}; "
                "owner roots must contribute only mapped eveye portions"
            )
