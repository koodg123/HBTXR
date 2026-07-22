from __future__ import annotations

import ast
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


ALGORITHM_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = ALGORITHM_ROOT / "utils" / "src" / "utils"
LEGACY_ROOT = ALGORITHM_ROOT / "common" / "src" / "EvEye" / "utils"
EXCLUDED_CACHE_MODULE = "cache/NpyCacheFrameStack.py"
EXCLUDED_OPERATIONAL_NOTEBOOKS = (
    "cache/MemmapCacheFrameStack.ipynb",
    "tonic/tonicLearning.ipynb",
)
OPTIONAL_IDENTITY_MODULE = "utils.tonic.functional.PlotDistribution"
EXPECTED_WRAPPERS = {
    "__init__.py": ("utils", ()),
    "PupilTracker.py": ("utils.PupilTracker", ("PupilTracker",)),
    "cache/__init__.py": ("utils.cache", ()),
    "cache/MemmapCacheStructedEvents.py": (
        "utils.cache.MemmapCacheStructedEvents",
        (
            "merge_structed_arrays", "get_indices", "create_memmap", "load_memmap",
            "cache_structed_events", "cache_structed_ellipses", "cache_structed_data",
            "load_cached_structed_events", "load_cached_structed_labels",
            "load_cached_structed_ellipses", "load_event_segment", "load_ellipse",
            "get_nums", "main",
        ),
    ),
    "dvs_common_utils/__init__.py": ("utils.dvs_common_utils", ()),
    "dvs_common_utils/base/__init__.py": ("utils.dvs_common_utils.base", ()),
    "dvs_common_utils/base/EventsIterator.py": (
        "utils.dvs_common_utils.base.EventsIterator", ("EventsIterator", "main")
    ),
    "dvs_common_utils/base/RawDtype.py": (
        "utils.dvs_common_utils.base.RawDtype",
        ("raw_event_type", "raw_label_type", "raw_ellipse_type"),
    ),
    "dvs_common_utils/processor/__init__.py": (
        "utils.dvs_common_utils.processor", ()
    ),
    "dvs_common_utils/processor/EventRandomAffine.py": (
        "utils.dvs_common_utils.processor.EventRandomAffine",
        ("rand_range", "temporal_shift", "temporal_scale", "EventRandomAffine", "main"),
    ),
    "dvs_common_utils/processor/NumpyEventFrameRandomAffine.py": (
        "utils.dvs_common_utils.processor.NumpyEventFrameRandomAffine",
        ("rand_range", "NumpyEventFrameRandomAffine"),
    ),
    "dvs_common_utils/representation/__init__.py": (
        "utils.dvs_common_utils.representation", ()
    ),
    "dvs_common_utils/representation/FrameStack.py": (
        "utils.dvs_common_utils.representation.FrameStack",
        ("FrameStackBuilder", "main"),
    ),
    "dvs_common_utils/representation/Histgram.py": (
        "utils.dvs_common_utils.representation.Histgram",
        ("HistgramBuilder", "main"),
    ),
    "dvs_common_utils/representation/TimeSurface.py": (
        "utils.dvs_common_utils.representation.TimeSurface",
        ("TimeSurfaceBuilder", "main"),
    ),
    "dvs_common_utils/representation/TorchFrameStack.py": (
        "utils.dvs_common_utils.representation.TorchFrameStack",
        ("TorchFrameStack",),
    ),
    "processor/__init__.py": ("utils.processor", ()),
    "processor/HDF5Processor.py": (
        "utils.processor.HDF5Processor", ("HDF5Processor", "main")
    ),
    "processor/TxtProcessor.py": (
        "utils.processor.TxtProcessor", ("TxtProcessor", "main")
    ),
    "tonic/__init__.py": ("utils.tonic", ()),
    "tonic/functional/__init__.py": ("utils.tonic.functional", ()),
    "tonic/functional/CutMaxCount.py": (
        "utils.tonic.functional.CutMaxCount",
        ("cut_max_count", "tensor_cut_max_count"),
    ),
    "tonic/functional/PlotDistribution.py": (
        "utils.tonic.functional.PlotDistribution", ("plot_histogram", "plot_KDE")
    ),
    "tonic/functional/ToFrameStack.py": (
        "utils.tonic.functional.ToFrameStack",
        ("normalize", "bilinear_interpolation", "to_frame_stack_numpy", "main"),
    ),
    "tonic/slicers/__init__.py": ("utils.tonic.slicers", ()),
    "tonic/slicers/SliceEventsAtIndices.py": (
        "utils.tonic.slicers.SliceEventsAtIndices",
        ("slice_events_at_timepoints",),
    ),
    "tonic/slicers/SliceWithTimestampAndCount.py": (
        "utils.tonic.slicers.SliceWithTimestampAndCount",
        ("slice_events_by_timestamp_and_count",),
    ),
    "visualization/__init__.py": ("utils.visualization", ()),
    "visualization/visualization.py": (
        "utils.visualization.visualization",
        (
            "visualize", "visualizeHWC", "load_image", "save_image",
            "save_batch_images", "resize_image", "ensure_same_size",
            "ensure_same_dtype", "convert_to_color", "get_color_map",
            "draw_points", "draw_label", "overlay_label", "draw_contour",
            "overlay_contour", "convert_to_ellipse", "draw_ellipse", "main",
        ),
    ),
}


def _modules_from_tree(tree: ast.AST) -> list[str]:
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return _modules_from_tree(tree)


def _notebook_imported_modules(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    modules = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        tree = ast.parse(source, filename=f"{path}:cell-{index}")
        modules.extend(_modules_from_tree(tree))
    return modules


def test_cache_event_processor_and_visualization_characterization(tmp_path: Path) -> None:
    cache = import_module("utils.cache.MemmapCacheStructedEvents")
    frame_stack = import_module("utils.tonic.functional.ToFrameStack")
    raw_dtype = import_module("utils.dvs_common_utils.base.RawDtype")
    txt = import_module("utils.processor.TxtProcessor")
    visualization = import_module("utils.visualization.visualization")

    arrays = [
        np.asarray([(0, 1, 2, 0)], dtype=raw_dtype.raw_event_type),
        np.asarray([(1, 2, 3, 1), (2, 3, 4, 0)], dtype=raw_dtype.raw_event_type),
    ]
    np.testing.assert_array_equal(cache.get_indices(arrays), [[0, 1], [1, 3]])
    merged = cache.merge_structed_arrays(arrays)
    data_file = tmp_path / "events.memmap"
    info_file = tmp_path / "events.txt"
    cache.create_memmap(merged, data_file, info_file)
    np.testing.assert_array_equal(cache.load_memmap(data_file, info_file), merged)

    frames = frame_stack.to_frame_stack_numpy(
        merged[:2], sensor_size=(5, 5, 2), n_time_bins=2, mode="nearest"
    )
    assert frames.shape == (2, 2, 5, 5)
    assert frames.sum() == 2

    text_path = tmp_path / "sample.txt"
    processor = txt.TxtProcessor(text_path)
    processor.write(["alpha\n", "beta\n"])
    assert processor.read_lines() == ["alpha\n", "beta\n"]
    assert visualization.convert_to_ellipse([0, 12.5, 20.25, 8, 6, 45]) == (
        (12.5, 20.25), (8.0, 6.0), 45.0
    )


def test_legacy_eveye_owner_is_retired() -> None:
    """AM-060: the EvEye compatibility owner is removed entirely."""
    legacy_package_root = ALGORITHM_ROOT / "common" / "src" / "EvEye"
    assert not legacy_package_root.exists()
    assert not LEGACY_ROOT.exists()


def test_optional_plot_wrapper_identity_when_dependencies_are_available() -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    canonical_module = import_module(OPTIONAL_IDENTITY_MODULE)
    legacy_module = import_module("EvEye.utils.tonic.functional.PlotDistribution")
    for export in ("plot_histogram", "plot_KDE"):
        assert getattr(legacy_module, export) is getattr(canonical_module, export)


def test_utils_are_leaf_and_dataset_uses_only_canonical_utils() -> None:
    forbidden = (
        "EvEye", "dataset", "common", "engine",
        "event", "eveye.hybrid", "src", "dvs_common_utils",
    )
    canonical_sources = [
        *CANONICAL_ROOT.rglob("*.py"),
        *CANONICAL_ROOT.rglob("*.ipynb"),
    ]
    for path in canonical_sources:
        modules = (
            _notebook_imported_modules(path)
            if path.suffix == ".ipynb"
            else _imported_modules(path)
        )
        assert not any(
            module == prefix or module.startswith(prefix + ".")
            for module in modules for prefix in forbidden
        ), f"upward or legacy dependency in {path}"

    dataset_root = ALGORITHM_ROOT / "dataset" / "src" / "dataset"
    for path in dataset_root.rglob("*.py"):
        assert not any(
            module == "EvEye.utils" or module.startswith("EvEye.utils.")
            for module in _imported_modules(path)
        ), f"legacy utility dependency in {path}"


def test_am040_exclusions_migrated_to_engine_tools() -> None:
    """AM-060: the three former exclusions now live under the engine tools owner."""
    tools_root = ALGORITHM_ROOT / "engine" / "src" / "engine" / "tools"

    cache_module = tools_root / "NpyCacheFrameStack.py"
    assert cache_module.is_file()
    assert not (CANONICAL_ROOT / EXCLUDED_CACHE_MODULE).exists()
    cache_source = cache_module.read_text()
    assert "engine.tools.CacheFrameStack" in cache_source
    assert "utils.dvs_common_utils" in cache_source

    notebooks_root = ALGORITHM_ROOT / "notebooks" / "engine" / "tools"
    cache_notebook_path = notebooks_root / "MemmapCacheFrameStack.ipynb"
    tonic_notebook_path = notebooks_root / "tonicLearning.ipynb"
    for notebook_path in (cache_notebook_path, tonic_notebook_path):
        assert notebook_path.is_file()
        json.loads(notebook_path.read_text())
    cache_notebook = cache_notebook_path.read_text()
    tonic_notebook = tonic_notebook_path.read_text()
    assert "engine.tools.CacheFrameStack" in cache_notebook
    assert "utils.dvs_common_utils" in cache_notebook
    assert "dataset.DavisEyeCenter" in tonic_notebook
    assert "utils" in tonic_notebook

    moved_prefixes = (
        "EvEye.utils.PupilTracker", "EvEye.utils.cache.MemmapCacheStructedEvents",
        "EvEye.utils.cache.MemmapCacheFrameStack", "EvEye.utils.dvs_common_utils",
        "EvEye.utils.processor", "EvEye.utils.tonic", "EvEye.utils.visualization",
    )
    active_roots = ("common", "dataset", "utils", "engine", "frame", "event", "hybrid")
    residual = []
    for owner in active_roots:
        for path in (ALGORITHM_ROOT / owner).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ipynb"}:
                continue
            if "hardware_reference" in path.parts or "handover" in path.parts:
                continue
            text = path.read_text()
            if any(prefix in text for prefix in moved_prefixes):
                residual.append(path.relative_to(ALGORITHM_ROOT).as_posix())
            if path.suffix == ".ipynb":
                json.loads(text)
    assert residual == []
