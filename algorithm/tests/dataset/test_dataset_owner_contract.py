from __future__ import annotations

import ast
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import torch


ALGORITHM_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = ALGORITHM_ROOT / "dataset" / "src" / "eveye" / "dataset"
LEGACY_ROOT = ALGORITHM_ROOT / "common" / "src" / "EvEye" / "dataset"
LEGACY_DATASET_PACKAGE = "EvEye" + ".dataset"
INACTIVE_ELLIPSE_MODULE = "Ellipse" + "MobileNet"
INACTIVE_ELLIPSE_PATH = (
    "common/src/EvEye/model/DavisEyeEllipse/"
    + INACTIVE_ELLIPSE_MODULE
    + ".py"
)
EXPECTED_DATASETS = {
    "DavisWithMaskDataset": "DavisWithMask.DavisWithMaskDataset",
    "TestDataset": "Test.TestDataset",
    "DavisEyeCenterDataset": "DavisEyeCenter.DavisEyeCenterDataset",
    "NpyDavisEyeCenterDataset": "DavisEyeCenter.NpyDavisEyeCenterDataset",
    "DatDavisEyeCenterDataset": "DavisEyeCenter.DatDavisEyeCenterDataset",
    "MemmapDavisEyeCenterDataset": "DavisEyeCenter.MemmapDavisEyeCenterDataset",
    "TestTextDavisEyeDataset": "DavisEyeCenter.TestTextDavisEyeDataset",
    "DavisEyeEllipseDataset": "DavisEyeEllipse.DavisEyeEllipseDataset",
    "DavisEyeEllipseFrameDataset": "DavisEyeEllipse.DavisEyeEllipseFrameDataset",
    "DavisEyeEllipseCenterSequenceDataset": (
        "DavisEyeEllipse.DavisEyeEllipseCenterSequenceDataset"
    ),
}
EXPECTED_WRAPPERS = {
    "__init__.py": ("eveye.dataset", ()),
    "dataset_factory.py": (
        "eveye.dataset.dataset_factory",
        ("DATASET_CLASSES", "worker_init_fn", "make_dataloader", "make_dataset"),
    ),
    "DavisWithMask/__init__.py": ("eveye.dataset.DavisWithMask", ()),
    "DavisWithMask/DavisWithMaskDataset.py": (
        "eveye.dataset.DavisWithMask.DavisWithMaskDataset",
        ("DavisWithMaskDataset",),
    ),
    "Test/__init__.py": ("eveye.dataset.Test", ()),
    "Test/TestDataset.py": (
        "eveye.dataset.Test.TestDataset",
        ("TestDataset",),
    ),
    "DavisEyeCenter/__init__.py": ("eveye.dataset.DavisEyeCenter", ()),
    "DavisEyeCenter/DavisEyeCenterDataset.py": (
        "eveye.dataset.DavisEyeCenter.DavisEyeCenterDataset",
        ("DavisEyeCenterDataset",),
    ),
    "DavisEyeCenter/NpyDavisEyeCenterDataset.py": (
        "eveye.dataset.DavisEyeCenter.NpyDavisEyeCenterDataset",
        ("NpyDavisEyeCenterDataset",),
    ),
    "DavisEyeCenter/DatDavisEyeCenterDataset.py": (
        "eveye.dataset.DavisEyeCenter.DatDavisEyeCenterDataset",
        ("DatDavisEyeCenterDataset",),
    ),
    "DavisEyeCenter/MemmapDavisEyeCenterDataset.py": (
        "eveye.dataset.DavisEyeCenter.MemmapDavisEyeCenterDataset",
        ("MemmapDavisEyeCenterDataset",),
    ),
    "DavisEyeCenter/TestTextDavisEyeDataset.py": (
        "eveye.dataset.DavisEyeCenter.TestTextDavisEyeDataset",
        ("TestTextDavisEyeDataset",),
    ),
    "DavisEyeCenter/losses.py": (
        "eveye.dataset.DavisEyeCenter.losses",
        (
            "OutputHook",
            "MacsEstimationHook",
            "RegularizationLoss",
            "regression_loss",
            "tracking_loss",
            "Losses",
            "process_detector_prediction",
            "p_acc",
        ),
    ),
    "DavisEyeEllipse/__init__.py": ("eveye.dataset.DavisEyeEllipse", ()),
    "DavisEyeEllipse/DavisEyeEllipseDataset.py": (
        "eveye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset",
        ("DavisEyeEllipseDataset",),
    ),
    "DavisEyeEllipse/DavisEyeEllipseFrameDataset.py": (
        "eveye.dataset.DavisEyeEllipse.DavisEyeEllipseFrameDataset",
        ("natural_key", "parse_frame_timestamp", "DavisEyeEllipseFrameDataset"),
    ),
    "DavisEyeEllipse/DavisEyeEllipseCenterSequenceDataset.py": (
        "eveye.dataset.DavisEyeEllipse.DavisEyeEllipseCenterSequenceDataset",
        ("DavisEyeEllipseCenterSequenceDataset",),
    ),
    "DavisEyeEllipse/utils.py": (
        "eveye.dataset.DavisEyeEllipse.utils",
        (
            "cal_ellipse_area",
            "convert_to_ellipse",
            "get_input_size",
            "get_dir",
            "get_3rd_point",
            "get_affine_transform",
            "gaussian2D",
            "draw_umich_gaussian",
            "affine_transform",
            "gaussian_radius",
        ),
    ),
}
# AM-040 disposed of the single allowlisted entry by removing the dead
# EllipseMobileNet owner, so no maintained source may reference the legacy
# dataset package any more.
INACTIVE_LEGACY_DATASET_ALLOWLIST: dict[str, dict[str, str]] = {}
FROZEN_PARTS = {
    "analysis",
    "archive",
    "artifacts",
    "references",
    "hardware_reference",
    "handover",
}


def _maintained_sources():
    for path in ALGORITHM_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".ipynb"}:
            continue
        if path.is_relative_to(LEGACY_ROOT):
            continue
        relative_path = path.relative_to(ALGORITHM_ROOT)
        if FROZEN_PARTS.intersection(relative_path.parts):
            continue
        yield path, relative_path


def test_factory_registry_names_and_origins_are_canonical() -> None:
    factory = import_module("eveye.dataset.dataset_factory")

    assert tuple(factory.DATASET_CLASSES) == tuple(EXPECTED_DATASETS)
    for class_name, relative_module in EXPECTED_DATASETS.items():
        module = import_module(f"eveye.dataset.{relative_module}")
        assert factory.DATASET_CLASSES[class_name] is getattr(module, class_name)
        assert factory.DATASET_CLASSES[class_name].__module__ == (
            f"eveye.dataset.{relative_module}"
        )
    assert Path(factory.__file__).resolve().is_relative_to(CANONICAL_ROOT.resolve())


def test_npy_sample_preserves_shapes_dtypes_and_center_coordinates(tmp_path: Path) -> None:
    module = import_module(
        "eveye.dataset.DavisEyeCenter.NpyDavisEyeCenterDataset"
    )
    data = np.arange(48, dtype=np.float64).reshape(2, 3, 4, 2)
    center = np.asarray([[12.5, 20.25], [31.75, 42.5]], dtype=np.float64)
    close = np.asarray([0.0, 1.0], dtype=np.float64)
    for directory, value in (
        ("np_data", data),
        ("np_label", center),
        ("np_close", close),
    ):
        target = tmp_path / "val" / directory
        target.mkdir(parents=True)
        np.save(target / "0.npy", value)

    dataset = module.NpyDavisEyeCenterDataset(tmp_path, split="val")
    sample_data, sample_center, sample_close = dataset[0]

    assert len(dataset) == 1
    assert tuple(sample_data.shape) == data.shape
    assert tuple(sample_center.shape) == center.shape
    assert tuple(sample_close.shape) == close.shape
    assert sample_data.dtype is torch.float32
    assert sample_center.dtype is torch.float32
    assert sample_close.dtype is torch.float32
    torch.testing.assert_close(sample_center, torch.tensor(center, dtype=torch.float32))


def test_ellipse_coordinate_order_is_preserved() -> None:
    utils = import_module("eveye.dataset.DavisEyeEllipse.utils")

    ellipse = utils.convert_to_ellipse(
        np.asarray([40000, 12.5, 20.25, 8.0, 6.0, 45.0], dtype=np.float32)
    )

    assert ellipse == ((12.5, 20.25), (8.0, 6.0), 45.0)


def test_legacy_dataset_owner_is_retired() -> None:
    """AM-060: the legacy EvEye dataset owner and all its wrappers are removed."""
    assert not LEGACY_ROOT.exists()
    assert not (ALGORITHM_ROOT / "common" / "src" / "EvEye").exists()


def test_dataset_owner_has_no_upward_imports_and_demos_are_extracted() -> None:
    forbidden_prefixes = (
        "EvEye.model",
        "eveye.engine.callback",
        "eveye.engine.logger",
        "eveye.common",
        "eveye.engine",
        "eveye.event",
        "src",
    )
    for path in CANONICAL_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
        assert not any(
            module == prefix or module.startswith(prefix + ".")
            for module in imported_modules
            for prefix in forbidden_prefixes
        ), f"upward dependency in {path}"
        assert not any(
            module.startswith("EvEye.utils.") for module in imported_modules
        ), f"legacy utility dependency in {path}"

    for relative_path in (
        "DavisEyeCenter/DavisEyeCenterDataset.py",
        "DavisEyeCenter/MemmapDavisEyeCenterDataset.py",
    ):
        tree = ast.parse((CANONICAL_ROOT / relative_path).read_text())
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
            for node in ast.walk(tree)
        )

    inspection_tool = import_module("eveye.engine.tools.inspect_davis_eye_center")
    assert callable(inspection_tool.inspect_davis_eye_center)
    assert callable(inspection_tool.inspect_memmap_davis_eye_center)
    assert callable(inspection_tool.main)


def test_maintained_legacy_census_matches_inactive_disposition() -> None:
    found_legacy_references = {}
    import_consumers = []
    inactive_names = {INACTIVE_ELLIPSE_MODULE, INACTIVE_ELLIPSE_MODULE + "V3"}

    for path, relative_path in _maintained_sources():
        text = path.read_text()
        matches = [
            line.strip()
            for line in text.splitlines()
            if LEGACY_DATASET_PACKAGE in line
        ]
        if matches:
            found_legacy_references[relative_path.as_posix()] = matches

        if relative_path.as_posix() == INACTIVE_ELLIPSE_PATH:
            continue
        if path.suffix == ".py":
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if INACTIVE_ELLIPSE_MODULE in module or any(
                        alias.name in inactive_names for alias in node.names
                    ):
                        import_consumers.append(relative_path.as_posix())
                elif isinstance(node, ast.Import) and any(
                    INACTIVE_ELLIPSE_MODULE in alias.name for alias in node.names
                ):
                    import_consumers.append(relative_path.as_posix())
        else:
            notebook = json.loads(text)
            code = "".join(
                source
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
                for source in cell.get("source", [])
            )
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith(("from ", "import ")) and any(
                    name in stripped for name in inactive_names
                ):
                    import_consumers.append(relative_path.as_posix())

    expected_references = {
        relative_path: [entry["line"]]
        for relative_path, entry in INACTIVE_LEGACY_DATASET_ALLOWLIST.items()
    }
    assert found_legacy_references == expected_references
    assert all(
        entry["reason"] for entry in INACTIVE_LEGACY_DATASET_ALLOWLIST.values()
    )
    assert import_consumers == []
