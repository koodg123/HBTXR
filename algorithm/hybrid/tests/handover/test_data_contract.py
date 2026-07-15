from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

HYBRID_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = HYBRID_ROOT / "src"
CONTRACTS_PATH = SRC_ROOT / "data" / "contracts.py"
CONTRACTS_SPEC = importlib.util.spec_from_file_location(
    "_hbtxr_data_contracts_under_test", CONTRACTS_PATH
)
if CONTRACTS_SPEC is None or CONTRACTS_SPEC.loader is None:
    raise RuntimeError(f"cannot load contracts module from {CONTRACTS_PATH}")
CONTRACTS_MODULE = importlib.util.module_from_spec(CONTRACTS_SPEC)
sys.modules[CONTRACTS_SPEC.name] = CONTRACTS_MODULE
CONTRACTS_SPEC.loader.exec_module(CONTRACTS_MODULE)

CanonicalSampleMetadata = CONTRACTS_MODULE.CanonicalSampleMetadata
ContractError = CONTRACTS_MODULE.ContractError
bind_event_tensor = CONTRACTS_MODULE.bind_event_tensor
resolve_event_tensor_spec = CONTRACTS_MODULE.resolve_event_tensor_spec

SHA256 = "a" * 64
EVENT_CASES = (
    ((2, 64, 64), "uint16", ("polarity", "height", "width"),
     "separate-negative-positive", "single-window-count"),
    ((2, 64, 64), "float32", ("polarity", "height", "width"),
     "separate-negative-positive", "single-window-count"),
    ((50, 2, 64, 64), "float32", ("sequence", "polarity", "height", "width"),
     "separate-negative-positive", "ordered-50-frame-sequence"),
    ((100, 2, 64, 64), "float32", ("sequence", "polarity", "height", "width"),
     "separate-negative-positive", "ordered-100-frame-sequence"),
    ((30, 3, 64, 64), "float32", ("sequence", "voxel-time-bin", "height", "width"),
     "signed-voxel", "ordered-30-frame-signed-voxel-sequence"),
)


def resolve_case(index: int = 0, **overrides: object):
    shape, dtype, axes, polarity, temporal = EVENT_CASES[index]
    values = {
        "shape": shape,
        "dtype": dtype,
        "axes": axes,
        "polarity_semantics": polarity,
        "temporal_semantics": temporal,
    }
    values.update(overrides)
    return resolve_event_tensor_spec(**values)


def metadata(**overrides: object) -> CanonicalSampleMetadata:
    values = {
        "event_ref": "events/sample-001",
        "frame_ref": "frames/sample-001.png",
        "target_ref": "targets/sample-001.json",
        "subject_id": "subject-01",
        "eye": "left",
        "sequence_id": "sequence-01",
        "start_timestamp_us": 100,
        "end_timestamp_us": 200,
        "dataset": "handover-fixture",
        "split": "train",
        "manifest_sha256": SHA256,
        "sample_id": "sample-001",
    }
    values.update(overrides)
    return CanonicalSampleMetadata(**values)


class EventTensorContractTests(unittest.TestCase):
    def test_accepts_four_layouts_and_both_count_volume_dtypes(self) -> None:
        for index, case in enumerate(EVENT_CASES):
            with self.subTest(case=case):
                spec = resolve_case(index)
                self.assertEqual(
                    (spec.shape, spec.dtype, spec.axes,
                     spec.polarity_semantics, spec.temporal_semantics),
                    case,
                )

    def test_binding_preserves_payload_identity(self) -> None:
        payload = object()
        case = EVENT_CASES[0]
        contracted = bind_event_tensor(
            payload,
            shape=case[0],
            dtype=case[1],
            axes=case[2],
            polarity_semantics=case[3],
            temporal_semantics=case[4],
        )
        self.assertIs(contracted.payload, payload)

    def test_rejects_unsupported_shape(self) -> None:
        with self.assertRaises(ContractError):
            resolve_case(shape=(2, 32, 32))

    def test_rejects_unsupported_dtype(self) -> None:
        with self.assertRaises(ContractError):
            resolve_case(dtype="float64")

    def test_rejects_axis_reordering(self) -> None:
        with self.assertRaises(ContractError):
            resolve_case(axes=("height", "width", "polarity"))

    def test_rejects_ambiguous_polarity(self) -> None:
        with self.assertRaises(ContractError):
            resolve_case(polarity_semantics="unknown")

    def test_rejects_temporal_semantic_mismatch(self) -> None:
        with self.assertRaises(ContractError):
            resolve_case(temporal_semantics="ordered-50-frame-sequence")


class CanonicalSampleMetadataTests(unittest.TestCase):
    def test_accepts_complete_metadata_and_optional_frame(self) -> None:
        self.assertEqual(metadata().sample_id, "sample-001")
        self.assertIsNone(metadata(frame_ref=None).frame_ref)

    def test_rejects_missing_split_provenance(self) -> None:
        for field_name, value in (
            ("dataset", ""), ("split", ""), ("manifest_sha256", ""),
            ("manifest_sha256", "A" * 64),
            ("manifest_sha256", "a" * 63),
            ("manifest_sha256", None), ("manifest_sha256", 123),
        ):
            with self.subTest(field=field_name, value=value):
                with self.assertRaises(ContractError):
                    metadata(**{field_name: value})

    def test_rejects_missing_identity_fields(self) -> None:
        for field_name in ("event_ref", "target_ref", "subject_id", "sequence_id", "sample_id"):
            with self.subTest(field=field_name):
                with self.assertRaises(ContractError):
                    metadata(**{field_name: " "})

    def test_rejects_invalid_eye(self) -> None:
        for eye in ("LEFT", "unknown", ""):
            with self.subTest(eye=eye):
                with self.assertRaises(ContractError):
                    metadata(eye=eye)

    def test_rejects_invalid_timestamps(self) -> None:
        for start, end in ((-1, 10), (100, 100), (101, 100), (False, 10), (0, True)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ContractError):
                    metadata(start_timestamp_us=start, end_timestamp_us=end)

    def test_metadata_is_frozen(self) -> None:
        sample = metadata()
        with self.assertRaises(FrozenInstanceError):
            sample.sample_id = "replacement"


class ImportBoundaryTests(unittest.TestCase):
    def test_contracts_imports_when_numpy_is_unavailable(self) -> None:
        script = """
import builtins
import importlib.util
import pathlib
import sys
original_import = builtins.__import__
def reject_numpy(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise ModuleNotFoundError("NumPy intentionally unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_numpy
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("isolated_data_contracts", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(SRC_ROOT / "data" / "contracts.py")],
            cwd=HYBRID_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
