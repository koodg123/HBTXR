from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "hybrid"
PACKAGE_DIR = SRC_ROOT / "evaluation"
SPEC = importlib.util.spec_from_file_location(
    "evaluation",
    PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(PACKAGE_DIR)],
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

BBox = MODULE.BBox
Center = MODULE.Center
Domain = MODULE.Domain
Ellipse = MODULE.Ellipse
Error = MODULE.EvaluationContractError
Policy = MODULE.EvaluationPolicy
Task = MODULE.EvaluationTask
Size = MODULE.Size
convert = MODULE.convert_record

ROI = Size(80, 60)
SENSOR = Size(640, 480)
POST = Size(64, 64)


class EvaluationBridgeTest(unittest.TestCase):
    def test_center_roi_to_sensor_corners_and_center(self) -> None:
        for source, expected in (
            ((0.0, 0.0), (0.0, 0.0)),
            ((40.0, 30.0), (320.0, 240.0)),
            ((80.0, 60.0), (640.0, 480.0)),
        ):
            with self.subTest(source=source):
                result = convert(
                    Center(Domain.ROI, ROI, *source),
                    Domain.SENSOR, SENSOR, Task.CENTER,
                )
                self.assertEqual((result.x, result.y), expected)

    def test_sensor_to_post_uses_per_axis_scale(self) -> None:
        result = convert(
            Center(Domain.SENSOR, SENSOR, 320.0, 120.0),
            Domain.POST_TRANSFORM, POST, Task.CENTER,
        )
        self.assertEqual((result.x, result.y), (32.0, 16.0))

    def test_bbox_scales_position_and_dimensions(self) -> None:
        result = convert(
            BBox(Domain.ROI, ROI, 10.0, 5.0, 20.0, 15.0),
            Domain.SENSOR, SENSOR, Task.BBOX,
        )
        self.assertEqual(
            (result.x, result.y, result.width, result.height),
            (80.0, 40.0, 160.0, 120.0),
        )

    def test_ellipse_scales_axes_and_preserves_theta(self) -> None:
        result = convert(
            Ellipse(Domain.SENSOR, SENSOR, 320.0, 240.0, 160.0, 120.0, 0.75),
            Domain.POST_TRANSFORM, POST, Task.ELLIPSE,
        )
        self.assertEqual(
            (result.cx, result.cy, result.axis_x, result.axis_y, result.theta),
            (32.0, 32.0, 16.0, 16.0, 0.75),
        )

    def test_rejects_unsupported_size_and_domain_mismatch(self) -> None:
        with self.assertRaises(Error):
            Size(128, 128)
        with self.assertRaises(Error):
            Center(Domain.SENSOR, ROI, 1.0, 1.0)
        with self.assertRaises(Error):
            convert(
                Center(Domain.ROI, ROI, 1.0, 1.0),
                Domain.SENSOR, POST, Task.CENTER,
            )
        with self.assertRaises(Error):
            convert(
                Center(Domain.ROI, ROI, 1.0, 1.0),
                Domain.SENSOR, (640, 480), Task.CENTER,
            )

    def test_rejects_task_type_mismatch(self) -> None:
        with self.assertRaises(Error):
            convert(
                Center(Domain.ROI, ROI, 1.0, 1.0),
                Domain.SENSOR, SENSOR, Task.BBOX,
            )

    def test_rejects_invalid_geometry(self) -> None:
        for factory in (
            lambda: BBox(Domain.ROI, ROI, 0.0, 0.0, 0.0, 1.0),
            lambda: Ellipse(Domain.ROI, ROI, 1.0, 1.0, 0.0, 1.0, 0.0),
            lambda: Center(Domain.ROI, ROI, "x", 1.0),
            lambda: Center(Domain.ROI, ROI, 10**10000, 1.0),
        ):
            with self.assertRaises(Error):
                factory()

    def test_policy_contract(self) -> None:
        policy = Policy(invalid_sample="exclude", subject_weighting="micro", aggregation="micro")
        self.assertEqual(policy.pixel_tolerances, (1, 5, 10))
        for kwargs in (
            {"invalid_sample": "ignore"},
            {"subject_weighting": "weighted"},
            {"aggregation": "subject"},
            {"invalid_sample": []},
            {"subject_weighting": []},
            {"aggregation": []},
            {"pixel_tolerances": (1, 3, 5)},
            {"pixel_tolerances": (True, 5, 10)},
            {"pixel_tolerances": (1.0, 5.0, 10.0)},
        ):
            with self.assertRaises(Error):
                Policy(**kwargs)

    def test_records_and_policy_are_frozen(self) -> None:
        center = Center(Domain.ROI, ROI, 1.0, 2.0)
        with self.assertRaises(FrozenInstanceError):
            center.x = 3.0
        with self.assertRaises(FrozenInstanceError):
            Policy().aggregation = "micro"


if __name__ == "__main__":
    unittest.main()
