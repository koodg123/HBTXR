from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO / "hardware" / "configs" / "xr_accel"
MODULE_PATH = REPO / "hardware" / "tools" / "xr_accel_config.py"
SPEC = importlib.util.spec_from_file_location("xr_accel_config", MODULE_PATH)
assert SPEC and SPEC.loader
xr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = xr
SPEC.loader.exec_module(xr)


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.json"))
        if path.name != "schema.json"
    }


class XRAccelConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_hashes = _tree_hash(SOURCE_ROOT)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "xr_accel"
        shutil.copytree(SOURCE_ROOT, self.root)

    def tearDown(self) -> None:
        self.assertEqual(self.original_hashes, _tree_hash(SOURCE_ROOT))
        self.temp.cleanup()

    def load(self, relative: str) -> dict:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def write(self, relative: str, value: dict) -> None:
        (self.root / relative).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def normalize(self, name: str = "zcu104_hbtxr_option_a_cyclic") -> dict:
        return xr.normalize_experiment(f"configs/experiments/{name}.json", config_root=self.root)

    def assert_invalid(self, error: str = "") -> None:
        with self.assertRaisesRegex(xr.XRConfigError, error):
            self.normalize()

    def test_schema_policy_is_explicit_and_has_no_defaults(self) -> None:
        schema = self.load("schema.json")
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", schema["$schema"])
        policy = schema["x-normalization-policy"]
        self.assertEqual("hbtxr.xr-accel-manifest/v1", policy["schema_version"])
        self.assertEqual("HANDOVER/XR_Accel", policy["source_repository"])
        self.assertIn("source_repository", schema["properties"]["provenance"]["required"])
        self.assertEqual("5041401aec53a1c85a58661203b8bee60dacbfca", policy["source_revision"])
        self.assertEqual(32, policy["axi_data_width_bits"])
        self.assertEqual("runtime_mmio", policy["memory_topology"])

        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item) for item in value.values()), set())
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value), set())
            return set()

        self.assertNotIn("default", keys(schema))

    def test_all_eight_experiments_reach_exactly_eighteen_references(self) -> None:
        manifests = xr.normalize_all_experiments(config_root=self.root)
        reached = {name for item in manifests for name in item["provenance"]["source_hashes"]}
        self.assertEqual(8, len(manifests))
        self.assertEqual(18, len(reached))
        self.assertEqual({"hbtxr.xr-accel-manifest/v1"}, {m["schema_version"] for m in manifests})

    def test_normalized_manifest_exact_contract_and_inheritance(self) -> None:
        manifest = xr.normalize_experiment(
            "configs/experiments/zcu104_deit_tiny_baseline_cyclic.json", config_root=self.root
        )
        self.assertEqual(
            {"schema_version", "experiment_name", "links", "records", "clocks", "interface", "memory_topology", "provenance"},
            set(manifest),
        )
        self.assertEqual(192, manifest["records"]["model"]["embed_dim"])
        self.assertEqual("cyclic_deit", manifest["records"]["model"]["kind"])
        self.assertEqual(32, manifest["interface"]["axi_data_width_bits"])
        self.assertEqual("runtime_mmio", manifest["interface"]["control"])
        self.assertEqual("structural-reference-only", manifest["provenance"]["evidence_policy"])
        self.assertEqual("HANDOVER/XR_Accel", manifest["provenance"]["source_repository"])
        self.assertEqual(
            ["dma", "gpio", "clock", "accelerator"],
            [item["name"] for item in manifest["memory_topology"]["regions"]],
        )
        xr.validate_normalized_manifest(manifest)
        bad = copy.deepcopy(manifest)
        bad["implicit"] = True
        with self.assertRaisesRegex(xr.XRConfigError, "unknown fields"):
            xr.validate_normalized_manifest(bad)

    def test_schema_contract_drift_is_rejected(self) -> None:
        cases = [
            (("provenance", "source_repository"), "other/repo"),
            (("provenance", "source_revision"), "0" * 40),
            (("provenance", "evidence_policy"), "promoted"),
            (("interface", "control"), "implicit"),
            (("memory_topology", "type"), "implicit"),
        ]
        for path, replacement in cases:
            with self.subTest(path=path):
                schema = self.load("schema.json")
                schema["properties"][path[0]]["properties"][path[1]]["const"] = replacement
                self.write("schema.json", schema)
                self.assert_invalid("policy mismatch")
                shutil.rmtree(self.root); shutil.copytree(SOURCE_ROOT, self.root)

    def test_normalized_records_and_links_are_revalidated(self) -> None:
        manifest = self.normalize()
        bad = copy.deepcopy(manifest)
        bad["records"]["model"]["injected"] = True
        with self.assertRaisesRegex(xr.XRConfigError, "unknown fields"):
            xr.validate_normalized_manifest(bad)
        bad = copy.deepcopy(manifest)
        bad["links"]["model"] = "configs/models/hbtxr_option_b.json"
        with self.assertRaisesRegex(xr.XRConfigError, "filename/name mismatch|link mismatch"):
            xr.validate_normalized_manifest(bad)
        bad = copy.deepcopy(manifest)
        bad["records"]["experiment"]["model"] = "configs/models/hbtxr_option_b.json"
        with self.assertRaisesRegex(xr.XRConfigError, "link mismatch"):
            xr.validate_normalized_manifest(bad)
        bad = copy.deepcopy(manifest)
        bad["experiment_name"] = "different"
        with self.assertRaisesRegex(xr.XRConfigError, "experiment_name"):
            xr.validate_normalized_manifest(bad)

    def test_unknown_and_missing_fields_are_rejected(self) -> None:
        experiment = self.load("experiments/zcu104_hbtxr_option_a_cyclic.json")
        experiment["surprise"] = 1
        self.write("experiments/zcu104_hbtxr_option_a_cyclic.json", experiment)
        self.assert_invalid("unknown fields")
        experiment.pop("surprise")
        experiment.pop("target")
        self.write("experiments/zcu104_hbtxr_option_a_cyclic.json", experiment)
        self.assert_invalid("missing fields")

    def test_unknown_nested_fields_are_rejected(self) -> None:
        cases = [
            ("models/hbtxr_option_a.json", "frame"),
            ("models/hbtxr_option_a.json", "event"),
            ("models/hbtxr_option_a.json", "stream_tiling"),
            ("targets/zcu104.json", "runtime"),
            ("experiments/zcu104_hbtxr_option_a_cyclic.json", "artifacts"),
        ]
        for relative, nested in cases:
            with self.subTest(nested=nested):
                value = self.load(relative)
                value[nested]["injected"] = 1
                self.write(relative, value)
                self.assert_invalid("unknown fields")
                shutil.rmtree(self.root)
                shutil.copytree(SOURCE_ROOT, self.root)

    def test_unsafe_path_variants_are_rejected(self) -> None:
        variants = [
            "/tmp/model.json", "C:/tmp/model.json", "C:\\tmp\\model.json",
            "\\\\server\\share\\model.json", "configs/models/../targets/zcu104.json",
            "configs//models/hbtxr_option_a.json", "configs\\models\\hbtxr_option_a.json",
        ]
        for path in variants:
            with self.subTest(path=path):
                value = self.load("experiments/zcu104_hbtxr_option_a_cyclic.json")
                value["model"] = path
                self.write("experiments/zcu104_hbtxr_option_a_cyclic.json", value)
                self.assert_invalid("path|namespace|portable")
                shutil.rmtree(self.root)
                shutil.copytree(SOURCE_ROOT, self.root)

    def test_unresolved_wrong_category_and_symlink_escape_are_rejected(self) -> None:
        value = self.load("experiments/zcu104_hbtxr_option_a_cyclic.json")
        value["model"] = "configs/models/missing.json"
        self.write("experiments/zcu104_hbtxr_option_a_cyclic.json", value)
        self.assert_invalid("unresolved")
        value["model"] = "configs/targets/zcu104.json"
        self.write("experiments/zcu104_hbtxr_option_a_cyclic.json", value)
        self.assert_invalid("namespace")

        outside = Path(self.temp.name) / "outside.json"
        outside.write_text(json.dumps(self.load("models/hbtxr_option_a.json")), encoding="utf-8")
        link = self.root / "models" / "escape.json"
        link.symlink_to(outside)
        value["model"] = "configs/models/escape.json"
        self.write("experiments/zcu104_hbtxr_option_a_cyclic.json", value)
        self.assert_invalid("escapes")

    def test_unsupported_board_part_and_filename_name_mismatch_are_rejected(self) -> None:
        target = self.load("targets/zcu104.json")
        target["part"] = "xczu9eg-ffvb1156-2-e"
        self.write("targets/zcu104.json", target)
        self.assert_invalid("unsupported board/part")
        shutil.rmtree(self.root); shutil.copytree(SOURCE_ROOT, self.root)
        model = self.load("models/hbtxr_option_a.json")
        model["name"] = "renamed"
        self.write("models/hbtxr_option_a.json", model)
        self.assert_invalid("filename/name mismatch")

    def test_inheritance_cycle_is_rejected(self) -> None:
        model = self.load("models/hbtxr_option_a.json")
        model["inherits"] = "configs/models/hbtxr_option_b.json"
        self.write("models/hbtxr_option_a.json", model)
        parent = self.load("models/hbtxr_option_b.json")
        parent["inherits"] = "configs/models/hbtxr_option_a.json"
        self.write("models/hbtxr_option_b.json", parent)
        self.assert_invalid("inheritance cycle")

    def test_invalid_clocks_are_rejected(self) -> None:
        for value in (0, -1, float("inf"), "6.0", True):
            with self.subTest(value=value):
                design = self.load("designs/hbtxr_cyclic_zcu104.json")
                design["clock_period_ns"] = value
                self.write("designs/hbtxr_cyclic_zcu104.json", design)
                self.assert_invalid("positive")
                shutil.rmtree(self.root); shutil.copytree(SOURCE_ROOT, self.root)
        with self.assertRaisesRegex(xr.XRConfigError, "positive finite"):
            xr._positive(10**10000, "clock")

    def test_noncanonical_and_duplicate_mmio_are_rejected(self) -> None:
        target = self.load("targets/zcu104.json")
        target["runtime"]["dma_base"] = "0xa0000000"
        self.write("targets/zcu104.json", target)
        self.assert_invalid("noncanonical MMIO")
        shutil.rmtree(self.root); shutil.copytree(SOURCE_ROOT, self.root)
        target = self.load("targets/zcu104.json")
        target["runtime"]["gpio_base"] = target["runtime"]["dma_base"]
        self.write("targets/zcu104.json", target)
        self.assert_invalid("distinct")

    def test_artifact_path_injection_is_rejected(self) -> None:
        experiment = self.load("experiments/zcu104_hbtxr_option_a_cyclic.json")
        experiment["artifacts"]["build_subdir"] = ["target", "../../escape"]
        self.write("experiments/zcu104_hbtxr_option_a_cyclic.json", experiment)
        self.assert_invalid("unsupported value")


if __name__ == "__main__":
    unittest.main()
