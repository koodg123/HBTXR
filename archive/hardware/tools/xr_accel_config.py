#!/usr/bin/env python3
"""Fail-closed normalizer for the versioned XR accelerator references."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class XRConfigError(ValueError):
    """Raised when an XR reference or normalized manifest is invalid."""


_SCHEMA_VERSION = "hbtxr.xr-accel-manifest/v1"
_SOURCE_REPOSITORY = "HANDOVER/XR_Accel"
_REVISION = "5041401aec53a1c85a58661203b8bee60dacbfca"
_POLICY_KEYS = {
    "schema_version", "source_repository", "source_revision", "evidence_policy", "supported_targets",
    "axi_data_width_bits", "interface_control", "memory_topology",
    "board_clock_rule", "required_runtime_regions",
}
_EXPECTED_POLICY = {
    "schema_version": _SCHEMA_VERSION,
    "source_repository": _SOURCE_REPOSITORY,
    "source_revision": _REVISION,
    "evidence_policy": "structural-reference-only",
    "supported_targets": [{
        "name": "zcu104", "part": "xczu7ev-ffvc1156-2-e",
        "board_part": "xilinx.com:zcu104:part0:1.1",
    }],
    "axi_data_width_bits": 32,
    "interface_control": "runtime_mmio",
    "memory_topology": "runtime_mmio",
    "board_clock_rule": "design_override_or_target",
    "required_runtime_regions": ["dma", "gpio", "clock", "accelerator"],
}
_EXPERIMENT_FIELDS = {"name", "model", "target", "design", "top_module", "artifacts"}
_DESIGN_FIELDS = {
    "name", "mode", "architecture", "allow_model_variant", "clock_period_ns",
    "board_clock_period_ns", "allow_board_clock_override", "case_root", "fifo_depth",
    "strategy_order", "benchmark_units", "parallelism_scale", "implementation_stage",
    "hls_cflags", "optimization_goal", "optimization_notes",
}
_MODEL_FIELDS = {
    "inherits", "name", "kind", "input_channels", "input_size", "patch_size", "seq_len",
    "patch_dim", "embed_dim", "mlp_ratio", "hidden_dim", "depth", "search_depth",
    "track_cut_point", "heads", "classes", "head_output_width", "frame", "event",
    "output_packet_words", "stream_tiling", "block_datawidth_in", "class_tokens",
}
_TARGET_FIELDS = {"name", "family", "shell_mode", "part", "board_part", "clock_period_ns", "runtime"}
_FRAME_FIELDS = {"channels", "height", "width", "patch_dim", "tokens"}
_TILING_FIELDS = {
    "tip_default", "top_default", "head_top", "patch_cip", "default_cip", "cop_default",
    "input_word_bits", "output_word_bits",
}
_ARTIFACT_FIELDS = {"instances_subdir", "reports_subdir", "build_subdir"}
_RUNTIME_FIELDS = {"dma_base", "gpio_base", "clock_base", "accelerator_base"}
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MMIO_RE = re.compile(r"^0x[0-9A-F]{8}$")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise XRConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise XRConfigError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise XRConfigError(f"{path}: root must be an object")
    return value


def _exact_keys(record: Mapping[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    unknown = set(record) - allowed
    missing = required - set(record)
    if unknown:
        raise XRConfigError(f"{where}: unknown fields: {sorted(unknown)}")
    if missing:
        raise XRConfigError(f"{where}: missing fields: {sorted(missing)}")


def _string(value: Any, where: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or (identifier and not _NAME_RE.fullmatch(value)):
        raise XRConfigError(f"{where}: expected non-empty{' portable' if identifier else ''} string")
    return value


def _positive(value: Any, where: str, *, integer: bool = False) -> None:
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or value <= 0:
        raise XRConfigError(f"{where}: expected positive {'integer' if integer else 'finite number'}")
    if not integer:
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError) as exc:
            raise XRConfigError(f"{where}: expected positive finite number") from exc
        if not finite:
            raise XRConfigError(f"{where}: expected positive finite number")


def _string_list(value: Any, where: str, allowed: set[str] | None = None) -> None:
    if not isinstance(value, list) or not value:
        raise XRConfigError(f"{where}: expected non-empty string array")
    for item in value:
        _string(item, where)
        if allowed is not None and item not in allowed:
            raise XRConfigError(f"{where}: unsupported value {item!r}")


def _relative_path(value: Any, where: str) -> str:
    text = _string(value, where)
    if "\\" in text or "\x00" in text or re.match(r"^[A-Za-z]:", text) or text.startswith("//"):
        raise XRConfigError(f"{where}: non-portable path")
    parts = text.split("/")
    if PurePosixPath(text).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise XRConfigError(f"{where}: path must be normalized and relative")
    return text


def _resolve(root: Path, reference: Any, category: str) -> tuple[Path, str]:
    text = _relative_path(reference, f"{category} reference")
    parts = text.split("/")
    if len(parts) != 3 or parts[:2] != ["configs", category] or not parts[2].endswith(".json"):
        raise XRConfigError(f"{category} reference: wrong namespace {text!r}")
    if not _NAME_RE.fullmatch(parts[2][:-5]):
        raise XRConfigError(f"{category} reference: invalid filename")
    root_resolved = root.resolve(strict=True)
    candidate = root / category / parts[2]
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise XRConfigError(f"{category} reference: unresolved {text!r}") from exc
    if not resolved.is_relative_to(root_resolved):
        raise XRConfigError(f"{category} reference: escapes configuration root")
    if not resolved.is_file():
        raise XRConfigError(f"{category} reference: not a file")
    return resolved, text


def _filename_name(record: Mapping[str, Any], reference: str, where: str) -> None:
    name = _string(record.get("name"), f"{where}.name", identifier=True)
    if name != PurePosixPath(reference).stem:
        raise XRConfigError(f"{where}: filename/name mismatch")


def _validate_experiment(value: dict[str, Any], reference: str) -> None:
    _exact_keys(value, _EXPERIMENT_FIELDS, {"name", "model", "target", "design"}, "experiment")
    _filename_name(value, reference, "experiment")
    for category in ("model", "target", "design"):
        _relative_path(value[category], f"experiment.{category}")
    if "top_module" in value and not _IDENT_RE.fullmatch(_string(value["top_module"], "experiment.top_module")):
        raise XRConfigError("experiment.top_module: invalid identifier")
    if "artifacts" in value:
        artifacts = value["artifacts"]
        if not isinstance(artifacts, dict):
            raise XRConfigError("experiment.artifacts: expected object")
        _exact_keys(artifacts, _ARTIFACT_FIELDS, _ARTIFACT_FIELDS, "experiment.artifacts")
        tokens = {"target", "model", "design", "experiment"}
        for key in sorted(_ARTIFACT_FIELDS):
            _string_list(artifacts[key], f"experiment.artifacts.{key}", tokens)


def _validate_model(value: dict[str, Any], reference: str, *, merged: bool) -> None:
    required = {"name"}
    if merged:
        required |= {
            "input_channels", "input_size", "patch_size", "seq_len", "patch_dim", "embed_dim",
            "mlp_ratio", "depth", "heads", "classes", "head_output_width", "stream_tiling",
            "block_datawidth_in",
        }
    _exact_keys(value, _MODEL_FIELDS, required, "model")
    _filename_name(value, reference, "model")
    if "inherits" in value:
        _relative_path(value["inherits"], "model.inherits")
    integer_fields = _MODEL_FIELDS & {
        "input_channels", "input_size", "patch_size", "seq_len", "patch_dim", "embed_dim",
        "hidden_dim", "depth", "search_depth", "track_cut_point", "heads", "classes",
        "head_output_width", "output_packet_words", "class_tokens",
    }
    for key in integer_fields & value.keys():
        _positive(value[key], f"model.{key}", integer=True)
    if "mlp_ratio" in value:
        _positive(value["mlp_ratio"], "model.mlp_ratio")
    for key in ("kind",):
        if key in value:
            _string(value[key], f"model.{key}", identifier=True)
    for key in ("frame", "event"):
        if key in value:
            nested = value[key]
            if not isinstance(nested, dict):
                raise XRConfigError(f"model.{key}: expected object")
            _exact_keys(nested, _FRAME_FIELDS, _FRAME_FIELDS, f"model.{key}")
            for field in _FRAME_FIELDS:
                _positive(nested[field], f"model.{key}.{field}", integer=True)
    if "stream_tiling" in value:
        tiling = value["stream_tiling"]
        if not isinstance(tiling, dict) or not tiling:
            raise XRConfigError("model.stream_tiling: expected non-empty object")
        _exact_keys(tiling, _TILING_FIELDS, set(), "model.stream_tiling")
        for key, item in tiling.items():
            _positive(item, f"model.stream_tiling.{key}", integer=True)
    if "block_datawidth_in" in value:
        widths = value["block_datawidth_in"]
        if not isinstance(widths, list) or not widths:
            raise XRConfigError("model.block_datawidth_in: expected non-empty array")
        for item in widths:
            _positive(item, "model.block_datawidth_in", integer=True)


def _validate_design(value: dict[str, Any], reference: str) -> None:
    required = {"name", "mode", "allow_model_variant", "clock_period_ns", "fifo_depth", "benchmark_units", "parallelism_scale"}
    _exact_keys(value, _DESIGN_FIELDS, required, "design")
    _filename_name(value, reference, "design")
    _string(value["mode"], "design.mode", identifier=True)
    if type(value["allow_model_variant"]) is not bool:
        raise XRConfigError("design.allow_model_variant: expected boolean")
    _positive(value["clock_period_ns"], "design.clock_period_ns")
    _positive(value["fifo_depth"], "design.fifo_depth", integer=True)
    _positive(value["parallelism_scale"], "design.parallelism_scale")
    _string_list(value["benchmark_units"], "design.benchmark_units")
    for key in ("strategy_order", "hls_cflags", "optimization_notes"):
        if key in value:
            _string_list(value[key], f"design.{key}")
    for key in ("architecture", "implementation_stage", "optimization_goal"):
        if key in value:
            _string(value[key], f"design.{key}")
    if "case_root" in value:
        _relative_path(value["case_root"], "design.case_root")
    clock_fields = {"board_clock_period_ns", "allow_board_clock_override"}
    if bool(clock_fields & value.keys()) != clock_fields.issubset(value):
        raise XRConfigError("design: board clock and override flag must appear together")
    if "board_clock_period_ns" in value:
        _positive(value["board_clock_period_ns"], "design.board_clock_period_ns")
        if value["allow_board_clock_override"] is not True:
            raise XRConfigError("design.allow_board_clock_override: must be true for override")


def _validate_target(value: dict[str, Any], reference: str, policy: Mapping[str, Any]) -> None:
    required = _TARGET_FIELDS
    _exact_keys(value, _TARGET_FIELDS, required, "target")
    _filename_name(value, reference, "target")
    for key in ("family", "shell_mode", "part", "board_part"):
        _string(value[key], f"target.{key}")
    _positive(value["clock_period_ns"], "target.clock_period_ns")
    supported = {(x["name"], x["part"], x["board_part"]) for x in policy["supported_targets"]}
    if (value["name"], value["part"], value["board_part"]) not in supported:
        raise XRConfigError("target: unsupported board/part tuple")
    runtime = value["runtime"]
    if not isinstance(runtime, dict):
        raise XRConfigError("target.runtime: expected object")
    _exact_keys(runtime, _RUNTIME_FIELDS, _RUNTIME_FIELDS, "target.runtime")
    addresses = []
    for key in sorted(_RUNTIME_FIELDS):
        address = runtime[key]
        if not isinstance(address, str) or not _MMIO_RE.fullmatch(address):
            raise XRConfigError(f"target.runtime.{key}: noncanonical MMIO address")
        addresses.append(address)
    if len(set(addresses)) != len(addresses):
        raise XRConfigError("target.runtime: MMIO addresses must be distinct")


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(parent)
    for key, value in child.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_policy(schema_path: Path) -> dict[str, Any]:
    schema = _read_json(schema_path)
    policy = schema.get("x-normalization-policy")
    if not isinstance(policy, dict):
        raise XRConfigError("schema: missing x-normalization-policy")
    _exact_keys(policy, _POLICY_KEYS, _POLICY_KEYS, "schema.x-normalization-policy")
    if policy != _EXPECTED_POLICY:
        raise XRConfigError("schema: normalization policy does not match pinned contract")
    try:
        properties = schema["properties"]
        provenance = properties["provenance"]["properties"]
        interface = properties["interface"]["properties"]
        memory = properties["memory_topology"]["properties"]
        bindings = {
            "schema_version": properties["schema_version"]["const"],
            "source_repository": provenance["source_repository"]["const"],
            "source_revision": provenance["source_revision"]["const"],
            "evidence_policy": provenance["evidence_policy"]["const"],
            "axi_data_width_bits": interface["axi_data_width_bits"]["const"],
            "interface_control": interface["control"]["const"],
            "memory_topology": memory["type"]["const"],
        }
    except (KeyError, TypeError) as exc:
        raise XRConfigError("schema: incomplete normalized-manifest contract") from exc
    for key, schema_value in bindings.items():
        if schema_value != policy[key]:
            raise XRConfigError(f"schema: {key} policy mismatch")
    required_provenance = set(properties["provenance"].get("required", ()))
    if {"source_repository", "source_revision", "evidence_policy", "source_hashes"} - required_provenance:
        raise XRConfigError("schema: incomplete provenance requirements")
    return policy


def _load_model(root: Path, reference: str, sources: dict[str, Path], chain: tuple[str, ...]) -> dict[str, Any]:
    path, namespace = _resolve(root, reference, "models")
    if namespace in chain:
        raise XRConfigError(f"model inheritance cycle: {' -> '.join(chain + (namespace,))}")
    raw = _read_json(path)
    _validate_model(raw, namespace, merged=False)
    sources[namespace] = path
    value: dict[str, Any] = {}
    if "inherits" in raw:
        value = _load_model(root, raw["inherits"], sources, chain + (namespace,))
    value = _merge(value, raw)
    _validate_model(value, namespace, merged=True)
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_experiment(
    experiment_reference: str,
    *,
    config_root: str | Path | None = None,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one experiment namespace into an immutable normalized manifest."""
    root = Path(config_root) if config_root is not None else Path(__file__).resolve().parents[1] / "configs" / "xr_accel"
    schema_file = Path(schema_path) if schema_path is not None else root / "schema.json"
    policy = _load_policy(schema_file)
    sources: dict[str, Path] = {}

    experiment_path, experiment_ref = _resolve(root, experiment_reference, "experiments")
    experiment = _read_json(experiment_path)
    _validate_experiment(experiment, experiment_ref)
    sources[experiment_ref] = experiment_path

    model = _load_model(root, experiment["model"], sources, ())
    design_path, design_ref = _resolve(root, experiment["design"], "designs")
    design = _read_json(design_path)
    _validate_design(design, design_ref)
    sources[design_ref] = design_path
    target_path, target_ref = _resolve(root, experiment["target"], "targets")
    target = _read_json(target_path)
    _validate_target(target, target_ref, policy)
    sources[target_ref] = target_path

    board_clock = design.get("board_clock_period_ns", target["clock_period_ns"])
    regions = []
    for name in policy["required_runtime_regions"]:
        regions.append({"name": name, "base_address": target["runtime"][f"{name}_base"]})
    manifest = {
        "schema_version": policy["schema_version"],
        "experiment_name": experiment["name"],
        "links": {"experiment": experiment_ref, "model": experiment["model"], "design": design_ref, "target": target_ref},
        "records": {"experiment": copy.deepcopy(experiment), "model": model, "design": design, "target": target},
        "clocks": {
            "target_period_ns": target["clock_period_ns"],
            "design_period_ns": design["clock_period_ns"],
            "board_period_ns": board_clock,
        },
        "interface": {
            "axi_data_width_bits": policy["axi_data_width_bits"],
            "control": policy["interface_control"],
            "top_module": experiment.get("top_module"),
        },
        "memory_topology": {"type": policy["memory_topology"], "regions": regions},
        "provenance": {
            "source_repository": policy["source_repository"],
            "source_revision": policy["source_revision"],
            "evidence_policy": policy["evidence_policy"],
            "source_hashes": {name: _hash(path) for name, path in sorted(sources.items())},
        },
    }
    validate_normalized_manifest(manifest, policy=policy)
    return manifest


def validate_normalized_manifest(manifest: Mapping[str, Any], *, policy: Mapping[str, Any] | None = None) -> None:
    """Validate the exact normalized-manifest envelope without third-party packages."""
    active = dict(_EXPECTED_POLICY if policy is None else policy)
    top = {"schema_version", "experiment_name", "links", "records", "clocks", "interface", "memory_topology", "provenance"}
    if not isinstance(manifest, Mapping):
        raise XRConfigError("manifest: expected object")
    _exact_keys(manifest, top, top, "manifest")
    if manifest["schema_version"] != active["schema_version"]:
        raise XRConfigError("manifest: schema version mismatch")
    _string(manifest["experiment_name"], "manifest.experiment_name", identifier=True)
    for key, required in (("links", {"experiment", "model", "design", "target"}), ("records", {"experiment", "model", "design", "target"})):
        value = manifest[key]
        if not isinstance(value, Mapping):
            raise XRConfigError(f"manifest.{key}: expected object")
        _exact_keys(value, required, required, f"manifest.{key}")
    for category in ("experiments", "models", "designs", "targets"):
        key = category[:-1]
        text = _relative_path(manifest["links"][key], f"manifest.links.{key}")
        if text.split("/")[:2] != ["configs", category]:
            raise XRConfigError(f"manifest.links.{key}: wrong category")
    records = manifest["records"]
    experiment = records["experiment"]
    model = records["model"]
    design = records["design"]
    target = records["target"]
    if not all(isinstance(item, dict) for item in (experiment, model, design, target)):
        raise XRConfigError("manifest.records: each record must be an object")
    _validate_experiment(experiment, manifest["links"]["experiment"])
    _validate_model(model, manifest["links"]["model"], merged=True)
    _validate_design(design, manifest["links"]["design"])
    _validate_target(target, manifest["links"]["target"], active)
    if manifest["experiment_name"] != experiment["name"]:
        raise XRConfigError("manifest: experiment_name does not match experiment record")
    for key in ("model", "design", "target"):
        if experiment[key] != manifest["links"][key]:
            raise XRConfigError(f"manifest: experiment {key} link mismatch")
    clocks = manifest["clocks"]
    if not isinstance(clocks, Mapping):
        raise XRConfigError("manifest.clocks: expected object")
    clock_keys = {"target_period_ns", "design_period_ns", "board_period_ns"}
    _exact_keys(clocks, clock_keys, clock_keys, "manifest.clocks")
    for key in clock_keys:
        _positive(clocks[key], f"manifest.clocks.{key}")
    interface = manifest["interface"]
    if not isinstance(interface, Mapping):
        raise XRConfigError("manifest.interface: expected object")
    interface_keys = {"axi_data_width_bits", "control", "top_module"}
    _exact_keys(interface, interface_keys, interface_keys, "manifest.interface")
    if interface["axi_data_width_bits"] != active["axi_data_width_bits"] or interface["control"] != active["interface_control"]:
        raise XRConfigError("manifest.interface: policy mismatch")
    if interface["top_module"] is not None and not _IDENT_RE.fullmatch(_string(interface["top_module"], "manifest.interface.top_module")):
        raise XRConfigError("manifest.interface.top_module: invalid identifier")
    memory = manifest["memory_topology"]
    if not isinstance(memory, Mapping):
        raise XRConfigError("manifest.memory_topology: expected object")
    _exact_keys(memory, {"type", "regions"}, {"type", "regions"}, "manifest.memory_topology")
    if memory["type"] != active["memory_topology"] or not isinstance(memory["regions"], list):
        raise XRConfigError("manifest.memory_topology: policy mismatch")
    names, addresses = [], []
    for region in memory["regions"]:
        if not isinstance(region, Mapping):
            raise XRConfigError("manifest.memory_topology.regions: expected objects")
        _exact_keys(region, {"name", "base_address"}, {"name", "base_address"}, "manifest.memory_topology.region")
        names.append(region["name"]); addresses.append(region["base_address"])
        if not isinstance(region["base_address"], str) or not _MMIO_RE.fullmatch(region["base_address"]):
            raise XRConfigError("manifest.memory_topology.region: noncanonical MMIO")
    if names != active["required_runtime_regions"] or len(set(addresses)) != len(addresses):
        raise XRConfigError("manifest.memory_topology: invalid region set or duplicate address")
    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping):
        raise XRConfigError("manifest.provenance: expected object")
    pkeys = {"source_repository", "source_revision", "evidence_policy", "source_hashes"}
    _exact_keys(provenance, pkeys, pkeys, "manifest.provenance")
    if (
        provenance["source_repository"] != active["source_repository"]
        or provenance["source_revision"] != active["source_revision"]
        or provenance["evidence_policy"] != active["evidence_policy"]
    ):
        raise XRConfigError("manifest.provenance: policy mismatch")
    hashes = provenance["source_hashes"]
    if not isinstance(hashes, Mapping) or len(hashes) < 4:
        raise XRConfigError("manifest.provenance.source_hashes: expected at least four entries")
    for name, digest in hashes.items():
        _relative_path(name, "manifest.provenance source")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise XRConfigError("manifest.provenance: invalid SHA-256")


def normalize_all_experiments(*, config_root: str | Path | None = None, schema_path: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(config_root) if config_root is not None else Path(__file__).resolve().parents[1] / "configs" / "xr_accel"
    experiments = sorted((root / "experiments").glob("*.json"))
    manifests = [normalize_experiment(f"configs/experiments/{path.name}", config_root=root, schema_path=schema_path) for path in experiments]
    reached = {name for manifest in manifests for name in manifest["provenance"]["source_hashes"]}
    inventory = {
        f"configs/{category}/{path.name}"
        for category in ("experiments", "models", "designs", "targets")
        for path in (root / category).glob("*.json")
    }
    if reached != inventory:
        raise XRConfigError(f"reference graph mismatch; unreachable={sorted(inventory - reached)}, unexpected={sorted(reached - inventory)}")
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", help="configs/experiments/<name>.json namespaces")
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--schema", type=Path)
    args = parser.parse_args(argv)
    if args.experiments:
        output: Any = [normalize_experiment(item, config_root=args.config_root, schema_path=args.schema) for item in args.experiments]
        if len(output) == 1:
            output = output[0]
    else:
        output = normalize_all_experiments(config_root=args.config_root, schema_path=args.schema)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
