#!/usr/bin/env python3
"""Expand an HGTXR sweep config into run manifests and compile-time defines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path


def load_config(path: Path):
    text = path.read_text()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return parse_minimal_yaml(text)


def parse_scalar(value: str):
    value = value.strip()
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [parse_scalar(part.strip()) for part in body.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_minimal_yaml(text: str):
    """Small YAML subset parser for this repo's sweep file shape.

    Supported:
    - nested mappings via indentation
    - scalar values
    - inline lists such as [1, 2, 4]

    This is not a general YAML parser. Install PyYAML for full YAML support.
    """
    root = {}
    stack = [(-1, root)]

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- "):
            raise ValueError("minimal YAML parser does not support block lists; install PyYAML")

        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"cannot parse line: {raw}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value.strip() == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = parse_scalar(value)
    return root


def flatten_parameter_map(config):
    matrix = config.get("sweep_matrix", config)
    params = matrix.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("expected sweep_matrix.parameters mapping")

    names = []
    values = []
    for name, spec in params.items():
        if not isinstance(spec, dict) or "values" not in spec:
            raise ValueError(f"parameter {name} must contain values")
        names.append(name)
        values.append(spec["values"])
    return names, values


def macro_name(prefix: str, name: str) -> str:
    return prefix + name.upper().replace("-", "_")


def stable_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def render_defines(params: dict, prefix: str, emit_format: str) -> str:
    macros = {macro_name(prefix, k): v for k, v in params.items()}
    if emit_format == "json":
        return json.dumps(macros, indent=2, sort_keys=True) + "\n"
    if emit_format == "cpp":
        lines = []
        for key, value in sorted(macros.items()):
            if isinstance(value, str):
                lines.append(f'#define {key} "{value}"')
            elif isinstance(value, bool):
                lines.append(f"#define {key} {1 if value else 0}")
            else:
                lines.append(f"#define {key} {value}")
        return "\n".join(lines) + "\n"
    if emit_format == "cmake":
        return "\n".join(f"set({k} {v})" for k, v in sorted(macros.items())) + "\n"
    if emit_format == "env":
        return "\n".join(f"{k}={v}" for k, v in sorted(macros.items())) + "\n"
    raise ValueError(f"unsupported emit format: {emit_format}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--defines-dir")
    parser.add_argument("--define-prefix", default="HGTXR_")
    parser.add_argument("--emit-format", choices=["json", "cpp", "cmake", "env"], default="cpp")
    parser.add_argument("--max-combos", type=int)
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--skip-keys", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.sweep_file))
    names, value_lists = flatten_parameter_map(config)

    filters = {}
    for item in args.filter:
      key, sep, value = item.partition("=")
      if not sep:
          raise ValueError(f"filter must be key=value: {item}")
      filters[key] = parse_scalar(value)

    combos = []
    for tuple_values in itertools.product(*value_lists):
        params = dict(zip(names, tuple_values))
        if any(params.get(k) != v for k, v in filters.items()):
            continue
        combos.append(params)

    if args.max_combos is not None and len(combos) > args.max_combos:
        raise ValueError(f"{len(combos)} combos exceeds --max-combos {args.max_combos}")

    if args.validate_only:
        print(f"valid sweep: {len(combos)} combinations")
        return 0

    out_dir = Path(args.out_dir)
    defines_dir = Path(args.defines_dir) if args.defines_dir else out_dir / "defines"
    manifest = Path(args.manifest)

    if not args.dry_run:
        if manifest.exists() and not args.overwrite:
            raise FileExistsError(f"{manifest} exists; pass --overwrite")
        out_dir.mkdir(parents=True, exist_ok=True)
        defines_dir.mkdir(parents=True, exist_ok=True)

    created = datetime.now(timezone.utc).isoformat()
    fixed_fields = [
        "run_id",
        "run_name",
        "sweep_name",
        "run_index",
        "combination_hash",
        "define_path",
        "build_root",
        "manifest_created_utc",
        "params_json",
    ]
    param_fields = [f"param_{name}" for name in names if name not in set(args.skip_keys)]
    rows = []

    matrix = config.get("sweep_matrix", config)
    sweep_name = matrix.get("project", "HGTXR")
    for idx, params in enumerate(combos):
        digest = stable_hash(params)
        run_id = f"run_{digest}"
        suffix = {"json": "json", "cpp": "h", "cmake": "cmake", "env": "env"}[args.emit_format]
        define_path = defines_dir / f"{run_id}.{suffix}"
        build_root = out_dir / run_id

        if not args.dry_run:
            define_path.write_text(render_defines(params, args.define_prefix, args.emit_format))

        row = {
            "run_id": run_id,
            "run_name": f"{sweep_name}::{idx:05d}",
            "sweep_name": sweep_name,
            "run_index": idx,
            "combination_hash": digest,
            "define_path": str(define_path),
            "build_root": str(build_root),
            "manifest_created_utc": created,
            "params_json": json.dumps(params, sort_keys=True),
        }
        for name in names:
            if name not in set(args.skip_keys):
                row[f"param_{name}"] = params[name]
        rows.append(row)

    if args.dry_run:
        print(json.dumps({"combinations": len(rows), "first": rows[:3]}, indent=2, default=str))
        return 0

    with manifest.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fixed_fields + param_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

