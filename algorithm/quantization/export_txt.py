"""Render an integer export as HG-PIPE-style ``.txt`` artifacts.

The HLS backend does not read JSON or ``.npy``; it ``#include``s comma-separated
integer literals. This module converts a dump written by :mod:`quantization.export`
into exactly the file layout the HG-PIPE reference produces
(``references/hardware/hg-pipe-quantization/src/lut_calibration.py``):

- ``{stem}_scalars.txt`` — the op's integer scalars,
- ``{stem}_{table}.txt``  — one file per table, where the generic key ``table``
  becomes ``table_m`` (the reference's name for a requant/GeLU table),
- content is a **single line** of comma-separated decimal integers **with a trailing
  comma and no newline** — a C array initializer body, nothing else.

It reads ``manifest.json`` rather than a live model, deliberately:

- everything it needs is already inlined there, so unlike the exporter it never
  touches an ``nn.Module``;
- it keeps ``export_integer_model`` single-purpose. That function has one sharp
  invariant — a stateful quantized module with no export branch raises — and a second
  output format would multiply its failure modes;
- it can be re-run against a directory exported before this module existed;
- ``.txt`` is *lossy* relative to the manifest (no dtype, no shape, no scales), so it
  is a **view** of the dump, not a source of truth. That is the same relationship
  ``verify_export`` already has with it: verify against the manifest, then render.

Weights stay in ``.npy`` unless ``include_weights=True``. They are the one payload
where the C-initializer form is a poor trade — a patch-embed weight is tens of
thousands of values — and the manifest records the ``.npy`` that holds them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from quantization.export import MANIFEST_NAME, _stem, load_integer_manifest

# Integer payloads a backend gets as their own file. Order fixed so a diff of two
# renders is stable. ``scalars``/``scalars_two`` are handled separately (they carry the
# ``_scalars`` suffix the reference uses).
TABLE_KEYS: tuple[str, ...] = (
    "table", "rsqrt_table", "rsqrt_table_two", "lnw", "lnb",
    "exp_table", "recip_table_one", "recip_table_two",
)
SCALAR_KEYS: tuple[str, ...] = ("scalars", "scalars_two")

# The reference renames the generic key; everything else keeps its manifest name, so a
# reader can map a file back to the field it came from.
_SUFFIX = {"table": "table_m"}


def _ints(values: Any) -> list[int]:
    """A flat list of python ints from a manifest field (list or materialized array)."""
    if isinstance(values, np.ndarray):
        return [int(v) for v in values.ravel().tolist()]
    return [int(v) for v in np.asarray(values).reshape(-1).tolist()]


def _write_csv(path: Path, values: Iterable[int]) -> None:
    """One line, comma separated, trailing comma, no newline — the reference's format."""
    path.write_text(",".join(str(int(v)) for v in values) + ",", encoding="ascii")


def write_hgpipe_txt(out_dir, *, txt_dir=None, include_weights: bool = False) -> list[Path]:
    """Render the export at ``out_dir`` as ``.txt``; returns the files written.

    ``load_integer_manifest`` is used rather than a raw ``json.load`` so the artifacts
    are validated (dtype, shape and declared value range) before anything is rendered —
    rendering a dump that does not load would produce plausible-looking garbage.
    """
    source = Path(out_dir)
    manifest = load_integer_manifest(source)
    dest = Path(txt_dir) if txt_dir is not None else source / "txt"
    dest.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    index: list[str] = []
    for name, entry in sorted(manifest["modules"].items(), key=lambda kv: kv[1]["order"]):
        stem = _stem(int(entry["order"]), name)
        for key in SCALAR_KEYS:
            if entry.get(key) is None:
                continue
            suffix = "scalars" if key == "scalars" else "scalars_two"
            path = dest / f"{stem}_{suffix}.txt"
            _write_csv(path, _ints(entry[key]))
            written.append(path)
            index.append(f"{path.name}\t{name}\t{entry['op']}\t{key}")
        for key in TABLE_KEYS:
            if entry.get(key) is None:
                continue
            path = dest / f"{stem}_{_SUFFIX.get(key, key)}.txt"
            _write_csv(path, _ints(entry[key]))
            written.append(path)
            index.append(f"{path.name}\t{name}\t{entry['op']}\t{key}")
        if include_weights and entry.get("weight_int") is not None:
            path = dest / f"{stem}_weight.txt"
            _write_csv(path, _ints(entry["weight_int"]))
            written.append(path)
            index.append(f"{path.name}\t{name}\t{entry['op']}\tweight_int (row-major "
                         f"{entry.get('weight_shape')})")

    # A .txt file carries no metadata at all, so the mapping back to the module it came
    # from would otherwise live only in the filename. This is the manifest's job for the
    # JSON dump; the index is its equivalent here.
    index_path = dest / "index.tsv"
    index_path.write_text(
        "file\tmodule\top\tfield\n" + "\n".join(index) + "\n", encoding="ascii")
    written.append(index_path)
    return written


def read_csv_ints(path) -> list[int]:
    """Parse a rendered ``.txt`` back to integers — the inverse of :func:`_write_csv`.

    Exists so a test can prove the render round-trips rather than merely that files
    appeared; the trailing comma means a naive ``split(",")`` yields an empty last field.
    """
    text = Path(path).read_text(encoding="ascii").strip()
    return [int(tok) for tok in text.split(",") if tok != ""]


__all__ = ["write_hgpipe_txt", "read_csv_ints", "TABLE_KEYS", "SCALAR_KEYS", "MANIFEST_NAME"]
