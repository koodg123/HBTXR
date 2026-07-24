# Quantization Source Provenance

## Active Baseline

- `HGTXR/HGTXR-etri-server/quantization`

This source provided the baseline package layout copied into
`HBTXR/quantization`.

## Canonical Reference Snapshot

- `HBTXR-etri-server/HBTXR/references/impl/HG-PIPE-Quantization`

Copied to:

```text
HBTXR/quantization/references/HG-PIPE-Quantization
```

## Equivalent HBTXR Branch References

The following branch reference directories were checked with relative-path
content hashing and matched the canonical snapshot:

```text
HBTXR-etri-server/HBTXR/references/impl/HG-PIPE-Quantization
HBTXR-etri-desktop/HBTXR/references/impl/HG-PIPE-Quantization
HBTXR-home/HBTXR/references/impl/HG-PIPE-Quantization
```

Relative content hash:

```text
9bbf8b645940f8a7bb02093d8fd528eb67584f74de6e8f414340e3e72e425899
```

## Overlay Additions From HBTXR Reference

The active package was overlaid with the canonical HBTXR reference package and
tests so the following additional LUT/scheme helpers are available in the main
package:

```text
src/lut_calibration.py
src/quantization_scheme.py
tests/test_lut_calibration.py
tests/test_quantization_scheme.py
docs/Quantization-LUT-Work-Summary.md
```

