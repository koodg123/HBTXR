# Quantization Integration Plan

## Goal

Integrate the HGTXR quantization package and the existing HBTXR branch
`HG-PIPE-Quantization` reference into the consolidated `HBTXR/quantization`
workspace.

## Sources

- Active base source:
  - `HGTXR/HGTXR-etri-server/quantization`
- Reference sources requested by the user:
  - `HBTXR-etri-server/HBTXR/references/impl/HG-PIPE-Quantization`
  - `HBTXR-etri-desktop/HBTXR/references/impl/HG-PIPE-Quantization`
  - `HBTXR-home/HBTXR/references/impl/HG-PIPE-Quantization`

The three `HBTXR-*` reference directories were checked with relative-path
content hashing and are identical. The consolidated tree therefore keeps one
canonical snapshot from `HBTXR-etri-server`.

## Applied Layout

```text
HBTXR/quantization/
  README.md
  pyproject.toml
  requirements-eval.txt
  requirements-torch-cu130.txt
  configs/
  docs/
  src/
  reports/
  scripts/
  tests/
  references/
    HG-PIPE-Quantization/
    SOURCES.md
```

## Merge Policy

1. Copy `HGTXR/HGTXR-etri-server/quantization` as the active package baseline.
2. Preserve one canonical `HBTXR-*` reference snapshot under
   `HBTXR/quantization/references/HG-PIPE-Quantization`.
3. Overlay the `HBTXR-*` reference package and tests onto the active package
   because the reference includes additional LUT calibration and quantization
   scheme modules:
   - `src/lut_calibration.py`
   - `src/quantization_scheme.py`
   - `tests/test_lut_calibration.py`
   - `tests/test_quantization_scheme.py`
   - `docs/Quantization-LUT-Work-Summary.md`
4. Keep `reports/` in place because the original CLI and tests expect the
   package-local report paths. The report payload is small enough for this
   integration step.

## Validation Plan

Minimum validation after integration:

```bash
cd HBTXR/quantization
python -m compileall src tests
python -m pytest tests
```

If the full test suite is too slow or environment-dependent, run focused smoke
checks first:

```bash
cd HBTXR/quantization
python -m src.cli audit-completion
python -m src.cli list --limit 3
python -m pytest tests/test_package_api.py tests/test_lut_calibration.py tests/test_quantization_scheme.py
```

## Risks

- The original repositories do not include final paper-equivalent checkpoints,
  final quantization tables, or full calibration assets.
- Some report files are smoke or reconstruction evidence, not final paper
  metric evidence.
- The active package now intentionally represents a union of the HGTXR
  quantization package and the identical HBTXR branch reference snapshot.

