# External Source Omissions

This file records what was intentionally not copied from the newly staged external algorithm sources.

## Source Aliases

| Alias | Meaning |
|---|---|
| `external_hybrid_package` | The pool-root hybrid algorithm package supplied after the initial three-branch consolidation. |
| `hgtxr_software` | `HGTXR/HGTXR-etri-server/software` |

The consolidated tree avoids carrying release-specific external package names into new directory names. The alias is used for provenance without making the new layout version-branded.

## external_hybrid_package

Copied:

```text
configs/ -> algorithm/hybrid/configs/external/
scripts/ -> algorithm/hybrid/scripts/external_pipeline/
src/hbtxr/ -> algorithm/hybrid/src/
tests/ -> algorithm/hybrid/tests/external_pipeline/
```

Adjusted while copying:

- package import references were changed from the source package name to `src`
- test file suffixes that encoded the old version name were removed
- release/version-branded strings were removed from copied files

Not copied:

```text
.git/
.github/
.gitignore
README.md
Third/
docs/
exps/
legacy/
packages/
pyproject.toml
ref_codes/
requirements.txt
uv.lock
```

Reasoning:

- repository metadata and package-manager state should not define the consolidated layout.
- `Third` and `packages` need reconciliation with root-level `third/` before import.
- `docs`, `exps`, `legacy`, and `ref_codes` should be reviewed separately before promotion.

## hgtxr_software

Copied:

```text
README.md -> algorithm/hybrid/hardware_reference/README.md
HANDOVER.md -> algorithm/hybrid/hardware_reference/HANDOVER.md
*.py -> algorithm/hybrid/hardware_reference/src/
modules/ -> algorithm/hybrid/hardware_reference/src/modules/
configs/ -> algorithm/hybrid/hardware_reference/configs/
docs/ -> algorithm/hybrid/hardware_reference/docs/
scripts/ -> algorithm/hybrid/hardware_reference/scripts/
tools/ -> algorithm/hybrid/hardware_reference/tools/
tests/ -> algorithm/hybrid/hardware_reference/tests/
```

Not copied:

```text
26.74489871433803
8.69557854788644
anlaysis/
data/
src/
configs/v3/
scripts/v3/
```

Reasoning:

- numeric files appear to be run artifacts or accidental output files.
- `anlaysis/` is not part of the executable software reference surface.
- `data/` should not be embedded in the algorithm source tree.
- `src/`, `configs/v3/`, and `scripts/v3/` duplicate the packaged hybrid source style; the compact HGTXR software reference is preserved through top-level modules under `hardware_reference/src`.
