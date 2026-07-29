> **작성** 2026-07-15 · **갱신** 2026-07-15
> **상태** active — governance
> **소유** repo

# External Artifact Policy

HBTXR keeps source code, configuration, schemas, and compact metadata in Git.
Large or generated payloads stay in an approved external store and are represented
by a validated manifest. This policy records metadata only; it never authorizes an
upload, download, move, deletion, build, reproduction run, or experiment.

## External by default

The following payloads are external by default: model checkpoints, datasets and
HDF5 files, raw predictions, run logs, HLS project state (`.Xil`), build outputs,
bitstreams, HWH/XSA/DCP files, and packaged IP archives. A small review fixture may
enter Git only through a separate explicit decision.

## Required manifest

Each payload has one JSON document conforming to
`docs/provenance/artifact-manifest.schema.json`. It records content hash and size,
source/config/split hashes, architecture/tool/board versions, a non-secret URI,
access/privacy/license state, the generation command and status, validation
ownership/date/claims, and an optional replacement hash.

The validator is fail-closed. It rejects unknown fields and enums, missing or
malformed hashes, negative sizes, credential-like data, subject identifiers,
unsafe URIs, unsupported validation claims, and inconsistent lifecycle states.
It validates metadata only and never opens a local artifact or accesses a URI.

## Lifecycle rules

- `BLOCKED` is mandatory while privacy is `UNKNOWN` or subject-sensitive, or the
  license is `UNVERIFIED`.
- `VALIDATED` requires `HASH_VERIFIED` and `SIZE_VERIFIED` plus owner and date.
- `RETIRED` requires a different `replacement_sha256`.
- Generation commands are provenance strings and must never be executed by the
  validator.

## Repository guardrail

Before committing, scan staged paths for prohibited payload names and extensions.
The current `.gitignore` already covers known run/checkpoint/log paths; it is not
broadened here because global HDF5/build/IP patterns could hide legitimate source
or configuration files. LFS, remote storage, and retention changes require a
separate approved decision.
