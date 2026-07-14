# HANDOVER license and artifact audit

## Decision summary

Local possession and Git history do not establish permission to redistribute or
promote third-party code, datasets, weights, or generated hardware products.
Any item without a verified source-to-license mapping is `LICENSE_BLOCKED` for
active integration. Large and generated artifacts remain external, referenced
by immutable manifests.

## License disposition

| Surface | Current evidence | Classification | Required action |
| --- | --- | --- | --- |
| HBTXR-owned current source | Existing repository history; file-level origin still varies | Case-by-case | Preserve history and confirm origin when importing from HANDOVER. |
| ERVT-derived scripts/configs | License mapping not established in this audit | `LICENSE_BLOCKED` | Identify upstream repository/revision, license text, modifications, and notice obligations. |
| TENNs-Eye material | License mapping not established | `LICENSE_BLOCKED` | Verify code and model/data rights separately before reuse. |
| TDTracker material | License mapping not established | `LICENSE_BLOCKED` | Record upstream provenance and compatible license before adaptation. |
| BRAT material | License mapping not established | `LICENSE_BLOCKED` | Do not promote until license/notice obligations are resolved. |
| HG-PIPE / ICCAD24 references | Local snapshots and historical provenance | `REFERENCE_ONLY` pending file-level check | Retain compact reference evidence; verify upstream license before copying implementation. |
| ViT/XR accelerator sources | Local HANDOVER repositories | `ADAPT` only after provenance | Carry original revision, license, notices, and modification record into any selective port. |
| Reports and locally authored analysis | Repository-local evidence | Documentation use | Preserve source revision and distinguish reproduced measurements from copied claims. |

`확실하지 않음`: the absence of a verified mapping in this audit does not mean
that no valid license exists. It means active reuse is blocked until evidence is
recorded. A directory-level LICENSE file is insufficient when bundled files have
different upstreams or dataset/model terms.

## Required provenance record

Every selectively adapted file or behavior must have a record containing:

- source repository, revision, original path, author/upstream where known;
- detected license and the exact license/notice files that govern it;
- copyright and attribution/notice requirements;
- whether modification, redistribution, model weights, and dataset use are
  permitted for the intended project distribution;
- destination path, adaptation summary, owner, review date, and tests;
- unresolved ambiguity and the decision that keeps it non-active.

If a behavior is reimplemented without copying source, retain the source as
design provenance and document that the implementation was rewritten against
current interfaces; this does not automatically remove patent, dataset, or
model-term concerns.

## Artifact classification

| Artifact class | Git disposition | Required external evidence |
| --- | --- | --- |
| Hand-written source, compact config, tests | Eligible after license and review | Revision/path provenance and validation. |
| Markdown/JSON/CSV evidence | Eligible when compact and non-sensitive | Source revision, generation command if known, and limitations. |
| Model checkpoints and optimizer state | `EXCLUDED` from normal Git | SHA-256, architecture/config, training revision, dataset/split, license/terms, storage URI. |
| HDF5/datasets/exports | `EXCLUDED` | Schema, subject/privacy approval, split, preprocessing, checksum, access controls. |
| Runs, logs, TensorBoard/W&B output | External or summarized | Command/config hash, environment, selected summary, sensitive-data review. |
| Vivado/Vitis generated projects, `.Xil`, build trees | `EXCLUDED` | Tool/board version, build manifest, source/config hash, artifact-store URI. |
| Bitstream, HWH/XSA/DCP/IP archives | `EXCLUDED` unless governed release artifact | Board/part, tool version, source/config hash, artifact hash, validation and rollback. |
| Virtual environments, caches, `__pycache__`, compiled intermediates | `EXCLUDED` | Recreate from pinned dependencies; no preservation need. |
| Secrets, tokens, keys, host credentials | Prohibited | Use approved secret storage; never place values in manifests. |

## Data and privacy risk

Eye/event datasets may contain subject identifiers or biometric-adjacent data.
Before importing data or per-subject predictions, record authorization,
redistribution constraints, de-identification, retention, and approved storage.
The committed C2 CSV/JSON evidence is compact repository evidence, but paths and
manifests may still disclose machine layout; they must not be interpreted as a
license or privacy clearance for the referenced datasets.

Subject-independent splits must be represented by stable, non-sensitive IDs and
validated for leakage. Raw data, frames, event streams, and linkable personal
metadata remain outside Git unless a separate approved data-governance process
explicitly permits them.

## Reproducibility versus provenance

Historic reports may be internally consistent while not reproducible on the
current machine because datasets, checkpoints, tools, boards, or absolute paths
are absent. Each report should therefore state one of:

- **reproduced** — rerun under a recorded current environment and matched;
- **validated artifact** — hash and contract checked, execution not repeated;
- **provenance only** — preserved from HANDOVER and not independently verified;
- **blocked** — required external input or authorization is unavailable.

The C1 archive sources and C2/C3 evidence belong to the latter two categories
unless a specific record says otherwise. Their presence does not prove runtime,
accuracy, timing, power, or board results.

## License/artifact promotion gate

An item can leave `LICENSE_BLOCKED`, `REFERENCE_ONLY`, or `EXCLUDED` only when:

1. a file/behavior-level provenance record is complete;
2. the governing license and notices are stored or linked and compatible with
   intended distribution;
3. dataset/model terms and privacy obligations are separately cleared;
4. no credentials, private paths, or sensitive subject data are included;
5. large/generated artifacts use a checksum manifest and approved storage;
6. the technical contract and tests in the software or hardware audit pass;
7. an explicit reviewer records the promotion decision.

## Conclusion

The safe default is to preserve uncertain material as local provenance and to
adapt only cleared, testable behavior. ERVT, TENNs-Eye, TDTracker, and BRAT remain
license-blocked in this audit. Datasets, weights, runs, and generated hardware
outputs remain external regardless of technical usefulness.
