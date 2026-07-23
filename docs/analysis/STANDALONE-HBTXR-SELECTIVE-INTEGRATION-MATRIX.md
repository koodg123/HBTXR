# Standalone HBTXR Selective Integration Matrix

## Source and target

- Source repository: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR`
- Source branch/revision: `annotation` at
  `2ff52628bec1eb4edd9227c5c9e720718a9accef`
- Target repository: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR`
- Target branch/revision at analysis time: `refactor/hbtxr-structure` at
  `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`
- Source authority: committed blobs at the pinned revision. The source working
  tree has 6,094 CRLF-only modifications and is not an import authority.

## Classification vocabulary

- `NO-OP`: equivalent or superseding target content already exists.
- `REFERENCE_ONLY`: useful evidence, not an active code owner.
- `ADAPT`: re-express bounded behavior through current target interfaces.
- `ACTIVE_CANDIDATE`: technically suitable after every named gate passes.
- `EXCLUDED`: generated, data, result, weight, environment, or bulk content.
- `LICENSE_BLOCKED`: no active copy or behavioral port until rights are proven.

## Mechanical inventory

| Surface | Source | Already in target | Source-only | Disposition |
|---|---:|---:|---:|---|
| `analysis/annotation-analysis` | 2,594 files | 14 common paths | 2,580 | Select pure contracts/metrics only |
| Common annotation files | 14 | 14 | 0 | 11 identical; 3 changed |
| Annotation samples | 2,422 | 0 active target need | 2,422 | `EXCLUDED` |
| `references/impl/FECET-HBTXR` | 70 | full archive equivalent | empty data marker only | `NO-OP` archive |
| `references/impl/HG-PIPE-Quantization` | 134 | active normalized package | package-name deltas | `NO-OP` code/tests |
| `references/impl/SWIFT-HBTXR` | 60 | full archive equivalent | empty data marker only | `NO-OP` archive |

The three changed common annotation paths are `README.md`,
`scripts/evlib.py`, and `scripts/requirements.txt`. The target versions remain
authoritative; no overwrite is planned.

## Annotation-analysis candidates

| ID | Exact source path(s) | Proposed target | Class | Decision and gates |
|---|---|---|---|---|
| ANN-00 | `analysis/annotation-analysis/{README.md,01_v2e_codebase_analysis.md,02_eveye_dataset_analysis.md,03_annotation_tool_evaluation.md}` and selected `docs/{00_goal_and_context,02_decision_log,06_analysis_design,11_eveye_dataset_provenance,14_pupil_iris_confusion,15_annotator_accuracy,17_progress_synthesis}.md` | `docs/analysis/EV-EYE-ANNOTATION-AUDIT.md` | `REFERENCE_ONLY` -> curated `ADAPT` | Summarize schema, derived-label, coordinate, and train-subject caveats. Do not copy numeric claims as current evidence. Require source revision links and claim review. |
| ANN-01 | `analysis/annotation-analysis/src/io_schema.py`, `src/align.py` | `algorithm/hybrid/src/evaluation/annotation_records.py`; `algorithm/hybrid/tests/evaluation/test_annotation_records.py` | `ADAPT` + `LICENSE_BLOCKED` | Re-express sample-key parsing and deterministic long-form records. Remove absolute paths, import-time directory creation, file writes, and global datasets. Requires license/authorship and frame-contract gates. |
| ANN-02 | `src/accuracy_view.py`, `src/corrected_error.py`, `src/frame64.py` | `algorithm/hybrid/src/evaluation/annotation_metrics.py`; `algorithm/hybrid/tests/evaluation/test_annotation_metrics.py` | `ADAPT` + `LICENSE_BLOCKED` | Add pure center error, per-axis scaling, P@1/5/10, invalid-sample policy, and macro/micro aggregation. Reuse `EvaluationPolicy`; no result files. |
| ANN-03 | `src/repeatability.py`, `src/reproducibility.py` | `algorithm/hybrid/src/evaluation/annotation_uncertainty.py`; `algorithm/hybrid/tests/evaluation/test_annotation_uncertainty.py` | `ADAPT` + `LICENSE_BLOCKED` | Add pure repeat sigma, bias/limits-of-agreement, pairwise RMS, and guarded three-cornered-hat functions. Explicitly represent source independence and negative-variance fallback. |
| ANN-04 | `scripts/10b_shape_metrics.py`, `scripts/10c_geometric_precision.py` | `algorithm/hybrid/src/evaluation/annotation_metrics.py` | `ADAPT` + `LICENSE_BLOCKED` | Only dependency-free IoU, Dice, radius ratio, angle delta, Hausdorff, and ASSD over supplied values/point sets. OpenCV contour extraction and mask fitting stay reference-only. |
| ANN-05 | `scripts/evlib.py`, `scripts/{01_analyze_dataset,02_analyze_labels}.py` | Existing `algorithm/hybrid/src/preprocess/io_utils.py` | `NO-OP` for event/frame/VIA parsing; gated `ADAPT` for HDF5/MAT inspection | Existing target parsing remains canonical. Add an optional inspector only after a real consumer is named; no current implementation task. |
| ANN-06 | `scripts/03_analyze_tobii_gaze.py` and Tobii helpers in `scripts/evlib.py` | Potential `algorithm/hybrid/src/evaluation/tobii_sync.py` | `ACTIVE_CANDIDATE` P2 + `LICENSE_BLOCKED` | No current consumer is established. Requires a separate requirement decision and synthetic TTL/unit fixtures. Deferred from the first integration slice. |
| ANN-07 | `scripts/{10_eval,10_label_noise,15_eval_annotators}.py` | Potential `algorithm/hybrid/scripts/evaluation/run_annotation_audit.py` | `REFERENCE_ONLY` | Orchestration is experiment-adjacent and excluded. Reconsider only after ANN-01 through ANN-04 become stable APIs. |
| ANN-08 | `src/motion_label.py` | Existing `algorithm/hybrid/src/preprocess/raw_ellipse_blink.py` | `REFERENCE_ONLY` | Fixed thresholds and session-specific rules are not a general contract. No integration until policy/config ownership is specified. |
| ANN-09 | `src/{crop_dataset,build_target_dataset}.py`, `scripts/{07_collect_samples,07b_reslice_events,11_build_perframe}.py` | Existing preprocess/data owners | `NO-OP` / `REFERENCE_ONLY` | Current canonicalization, ROI, event-builder, manifest, and dataset contracts supersede the source. Hard-coded crop and marker values are not imported. |
| ANN-10 | `scripts/{08_run_gsam2,09_inject_pred,13_gsam2_mislabel,14_flag_gsam2_mislabel}.py`, `scripts/annotators/**` | Existing `algorithm/hybrid/src/preprocess/annotation_backends.py` | Backend `NO-OP`; adapters `LICENSE_BLOCKED` | Do not copy external model wrappers, weights, or `_deepvog_model.py` GPLv3 code. New backends need separate upstream, model, and checkpoint terms. |
| ANN-11 | `src/{event_overlay,render_all_masks,figures,tables_out,run_all}.py`, presentation scripts | None in first slice | `REFERENCE_ONLY` | Report/presentation code is not needed to establish contracts. No integration. |
| ANN-12 | `samples/**`, `datasets/**`, `results/**`, `tables/**`, `fig/**`, `backup/**` | External artifact/data store only | `EXCLUDED` | Never import raw samples, predictions, PNGs, CSVs, result JSON, or generated reports. A later request may approve schema/hash manifests only. |
| ANN-13 | `pyproject.toml`, `uv.lock`, `scripts/requirements.txt` | Existing target dependency owners | `NO-OP` | Do not merge lockfiles or install dependencies. The first slice is standard-library-only. |
| ANN-14 | `weights/**`, download scripts and manifests | External model store only | `EXCLUDED` | No weights, downloads, installation, or model execution. |

## `references/impl` candidates

No `LICENSE`, `COPYING`, `NOTICE`, SPDX identifier, copyright statement, nested
Git history, or individual upstream remote was found for any of the three
vendored roots. Active promotion is therefore `LICENSE_BLOCKED` even when a
technical candidate exists.

| ID | Exact source path(s) | Existing target | Class | Decision |
|---|---|---|---|---|
| IMP-FECET-00 | `references/impl/FECET-HBTXR/**` | `algorithm/archive/imports/fecet_hbtxr/**` | `NO-OP`; active use `LICENSE_BLOCKED` | CRLF-normalized content is equivalent. Keep the inactive archive and exclude the empty `data/.gitkeep` tree. |
| IMP-FECET-01 | FECET event, transform, dataset, preparation, model, runtime, training and metric code | Current hybrid data/preprocess/model/training owners | `NO-OP` / `REFERENCE_ONLY` | Current target owners supersede or cover the behavior. No active port is planned. |
| IMP-HG-00 | `references/impl/HG-PIPE-Quantization/hgpipe_quantization/**`, `tests/**`, `configs/**` | `quantization/{src,tests,configs}/**` | `NO-OP` | Differences are the intentional package import rewrite `hgpipe_quantization` -> `src`. Never overwrite active code/tests. |
| IMP-HG-01 | HG-PIPE `docs/**`, `reports/**` | `quantization/{docs,reports}/**` | `REFERENCE_ONLY` | Retain current provenance. Differing generated values require their own source manifest and are not replacement candidates. |
| IMP-SWIFT-00 | `references/impl/SWIFT-HBTXR/**` | `algorithm/archive/imports/swift_hbtxr/**` | `NO-OP`; active use `LICENSE_BLOCKED` | CRLF-normalized content is equivalent. Keep the inactive archive and exclude the empty data marker. |
| IMP-SWIFT-01 | `references/impl/SWIFT-HBTXR/swift_hbtxr/antiblink.py` | Potential `algorithm/hybrid/src/models/antiblink.py` | `ACTIVE_CANDIDATE` + `LICENSE_BLOCKED` | Only candidate not clearly covered by active code: optional anti-blink state/output contract. Requires ownership/license, consumer, PyTorch/runtime, and state-reset review in a separate plan. |
| IMP-SWIFT-02 | `references/impl/SWIFT-HBTXR/tools/import_swift_eye_checkpoint.py` | Potential `algorithm/hybrid/src/preprocess/swift_eye_checkpoint.py` | `ACTIVE_CANDIDATE` + `LICENSE_BLOCKED` | Consider only key-filter/report behavior, never checkpoints. Requires a separately approved checkpoint compatibility plan and runtime. |
| IMP-SWIFT-03 | SWIFT geometry, event, dataset, interpolation, model, runtime, trainer and loss code | Existing hybrid owners | `NO-OP` / `REFERENCE_ONLY` | No whole-module promotion. TimeLens and target-FPS paths already exist. |
| IMP-ENV-00 | setup scripts, CUDA requirements, environments, model downloads | None | `EXCLUDED` | Installation and runtime setup are outside this plan. |

## Mandatory gates

1. **G0 Source lock:** use committed blobs at source revision `2ff5262`; never
   copy the CRLF-dirty worktree implicitly.
2. **G1 Rights:** record author, upstream, license, modification notice, and
   code/model/data rights. Unresolved candidates remain blocked.
3. **G2 Coordinate contract:** resolve `346x260` source metrics versus target
   `346x240` preprocessing and `640x480` evaluation sensor domains.
4. **G3 No-op drift:** re-run normalized, path-aware comparisons before changing
   FECET, HG-PIPE, SWIFT, or the 14 common annotation files.
5. **G4 Pure API:** selected annotation code has no absolute paths, import-time
   I/O, datasets, model weights, or optional runtime imports.
6. **G5 Unit evidence:** standard-library synthetic fixtures cover invalid,
   empty, boundary, anisotropic, macro/micro, and negative-variance cases.
7. **G6 Scope review:** staged files match the exact target allowlist; no data,
   results, weights, archive replacement, installation, experiment, or push.

## Recommended first slice

After G0 through G2 are approved, the smallest maintainable slice is:

1. `annotation_records.py` and its tests.
2. `annotation_metrics.py` and its tests.
3. `annotation_uncertainty.py` and its tests.
4. One curated annotation audit document with pinned source references.

FECET and HG-PIPE have no first-slice code task. SWIFT anti-blink/checkpoint
work remains a separately planned, license- and runtime-gated future lane.
