# HGTXR Software Paper-Reproduction Status - 2026-06-10

## Purpose

This document records the work completed so far to make HGTXR/software reproduce the submitted HBTXR paper located under PAPER_PRJXR/20_submission_correction/submitted_version. It separates three levels of progress:

1. Structure/procedure reproduction: code, config, scripts, and validation gates exist.
2. Algorithm contract reproduction: the software follows the paper concepts such as search/track, anchor-relative residual tracking, scheduler mode selection, and reduced-depth track execution.
3. Exact numeric reproduction: the reported paper metrics are reproduced from the final dataset split, checkpoint, quantization artifacts, LUTs, golden vectors, and board/HLS evidence.

Current conclusion: level 1 is substantially implemented, level 2 is partially implemented but has a known depth-routing bug in the canonical v3 path, and level 3 is not achieved yet.

## Source Inputs Used

- Submitted paper source: /home/user/project/PRJXR/PAPER_PRJXR/20_submission_correction/submitted_version/jetcas_draft_final/main.tex
- Submitted paper PDF: /home/user/project/PRJXR/PAPER_PRJXR/20_submission_correction/submitted_version/jetcas_draft_final.pdf
- Reference implementation source: /home/user/project/PRJXR/impl_repos/HBTXR_v3_0
- Current target workspace: /home/user/project/PRJXR/HGTXR/software

## Implemented Software Structure

The canonical paper-reproduction package was added under:

- /home/user/project/PRJXR/HGTXR/software/src/hbtxr

This package was ported from:

- /home/user/project/PRJXR/impl_repos/HBTXR_v3_0/src/hbtxr

The existing flat HGTXR software package was intentionally preserved for compatibility:

- /home/user/project/PRJXR/HGTXR/software/*.py
- /home/user/project/PRJXR/HGTXR/software/modules

A root import shim was added so that import hbtxr works from the HGTXR project root:

- /home/user/project/PRJXR/HGTXR/hbtxr/__init__.py

The HGTXR packaging metadata was updated:

- /home/user/project/PRJXR/HGTXR/pyproject.toml

The pyproject now exposes the canonical package from software/src and includes the runtime dependencies needed by the v3 software path.

## Ported v3 Execution Surfaces

The v3 scripts were copied into:

- /home/user/project/PRJXR/HGTXR/software/scripts/v3

The copied scripts include train, eval, infer, export, dataset preparation, canonicalization, manifest building, Grounded-SAM annotation hooks, package sync, and visualization entrypoints.

The v3 configs were copied into:

- /home/user/project/PRJXR/HGTXR/software/configs/v3

The script bootstrap logic was patched so scripts under software/scripts/v3 resolve HGTXR/software as their project root and HGTXR/software/src as their source root.

## Submitted-Paper Reproduction Config

The submitted-paper reproduction preset was added at:

- /home/user/project/PRJXR/HGTXR/software/configs/paper/zcu104_option_a.yaml

It records the paper target settings:

- Platform: ZCU104
- Clock: 300 MHz
- Frame search input: 1 x 128 x 128
- Event track input: 2 x 64 x 64
- Search configuration: depth 8, embed dim 192, mlp ratio 4
- Track configuration: depth 4, embed dim 192, mlp ratio 4
- Stage 1: 100 epochs, AdamW, lr 1e-3
- Stage 2: 200 epochs, AdamW, lr 1e-4
- Stage 3: self-supervised distillation / slimming, selected option A
- Paper hybrid target: P10 99.97, P5 99.72, P1 99.61, pixel error 0.1812 px, latency 0.43 ms, power 1.34 W

The experiment config was repaired:

- /home/user/project/PRJXR/HGTXR/software/configs/experiments/paper_reproduce.yaml

It now extends the submitted-paper config instead of the previously broken software/stage2_hybrid.yaml path.

Known config issue found during review:

- The paper.submitted_version field in zcu104_option_a.yaml currently points to ../../PAPER_PRJXR/... and does not resolve from HGTXR or from the config directory.
- The actual submitted paper source exists at /home/user/project/PRJXR/PAPER_PRJXR/20_submission_correction/submitted_version/jetcas_draft_final/main.tex.
- From HGTXR, the correct relative path is ../PAPER_PRJXR/20_submission_correction/submitted_version/jetcas_draft_final/main.tex.

## Algorithm and Runtime Changes

Reduced-depth support was added in the canonical v3 path:

- /home/user/project/PRJXR/HGTXR/software/src/hbtxr/models/backbone.py
- /home/user/project/PRJXR/HGTXR/software/src/hbtxr/models/tracker/encoder.py
- /home/user/project/PRJXR/HGTXR/software/src/hbtxr/models/hybrid_tracker.py
- /home/user/project/PRJXR/HGTXR/software/src/hbtxr/training/model_factory.py

New or wired configuration fields:

- model.frame_input_size
- model.event_input_size
- model.event_cut_depth
- model.track_depth

Runtime selected-state and reliability outputs were added in:

- /home/user/project/PRJXR/HGTXR/software/src/hbtxr/models/tracker/runtime_policy.py

Added runtime outputs:

- runtime/ellipse_state
- runtime/search_conf
- runtime/track_conf
- runtime/track_quality

The runtime scheduler was extended with a maximum track-duration gate:

- /home/user/project/PRJXR/HGTXR/software/src/hbtxr/models/controller.py

Added field:

- max_track_updates

## Reproduction Manifest Gate

The reproduction manifest helper was added at:

- /home/user/project/PRJXR/HGTXR/software/src/hbtxr/reproduction.py

It generates a manifest with:

- Paper target values
- Actual metrics when available
- Artifact paths when available
- Missing-artifact status
- Overall status

Current status from the manifest logic:

- structure_reproduction_only

The manifest reports exact metric reproduction as not ready unless these artifact groups exist:

- dataset_split
- checkpoint
- quantization
- nonlinear_luts
- golden_vectors

This is intentional. The repository should not claim exact paper reproduction until those artifacts are present and verified.

## Tests Added

Focused paper-reproduction tests were added at:

- /home/user/project/PRJXR/HGTXR/software/tests/test_paper_reproduction_v3.py

Covered assertions:

- paper_reproduce config loads
- paper target fields are present
- frame/event input-size fields are preserved
- v3 model factory wires depth fields
- runtime_step emits selected-state and reliability outputs
- reproduction manifest marks missing final artifacts correctly

## Validation Performed

The following validation passed after implementation:

- Focused paper-reproduction tests: .venv/bin/python -m pytest -q -s software/tests/test_paper_reproduction_v3.py -> 5 passed
- Full software suite: .venv/bin/python -m pytest -q -s software/tests -> 29 passed
- v3 CLI help smoke: train_hbtxr.py, eval_hbtxr.py, infer_hbtxr.py, export_hbtxr.py all returned help successfully
- YAML load sweep: 13 configs checked, 0 failures
- Direct root import smoke: import hbtxr succeeded
- Compile sweep: .venv/bin/python -m compileall -q hbtxr software/src/hbtxr software/scripts/v3 software/tests/test_paper_reproduction_v3.py -> compile_ok

These checks validate importability, config integrity, CLI surface availability, runtime ABI, and unit-level behavior. They do not validate final EV-Eye paper accuracy or hardware latency.

## Current Reproduction Level

### Completed or Mostly Completed

- v3 source package ported into HGTXR/software
- v3 script/config surfaces copied without overwriting legacy wrappers
- submitted-paper target config created
- paper metric targets encoded as config targets
- reduced-depth fields added to model factory/model constructor
- runtime selected-state and reliability outputs added
- reproduction manifest gate added
- focused and full software tests pass
- durable docs/tracking entries updated

### Partially Completed

- Search/track algorithm contract: present in code structure, but not yet fully trustworthy because of the encoder depth-routing bug described below.
- Time-synchronous / geometry-stable supervision pipeline: v3 preprocessing code is ported, but not run on a real EV-Eye split in HGTXR.
- Multi-stage training: configs and scripts are present, but final stage1/stage2/stage3 paper training was not executed.
- Scheduler-driven hybrid evaluation: runtime outputs and scripts exist, but no final dataset/checkpoint evaluation was executed.
- Hardware/software co-design linkage: HGTXR hardware has extensive scaffold and HLS evidence, but not final paper-equivalent bitstream/board reproduction.

### Not Completed

- Final EV-Eye train/val/test split reproduction
- Final trained HBTXR checkpoint reproduction
- Quantization table export and validation
- Nonlinear LUT export and validation
- Golden vector export and replay
- End-to-end reproduction of paper metrics: 0.1812 px, 0.43 ms, 1.34 W
- Board-level ZCU104 verification matching the submitted paper table

## Important Known Issues

### 1. Canonical v3 depth-routing bug

File:

- /home/user/project/PRJXR/HGTXR/software/src/hbtxr/models/tracker/encoder.py

Observed issue:

- encode_tokens accepts a depth_limit argument, but currently calls the backbone using self.track_depth_limit instead of the passed depth_limit.

Impact:

- For the paper config depth 8 / track_depth 4, the frame search path may also run at depth 4 instead of full depth 8.
- This undermines the paper contract: full-depth frame search plus reduced-depth event/track.

Required fix:

- encode_tokens should call self.backbone(tokens, depth_limit=depth_limit).
- encode_frame should pass depth_limit=None.
- encode_event should pass depth_limit=self.event_depth_limit.
- encode_track should continue using self.track_depth_limit.
- Add a test that distinguishes search full depth from track reduced depth using counters or deterministic patched stages.

### 2. Submitted paper source path in config is wrong

File:

- /home/user/project/PRJXR/HGTXR/software/configs/paper/zcu104_option_a.yaml

Observed issue:

- paper.submitted_version does not resolve to an existing file from HGTXR.

Required fix:

- Change it to ../PAPER_PRJXR/20_submission_correction/submitted_version/jetcas_draft_final/main.tex if interpreting from HGTXR project root, or store an absolute path.

### 3. Dataset/manifests are absent

Current state:

- /home/user/project/PRJXR/HGTXR/software/data/_internal/manifests contains only .gitkeep.
- /home/user/project/PRJXR/HGTXR/software/data/_internal/canonical contains only .gitkeep.

Impact:

- v3 train/eval/infer scripts cannot reproduce paper metrics yet.

### 4. External package/runtime roots are not materialized under HGTXR/software

Current state:

- HGTXR/software does not contain packages or Third directories for Grounded-SAM, TimeLens, or v2e.
- The ported paths examples still point to reference-era locations.

Impact:

- The preprocessing pipeline is present as code, but external package backed execution is not ready without path repair and package sync.

### 5. Final checkpoint/quantization/golden artifacts are absent

Observed local candidates:

- /home/user/project/PRJXR/HGTXR/hardware/generated/runs/uv_smoke_stage1/last.pt
- /home/user/project/PRJXR/HGTXR/hardware/refs/weights/software_initial_weights.pt

These are not sufficient to claim submitted paper metric reproduction.

## Accurate Status Statement

The current implementation reproduces much of the paper-facing software structure and procedure, but it does not yet exactly reproduce the submitted paper results.

Best current description:

- Structure/procedure reproduction: largely implemented.
- Algorithm contract reproduction: partially implemented; must fix the v3 depth-routing bug before claiming full-depth search and reduced-depth track correctness.
- Data/training reproduction: not completed; no EV-Eye manifests/canonical data or final checkpoint are present.
- Hardware numeric reproduction: not completed; HLS scaffold and validation evidence exist, but final paper latency/power/resource equality is not proven.
- Exact paper table reproduction: not achieved.

## Next Required Work

Priority 1:

- Fix encoder.py depth_limit routing.
- Fix zcu104_option_a.yaml paper.submitted_version path.
- Add a depth-specific regression test that proves search uses depth 8 and track uses depth 4.

Priority 2:

- Materialize EV-Eye canonical/manifests under HGTXR/software/data/_internal.
- Repair HGTXR-local paths config for Grounded-SAM, TimeLens, v2e, annotation_root, canonical_root, and manifests_root.
- Run v3 train/eval/infer smoke on a real small manifest.

Priority 3:

- Train or import final Stage 1 / Stage 2 / Stage 3 checkpoints.
- Export quantization tables, nonlinear LUTs, and golden vectors.
- Generate reproduction_manifest.json with actual metrics and artifact paths.

Priority 4:

- Run full paper metric evaluation.
- Run HLS/board validation against the exported golden vectors.
- Only then compare against 0.1812 px, 0.43 ms, and 1.34 W as exact reproduction targets.
