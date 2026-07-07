# V3 Progress Checklist

Last updated: 2026-04-08 KST

Reference plan:
- `docs/plan/20260327_163915_04_v3_2_update_plan.md`

Status legend:
- `[x]` done
- `[~]` partial
- `[ ]` pending

## 1. Data Modes / Canonical / Manifest

- [x] `mode0` raw-CSV supervision path added
- [x] `canonical0` / `manifest0` naming and provenance fields added
- [x] `mode0` forces `manual_csv` supervision and blocks `groundedsam`
- [x] `mode0` raw ellipse now generates:
  - pupil state
  - rasterized pupil mask
  - heuristic eye ROI bbox
- [x] `mode1` baseline event-voxel path retained
- [x] `mode2` config / manifest / loader scaffold added
- [x] `canonical1` / `canonical2` naming and provenance fields added
- [x] `canonical1` / `canonical2` physical dataset roots resolved under the canonical workspace root
- [x] `manifest1` / `manifest2` naming and provenance fields added
- [x] mode-aware manifest fallback path resolved as `manifests/<manifest_name>/<split>_manifest.jsonl`
- [x] `mode2` interpolation-before-annotation preprocess path
  - canonicalization now generates synthetic interpolated frames
  - interpolation provenance is written into canonical rows, session packages, and manifests
  - synthetic-frame Grounded-SAM rerun is available when the external Grounded-SAM runtime is configured

## 2. Dataloader / Batch ABI

- [x] `make_loader_by_mode()` factory added
- [x] mode-aware dataset selection added
- [x] batch `meta` includes `data_mode`, `canonical_name`, `manifest_name`, `frame_source`, `effective_event_window`
- [x] baseline model ABI remains `frame [B,1,256,256]`, `event [B,2,256,256]`, `prev_state [B,6]`

## 3. Event Builder Alignment

- [x] default builder aligned to `FACET Table 3`
- [x] default policy is `fixed_count`
- [x] default event count target is `5000`
- [x] default accumulation is `fast_causal_linear`
- [x] `fast_causal_limit=25.0` is propagated through config, manifest, dataset, and trainer
- [x] event binning / accumulation / tensor-shape flow documented

## 4. Run Contract / CLI

- [x] official run root aligned to `runs/<experiment_name_timestamp>/train|eval|infer|vis|hypers`
- [x] `hypers/` artifact writing added
- [x] `train_hbtxr.py` supports `--mode`, `--stage`, `--resume`, `--experiment-name`, `--device`, `--override`
- [x] `eval_hbtxr.py` supports `--resume` and default manifest/checkpoint resolution
- [x] `infer_hbtxr.py` supports `--resume` and default manifest/checkpoint resolution
- [x] `visualize_dataset.py` supports run-contract-aware path resolution
- [x] `visualize_inference_results.py` supports run-contract-aware path resolution
- [x] `visualize_runtime.py` supports run-contract-aware path resolution
- [x] `scripts/hbtxr_mode_pipeline.sh` rewritten around `train | eval | infer | vis`
- [x] old dedicated `resume` action removed from the official shell path
- [x] compatibility alias `visualize -> vis` added
- [x] `scripts/run_vis.sh` added

## 5. Pretrained Loading / Loss Library

- [x] DeiT-Tiny selective preload path added for Stage1
- [x] pretrained report writing added under `hypers/`
- [x] loss catalog expanded in `src/hbtxr/training/losses.py`
- [x] active stage loss wiring preserved
- [x] actual `SSL` wiring added to the trainer path
- [x] actual `pruning` wiring added to the model/trainer path
- [x] `README.md` expanded with project structure, training modes, and training stages
- [x] `README.md` synced with canonical workspace split and `mode2` synthetic Grounded-SAM rerun status
- [x] `mode0` head composition clarified
  - no `mode0`-exclusive new head family was added
  - `mode0_stage1` uses `event=false`, `mask=true`
  - `mode0_stage2` uses `event=true`, `mask=false`

## 6. Documentation / Tracking

- [x] update history document refreshed
- [x] conversation and decision log refreshed
- [x] progress checklist refreshed
- [x] v3.1 update plan progress snapshot refreshed
- [x] active config YAML files annotated with inline option comments
- [x] `README.md` synced with `mode0`, `canonical0`, and `manifest0`

## 7. Verification

- [x] `compileall` validation completed in the available standalone Python environment
- [x] config helper tests extended for `vis_dir`, resume-root lookup, and manifest fallback
- [x] runtime smoke validation for mode-aware `train/eval/infer/vis`
  - synthetic checkpoint + manifest end-to-end execution completed under `pytest`
- [x] `mode0` preprocess / dataset / config regression coverage added
- [x] full repository regression passed after the `mode0` and YAML-comment refresh
  - result: `95 passed`
- [x] full `pytest` execution completed
- [x] synthetic checkpoint + manifest end-to-end `vis` validation completed
- [ ] real dataset / checkpoint end-to-end validation completed
- [x] `mode2` interpolation path end-to-end validation completed
  - synthetic canonicalize + manifest + dataset validation completed under `pytest`
  - synthetic `mode2` train/eval/infer/vis` runtime sweep completed under `pytest`

## 8. Overall

- Current status: `PARTIAL`
- The highest-priority structural refactor items are in place.
- The main remaining blocker is real dataset runtime validation.

## Checklist Snapshot

- [x] `mode1/mode2` contract introduced
- [x] `canonical1/2`, `manifest1/2`, `loader_mode1/2` scaffolding introduced
- [x] `runs/<experiment_name_timestamp>/...` contract introduced
- [x] DeiT-Tiny preload path introduced
- [x] loss library expanded
- [x] `vis` action moved onto the common run contract
- [x] mode-aware direct execution validated
- [x] actual `SSL` wiring completed
- [x] actual `pruning` wiring completed
- [x] full regression run completed

## 9. 2026-03-26 Documentation Sync Check

- [x] update history synced with the latest implementation snapshot
- [x] conversation log synced with the latest decisions and requests
- [x] `v3.1` update plan synced with current plan-vs-progress status
- [x] JETCAS update log synced with the active draft workspace state
- [x] implementation progress re-checked against the active plan
- [~] runtime validation status changed
  - no new runtime pass was completed in this sync cycle
- [x] full `pytest` execution completed
- [ ] real checkpoint + manifest end-to-end validation completed
- [x] `mode2` interpolation preprocess path completed

## 10. 2026-03-26 SSL / Pruning / Pytest Check

- [x] `ssl` reference-only block added to active config
- [x] `pruning` reference-only block added to active config
- [x] `hypers/ssl_reference.json` artifact writing added
- [x] `hypers/pruning_reference.json` artifact writing added
- [x] placeholder reporting covered by config tests
- [x] full `pytest` execution completed in WSL
- [~] warnings cleaned up
  - the suite passes
  - deprecation warnings remain in AMP and pin-memory code paths

## 11. 2026-03-26 Rollback / Policy / Status Recheck

- [x] recent `JETCAS` helper-document edits rolled back on request
- [x] `docs/jetcas/` is now excluded from default edit scope
- [x] future commit requests should include existing `JETCAS` changes
- [x] plan-vs-progress status rechecked after the rollback
- [x] no implementation milestone regressed because the rollback was document-only
- [x] full `pytest` remains completed
- [x] runtime validation milestone completed for synthetic test coverage
  - helper/tests level validation exists
  - synthetic end-to-end checkpoint + manifest execution exists for `mode1` and `mode2`
- [x] `mode2` preprocessing milestone completed
  - synthetic frame generation, interpolation index export, and manifest provenance are implemented
  - synthetic runtime train/eval/infer/vis coverage is also completed

## 12. 2026-03-26 Actual SSL / Pruning / Warning / Runtime Validation Check

- [x] `SSL` moved from placeholder-only reporting to actual optional trainer wiring
- [x] `Pruning` moved from placeholder-only reporting to actual optional width-slicing wiring
- [x] AMP deprecation cleanup applied through `torch.amp`
- [x] loader-side `pin_memory` decision path tightened
- [x] synthetic checkpoint + manifest end-to-end validation added
- [x] full `pytest` passed in the repository-local `.venv`
- [x] deprecation-focused `pytest` rerun passed
- [ ] real dataset / checkpoint validation completed
- [x] `mode2` interpolation preprocessing completed

## 13. 2026-03-26 Grounded-SAM Runtime / Docs / Commit Preparation Check

- [x] repository tree and `src/hbtxr` dependency map documented
- [x] Grounded-SAM script entry surface documented
- [x] `groundedsam_root` meaning and export layout documented
- [x] nested `segment_anything` repo layout handled in runtime bootstrap
- [x] `transformers 5.x` BERT helper compatibility handled
- [x] GroundingDINO `device` propagation fixed
- [x] missing `_C` deformable-attention custom-op path downgraded to PyTorch fallback
- [x] CPU dummy-image annotation smoke completed
- [~] real CUDA annotation execution validated
  - the crash path is patched
  - actual CUDA rerun still depends on the user machine
- [ ] GroundingDINO custom C++ op build completed
- [ ] real dataset `annotate -> canonicalize -> build_manifests` rerun completed

## 14. Plan-Vs-Progress Snapshot For Current Session

- [x] `P0` contract and CLI milestones remain completed
- [x] `P1` preload integration remains completed
- [x] `P2` loss-library milestone remains completed
- [x] `P3` documentation/tracking remains synchronized
- [~] `P3` runtime validation remains partial
  - synthetic/runtime smoke validation exists
  - real dataset / checkpoint validation is still pending
- [x] `P4` actual `SSL` and `pruning` wiring remains completed
- [~] Grounded-SAM integration moved from scaffold-only toward operational compatibility
  - local import/runtime issues were patched
  - real end-to-end dataset rerun is still pending
- [ ] `mode2` interpolation preprocessing milestone completed

## 15. 2026-03-26 Patch-Embed `split1/split2` Progress Check

- [x] `model.patch_embed.variant` introduced with `legacy | split1 | split2`
- [x] `legacy` kept as the default execution path
- [x] `split1` implemented as shared `3ch` front-end with mode affine and frame psum cache
- [x] `split2` implemented as split `1ch` frame and `2ch` event front-end with frame cache
- [x] tracker now uses a unified patch front-end instead of fixed direct frame/event patch modules
- [x] runtime state now carries patch-cache information
- [x] `split1` warmup gating is now epoch-aware
- [x] variant-aware pretrained mapping is implemented for `legacy`, `split1`, and `split2`
- [x] repository-path model ABI tests passed
- [x] local temp-clone preload pytest passed
- [x] local temp-clone runtime e2e pytest passed
- [ ] real hardware-oriented cache/performance validation completed

## 14. 2026-03-26 Mode2 Interpolation Completion Check

- [x] canonicalization now generates synthetic interpolated frames for `mode2`
- [x] `interpolation_index.json` is written per session
- [x] `session_package.json` now carries interpolation provenance
- [x] `manifest2` rows now carry synthetic frame provenance from canonicalization output
- [x] `Mode2Dataset` loading is covered by dedicated pytest checks
- [x] full repository `pytest` completed after the `mode2` changes
- [x] full `mode2` train/eval/infer runtime sweep completed
  - preprocess, manifest, dataset, and runtime coverage is complete in synthetic pytest
  - real dataset / real checkpoint validation is still separate

## 15. 2026-03-26 Mode2 Runtime Sweep Completion Check

- [x] `mode2` synthetic canonicalize -> manifest2 -> stage1 train completed
- [x] `mode2` stage2 train from Stage1 checkpoint completed

## 16. 2026-03-27 Target-FPS Dataset Transition Planning Check

- [x] target-FPS transition plan document added
- [x] `docs/plan/00_index.md` updated with the new plan entry
- [x] update history synced with the new target-FPS transition plan
- [ ] target-FPS grid-based dataset build implemented
- [ ] rebinned existing-event packet generation implemented
- [ ] per-session `H5` storage contract implemented
- [ ] fused annotation path implemented
  - `Raw-CSV` pupil supervision
  - `Grounded-SAM` eye ROI supervision
  - `Grounded-SAM-in-ROI` pupil fallback for no-CSV sessions
- [ ] canonical rows extended with `H5` and frame-level status/reason fields
- [ ] manifest rows extended with skip/invalid reason visibility
- [ ] end-to-end mode1/mode2 target-FPS smoke validation completed

## 17. 2026-03-27 v3.2.1 Immediate Execution Planning Check

- [x] `v3.2.1` immediate execution scope documented under the active `v3.2` plan
- [x] raw dataset root fixed as `/mnt/d/dataset/EV_Eye/raw_data/Data_davis`
- [x] target dataset root fixed as `/mnt/d/dataset/EV_Eye/target_data`
- [x] current phase constrained to:
  - interpolated frame generation
  - rebinned existing-event packet generation
  - `Raw-CSV`-only label generation
- [x] deferred `Grounded-SAM` work recorded for the next phase
- [ ] target dataset root `fps_<TARGET_FPS>` builder implemented
- [ ] `session.h5` writer implemented
- [ ] `session_meta.json` writer implemented
- [ ] `frame_labels.jsonl` writer implemented
- [ ] `groundedsam_todo.jsonl` writer implemented
- [x] `mode2` eval completed
- [x] `mode2` infer completed
- [x] `mode2` inference visualization completed
- [x] `mode2` runtime trace visualization completed
- [x] full repository `pytest` completed after the runtime addition
- [x] deprecation-focused full repository `pytest` completed after the runtime addition
- [ ] real dataset / real checkpoint runtime validation completed

## 16. 2026-03-26 Canonical Split / Synthetic Grounded-SAM / README Sync Check

- [x] physical `canonical1/canonical2` dataset split implemented under the canonical workspace root
- [x] mode-aware canonical root resolution added to preprocess, manifest build, dataset load, and overlay preview paths
- [x] `mode1` canonicalization now validates the nested `canonical1` layout under `pytest`
- [x] `mode2` synthetic-frame Grounded-SAM rerun implemented
- [x] synthetic Grounded-SAM rerun is covered by dedicated preprocess `pytest`
- [x] `configs/paths.example.json` now reflects workspace-root canonical resolution
- [x] `README.md` synced with the current implementation state
- [x] full repository `pytest` completed after this sync
- [x] deprecation-focused full repository `pytest` completed after this sync
- [ ] real dataset / real checkpoint runtime validation completed

## 17. 2026-03-26 `exp` Merge / `ref_cods` Removal / Repository Cleanup Check

- [x] remote `origin/exp` fetched and merged into `main`
- [x] expected `docs/chat` merge conflicts resolved while preserving both histories
- [x] merged `main` pushed to `origin/main`
- [x] follow-up local HBTXR sync committed after the merge
- [x] `ref_cods` removal committed and pushed
- [x] existing `docs/jetcas` worktree changes included in the requested commit/push flow
- [x] repository regression suite rerun after the follow-up commit
  - result: `29 passed`
- [x] worktree returned to clean state
- [x] leftover pre-merge stash dropped
- [x] `origin/exp` compatibility work is now part of `main`
- [ ] real dataset / real checkpoint runtime validation completed
- [ ] real dataset `annotate -> canonicalize -> build_manifests` rerun completed

## 18. 2026-03-26 Structural Distillation / Regularization SSL Gating Check

- [x] `distillation` separated from the old mixed `ssl` surface

## 19. 2026-03-27 Mode2 Target-FPS / Synthetic Event Generation Check

- [x] `mode2` interpolation supports `target_fps`-driven synthetic-frame count calculation
- [x] `fixed_insert` override remains available above target-FPS auto-counting
- [x] `count_policy: round | floor | ceil` is implemented
- [x] interpolation provenance now records `interp_rank`, `interp_insert_count`, `interp_target_fps`, and `interp_count_policy`
- [x] `linear_blend` multi-frame interpolation is implemented
- [x] `TimeLens` preprocessing adapter is wired into the `mode2` interpolation backend surface
- [x] local `.venv` dependency blocker for `TimeLens` import was cleared by installing `scipy`
- [~] real `TimeLens` end-to-end canonicalize smoke completed
  - adapter wiring and local import succeeded
  - the external `attention.bin` checkpoint is still missing
- [x] `mode2` synthetic event generation now supports `generation_strategy: raw_window | source_pair_average`
- [x] `mode2` synthetic event generation now supports `average_weighting: mean | alpha`
- [x] `source_pair_average + mean` path is covered by dataset tests
- [x] `source_pair_average + alpha` path is covered by dataset tests
- [x] `raw_window` synthetic path is covered by dataset tests
- [x] `mode2_stage1.yaml` now defaults to `source_pair_average + alpha`
- [x] `mode2_stage2.yaml` now defaults to `source_pair_average + alpha`
- [x] `README.md` documents the synthetic-event generation contract
- [x] full repository `pytest` completed after the interpolation / synthetic-event changes
  - result: `46 passed`
- [ ] real `TimeLens` checkpoint-backed preprocessing validation completed
- [x] `regularization_ssl` separated as a teacher-free auxiliary path
- [x] explicit full-teacher / structural slim-student path implemented
- [x] structural student export implemented
- [x] slim-student partial preload implemented
- [x] recommended mode/stage presets updated for structural distillation
- [x] strict stage-aware distillation gating implemented
- [x] strict stage-aware regularization SSL gating implemented
- [x] config-level stage target override hooks implemented
- [x] targeted config tests passed
- [x] targeted loss gating tests passed
- [x] targeted structural training smoke tests passed
- [x] full repository `pytest` completed after the redesign
  - result: `37 passed`
- [ ] real dataset / real checkpoint structural-student validation completed

## 20. 2026-03-27 `qc.py` Legacy Migration Check

- [x] active `src/hbtxr/preprocess/qc.py` retired
- [x] protocol/session-prior helpers split into `src/hbtxr/preprocess/protocol.py`
- [x] legacy QC helper code archived under `legacy/src/hbtxr/preprocess/qc.py`
- [x] active canonicalization no longer emits `session_qc`
- [x] active `session_package.json` generation no longer emits `session_qc`
- [x] repository dependency-map documentation updated for the new active/legacy boundary
- [x] update history document updated
- [x] conversation/decision log updated
- [x] targeted preprocess regression passed
  - result: `8 passed`
- [x] full repository regression passed after the migration
  - result: `46 passed`
- [ ] real dataset / real checkpoint validation completed after the QC-surface retirement

## 21. 2026-03-27 External Packages / README / Script Surface Check

- [x] managed `packages/` surface created
- [x] default package manifest added for `Grounded-Segment-Anything`
- [x] default package manifest added for `timelens`
- [x] `scripts/sync_packages.py` added for clone/update workflow
- [x] `packages/.gitignore` added to keep cloned repositories out of version control
- [x] path resolution now auto-detects `groundedsam_root` from `packages/`
- [x] path resolution now auto-detects `timelens_root` from `packages/`
- [x] top-level `README.md` expanded into an end-to-end execution handbook
- [x] README now documents:
  - dataset relocation / generation
  - annotation
  - canonicalization
  - manifest generation
  - dataloader inspection
  - Stage1 / Stage2 training for both `mode1` and `mode2`
  - inference
  - export
- [x] missing dataloader inspection script added
- [x] missing standalone export script added
- [x] missing prepare wrapper added
- [x] missing relocate wrapper added
- [x] missing package-sync wrapper added
- [x] `hbtxr_mode_pipeline.sh` extended to cover `dataloader` and `export`
- [x] targeted workflow regression passed
  - result: `2 passed`
- [x] targeted package-surface regression passed
  - result: `6 passed`
- [x] full repository regression passed after the surface expansion
  - result: `54 passed`
- [ ] default `Swift-Eye` package manifest entry added
- [ ] real package download / clone validation completed against live remotes

## 22. 2026-03-27 `Swift-Eye` Default Manifest / Reference Analysis Activation Check

- [x] default `Swift-Eye` package manifest entry added
- [x] `packages/README.md` updated for the three-package default surface
- [x] top-level `README.md` updated for the three-package default surface
- [x] `docs/prj/20260325_140031_00_docs_index.md` now links the active `docs/references_analysis/` tree
- [x] `docs/prj/references/20260330_170419_00_index.md` added
- [x] `docs/references_analysis/facet/00_index.md` added
- [x] `docs/references_analysis/swift-eye/00_index.md` added
- [x] package manifest dry-run validation completed
- [x] targeted package-surface regression passed
  - result: `6 passed`

## 23. 2026-03-27 Real External Validation And WSL GPU 5-Epoch Smoke

- [x] user-provided Grounded-SAM root validated
- [x] user-provided TimeLens root validated
- [x] user-provided Swift-Eye root validated
- [x] Grounded-SAM weights downloaded into the external repository
- [x] WSL GPU runtime validated
  - `torch 2.11.0+cu130`
  - `cuda_available=True`
  - device: `NVIDIA GeForce RTX 4070 Ti`
- [x] real Grounded-SAM annotation smoke completed on WSL GPU
- [x] real TimeLens checkpoint-backed `mode2` canonicalization completed on WSL GPU
- [~] real Swift-Eye consumer runtime validation completed
  - repository and weights are present
  - import is currently blocked by missing `mmrotate.core`
- [x] `mode1` canonicalization / manifest / dataloader smoke completed on WSL GPU
- [x] `mode2` canonicalization / manifest / dataloader smoke completed on WSL GPU
- [x] `mode1-stage1` 5-epoch WSL GPU smoke completed
- [x] `mode1-stage2` 5-epoch WSL GPU smoke completed
- [x] `mode2-stage1` 5-epoch WSL GPU smoke completed
- [x] `mode2-stage2` 5-epoch WSL GPU smoke completed
- [x] `mode1-stage2` inference and export smoke completed on WSL GPU
- [x] `mode2-stage2` inference and export smoke completed on WSL GPU
- [~] GroundingDINO custom C++ op build completed
  - the real annotation run succeeded via the PyTorch fallback path
- [x] worktree was clean before the current doc-sync pass

## 24. 2026-03-27 Grounded-SAM Eye BBox / Raw-Ellipse Blink Heuristic Check

- [x] multi-GPU Grounded-SAM annotation support is in place for non-overlapping session shards
- [x] bbox-only Grounded-SAM eye-region preview path added
- [x] heuristic eye-region expansion excluded from the bbox-only path
- [x] near-full-frame bbox rejection added for near-eye false positives
- [x] `user01/session101` bbox export results analyzed and summarized
- [x] raw dataset inspected for explicit `blink/open/closed` metadata
- [x] raw-ellipse blink-candidate heuristic implemented
- [x] raw-ellipse blink CLI and shell runner added
- [x] manual CSV canonicalization now derives `closed_eye_flag` from raw-ellipse metadata
- [x] manifest rows now propagate blink score/source metadata
- [~] blink-state coverage across annotation sources
  - manual CSV sessions are covered
  - Grounded-SAM-generated annotation stores were not yet classified at this step
- [~] runtime validation status
  - helper, syntax, and synthetic propagation checks are complete
  - full real-dataset `canonicalize -> build_manifests` rerun is still pending

## 25. 2026-03-27 Grounded-SAM Store Blink Classification Check

- [x] Grounded-SAM annotation-store rows now receive blink metadata from the session-level ellipse heuristic
- [x] existing Grounded-SAM stores can be backfilled in place when reused
- [x] canonicalization now preserves or derives Grounded-SAM blink metadata instead of marking it not-applicable
- [x] Grounded-SAM pipeline documentation refreshed for the new blink fields and reports
- [~] validation status
  - `py_compile` and manual helper propagation checks are complete
  - focused `pytest` could not run because `pytest` is absent from the repository-local `.venv`
- [ ] full real Grounded-SAM session rerun with the new store-level blink metadata completed

## 26. 2026-03-27 Focused Real Grounded-SAM Blink Validation Check

- [x] focused real Grounded-SAM runtime validation completed on previously problematic blink frames
- [x] `session101` blink-region subset annotated with actual Grounded-SAM runtime on `cuda:0`
- [x] `session102` raw-candidate-adjacent subset annotated with actual Grounded-SAM runtime on `cuda:0`
- [x] Grounded-SAM store blink flags observed in actual exported `frame_annotations.jsonl`
- [x] blink-region interpretation documented in active history / conversation tracking
- [~] result consistency assessment
  - `session101` blink flags were a selective subset of previous bbox rejects
  - `session102` blink flags aligned with the previous raw-CSV blink region within about one frame
- [ ] full-session / full-dataset Grounded-SAM rerun completed

## 27. 2026-03-27 Branch Sync / v3.2 Integrated Plan Check

- [x] local `exp` and local `main` repositories compared for overlap/conflict analysis
- [x] latest local `main` changes merged into `exp`
- [x] merge conflicts resolved while preserving both:
  - `main` interpolation / package / documentation expansion
  - `exp` Grounded-SAM blink and bbox-preview work
- [x] local `main` repository fast-forwarded to the merged `exp` commit
- [x] `v3.2` plan clarified with concrete contracts for:
  - target timestamp grid
  - event packet boundaries
  - H5 physical schema baseline
  - loader/canonical migration bridge
- [x] split `v3.2.1` wording removed in favor of one integrated `v3.2` execution plan
- [x] user-requested four-step flow is now reflected in the plan:
  - dataset build
  - dataset annotation
  - H5/session data structure
  - canonical / manifest generation
- [x] integrated `v3.2` implementation work breakdown added
- [x] active history / checklist / conversation docs synchronized for this planning pass
- [x] end-of-work checklist output adopted as the active workflow convention for future close-outs
- [~] implementation status
  - planning/documentation surface is clarified
  - actual `v3.2` builder / annotation / H5 / bridge code is still pending
- [ ] `v3.2` Task 1 CLI / config surface implemented
- [ ] `v3.2` Task 2 session scan / build-job enumeration implemented
- [ ] `v3.2` Task 3 target timestamp grid builder implemented

## 28. 2026-03-27 Target-FPS Build / Bridge / Loader / Validation Check

- [x] target-FPS `build -> canonical -> manifest` bridge baseline committed
  - checkpoint: `0b7c79f`
- [x] target-FPS manifests now preserve session-store references
  - `session_store_path`
  - `session_store_format`
  - `frame_index`
  - `event_index_range`
- [x] dataset loader now reopens target-FPS samples directly from session stores
  - `npz` supported
  - `h5` supported

## 29. 2026-04-08 TSGSS Sampled Dataset Refinement / Guided Feature Tracking

- [x] all-user / all-session sampled raw-frame surface generated under `tsgss/`
  - `388` sessions
  - `3104` sampled frames
  - `8` consecutive frames per session
- [x] sampled dense annotation merge completed
  - eye ROI / pupil geometry / source-status aggregation recorded
- [x] overlay preview gallery, failure gallery, and failure analysis document generated
- [x] sampled frame interpolation artifact generated
  - `3104 raw -> 2716 interpolated -> 5820 total frames`
- [x] frame-to-frame fixed-count event alignment generated
  - `5432` intervals
  - `target_event_count = 1000`
- [x] raw sparse-interval linear tuple interpolation artifact documented
  - event-plane / xyz / GIF previews produced
  - limitation recorded in the new progress document
- [x] real-vs-interpolated side-by-side event animation generated
- [x] density-adaptive crop added to the dataset path
- [x] density-adaptive crop preview generated for sampled event comparison
- [x] guided derived event feature interpolation generated for all sampled intervals
- [x] ROI-first guided event feature interpolation generated for all sampled intervals
- [x] top-level `tsgss` folder-order alias mapping generated
- [x] consolidated `tsgss` progress document added
- [~] version-control policy for generated `tsgss` artifacts
  - reproduction scripts and documentation are ready
  - large local artifacts remain workspace outputs rather than canonical tracked assets
- [x] build stage now writes queryable frame-level `annotation_failures.jsonl`
- [x] canonical stage now preserves `canonical_annotation_failures.jsonl`
- [x] manifest stage now preserves both:
  - `skipped_sessions.jsonl`
  - `annotation_failures.jsonl`
- [x] pending-eye-ROI path validated for failure propagation
  - `81` failure rows preserved through build, canonical, and manifest
- [x] H5 end-to-end reopening path validated
  - `session.h5 -> canonical -> manifest -> dataset`
- [x] repository-local `.venv` now includes:
  - `h5py`
  - `pytest`
- [x] `requirements.txt` synchronized with those validation dependencies
- [x] focused target-FPS regression suite passed
  - result: `18 passed`
- [x] plan checklist synchronized to the current v3.2 implementation state
- [ ] full real EV-Eye target-FPS dataset generation rerun completed
- [ ] broader training/runtime integration pass completed on the new target-FPS artifacts

## 29. 2026-03-28 Stage2 Pair-Local Synthetic Event Mixing Check

- [x] `mode2` synthetic-event generation now supports `source_pair_scope`
  - `anchor_window`
  - `pair_local`
- [x] legacy `source_pair_average` anchor-window path preserved for backward compatibility
- [x] pair-local source-pair mixing implemented in `src/hbtxr/data/dataset.py`
  - left interval: `[t0, t_syn]`
  - right interval: `(t_syn, t1]`
  - mixed output: `(1 - alpha) * left + alpha * right`
- [x] pair-local `selected_event_count` now reports the full pair-local interval count
- [x] sample metadata now exports `event_generation_source_pair_scope`
- [x] base config updated with default `source_pair_scope: anchor_window`
- [x] `mode2_stage1` preset simplified to `generation_strategy: raw_window`
- [x] `mode2_stage2` preset now uses:
  - `generation_strategy: source_pair_average`
  - `source_pair_scope: pair_local`
  - `average_weighting: alpha`
- [x] dataset regression added for pair-local alpha mixing
- [x] preset/config regression updated for the new Stage1/Stage2 split
- [x] focused dataset regression passed
  - result: `8 passed`
- [x] focused config regression passed
  - result: `4 passed`
- [x] full repository regression passed after the pair-local change
  - result: `88 passed`
- [x] README explanation of `source_pair_scope` added
- [ ] real `mode2-stage2` long-run accuracy comparison completed between:
  - `raw_window`
  - `source_pair_average(anchor_window)`
  - `source_pair_average(pair_local)`

## 30. 2026-03-28 Lazy `mode2-stage2` 2000FPS Planning Check

- [x] dedicated `v3.3` plan document added for lazy `mode2-stage2`
- [x] plan index updated to expose the new `v3.3` planning surface
- [x] problem statement documented using the observed:
  - `500 FPS`
  - `1000 FPS`
  - projected `2000 FPS`
  frame counts
- [x] lazy Stage2 target architecture documented
  - schedule-driven path
  - compact canonical / manifest rows
  - on-the-fly frame generation
  - on-the-fly or cached event generation
- [x] file-by-file implementation plan documented
- [x] migration phases documented
- [x] risks and acceptance criteria documented
- [~] implementation status
  - planning surface is complete
  - lazy `mode2-stage2` code path is not yet implemented
- [ ] `data.mode2.execution: lazy_target_fps` added to config/loader
- [ ] compact lazy schedule metadata exported by target-FPS build path
- [ ] dataset lazy frame synthesis implemented
- [ ] dataset interval-backed Stage2 event generation implemented
- [ ] lazy `mode2-stage2` smoke validation completed

## 31. 2026-03-28 `interval_all` Stage2 / Lazy `mode2-stage2` Implementation Check

- [x] `mode1-stage2` preset now explicitly uses `event_builder.policy: interval_all`
- [x] `build_manifests.py` now supports `event_policy=interval_all`
- [x] manifest rows now preserve `prev_sample_timestamp_us`
- [x] dataset event selection now supports direct `interval_all` slicing
- [x] target-FPS build now supports `frame_storage_mode=lazy_source_frames`
- [x] target-FPS session stores now preserve:
  - `source_frame_images`
  - `source_frame_timestamps_us`
- [x] canonical bridge now emits `session_store_layout`
- [x] dataset can synthesize lazy target-FPS frames from source frame pairs at load time
- [x] loader/config surface now exposes `data.mode2.execution`
- [x] dedicated lazy preset added:
  - `configs/mode2_stage2_lazy_2000fps.yaml`
- [x] README now documents:
  - `mode1-stage2 interval_all`
  - target-FPS lazy Stage2 preparation
  - `mode2_stage2_lazy_2000fps`
- [x] focused regression passed
  - result: `31 passed`
- [x] full repository regression passed after the implementation
  - result: `92 passed`
- [~] lazy `mode2-stage2` runtime status
  - synthetic/session-store regression is complete
  - real large-scale `2000 FPS` runtime validation is still pending

## 32. 2026-03-28 Head / Mode Data Documentation Check

- [x] active head inventory document added
  - `docs/prj/20260330_170419_08_v3_heads_outputs_and_stage_losses.md`
- [x] head document now captures:
  - head modules
  - output keys / shapes / meanings
  - head-to-loss mapping
  - Stage1 / Stage2 training coverage
- [x] active mode data-contract document added
  - `docs/prj/20260330_170419_09_v3_mode_data_pipeline_contracts.md`
- [x] mode data-contract document now captures:
  - `mode1`
  - `mode2` materialized
  - `mode2` lazy target-FPS
  - canonical / manifest / dataset / dataloader contracts
- [x] generated artifact tables added to the mode data-contract document
- [x] representative raw `JSON` / `JSONL` examples added for:
  - `session_package.json`
  - `interpolation_index.json`
  - manifest row
  - target-FPS `session_meta.json`
  - target-FPS `frame_labels.jsonl`
  - target-FPS canonical `frame_annotations.jsonl`
- [x] `docs/hbtxr/00_index.md` updated to expose the two new analysis documents
- [~] validation status
  - document-only synchronization is complete
  - no additional pytest or runtime execution was performed in this pass

## 33. 2026-03-28 Stage-Specific Head Gating / Deployment Export Check

- [x] `model.heads` surface added to the normalized config path
- [x] `build_model()` now forwards head-enable flags into `HBTXRTracker`
- [x] tracker head modules are now conditionally instantiated
- [x] `mode1_stage1` preset now disables `event_head`
- [x] `mode2_stage1` preset now disables `event_head`
- [x] `mode1_stage2` preset now disables `mask_head`
- [x] `mode2_stage2` preset now disables `mask_head`
- [x] `mode2_stage2_lazy_2000fps` preset now disables `mask_head`
- [x] structural student export now forces deployment head stripping:
  - `event_head`
  - `mask_head`
- [x] exported checkpoint state dict no longer contains:
  - `event_head.*`
  - `mask_head.*`
- [x] README synced with:
  - Stage1 head policy
  - Stage2 head policy
  - deployment export head policy
- [x] focused regression passed
  - result: `15 passed`
- [x] full repository regression passed after the head-gating change
  - result: `93 passed`
- [~] deployment simplification status
  - `event_head` and `mask_head` are removed from exported structural-student checkpoints
  - `eye_head` is still preserved and remains an optional future deployment simplification target

## 34. 2026-03-28 YAML Config Reference Documentation Check

- [x] dedicated YAML config reference document added
  - `docs/prj/20260330_170419_10_v3_yaml_config_reference.md`
- [x] `docs/hbtxr/00_index.md` updated to expose the new YAML reference
- [x] config-system structure documented:
  - `base.yaml`
  - `extends`
  - dotted CLI override behavior
- [x] top-level config sections documented:
  - `model`
  - `data`
  - `training`
  - `experiment`
  - `run`
  - `distillation`
  - `regularization_ssl`
  - `ssl`
  - `pruning`
  - `loss`
  - `runtime`
- [x] code-confirmed option values documented for:
  - `data.mode`
  - event window `policy`
  - `generation_strategy`
  - `source_pair_scope`
  - `average_weighting`
  - `scheduler.type`
  - `data.mode2.execution`
- [x] operational preset meanings documented for:
  - `mode1_stage1`
  - `mode1_stage2`
  - `mode2_stage1`
  - `mode2_stage2`
  - `mode2_stage2_lazy_2000fps`
- [~] validation status
  - document synchronization is complete
  - no additional pytest or runtime execution was performed in this pass

## 35. 2026-03-28 Temporal Sampling / ROI Experiment Planning Check

- [x] dedicated `v3.4` experiment-planning document added
  - `docs/plan/20260328_190651_06_v3_4_temporal_sampling_roi_experiments_plan.md`
- [x] `docs/plan/00_index.md` updated to expose the new planning surface
- [x] plan now separates research variables into independent tracks:
  - temporal sample definition
  - frame-event synchronization
  - annotation coordinate policy
  - event voxel scale and normalization
  - ROI / pupil detection heads
  - synthetic frame / event generation
- [x] baseline-safe vs controlled-ablation vs research-only guidance documented
- [x] recommended execution order documented
- [x] acceptance gates and logging contract documented
- [~] implementation status
  - planning surface is complete
  - no code implementation has been started from this `v3.4` plan

## 36. 2026-03-28 Optimizer Pool / Flat-File Optimizer Surface Check

- [x] `src/hbtxr/optim` package added
- [x] flat-file optimizer surface added for:
  - `adamw`
  - `lion`
  - `prodigy`
  - `adopt`
  - `musgd`
- [x] placeholder flat-file registry surface added for:
  - `sophia_g`
  - `ivon`
  - `adam_mini`
  - `adema_mix`
  - `mars`
  - `soap`
- [x] optimizer registry added
- [x] optimizer modifiers added:
  - `cautious`
  - `schedule_free`
- [x] trainer no longer hard-codes `AdamW`
- [x] trainer now writes optimizer hypers reports:
  - `optimizer_resolved.json`
  - `optimizer_diff_summary.json`
- [x] script-level optimizer-pool sweep path added to `train_hbtxr.py`
- [x] config surface added to `base.yaml`:
  - `training.optimizer`
  - `training.optimizer_modifiers`
  - `training.optimizer_pool`
- [x] per-optimizer diff documentation added under:
  - `docs/hbtxr/optimizers/`
- [x] focused optimizer regression passed
  - result: `11 passed`
- [x] focused train/config regression passed
  - result: `7 passed`
- [x] full repository regression passed after optimizer-pool integration
  - result: `106 passed`
- [~] implementation status
  - the pool is fully wired for:
    - `adamw`
    - `lion`
    - `prodigy`
    - `adopt`
    - `musgd`
  - the remaining optimizer names are registered placeholders and are intentionally excluded by `implemented_only=true`
- [ ] README optimizer-pool usage guide added
- [ ] dedicated active-reference doc for optimizer pool semantics added

## 37. 2026-03-28 Optimizer Pool Documentation Synchronization Check

- [x] README optimizer-pool usage guide added
- [x] dedicated active-reference doc added:
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`
- [x] docs index updated:
  - `docs/hbtxr/00_index.md`
- [x] README now documents:
  - single optimizer selection
  - `MuSGD`
  - schedule-free AdamW
  - optimizer pool sweep outputs
- [x] optimizer-pool reference now documents:
  - package layout
  - config surface
  - implemented vs placeholder coverage
  - `MuSGD` group policies
  - report semantics
- [~] validation status
  - document synchronization is complete
  - no extra pytest run was needed for this doc-only follow-up

## 38. 2026-03-28 Trainer Console Logging / CUDA Fail-Fast Check

- [x] live trainer console logging added
- [x] `training.console_log` config surface added to `base.yaml`
- [x] `training.log_every` now drives step-level console update cadence when `console_log.step_interval` is unset
- [x] `training.log_epoch_every` now drives epoch-summary cadence
- [x] run-start console summary added
- [x] epoch progress logging added for:
  - `train`
  - `val`
- [x] checkpoint / pretrained / resume / export / early-stop console events added
- [x] device parser extended for:
  - `cuda:0,cuda:1`
  - `cuda:0,1`
  - `0,1`
  - `multi-gpu`
- [x] CUDA request now fails immediately when CUDA is unavailable
- [x] `requirements.txt` updated with `tqdm`
- [x] `pyproject.toml` updated with `tqdm`
- [x] README synced for trainer logging and CUDA fail-fast semantics
- [x] focused trainer/config regression passed
  - result: `10 passed`
- [x] full repository regression passed
  - result: `115 passed`
- [x] deprecation-focused full repository regression passed
  - result: `115 passed`
- [~] environment status
  - local `.venv` still did not have `tqdm` preinstalled during validation
  - trainer now falls back to plain text when `tqdm` is missing

## 38. 2026-03-28 Optimizer Pool Expansion Check

- [x] `adam_mini` promoted from placeholder to implemented optimizer
- [x] `adema_mix` promoted from placeholder to implemented optimizer
- [x] `soap` promoted from placeholder to implemented optimizer
- [x] optimizer registry now exposes the expanded implemented set:
  - `adamw`
  - `adam_mini`
  - `adema_mix`
  - `lion`
  - `prodigy`
  - `adopt`
  - `soap`
  - `musgd`
- [x] per-optimizer diff notes updated for:
  - `adam_mini`
  - `adema_mix`
  - `soap`
- [x] README optimizer list updated
- [x] optimizer-pool reference document updated
- [x] focused optimizer regression passed
  - result: `17 passed`
- [x] full repository regression passed after the expansion
  - result: `112 passed`
- [~] remaining placeholder set narrowed to:
  - `sophia_g`
  - `ivon`
  - `mars`

## 39. 2026-03-28 Optimizer Pool Completion Check

- [x] `ivon` promoted from placeholder to implemented optimizer
- [x] `sophia_g` promoted from placeholder to implemented optimizer
- [x] `mars` promoted from placeholder to implemented optimizer
- [x] optimizer registry now exposes the completed implemented set:
  - `adamw`
  - `adam_mini`
  - `adema_mix`
  - `ivon`
  - `lion`
  - `mars`
  - `prodigy`
  - `adopt`
  - `sophia_g`
  - `soap`
  - `musgd`
- [x] trainer sampled-parameter path added for:
  - `ivon`
- [x] trainer hessian-refresh path added for:
  - `sophia_g`
- [x] generic optimizer `step(**kwargs)` routing added for batch-scale aware optimizers
- [x] per-optimizer diff notes updated for:
  - `ivon`
  - `sophia_g`
  - `mars`
- [x] README optimizer list updated
- [x] optimizer-pool reference document updated
- [x] focused optimizer regression passed
  - result: `24 passed`
- [x] focused train-pipeline regression passed
  - result: `3 passed`
- [x] full repository regression passed after the expansion
  - result: `122 passed`
- [x] remaining placeholder set resolved

## 40. 2026-03-28 Exact MARS Multi-Pass Trainer Check

- [x] `MARS` builder-side `is_approx=false` block removed
- [x] trainer exact multi-pass future-group/current-group loop added for `MARS`
- [x] exact MARS path now supports finite-loader execution instead of relying on an infinite upstream `get_batch(...)` loop
- [x] `MARS` optimizer state handling hardened for stage-gated parameters:
  - zero-fill `previous_grad` when no future-pass gradient exists
  - tolerate missing `previous_grad` during `update_last_grad()`
- [x] registry metadata updated so `mars` now reports `algorithmic_diff=true`
- [x] README exact-MARS usage example added
- [x] optimizer-pool reference updated for exact MARS semantics
- [x] `mars_diff.md` updated to reflect the new support boundary
- [x] focused optimizer regression passed
  - result: `27 passed`
- [x] focused train-pipeline regression passed
  - result: `3 passed`
- [x] full repository regression passed
  - result: `125 passed`
- [~] validation status
  - CPU regression coverage is complete for this pass
  - dedicated CUDA exact-MARS benchmarking remains undone

## 41. 2026-03-29 Sophia-G Hessian Refresh AMP Compatibility Check

- [x] `Sophia-G` Hessian-refresh path no longer raises on `GradScaler`
- [x] refresh loss is now scaler-scaled and then manually unscaled before `optimizer.update_hessian()`
- [x] README updated for the new Sophia-G AMP status
- [x] optimizer-pool reference updated for the new Sophia-G AMP status
- [x] `sophia_g_diff.md` updated
- [x] direct helper coverage added using `amp.GradScaler("cpu", enabled=True)`
- [x] focused optimizer regression passed
  - result: `28 passed`
- [x] full repository regression passed
  - result: `126 passed`
- [~] validation status
  - helper-level AMP compatibility is validated
  - dedicated CUDA end-to-end Sophia-G AMP benchmarking remains undone

## 42. 2026-03-29 Mode2 Optimizer-Pool Presets / Stage-Aware Stats Check

- [x] stage-aware stats filter added to trainer aggregation
- [x] Stage1 now hides stage-irrelevant:
  - `metric_event_*`
  - `metric_track_*`
  - `loss_event_*`
  - `loss_track_*`
  - `loss_consistency`
- [x] Stage2 now hides stage-irrelevant:
  - `loss_eye`
  - `loss_mask`
- [x] stage1 presets now disable `track_head` by default for:
  - `mode0_stage1`
  - `mode1_stage1`
  - `mode2_stage1`
- [x] `mode2` optimizer-pool presets added:
  - `mode2_stage1_optimizer_pool.yaml`
  - `mode2_stage2_optimizer_pool.yaml`
- [x] README updated
- [x] optimizer-pool reference updated
- [x] focused config/console regression passed
  - result: `8 passed`
- [x] focused train-pipeline regression passed
  - result: `3 passed`
- [x] full repository regression passed
  - result: `126 passed`
- [~] execution status
  - `mode2` pool sweep surface is now exposed through presets
  - real multi-run `mode2` pool benchmarking has not been launched in this pass

## 43. 2026-03-29 Mode2 Optimizer-Pool Synthetic Sweep Validation Check

- [x] `mode2_stage1_optimizer_pool.yaml` script-level smoke executed under `pytest`
- [x] `mode2_stage2_optimizer_pool.yaml` script-level smoke executed under `pytest`
- [x] synthetic target-FPS canonical workspace bootstrap added for mode2 pool validation
- [x] pool run artifact verification added:
  - `optimizer_pool_report.json`
  - expected candidate label set
- [x] README updated with the new mode2 pool smoke status
- [x] optimizer-pool reference updated with the new mode2 pool smoke status
- [x] focused optimizer regression passed
  - result: `30 passed`
- [x] focused config / console / train-pipeline regression passed
  - result: `11 passed`
- [x] full repository regression passed
  - result: `129 passed`
- [~] execution status
  - synthetic script-level mode2 pool execution is now validated
  - real multi-run mode2 benchmark collection is still pending

## 44. 2026-03-29 Mode2 Optimizer-Pool CPU Pilot Benchmark Check

- [x] stage1 bootstrap checkpoint generated for a normal `mode2 stage2` initialization path
- [x] full implemented optimizer set executed under `mode2 stage1` synthetic CPU pilot
- [x] full implemented optimizer set executed under `mode2 stage2` synthetic CPU pilot
- [x] actual pool artifacts generated for both stages:
  - `optimizer_pool_report.json`
  - `optimizer_pool_summary.csv`
  - `optimizer_pool_rankings.json`
- [x] pilot result document added:
  - `docs/exps/20260330_170419_12_v3_mode2_optimizer_pool_cpu_pilot.md`
- [x] README updated with pilot-result reference
- [x] optimizer-pool reference updated with pilot-result reference
- [~] benchmark status
  - synthetic CPU pilot collection is complete
  - real EV-Eye and CUDA benchmark collection is still pending

## 45. 2026-03-29 Real EV-Eye Mode2 Pilot / CUDA Benchmark Readiness Check

- [x] previous synthetic pilot temp directory removed
  - `.tmp_mode2_pool_pilot/`
- [x] bounded real EV-Eye raw subset bridge created for:
  - `user01/left/session_102`
  - `user01/right/session_102`
- [x] canonical `events.npz` + bbox-only ROI rows used to unblock target-FPS build on real data
- [x] target-FPS `100` real-data build completed
- [x] bounded `canonical2` / `manifest2` real-data pilot surface completed
- [x] stage1 bootstrap checkpoint generated on the real-data subset
- [x] full implemented optimizer set executed under bounded real-data `mode2 stage1`
- [x] full implemented optimizer set executed under bounded real-data `mode2 stage2`
- [x] result document added:
  - `docs/exps/20260330_170419_13_v3_real_ev_eye_mode2_optimizer_pool_cpu_pilot.md`
- [x] README updated with real-data pilot reference
- [x] optimizer-pool reference updated with real-data pilot and CUDA status
- [x] temporary real-data pilot workspace removed after result capture
- [~] CUDA benchmark status
  - GPU hardware is present
  - current project `.venv` is CPU-only for PyTorch
  - CUDA throughput / memory benchmark is still blocked by environment setup
- [~] result interpretation status
  - stage1 real-data CPU pilot completed successfully
  - stage2 one-epoch CPU pilot completed but showed `NaN` train loss across candidates, so it is a stability screen rather than a final optimizer ranking

## 46. 2026-03-29 CUDA Environment / Loss Cluster / Import Dependency Check

- [x] WSL CUDA virtual environment created
  - `.venv_wsl_cuda`
- [x] official CUDA PyTorch installed into WSL environment
  - `torch 2.11.0+cu128`
  - `torchvision 0.26.0+cu128`
- [x] CUDA device visibility verified in WSL
  - `torch.cuda.is_available() == True`
  - `device_count == 1`
  - `NVIDIA GeForce RTX 4070 Ti`
- [x] monolithic loss implementation split into clustered files
  - `src/hbtxr/loss_common.py`
  - `src/hbtxr/loss_primitives.py`
  - `src/hbtxr/loss_stage.py`
  - `src/hbtxr/loss_distillation.py`
- [x] `src/hbtxr/training/losses.py` kept as compatibility facade
- [x] package export surface updated
  - `src/hbtxr/__init__.py`
- [x] static import/path dependency regression test added
  - `tests/test_import_graph_v3.py`
- [x] full repository regression passed
  - result: `132 passed`
- [x] compile validation passed
  - `python -m compileall src/hbtxr`
- [x] README updated with WSL CUDA usage note
- [x] `.gitignore` updated for `.venv_wsl_cuda`
- [~] environment status
  - Windows `.venv` remains CPU-only
  - WSL `.venv_wsl_cuda` is the active CUDA path

## 47. 2026-03-29 YOLO26 Head / BBox Integration Plan Check

- [x] current `HBTXR` head inventory re-analyzed against local code
- [x] local `ultralytics-main` reference reviewed for:
  - `Detect`
  - `BboxLoss`
  - `OBB26`
  - `Segment26 / Proto26`
  - `STAL / TAL`
- [x] focused `v3.5` plan document added
  - `docs/plan/20260329_023122_07_v3_5_yolo26_head_bbox_integration_plan.md`
- [x] `docs/plan/00_index.md` updated
- [x] integration scope locked to:
  - `Eye ROI` bbox loss upgrade
  - optional `Pupil BBox / OBB auxiliary branch`
- [x] explicit exclusion list recorded for this wave:
  - full `Detect` replacement
  - `TaskAlignedAssigner`

## 48. 2026-03-30 YOLO / SOT / MGIoU / MPDIoU Dependency Audit Check

- [x] repository-wide import dependency regression rerun completed
- [x] `compileall` validation rerun completed for `src/hbtxr`
- [x] `training/losses.py` compatibility facade updated to re-export:
  - `MGIoU2DLoss`
  - `MPDIoULoss`
- [x] `Eye ROI` now supports `loss.eye_box_mode: yolo26_mpdiou`
- [x] `search/event bbox_aux` now support `*_bbox_aux_mode: mpdiou`
- [x] `search/event obb_aux` now support `*_obb_aux_mode: mgiou2d`
- [x] global `loss.mgiou_fast_mode` option added
- [x] targeted config / model / train smoke validation completed
- [x] full repository `pytest` completed after the new loss integration
  - result: `150 passed`
- [x] active `YOLO26` integration plan document synchronized with the actual code state
  - `STAL`
  - `YOLOE`
- [~] execution status
  - planning surface is complete
  - implementation has not started in this pass

## 61. 2026-03-31 P0-P2 Detector / Assigner / ProtoMask Check

- [x] `P0` detector ABI normalization added
  - `search/eye`
  - `search/eye_det_cls_logits`
  - `search/eye_det_reg_logits`
  - optional `search/eye_det_quality_logits`
- [x] `TaskAlignedAssigner` implemented
- [x] `STALTaskAlignedAssigner` implemented
- [x] `yolo_detect` eye-head variant implemented
- [x] `yolo26_point` loss path migrated onto the assigner stack
- [x] `proto26` mask decoder implemented
- [x] `model.mask.variant` / `model.mask.num_prototypes` config surface added
- [x] model/test/train regression updated for detector and mask variants
- [x] focused regression completed
  - `32 passed`
- [x] full repository regression completed
  - `156 passed`
- [x] README synchronized with the new detector / mask variants
- [~] remaining research backlog
  - tracker actor replacement not started
  - candidate elimination tracker not started
  - YOLOE prompt head not started

## 48. 2026-03-29 Selective Unmerged-Branch Integration Check

- [x] `codex/ipex` excluded from this integration wave
- [x] `codex/startpoint` excluded from this integration wave
- [x] `codex-exp-deepmicro` code integrated additively
  - Grounded-SAM filters
  - checkpoint auto-resolution
  - raw `events.txt` target-FPS input
  - canonical-root nesting fix
- [x] `codex-exp-deepmicro` / `codex/home-wsl` docs and preview tooling imported
  - `docs/hbtxr/14_*`
  - `docs/hbtxr/15_*`
  - `docs/hbtxr/16_*`
  - `docs/hbtxr/17_*`
  - review resources
- [x] `exp_wsl` heuristic ROI docs / plans / preview scripts imported
  - `docs/hbtxr/18_*`
  - `docs/plan/08_*`
  - `docs/plan/09_*`
  - `docs/plan/10_*`
- [x] `exp-wsl` helper script imported
  - `scripts/run_mode1_event_sweep.sh`
- [x] `codex-exp-ubeeslab` experimental surface integrated additively
  - `model.search.xy_from_mask_centroid`
  - mask-centroid search replacement path
  - dynamic custom best checkpoint files
  - `mode0_stage1` eye-only preset
  - `mode0_stage2` mask-centroid fusion preset
  - `docs/hbtxr/19_*`
- [x] focused Wave 4 regression passed
  - model/config/train: `17 passed`
  - trainer/runtime: `36 passed`
- [x] final sync completed
  - README / docs indexes / chat tracking sync completed
  - full repository regression completed: `140 passed`
  - push to `origin/codex/home` completed

## 49. 2026-03-29 Selective Integration Traceability Check

- [x] dedicated selective-integration trace report added
  - `docs/others/update/20260330_170419_20_v3_selective_branch_integration_report.md`
- [x] `docs/hbtxr/00_index.md` updated with the new report
- [x] update history synchronized with the trace report addition
- [x] progress checklist synchronized with the trace report addition
- [x] conversation log synchronized with the trace report addition
- [~] git status
  - documentation sync completed
  - commit / push not yet performed in this specific trace-report pass

## 50. 2026-03-30 Docs Canonical Restructure / Koreanization Check

- [x] `docs/chat`, `docs/jetcas` 유지 원칙 유지
- [x] canonical folders established and kept active
  - `docs/analysis`
  - `docs/experiments`
  - `docs/update`
  - `docs/plan`
  - `docs/progress`
- [x] `docs/hbtxr`, `docs/references_analysis`, `docs/others` compatibility-layer interpretation fixed
- [x] canonical `analysis` documents translated into Korean
- [x] canonical `analysis/optimizers` diff documents translated into Korean
- [x] canonical `analysis/references` session documents translated into Korean
- [x] canonical `experiments` documents translated into Korean
- [x] canonical `update` documents synchronized and kept in Korean
- [x] canonical `plan` documents `00` to `10` translated into Korean
- [x] canonical `progress` active document kept in Korean
- [x] top-level docs index updated with current canonical status
- [x] canonicalization completion report added
  - `docs/others/update/20260330_170419_03_docs_restructure_and_korean_canonicalization_report.md`
- [~] repository state
  - documentation changes are present in the worktree
  - commit / push not performed in this sync pass

## 51. 2026-03-31 P3-P5 Tracking / Prompt Backlog Marker

- [ ] `P3` tracker actor whole-stack replacement
- [ ] `P4` candidate elimination tracker structure integration
- [ ] `P5` YOLOE-style prompt head integration
- [~] `P0-P2` implementation review
  - code review requested
  - implementation status is already tracked in the earlier checklist entries

## 52. 2026-04-01 P1/P2 Review Follow-up Check

- [x] `P3/P4/P5` remained checklist-only backlog items
- [x] `P1` dense eye assigner no longer hard-restricts positives to inside-box points
- [x] `P1` `yolo_detect` now separates binary cls targets from IoU quality targets
- [x] `P2` `proto26` now has auxiliary coarse-mask supervision
- [x] `loss.mask_coarse_weight` config surface added
- [x] focused regression rerun passed
  - `30 passed`
- [x] full repository regression rerun passed
  - `157 passed`
