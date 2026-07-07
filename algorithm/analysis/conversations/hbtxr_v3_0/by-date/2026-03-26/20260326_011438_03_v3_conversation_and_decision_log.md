# V3 Conversation And Decision Log

Last updated: 2026-03-29 KST

이 문서는 `HBTXR_v3_0` 작업 과정에서 사용자 요청, 설계 결정, 최근 대화에서 정리된 코드 수준 설명을 요약한 로그다.
원문 transcript를 그대로 복제하지 않고, 구현과 문서에 영향을 준 결정만 남긴다.

## 1. Initial Direction

요청 요약:

- `baseline/HBTXR-v2_1-main` 계열을 기반으로 `HBTXR_v3_0`를 정리
- `FACET`를 주요 reference로 활용
- Grounded-SAM 계열 dense annotation 흐름을 고려
- hybrid frame-event eye tracking 구조 유지

결정:

- backbone과 runtime split 철학은 HBTXR 계열을 유지
- dataset/supervision 정렬은 FACET에서 많이 차용
- annotation provenance와 canonical package를 분리된 단계로 관리

## 2. ABI And Schema Clarification

요청 요약:

- annotation / canonical / manifest / dataloader / model / scheduler / training pipeline / loss를 코드 수준까지 구체화

결정:

- 공식 입력 ABI를 `frame [B,1,256,256]`, `event [B,2,256,256]`, `prev_state [B,6]`로 고정
- canonical, manifest, dataloader sample contract를 문서와 코드 모두에서 맞춤
- `search / event / track` branch를 분리된 역할로 유지

## 3. Mandatory Scope Confirmation

요청 요약:

- 다음 항목이 계획과 구현에 반드시 포함되어야 함
  - Dense Annotation using Grounded-SAM
  - Fixed Count Event Voxel
  - Multimodality End-to-end Frame-Event Hybrid Eye Tracking Transformer
  - Partial DeiT-Tiny
  - Decoupled Head
  - 2-stage Track and Search Scheduler
  - 2-Stage Training
  - ROI Restricted Eye Tracking (`constraint_center`)
  - CPU-FPGA workload partition / DeiT-Tiny accelerator / Scheduler FSM

결정:

- active 문서와 코드 설명은 위 항목을 기준으로 정리
- 다만 Grounded-SAM과 hardware integration은 완성 구현이 아니라 scaffold/documented intent 수준임을 분리해서 표시

## 4. Experiment Root Workflow

요청 요약:

- `EXPERIMENT_NAME` 아래 `train / eval / inference / visualization`이 공존해야 함
- shell에서 `resume / eval / inference / visualization`이 바로 가능해야 함
- `train_config`, `hyper_parameter`, `device`, `override`, YAML `experiment.name`가 workflow에 반영되어야 함

결정:

- `scripts/hbtxr_mode_pipeline.sh`를 공식 진입점으로 사용
- resolved config, CLI args, overrides, device를 `Hyperparameters/` 아래 저장
- shell workflow를 active docs와 코드 분석 문서에서 일치하게 설명

## 5. Event Builder And Resize Policy

요청 요약:

- event input dataloader에서 `time_bin`과 `fixed_count` 둘 다 지원해야 함
- resizing은 FACET-style을 기준으로 지원해야 함

결정:

- event loader는 `fixed_count | time_bin` 유지
- resize 정책은 `facet_square_direct`, `letterbox_square`, `sensor_full_square` 유지
- canonical/manifest에는 provenance를 남기고, dataset config override가 실제 실행 정책을 덮을 수 있게 함

## 6. Repository Cleanup And Documentation

요청 요약:

- legacy 코드와 문서를 `legacy/`로 이동
- update history, conversation log, progress checklist를 active docs에 유지
- 계획 대비 진행상황을 체크리스트 형태로 볼 수 있어야 함

결정:

- 구버전 docs/scripts/config는 `legacy/`로 이동
- active docs는 `docs/`에 유지
- `00_docs_index`, `01_v3_update_history`, `02_v3_progress_checklist`, `03_v3_conversation_and_decision_log`를 active tracking 문서로 유지

## 7. Reference-Driven Design Decision

요청 요약:

- reference paper comparison을 실제 HBTXR 설계 의사결정으로 연결

결정:

- `FACET`는 dataset contract, resize policy, ellipse-aware supervision의 1차 reference
- `EX-Gaze`는 runtime `Search | Track` split, previous-state usage, online gating 철학의 1차 reference
- `Swift-Eye`는 selective quality/occlusion gating 아이디어만 제한적으로 차용

문서 결과:

- `docs/others/jetcas/20260326_011438_04_reference_ex_gaze_facet_swift_eye_analysis.md`
- `docs/others/jetcas/20260326_011438_05_reference_driven_hbtxr_design_adoption.md`

## 8. Code Architecture Review Request

요청 요약:

- `E:\WSL\Shared\ETRI_SYNC\HBTXR\paper_works\CODE\HBTXR_v3_0` 코드를 자세히 분석

결정 및 결과:

- active `v3` surface 기준 정적 구조 분석 수행
- `raw -> canonical -> manifest -> dataset -> model -> trainer -> runtime` 흐름을 문서화
- 결과 문서: `docs/prj/20260330_170419_01_v3_code_architecture_analysis.md`

주요 결론:

- 현재 코드는 contract-driven research baseline에 가깝다
- preprocessing, manifest, loader, model, trainer, runtime가 명확히 분리되어 있다
- runtime FSM은 존재하지만 active CLI `eval/infer` 경로에 직접 통합되어 있지 않다

## 9. Dataflow Walkthrough Request

요청 요약:

- annotation 생성, 데이터셋 처리, canonical 생성, manifest 생성, dataloader 흐름을 코드 수준으로 설명

결정 및 결과:

- 다음 흐름이 코드 기준으로 정리되었다
  - CSV 또는 scaffold annotation row 생성
  - `canonicalize_dataset()` / `canonicalize_session()`
  - `frame_annotations.jsonl`, `session_package.json`, `events.npz` 생성
  - `build_manifests()`로 sample row 생성
  - `EVEyeHBTXRDataset.__getitem__()`에서 학습용 tensor sample 조립
  - `trainer.make_loader()`로 DataLoader 구성

핵심 해석:

- canonical은 source of truth
- manifest는 sample index
- dataset은 supervised sample assembler
- dataloader는 batching wrapper

## 10. Dataloader And Loss Explanation Request

요청 요약:

- dataloader와 손실함수까지 이어서 설명

결정 및 결과:

- sample field별 소비처를 `model 입력 / head 출력 / loss / metric` 기준으로 정리
- `stage1`과 `stage2` 손실 구조를 분리해 설명
- `search / event / track` head별 입력 차원, 출력 차원, target 의미, loss 의미를 정리

핵심 해석:

- `frame`, `event`, `prev_state`만이 실질적인 model 입력
- `search/pupil`과 `event/pupil`은 direct state prediction
- `track/pupil`은 residual parameter prediction
- `track/state`는 residual decode 결과

## 11. Static Gaps Identified During Recent Review

최근 정적 분석에서 아래 gap들이 합의된 known issue로 정리되었다.

- `similarity_target`
  - dataset batch에는 들어가지만 active loss/metric에서 미사용
- `polarity_split`
  - config에 있으나 event voxel 생성에서 실질 분기 없음
- `grad_clip_norm`
  - config에 있으나 trainer에서 실제 적용 없음
- `RuntimeHBTXRTracker` / `model.runtime_step()`
  - 존재하나 main CLI path와 분리
- `split_users_random()`
  - 이름과 달리 deterministic ordered split
- `constraint_center`
  - 현재 eye box center anchor 수준 구현

## 12. Current Known Gaps

- 실행 환경 검증
  - 현재 셸에서는 `python`/`pytest` 직접 확인이 제한적이므로 runtime verification 미완료
- Grounded-SAM 실제 inference integration
  - scaffold는 존재하나 external pipeline 자동 연결은 후속 작업
- hardware/export 검증
  - 문서와 boundary 정의는 존재하나 synthesis/export/integration 검증은 후속 작업

## Decision Checklist

- [x] Active `v3` surface 기준 확정
- [x] Official ABI 확정
- [x] Experiment root workflow 확정
- [x] Annotation -> canonical -> manifest -> dataloader 흐름 설명 완료
- [x] Dataloader / loss / head 구조 설명 완료
- [x] Static gap 목록 정리
- [ ] Runtime verification completed

## 13. FACET Table 3 Event Builder Alignment

Requested change:

- align the active `v3` default with the FACET ablation conclusion
- explain exactly where event binning happens
- explain exactly where event accumulation happens
- explain how the event tensor shape changes along the path

Decision:

- adopt `fixed_count 5000 + fast_causal_linear(limit=25)` as the active default builder
- keep `time_bin` and `causal_linear` as supported ablation paths
- document the execution path instead of leaving it implicit

Implementation result:

- `configs/base.yaml`
  - default accumulation changed to `fast_causal_linear`
- `src/hbtxr/data/dataset.py`
  - added fast-causal saturating accumulation
- `src/hbtxr/preprocess/build_manifests.py`
  - manifest rows now store `fast_causal_limit`
- `docs/prj/20260330_170419_02_event_binning_and_accumulation_flow.md`
  - records the exact function-level flow and shape transitions

Key technical conclusion:

- binning happens in `_event_indices_for_window()`
- accumulation weights are generated in `_accumulation_weights()`
- fast-causal saturation is applied in `_accumulate_events()`
- dense sensor voxel generation happens in `_build_event_frame()`
- resize to model input happens in `_apply_transform_to_event()` and `_build_event_input()`
- final sample tensor is `event [2,256,256]`, and final batch tensor is `event [B,2,256,256]`

Validation note:

- `compileall` succeeded
- full `pytest` execution is still blocked by the current environment

## Conversation Update Checklist

- [x] FACET Table 3 question resolved
- [x] default builder decision recorded
- [x] function-level event flow recorded
- [x] tensor-shape transition recorded
- [ ] runtime test execution recorded

## 14. Grounded-SAM Pipeline Request

Request summary:

- use the provided raw dataset path as read-only input
- use the provided Grounded-SAM repository as read-only input
- do not use UNet training for annotation
- add scripts for annotation, canonicalization, manifest generation, overlay preview, train, eval, infer
- update README, requirements, `.gitignore`, and docs history/conversation/progress

Decision:

- treat both external paths as immutable inputs
- export Grounded-SAM annotations into a project-local workspace root
- let canonicalization consume those exports via `annotation_mode=groundedsam`
- add `sh` wrappers so the main user flow is runnable from Linux shell without manual Python command assembly
- keep overlay preview downstream of canonical data so preview renders the same annotation contract used by training

Implementation result:

- `src/hbtxr/preprocess/groundedsam_build.py`
- `scripts/annotate_groundedsam_ev_eye.py`
- `scripts/build_groundedsam_dataset.py`
- `scripts/overlay_preview.py`
- `scripts/run_groundedsam_annotation.sh`
- `scripts/run_canonicalize.sh`
- `scripts/run_build_manifests.sh`
- `scripts/build_dataset_groundedsam.sh`
- `scripts/run_overlay_preview.sh`
- `scripts/run_train.sh`
- `scripts/run_eval.sh`
- `scripts/run_infer.sh`
- `configs/paths/ev_eye_groundedsam_paths.json`
- `docs/prj/20260330_170419_03_groundedsam_dataset_pipeline.md`

## 15. Vis Action Run Contract Integration

Request summary:

- continue the next planned step
- explain and then implement full `vis` action integration
- update the history, conversation log, and progress tracking documents together with the code change

Decision:

- treat `vis` as a first-class action in the same family as `train`, `eval`, and `infer`
- remove the old dedicated `resume` shell action from the official path
- let `eval`, `infer`, and `vis` reuse an existing timestamped run root through `--resume`
- move manifest / results / output fallback resolution into shared helper-backed Python entrypoints instead of leaving it only in shell glue

Implementation result:

- `scripts/_config.py`
  - added helpers for latest run lookup, resume root resolution, stage checkpoint defaults, and manifest fallback resolution
- `scripts/hbtxr_mode_pipeline.sh`
  - rewritten around `train | eval | infer | vis`
  - added `visualize -> vis` compatibility alias
- `scripts/eval_hbtxr.py`
  - now supports `--resume`
  - now resolves default manifest/checkpoint paths from the run contract when possible
- `scripts/infer_hbtxr.py`
  - same run-root reuse policy as `eval`
- `scripts/visualize_dataset.py`
- `scripts/visualize_inference_results.py`
- `scripts/visualize_runtime.py`
  - now resolve `vis` output roots from `run_contract`
  - now emit `hypers/` records like the other action entrypoints
- `scripts/run_vis.sh`
  - added as the direct shell wrapper for the new official action

Current gap after the change:

- the structural path is now aligned
- full runtime validation still depends on an environment with the required Python packages and runnable checkpoints

## 16. Documentation / History / Plan Status Sync

Request summary:

- update the latest project changes in the documentation
- summarize recent conversation and history
- re-check implementation progress against the active plan
- output the current status in checklist form

Decision:

- keep the implementation plan status unchanged unless a real code/runtime milestone moved
- synchronize `chat`, `plan`, and `jetcas` logs on the same day so history and manuscript status do not drift apart
- explicitly record that the current sync cycle is documentation-driven, not a new runtime-validation cycle

Implementation result:

- refreshed `docs/chat/20260326_011438_01_v3_update_history.md`
- refreshed `docs/chat/20260326_011438_02_v3_progress_checklist.md`
- refreshed `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`
- refreshed `docs/plan/20260326_011438_03_v3_1_update_plan.md`
- refreshed `docs/others/jetcas/20260326_011438_02_jetcas_update_log.md`

Status conclusion:

- plan progress remains structurally the same as the previous implementation pass
- completed items remain `mode1/mode2` contract, run contract, preload path, loss catalog, and `vis` integration
- pending items remain runtime validation, full `pytest`, and full `mode2` preprocessing completion

## 17. SSL / Pruning Reference-Only Follow-Up And Full Test Run

Request summary:

- implement `SSL reference-only follow-up reflected in code/config`
- implement `pruning reference-only follow-up reflected in code/config`
- run the full `pytest` suite

Decision:

- keep both features inactive in the current pipeline
- represent both features explicitly in the active config
- emit run-time artifacts so future experiments can track that the features are reserved but not wired
- validate the result with the full test suite instead of another partial static-only pass

Implementation result:

- `configs/base.yaml`
  - added `ssl` reference-only placeholder block
  - added `pruning` reference-only placeholder block
- `scripts/_config.py`
  - added reference-only report collection
  - `write_run_artifacts()` now emits placeholder reports into `hypers/`
- `tests/test_config_v3.py`
  - added coverage for the new placeholder surface

Validation result:

- full `pytest` completed successfully in WSL
- outcome:
  - `11 passed`
- remaining non-blocking issues:
  - AMP deprecation warnings
  - `pin_memory(device=...)` deprecation warnings

## 18. JETCAS Rollback And Future Edit Boundary

Request summary:

- roll back the most recent modifications that had just been applied to selected `JETCAS`-adjacent draft and tracking files
- avoid touching `JETCAS` internal directories in future work
- when a commit is requested later, include the existing `JETCAS` changes together with that commit

Decision:

- treat the rollback as a document-only recovery step
- revert only the newly added `JETCAS`-related notes and helper-file changes without disturbing other active implementation work
- record a new working boundary:
  - do not modify `docs/jetcas/` by default
  - still include existing `JETCAS` changes when the user explicitly asks for a commit

Implementation result:

- reverted the requested document-only changes
- removed the newly created `41.table_examples_latex.md`
- preserved the rest of the `HBTXR_v3_0` implementation state
- applied the future no-touch policy for `docs/jetcas/`

Status conclusion:

- no code-path or training/runtime milestone changed in this step
- the current implementation status remains:
  - `mode1/mode2` contract in place
  - run contract in place
  - preload path in place
  - loss catalog in place
  - `vis` integration in place
  - full `pytest` completed
  - runtime end-to-end validation still pending

## 19. Actual SSL / Pruning Wiring, Warning Cleanup, README Expansion, And Runtime Validation

Request summary:

- implement the previously pending items:
  - warning cleanup
  - actual `SSL` wiring
  - actual `pruning` wiring
  - end-to-end checkpoint + manifest runtime validation
- expand `README.md` with the full project structure and detailed training mode / stage explanations
- then update the project tracking documents and commit the result

Decision:

- promote `SSL` and `pruning` from placeholder-only tracking items to actual optional code paths
- keep both features disabled by default in `configs/base.yaml`
- treat synthetic end-to-end checkpoint + manifest execution as the required validation milestone for this phase
- keep `docs/jetcas/` untouched and record the implementation only in `docs/chat/` and `docs/plan/`

Implementation result:

- `src/hbtxr/training/trainer.py`
  - active `SSL` integration
  - active pruning integration
  - `torch.amp` migration
- `src/hbtxr/training/losses.py`
  - feature-consistency, KD, and RKD additions
- `src/hbtxr/models/hybrid_tracker.py`
  - width masking and active-width reporting
- `src/hbtxr/data/loader.py`
  - refined `pin_memory` resolution
- `tests/test_runtime_e2e_v3.py`
  - new synthetic end-to-end runtime test
- `README.md`
  - project map
  - `mode1` / `mode2`
  - `stage1` / `stage2`

Validation result:

- local `.venv` repaired and populated for this repository
- full regression suite:
  - `14 passed`
- deprecation-focused rerun:
  - `14 passed`
- current remaining gap:
  - real dataset / checkpoint validation is still not completed

## 20. Repository Tree Mapping, Grounded-SAM Runtime Compatibility, And Commit Preparation

Request summary:

- focus only on `HBTXR_v3_0`
- re-draw the repository tree and explain `src/hbtxr` module dependencies
- document the current updates, conversation history, and progress state
- resolve the sequential Grounded-SAM runtime errors seen during annotation execution
- prepare the current state for an `exp` branch handoff and commit

Decision:

- keep all compatibility fixes inside `HBTXR_v3_0` rather than editing the external `grounded-segment-anything` repository
- treat `groundedsam_root` as a read-only external dependency root
- accept the PyTorch deformable-attention fallback when the `_C` extension is unavailable
- keep technical flow changes in `docs/hbtxr/` and process/history changes in `docs/chat/`

Implementation result:

- added `docs/hbtxr/06_v3_repository_tree_and_module_dependency_map.md`
- expanded `docs/prj/20260330_170419_03_groundedsam_dataset_pipeline.md`
- patched `src/hbtxr/preprocess/groundedsam_build.py` for:
  - nested `segment_anything` layout handling
  - `transformers 5.x` BERT helper compatibility
  - explicit GroundingDINO `device` propagation
  - `_C`-missing deformable-attention fallback
- updated path/dependency tracking files:
  - `configs/paths/ev_eye_groundedsam_paths.json`
  - `configs/paths/reference_path.json`
  - `requirements.txt`
  - `.gitignore`
- synchronized update history and progress checklist

Validation result:

- local `.venv` import check passed for `segment_anything`
- CPU runtime initialization passed
- CPU dummy-image annotation passed
- remaining gap:
  - no CUDA validation in the current sandbox
  - no full real-dataset rerun in this session

## 21. Patch-Embed Bundle Integration Request

Request summary:

- implement the external `hbtxr_patch_embed_bundle` ideas directly inside `HBTXR_v3_0`
- keep the current path intact and expose the new front-end as optional variants
- rename the planned variants to:
  - `split1`
  - `split2`
- preserve the current dataset and training ABI
- integrate the result into tracker, runtime, trainer, config, pretrained loading, and tests

Decision:

- keep `legacy` as the default path to avoid breaking existing checkpoints and configs
- absorb the bundle logic into the in-repo model surface instead of adding a runtime dependency on the external folder
- define the variants as:
  - `split1 = shared 3ch + mode affine + frame psum cache`
  - `split2 = split 1ch frame + 2ch event + frame cache`
- keep pseudo-edge generation inside the model, not in the dataset
- extend validation in two layers:
  - repository-path ABI/cache tests
  - local temp-clone pytest rerun to avoid shared-drive temp permission issues

Implementation result:

- `src/hbtxr/models/patch_embeddings.py`
  - added the variant-aware patch front-end stack
- `src/hbtxr/models/hybrid_tracker.py`
  - switched to a unified patch-front-end interface
  - added patch-cache helpers and variant-aware pretrained mapping
- `src/hbtxr/runtime/tracker.py`
  - added runtime patch-cache propagation
- `src/hbtxr/training/trainer.py`
  - added epoch-context propagation for `split1` warmup control
- `src/hbtxr/models/preload/pretrained_loader.py`
  - now respects tracker-side patch-weight remapping
- `configs/base.yaml`
  - added patch-embed defaults
- `tests/test_model_v3.py`
  - added `legacy/split1/split2` ABI and cache tests
- `tests/test_pretrained_loader_v3.py`
  - added `split1/split2` pretrained loader checks

Validation result:

- direct repository-path ABI/cache suite:
  - `tests/test_model_v3.py`
  - `6 passed`
- local temp-clone rerun:
  - `tests/test_pretrained_loader_v3.py`
  - `3 passed`
  - `tests/test_runtime_e2e_v3.py`
  - `1 passed`

Status conclusion:

- the optional patch-embed integration is now implemented and validated
- the earlier temp-path test failures were caused by the shared-drive temp environment, not by the feature itself

## 21. Mode1 / Mode2 Completion Follow-up

Request summary:

- continue the unfinished `mode1/mode2` work
- focus on the still-open `mode2 interpolation preprocess path`

Decision:

- treat the remaining gap as a preprocessing problem, not a loader-only metadata problem
- implement synthetic frame generation during canonicalization for adjacent labeled pairs
- export interpolation provenance so `canonical2`, `manifest2`, and `Mode2Dataset` share the same source of truth
- keep `docs/jetcas/` untouched and record the status only in `docs/chat/` and `docs/plan/`

Implementation result:

- `src/hbtxr/preprocess/canonicalize.py`
  - added `mode2` synthetic frame generation
  - added interpolation summary export and row-level provenance
- `src/hbtxr/preprocess/session_package.py`
  - now stores interpolation provenance when available
- `scripts/prepare_ev_eye.py`
  - now accepts interpolation CLI arguments
- `tests/conftest.py`
  - added a raw fixture for `mode2`
  - switched test temp creation to a writable Windows temp root
- `tests/test_preprocess_v3.py`
  - added `mode2` canonicalize + manifest coverage
- `tests/test_dataset_v3.py`
  - added `Mode2Dataset` synthetic metadata coverage

Validation result:

- targeted preprocess/dataset suite:
  - `6 passed`
- full regression suite:
  - `22 passed`
- deprecation-focused rerun:
  - `22 passed`

Status conclusion:

- `mode2` preprocessing is no longer scaffold-only
- remaining `mode2` work is now limited to full runtime train/eval/infer validation on the mode-aware path

## 22. Mode2 Runtime Sweep Follow-up

Request summary:

- continue from the completed `mode2` preprocessing work
- validate the remaining `mode2` runtime path end-to-end

Decision:

- keep the runtime harness synthetic for fast regression coverage
- generalize the existing `mode1` end-to-end runtime test rather than add a separate ad hoc execution path
- treat completion as:
  - `stage1 train`
  - `stage2 train`
  - `eval`
  - `infer`
  - `visualize_inference_results`
  - `visualize_runtime`

Implementation result:

- `tests/test_runtime_e2e_v3.py`
  - `_write_config()` now supports `mode1` and `mode2`
  - added `_run_runtime_cycle()` shared helper
  - existing `mode1` runtime test now uses the shared helper
  - added dedicated `mode2` runtime validation from canonicalization output through visualization

Validation result:

- targeted runtime suite:
  - `2 passed`
- full regression suite:
  - `23 passed`
- deprecation-focused rerun:
  - `23 passed`

Status conclusion:

- synthetic `mode2` runtime validation is now complete
- the remaining runtime gap is limited to real dataset / real checkpoint execution

## 23. Canonical Split / Synthetic Grounded-SAM / README Sync Follow-up

Request summary:

- complete the remaining partial items:
  - physical `canonical1/canonical2` artifact split
  - `mode2` synthetic-frame Grounded-SAM rerun
  - README synchronization with the latest implementation state

Decision:

- keep the storage model centered on a single configurable canonical workspace root
- physically separate datasets by resolving `canonical_root/<canonical_name>/...`
- preserve backward-compatible read behavior for older flat canonical trees
- implement synthetic-frame Grounded-SAM rerun as an optional canonicalization refinement that only activates when the external Grounded-SAM runtime is configured
- keep `docs/jetcas/` outside the edit scope

Implementation result:

- `src/hbtxr/utils/paths.py`
  - added canonical dataset/index root resolution helpers
- `src/hbtxr/preprocess/canonicalize.py`
  - now writes `mode1` and `mode2` into physically separated canonical roots
  - added synthetic-frame Grounded-SAM rerun and interpolation-summary reporting
- `src/hbtxr/preprocess/build_manifests.py`
  - now resolves nested canonical roots before reading session metadata
- `src/hbtxr/preprocess/groundedsam_build.py`
  - exposes a reusable runtime builder for synthetic rerun
- `src/hbtxr/data/dataset.py`
  - resolves nested canonical roots for dataset loading
- `scripts/prepare_ev_eye.py`
  - forwards the Grounded-SAM root and mode-aware canonical resolution
- `scripts/build_groundedsam_dataset.py`
  - forwards the Grounded-SAM root and mode-aware canonical resolution
- `scripts/overlay_preview.py`
  - now resolves indexes and stored paths against the active canonical dataset root
- `src/hbtxr/preprocess/path_utils.py`
  - example path generation now treats `canonical_root` as a workspace root
- `configs/paths.example.json`
  - synced to the workspace-root canonical layout
- `README.md`
  - synced to the physical split and synthetic rerun implementation
- `tests/test_preprocess_v3.py`
  - added mode1 nested-root coverage and mode2 synthetic rerun coverage

Validation result:

- preprocess suite:
  - `4 passed`
- dataset suite:
  - `4 passed`
- runtime suite:
  - `2 passed`
- full repository suite:
  - `25 passed`
- deprecation-focused full repository suite:
  - `25 passed`

Status conclusion:

- the three previously partial items are now implemented
- the remaining unresolved runtime item is still real dataset / real checkpoint validation

## 24. `origin/exp` Merge Into `main`

Request summary:

- check whether remote `exp` can be reflected into the current `main`
- identify conflict points
- proceed with the merge work

Decision:

- treat the merge as a real integration task instead of a read-only comparison
- preserve the current local work by stashing before merge
- resolve `docs/chat` conflicts by keeping both sides because they were log/history additions, not mutually exclusive logic changes

Implementation result:

- fetched `origin/exp` and confirmed the common base against `main`
- merged `origin/exp` into `main`
- resolved conflicts in:
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`
- restored the pre-merge local work and resolved the additional conflict in:
  - `tests/test_preprocess_v3.py`
- created and pushed merge commit:
  - `45af406`
  - `Merge origin/exp into main`

Validation result:

- repository regression suite after merge recovery:
  - `29 passed`

Status conclusion:

- `origin/exp` is now integrated into `main`
- the merge conflict surface stayed limited to the expected tracking documents

## 25. `ref_cods` Removal / Follow-up Commit / Stash Cleanup

Request summary:

- inspect the remaining dirty state after the merge
- treat `ref_cods` deletion as intentional
- complete the follow-up commit and push
- decide whether the remaining stash should be kept

Decision:

- split the remaining state into:
  - meaningful HBTXR local work
  - large `ref_cods` deletion set
- honor the user request by keeping the `ref_cods` removal in the follow-up commit
- include existing `docs/jetcas` worktree files when committing because the user previously requested that behavior
- drop the stash after confirming the committed `main` already contained its payload

Implementation result:

- committed the follow-up repository state:
  - `dd9a1ee`
  - `Finalize mode2 GroundedSAM sync and remove ref_cods`
- pushed `dd9a1ee` to `origin/main`
- dropped the leftover stash:
  - `stash@{0}` removed

Validation result:

- repository regression suite before the follow-up commit:
  - `29 passed`
- final repository state:
  - worktree clean
  - stash list empty

Status conclusion:

- `main` now contains:
  - merged `origin/exp`
  - the latest local `mode2` / Grounded-SAM sync work
  - intentional `ref_cods` removal
- the primary remaining gap is still real dataset / real checkpoint validation

## 26. Structural Distillation Redesign Follow-up

Request summary:

- continue the compression redesign work
- provide concrete recommended presets for:
  - `mode1/stage1`
  - `mode1/stage2`
  - `mode2/stage1`
  - `mode2/stage2`
- clarify the design basis for those presets

Decision:

- keep `distillation` as the primary compression path
- keep `regularization_ssl` disabled in the recommended teacher-guided presets
- use `metric_search_p10_pct` as the Stage1 control metric
- use `metric_track_p10_pct` as the Stage2 control metric
- use cosine scheduling for Stage1 foundation learning
- use plateau scheduling for Stage2 hybrid refinement

Implementation result:

- updated:
  - `configs/mode1_stage1.yaml`
  - `configs/mode1_stage2.yaml`
  - `configs/mode2_stage1.yaml`
  - `configs/mode2_stage2.yaml`
- updated `README.md` to explain the new compression structure and recommended presets
- added preset-resolution coverage in `tests/test_config_v3.py`

## 27. Stage-Aware Distillation And Regularization SSL Gating

Request summary:

- explain whether `SSL` and `Distillation` are the same in this project
- explain how strict stage-aware distillation gating should be implemented
- then implement the gating

Decision:

- treat `Distillation` and `regularization_ssl` as different mechanisms
- enforce stage policies in the loss layer rather than relying only on preset convention
- keep default stage policies in config and allow explicit override for ablations

Implementation result:

- `compute_distillation_losses(..., stage=...)`
  - now uses stage-aware key allowlists
- `compute_regularization_ssl_losses(..., stage=...)`
  - now uses stage-aware key allowlists
- `configs/base.yaml`
  - now includes default stage target policies for both paths
- `tests/test_loss_catalog_v3.py`
  - verifies:
    - Stage1 blocks event/track distillation by default
    - Stage1 override can reopen blocked keys
    - Stage1 blocks regularization SSL by default
    - Stage2 enables hybrid regularization keys

Validation result:

- targeted loss tests passed
- targeted train-pipeline tests passed
- full repository regression passed:
  - `37 passed`

Current conclusion:

- the implementation now matches the intended `stage1` and `stage2` preset semantics
- remaining open items are external runtime validation tasks, not internal stage-policy inconsistencies

## 28. Active `qc.py` Retirement / Legacy Migration

Request summary:

- move all `qc.py`-related active preprocess content into the legacy surface
- include the documentation updates in the same pass
- update:
  - update history
  - progress checklist
  - conversation/decision tracking
- report plan-vs-progress status as a checklist

Decision:

- do not keep `qc.py` active just to preserve two protocol helpers
- split protocol/session-prior helpers into a new active module
- remove `session_qc` from the active canonical/session-package contract instead of carrying a dormant QC payload
- preserve the original QC helper implementation in `legacy/src/hbtxr/preprocess/qc.py`

Implementation result:

- added active module:
  - `src/hbtxr/preprocess/protocol.py`
- archived old helper surface:
  - `legacy/src/hbtxr/preprocess/qc.py`
- removed active file:
  - `src/hbtxr/preprocess/qc.py`
- updated active preprocess wiring:
  - `src/hbtxr/preprocess/canonicalize.py`
  - `src/hbtxr/preprocess/session_package.py`
- updated active documentation:
  - `docs/hbtxr/06_v3_repository_tree_and_module_dependency_map.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Validation result:

- targeted preprocess regression:
  - `8 passed`
- full repository regression:
  - `46 passed`
- active `.py` search confirmed:
  - no remaining `hbtxr.preprocess.qc` imports
  - no remaining active `session_qc` references

Current conclusion:

- `qc.py` is now purely a legacy/archive concern
- active preprocess retains only protocol/session-prior metadata helpers
- the remaining unresolved validation item is still external real dataset / real checkpoint execution

## 29. External Packages Surface / End-To-End README / Script Coverage

Request summary:

- create a new `packages/` surface for external dependencies used by the active project
- generate scripts so those packages can be downloaded / updated from inside the workspace
- cover the full operational flow in `README.md`:
  - dataset generation
  - annotation
  - canonical
  - manifest
  - dataloader
  - Stage1 training for `mode1` and `mode2`
  - Stage2 training for `mode1` and `mode2`
  - inference
  - export
- verify each step has a concrete executable script
- add missing scripts where the surface is incomplete

Decision:

- standardize external repositories under `./packages/` instead of ad hoc absolute paths
- keep package clones out of version control via `packages/.gitignore`
- expose package sync as a first-class script surface
- treat README as the operator handbook, not a short marketing note
- add explicit utility entrypoints where the end-to-end flow was previously implicit:
  - dataloader inspection
  - standalone export
  - prepare wrapper
  - relocate wrapper
  - package sync wrapper
- do not include `docs/references_analysis/` in this runtime-focused change set

Implementation result:

- added:
  - `src/hbtxr/utils/external_packages.py`
  - `scripts/sync_packages.py`
  - `scripts/check_dataloader.py`
  - `scripts/export_hbtxr.py`
  - `scripts/run_sync_packages.sh`
  - `scripts/run_prepare.sh`
  - `scripts/run_relocate.sh`
  - `scripts/run_dataloader.sh`
  - `scripts/run_export.sh`
  - `packages/README.md`
  - `packages/repositories.json`
  - `packages/.gitignore`
  - `tests/test_external_packages_v3.py`
  - `tests/test_script_workflows_v3.py`
- updated:
  - `README.md`
  - `scripts/hbtxr_mode_pipeline.sh`
  - `scripts/prepare_ev_eye.py`
  - `src/hbtxr/preprocess/path_utils.py`
  - `src/hbtxr/preprocess/canonicalize.py`
  - `configs/paths.example.json`

Validation result:

- targeted workflow regression:
  - `2 passed`
- targeted package/path regression:
  - `6 passed`
- full repository regression:
  - `54 passed`

Current conclusion:

- the repository now has a concrete command surface for the entire preprocess-to-export flow
- package path handling is normalized under `packages/`
- the only missing default package entry is `Swift-Eye`, pending the repository URL

## 30. `Swift-Eye` Default Package Manifest And Reference Analysis Activation

Request summary:

- add `Swift-Eye` to the default `packages/` manifest using:
  - `https://github.com/ztysdu/Swift-Eye`
- explain how the remaining `docs/prj/20260325_140031_00_docs_index.md` and `docs/references_analysis/` state should be handled
- if the docs structure is now stable, commit both together as one coherent doc-surface change

Decision:

- treat `Swift-Eye` as a first-class managed external dependency alongside Grounded-SAM and TimeLens
- do not leave `docs/references_analysis/` half-active
- activate `docs/references_analysis/` and sync `docs/prj/20260325_140031_00_docs_index.md` in the same pass

Implementation result:

- updated:
  - `packages/repositories.json`
  - `packages/README.md`
  - `README.md`
  - `docs/prj/20260325_140031_00_docs_index.md`
- added:
  - `docs/prj/references/20260330_170419_00_index.md`
  - `docs/references_analysis/facet/00_index.md`
  - `docs/references_analysis/swift-eye/00_index.md`

Validation result:

- package manifest dry-run succeeded
- targeted package-surface regression:
  - `6 passed`

Current conclusion:

- `Swift-Eye` is now part of the default external package manifest
- `docs/references_analysis/` is now an active tracked documentation subtree
- the package and reference-analysis surfaces are structurally consistent

## 31. Real External Validation And WSL GPU 5-Epoch End-To-End Smoke

Request summary:

- use the real external roots supplied by the user:
  - Grounded-SAM:
    - `E:\WSL\Shared\ETRI_SYNC\HBTXR\annotation_tools\Grounded-Segment-Anything-main`
  - TimeLens / Swift-Eye:
    - `E:\WSL\Shared\ETRI_SYNC\HBTXR\references`
- download Grounded-SAM weights
- finish the three previously unexecuted external validations
- rerun the whole project smoke on WSL GPU for about `5 epochs`

Decision:

- use WSL as the authoritative GPU validation environment
- validate Grounded-SAM with real weights and real runtime execution
- validate TimeLens through the real checkpoint-backed `mode2` canonicalization path
- validate Swift-Eye as a downstream consumer reference and record any environment blocker explicitly
- keep the smoke dataset intentionally tiny because the goal is execution-contract validation, not accuracy benchmarking

Implementation result:

- Grounded-SAM weights downloaded into the external repository:
  - `groundingdino_swint_ogc.pth`
  - `sam_vit_h_4b8939.pth`
- WSL GPU runtime confirmed:
  - `torch 2.11.0+cu130`
  - `cuda_available=True`
  - device: `NVIDIA GeForce RTX 4070 Ti`
- completed:
  - real Grounded-SAM annotation smoke
  - real TimeLens checkpoint-backed `mode2` canonicalization smoke
  - `mode1` canonicalization / manifest / dataloader smoke
  - `mode2` canonicalization / manifest / dataloader smoke
  - `mode1-stage1` 5-epoch GPU train smoke
  - `mode1-stage2` 5-epoch GPU train smoke
  - `mode2-stage1` 5-epoch GPU train smoke
  - `mode2-stage2` 5-epoch GPU train smoke
  - `mode1-stage2` inference / export smoke
  - `mode2-stage2` inference / export smoke

Validation result:

- Grounded-SAM:
  - passed with real weights and real annotation output
  - custom C++ op remained on fallback path
- TimeLens:
  - passed with `refined_model/attention.bin`
- Swift-Eye:
  - repository and weights confirmed
  - import blocked by missing `mmrotate.core`

Current conclusion:

- the main HBTXR pipeline is now validated on WSL GPU across annotation through export
- the remaining unresolved external-runtime gap is the dedicated `Swift-Eye` consumer environment

## 32. Grounded-SAM Eye-Region BBox Preview, Blink Investigation, And Raw-Ellipse Heuristic Wiring

Request summary:

- make `gpu0` and `gpu1` work on disjoint Grounded-SAM annotation shards
- build a Grounded-SAM path that exports only eye-region bounding boxes and overlay previews for `user01/session101`
- avoid heuristic eye-region synthesis in that bbox-only path
- analyze the bbox results and explain why blink-like frames sometimes keep a box and sometimes do not
- check whether the raw EV-Eye dataset already includes `open/closed/blink` metadata
- if not, derive blink candidates from raw CSV ellipse annotations and connect them to the pipeline

Decision:

- use session-level sharding for multi-GPU work instead of a single multi-device `torch.device`
- keep the bbox-only preview path separate from the full Grounded-SAM annotation path
- reject near-full-frame zero-shot `eye` detections rather than exporting obviously wrong near-eye boxes
- treat raw EV-Eye CSV annotation as the authoritative source for blink heuristics because it contains ellipse geometry but no explicit blink label
- wire blink candidates into manual-CSV canonicalization and manifest generation first

Implementation result:

- added a bbox-only Grounded-SAM eye-region preview/export surface
- added overlay preview generation and conservative large-box rejection
- added raw-user directory resolution for both `user1` and `user01`
- added raw-ellipse blink-candidate analysis and CLI wrappers
- added manual-CSV `closed_eye_flag` and blink metadata propagation through canonicalization and manifests
- synchronized active chat/history/progress documents for the session

Validation result:

- bbox-only output for `user01/session101` was analyzed:
  - `10111` total frames
  - `10080` detected
  - right eye `5041 / 5041`
  - left eye `5039 / 5070`
  - all `31` failures were near-full-frame rejects
- raw dataset inspection confirmed there is no explicit `blink/open/closed/eye_state` field in the raw CSV annotations
- raw-ellipse real-session analysis on `user1/left/session102` found `2` blink candidates across `27` annotations

Current conclusion:

- the project now has a dedicated bbox-only Grounded-SAM preview path for near-eye ROI inspection
- blink-state handling became available for manual CSV sessions via a raw-ellipse heuristic

## 33. Grounded-SAM Annotation Store Blink Classification Follow-Up

Request summary:

- proceed with Grounded-SAM annotation-store-level blink classification
- make the store itself carry blink metadata rather than leaving blink handling only to raw CSV sessions

Decision:

- reuse the same ellipse-collapse heuristic family rather than inventing a separate blink rule for Grounded-SAM stores
- classify Grounded-SAM stores from their exported ellipse sequence after session export completes
- support both new store generation and existing store reuse via backfill
- also derive the same metadata in-memory during canonicalization so older stores can still flow downstream correctly

Implementation result:

- store-level blink metadata builder and backfill helper added
- Grounded-SAM export now applies blink metadata after session rows are sorted
- reused stores are rewritten only when blink fields were missing
- canonicalization now preserves the same metadata path for Grounded-SAM rows
- pipeline docs and tests updated for the new store-level fields

Validation result:

- `py_compile` passed for the updated blink, Grounded-SAM build, canonicalize, and test files
- manual helper verification confirmed:
  - `closed_eye_flag` backfill
  - `blink_candidate_source=groundedsam_ellipse_heuristic`
  - `Grounded-SAM store -> canonical row` propagation

Current conclusion:

- Grounded-SAM sessions are no longer blocked from blink classification at the data-contract level
- the remaining gap moved from implementation to real-session runtime coverage

## 34. Focused Real Grounded-SAM Blink Validation On Previously Discussed Blink Regions

Request summary:

- test the previously discussed blink cases with the updated Grounded-SAM store blink classifier
- validate the result on real inference rather than helper-only synthetic rows
- compare the new blink flags against earlier bbox-only rejects and raw-CSV blink candidates

Decision:

- do not run a full dataset rerun yet
- build a temporary raw subset containing only the previously discussed blink-region frames
- use the actual `annotate_groundedsam_ev_eye.py` runtime on `cuda:0`
- compare store-level blink flags against both prior analysis tracks

Implementation result:

- created a temporary raw subset for:
  - `user01/left/session101`: `005004` to `005012`
  - `user01/left/session102`: `000309` to `000311`, `001409` to `001411`
- ran actual Grounded-SAM annotation on the subset with the configured checkpoints
- inspected the generated stores and compared blink flags with earlier bbox-only and raw-CSV analyses

Validation result:

- `session101`
  - `9` annotated frames
  - `3` blink-flagged frames:
    - `005008_1657710986457638.png`
    - `005009_1657710986497638.png`
    - `005012_1657710986617638.png`
- `session102`
  - `6` annotated frames
  - `2` blink-flagged frames:
    - `000311_1657711096857726.png`
    - `001411_1657711140857761.png`

Current conclusion:

- the Grounded-SAM store blink classifier has been exercised on real inference output, not only helper-level synthetic rows
- remaining work is now about full-session and full-dataset coverage expansion

## 35. Local Main Sync And Integrated v3.2 Planning Request

Request summary:

- compare the current `exp` repository against the latest local `main` repository under `/home/kjm26/project/PRJXR/HBTXR-v3_0-main`
- analyze overlap and possible conflict points in detail
- merge the current `exp` state with the latest local `main` state
- reflect the merged result back into the local `main` repository as well
- analyze the `v3.2` update plan in detail
- stop splitting execution into `v3.2.1` and rewrite the plan as one integrated `v3.2` plan
- produce an implementable work breakdown matching the requested four-step flow:
  - dataset build
  - dataset annotation
  - H5/session data structure
  - canonical / manifest generation
- keep history / progress / conversation tracking synchronized and print a plan-vs-progress checklist at the end of each work cycle

Decision:

- merge `main` into `exp` first, because the blink-related work lived on `exp` and needed to be preserved on top of the broader `main` changes
- resolve manual conflicts in:
  - `src/hbtxr/preprocess/canonicalize.py`
  - `docs/chat/*.md`
- keep the latest `main` structure for:
  - `protocol.py`-based canonicalization
  - interpolation/package/documentation expansion
  - `ref_codes/` tree
- preserve the `exp` Grounded-SAM bbox-preview and blink-classification work inside that merged structure
- fast-forward the local `main` repository from the merged `exp` result instead of creating a second divergent merge
- collapse the plan back into one integrated `v3.2` execution scope
- treat `eye_state_flag` as a binary `open|closed` field and map blink/closed-eye cases to `closed`, while using status/reason fields to represent mismatch cases

Implementation result:

- completed local branch sync:
  - merge commit created on `exp`:
    - `dfd4b92` `Merge main updates into exp with Grounded-SAM blink flow`
  - local `main` repository fast-forwarded to the same commit
- updated `docs/plan/20260327_163915_04_v3_2_update_plan.md` to:
  - remove the split `v3.2.1` execution structure
  - define one integrated `v3.2` execution scope
  - align the plan to the requested four-step pipeline
  - add a concrete implementation work breakdown
  - add a concrete validation checklist
- updated active tracking documents:
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Validation result:

- merge-step validation passed:
  - `python3 -m py_compile` passed for merged preprocess files
  - repository-local import check passed for `hbtxr.preprocess.canonicalize`
- repository sync state confirmed:
  - local `exp` HEAD = `dfd4b923f5afbc744313f36989833f6987c9221d`
  - local `main` HEAD = `dfd4b923f5afbc744313f36989833f6987c9221d`
  - both worktrees were clean after the sync
- current planning pass:
  - document-only update
  - no new code/runtime execution performed in this pass

Current conclusion:

- the local repositories are synchronized on one merged branch state
- the `v3.2` plan is now expressed as one integrated execution contract instead of a split phase document
- the next concrete implementation starting point is:
  - `Task 1` CLI/config surface
  - `Task 2` session/job enumeration
  - `Task 3` target timestamp grid builder

## 36. Target-FPS Bridge Checkpoint / Session-Store Loader / Failure-Queryability Completion

Request summary:

- commit the current target-FPS implementation state and continue to the next blocking work
- make the new target-FPS manifests consumable by the active dataset loader
- preserve skipped-session reasons and frame-level annotation failures as queryable outputs
- validate the H5-backed path, not only the fallback `npz` path
- synchronize chat/history documents and include the `requirements.txt` dependency updates in the final commit

Decision:

- first create a clean checkpoint commit for the initial target-FPS `build -> canonical -> manifest` bridge work
- keep backward compatibility in the dataset loader by preserving the old `frame_path/events_npz` path and only using session-store loading when those fields are absent
- represent partial/pending annotation behavior as explicit failure-index artifacts instead of trying to infer it later from raw frame-label scans
- preserve skip/failure visibility at all three levels:
  - target-FPS build
  - canonical bridge
  - manifest outputs
- install `h5py` and `pytest` into the repository-local `.venv` because H5 and focused regression validation had become the main remaining blockers
- sync `requirements.txt` to match the validated local environment

Implementation result:

- created checkpoint commit:
  - `0b7c79f`
  - `Add target-fps build and canonical manifest bridge`
- extended `src/hbtxr/data/dataset.py` to:
  - load frames from `session_store_path + frame_index`
  - rebuild event inputs from `session_store_path + event_index_range`
  - support both `npz` and `h5` session stores
- extended `src/hbtxr/preprocess/target_fps_build.py` to:
  - write per-session `annotation_failures.jsonl`
  - write aggregated `indexes/annotation_failures.jsonl`
  - export failure counts into `session_meta.json` and `build_summary.json`
- extended `src/hbtxr/preprocess/target_fps_canonical.py` to:
  - write `indexes/canonical_annotation_failures.jsonl`
- extended `src/hbtxr/preprocess/build_manifests.py` to:
  - preserve `skipped_sessions.jsonl`
  - write `annotation_failures.jsonl`
  - keep status / reason / ROI / pupil metadata and session-store references in manifest rows
  - use a safe ROI fallback for pending rows with missing eye ROI
- added or updated target-FPS regression files:
  - `tests/test_target_fps_build_v3.py`
  - `tests/test_target_fps_canonical_v3.py`
  - `tests/test_target_fps_manifest_v3.py`
  - `tests/test_target_fps_dataset_v3.py`
- updated `requirements.txt` with:
  - `h5py`
  - `pytest`
- synchronized `docs/plan/20260327_163915_04_v3_2_update_plan.md` with the validated implementation state

Validation result:

- compile validation passed for the updated preprocess, dataset, and target-FPS test surfaces
- target-FPS dataset loader smoke passed for `mode1`
  - session-store-backed frame/event loading worked without `frame_path/events_npz`
- target-FPS dataset loader smoke passed for `mode2`
  - interpolation metadata survived through the session-store manifest path
- build-stage failure index smoke passed
  - `81` pending rows exported into `indexes/annotation_failures.jsonl`
- canonical + manifest failure/skip propagation smoke passed
  - `81` failure rows preserved into canonical and manifest outputs
  - skipped session reason preserved as `event_file_missing`
- H5 end-to-end smoke passed
  - `session.h5 -> canonical -> manifest -> dataset`
- focused pytest suite passed:
  - `18 passed in 1.44s`

Current conclusion:

- the integrated target-FPS path now closes end-to-end for:
  - build
  - annotation fusion
  - session-store writing
  - canonical bridge
  - manifest bridge
  - dataset loading
- skip reasons and frame-level annotation failures are now queryable artifacts instead of implicit side effects
- the main remaining work has shifted beyond the base target-FPS bridge and toward real-dataset runs and higher-level integration passes

## 37. Stage2 Pair-Local `source_pair_average + alpha` Request

Request summary:

- user pointed out that the `mode2_stage1.yaml` synthetic-event options do not materially matter for the Stage1 objective
- user asked to make `Stage2` support:
  - `pair-local + source_pair_average + alpha`
- user explicitly wanted the Stage2 path to operate on events local to the source frame pair rather than the old anchor-window approximation

Decision:

- do not globally redefine `source_pair_average`, because that would silently break older experiments and invalidate the existing tests
- instead, add a new scope knob:
  - `source_pair_scope: anchor_window | pair_local`
- keep `anchor_window` as the default for backward compatibility
- move the recommended presets to:
  - `mode2_stage1`
    - `generation_strategy: raw_window`
  - `mode2_stage2`
    - `generation_strategy: source_pair_average`
    - `source_pair_scope: pair_local`
    - `average_weighting: alpha`
- for the new pair-local path, interpret the synthetic sample timestamp `t_syn` inside the source pair `[t0, t1]` as:
  - left interval: `[t0, t_syn]`
  - right interval: `(t_syn, t1]`
  - mixed event tensor: `(1 - alpha) * left + alpha * right`
- make `selected_event_count` report the full pair-local interval count instead of a weighted pseudo-count

Implementation result:

- updated `src/hbtxr/data/dataset.py` to:
  - add `source_pair_scope`
  - preserve legacy `source_pair_average(anchor_window)` behavior
  - add pair-local interval slicing and alpha mixing
  - export `event_generation_source_pair_scope` in metadata
- updated `configs/base.yaml` with:
  - `source_pair_scope: anchor_window`
- updated `configs/mode2_stage1.yaml` to:
  - `generation_strategy: raw_window`
- updated `configs/mode2_stage2.yaml` to:
  - `generation_strategy: source_pair_average`
  - `source_pair_scope: pair_local`
  - `average_weighting: alpha`
- updated tests:
  - `tests/test_dataset_v3.py`
  - `tests/test_config_v3.py`

Validation result:

- focused dataset regression passed:
  - `8 passed`
- focused config regression passed:
  - `4 passed`
- full repository regression passed:
  - `88 passed`

Current conclusion:

- `Stage1` no longer carries a misleading recommended preset for source-pair synthetic-event mixing
- `Stage2` now has an explicit pair-local synthetic-event path without breaking older anchor-window experiments
- the next optional follow-up is documentation and longer real-dataset comparison between:
  - `raw_window`
  - `source_pair_average(anchor_window)`
  - `source_pair_average(pair_local)`

## 38. Lazy `mode2-stage2` 2000FPS Planning Request

Request summary:

- user observed that projected frame counts at higher target FPS make full materialization unrealistic
- user asked whether `2000 FPS` generation is feasible
- after the feasibility discussion, user asked for a concrete plan
- user explicitly requested documentation only for the next step, not code implementation

Decision:

- do not overload the existing `v3.2` document with another implementation-scale topic
- create a new focused planning surface for:
  - `mode2-stage2`
  - lazy execution
  - `target_fps=2000`
- keep the planning scope centered on:
  - compact schedule metadata
  - compact canonical / manifest rows
  - on-the-fly frame generation
  - pair-local event generation
- treat this as a `v3.3` planning pass rather than an immediate code change

Implementation result:

- added:
  - `docs/plan/20260328_015046_05_v3_3_mode2_stage2_lazy_2000fps_plan.md`
- updated:
  - `docs/plan/00_index.md`

Documented content:

- why `2000 FPS` full materialization is impractical
- why `Stage2` is the correct target for the lazy path
- the recommended lazy data contract
- lazy frame generation rules
- lazy event generation rules
- file-by-file implementation plan
- migration phases
- risk items
- acceptance criteria

Validation result:

- document-only pass
- no code execution performed in this step

Current conclusion:

- the repository now has a dedicated planning surface for scalable `mode2-stage2` at `2000 FPS`
- the next concrete implementation step is to add:
  - `data.mode2.execution: lazy_target_fps`
  - compact schedule metadata in the target-FPS path
  - dataset-side on-the-fly frame/event synthesis

## 39. Head / Mode Data Documentation Request

Request summary:

- user asked to turn the recent head / output / loss / stage explanation into a persistent project document
- user then asked to do the same for the data side:
  - dataset build
  - canonical
  - manifest
  - dataset / dataloader
  - mode-specific contracts
- user additionally asked that the data document include:
  - generated artifacts by mode/path
  - representative raw `JSON` / `JSONL` examples
- final request for this pass:
  - synchronize `update history`
  - synchronize `progress checklist`
  - synchronize `conversation log`
  - commit, but do not push

Decision:

- keep the new technical docs under `docs/hbtxr/` because they describe active repository contracts rather than planning or external references
- split the documentation into two focused surfaces instead of one oversized document:
  - head / output / loss / stage coverage
  - mode-specific data-pipeline and artifact coverage
- include abbreviated raw `JSON` examples using real field names from the current codebase, but keep them compact enough to be used as reference rather than dumps
- treat the final synchronization step as documentation-only and avoid rerunning runtime validation

Implementation result:

- added:
  - `docs/prj/20260330_170419_08_v3_heads_outputs_and_stage_losses.md`
  - `docs/prj/20260330_170419_09_v3_mode_data_pipeline_contracts.md`
- updated:
  - `docs/hbtxr/00_index.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`
- documented in the new head/loss surface:
  - active head inventory
  - output ABI
  - derived state outputs
  - supervised loss mapping
  - Stage1 / Stage2 training matrix
- documented in the new mode data surface:
  - `mode1`
  - `mode2` materialized
  - `mode2` lazy target-FPS
  - generated artifact tables
  - representative raw `JSON` / `JSONL` structures

Validation result:

- document-only pass
- no additional code execution performed in this step

Current conclusion:

- the repository now has dedicated active-reference documents for both:
  - model head / loss / stage structure
  - mode-specific data-pipeline / artifact structure
- the chat/history/progress tracking surfaces are synchronized to that documentation expansion

## 40. YAML Config Reference Documentation Request

Request summary:

- user asked for a concrete analysis of the YAML files
- user specifically wanted:
  - which options can be specified
  - what each option means
  - how those options are actually interpreted by the code
- user then asked to persist that explanation as project documentation
- final request for this pass:
  - synchronize `update history`
  - synchronize `progress checklist`
  - synchronize `conversation log`
  - commit and push

Decision:

- add a dedicated YAML reference document under `docs/hbtxr/` because the request is about active repository contracts, not planning
- structure the document around:
  - config inheritance and overrides
  - top-level section map
  - option reference by section
  - code-confirmed enum values only where they can be justified directly from the codebase
- keep the tone conservative where the full accepted value set is not strongly validated
- exclude unrelated `docs/plan` changes from the commit

Implementation result:

- added:
  - `docs/prj/20260330_170419_10_v3_yaml_config_reference.md`
- updated:
  - `docs/hbtxr/00_index.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`
- documented:
  - `base.yaml` + `extends` merge semantics
  - dotted CLI override behavior
  - active top-level config surfaces
  - `model`, `data`, `training`, `distillation`, `regularization_ssl`, `pruning`, `loss`, and `runtime` options
  - code-confirmed values for the main mode/data/scheduler enums
  - the intent of the active mode/stage presets

Validation result:

- document-only pass
- no additional code execution performed in this step

Current conclusion:

- the repository now has a dedicated active-reference document for the YAML configuration system
- active config analysis is no longer scattered across ad hoc explanations
- commit scope is intentionally limited to:
  - `docs/hbtxr`
  - `docs/chat`
  while leaving unrelated `docs/plan` work untouched

## 41. Unrelated `docs/plan` Cleanup Request

Request summary:

- after the YAML reference documentation was committed and pushed, the remaining dirty worktree contained only:
  - `docs/plan/00_index.md`
  - `docs/plan/20260328_190651_06_v3_4_temporal_sampling_roi_experiments_plan.md`
- user asked to clean up those unrelated `docs/plan` changes

Decision:

- treat the remaining plan files as a legitimate standalone documentation change rather than leaving them as untracked residue
- keep the scope limited to:
  - `docs/plan`
  - matching `docs/chat` synchronization
- do not mix this pass with code or runtime changes

Implementation result:

- added:
  - `docs/plan/20260328_190651_06_v3_4_temporal_sampling_roi_experiments_plan.md`
- updated:
  - `docs/plan/00_index.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`
- documented:
  - temporal sampling ablation planning
  - frame-event synchronization experiment policy
  - annotation-space policy split
  - voxel scaling / normalization plan
  - ROI / pupil detection-head experiment tracks

Validation result:

- document-only planning cleanup pass
- no additional code execution performed in this step

Current conclusion:

- the remaining unrelated `docs/plan` changes are now treated as a proper tracked planning update
- worktree cleanup can now be completed with a standalone documentation commit

## 42. Mode0 Raw-CSV Supervision / YAML Inline Comments Request

Request summary:

- user asked to create a new `mode0`
- `mode0` should use:
  - raw CSV pupil state
  - pupil mask generated from raw annotations
  - the existing heuristic eye-region ROI bbox as label
- after that, user asked to annotate each YAML with usable options as inline comments
- final request for this pass:
  - synchronize `update history`
  - synchronize `progress checklist`
  - synchronize `conversation log`
  - synchronize `plan`
  - commit and push

Decision:

- treat `mode0` as a first-class active data mode rather than a one-off script flag
- keep `mode0` strictly tied to `manual_csv` supervision so it does not ambiguously overlap with `mode1` Grounded-SAM flows
- document YAML options directly in the config files instead of adding another detached doc, because the user explicitly asked for comments inside each yaml
- keep `docs/jetcas/` untouched in this implementation pass

Implementation result:

- added:
  - `configs/mode0_stage1.yaml`
  - `configs/mode0_stage2.yaml`
- updated:
  - `src/hbtxr/preprocess/canonicalize.py`
  - `src/hbtxr/preprocess/build_manifests.py`
  - `src/hbtxr/data/dataset.py`
  - `src/hbtxr/data/loader.py`
  - `scripts/_config.py`
  - `scripts/prepare_ev_eye.py`
  - mode-aware CLI entry scripts
  - `scripts/hbtxr_mode_pipeline.sh`
  - `scripts/run_prepare_and_train.sh`
  - `README.md`
  - `configs/base.yaml`
  - active preset YAML files with inline option comments
  - `tests/conftest.py`
  - `tests/test_config_v3.py`
  - `tests/test_preprocess_v3.py`
  - `tests/test_dataset_v3.py`
- implemented:
  - `mode0 -> canonical0 -> manifest0`
  - raw-CSV-only supervision contract
  - pupil-mask rasterization from raw ellipse
  - heuristic eye ROI labeling
  - mode-aware preset and shell workflow support

Validation result:

- focused regression passed:
  - `tests/test_config_v3.py`
  - `tests/test_preprocess_v3.py`
  - `tests/test_dataset_v3.py`
  - result: `23 passed`
- full repository regression passed:
  - `pytest -q`
  - result: `95 passed`

Current conclusion:

- `mode0` is now an active raw-supervision path rather than a planning-only idea
- the YAML surface is now self-describing enough for direct preset editing without cross-reading the code first
- remaining non-synthetic validation gaps are still:
  - real dataset / real checkpoint runtime validation
  - real dataset `annotate -> canonicalize -> build_manifests` rerun

## 43. Mode0 Head-Composition Clarification

Request summary:

- user asked whether the head composition had also been changed to match `mode0`

Decision:

- answer conservatively using the implemented config and model-construction path
- distinguish between:
  - `new mode0-specific head architecture`
  - `stage-specific gating of the existing head family`

Confirmed implementation:

- `mode0_stage1`
  - disables `event_head`
  - keeps `mask_head`
- `mode0_stage2`
  - enables `event_head`
  - disables `mask_head`
- the tracker still uses the same common head family as the other modes
- the implementation therefore changes head activation policy, not head taxonomy

Code basis cited in the answer:

- `configs/mode0_stage1.yaml`
- `configs/mode0_stage2.yaml`
- `src/hbtxr/training/trainer.py`
- `src/hbtxr/models/hybrid_tracker.py`

Current conclusion:

- yes, `mode0` is aligned to the stage-specific head policy
- no, there is still no separate `mode0-only` head architecture

## 44. Optimizer Pool / Public-Code-First Integration Request

Request summary:

- user asked to implement the previously locked optimizer-pool plan
- the locked structure was:
  - flat files under `src/hbtxr/optim/<name>.py`
  - public-code-first intake
  - upstream diff notes in `docs/hbtxr/optimizers/<name>_diff.md`
- user explicitly rejected:
  - nested per-optimizer directories like `src/hbtxr/optim/<name>/vendor.py`
- user also required:
  - `MuSGD` support
  - optimizer registry / modifiers / pool orchestration
  - actual integration into trainer/config/script paths

Decision:

- implement the optimizer system as a flat-file package:
  - `src/hbtxr/optim/__init__.py`
  - `src/hbtxr/optim/registry.py`
  - `src/hbtxr/optim/modifiers.py`
  - `src/hbtxr/optim/pool.py`
  - `src/hbtxr/optim/common.py`
  - `src/hbtxr/optim/<name>.py`
- keep public-code-first semantics where practical
- use the local Ultralytics reference snapshot as the source of truth for `MuSGD`
- implement a production-ready first wave rather than faking full coverage:
  - `adamw`
  - `lion`
  - `prodigy`
  - `adopt`
  - `musgd`
- expose the remaining planned optimizer names as explicit placeholders so the registry surface matches the plan without silently implying support

Implementation result:

- added:
  - `src/hbtxr/optim/`
  - `tests/test_optimizer_pool_v3.py`
  - `docs/hbtxr/optimizers/*.md`
- updated:
  - `src/hbtxr/training/trainer.py`
  - `scripts/train_hbtxr.py`
  - `configs/base.yaml`
- implemented:
  - config-driven optimizer construction
  - `cautious` wrapper
  - `schedule_free` path for `adamw`
  - optimizer hypers reporting
  - pool sweep report writing
  - `MuSGD` auto param-group split:
    - `2D+ -> use_muon=True`
    - `1D/scalar/bias/norm -> use_muon=False`

Validation result:

- focused optimizer-pool regression passed:
  - `11 passed`
- focused train/config regression passed:
  - `7 passed`
- full repository regression passed:
  - `106 passed`

Current conclusion:

- the repository now has a real optimizer-pool execution surface rather than only a planning document
- the first supported benchmarkable optimizer set is:
  - `adamw`
  - `lion`
  - `prodigy`
  - `adopt`
  - `musgd`
- the rest of the planned optimizer names remain deliberately visible as placeholders pending later vendoring/validation

## 45. Optimizer Pool Documentation Follow-up

Request summary:

- after the optimizer pool implementation was committed and pushed, the remaining next-step items were:
  - README optimizer-pool usage guide
  - dedicated active-reference document for optimizer-pool semantics
- user asked to proceed with the next task without redefining the technical scope

Decision:

- treat the next task as a documentation follow-up rather than another code change
- add one active-reference document under `docs/hbtxr`
- add concise but directly runnable README examples
- sync the chat tracking documents so the implementation and documentation surfaces stay aligned

Implementation result:

- added:
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`
- updated:
  - `README.md`
  - `docs/hbtxr/00_index.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Documented topics:

- `src/hbtxr/optim` package layout
- implemented optimizer set vs placeholder registry surface
- optimizer config priority rules
- `MuSGD` policy semantics
- schedule-free semantics
- pool sweep outputs
- README usage examples for:
  - `lion`
  - `musgd`
  - `schedule_free`
  - pool sweep

Current conclusion:

- optimizer-pool implementation now has both:
  - executable code surface
  - active-reference documentation surface

## 46. Trainer Console Logging And CUDA Fail-Fast Follow-up

Request summary:

- user observed that training progress was written to files but was effectively invisible in the terminal
- user also hit ambiguous multi-GPU device syntax and a silent CPU-fallback style failure mode after CUDA initialization warnings
- the requested follow-up was to implement:
  - real-time console progress
  - stricter CUDA device handling
  - documentation / checklist synchronization

Decision:

- keep the existing file-log contract unchanged
- add console logging as an additional channel rather than replacing file logs
- treat `tqdm` progress as the preferred console style
- keep a plain-text fallback when `tqdm` is not installed
- remove silent CPU fallback for requested CUDA devices and fail immediately with an explicit error
- accept multi-GPU aliases that users naturally type:
  - `cuda:0,cuda:1`
  - `cuda:0,1`
  - `0,1`
  - `multi-gpu`

Implementation result:

- updated:
  - `src/hbtxr/training/trainer.py`
  - `configs/base.yaml`
  - `README.md`
  - `requirements.txt`
  - `pyproject.toml`
  - `tests/test_config_v3.py`
- added:
  - `tests/test_trainer_console_v3.py`

Implemented behavior:

- trainer now emits:
  - run-start summary
  - epoch progress for train / val
  - epoch summaries
  - checkpoint / pretrained / resume / export / early-stop events
- console progress is controlled by:
  - `training.console_log.enabled`
  - `training.console_log.style`
  - `training.console_log.step_interval`
  - `training.console_log.show_lr`
  - `training.console_log.show_best_metric`
  - `training.console_log.show_width`
- CUDA requests now fail fast with an explicit message that includes:
  - requested device string
  - parsed device ids
  - CUDA availability
  - CUDA device count
  - example valid syntax

Validation result:

- focused trainer/config regression passed:
  - `10 passed`
- full repository regression passed:
  - `115 passed`
- deprecation-focused full repository regression passed:
  - `115 passed`

Current conclusion:

- training progress is now visible during execution instead of only after file writes complete
- multi-GPU device parsing is less ambiguous
- CUDA failure modes are now explicit instead of silently turning into CPU training

## 46. Optimizer Pool Expansion Follow-up

Request summary:

- user asked to:
  - update `README`
  - update history / progress
  - commit and push
  - continue with the next optimizer batch chosen from:
    - `MARS`
    - `Sophia-G`
    - `IVON`
    - `SOAP`

Decision:

- choose `SOAP` as the next optimizer batch item because it has an official single-file optimizer implementation and fits the current trainer contract without new callback surfaces
- fold the already-completed `adam_mini` / `adema_mix` local implementation work into the same documentation / release pass
- keep:
  - `sophia_g`
  - `ivon`
  - `mars`
  as the remaining placeholder set after this pass

Implementation result:

- updated:
  - `src/hbtxr/optim/adam_mini.py`
  - `src/hbtxr/optim/adema_mix.py`
  - `src/hbtxr/optim/soap.py`
  - `src/hbtxr/optim/registry.py`
  - `tests/test_optimizer_pool_v3.py`
  - `README.md`
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`
  - `docs/hbtxr/optimizers/adam_mini_diff.md`
  - `docs/hbtxr/optimizers/adema_mix_diff.md`
  - `docs/hbtxr/optimizers/soap_diff.md`
- promoted to implemented:
  - `adam_mini`
  - `adema_mix`
  - `soap`

Validation result:

- focused optimizer regression passed:
  - `17 passed`
- full repository regression passed:
  - `112 passed`

Current conclusion:

- the optimizer pool is now materially broader while still staying inside the current trainer/config contract
- the most likely next heavy-lift optimizer items are now:
  - `mars`
  - `sophia_g`
  - `ivon`
  because they are the remaining registered placeholders

## 47. Optimizer Pool Completion Follow-up

Request summary:

- user asked to continue the interrupted optimizer expansion work and explicitly requested:
  - `README / history / progress` synchronization
  - implementation of the next remaining placeholders:
    - `MARS`
    - `Sophia-G`
- during continuation, `IVON` had already been implemented locally but had not yet been documented or committed in the same progress thread

Decision:

- fold `IVON` into the same follow-up because it was already validated but not yet synchronized into the documentation trail
- implement `Sophia-G` from the official optimizer core and add the minimal trainer hessian-refresh hook required by HBTXR
- implement `MARS` from the official approximate optimizer path and keep exact multi-pass MARS explicitly unsupported for now
- finish the optimizer-pool surface by eliminating the last registered placeholders instead of leaving a mixed implemented/placeholder registry

Implementation result:

- updated:
  - `src/hbtxr/optim/ivon.py`
  - `src/hbtxr/optim/sophia_g.py`
  - `src/hbtxr/optim/mars.py`
  - `src/hbtxr/optim/registry.py`
  - `src/hbtxr/training/trainer.py`
  - `tests/test_optimizer_pool_v3.py`
  - `README.md`
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`
  - `docs/hbtxr/optimizers/ivon_diff.md`
  - `docs/hbtxr/optimizers/sophia_g_diff.md`
  - `docs/hbtxr/optimizers/mars_diff.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Implemented behavior:

- `IVON`
  - trainer now supports sampled-parameter optimizers through `optimizer.sampled_params(train=True)`
  - `ess` is auto-filled from the training dataset size when omitted
- `Sophia-G`
  - trainer now supports periodic hessian refresh through `optimizer.update_hessian()`
  - refresh cadence is controlled by:
    - `training.optimizer.kwargs.hess_interval`
  - batch-scale `bs` for `step(bs=...)` is inferred from the effective sample batch size
- `MARS`
  - official approximate optimizer path is now wired
  - `mars_type`, `gamma`, `optimize_1d`, and 1D fallback hypers are exposed through `training.optimizer.kwargs`
  - exact `is_approx=false` remains intentionally blocked

Validation result:

- focused optimizer regression passed:
  - `24 passed`
- focused train-pipeline regression passed:
  - `3 passed`
- full repository regression passed:
  - `122 passed`

Current conclusion:

- the optimizer pool no longer has any registered placeholders
- the remaining follow-up items are no longer placeholder implementation work, but rather:
  - broader benchmark coverage
  - `mode2` optimizer-pool expansion
  - optional support for exact MARS and AMP-compatible Sophia-G hessian refresh

## 48. Exact MARS Multi-Pass Follow-up

Request summary:

- user asked to continue with the next optimizer follow-up item after the placeholder implementations were finished
- the remaining concrete functional gap was:
  - exact MARS multi-pass trainer support

Decision:

- implement exact MARS before any broader benchmark expansion
- follow the official MARS ordering exactly at the trainer level:
  - future-group gradient pass
  - `update_previous_grad()`
  - current-group gradient pass
  - optimizer step
  - `update_last_grad()`
- adapt the upstream infinite-batch loop to HBTXR's finite manifest loader
- explicitly harden the optimizer state path for stage-gated parameters that may not receive gradients in every pass

Implementation result:

- updated:
  - `src/hbtxr/optim/mars.py`
  - `src/hbtxr/optim/registry.py`
  - `src/hbtxr/training/trainer.py`
  - `tests/test_optimizer_pool_v3.py`
  - `README.md`
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`
  - `docs/hbtxr/optimizers/mars_diff.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Implemented behavior:

- `training.optimizer.name=mars` with:
  - `training.optimizer.kwargs.is_approx=false`
  now activates the exact multi-pass path
- trainer now prefetches a future batch-group before each current step
- `MARS.update_previous_grad()` now zero-fills inactive parameters instead of leaving missing state behind
- `MARS.update_last_grad()` now tolerates the absence of `previous_grad`
- `mars` metadata now reports `algorithmic_diff=true` because the missing-gradient handling is an HBTXR-specific integration behavior

Validation result:

- focused optimizer regression passed:
  - `27 passed`
- focused train-pipeline regression passed:
  - `3 passed`
- full repository regression passed:
  - `125 passed`

Current conclusion:

- exact MARS is no longer a planned follow-up item; it is now an implemented optimizer path
- the remaining optimizer-specific follow-up items are narrower:
  - dedicated CUDA exact-MARS benchmark runs
  - optional AMP-focused validation for exact MARS
  - optional AMP-compatible Sophia-G hessian-refresh support

## 49. Sophia-G AMP Hessian-Refresh Follow-up

Request summary:

- continue with the next remaining optimizer follow-up item after exact MARS
- the most concrete remaining implementation gap was:
  - AMP-compatible `Sophia-G` Hessian refresh

Decision:

- implement AMP compatibility before any new benchmark sweep
- keep the existing HBTXR trainer contract unchanged except for the refresh helper:
  - the refresh pass still uses supervised `loss_total`
  - only the `GradScaler` interaction is relaxed
- validate the helper path without requiring CUDA by using a CPU `GradScaler` smoke test

Implementation result:

- updated:
  - `src/hbtxr/training/trainer.py`
  - `tests/test_optimizer_pool_v3.py`
  - `README.md`
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`
  - `docs/hbtxr/optimizers/sophia_g_diff.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Implemented behavior:

- `_maybe_run_hessian_refresh(...)` now:
  - scales the refresh loss when a scaler is active
  - manually unscales gradients using the current scaler scale
  - then calls `optimizer.update_hessian()`
- the previous hard failure:
  - `Hessian-refresh optimizers are not yet supported with AMP/GradScaler in HBTXR.`
  has been removed

Validation result:

- focused optimizer regression passed:
  - `28 passed`
- full repository regression passed:
  - `126 passed`

Current conclusion:

- `Sophia-G` no longer has an explicit AMP blocker at the helper level
- the remaining work is now primarily empirical:
  - CUDA-backed end-to-end benchmark validation
  - broader optimizer-pool benchmark expansion into `mode2`

## 50. Mode2 Benchmark Surface And Stage-Aware Stats Follow-up

Request summary:

- continue with the next follow-up item after the Sophia-G AMP helper work
- user also reported that the terminal still showed losses / metrics that were not actually relevant to the active stage

Decision:

- address both items together:
  - expose `mode2` optimizer-pool benchmark entry presets
  - clean stage-irrelevant stat reporting at the trainer aggregation layer
- keep the filtering at the reporting/history boundary instead of changing the supervised objective functions
- also harden stage1 presets by disabling `track_head` directly

Implementation result:

- updated:
  - `src/hbtxr/training/trainer.py`
  - `configs/mode0_stage1.yaml`
  - `configs/mode1_stage1.yaml`
  - `configs/mode2_stage1.yaml`
  - `tests/test_config_v3.py`
  - `tests/test_trainer_console_v3.py`
  - `README.md`
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`
- added:
  - `configs/mode2_stage1_optimizer_pool.yaml`
  - `configs/mode2_stage2_optimizer_pool.yaml`

Implemented behavior:

- Stage1 reporting now drops:
  - `metric_event_*`
  - `metric_track_*`
  - `loss_event_*`
  - `loss_track_*`
  - `loss_consistency`
- Stage2 reporting now drops:
  - `loss_eye`
  - `loss_mask`
- the filtering applies consistently to:
  - terminal summaries
  - `history.json`
  - `history.jsonl`
  - `train_log.txt`
- `mode2` pool sweep can now be launched directly from dedicated presets instead of only by manual override

Validation result:

- focused config/console regression passed:
  - `8 passed`
- focused train-pipeline regression passed:
  - `3 passed`
- full repository regression passed:
  - `126 passed`

Current conclusion:

- the stage-specific terminal/history noise issue is resolved at the reporting layer
- `mode2` optimizer-pool experimentation is now exposed through first-class preset files
- the remaining follow-up is primarily empirical execution:
  - actually running the `mode2` pool sweeps

## 51. Mode2 Optimizer-Pool Execution Follow-up

Request summary:

- continue with the next remaining optimizer-pool follow-up item
- the most concrete remaining gap after the preset work was:
  - actually executing the `mode2` pool sweeps instead of only exposing config presets

Decision:

- validate the mode2 pool surface through synthetic smoke execution first
- use the real `train_hbtxr.py` script entrypoint rather than only direct helper calls
- keep the scope lightweight and deterministic by generating a temporary target-FPS canonical workspace inside `pytest`
- preserve the normal stage2 contract:
  - `mode2_stage2_optimizer_pool` must still bootstrap from a stage1 checkpoint

Implementation result:

- updated:
  - `tests/test_optimizer_pool_v3.py`
  - `README.md`
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`

Implemented behavior:

- `test_optimizer_pool_v3.py` now:
  - builds a synthetic target-FPS canonical workspace
  - writes temporary mode2 pool configs
  - invokes the real `train_hbtxr.py` main entrypoint
  - verifies `optimizer_pool_report.json`
  - verifies candidate labels for the emitted pool run
- stage2 mode2 pool smoke first creates a stage1 checkpoint and then runs the stage2 pool sweep against it

Validation result:

- focused optimizer regression passed:
  - `30 passed`
- focused config / console / train-pipeline regression passed:
  - `11 passed`
- full repository regression passed:
  - `129 passed`

Current conclusion:

- the mode2 optimizer-pool surface is now validated at the script-execution layer for synthetic target-FPS data
- the remaining follow-up is no longer basic plumbing; it is empirical benchmark collection on larger mode2 runs

## 52. Mode2 Optimizer-Pool CPU Pilot Benchmark Follow-up

Request summary:

- continue with the next concrete follow-up after the mode2 script-level smoke validation
- the next actionable gap was:
  - collect an actual mode2 optimizer-pool benchmark instead of only smoke coverage

Decision:

- run a lightweight but real pilot benchmark before any CUDA-specific attempt
- since CUDA is unavailable in the current environment, explicitly scope this pass as:
  - synthetic target-FPS
  - CPU only
- keep the candidate set equal to the full currently implemented optimizer set
- use a normal stage1 bootstrap checkpoint before the stage2 pool run

Implementation result:

- executed:
  - stage1 bootstrap
  - mode2 stage1 pool
  - mode2 stage2 pool
- added:
  - `docs/exps/20260330_170419_12_v3_mode2_optimizer_pool_cpu_pilot.md`
- updated:
  - `README.md`
  - `docs/hbtxr/00_index.md`
  - `docs/prj/20260330_170419_11_v3_optimizer_pool_reference.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Observed result:

- all implemented optimizers completed in both stage1 and stage2 pilot runs
- stage1 main search metric tied at `0.0`, so `val.loss_total` became the most useful secondary discriminator
- stage2 main track percentage metric tied at `98.7654`, so `metric_track_center_px` and `val.loss_total` became the useful secondary signals
- the pilot therefore serves as:
  - real execution confirmation
  - early relative-behavior inspection
  and not as a final benchmark recommendation

Current conclusion:

- `mode2` optimizer-pool execution is now validated beyond smoke level
- the remaining empirical gap is specifically:
  - larger-budget synthetic reruns
  - real EV-Eye reruns
  - CUDA runtime / throughput collection

## 54. Real EV-Eye Mode2 Pilot / CUDA Benchmark Request

Request summary:

- continue with the next empirical benchmark step after the synthetic CPU pilot
- specifically:
  - run a real EV-Eye based `mode2` benchmark
  - collect CUDA throughput / memory results if possible
  - clean `.tmp_mode2_pool_pilot/`

Decision:

- first remove the previous synthetic pilot temp directory
- then run a bounded real-data pilot instead of waiting for a full-dataset build
- use a temporary bridge because the current target-FPS builder requires `events.npz`, while raw EV-Eye sessions expose `events.txt`
- explicitly separate:
  - what can be benchmarked now on CPU
  - what remains blocked for CUDA

Implementation result:

- removed:
  - `.tmp_mode2_pool_pilot/`
- created a temporary bounded real-data bridge for:
  - `user01/left/session_102`
  - `user01/right/session_102`
- bridged:
  - raw frames / timestamps / CSV
  - canonical `events.npz`
  - bbox-only ROI rows derived from canonical metadata
- built:
  - target-FPS `100`
  - `canonical2`
  - bounded real-data pilot manifests
- executed:
  - real-data `mode2 stage1` bootstrap
  - real-data `mode2 stage1` optimizer-pool sweep
  - real-data `mode2 stage2` optimizer-pool sweep
- added:
  - `docs/exps/20260330_170419_13_v3_real_ev_eye_mode2_optimizer_pool_cpu_pilot.md`

Observed result:

- stage1:
  - all implemented optimizers completed
  - primary search metric tied at `0.0`
  - `adam_mini` had the lowest validation loss in this bounded pilot
- stage2:
  - validation remained finite for `10/11` optimizers
  - `soap` collapsed in this bounded pilot
  - all candidates showed `NaN` train loss in the one-epoch CPU run
  - this makes the stage2 result a stability screen, not a trustworthy ranking
- CUDA:
  - GPU hardware is present
  - current project `.venv` is CPU-only for PyTorch
  - real CUDA throughput / memory collection could not be executed in this environment

Current conclusion:

- bounded real EV-Eye `mode2` execution is now validated
- the remaining blocker is no longer data-path plumbing
- the next hard requirement is a CUDA-enabled PyTorch environment for true GPU throughput / memory benchmarking

## 55. CUDA Environment / Loss Cluster / Import Dependency Request

Request summary:

- connect a real CUDA-enabled PyTorch environment
- split the loss implementation by cluster into flat files under `src/hbtxr/`
- analyze project-wide path/import dependencies and fix issues if found

Decision:

- keep the existing Windows `.venv` unchanged because it is still the stable CPU regression environment
- create a separate WSL CUDA venv because the current Windows `.venv` is `Python 3.14 + torch+cpu`, which is not the right CUDA install surface
- refactor losses by moving implementation into:
  - `loss_common`
  - `loss_primitives`
  - `loss_stage`
  - `loss_distillation`
  while preserving `hbtxr.training.losses` as a compatibility layer
- validate import/path dependencies with a static graph test rather than ad-hoc manual checks only

Implementation result:

- created:
  - `.venv_wsl_cuda`
- installed:
  - `torch 2.11.0+cu128`
  - `torchvision 0.26.0+cu128`
- verified in WSL:
  - `cuda_available=True`
  - `device_count=1`
  - `NVIDIA GeForce RTX 4070 Ti`
- added:
  - `src/hbtxr/loss_common.py`
  - `src/hbtxr/loss_primitives.py`
  - `src/hbtxr/loss_stage.py`
  - `src/hbtxr/loss_distillation.py`
  - `tests/test_import_graph_v3.py`
- converted:
  - `src/hbtxr/training/losses.py`
  into a re-export facade
- updated:
  - `.gitignore`
  - `README.md`
  - `src/hbtxr/__init__.py`

Observed result:

- the repo now has a verified CUDA-capable execution surface through WSL
- the existing Windows regression flow remains intact
- loss code is separated by concern without changing the public trainer import path
- static analysis found:
  - no unresolved `hbtxr.*` absolute imports
  - no intra-package static import cycle

Validation:

- `pytest tests -q` -> `132 passed`
- `python -m compileall src/hbtxr` -> passed

Current conclusion:

- CUDA environment connection is no longer blocked
- the next GPU benchmark work can use `.venv_wsl_cuda`
- loss codebase maintainability is improved without breaking the current training surface

## 56. YOLO26 Head / BBox Integration Planning Request

Request summary:

- analyze whether `YOLO26` bbox detection ideas should be applied to:
  - eye region ROI bbox
  - pupil bbox
- then convert the analysis into an update plan

Decision:

- separate the problem into:
  - current `HBTXR` head architecture
  - `Ultralytics YOLO26` head / loss mechanisms
  - adoptable mechanisms vs non-adoptable mechanisms
- treat this as a focused planning item instead of expanding the broader `v3.4` experiment document
- create a new `v3.5` plan dedicated to:
  - `Eye ROI` bbox loss upgrade
  - optional `Pupil BBox / OBB auxiliary branch`

Analysis result:

- `Eye ROI` is the cleanest insertion point for `YOLO26`-style bbox regression / loss
- `Pupil` should not be converted to bbox-only supervision immediately
- the recommended path is:
  - keep ellipse / state as the primary target
  - add bbox or OBB as an auxiliary branch later
- `track_head` should remain unchanged in this wave
- the following `Ultralytics` mechanisms are explicitly out of scope for now:
  - full `Detect` head replacement
  - `TaskAlignedAssigner`
  - `STAL`
  - `YOLOE`

Implementation result:

- added:
  - `docs/plan/20260329_023122_07_v3_5_yolo26_head_bbox_integration_plan.md`
- updated:
  - `docs/plan/00_index.md`

Current conclusion:

- there is now a focused `v3.5` active plan for head-level `YOLO26` integration
- the project now separates:
  - broad ROI / temporal experiments in `v3.4`
  - concrete `YOLO26` head / loss adoption work in `v3.5`

## 57. Selective Unmerged-Branch Integration Decision

Request summary:

- implement the previously approved selective-integration plan
- do not integrate:
  - `codex/ipex`
  - `codex/startpoint`
- do not merge branches wholesale
- only import additive code, docs, scripts, and experiments from:
  - `codex-exp-deepmicro`
  - `codex/home-wsl`
  - `exp_wsl`
  - `exp-wsl`
  - `codex-exp-ubeeslab`

Decision:

- keep the integration strictly wave-based and commit each wave independently
- treat `codex-exp-deepmicro` as the primary code source for:
  - Grounded-SAM build
  - target-FPS build
  - target-FPS canonical follow-ups
- treat `codex/home-wsl` as a review / handoff / resource source only
- treat `exp_wsl` and `exp-wsl` as documentation / preview-tooling / helper-script sources only
- treat `codex-exp-ubeeslab` as a semantic-patch source for:
  - mask-centroid search experimentation
  - `mode0` experimental presets
  - custom best-checkpoint handling
- preserve all current-branch newer systems:
  - optimizer pool
  - loss clusters
  - WSL CUDA guidance
  - lazy `mode2`
  - stage-aware logging

Implementation result:

- Wave 1:
  - imported deepmicro preprocess patches and tests
  - commit `51a26af`
- Wave 2:
  - imported Grounded-SAM / `all48` review docs, resources, and preview scripts
  - commit `633a9f3`
- Wave 3:
  - imported heuristic ROI docs, plans, preview utilities, and mode1 sweep helper
  - commit `ca65d4b`
- Wave 4:
  - integrated `model.search.xy_from_mask_centroid`
  - restored dynamic custom best checkpoints
  - imported `mode0` experimental presets and debug history doc
  - commit `203ce9e`

Current conclusion:

- the branch-selection policy was preserved exactly
- no excluded branch content was integrated
- the imported features remain additive and do not roll back the current `codex/home` architecture
- the remaining work after Wave 4 is only final documentation/index/chat synchronization plus one full regression run and push

## 58. Selective Integration Traceability Documentation Request

Request summary:

- after the selective integration implementation was finished, document exactly:
  - where each imported change came from
  - what file it landed in
  - what behavior was imported
  - how it was integrated

Decision:

- keep the existing update history / progress checklist summaries short
- add one dedicated technical trace document under `docs/hbtxr/`
- organize the new document by:
  - source branch
  - target file
  - imported behavior
  - integration method
  - wave / commit mapping

Implementation result:

- added:
  - `docs/others/update/20260330_170419_20_v3_selective_branch_integration_report.md`
- updated:
  - `docs/hbtxr/00_index.md`

Current conclusion:

- the branch-integration work is now documented at two levels:
  - summary level in `docs/chat/*`
  - traceability level in `docs/others/update/20260330_170419_20_v3_selective_branch_integration_report.md`

## 59. Documentation Restructure And Canonical Koreanization Continuation

Request summary:

- keep `docs/chat` and `docs/jetcas` unchanged
- reorganize the remaining documentation under:
  - `update`
  - `experiments`
  - `analysis`
  - `plan`
  - `progress`
- convert non-chat/non-jetcas canonical documents into Korean
- continue the work wave by wave instead of doing a risky one-shot move

Decision:

- keep old folders as compatibility layers instead of deleting them immediately
- establish new canonical folders first
- translate and rewrite canonical documents in place
- preserve filenames where possible and move the semantic “source of truth” to the new canonical folders

Implementation result:

- canonical folders now hold the preferred documentation paths:
  - `docs/analysis`
  - `docs/experiments`
  - `docs/update`
  - `docs/plan`
  - `docs/progress`
- compatibility folders were converted into redirect-style or stub-style legacy layers:
  - `docs/hbtxr`
  - `docs/references_analysis`
  - `docs/others`
- the following canonical documentation groups were rewritten into Korean:
  - analysis core docs
  - analysis optimizer diff docs
  - analysis reference-session docs
  - experiments docs
  - update docs
  - plan docs
  - progress active doc

Current conclusion:

- the canonical documentation structure is now stable
- old paths remain only for backward-link compatibility
- a dedicated completion report was added under:
  - `docs/others/update/20260330_170419_03_docs_restructure_and_korean_canonicalization_report.md`

## 60. YOLO / SOT / MGIoU / MPDIoU Integration And Dependency Audit

Request summary:

- integrate additional bbox/obb loss ideas from:
  - `YOLO`
  - `SOT`
  - `ldtho/MGIoU`
  - `arXiv:2307.07662`
- analyze dependency risks across the current codebase
- fix any discovered compatibility issue
- synchronize the active plan and tracking documents

Decision:

- treat `ldtho/MGIoU` and `arXiv:2307.07662` as different sources
  - `ldtho/MGIoU` corresponds to `MGIoU`
  - `2307.07662` is treated as `MPDIoU`
- integrate `MGIoU2D` only for rotated OBB auxiliary paths
- integrate `MPDIoU` for axis-aligned eye/bbox auxiliary paths
- rerun import-graph, `compileall`, targeted smoke, and full regression after the changes
- fix any stale compatibility facade that does not re-export the newly added losses

Implementation result:

- added `MGIoU2DLoss` and `MPDIoULoss`
- wired:
  - `loss.eye_box_mode: yolo26_mpdiou`
  - `loss.search_bbox_aux_mode: mpdiou`
  - `loss.event_bbox_aux_mode: mpdiou`
  - `loss.search_obb_aux_mode: mgiou2d`
  - `loss.event_obb_aux_mode: mgiou2d`
  - `loss.mgiou_fast_mode`
- fixed:
  - `src/hbtxr/training/losses.py` legacy export surface
- synchronized:
  - `docs/plan/20260329_023122_07_v3_5_yolo26_head_bbox_integration_plan.md`
  - `docs/chat/20260326_011438_01_v3_update_history.md`
  - `docs/chat/20260326_011438_02_v3_progress_checklist.md`
  - `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

Validation result:

- import graph regression passed
- `compileall` passed
- targeted config/model/train regressions passed
- full repository `pytest` passed
  - `150 passed`

Current conclusion:

- the recent YOLO/SOT head-loss expansion remains dependency-safe
- the active plan document now matches the actual implementation state

## 61. P0-P2 Detector / Assigner / ProtoMask Integration

Request summary:

- explain and then implement the early part of the planned:
  - detector-stack expansion
  - assigner-stack expansion
  - prototype-style mask decoder path
- cover:
  - `P0`
  - `P1`
  - `P2`
- avoid touching `docs/jetcas`

Decision:

- keep the existing baseline presets and ABI stable by default
- add the new stack as opt-in config variants rather than replacing the baseline immediately
- treat `P0` as ABI normalization, `P1` as detector/assigner integration, and `P2` as mask-decoder integration
- route the existing `yolo26_point` dense eye-loss path through the new assigner stack so the TAL/STAL work is not isolated to a brand-new variant only

Implementation result:

- added:
  - `src/hbtxr/loss/assigners.py`
  - `tests/test_eye_detector_stack_v3.py`
- updated:
  - eye head stack
  - stage eye-loss stack
  - model builder config surface
  - train smoke/model tests
  - README and progress tracking docs

Technical result:

- normalized eye detector ABI:
  - decoded eye box remains `search/eye`
  - dense detector logits now use `search/eye_det_*`
- implemented:
  - `TaskAlignedAssigner`
  - `STALTaskAlignedAssigner`
  - `yolo_detect` eye head
  - `proto26` mask decoder
- wired:
  - `yolo26_point` -> assigner-backed dense eye loss
  - `yolo_detect` -> assigner-backed dense eye loss + quality term

Validation result:

- focused regression passed:
  - `32 passed`
- full repository regression passed:
  - `156 passed`

Current conclusion:

- `P0-P2` is now implemented in code, config surface, and regression coverage
- later roadmap items still remain:
  - tracker actor replacement
  - candidate elimination tracker
  - YOLOE prompt-conditioned head

## 62. Review-Driven P1/P2 Follow-up Fixes

Request summary:

- keep `P3-P5` as backlog only
- review the already-implemented `P0-P2` work
- then fix the concrete gaps that were found
- finish with regression and keep `docs/jetcas` untouched

Review conclusion:

- `P1` was still too close to an inside-box-only heuristic because the assigner filtered candidates through `inside_mask` before top-k selection
- `yolo_detect` quality training was too close to class-target training because both branches consumed nearly the same normalized target
- `P2` `proto26` emitted prototype/coarse tensors but only the final mask logits participated in supervision

Implementation result:

- `TaskAlignedAssigner` now ranks aligned candidates globally and only uses inside-box / nearest-point logic as a recovery path
- `yolo_detect` now uses:
  - binary dense cls targets
  - IoU-based dense quality targets
- `proto26` now adds `loss_mask_coarse`
- `loss.mask_coarse_weight` added to config
- checklist backlog marker for:
  - tracker actor
  - candidate elimination
  - YOLOE prompt head

Validation result:

- focused regression passed:
  - `30 passed`
- full repository regression passed:
  - `157 passed`

Current conclusion:

- the main review findings for `P1/P2` have been closed
- `P3-P5` remain intentionally unimplemented
