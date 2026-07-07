# HBTXR_v3_0 전체 구조와 파이프라인 분석

최종 갱신: 2026-04-08 KST

Role: `프로젝트 아키텍트`, `파이프라인 분석가`, `운영 문서 작성자`

## 요약

- 현재 `HBTXR_v3_0`는 하나의 학습 스크립트 프로젝트가 아니라, `preprocess -> manifest -> dataset -> model -> loss -> trainer -> runtime`으로 분리된 계약 중심 파이프라인이다.
- 루트 표면은 `main surface`, `experimental surface`, `archive surface`로 분리되어 있다.
- 공식 운영 표면은 `scripts/`와 `configs/`이고, 실험용 스크립트와 preset은 `exps/`로 격리되어 있다.
- 핵심 허브 파일은 `scripts/_config.py`, `src/hbtxr/preprocess/*`, `src/hbtxr/data/*`, `src/hbtxr/models/hybrid_tracker.py`, `src/hbtxr/training/trainer.py`, `src/hbtxr/loss/stage1.py`, `src/hbtxr/loss/stage2.py`이다.

## 1. 루트 구조

현재 루트는 아래 8개 축으로 이해하는 것이 가장 정확하다.

```text
HBTXR_v3_0/
  configs/
  docs/
  exps/
  legacy/
  packages/
  scripts/
  src/hbtxr/
  tests/
```

### 1.1 main surface

- `configs/`
  - 공식 runtime preset
  - `base.yaml`, `mode1_stage1.yaml`, `mode1_stage2.yaml`, `mode2_stage1.yaml`, `mode2_stage2.yaml`
- `scripts/`
  - 공식 CLI와 shell wrapper
- `src/hbtxr/`
  - 실제 importable 구현
- `tests/`
  - regression, contract, snapshot, CI guard

### 1.2 experimental surface

- `exps/`
  - mode0 preset
  - optimizer pool preset
  - lazy target-FPS preset
  - Grounded-SAM option JSON
  - preview / render / analysis / one-off helper

### 1.3 archive surface

- `docs/others/`
  - 업데이트 리포트, archive map, 분석 보관 문서
- `legacy/`
  - 이전 세대 wrapper/config/doc
- `ref_codes/`
  - 외부 reference code snapshot

## 2. src/hbtxr 계층 구조

현재 `src/hbtxr`는 아래 순서로 읽는 것이 좋다.

```text
utils
  -> preprocess
  -> data
  -> models
  -> loss / optim
  -> training
  -> runtime
```

### 2.1 utils

공통 하부 계층이다.

- `utils/paths.py`
  - canonical root, stored path, cross-platform path 정규화
- `utils/io.py`
  - JSON/JSONL 입출력
- `utils/cache.py`
  - sample cache helper
- `utils/state6.py`
  - `xyabuv`, `xywht` 관련 state 변환

### 2.2 preprocess

offline artifact 생성 계층이다.

- `path_utils.py`
  - CLI, env, paths-config, external package root를 통합 해석
- `groundedsam_build.py`
  - raw frame에 Grounded-SAM annotation 수행
- `canonicalize.py`
  - raw session을 canonical session package로 정규화
- `build_manifests.py`
  - canonical session package를 train/val/test sample row로 분해
- `target_fps_build.py`, `target_fps_canonical.py`
  - target-FPS session store와 lazy mode2 bridge

### 2.3 data

manifest row를 학습용 tensor sample로 materialize하는 계층이다.

- `loader.py`
  - mode별 dataset 선택과 DataLoader 구성
- `dataset.py`
  - orchestration facade
- `session_reader.py`
  - annotation, frame, mask, event, session-store I/O
- `event_builder.py`
  - raw window / source pair average / target-FPS session-store event voxel 생성
- `components.py`
  - adaptive ROI, supervision target, sample assembly
- `transform.py`
  - frame / mask / event에 일관된 transform 적용

### 2.4 models

멀티모달 추론 계층이다.

- `patch_embeddings.py`
  - frame/event patch frontend
- `adapters.py`
  - modality adapter
- `backbone.py`, `blocks.py`
  - token backbone
- `heads.py`
  - eye/search/event/track/mask/aux head
- `hybrid_tracker.py`
  - 전체 tracker facade
- `models/tracker/*`
  - encoder, head factory, search/event/track branch, runtime policy, codec 분리 구현
- `controller.py`
  - search-track runtime FSM

### 2.5 loss / optim

- `loss/`
  - loss primitive, assigner, stage1/stage2 orchestration
- `optim/`
  - optimizer registry와 modifier, optimizer pool 확장

### 2.6 training

전체 학습 orchestration 허브다.

- `training/losses.py`
  - public loss facade
- `training/trainer.py`
  - loader, model, optimizer, scheduler, distillation, checkpoint, export, logging 통합

### 2.7 runtime

온라인 상태 유지와 runtime scheduler 래퍼다.

- `runtime/similarity.py`
  - ellipse similarity
- `runtime/tracker.py`
  - stateful runtime wrapper

## 3. 전체 파이프라인

프로젝트의 공식 파이프라인은 아래 10단계로 읽을 수 있다.

```text
1. external package sync
2. dataset generation / relocation
3. Grounded-SAM annotation
4. canonicalization
5. manifest generation
6. dataloader inspection
7. Stage1 training
8. Stage2 training
9. inference / eval / vis
10. export
```

### 3.1 Step 0: external package sync

- entry
  - `scripts/sync_packages.py`
  - `scripts/run_sync_packages.sh`
- output
  - `packages/Grounded-Segment-Anything`
  - `packages/timelens`
  - `packages/Swift-Eye`

### 3.2 Step 1: dataset relocation

- entry
  - `scripts/relocate_ev_eye_dataset.py`
  - `scripts/run_relocate.sh`
- 역할
  - canonical/manifests 내부 경로 rewrite
  - symlink 복구

### 3.3 Step 2: Grounded-SAM annotation

- entry
  - `scripts/annotate_groundedsam_ev_eye.py`
  - `scripts/run_groundedsam_annotation.sh`
- core
  - `parse_groundedsam_devices`
  - `annotate_dataset_with_groundedsam`
  - `annotate_dataset_with_groundedsam_multi`
- output
  - `workspace/groundedsam_annotations/.../frame_annotations.jsonl`

### 3.4 Step 3: canonicalization

- entry
  - `scripts/canonicalize_hbtxr.py`
  - `scripts/run_canonicalize.sh`
- core
  - `resolve_paths`
  - `canonicalize_dataset`
  - session별 `session_package.json`, `meta.json`, `frame_annotations.jsonl`, `events.npz`
- output
  - `canonical0`, `canonical1`, `canonical2`
  - `indexes/sessions.jsonl`
  - `indexes/canonical_summary.json`

### 3.5 Step 4: manifest generation

- entry
  - `scripts/build_mode_manifests.py`
  - `scripts/run_build_manifests.sh`
- core
  - `build_manifests`
- output
  - `manifests/manifest*/train_manifest.jsonl`
  - `manifest_summary.json`

### 3.6 Step 5: dataloader inspection

- entry
  - `scripts/check_dataloader.py`
  - `scripts/run_dataloader.sh`
- 역할
  - batch key, dtype, shape, first meta 요약

### 3.7 Step 6: Stage1 training

- entry
  - `scripts/train_hbtxr.py`
  - `scripts/run_train.sh`
- 의미
  - eye/search/mask foundation training

### 3.8 Step 7: Stage2 training

- entry
  - `scripts/train_hbtxr.py`
  - `scripts/run_train.sh`
- 의미
  - search + event + track hybrid training
- 특징
  - Stage1 best checkpoint를 init checkpoint로 warm start하는 경우가 많다

### 3.9 Step 8: inference / eval / vis

- eval
  - `scripts/eval_hbtxr.py`
- infer
  - `scripts/infer_hbtxr.py`
- vis
  - `scripts/visualize_dataset.py`
  - `scripts/visualize_inference_results.py`
  - `scripts/visualize_runtime.py`

### 3.10 Step 9: export

- entry
  - `scripts/export_hbtxr.py`
  - `scripts/run_export.sh`
- core
  - `export_structural_student`
- output
  - `train/export/student_export.pt`
  - `student_export_config.json`
  - `student_export_report.json`

## 4. mode / stage 관점에서 본 파이프라인

### 4.1 mode

- `mode0`
  - raw CSV supervision experimental path
  - main surface가 아니라 `exps/` 중심
- `mode1`
  - original frame + event voxel baseline
- `mode2`
  - interpolated frame / target-FPS synthetic path

### 4.2 stage

- `stage1`
  - eye/search/mask foundation
- `stage2`
  - event + track를 포함한 hybrid tracking

## 5. 구조적으로 가장 중요한 허브

### 5.1 scripts/_config.py

이 파일은 실행 표면의 중심이다.

역할:

- `extends` 체인 config load
- experimental YAML old-path alias
- CLI dotted override 적용
- experiment/run contract 계산
- manifest, checkpoint, output 경로 해석
- run artifact 기록

즉 shell wrapper 아래에서 실제 “실행 계약”을 결정하는 위치다.

### 5.2 trainer.py

이 파일은 현재 코드베이스의 제일 큰 orchestration hub다.

역할:

- dataset loader build
- model build
- optimizer/scheduler build
- pretrained / distillation / pruning wiring
- epoch loop
- checkpoint / history / export

### 5.3 dataset.py

현재는 데이터 파이프라인 facade 역할이다.

실제 세부 책임은 이미 분리되어 있다.

- reader
- transform resolver
- event builder
- ROI resolver
- target builder
- sample assembler

### 5.4 hybrid_tracker.py

현재는 model facade다.

실제 세부 책임은 tracker component로 분해돼 있다.

- token encoder
- head factory
- search branch
- event branch
- track branch
- runtime policy
- track codec

## 6. 현재 구조가 의미하는 것

이 프로젝트의 본질은 “raw 데이터 처리 코드 + 모델 코드”가 아니라 “계약을 단계별로 물고 내려가는 시스템”이라는 점이다.

핵심 계약은 아래 순서로 이어진다.

```text
raw layout contract
-> canonical contract
-> manifest contract
-> dataset sample contract
-> tracker output contract
-> loss log contract
```

이 계약들이 테스트와 snapshot으로 고정되어 있기 때문에, 내부 클래스를 분할하거나 엔트리포인트를 정리해도 ABI를 유지할 수 있다.

## 7. 권장 읽기 순서

1. [README.md](../../README.md)
2. [20260330_170419_06_v3_repository_tree_and_module_dependency_map.md](20260330_170419_06_v3_repository_tree_and_module_dependency_map.md)
3. [20260330_170419_09_v3_mode_data_pipeline_contracts.md](20260330_170419_09_v3_mode_data_pipeline_contracts.md)
4. [20260408_190000_tracker_dataset_output_contracts.md](20260408_190000_tracker_dataset_output_contracts.md)
5. [20260408_233200_train_hbtxr_actual_execution_flow.md](20260408_233200_train_hbtxr_actual_execution_flow.md)
6. [20260408_233400_dataset_model_loss_detailed_call_stack.md](20260408_233400_dataset_model_loss_detailed_call_stack.md)
