# 선택적 브랜치 반영 보고서

마지막 갱신: 2026-03-29 KST

## 요약

이 문서는 아래 항목을 추적 가능하게 정리합니다.

- 어떤 원격 브랜치를 선택적으로 반영했는지
- 어떤 브랜치를 명시적으로 제외했는지
- 현재 브랜치에서 어떤 파일이 추가되거나 수정되었는지
- 어떤 동작을 가져왔는지
- 그 반영을 어떤 방식으로 수행했는지

이번 통합의 고정 원칙은 아래와 같았습니다.

- branch merge 금지
- wholesale cherry-pick 금지
- 구형 브랜치 상태로의 full-file rollback 금지
- 현재 `codex/home` 위에서 필요한 의미만 additive semantic patch 방식으로 반영

제외한 브랜치:

- `codex/ipex`
- `codex/startpoint`

반영한 source branch:

- `codex-exp-deepmicro`
- `codex/home-wsl`
- `exp_wsl`
- `exp-wsl`
- `codex-exp-ubeeslab`

## Commit Map

| Wave | Commit | Message |
| --- | --- | --- |
| 1 | `51a26af` | `Integrate deepmicro Grounded-SAM and target-fps patches` |
| 2 | `633a9f3` | `Import Grounded-SAM review docs and preview tooling` |
| 3 | `ca65d4b` | `Add heuristic ROI docs and preview helpers` |
| 4 | `203ce9e` | `Integrate mode0 experimental search controls` |
| 5 | `6476b10` | `Sync selective integration docs and indexes` |

## 통합 규칙

### 허용한 것

- source branch에만 있던 옵션이나 함수를 현재 코드에 선택적으로 이식
- 새 문서, 자원, helper script, test 추가
- 현재 파일의 필요한 semantic 지점만 patch
- 현재 브랜치의 최신 시스템 유지:
  - optimizer pool
  - loss cluster 분리
  - WSL CUDA 가이드
  - lazy `mode2`
  - stage-aware stat filtering

### 허용하지 않은 것

- source branch 직접 merge
- 현재 파일 전체를 구형 브랜치 버전으로 교체
- 현재 `codex/home`의 신규 시스템을 제거하여 예전 브랜치에 맞추는 회귀

## Wave 1: `codex-exp-deepmicro` 코드 반영

### source branch 역할

주요 source:

- Grounded-SAM build 보강
- target-FPS build 강건화
- target-FPS canonical path 보정

### 대상 파일

| 현재 파일 | source branch | 반영한 내용 | 반영 방식 |
| --- | --- | --- | --- |
| `src/hbtxr/preprocess/groundedsam_build.py` | `codex-exp-deepmicro` | user/eye/session filtering, nonstandard-session 제어, default checkpoint auto-resolution, `sam_hq_model_registry` 지원 | additive function/config patch |
| `src/hbtxr/preprocess/target_fps_build.py` | `codex-exp-deepmicro` | `.npz`와 함께 raw `events.txt` 입력 지원 | additive loader patch, 기존 lazy/session-store logic 유지 |
| `src/hbtxr/preprocess/target_fps_canonical.py` | `codex-exp-deepmicro` | `canonical_root == target_fps_root`일 때의 nesting fix | 좁은 범위 path-resolution patch |
| `tests/test_preprocess_v3.py` | `codex-exp-deepmicro` | Grounded-SAM helper/filter regression | test 추가 |
| `tests/test_target_fps_build_v3.py` | `codex-exp-deepmicro` | txt-event target-FPS regression | test 추가 |
| `tests/test_target_fps_canonical_v3.py` | `codex-exp-deepmicro` | canonical nesting regression | test 추가 |

### 반영한 동작

- `groundedsam_build.py`에 아래 필터를 추가
  - `user_id`
  - `eye`
  - `session_codes`
  - `include_nonstandard_sessions`
- 기본 후보 checkpoint가 있을 때 GroundingDINO와 SAM checkpoint를 자동 해석
- checkpoint가 HQ variant일 경우 `sam_hq_model_registry` 지원
- target-FPS build가 `load_events_from_txt(...)`를 통해 raw text event stream도 읽을 수 있게 확장
- canonical root와 target-FPS root가 동일할 때도 output nesting이 안정적으로 유지되도록 보정

### 유지한 guardrail

- 현재 lazy `mode2`의 `session_store_layout` 제거 금지
- 현재 `frame_storage_mode=lazy_source_frames` 유지
- optimizer / CUDA / loss 체계의 회귀 금지

## Wave 2: `codex-exp-deepmicro` + `codex/home-wsl` 문서 / 자원 / preview tooling 반영

### source branch 역할

- `codex-exp-deepmicro`
  - 기술 실험 요약
  - sampled dataset notes
- `codex/home-wsl`
  - review / handoff / failure-analysis 문서

### 추가한 문서

| 현재 문서 | 주 source | 목적 |
| --- | --- | --- |
| `docs/exps/20260330_170419_14_groundedsam_roi_crop_prompted_experiment_summary.md` | `codex-exp-deepmicro` | Grounded-SAM ROI-crop 실험 요약 |
| `docs/exps/20260330_170419_15_all48_v2_baseline_failure_review.md` | `codex/home-wsl` | `all48` failure review |
| `docs/exps/20260330_170419_16_all48_v2_handoff.md` | `codex/home-wsl` | handoff 및 follow-up context |
| `docs/exps/20260330_170419_17_all48_sampled_dataset_construction_experiment.md` | `codex-exp-deepmicro` | sampled dataset construction notes |

### 추가한 자원 / 스크립트

| 현재 경로 | source | 목적 |
| --- | --- | --- |
| `docs/experiments/resources/08_all48_v2_baseline_failure_review/` | `codex/home-wsl` | review overlay와 manifest 자산 |
| `scripts/review_roi_crop_prompted_batch.py` | `codex-exp-deepmicro` | batch review helper |
| `scripts/run_roi_crop_prompted_preview_batch.py` | `codex-exp-deepmicro` | preview batch launcher |

### 반영 방식

- 현재 `codex/home`에 없던 문서/자원/스크립트는 신규 import
- 문서 번호는 현재 브랜치 규칙에 맞춰 `14`~`17`로 재배치
- docs index 구조는 현재 브랜치 체계를 유지

## Wave 3: `exp_wsl` + `exp-wsl` heuristic ROI / preview / helper script 반영

### source branch 역할

- `exp_wsl`
  - heuristic ROI reference material
  - preview / overlay tooling
- `exp-wsl`
  - mode1 event-sweep helper script

### 추가한 문서와 계획

| 현재 경로 | source | 목적 |
| --- | --- | --- |
| `docs/prj/20260330_170419_18_current_heuristic_eye_region_roi_algorithm.md` | `exp_wsl` | heuristic eye ROI 알고리즘 참조 문서 |
| `docs/plan/20260329_124101_08_v3_6_heuristic_eye_roi_transition_plan.md` | `exp_wsl` | heuristic ROI transition plan |
| `docs/plan/20260329_124101_09_v3_7_target_fps_groundedsam_closure_plan.md` | `exp_wsl` | target-FPS / Grounded-SAM closure plan |
| `docs/plan/20260329_124101_10_v3_8_all48_sampled_dataset_construction_plan.md` | `exp_wsl` | `all48` sampled dataset construction plan |

### 추가한 스크립트

| 현재 경로 | source | 목적 |
| --- | --- | --- |
| `scripts/groundedsam_eye_bbox_overlay.py` | `exp_wsl` | bbox overlay review |
| `scripts/raw_overlay_preview_eye_region_bbox.py` | `exp_wsl` | raw eye-region bbox preview |
| `scripts/raw_overlay_preview_groundedsam.py` | `exp_wsl` | Grounded-SAM raw preview |
| `scripts/raw_overlay_preview_groundedsam_eye_region.py` | `exp_wsl` | Grounded-SAM eye-region preview |
| `scripts/raw_overlay_preview_groundedsam_mask_bbox.py` | `exp_wsl` | mask/bbox preview |
| `scripts/raw_overlay_preview_h5_mask.py` | `exp_wsl` | H5 mask preview |
| `scripts/raw_overlay_preview_manual_csv.py` | `exp_wsl` | manual CSV preview |
| `scripts/spotcheck_groundedsam_annotations.py` | `exp_wsl` | annotation spotcheck |
| `scripts/run_mode1_event_sweep.sh` | `exp-wsl` | mode1 event-policy sweep helper |

### 이 wave에서 반영한 코드

| 현재 파일 | source | 반영 내용 | 방식 |
| --- | --- | --- | --- |
| `src/hbtxr/preprocess/io_utils.py` | `exp_wsl` | asymmetric eye-region margin | `derive_eye_region(...)`에 대한 최소 patch |

### 반영 방식

- `exp_wsl`의 구형 training / optimizer / target-FPS 코드는 가져오지 않음
- standalone script와 문서만 반영
- 기존 코드와 문서 서술이 어긋나는 부분만 최소 수정

## Wave 4: `codex-exp-ubeeslab` experimental surface 반영

### source branch 역할

- trainer safety
- experimental search-center fusion
- `mode0` debug preset 및 관련 notes

### 대상 파일

| 현재 파일 | source | 반영 내용 | 방식 |
| --- | --- | --- | --- |
| `configs/base.yaml` | `codex-exp-ubeeslab` | `model.search.xy_from_mask_centroid` config surface | additive config field |
| `configs/mode0_stage1.yaml` | `codex-exp-ubeeslab` | eye-only sanity preset | `mode0_stage1` 범위 안에서 재구성 |
| `configs/mode0_stage2.yaml` | `codex-exp-ubeeslab` | mask-centroid search fusion preset | additive experimental preset change |
| `src/hbtxr/models/hybrid_tracker.py` | `codex-exp-ubeeslab` | `search_cfg`, soft mask centroid, search `x/y` replacement | semantic patch only |
| `src/hbtxr/training/trainer.py` | `codex-exp-ubeeslab` | custom `best_metric_name`용 dynamic checkpoint | semantic patch only |
| `docs/exps/progress/20260330_170419_19_mode0_stage1_debug_history_and_progress.md` | `codex-exp-ubeeslab` | debug history와 findings | 현재 브랜치 번호 체계에 맞춰 반영 |

### 반영한 동작

- model-level experimental switch:
  - `model.search.xy_from_mask_centroid`
- 활성화 시 `mask_head`가 있을 경우:
  - `search/mask_logits`의 soft centroid로 search `x/y` 대체
- custom `training.best_metric_name`에 대해서:
  - `best_<metric>.pt`를 별도 생성
- `mode0_stage1`은 eye-only sanity preset 역할 수행
- `mode0_stage2`는 mask-centroid 기반 search-fusion experiment preset 역할 수행

### 의도적으로 반영하지 않은 것

- 구형 optimizer 로직
- 구형 console logger 교체
- 구형 base trainer rollback
- optimizer pool / loss-cluster / CUDA WSL guidance 제거

## Wave 5: 최종 동기화

### 대상 파일

| 현재 파일 | 목적 |
| --- | --- |
| `README.md` | 새 문서/스크립트를 메인 실행 안내에 반영 |
| `docs/hbtxr/00_index.md` | 당시 `14`~`19` 문서와 자원을 인덱싱 |
| `docs/plan/00_index.md` | `08`~`10` 계획 문서 인덱싱 |
| `docs/chat/20260326_011438_01_v3_update_history.md` | 통합 이력 반영 |
| `docs/chat/20260326_011438_02_v3_progress_checklist.md` | plan-vs-progress 상태 반영 |
| `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md` | 결정 근거 반영 |

### 반영 방식

- 이 wave에서는 코드 동작을 바꾸지 않음
- README, index, tracking docs만 동기화

## 검증 요약

| Wave | 검증 |
| --- | --- |
| 1 | preprocess / target-FPS targeted regression 통과 |
| 2 | imported preview script `--help` smoke 통과 |
| 3 | preview script `--help` smoke 및 preprocess regression 통과 |
| 4 | focused model/config/train regression, broader trainer/runtime regression 통과 |
| 최종 | 전체 저장소 regression 통과: `140 passed` |

## 최종 해석

이번 작업은 merge가 아니라, 특정 capability만 현재 브랜치에 backport한 통합입니다.

- `codex-exp-deepmicro`
  - code-first source
- `codex/home-wsl`
  - review / handoff / resource source
- `exp_wsl`
  - heuristic ROI 문서 및 preview tooling source
- `exp-wsl`
  - helper script source
- `codex-exp-ubeeslab`
  - experimental search / mode0 semantic-patch source

결과적으로 현재 `codex/home`는 자기 자신의 최신 구조를 유지하면서, 필요하다고 판단된 추가 기능만 흡수한 상태입니다.
