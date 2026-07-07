# HBTXR_v3_0 Mode0 Stage1 디버그 이력 및 진행 상태

마지막 갱신: 2026-03-29 KST

## 요약

이 문서는 최근 수행한 `mode0-stage1` 디버그 작업의 흐름, 주요 코드/설정 변경, 변경 판단 근거, 그리고 원래 조사 계획 대비 현재 상태를 정리합니다.

이번 디버그는 아래 네 단계로 진행되었습니다.

1. 런타임 안정화
2. 데이터 계약 재점검
3. eye ROI 회귀를 위한 target 및 resize 경로 수정
4. `eye-only` sanity 학습과 시각 검증

## 1. 이번 패스에서 완료된 변경

- `--device cuda:0,cuda:1` 지정 시 CPU로 조용히 fallback 되던 런처/런타임 동작을 수정
- epoch 로그를 `train_log.txt`뿐 아니라 터미널에도 출력하도록 정리
- `mode0` 데이터 계약을 [07_ev_eye_dataset_analysis_results.md](../../others/analysis/20260330_170419_07_ev_eye_dataset_analysis_results.md) 기준으로 다시 확인하여, 현재 `mode0`가 `clean 366`이 아니라 `valid 384`에 해당함을 명확히 함
- manifest metadata와 runtime transform size 간의 불일치를 분석하고 원인을 설명
- `mode0-stage1` eye ROI 회귀를 `eye-only` sanity preset으로 분리
- 종횡비를 유지하는 새 resize 정책 `sensor_full_letterbox` 추가 및 검증
- eye-only Stage1 30 epoch 학습 완료 및 유의미한 best checkpoint 확보
- random validation sample 8장에 대해 best checkpoint overlay preview 생성

## 2. 핵심 발견

### 2.1 런타임과 로깅

- CUDA 인자는 이미 학습 스크립트까지 전달되고 있었지만, trainer가 명시적으로 실패하지 않고 CPU로 fallback 할 수 있었습니다.
- 기존에는 `train_log.txt`가 제대로 기록되더라도, epoch 단위 요약이 터미널에 바로 보이지 않았습니다.

### 2.2 데이터세트와 계약

- 현재 `mode0` canonicalization은 `388 total`, `384 valid`, `4 skipped`에 대응합니다.
- 현재 `mode0`는 `clean 366` subset을 자동으로 강제하지 않습니다.
- manifest metadata가 한 resize 계약을 보고하더라도, dataset loader에서 runtime config override가 다른 transform 경로를 적용할 수 있습니다.

### 2.3 Eye ROI supervision

- `facet_square_direct`에서는 학습 입력이 `roi_xywh` 기준으로 만들어져, transform 이후 `eye_target`이 거의 full-canvas 상수 박스로 수렴하는 문제가 있었습니다.
- 반면 sensor frame 기준 overlay preview는 sensor 좌표계에서 그려졌기 때문에 정상처럼 보였습니다.
- `sensor_full_square`는 full-frame supervision을 복구하지만 종횡비 왜곡을 유발했습니다.
- `sensor_full_letterbox`는 종횡비를 유지하면서 full-frame supervision을 복구했습니다.

### 2.4 Search center 조사

- 더 넓은 Stage1 불안정성 조사 결과, search center head가 mask branch보다 약할 가능성이 보였습니다.
- 코드에는 mask centroid 기반 search-center fusion 경로를 추가했고, [mode0_stage2.yaml](../../../exps/configs/mode0_stage2.yaml)에서 활성화했습니다.
- 다만 이번 패스에서 실제로 검증된 학습 run은 여전히 `eye-only` Stage1 sanity run입니다.

## 3. 진행 연표

| 날짜 | 주제 | 결과 |
| --- | --- | --- |
| 2026-03-28 | 브랜치 분리와 초기 커밋 | 작업을 `codex-exp-ubeeslab`로 옮기고 초기 config/runtime 변경을 checkpoint |
| 2026-03-28 | GPU 미사용 문제 | 런처/런타임 동작 수정 후 GPU 학습 사용 확인 |
| 2026-03-28 | 터미널 로그 가시성 | trainer가 epoch JSON row를 stdout에도 출력하도록 수정 |
| 2026-03-28 | `clean 366` vs 현재 `mode0` | 현재 파이프라인이 `clean 366`이 아니라 `valid 384`에 대응함을 확인 |
| 2026-03-29 | 학습 붕괴 원인 조사 | 초기 Stage1 붕괴가 eye ROI 자체보다는 pupil/search supervision 문제에 더 가깝다는 점 확인 |
| 2026-03-29 | `target_size_wh` 감사 | manifest의 `256x160` metadata가 현재 loader의 실제 runtime transform과 다름을 확인 |
| 2026-03-29 | eye ROI target 분석 | `facet_square_direct`가 입력 좌표계에서 `eye_target`을 사실상 상수 박스로 만들고 있음을 확인 |
| 2026-03-29 | full-frame eye regression 경로 | `mode0_stage1`을 full-sensor supervision으로 전환한 뒤 `sensor_full_letterbox`로 정리 |
| 2026-03-29 | eye-only Stage1 sanity run | 30 epoch 완료, epoch 10에서 best eye-only checkpoint 획득 |
| 2026-03-29 | best checkpoint overlay 검증 | sensor 좌표계에서 random validation overlay 8장을 생성 |

## 4. 현재 코드 및 설정 변경

### 4.1 Runtime / trainer

- [src/hbtxr/training/trainer.py](../../../src/hbtxr/training/trainer.py)
  - epoch row를 stdout에 출력
  - 임의의 `training.best_metric_name`을 받아 `best_<metric>.pt`를 기록
  - `model.search` config를 모델 build 경로로 전달

### 4.2 Stage1 loss 유연화

- [src/hbtxr/training/losses.py](../../../src/hbtxr/training/losses.py)
  - `compute_stage1_losses()`가 비활성 head를 안전하게 건너뛰도록 정리
  - `eye-only` Stage1 sanity configuration을 가능하게 함

### 4.3 Resize 정책

- [src/hbtxr/data/dataset.py](../../../src/hbtxr/data/dataset.py)
  - `sensor_full_letterbox` 추가
  - `letterbox_square`, `sensor_full_letterbox`를 모두 종횡비 유지형 letterbox transform으로 처리
- [scripts/prepare_ev_eye.py](../../../scripts/prepare_ev_eye.py)
- [scripts/build_groundedsam_dataset.py](../../../scripts/build_groundedsam_dataset.py)
- [src/hbtxr/preprocess/build_manifests.py](../../../src/hbtxr/preprocess/build_manifests.py)
  - CLI choices에 `sensor_full_letterbox` 추가

### 4.4 모델 동작

- [src/hbtxr/models/hybrid_tracker.py](../../../src/hbtxr/models/hybrid_tracker.py)
  - `search.xy_from_mask_centroid` 추가
  - `search/mask_center` 노출
  - `search/mask_logits`의 differentiable centroid로 search-center `x/y`를 대체할 수 있게 함

### 4.5 Preset

- [exps/configs/mode0_stage1.yaml](../../../exps/configs/mode0_stage1.yaml)
  - 현재 `eye-only` sanity preset
  - `sensor_full_letterbox` 사용
  - `distillation`, `pruning` 비활성화
  - eye 외 Stage1 loss weight를 모두 0으로 설정
- [exps/configs/mode0_stage2.yaml](../../../exps/configs/mode0_stage2.yaml)
  - mask branch 유지
  - `model.search.xy_from_mask_centroid: true` 활성화

## 5. 검증된 산출물

### 5.1 Eye-only Stage1 run

- run root:
  - `runs/mode0_stage1_eye_only_letterbox30_20260329_013549`
- best checkpoint:
  - `runs/mode0_stage1_eye_only_letterbox30_20260329_013549/train/best.pt`
- history:
  - `runs/mode0_stage1_eye_only_letterbox30_20260329_013549/train/history.json`

best snapshot:

- best validation `loss_eye`: `10.081159`
- best epoch: `10`
- best epoch train `loss_eye`: `6.410475`
- epoch 30 train `loss_eye`: `1.582119`
- epoch 30 val `loss_eye`: `11.709246`

해석:

- 학습 자체는 안정적입니다.
- `sensor_full_letterbox`가 의미 있는 eye ROI bbox regression을 가능하게 했습니다.
- epoch 10 전후가 best window이며, 이후에는 overfitting 조짐이 보입니다.

### 5.2 Overlay 검증

- output directory:
  - `workspace/previews/mode0_stage1_eye_only_best_eye_bbox_overlay_8`
- contact sheet:
  - `workspace/previews/mode0_stage1_eye_only_best_eye_bbox_overlay_8/contact_sheet.png`
- summary:
  - `workspace/previews/mode0_stage1_eye_only_best_eye_bbox_overlay_8/summary.json`

overlay summary:

- split: `val`
- random seed: `42`
- sample count: `8`
- sensor 좌표계 mean IoU: `0.663755`
- min IoU: `0.540969`
- max IoU: `0.725800`

## 6. 원래 계획 대비 현재 상태

### 6.1 조사 체크리스트

- [x] GPU runtime selection 확인 및 silent CPU fallback 제거
- [x] epoch 단위 터미널 로깅 복구
- [x] `valid 384` vs `clean 366` 차이 명확화
- [x] manifest `target_size_wh`가 runtime에서 실제 쓰이는지 감사
- [x] sensor-frame overlay 품질과 training-target 붕괴가 동시에 가능했던 이유 설명
- [x] `eye-only` Stage1 supervision 분리
- [x] 종횡비 왜곡 없는 `sensor_full_letterbox` 도입
- [x] 30 epoch eye-only sanity 실험 실행
- [x] random sample 기준 best-checkpoint overlay 생성
- [ ] 수정된 `mode0` full-sensor 계약 위에 `search`, `mask` supervision 재통합
- [ ] `xy_from_mask_centroid`가 full Stage1 안정성에 도움을 주는지 검증
- [ ] `mode0`를 `valid 384`로 유지할지 `clean 366` 기준으로 재구성할지 결정

### 6.2 현재 상태 판단

- 완료:
  - eye ROI regression 경로 자체는 종횡비 보존 full-sensor transform 기준으로 검증 완료
- 부분 완료:
  - search-center fusion 코드는 존재하고 `mode0_stage2`에 연결되어 있으나, 이번 패스에서 full Stage1 학습 run으로 재검증하지는 않음
- 보류:
  - `eye-only` 분리 이후, 전체 `mode0-stage1` recipe를 다시 통합하는 단계가 남아 있음

## 7. 권장 다음 단계

가장 자연스러운 후속 순서는 아래와 같습니다.

1. [exps/configs/mode0_stage1.yaml](../../../exps/configs/mode0_stage1.yaml)에 `search`, `mask` head를 다시 복구하되 `sensor_full_letterbox`는 유지
2. 수정된 full-sensor 계약으로 짧은 Stage1 sanity experiment 재실행
3. 기본 search-center 회귀와 `xy_from_mask_centroid`를 비교
4. full Stage1 경로가 안정화된 뒤에야 `clean 366`을 manifest-only filter로 둘지 canonical-level filter로 올릴지 결정
