# Swift-Eye 분석 대화 세션 요약

최종 갱신: 2026-03-27 KST

## 목적

이 문서는 Swift-Eye / TimeLens / EV-Eye 분석 대화를 주제별 세션으로 다시 구성한 요약본입니다. 원문 transcript를 그대로 복제하지 않고, 각 세션의 요청, 분석 범위, 핵심 결론, 남은 불확실성을 정리합니다.

## 세션 1. Swift-Eye 전체 구조와 dataset / dataloader 분석

### 사용자 요청

- Swift-Eye 논문과 코드를 자세히 분석
- 데이터셋 처리 및 데이터 로더 구성을 설명

### 범위

- 논문 핵심 주장 확인
- training / inference 파이프라인 복원
- TimeLens와 Swift-Eye의 역할 분리
- dataset schema, loader, collate 구조 파악

### 핵심 결론

- Swift-Eye는 raw event를 detector/tracker에 직접 넣지 않습니다.
- TimeLens가 먼저 `images + events -> interpolated_frames/*.png`를 생성합니다.
- Swift-Eye는 이 interpolated PNG sequence 위에서 detector/tracker를 수행합니다.
- 학습용 데이터셋은 하나가 아니라 다음 3종으로 나뉩니다.
  - backbone/neck용 DOTA-style single-frame dataset
  - detection head용 single-frame pickle dataset
  - temporal fusion / tracking head용 template-search pair pickle dataset
- temporal fusion 학습의 핵심은 template-search에 동일한 flip/rotate augmentation을 적용하는 점입니다.

### 불확실성

- 실제 `detection_train.pickle`, `tracking_train_dataset.pickle` 파일은 저장소에 없어서 schema는 loader 코드 기준으로 복원했습니다.
- 논문 본문 전체 텍스트 추출은 제한적이어서 paper 주장과 코드를 교차 확인했습니다.

## 세션 2. 데이터셋 폴더 구조와 pickle 스키마 정리

### 사용자 요청

- Swift-Eye용 데이터셋 폴더 구조를 다이어그램으로 정리
- pickle 내부 컬럼 정의와 예시 row 정리

### 핵심 결론

- `train_backbone_and_neck/`는 `images/ + annotations/` 구조를 기대합니다.
- detection pickle 최소 스키마:
  - `image_path`
  - `poly`
- tracking pickle 최소 스키마:
  - `template_path`
  - `origin_poly`
  - `search_path`
  - `occlusion_poly`
- `poly`는 8개 좌표를 가지며 loader 내부에서 oriented bbox `(x,y,w,h,a)`로 변환됩니다.

### 산출물

- 폴더 구조 다이어그램
- pickle row 예시
- 추천 부가 컬럼
- 저장 전 validation 규칙

## 세션 3. raw dataset -> dataloader -> collate -> model forward 텐서 차원 분석

### 사용자 요청

- raw dataset부터 input tensor까지 텐서 차원 단위로 정리
- 표 형식으로 다시 압축 정리

### 핵심 결론

- TimeLens:
  - raw event는 개념적으로 `[E,4]`
  - voxel grid는 `[5,260,346]`
  - 최종 interpolated frame은 `[3,260,346]`
- Swift-Eye detection:
  - pad 후 입력은 `[3,288,352]`
  - 첫 feature map은 `[256,72,88]`
- Swift-Eye tracking:
  - outer batch 기준 image tensor는 `[2B,3,288,352]`
  - crop 후 `template_feature [B,256,13,13]`
  - crop 후 `search_feature [B,256,33,33]`

### 산출물

- 텐서 shape flow 설명
- 4개 표 형태의 pipeline 문서 초안

## 세션 4. frame interpolation 동작 원리 설명

### 사용자 요청

- frame interpolation이 각 frame interval 사이의 event timestamp에 대해 frame을 생성하는 것인지 확인
- `t0 ~ t1` 구간에서 left/right split, voxel grid, warp, fusion, refine, attention까지 설명

### 핵심 결론

- 이해 방향은 대체로 맞지만, 정확히는 “모든 event timestamp마다 frame 1장”이 아니라 “선택된 target timestamp들”에 대한 frame 생성입니다.
- 각 `t*`에 대해
  - `left_events = [t0, t*)`
  - `right_events = [t*, t1)`
  를 만듭니다.
- TimeLens는 이벤트만으로 frame을 직접 그리는 것이 아니라, 양끝 frame을 target 시점으로 warp/fuse/refine 하는 구조입니다.

## 세션 5. time-bin = 5의 의미와 이유 해석

### 사용자 요청

- voxel time-bin을 `5`로 둔 이유 설명

### 핵심 결론

- `5`는 코드상 하드코딩된 기본 voxel representation 값입니다.
- flow network, fusion network 입력 채널 설계도 이 값에 맞춰져 있습니다.
- 논문이나 저장소에서 “왜 꼭 5인지”를 설명하는 명시적 ablation 근거는 확인되지 않았습니다.
- 가장 합리적인 해석은 temporal detail과 sparsity의 절충값이라는 것입니다.

### 실무적 의미

- `5`는 단순 전처리 옵션이 아니라 모델 구조와 결합된 값입니다.
- 값을 바꾸면 pretrained weight 호환성이 깨질 가능성이 큽니다.

## 세션 6. EV-Eye raw dataset frame-interval event count 집계

### 사용자 요청

- 실제 raw dataset `E:\\WSL\\Shared\\dataset\\Eye\\EV_Eye\\raw_data\\Data_davis`를 분석
- frame 간 event가 몇 개인지 확인

### 핵심 결론

- 총 session dir: `388`
- missing frame timestamp file: `4`
- frame timestamp 역전 세션: `16`
- frame/event 시간축 불일치로 모든 interval count가 `0`인 세션: `2`
- 정상 정렬 세션: `366`

### clean-session 통계

- total clean intervals: `1,411,213`
- mean events per interval: `1851.93`
- median events per interval: `812`
- min: `0`
- max: `69039`
- p01: `11`
- p05: `23`
- p25: `179`
- p75: `1382`
- p95: `9007`
- p99: `17857`

### 해석

- EV-Eye의 frame interval event density는 매우 불균일합니다.
- 일부 interval은 수만 개 이벤트를 가지지만, 다수는 수백 개 수준입니다.
- 따라서 199 inserted frame 설정에서 “이벤트가 항상 충분한가”는 motion intensity에 크게 좌우됩니다.

## 세션 7. `정상 정렬 세션 366개`와 interval 비율표 해석

### 사용자 요청

- `정상 정렬 세션 366개` 표와 `이벤트 interval 비율` 표를 자세히 설명

### 핵심 결론

- `366`은 timestamp 기준으로 frame-interval counting이 가능한 usable subset입니다.
- interval 비율표는 누적 threshold 표 성격이 강해서, 배타 구간으로 다시 읽는 편이 더 직관적입니다.

### 배타 구간 기준 interval 분포

- `0 ~ 99`: `272,385` intervals, `19.30%`
- `100 ~ 499`: `204,836` intervals, `14.51%`
- `500 ~ 999`: `440,937` intervals, `31.25%`
- `1000 ~ 4999`: `354,575` intervals, `25.13%`
- `5000 ~ 9999`: `78,719` intervals, `5.58%`
- `>= 10000`: `59,761` intervals, `4.23%`

### 해석

- 약 19%는 매우 sparse interval입니다.
- 약 65%는 `1000` 미만입니다.
- `10000` 이상인 매우 event-rich interval은 4%대에 불과합니다.
- 따라서 EV-Eye는 평균만 보면 dense해 보이지만 실제로는 long-tail 분포입니다.

## 통합 결론

- Swift-Eye는 “event-to-frame interpolation + frame-based detector/tracker” 파이프라인입니다.
- TimeLens의 핵심은 per-target tiny slice가 아니라 left/right 누적 event packet과 boundary frame을 같이 쓰는 점입니다.
- EV-Eye raw dataset은 usable하지만 timestamp 이상 세션이 섞여 있어 sanity check 없이 바로 쓰면 안 됩니다.
- Swift-Eye/TimeLens 계열 이식 전에 다음 전처리가 필요합니다.
  - session timestamp consistency check
  - frame/event time-axis alignment check
  - clean session filtering
  - interpolation input packaging
