# Swift-Eye 논문 및 코드 분석

최종 갱신: 2026-03-27 KST

## 목적

이 문서는 Swift-Eye 논문과 로컬 코드 분석 결과를 한 곳에 정리합니다. 논문이 주장하는 방법과 실제 공개 코드가 어떻게 대응되는지, 그리고 dataset / loader / inference 관점에서 어떤 구조를 취하는지를 요약합니다.

## 분석 범위

- 논문 핵심 문제 정의
- 논문 핵심 기여와 방법 요약
- 공개 코드의 실제 training / inference 구조
- 논문과 코드의 일치 지점
- 코드 기준으로 확인 가능한 구조와 확실하지 않은 부분 구분

## 1. 논문 요약

### 문제 정의

Swift-Eye는 near-eye 환경에서 blink, eyelid occlusion, fast eye motion 때문에 발생하는 pupil tracking 불안정을 줄이려는 방법입니다. 핵심 문제는 저프레임 카메라만으로는 blink 전후의 pupil motion과 shape 변화를 안정적으로 복원하기 어렵다는 점입니다.

### 중심 아이디어

1. event camera와 저속 frame을 함께 사용합니다.
2. TimeLens 계열 interpolation으로 저프레임 입력을 dense frame sequence로 보강합니다.
3. dense frame sequence 위에서 pupil detection, tracking, interpolation을 조합해 robust trajectory를 복원합니다.
4. occlusion 정도를 판단해 detection / tracking / interpolation 모드를 전환합니다.

### 논문이 내세우는 기여

- anti-blink pupil tracking framework
- event-based high-frequency near-eye motion analysis
- heavy occlusion과 blink 구간에서의 robustness 향상
- offline precise and robust pupil estimation pipeline

### 실무적 해석

Swift-Eye는 “event camera를 detector에 직접 넣는 방식”보다는, event를 이용해 frame sequence를 먼저 보강한 뒤 그 위에서 pupil estimation을 안정화하는 전체 시스템 제안으로 읽는 편이 맞습니다.

## 2. 코드 수준 아키텍처 요약

### 상위 파이프라인

공개 코드 기준 파이프라인은 다음 4단계입니다.

1. TimeLens가 `images/ + events/ + timestamps`를 읽어 `interpolated_frames/*.png`를 생성
2. Swift-Eye backbone + neck를 single-frame rotated detection dataset으로 학습
3. detection head와 temporal fusion / tracking head를 별도 단계로 학습
4. 추론 시 mask 기반 open-extent 판단으로 detection / tracking / interpolation 모드를 전환

### Swift-Eye가 하지 않는 일

중요한 점은 Swift-Eye 본체가 raw event tensor를 detector/tracker 입력으로 직접 사용하지 않는다는 것입니다. raw event는 TimeLens interpolation 단계에서만 직접 사용되고, Swift-Eye detector/tracker는 보간된 PNG frame sequence를 사용합니다.

### 주요 runtime 구성요소

- backbone
- neck
- detection head
- correlation head
- tracking head
- pupil mask UNet

runtime state:

- `template`
- `last_pred_rbbox`
- `last_open_extent`
- `mode`

즉 Swift-Eye inference는 완전한 stateless detector가 아니라 이전 프레임 상태를 유지하는 online tracker 구조입니다.

## 3. 코드 기준 학습 구조

### Stage A. backbone and neck

- MMRotate `DOTADataset` 기반 single-frame rotated detection 학습
- class는 `pupil` 하나
- image와 rotated bbox annotation 사용

### Stage B. detection head 학습

- pickle row 1개가 frame 1장과 polygon GT 1개를 의미
- 최소 row schema:
  - `image_path`
  - `poly`

### Stage C. temporal fusion / tracking 학습

- pickle row 1개가 template-search pair를 의미
- 최소 row schema:
  - `template_path`
  - `origin_poly`
  - `search_path`
  - `occlusion_poly`

### tracking dataset이 pair-based인 이유

tracking head는 template feature와 search feature의 상관관계를 학습해야 하므로 dataset이 처음부터 pair 구조를 가집니다. 여기서 중요한 구현 포인트는 template와 search에 동일한 geometric augmentation을 적용해 relative geometry가 깨지지 않게 하는 점입니다.

## 4. 코드 기준 추론 구조

### 첫 프레임

첫 프레임은 detection mode로 시작합니다. detector가 pupil을 찾으면 해당 위치의 feature patch를 `template`로 저장합니다.

### 이후 프레임

이후 프레임은 다음 세 모드 중 하나를 선택합니다.

- detection
- tracking
- interpolation

### 모드 전환 기준

모드 선택의 중심은 `open_extent`입니다. 이는 predicted ellipse와 mask UNet segmentation 결과의 overlap 정도를 사용해 계산됩니다.

실질적 의미:

- pupil이 충분히 보이면 detection
- 부분적으로 보이면 tracking
- 거의 닫히거나 검출이 약하면 interpolation

## 5. 전체 시스템에서 TimeLens의 역할

### 기능

TimeLens는 Swift-Eye의 단순 전처리 이전 단계가 아니라, 시스템 전체에서 중요한 high-speed frame synthesizer 역할을 합니다.

입력:

- boundary frame pair
- frame 사이 event stream

출력:

- dense interpolated frame sequence

### 왜 중요한가

논문이 말하는 high-frequency eye movement analysis는 이 interpolated sequence가 있어야 성립합니다. 즉 Swift-Eye의 시간 해상도 증가는 detector 자체가 아니라 TimeLens가 만든 dense frame stream에 크게 의존합니다.

## 6. 논문-코드 정렬 표

| 항목 | 논문 수준 주장 | 코드에서 확인된 구조 | 판단 |
|---|---|---|---|
| anti-blink tracking | blink/occlusion 상황에서 robust pupil estimation | mask 기반 open-extent + detection/tracking/interpolation mode switching | 일치 |
| high-frequency analysis | event를 이용한 고주파 pupil trajectory 복원 | TimeLens로 dense frame sequence 생성 후 Swift-Eye 추론 | 일치 |
| event 활용 방식 | event camera 활용 | event는 TimeLens interpolation 단계에서 직접 사용 | 일치 |
| direct event detector | event를 detector 입력으로 직접 사용한다는 뉘앙스는 약함 | Swift-Eye 본체는 interpolated PNG 기반 | direct event detector는 아님 |
| temporal fusion | 시간 정보 활용 | template-search pair tracking head, correlation head | 일치 |

## 7. 코드에서 보이는 장점

- detector-only 파이프라인보다 blink/occlusion 상황에 강한 mode-switching 구조
- tracking 학습을 pair dataset으로 분리해 목적이 명확함
- TimeLens를 분리해 interpolation과 detection/tracking 책임이 분명함
- feature-template cache를 사용해 inference가 완전 재검출 기반이 아님

## 8. 코드에서 보이는 한계

- pipeline이 multi-stage라 재현성이 낮아질 수 있음
- raw event -> detector 직접 연결이 아니라, 최종 detector가 event를 직접 학습하지는 않음
- 실제 dataset 생성 스크립트가 없어서 pickle schema는 loader 역추적이 필요함
- 일부 `img_metas` 하드코딩과 shape 표기가 깔끔하지 않음
- 공개 코드만으로는 full training data provenance를 완전히 복원하기 어려움

## 9. Dataset / Dataloader 관점 함의

Swift-Eye 구조를 dataset 관점에서 요약하면 다음과 같습니다.

- interpolation 입력 dataset
  - `images/*.png`
  - `events/*.npz`
  - `timestamp.txt`
- detection dataset
  - frame-level pickle 또는 rotated detection annotation
- tracking dataset
  - template-search pair pickle
- inference dataset
  - `interpolated_frames/*.png`

즉 unified dataset 하나가 아니라, stage별로 sample contract가 다른 multi-dataset system입니다.

## 10. 확인된 사실과 추론의 구분

### 코드에서 확인된 사실

- TimeLens는 frame 사이 event를 split해 interpolated frame을 생성합니다.
- Swift-Eye detection / tracking 학습은 별도 dataset과 trainer를 사용합니다.
- tracking dataset은 template-search pair 구조입니다.
- inference는 detection / tracking / interpolation mode를 오갑니다.

### 주변 근거를 통한 추론

- backbone/neck 학습셋은 DOTA-style rotated detection annotation일 가능성이 높습니다.
- 논문의 실험용 full preprocessing pipeline은 공개 저장소만으로 100% 재현되지는 않습니다.
- TimeLens의 `time-bin=5`는 explicit ablation보다는 inherited design choice일 가능성이 높습니다.

## 11. 결론

Swift-Eye는 “event detector”라기보다 다음 구조에 가깝습니다.

- event-assisted dense frame generation
- frame-based pupil detection and template tracking
- segmentation-guided mode switching for anti-blink robustness

따라서 이 방법을 다른 데이터셋에 이식하려면 detector 자체보다 먼저 다음 세 가지를 맞춰야 합니다.

1. frame-event timestamp alignment
2. TimeLens input packaging
3. template-search pair dataset construction

## 로컬 참조 경로

- 논문 PDF
  - `E:\WSL\Shared\ETRI_SYNC\HBTXR\references\(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`
- Swift-Eye 코드
  - `E:\WSL\Shared\ETRI_SYNC\HBTXR\references\Swift-Eye-main\Swift-Eye-main`
- Swift-Eye가 사용하는 TimeLens 코드
  - `E:\WSL\Shared\ETRI_SYNC\HBTXR\references\timelens`
