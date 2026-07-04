# EyeLoRiN 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\EyeLoRiN`
> 분석일: 2026-06-20 / 도구: Glob, Grep, Read (실제 코드 라인 근거 기반)
> 근거 표기 규칙: 라인 인용은 `파일:라인` 형식. 코드로 직접 확인되지 않은 내용은 "추정" 또는 "확인 불가"로 명시.

---

## 0. 한눈에 보는 결론 (TL;DR)

- 이 repo는 **모델 학습/추론 코드가 아니라, 이미 학습된 이벤트 기반 시선추적 모델의 "출력(동공 좌표 CSV)"을 다듬는 추론시점(inference-time) 후처리(post-processing) 코드베이스**다. (근거: README.md:8-10, method.py 전체, jitter_metric.py 전체)
- 이름 "EyeLoRiN"은 백본 아키텍처 명칭이 **아니다**. repo에는 backbone/neck/head/loss/dataloader/train loop가 **존재하지 않는다** (확인: 전체 파일이 README.md + method.py + jitter_metric.py 3개뿐).
- 핵심 기여 2가지는 (i) **Motion-Aware (적응형) Median Filtering**, (ii) **Optical Flow 기반 Local Refinement**, 그리고 평가용 (iii) **Jitter Metric** (속도 엔트로피 + SPARC 기반)이다. (근거: README.md:10)
- "LoRiN = Low-Rank / Recurrent / SSM" 같은 신경망 구조는 **이 repo 코드에서 확인 불가**. 실제 코드는 NumPy/pandas/scipy/OpenCV 기반 고전 신호처리다. (근거: method.py:1-16 import 목록 — torch 없음)

---

## 1. 개요 (Overview)

### 1.1 목적
기존 이벤트 기반 gaze estimation(동공/시선 추정) 모델을 **재학습하거나 구조를 바꾸지 않고**, 모델 출력 좌표 시퀀스의 **시간적 부드러움(temporal smoothness)과 공간 jitter**를 개선하는 model-agnostic 후처리 프레임워크. (근거: README.md:10 "model-agnostic, inference-time refinement framework ... without modifying their architecture or requiring retraining")

### 1.2 원논문 / 챌린지
- 논문: "Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing", **IJCAI 2025 Workshop (4DMR2025, Micro-expression Recognition)** 채택. arXiv:2506.12524. (근거: README.md:3, README.md:6, README.md:16-24)
- 챌린지: **CVPR 2025 Event-based Eye Tracking Competition 2위 수상**. (근거: README.md:4)

### 1.3 입력 / 출력
- 입력 1: **이벤트 데이터** — `.h5` 파일의 `events` 배열. 필드 `(t, x, y, p)`, t는 마이크로초(us), polarity는 0/1. (근거: method.py:29-33, method.py:30 주석 "t is in us")
- 입력 2: **베이스라인 모델의 동공 좌표 예측 CSV** (`x`, `y`, `row_id` 컬럼). (근거: method.py:166-168, method.py:227 `./original_bigBrains/submission_check_*.csv`)
- 출력: **정제된 동공 좌표 CSV** (`row_id`, `x`, `y`). 좌표는 1/8 스케일로 환산되어 저장. (근거: method.py:220-221, method.py:150-151)
- 출력 형태는 **동공 중심 좌표(x, y) 회귀**이며, **세그멘테이션 마스크 출력은 이 repo 코드에서 확인 불가** (CSV는 좌표만 다룸).

### 1.4 센서/해상도 가정
- 센서 크기 `(640, 480, 2)` (W, H, polarity 채널), DVS류 이벤트 카메라 가정. (근거: method.py:25)
- 좌표 스케일 `scale = 8` — 베이스라인 예측은 1/8 다운스케일 좌표계로 보이며, 후처리 시 ×8로 원해상도 복원 후 다시 ÷8 저장. (근거: method.py:164, method.py:167-168, method.py:220)

---

## 2. 디렉토리 구조

### 2.1 실제 구조 (자체 핵심 소스)
```
EyeLoRiN/
├── README.md          # 프로젝트 개요, 논문/챌린지 링크, citation (28 lines)
├── method.py          # 핵심 후처리: median filter + optical-flow refinement (227 lines)
└── jitter_metric.py   # 평가 메트릭: velocity-entropy + SPARC + 노이즈 시뮬레이션 (159 lines)
```
(근거: Glob 전체 스캔 — 위 3개 외 소스 파일 없음. `.h5/.csv/.png/.pth/.pt/.ckpt` 데이터·체크포인트 0건)

### 2.2 제외 대상 (분석 제외 명시)
- `.git/` 전체 (버전관리 메타데이터, 분석 대상 아님)
- 외부 프레임워크/vendor/third_party: **존재하지 않음** (의존성은 pip 패키지로만 import, 벤더링된 원본 소스 없음)
- 빌드 산출물/대용량 데이터/체크포인트: **존재하지 않음**

### 2.3 코드에서 참조하지만 repo에 미포함된 경로 (런타임 입력, 확인 불가)
- `./event_files/{file_name}.h5` (이벤트 원본) — method.py:29
- `./original_bigBrains/submission_check_{file_name}.csv` (베이스라인 예측) — method.py:227
- `./refined/refined_predictions_{file_name}.csv` (출력) — method.py:227
- jitter_metric.py의 `true_labels` 전역 변수 — method.py가 아닌 jitter_metric.py:131에서 정의 없이 사용 → **실행 시 NameError 발생, 외부에서 주입 가정** (확인: jitter_metric.py:131 `true_labels[400:480,0]` 정의부 부재)

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

> 이 repo에는 backbone/neck/head/loss/dataset/train loop가 없으므로, 해당 항목은 "해당 없음(코드 부재)"으로 명시하고, 실제 존재하는 후처리·메트릭 함수들을 함수 단위로 정밀 분석한다.

### 3.0 부재 확인 (CV/DL 표준 모듈)
| 표준 모듈 | 존재 여부 | 근거 |
|---|---|---|
| Backbone (CNN/ViT/SSM 등) | **없음** | torch import 부재 (method.py:1-16) |
| Neck/FPN | **없음** | 동일 |
| Head (회귀/세그) | **없음** | 동일 |
| Loss function | **없음** (대신 평가 score) | jitter_metric.py:49-53는 학습 loss가 아닌 평가 지표 |
| Dataset/DataLoader | **없음** | h5/csv를 직접 read (method.py:28-36, 82) |
| Train loop | **없음** | optimizer/backward 호출 0건 |
| Inference loop | 후처리 루프만 존재 | method.py:180-218 (모델 forward 아님) |

### 3.1 `method.py` — 이벤트→프레임 변환 및 I/O 유틸

- `events_to_frame(events, width, height)` (method.py:19-23): 흰 배경(255) 위에 이벤트 좌표를 0(검정)으로 찍는 **단순 누적 프레임(event frame)** 생성. polarity는 무시(존재만 표시). voxel/time-surface가 아닌 **2D binary accumulation frame**. (근거: method.py:20-22)
- `event_file_to_array(file_name)` (method.py:28-36): `.h5`의 `events`를 읽어 `(t,x,y,p)` dtype으로 캐스팅. **polarity를 0/1 → -1/+1로 변환 후 다시 -1을 0으로 되돌림** (method.py:33-35) — 결과적으로 polarity는 0/1로 남으며 33행 변환이 35행에서 상쇄되는 **불필요/혼란 코드**(코드 스멜).
- `extract_numbers`, `combine_csv_files` (method.py:38-58): 파일명 숫자 기준 정렬 후 CSV 병합 유틸. 제출 파일 합치기용. 핵심 알고리즘 아님.

### 3.2 `local_frequency_variance` (method.py:60-64) — STFT 기반 주파수 분산
- `scipy.signal.stft`로 좌표 신호의 단시간 푸리에 변환 → power spectrum의 분산을 시간축으로 반환. 적응형 윈도우 결정의 "frequency" 모드에서 motion 변동성 척도로 사용. fs 기본 30Hz. (근거: method.py:62-64)

### 3.3 `adaptive_smoothing` (method.py:66-151) — **기여 (i): Motion-Aware (적응형) Median Filtering** [핵심]
입력 CSV의 `x_smooth`, `y_smooth` 컬럼을 읽어(method.py:83-84) **국소 운동 변동성(motion variance)에 따라 median 필터 윈도우 크기를 점마다 동적으로 바꾸는** 적응형 median smoothing.

- motion variance 추정 방식 5종(인자 `method`로 선택):
  - `"raw"` / `"velocity"`: 인접 프레임 좌표 차분의 L2 norm = 속도. rolling mean으로 평활. (method.py:86-98)
  - `"acceleration"`: 2차 차분(가속도) 기반. (method.py:100-105)
  - `"covariance"` (기본값): rolling 윈도우 내 공분산. (method.py:107-114)
  - `"frequency"`: 3.2의 STFT 분산. (method.py:116-131)
- 적응형 윈도우 산출: variance를 `[min_window, max_window]`(기본 5~20)로 clip → rolling percentile(기본 75)로 점별 윈도우 결정. (method.py:136-142)
- 점별로 서로 다른 윈도우 크기 `w`로 median 적용 (method.py:145-148). **주의: 매 점마다 전체 Series에 rolling median을 다시 계산하고 인덱스 i 하나만 취함 → O(N²) 비효율**(코드 스멜, 성능 이슈).
- 최종 좌표를 ÷8 스케일로 저장. (method.py:150-151)
- 설계 의도: blink(눈 깜빡임)으로 인한 좌표 스파이크는 큰 윈도우로 억제하고, 빠른 saccade(자연 시선 이동)는 작은 윈도우로 보존하는 **blink-spike 억제 + 자연 동역학 보존** 트레이드오프. (README.md:10 "suppresses blink-induced spikes while preserving natural gaze dynamics"와 일치)

### 3.4 `post_process_pupil_coordinates_optical_flow` (method.py:153-221) — **기여 (ii): Optical Flow 기반 Local Refinement** [핵심]
median 필터로 1차 평활한 좌표를, **국소 ROI 내 이벤트의 누적 이동 방향(근사 optical flow)에 맞춰 ±1픽셀씩 미세 정렬**한다.

라인 단위 흐름:
1. 예측 좌표 ×8 복원 (method.py:166-168).
2. 이벤트 총 지속시간을 예측 개수로 나눠 프레임당 시간 간격 `time_step` 산출 (method.py:170-172).
3. 윈도우 20의 median으로 1차 평활 (method.py:174-176).
4. 각 예측점마다:
   - 현재 시간창 `[prev_timestamp, timestamp]` 결정 (method.py:181, 200).
   - **적응형 ROI 크기**: 최근 5점 평균 대비 변화가 크면(빠른 이동) ROI 확대(15×8), 작으면 축소(8×8) (method.py:185-192). → 운동 인지(motion-aware) 요소.
   - ROI(공간) ∩ 시간창 내 이벤트만 추출 (method.py:194-198).
   - 이벤트 수가 임계(`10*scale=80`) 초과 시: 인접 이벤트 좌표 차분을 누적해 `(dx, dy)` 산출 → 정규화 후 부호만 취해 **±1픽셀 shift** 적용 (method.py:202-218). 이것이 "simplified optical flow approximation"(method.py:203 주석).
5. 결과 ÷8 저장 (method.py:220-221).

- **주의**: 이벤트 마스킹(method.py:194-198)이 매 예측점마다 전체 이벤트 배열을 boolean 인덱싱 → 이벤트 수 N, 예측 수 M에 대해 O(N·M)으로 **대규모 이벤트에서 매우 느림**(성능 병목, FPGA 이식 시 핵심 고려점).
- optical flow는 Lucas-Kanade/Farneback 같은 정식 알고리즘이 아니라 **이벤트 좌표 1차 차분 누적의 부호**라는 극단적 단순 근사. (method.py:206-216)

### 3.5 메인 실행부 (method.py:223-228)
- 하드코딩된 11개 파일명 리스트(`'1_1'...'12_4'`)에 대해 `event_file_to_array` → `post_process_pupil_coordinates_optical_flow` 순차 실행. (method.py:223-227)
- `adaptive_smoothing`는 정의되어 있으나 **메인 루프에서 호출되지 않음**(method.py:225-227은 optical_flow 함수만 호출) → 두 기여가 파이프라인으로 결합된 코드는 이 repo에서 **확인 불가**(논문상 결합, 코드상 분리 실행 추정).
- `combine_csv_files` 호출은 주석 처리됨 (method.py:228).

### 3.6 `jitter_metric.py` — **기여 (iii): Jitter Metric (평가)** [핵심]
gaze 궤적의 **시간적 부드러움**을 정량화하는 신규 메트릭. 공간 정확도(p-error/거리)만으로 못 잡는 jitter를 보완. (README.md:10 "novel Jitter Metric ... velocity regularity and local signal complexity"와 일치)

- `velocity`, `velocity_series` (jitter_metric.py:7-11): gradient/차분 기반 속도.
- `comparative_velocity_entropy(pred, gt)` (jitter_metric.py:13-34): pred·gt 속도 분포를 히스토그램(Freedman-Diaconis bin)으로 만들고 **KL divergence → log1p 정규화**. 예측 속도 분포가 GT와 얼마나 다른지(=비자연적 jitter) 측정. (jitter_metric.py:32-33)
- `sparc_1d` (jitter_metric.py:36-42): 속도의 FFT 스펙트럼 기반 **SPARC(Spectral Arc Length, 운동 부드러움 지표)** 1D 버전. (DC 성분 제외, jitter_metric.py:41)
- `comparative_sparc` (jitter_metric.py:44-47): pred/gt SPARC 상대 차이.
- `compute_combined_score` (jitter_metric.py:49-53): **0.75·SPARC + 0.25·CVE** 가중합 최종 점수. (jitter_metric.py:52)
- `add_composite_noise` (jitter_metric.py:55-97): Gaussian + blink 스파이크 + 상수 shift + 50Hz 사인 jitter를 합성하는 **노이즈 시뮬레이터**(메트릭 검증/시연용).
- `generate_prediction` (jitter_metric.py:99-126): 목표 MSE·목표 noisiness를 동시에 만족하는 가짜 예측을 **랜덤 탐색(최대 1e6회)**으로 생성 — 메트릭의 분별력을 보이기 위한 합성 케이스(a~g, jitter_metric.py:131-159).
- **주의**: `true_labels`(jitter_metric.py:131)가 미정의 → 이 스크립트는 그대로 실행 불가, 데모/실험 노트 성격. (확인: 정의부 부재)

---

## 4. 알고리즘 / 데이터 표현

### 4.1 이벤트 표현
- **누적 binary frame** (`events_to_frame`, method.py:19-23)이 유일한 명시적 프레임화. **Voxel grid / time-surface는 코드에 없음**(확인 불가). 다만 실제 refinement 단계는 프레임이 아니라 **raw 이벤트 좌표 자체를 ROI/시간창으로 필터링**해 사용(method.py:194-198) → "이벤트 표현"보다는 "이벤트 스트림 직접 질의" 방식.

### 4.2 신경망 구조 (ConvLSTM/SSM/Transformer/Low-rank)
- **모두 코드 부재 → 확인 불가/해당 없음.** torch/tensorflow import 0건(method.py:1-16, jitter_metric.py:1-5). 베이스라인 모델(예: "bigBrains" 제출물, method.py:227 경로명)은 외부 산출물이며 이 repo는 그 출력만 받는다.
- 따라서 "LoRiN = Low-Rank/Recurrent" 가설은 **이 repo 코드로는 입증 불가**. repo명일 뿐 구조와 무관(추정).

### 4.3 후처리 알고리즘 요약
| 단계 | 알고리즘 | 라인 |
|---|---|---|
| 1차 평활 | 고정 윈도우(20) median filter | method.py:174-176 |
| 적응형 평활 | motion-variance 적응형 윈도우 median | method.py:136-148 |
| blink 억제 | 큰 윈도우 median으로 스파이크 제거 | 3.3, README.md:10 |
| 공간 정렬 | 이벤트 차분 누적 부호 기반 ±1px shift (근사 optical flow) | method.py:202-218 |
| 평가 | velocity-entropy(KL) + SPARC 결합 score | jitter_metric.py:49-53 |

### 4.4 메트릭
- 신규: **Jitter Metric** = 0.75·SPARC + 0.25·CVE (jitter_metric.py:52).
- 전통 메트릭(p-error / 거리 / IoU)은 **이 repo 코드에 구현 없음**. 챌린지 자체는 p-error(픽셀 거리) 사용으로 알려져 있으나(추정), repo에서는 미구현(확인 불가).

---

## 5. 학습 · 평가

### 5.1 데이터셋
- CVPR 2025 Event-based Eye Tracking Competition(3ET+) 데이터로 추정. (근거: README.md:4 챌린지 링크) repo에는 데이터 미포함.
- 입력 파일 11개 식별자 하드코딩: `1_1 ~ 12_4` (method.py:223) — 피험자_세션 형식 추정.

### 5.2 학습
- **학습 코드 없음.** 이 방법은 정의상 inference-time, retraining-free. (README.md:10)

### 5.3 평가 / 실행 명령어
- 후처리 실행: `python method.py` (method.py:225-227 메인 루프 자동 실행). 단, `./event_files/`, `./original_bigBrains/`, `./refined/` 디렉토리·파일이 사전 준비되어야 함.
- 메트릭 데모: `python jitter_metric.py` — 단, `true_labels` 미정의로 **그대로는 실패**, 외부에서 GT 배열 주입 필요(jitter_metric.py:131).
- CLI 인자 파서(`argparse` import, method.py:8)는 import만 되고 **실제 인자 정의·사용 없음**(코드 스멜).

---

## 6. 의존성

- method.py: `h5py, os, pandas, re, cv2(OpenCV), glob, io.BytesIO, argparse, struct, sys, matplotlib, numpy, PIL.Image, scipy.signal` (method.py:1-16). 이 중 다수(`BytesIO, struct, sys, matplotlib, PIL, argparse, glob, cv2`)는 **실제로 사용되지 않는 죽은 import**(핵심 로직은 numpy/pandas/scipy만 사용).
- jitter_metric.py: `numpy, matplotlib, scipy.signal.savgol_filter, scipy.stats.entropy, numpy.fft` (jitter_metric.py:1-5). `savgol_filter`, `matplotlib`는 사용 안 됨(죽은 import).
- **딥러닝 프레임워크 의존성 0** (torch/tf 없음). requirements.txt/setup.py/pyproject.toml **없음** → 재현 환경 명세 부재(한계).

---

## 7. 강점 · 한계

### 7.1 강점
- **재학습 불요, 모델 비종속**: 어떤 베이스라인의 CSV 출력에도 적용 가능(model-agnostic). 실전 가치 높음. (README.md:10)
- **운동 인지 적응성**: motion variance에 따라 median 윈도우·ROI를 동적 조절 → saccade 보존하며 blink 스파이크 억제. (method.py:136-142, 185-192)
- **신규 평가축 제시**: 공간 정확도가 같아도 jitter가 다른 케이스를 분별하는 SPARC+엔트로피 메트릭. (jitter_metric.py, 합성 케이스 a~g로 시연)
- 알고리즘이 NumPy/scipy 수준이라 **이식·경량화에 매우 유리**(딥러닝 가속기 없이도 구현 가능).

### 7.2 한계 / 코드 품질 이슈
| 심각도 | 위치 | 문제 |
|---|---|---|
| High(성능) | method.py:145-148 | 점별 rolling-median을 전체 Series에 반복 계산 → O(N²) |
| High(성능) | method.py:194-198 | 예측점마다 전체 이벤트 boolean 인덱싱 → O(N·M) |
| Medium(정확성) | method.py:33-35 | polarity -1/+1 변환을 곧바로 0으로 되돌림(상쇄, 무의미) |
| Medium(재현성) | repo 전역 | requirements/seed/CLI 인자 부재, 경로·파일명 하드코딩 |
| Medium(실행성) | jitter_metric.py:131 | `true_labels` 미정의 → 스크립트 단독 실행 불가 |
| Low(청결성) | method.py:1-16 등 | 다수의 미사용 import(argparse/struct/PIL/matplotlib 등) |
| Low(설계) | method.py:225-227 | `adaptive_smoothing`(기여 i)가 메인에서 미호출 → 논문 파이프라인 결합 코드 부재 |
- optical flow 근사가 **±1픽셀 단위**라 미세 정렬 효과가 제한적이고 ROI 통계의 노이즈에 민감(method.py:215-218).

---

## 8. 우리 프로젝트 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)

> 전제: 우리 프로젝트 목표를 "XR 시선추적의 FPGA 저지연 on-device 가속"으로 추정.

1. **후처리 블록의 HW 친화성이 높다.** 이 repo의 핵심(median filter, 차분/누적, 부호 결정)은 곱셈이 거의 없는 정수·비교 연산 위주 → **FPGA에 매우 적합**. median 필터는 정렬 네트워크(sorting network)로, ROI 누적은 시간창 슬라이딩 윈도우 + 누산기로 RTL/HLS 매핑 용이. (근거: method.py:174-176, 202-218)
2. **저지연 관건은 알고리즘 복잡도다.** Python 구현의 O(N²)·O(N·M)(method.py:145-148, 194-198)는 SW 비효율일 뿐, FPGA에서는 **스트리밍 처리(이벤트 도착 즉시 ROI 누산, 고정 크기 윈도우 시프트 레지스터)**로 재설계하면 O(1)/이벤트로 바꿀 수 있다 → on-device 실시간화의 핵심 리팩토링 포인트.
3. **양자화·경량화 부담이 거의 없다.** 딥러닝 가중치가 없으므로 PTQ/QAT 불필요. 좌표는 정수 픽셀, median 윈도우·ROI는 작은 정수 상수(5~20, 8~15) → **고정소수점/정수 RTL로 바로 이식 가능**(method.py:137, 185-192).
4. **메트릭(SPARC/엔트로피)은 평가 단계라 on-device 불필요**하지만, jitter를 줄이는 목적함수로서 **HW 후처리 파라미터(윈도우 크기·ROI·shift) 튜닝의 평가 기준**으로 활용 가능(jitter_metric.py:49-53).
5. **베이스라인 모델 가속과 분리 설계.** 무거운 부분은 이 repo가 아닌 외부 gaze 모델(추정: CNN/RNN 계열)이므로, FPGA 가속은 (a) 베이스라인 추론 가속(별도 repo 대상) + (b) 본 repo의 경량 후처리 파이프라인(시프트 레지스터 기반)으로 **2단 파이프라인** 구성이 자연스럽다.
6. **권장 이식 우선순위**: ① 고정 윈도우 median(method.py:174-176) → 정렬 네트워크 RTL ② ROI 이벤트 누산 + 부호 shift(method.py:202-218) → 스트리밍 누산기 ③ 적응형 윈도우 결정(method.py:136-142) → 룩업/임계 비교 로직.

---

## 9. 근거 표기 요약

- **코드로 확인됨**: 후처리 2종(method.py), Jitter Metric(jitter_metric.py), 입출력 형식(CSV/h5), 센서 크기·스케일, 딥러닝 프레임워크 부재, 미사용 import, 성능 이슈.
- **추정(코드 외 근거/관례)**: 데이터셋이 3ET+ 챌린지셋, 베이스라인이 CNN/RNN 계열, "피험자_세션" 파일명 의미, 챌린지 메트릭 p-error, 우리 프로젝트 목표.
- **확인 불가(코드 부재)**: backbone/neck/head/loss/dataloader/train·infer(모델) 루프, ConvLSTM/SSM/Transformer/Low-rank 구조, voxel/time-surface 표현, 세그멘테이션 출력, p-error/IoU 구현, "LoRiN" 명칭의 아키텍처적 의미.

---

## 부록: 참고 파일

- `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\EyeLoRiN\README.md`
- `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\EyeLoRiN\method.py`
- `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\EyeLoRiN\jitter_metric.py`
