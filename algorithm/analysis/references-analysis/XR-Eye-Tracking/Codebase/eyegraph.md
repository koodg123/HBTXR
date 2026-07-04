# EyeGraph 코드베이스 정밀 분석

> 기준 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\eyegraph\`
> 분석 도구: Glob/Grep/Read 만 사용 (bash 미사용). 라인 근거는 `파일명:라인` 형식.

---

## 1. 개요

- **목적**: **이벤트 카메라(DVS) 기반 연속 시선/동공 추적**을 위한 데이터셋 + 비지도 그래프 클러스터링 벤치마크. 핵심 방법은 ① GMM으로 동적 그래프 구성 → ② **modularity-aware(모듈성 인식) 그래프 클러스터링**으로 동공 이동 추적. (`README.md:9-10`, `EyeGraph_DatasetDetails.md:6`)
- **원논문/챌린지**: Bandara et al., "EyeGraph: Modularity-aware Spatio Temporal Graph Clustering for Continuous Event-based Eye Tracking", **NeurIPS 2024** (Datasets & Benchmarks Track) (`README.md:1, 37-44`).
- **입력**: **이벤트 스트림**(DAVIS346 DVS, 해상도 346x260 — `noise_analysis.py:8`). aedat4 포맷(`gt_annotator.py:20`, `dv_processing` 사용). 보조로 grayscale 프레임·Pupil-Core 기준데이터(다중모달, `README.md:24, 27`).
- **출력**: **동공(pupil) 좌표 추적** (그래프 클러스터링으로 동공 영역 군집 → 중심). (`README.md:21` Tracking End Goal=Pupil, Representation=Graph, Learning=unsupervised)
- **차별점**: 기존 이벤트 시선추적 대비 비지도·그래프 표현·실환경(조명변화·이동성·머리움직임 허용) 40명 데이터 (`README.md:19-31`).

---

## 2. 디렉토리 구조

### 자체 소스 (분석 대상)
```
eyegraph/
├── README.md                      # 논문 초록 + 데이터셋 비교표
├── EyeGraph_DatasetDetails.md     # 데이터 수집 프로토콜 / 파일 구조 상세
├── LICENSE.txt
├── resources/
│   ├── EyeGraph_overview.png
│   └── Readme.md
└── src/
    ├── Readme.md                  # "EyeGraph method codes" (1줄)
    ├── noise_analysis.py          # 이벤트 노이즈 필터링 분석 (DVS)
    ├── gt_annotator.py            # 수동 GT 동공 좌표 라벨링 도구
    ├── pupil_gaze_analysis.py     # Pupil-Core CSV 기반 시선/동공 통계·시각화
    └── sample_interactive_graph.html
```

### 제외/부재 항목 (중요 발견)
- `.git/` 내부 — 제외.
- **★ 핵심 발견 — 논문의 핵심 알고리즘(GMM 그래프 구성 + modularity-aware 그래프 클러스터링) 소스가 본 repo에 부재.** `src/`에는 ① 이벤트 노이즈 필터 분석, ② GT 수동 주석 도구, ③ Pupil-Core 데이터 시각화 스크립트만 존재. `src/Readme.md`는 "EyeGraph method codes" 한 줄(`src/Readme.md:1`)이지만 실제 GMM/그래프/클러스터링 구현 파일은 존재하지 않음(확인: Glob 결과 `src/*.py` = 3개뿐). 본 repo는 사실상 **데이터셋 공개 + 전처리/분석 유틸리티** 저장소이며, 방법론 재현 코드는 미포함(별도 미공개 추정).

---

## 3. 핵심 모듈·파일별 정밀 분석

> 실제 존재하는 3개 .py만 정밀 분석. (논문 방법론 코드는 부재하여 분석 불가)

### 3.1 `src/noise_analysis.py` — 이벤트 노이즈 필터 비교 (1~173)

- **목적**: DVS 이벤트 스트림에 3종 노이즈 필터를 적용해 reduction factor(폐기 이벤트 비율) 비교.
- **입출력/처리**:
  - `dv_processing`(iniVation DV SDK)로 aedat4 읽기(`MonoCameraRecording`, `:23, 55`). 스트림 가용성(event/frame/IMU/trigger) 점검(`:29-52`).
  - 디렉토리 내 `.aedat4` 순회(`:17-19`), `getNextEventBatch()`로 배치 단위 이벤트 획득(`:64`).
  - 필터 3종:
    1. **`BackgroundActivityNoiseFilter`** (backgroundActivityDuration=100ms) (`:76`) → `noise_1`.
    2. **`FastDecayNoiseFilter`** (halfLife=100ms, subdivisionFactor=4, noiseThreshold=1.0) (`:122-125`) → `noise_2`.
    3. **`RefractoryPeriodFilter`** (100ms) (`:162`) → `noise_3`.
  - 각 필터의 `getReductionFactor()` 또는 잔존 비율을 누적 후 평균 출력(`:108, 150, 173`).
  - `EventVisualizer`로 입력/출력 이벤트 프레임 시각화(`:90-102`).
- **역할**: 이벤트 전처리(노이즈 억제) 파라미터 탐색용 실험 스크립트. 하드코딩 경로(`:10, 106`)·`cv.waitKey()` 블로킹(`:102`) → 대화형 분석 도구 성격.

### 3.2 `src/gt_annotator.py` — 동공 GT 수동 주석 도구 (1~106)

- **목적**: 이벤트 프레임을 사람이 보고 **동공 중심을 클릭**해 ground-truth 좌표(JSON) 생성.
- **처리 흐름**:
  - aedat4 읽기 + `EventVisualizer` 색상 설정(배경 흰색/+파랑/−회색) (`:24-27`).
  - 이벤트 배치를 이미지로 누적(`generateImage`), 180° 회전(`:48-50`), 이벤트 수 1000 초과 프레임만 처리(`:46`).
  - 마우스 콜백 `on_mouse_click`(`:12-16`): 좌클릭 좌표 2회 수집(중심+반경 산정용). 첫 클릭을 중심으로, 반경 고정 50(`:71-72`).
  - 'q'=종료, 's'=프레임 skip (`:61-66`).
  - 주석을 `{frame_number, filename, center_x, center_y, radius}`로 JSON append 저장(`:80-97`). 출력명 `annotations_{datetime}.json` (`:33`).
- **역할**: 데이터셋의 in-the-wild 이벤트 녹화에 대한 동공 좌표 GT 라벨링(데이터셋 구조의 `annotations_*.json`, `EyeGraph_DatasetDetails.md:105, 176` 대응).

### 3.3 `src/pupil_gaze_analysis.py` — Pupil-Core 데이터 분석/시각화 (1~362)

- **목적**: 기준(reference)용 **Pupil-Core 아이트래커**의 export CSV에서 시선/동공 통계 산출·플롯(논문 figure 생성용).
- **주요 함수**:
  - `cart_to_spherical` (`:24-36`): 3D gaze 벡터 → 구면좌표 `(r, θ, ψ)`. (z<0 부호 보정, `:20-22`)
  - `sphere_pos` / `sphere_pos_over_time` (`:38-61`): 시선 방향 산점도(거리 컬러맵)·시계열.
  - **시선 속도(gaze velocity)** (`:125-131, 233-237`): θ·ψ 차분의 유클리드/시간차 → deg/sec. 히스토그램(`:148-156`) — 고정/saccade 구분 근거.
  - **동공(pupil)** (`:257-360`): `pupil_positions.csv`에서 method='2d c++' 필터(`:271`), 3D 모델 수렴 위해 첫 5초 skip(`:273-275`), 좌/우안(eye_id) 분리(`:278-279`), confidence>0.6 필터(`:288-289`). 동공 지름 시계열(`:291-297`), norm_pos 시·공간 산점도(`:304-360`).
- **역할**: 이벤트 기반 추적의 **검증용 기준데이터(Pupil-Core)** 분석. 40명 subject 루프 처리(`:109-123, 161-196, 206-256`). Colab 경로(`/content/`)·하드코딩 인덱스 → 분석 노트북 성격.

---

## 4. 알고리즘/데이터 표현

- **이벤트 표현(논문)**: 비동기 이벤트 스트림을 GMM 기반 **동적 그래프(노드=eye morphology feature)**로 표현 → modularity-aware 클러스터링. (`README.md:10`, `EyeGraph_DatasetDetails.md:6`) — **단, 이 표현/클러스터링 코드는 repo에 부재**(3장 참조). 따라서 voxel/frame/SSM/ConvLSTM 등 구현 세부는 **코드로 확인 불가**.
- **repo에 실제 존재하는 데이터 처리**:
  - 이벤트 → 누적 프레임 이미지화(`EventVisualizer.generateImage`, `gt_annotator.py:48`, `noise_analysis.py:93`).
  - 이벤트 노이즈 필터링(background activity / fast decay / refractory) — DVS 전처리.
  - 기준 시선의 구면좌표·속도·동공지름 분석.
- **후처리(ellipse fitting)**: 본 repo에는 없음(GT 주석은 원/반경 50 고정, `gt_annotator.py:72`).

---

## 5. 학습/평가 파이프라인

- **학습 코드 부재**: 비지도 방법이며(`README.md:23`), 본 repo에 학습/추론 루프·모델 정의 없음. 평가 메트릭 코드(p-error/IoU 등)도 부재(확인: Glob).
- **데이터셋** (`EyeGraph_DatasetDetails.md`):
  - 40명, DAVIS346 이벤트(+grayscale 30FPS, IMU, trigger) + Pupil-Core 기준 (`:171`).
  - 3개 실험 셋업: (i) 일반 lab(머리움직임 허용), (ii) 조명변화(348 Lux↔24 Lux), (iii) 사용자 이동성 (`:28`).
  - 파일 구조: parent/child 폴더, `[subject]_[session].aedat4` + Pupil-Core export(pupil/gaze/fixation/blink CSV) + in-the-wild 녹화의 `annotations_*.json` GT (`:36-176`).
  - 데이터 접근은 신청 폼 필요(`README.md:7`).
- **실행 명령어**: 정식 학습/평가 CLI 없음. 스크립트는 경로 하드코딩 후 직접 실행(예: `noise_analysis.py`의 `directory='./userStudy_eyeTracking_Davis346'`, `:10`).

---

## 6. 의존성

- `dv_processing` (iniVation DV SDK — 이벤트/aedat4 I/O·노이즈필터·시각화) (`noise_analysis.py:1`, `gt_annotator.py:1`).
- OpenCV(cv2), numpy, pandas, seaborn, matplotlib, csv, json (`pupil_gaze_analysis.py:1-10`, `gt_annotator.py:1-4`).
- IPython.display(Colab 환경) (`pupil_gaze_analysis.py:268`).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **데이터셋 자체의 가치**: 40명, 실환경(조명/이동성/머리움직임), 다중모달(이벤트+프레임+IMU+Pupil-Core 기준), 동공좌표 GT — 이벤트 기반 시선추적 연구의 드문 in-the-wild 벤치마크 (`README.md:19-31`).
- 이벤트 노이즈 필터 3종 비교 코드 → DVS 전처리 파라미터 선정 재사용 가능.
- GT 주석 도구가 단순·재현 용이.

**한계 / 리스크**
- **★ 방법론 재현 불가**: 논문의 핵심(GMM 동적 그래프 + modularity-aware 클러스터링) 코드 부재 → 본 repo만으로 알고리즘 재현/이식 불가. (가장 큰 리스크)
- 스크립트 품질: 하드코딩 경로/인덱스(`noise_analysis.py:10`, `pupil_gaze_analysis.py:12,14`), Colab 의존, 함수화 미흡 → 재사용성 낮음.
- 평가 메트릭/정량 비교 코드 없음 — 성능 검증은 논문/supplementary 의존.
- `dv_processing`는 iniVation 전용 SDK → DAVIS 외 센서/플랫폼 이식성 제약.

---

## 8. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 가속, 추정)

> "XR 시선추적 + FPGA 가속(on-device/저지연/경량화)" 맥락은 **추정**.

- **이벤트 기반 = 저지연/저전력 본질적 강점**: DVS는 sub-µs 지연·sparse 데이터(`README.md:9`) → 이벤트 구동 회로(asynchronous/sparse compute)는 FPGA 가속과 궁합이 좋음. 프레임 누적 대신 **이벤트-스트림 파이프라인**을 HW로 구성하면 전력·지연 이점.
- **이벤트 노이즈 필터의 HW화**: background-activity / refractory-period 필터(`noise_analysis.py:76, 162`)는 픽셀별 타임스탬프 메모리 + 비교 로직으로 **FPGA에 직접 매핑 가능**(흔한 DVS 전처리 IP). 우리 가속기의 전단(front-end) 전처리로 재사용 후보.
- **그래프 클러스터링의 가속 난점**: 논문 방법(GMM+modularity 클러스터링)은 동적 그래프·반복 최적화 → FPGA 직접 가속이 어렵고 비결정적 지연. **온디바이스 대안으로는 CNN(RITnet/EllSeg)+soft-argmax 같은 결정적 dataflow가 더 적합**할 수 있음(트레이드오프 비교 근거).
- **데이터/검증 자산**: 이 데이터셋(실환경 이벤트 + 동공 GT + Pupil-Core 기준)은 우리가 설계할 경량 이벤트 추적기/FPGA IP의 **벤치마크·정량평가**에 활용 가능(접근 신청 전제).
- **표현 변환 경로**: 이벤트 → 누적 프레임(`generateImage`) → 기존 프레임 기반 경량 CNN(EllSeg/RITnet) 추론은, 이벤트 입력을 우리 FPGA CNN 파이프라인에 연결하는 **현실적 가교**가 될 수 있음.

---

## 9. 근거 표기

- **확인(코드 라인 근거)**: `noise_analysis.py`(필터 3종·DVS I/O), `gt_annotator.py`(주석 도구), `pupil_gaze_analysis.py`(시선/동공 분석) 전체. 데이터셋 구조·프로토콜(`EyeGraph_DatasetDetails.md`), 비교표(`README.md`).
- **★ 확인된 부재(중요)**: 논문 핵심 알고리즘(GMM 그래프 구성, modularity-aware 그래프 클러스터링, 학습/평가 루프, 모델) 소스가 repo에 **존재하지 않음** (Glob로 `src/*.py`=3개 전수 확인).
- **추정**: GMM/그래프/시공간 모델 세부는 논문 기반 서술이며 **코드로 확인 불가**; "FPGA 가속" 프로젝트 맥락(8장).
- **확인 불가(미열람)**: `resources/Readme.md`, `sample_interactive_graph.html` 내부 — 분석 핵심과 무관하여 제외.
