# cb-convlstm-eyetracking 정밀 분석 (3ET: Change-Based ConvLSTM Eye Tracking)

> 분석 기준 경로: `REF/XR-Eye-Tracking/Codebase/cb-convlstm-eyetracking/`
> 분석 도구: Glob/Grep/Read (실제 코드 라인 근거). 추론은 "추정"/"확인 불가" 명기.
> 비고: gg_ssms/eye_tracking_lpw의 데이터/학습 파이프라인 원본(upstream)이며, 두 repo의 Dataset 코드는 거의 동일.

---

## 1. 개요

- **목적**: 이벤트 카메라 스트림으로부터 **ConvLSTM**을 이용해 동공 중심(pupil center)을 검출. 시간적 희소성(change-based)을 활용해 경량·저전력 추론 지향.
- **원논문/챌린지**: *3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network*, Chen, Wang, Liu, Gao, **IEEE BioCAS 2023** (arXiv:2308.11771). README L11-22. 3ET 데이터셋 공개(Tonic 로더 지원, README L47-65).
- **입출력**:
  - 입력 = **이벤트 프레임 시퀀스** `[B, T, C=1, H=60, W=80]` (이벤트를 프레임으로 누적, 그레이스케일). `convlstm-et-pytorch-event.py` L21-22, EventDataset L97-113.
  - 출력 = **동공 중심 좌표 (x, y)** 정규화 회귀, `[B, T, 2]` (forward L247-249, fc2→2).
  - 세그멘테이션 없음. 좌표 회귀만. 확인됨.
- **데이터셋**: SEET(Synthetic Event-based Eye Tracking, event frame/raw), LPW(원본 비이벤트), 3ET(Tonic). README L24-45.

---

## 2. 디렉토리 구조

```
cb-convlstm-eyetracking/
├── README.md
├── neuromorphic_eye.gif
└── eyetracking-convlstm/
    ├── convlstm-et-pytorch-event.py  ★ 메인 학습 스크립트 (Dataset/MyModel/Train/Val/Plot)
    ├── convlstmbak.py                ★ 표준 ConvLSTM (메인이 import: L9)
    ├── convlstm_cell.py              ★ change-based ConvLSTM cell + sparse-rate 계측
    ├── convlstm.py                   # 전체 ConvLSTM(cell 포함) 변형
    ├── convlstm_delta.py             # delta(변화량) 기반 변형
    ├── convlstm_sp.py                # sparse 변형
    ├── process_event.py             ★ 원시 h5 → 슬라이딩 윈도우 h5 전처리
    ├── train_files.txt / val_files.txt / files.txt
    ├── checkpoint.pth                ── [제외] 체크포인트
    ├── plot/ (logo, event_plot, neuromorphic_eye.gif)
    └── log/training_log.txt
```

★ = 정밀 분석 파일.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 표준 ConvLSTM — `convlstmbak.py` (메인이 실제 사용)

`convlstm-et-pytorch-event.py` L9가 `from convlstmbak import ConvLSTM`로 임포트하는 실제 모델 본체.

- **`ConvLSTMCell` (L5-60)**:
  - 생성자(L7-36): 입력+은닉 채널을 받아 `Conv2d(in+hidden → 4·hidden, kernel, padding=k//2)`(L32-36). 게이트 4개(i,f,o,g)를 한 conv로 계산.
  - forward(L38-55): `combined = cat(input, h_cur)`(L41) → **`relu(conv(combined))`**(L45, 표준 ConvLSTM과 달리 conv 출력에 ReLU 적용) → split into cc_i/f/o/g(L46) → `i=σ, f=σ, o=σ, g=tanh`(L47-50) → `c_next = f·c_cur + i·g`, `h_next = o·tanh(c_next)`(L52-53). **상태 전이**: 셀 상태 c와 은닉 h를 표준 LSTM 게이팅으로 갱신하되 공간 conv 사용.
  - `init_hidden` (L57-60): (h, c) 0 초기화.
- **`ConvLSTM` (L63-193)**: 다층 스택 컨테이너. batch_first 지원(L136-138), 각 layer마다 cell 생성(L111-120), 시퀀스 길이 T를 시간축으로 순회(L160-163)하며 h 누적. `return_all_layers`로 전 레이어/마지막 레이어 출력 선택(L171-174).

### 3.2 Change-Based ConvLSTM cell — `convlstm_cell.py`

논문의 "change-based" 핵심 아이디어를 구현한 변형(메인 스크립트가 직접 쓰진 않으나 논문 기여의 핵심).

- **`ConvLSTMCell` (L13-120)**:
  - 생성자: 표준과 동일 conv(L40-46) + **추가 `BatchNorm2d(hidden_dim)`**(L47).
  - forward(L49-94): cur_state가 `(h_cur, c_cur, h_pre)` **3-튜플**(이전 은닉 h_pre 추가) → `delta = h_cur`(L51) → `combined = cat(input_tensor, delta)`(L54-56). 주석(L52-53)에 threshold로 작은 변화 제거(`delta < threshold → 0`)하는 change-based 희소화 코드가 비활성으로 존재.
  - **희소율 계측(eval 모드, L57-71)**: `combined`/`input`/`delta`의 0 비율(sparse_rate)을 로그 파일에 기록. → change-based 입력이 실제로 얼마나 희소한지(연산 절감 가능성) 측정용 instrumentation.
  - 나머지 게이팅은 표준과 동일(L81-92). 반환은 `(h_next, c_next, h_cur)` — h_cur를 다음 step의 h_pre로 전달.
- **요지**: 입력에 "현재 은닉 상태(=직전까지의 변화 누적)"를 delta로 concat하고, 작은 변화를 0으로 잘라 **희소 연산**을 유도. 이벤트 데이터의 시간적 희소성을 ConvLSTM 단계까지 전파하는 것이 논문 핵심.

### 3.3 메인 학습/모델 — `convlstm-et-pytorch-event.py`

- **하이퍼파라미터**(L21-28): height=60, width=80, batch_size=16, seq=40, stride=1(train)/40(val), chunk_size=500, epochs=100, seed=1.
- **`normalize_data` (L44-61)**: 프레임 z-score 정규화.
- **`create_samples` (L63-83)**: chunk(500) 단위 슬라이딩 윈도우(길이 seq, stride).
- **`EventDataset` (L86-127)**:
  - `__getitem__` (L97-113): h5 `file.root.vector`에서 sample 읽어 `cv2.resize(80,60)`+정규화 → `[seq,1,H,W]`. 라벨 `label1=x/M/8, label2=y/N/8`(L109-110) → `[seq,2]`.
  - `_concatenate_files` (L115-127): 라벨 txt `lines[3::4]`(4줄마다 1개) 추출 → 윈도우화.
  - **gg_ssms/eye_tracking_lpw의 EventDataset과 사실상 동일** → cb-convlstm이 upstream 원본임을 확인.
- **`MyModel` (L164-249)** — 4-stage ConvLSTM CNN:
  - `convlstm1`(1→8) → BN3d → ReLU → MaxPool3d(1,2,2) (L168-171, L201-205).
  - `convlstm2`(8→16), `convlstm3`(16→32), `convlstm4`(32→64), 각 BN3d+ReLU+MaxPool3d(1,2,2)로 **공간만 2× 다운샘플**(시간 유지)(L176-227).
  - head: 시간 step별로 flatten → `fc1(960→128)+ReLU+Dropout0.5+fc2(128→2)`(L195-197, L238-249). 출력 `[B,T,2]`.
  - **구조(backbone/neck/head)**: backbone = ConvLSTM×4(시공간 동시), neck = 시간축 유지 MaxPool3d, head = 프레임별 MLP 회귀.
- **학습 루프(L280-356)**: criterion=`SmoothL1Loss`(L262), optimizer=Adam(lr=1e-3)(L263), 100 epoch. thop(profile) import로 FLOPs 측정 가능(L16).
- **검증·메트릭(L308-344)**: `dis[...,0]*=height, dis[...,1]*=width` → `dist=norm` → `dist > {1,3,5,10}` 초과 비율 err_rate 기록(L327-337). training_log.txt 저장(L342-344). best val loss 시 checkpoint.pth 저장(L346-356).
- **플롯(L359-428)**: 좌표 시계열, 프레임 위 예측점 오버레이.

### 3.4 전처리 — `process_event.py`

- height=180, width=240(원해상도), chunk_size=500(L17-20).
- `create_samples` (L22-39): chunk 단위 슬라이딩 윈도우 생성(메인과 동일).
- **`get_data` (L46-58)**: 원시 h5(`h5_file.root.vector`)를 읽어 윈도우화한 뒤 blosc 압축(complevel=5)으로 새 h5 저장(L56-58).
- 입력 `data_ts_500` → 출력 `data_ts_pro/{train,val}`(L60-62). seq=40, stride=1(train)/40(val)(L71-74).

---

## 4. 알고리즘 / 데이터 표현

- **이벤트 표현**: 이벤트를 시간 슬라이스로 누적한 **이벤트 프레임 시퀀스**(80×60, 1채널). chunk(500) 내 슬라이딩 윈도우(40프레임)로 시퀀스 샘플 구성.
- **시공간 모델링**: **ConvLSTM cell**(conv 게이트 + 셀/은닉 상태)을 4단 적층. 공간은 conv·pool로, 시간은 LSTM 재귀로 처리.
  - 상태 전이: `c_t = f_t·c_{t-1} + i_t·g_t`, `h_t = o_t·tanh(c_t)`, 게이트는 `Conv2d(cat(x_t, h_{t-1}))`로 산출.
  - change-based 변형(`convlstm_cell.py`): 입력에 delta(은닉) concat + threshold 희소화 + eval 희소율 로깅.
- **후처리**: 정규화 좌표를 픽셀 환산하여 거리 기반 err_rate 산출. (median_filter import는 있으나 학습 루프에서 직접 사용 안 함 — **확인됨**.)

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: SEET(event frame), 3ET(Tonic 로더, `tonic.datasets.ThreeET_Eyetracking`, README L59-65). LPW 원본은 비이벤트(README L44-45).
- **메트릭**: `dist > {1,3,5,10}px` 초과 비율(err_rate, p-error 보완). 작을수록 우수. (논문 본문은 distance/검출율 사용 — README는 28 epoch 후 좌표 예측 그림 제시 L38-40.)
- **명령어**(README L30-37):
  1. SEET 다운로드 → `/DATA/`.
  2. `cd eyetracking-convlstm`.
  3. `python process_event.py` (seq 파라미터로 시퀀스 길이 조정).
  4. `python convlstm-et-pytorch-event.py`.
- **Tonic 로더**(README L54-65): `pip install tonic --pre` 후 `ThreeET_Eyetracking(save_to, split)`로 raw event+label 자동 다운로드.

---

## 6. 의존성

- torch, torchvision, numpy, opencv-python(cv2), pandas, tables(PyTables/h5), tqdm, matplotlib, scipy(median_filter), thop(FLOPs).
- (선택) tonic(3ET 로더).
- **CUDA 커스텀 커널 없음** → 순수 PyTorch. 이식성 측면 최상.

---

## 7. 강점 / 한계 / 리스크

- **강점**:
  - **순수 PyTorch, 커스텀 커널 무의존** → 이식/포팅 용이(CPU/엣지/FPGA HLS 출발점으로 최적).
  - ConvLSTM은 잘 정립된 정형 연산(conv+elementwise)으로 HW 매핑이 단순.
  - change-based 희소화 + 희소율 계측으로 저전력 가능성 정량화.
  - 작은 입력(80×60), 적은 채널(8/16/32/64) → 경량.
- **한계/리스크**:
  - change-based threshold 희소화가 메인 모델(`convlstmbak`)엔 미적용 — 실제 학습은 표준 ConvLSTM. 희소화 효과는 별도 cell(`convlstm_cell.py`)에서 측정만 — **확인됨**.
  - `relu(conv(...))`(convlstmbak L45) 적용으로 표준 ConvLSTM과 다름(게이트 입력에 ReLU). 의도/영향 문서화 없음 — **추정**.
  - 라벨 `/8` 정규화(L109-110) 근거 주석 없음 — **추정**(다운샘플 스케일 보정).
  - SmoothL1만 사용, 폐안(closed-eye) 처리 없음(EventMamba와 대비).
  - 코드가 스크립트형(전역 실행) — 모듈화/재사용성 낮음.

---

## 8. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 가속, 추정)

- **FPGA 매핑 적합성 최상**: ConvLSTM은 conv + elementwise sigmoid/tanh + 셀 상태 누적으로 구성 → **정형 데이터플로**로 HLS/RTL 매핑이 세 repo 중 가장 직접적. 동적 그래프(gg_ssms)·FPS(EventMamba) 같은 불규칙 연산 없음.
- **change-based 희소성 활용**: `convlstm_cell.py`의 threshold 희소화를 HW에서 zero-skipping(0 입력 MAC 생략)으로 구현하면 저전력 가능. 희소율 로깅 자산이 정량 근거 제공.
- **저지연/경량**: 80×60×1 입력, 채널 8~64, seq 40 → 작은 모델. on-device XR 시선추적의 강력한 baseline. 양자화(INT8) 적용 시 추가 경량화 여지(본 repo엔 양자화 코드 없음 → gg_ssms/retina의 lsq/dorefa 재활용 검토).
- **데이터 파이프라인 재사용**: gg_ssms/eye_tracking_lpw와 동일 Dataset/전처리 → 동일 입력으로 ConvLSTM vs GG-SSM vs Mamba **정확도/지연/리소스 직접 비교** 가능(공정 벤치마크 기반).
- **권장 활용**: FPGA 1차 타깃 모델로 ConvLSTM 채택 → systolic conv array + 셀상태 SRAM + 희소 스킵 → 정확도-리소스 곡선의 reference point로 사용. GG-SSM/Mamba는 정확도 상한 비교군 — **추정**.

---

## 9. 근거 표기

- 모델/cell/학습/전처리: 실제 코드 Read(파일:라인)로 확인.
- "확인됨": 메인이 convlstmbak 사용(L9), change-based 희소화는 별도 cell에서만, median_filter 미사용.
- "추정": convlstmbak의 conv-ReLU 의도, `/8` 라벨 정규화 의미, FPGA reference 전략.
- "확인 불가": 논문 본문 정량 수치(메트릭은 README/코드 err_rate 기준으로 기술), checkpoint.pth 내부(제외).
