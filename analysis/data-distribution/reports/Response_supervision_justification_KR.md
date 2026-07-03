# Response — Supervision(데이터 증강) 정당성: Frame/Event Interpolation

본 절은 "왜 학습 단계에서 Frame/Event Interpolation 기반 supervision을 도입했는가"를 데이터 분포 근거로 정당화한다. 평가·GT는 보간이 아닌 원본 EV-Eye 실측에 대해서만 수행하며(아래 §4), 보간은 **학습 입력 증강**에 한정된다.

## 1. 원본 데이터 분포의 한계 (Saccade 과소표현)

### 1.1 Frame 데이터는 40ms 간격이라 saccade를 잘 포착하지 못한다
- **판정 물리량**: 동공 위치 속도(px/s) — 인접 프레임 동공중심의 중앙차분.
- **임계(Threshold)**: `speed > 493 px/s` (saccade-세션 속도분포의 90퍼센타일, I-VT).
- **시간해상도**: **25 Hz (40 ms)** → saccade(지속 30–80 ms)가 두 프레임 사이에서 시작·종료되어, 인접 프레임 사이 변위가 작아 속도가 임계를 거의 넘지 못함 → **saccade를 놓침**.

### 1.2 실제 4-state 분포 (Frame)
EV-Eye 전체(48명, 약 1.49M 프레임)에 대해 4-state(Fixation/Saccade/Smooth/Blink)를 산출한 결과, **Saccade 비율이 0.00%** 수준이다(아래 Figure).

**[Figure: Subject-wise 4-state Distributions (Frame, before interpolation)]** — `fig_split_4state_frame_before.png`

| split | Fixation | **Saccade** | Smooth | Blink |
|---|--:|--:|--:|--:|
| Train(1–36) | 63.2% | **0.00%** | 34.5% | 2.3% |
| Val(37,40,45,48) | 62.7% | **0.00%** | 35.3% | 2.0% |
| Test(37–48) | 62.9% | **0.00%** | 35.2% | 2.0% |

→ Saccade 표본이 사실상 부재하므로 **해당 state는 학습이 거의 진행되지 않는다**(극단적 클래스 불균형).

### 1.3 Event 데이터 분포
이벤트 스트림은 빠른 운동에서 이벤트가 급증하므로 saccade를 포착한다.
- (레이트-bin 방식) 원시 이벤트를 **20 ms bin** → `rate(ev/s)`, 세션중앙을 `m`이라 할 때
  **`Saccade = 4·m < rate ≤ 12·m`** (그 위 `>12·m`는 Blink), 비버스트는 세션 pattern으로 Fixation/Smooth.
  - **판정 물리량**: 이벤트 레이트(ev/s) · **임계**: `rate > 4× 세션중앙` · **시간해상도**: μs~20 ms → **saccade 포착**.
- (융합 방식) 각 프레임 **±20 ms** 창의 이벤트 수 `cnt`, 세션중앙 `m` →
  **`Saccade = (눈뜸) AND cnt > 4·m`** (Blink는 프레임 마스크 기준으로 결정).

**[Figure: Subject-wise 4-state Distributions (Event, before interpolation)]** — `fig_split_4state_event_before.png`

| split | Fixation | **Saccade** | Smooth | Blink |
|---|--:|--:|--:|--:|
| Train | 54.0% | **8.42%** | 28.5% | 9.1% |
| Val | 51.8% | **8.69%** | 30.8% | 8.7% |
| Test | 51.6% | **8.60%** | 30.2% | 9.6% |

→ 같은 데이터라도 **Event에서는 Saccade가 ~8.5%**로, Frame(0%) 대비 수백 배 표현된다(시간해상도 차이).

## 2. Frame Interpolation (Saccade region 보강)
위 한계를 해소하기 위해, **Saccade 구간에 TimeLens-XL을 적용하여 200 Hz(200 FPS, 5 ms)로 Frame Interpolation**을 수행한다. 5 ms 간격에서는 saccade가 6–16 프레임에 걸쳐 펼쳐져, 인접 프레임 속도가 임계를 넘어 **Frame 표현에서도 saccade가 검출 가능**해진다.

**[Figure: Subject-wise 4-state Distributions (after Frame Interpolation)]** — 실측 TimeLens-XL 산출 분포로 삽입 `[TBD: 실측]`
보간 후 분포(추정/emulated, `fig_split_4state_after_emulated.png`): Saccade **≈ 9.3%** (Train 9.33 / Val 9.35 / Test 9.33%)로, Frame에서 saccade state가 학습 가능한 수준으로 회복된다. *(주: 이 분포는 실 보간 산출이 아직 반영되지 않은 경우의 모델링 추정치이며, TimeLens-XL 실행 결과로 교체한다 — 무결성상 실측 미확정 수치는 [TBD]로 표기.)*

## 3. Event Interpolation (Sparse Event 보강)
- **New Event Interval**: 보간된 200 Hz 프레임을 기준으로, **이벤트가 희소한 프레임-사이(주로 saccade 전이) 구간**에 대해 인접 보간 프레임 쌍의 밝기 변화를 임계화(V2E 방식)하여 **합성 이벤트를 생성**한다.
- 즉, 200 Hz로 조밀해진 프레임 타임라인과 **시간 정합을 유지한 채**, 실제 이벤트만으로는 sparse했던 빠른-전이 구간의 이벤트 표현을 보강하여 saccade의 event 표현(버스트)을 학습에 충분히 제공한다.
- 프레임–이벤트는 동일 DAVIS 클럭을 공유하므로(확인됨), 보간 프레임 타임스탬프 기준으로 합성 이벤트를 일관되게 정렬할 수 있다.

## 4. 검증·무결성·한계 (리뷰어 방어)
- **증강은 학습 전용**이다. **Val/Test 평가와 GT는 보간이 아닌 원본 EV-Eye(실 프레임·실 이벤트·VGG 실 라벨)** 에 대해서만 수행한다. 보간으로 만든 프레임/이벤트에 대고 정확도를 측정하지 않는다(순환평가 방지).
- 보간 프레임은 추정값이므로 **ballistic saccade의 평활화** 가능성이 있다 → 보간 데이터를 saccade의 *정답(GT)* 으로 쓰지 않고 **입력 표현 보강**으로만 사용한다.
- 학습(실+보간)과 평가(실)의 분포 차이는 **sim-to-real gap**으로 별도 보고한다.
- Split: Train 1–36 / Test 37–48 (subject-independent), Val={37,40,45,48}. (Val⊂Test 중첩은 별도 표기.)

## 5. 요약
| | 판정 물리량 | 임계 | 시간해상도 | Saccade 비율 |
|---|---|---|---|--:|
| Frame (전) | 동공 속도(px/s) | >493 px/s | 25 Hz(40 ms) | ~0.00% |
| Event (전) | 이벤트 레이트 | >4× 세션중앙 | μs~20 ms | ~8.5% |
| **Frame (보간 후, 200 Hz)** | 동공 속도(px/s) | >493 px/s | 200 Hz(5 ms) | **≈9.3% (실측 [TBD])** |

원본 분포에서 Frame의 saccade가 사실상 학습되지 않는 문제를, **(i) TimeLens-XL 200 Hz 프레임 보간 + (ii) 보간 프레임 기반 이벤트 합성**으로 보강하여, saccade state의 학습 가능성을 확보하였다. 평가·GT는 실데이터로 유지한다.

### 파일/근거
그림: `fig_split_4state_{frame_before,event_before,after_emulated}.png` · 표(비율 포함): `split_4state_{frame_before,event_before,after_emulated}.csv`, `split_4state_summary.csv` · 기준: `CRITERIA_4state.md`
