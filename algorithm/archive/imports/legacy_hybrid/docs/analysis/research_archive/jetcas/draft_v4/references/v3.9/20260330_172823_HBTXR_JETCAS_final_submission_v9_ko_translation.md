# HBTXR 최종 제출본(v9) 섹션별 한국어 번역

원문: `HBTXR_JETCAS_final_submission_v9.tex`  
번역 기준: 현재 최종 제출본의 섹션 구조를 그대로 유지하되, **본문 문장과 표 캡션/핵심 표 내용**을 한국어로 옮겼습니다.  
참고: 수식 기호와 변수 표기는 원문과 동일하게 유지했습니다.

---

## 제목

**HBTXR: 온디바이스 확장현실을 위한 알고리즘–하드웨어 공동 설계 하이브리드 안구 추적**

## 초록 (Abstract)

확장현실(XR) 장치에서의 실시간 안구 추적은, 신뢰할 수 있는 동공 위치 추정과 시선 추정을 유지하면서도 엄격한 지연시간, 에너지, 메모리, 폼팩터 제약을 동시에 만족해야 한다. 기존 연구는 대체로 알고리즘 또는 가속기 중 한쪽만을 독립적으로 최적화해 왔으며, 그 결과 모델의 동작 특성과 실제 배치 비용 사이에 불일치가 발생하는 경우가 많았다. 본 논문에서는 프레임 기반 Search, 이벤트 기반 Track, 런타임 스케줄링, 그리고 가속기 매핑을 하나의 시스템 파이프라인으로 통합한 알고리즘–하드웨어 공동 설계 하이브리드 안구 추적 프레임워크인 **HBTXR**를 제안한다. 알고리즘 측면에서 HBTXR은 시간 동기화 및 기하 안정성을 갖는 하이브리드 감독(supervision) 흐름, 모달리티 인지 트랜스포머, 그리고 Search는 전체 백본을 실행하고 Track은 전반부 cut point에서 종료되는 배치용 학생 모델을 증류하는 암묵적 모델 프루닝 절차를 도입한다. 또한 품질 신호에 기반하여 모드 전환을 제어하는 런타임 인지 스케줄러를 설계한다. 하드웨어 측면에서 HBTXR은 양자화 친화적 연산자 재구성, 테이블 기반 비선형 근사, 그리고 intra-layer dataflow와 cross-layer coupling을 함께 최적화하는 스트리밍형 멀티스테이지 파이프라인을 채택한다. 실험 결과, HBTXR은 동공 중심 오차 0.42 pixel, 시선 오차 $0.68^{\circ}$, 종단간 Track 지연시간 0.57 ms를 달성했으며, 동시에 97.8\%의 Search 성공률과 1754 Hz의 유효 Track 업데이트율을 유지하였다. 이러한 결과는 HBTXR이 정확도, 저지연성, 배치 지향성을 동시에 만족하는 XR 안구 추적의 유효한 동작점을 제공함을 보여준다.

## 핵심어 (Keywords)

안구 추적, 확장현실, 하이브리드 센싱, 이벤트 카메라, 트랜스포머 가속기, FPGA, 알고리즘–하드웨어 공동 설계.

---

## 1. 서론 (Introduction)

안구 추적은 확장현실(XR)에서 핵심적인 기반 기능으로 자리 잡았으며, 시선 기반 상호작용, 포비에이티드 렌더링(foveated rendering), 사용자 상태 이해, 프라이버시 보존형 온디바이스 인지 기능을 지원한다. 그러나 실제 XR 안구 추적 시스템은 세 가지 요구를 동시에 만족해야 한다. 첫째, 외형 변화가 있더라도 의미적(semantic)으로 강건해야 한다. 둘째, 빠른 안구 운동 하에서도 시간적으로 민감하게 반응해야 한다. 셋째, 엄격한 전력 및 메모리 예산을 가진 임베디드 하드웨어에서 실행될 만큼 충분히 효율적이어야 한다. 따라서 XR 안구 추적은 단순한 인지(perception) 문제가 아니라 본질적으로 **시스템 수준의 설계 문제**라고 볼 수 있다.

기존 접근법은 구조적인 절충 관계를 보여준다. 프레임 기반 방법은 의미 복원과 보정(calibration) 인지 회귀 측면에서 강점을 가지지만, 밀집 영상(dense image)을 반복적으로 처리해야 하므로 높은 대역폭과 지연시간 비용이 발생한다. 이벤트 기반 방법은 희소한 센싱과 높은 시간 해상도를 제공하지만, 이벤트가 약한 구간, 누적 드리프트, 재초기화 실패 상황에서는 신뢰성이 저하된다. 하이브리드 방법은 이러한 균형을 어느 정도 개선하지만, 많은 경우 모달리티 융합, 런타임 모드 전환, 하드웨어 배치가 여전히 느슨하게 결합된 단계들로 취급된다.

본 논문은 이러한 간극을 메우기 위해, 온디바이스 XR을 위한 알고리즘–하드웨어 공동 설계 하이브리드 안구 추적 시스템 **HBTXR**를 제안한다. 단일한 멀티모달 회귀(monolithic multimodal regression)와 달리, HBTXR은 **Search**와 **Track**을 명시적으로 분리한다. Search는 프레임 기반이며 refresh-oriented한 경로로서, 강건한 재탐색(relocalization)과 anchor 복구를 담당한다. Track은 이벤트 기반이며 state-persistent한 경로로서, 현재 동공 추정치 주변에서 저지연 잔차 업데이트를 수행한다. 이러한 분해는 모델 구조뿐 아니라 감독 흐름, 학습 파이프라인, 런타임 스케줄러, 하드웨어 구조 전반에 반영된다. 전체 모델(full model)에서는 두 모달리티가 하나의 공유 트랜스포머 계열 아래에서 함께 학습되며, 배치 지향 암묵적 모델 프루닝 단계에서는 Search는 전체 백본을 유지하고 Track은 전반부 cut point에서 종료하여 중간 표현을 Track head로 직접 전달하는 슬림한 학생 모델을 증류한다.

그림 1은 전체 설계를 요약한다. HBTXR은 먼저 프레임 입력과 이벤트 입력이 하나의 안정적인 ellipse 중심 목표를 공유하도록 시간 동기식 supervision을 구성한다. 이후 공유 백본과 분리된 heads를 갖는 모달리티 인지 트랜스포머를 사용하고, 이어서 Track 측 cut point를 갖는 더 작은 배치용 학생 모델을 내보내는 암묵적 모델 프루닝 단계를 수행한다. 런타임 스케줄러는 confidence, similarity, event density, eye-state validity에 따라 Search와 Track을 전환한다. 마지막으로 CPU–FPGA 배치 구조는 공유 백본과 경량 head들을 차별화된 실행 경로로 매핑한다. 즉, HBTXR은 알고리즘 설계 단계에서부터 배치 제약을 내부화하고, 그 결과 얻어지는 Search/Track 런타임 의미론을 하드웨어 구조에 직접 대응시킨다.

**그림 1 캡션 번역**  
HBTXR 개요. 본 프레임워크는 시간 동기식 supervision contract, 모달리티 인지 Search/Track 트랜스포머, 런타임 스케줄러, 그리고 배치 지향 CPU–FPGA 파이프라인을 결합한다. Search는 학생 모델의 전체 백본을 통과하는 반면, Track은 암묵적 모델 프루닝에 의해 도입된 전반부 cut point에서 종료된다.

본 연구의 주요 기여는 다음과 같다.

1. 하이브리드 안구 추적을 Search–Track 문제로 재정식화하고, 프레임, 이벤트, 상태 목표를 정렬하기 위한 시간 동기·기하 안정 supervision contract를 제안한다.
2. 공유 백본과 분리된 Search, Event-Validation, Track, Mask, Eye-Region head를 갖는 모달리티 인지 트랜스포머를 설계하고, Search는 전체 백본을 사용하고 Track은 전반부 cut point에서 종료되는 배치용 학생 모델을 증류하는 암묵적 모델 프루닝 흐름과 결합한다.
3. 품질 평가 신호에 따라 Search 모드, Track 모드, 모드 전환을 명시적으로 관리하는 런타임 인지 스케줄러를 개발한다.
4. 양자화된 연산자, 테이블 기반 비선형 근사, 스트리밍 실행, 멀티스테이지 파이프라인 균형, CPU–FPGA 공동 스케줄링을 포함하는 배치 지향 가속기 설계를 제시한다.

---

## 2. 관련 연구 (Related Works)

### 2.1 프레임 기반 안구 추적 (Frame-based Eye Tracking)

프레임 기반 안구 추적은 근안(near-eye) 및 HMD 시스템에서 여전히 지배적인 패러다임이다. 그 이유는 그레이스케일 또는 적외선 프레임이 동공 위치 추정, 홍채 피팅, 시선 회귀에 필요한 강한 의미 구조를 보존하기 때문이다. 이러한 방법은 전역적 재탐색과 보정 인지 추정에서 특히 효과적이다. 그러나 이러한 장점은 밀집 센싱과 반복적인 전체 프레임 처리 비용을 수반하며, 이는 높은 메모리 트래픽, 대역폭 오버헤드, 그리고 빠른 안구 운동 시 제한된 응답성으로 이어진다.

### 2.2 이벤트 기반 안구 추적 (Event-based Eye Tracking)

이벤트 기반 안구 추적은 비동기적 밝기 변화 센싱을 활용하여 희소하고 높은 시간 해상도의 처리를 달성한다. 선행 연구는 이벤트 기반 방법이 빠른 saccade 동안의 저지연 업데이트와 fixation 구간의 중복 처리 감소에 매력적임을 보여주었다. 그럼에도 불구하고 이벤트-only 파이프라인은 의미적 안정화, 깜빡임(blink) 복구, 장시간 재탐색 측면에서 여전히 약점을 가지며, 특히 이벤트 밀도가 낮아지거나 추적기가 유효한 동공 윤곽에서 벗어날 때 이러한 한계가 두드러진다.

### 2.3 하이브리드 프레임–이벤트 안구 추적 (Hybrid Frame-Event Eye Tracking)

하이브리드 안구 추적은 프레임 기반의 의미 복원 능력과 이벤트 기반의 시간 응답성을 결합한다. 대표적인 시스템들은 저주기 프레임에서 relocalization을 수행하고, 이벤트 스트림이 고주기 업데이트를 지원하도록 구성될 수 있음을 보여주었다. 이러한 방향은 센싱 모달리티를 추적 단계와 정렬시킨다는 점에서 XR에 특히 적합하다. 그러나 많은 하이브리드 시스템은 융합 전략, 런타임 모드 정책, 하드웨어 매핑 사이의 관계를 여전히 충분히 명시하지 못하고 있다.

### 2.4 배치 지향 설계와 하드웨어 가속기 (Deployment-Oriented Designs and Hardware Accelerators)

배치 지향 안구 추적은 프레임 중심 가속기, 이벤트 중심 희소 설계, 그리고 ASIC 기반 저전력 추적기 등을 통해 연구되어 왔다. 이러한 연구는 안구 추적이 단지 인지 문제만이 아니라 하드웨어 설계 문제이기도 하다는 점을 보여준다. 동시에, 모델 설계·양자화·가속기 매핑을 하나의 워크플로 안에 통합하는 알고리즘–하드웨어 공동 최적화가 엣지 비전 시스템에서 효과적임도 입증되었다. 하이브리드 XR 안구 추적에서 아직 남아 있는 과제는, Search/Track 의미론, 트랜스포머 추론, 그리고 실제 배치 동작을 하나의 통합 프레임워크 안에서 공동 설계하는 것이다.

### 2.5 요약 (Summary)

기존 문헌은 네 가지 결론을 시사한다. 첫째, 프레임 중심 방법은 전역 refresh를 위한 가장 신뢰할 수 있는 선택지이다. 둘째, 이벤트 중심 방법은 초저지연의 국소 업데이트를 위한 가장 유력한 선택지이다. 셋째, 하이브리드 시스템은 그 운용 모드가 명시적으로 정의될 때에만 설득력이 높아진다. 넷째, 하드웨어 제약을 모델 설계 단계에서부터 내재화할수록 배치 효율은 향상된다. HBTXR은 바로 이러한 결론들 위에 구축되었다.

---

## 3. 제안하는 HBTXR 알고리즘 (Proposed HBTXR Algorithm)

### 3.1 프레임워크 설계 (Framework Design)

#### 3.1.1 문제 정식화 (Problem Formulation)

HBTXR은 프레임 스트림, 이벤트 스트림, 그리고 순환 상태(recurrent state)로 이루어진 하이브리드 센싱 입력을 사용한다. 시각 $t$에서 획득한 프레임을 $I_t \in \mathbb{R}^{H\times W}$라 하고, 현재 윈도우 안의 이벤트 집합을 $E_t = \{e_k=(x_k,y_k,p_k,\tau_k)\}_{k=1}^{N_t}$라 하자. 여기서 $(x_k,y_k)$는 이벤트 좌표, $p_k \in \{-1,+1\}$는 극성, $\tau_k$는 타임스탬프이다. 이벤트 스트림은 causal accumulation 연산자를 통해 polarity-split tensor $V_t \in \mathbb{R}^{2\times H\times W}$로 변환된다.

**Eye Model**  
목표 동공 상태는 다음과 같이 파라미터화된다.

```tex
\mathbf{s}_t = [x_t, y_t, a_t, b_t, u_t, v_t] \in \mathbb{R}^{6}
```

여기서 $(x_t,y_t)$는 동공 중심, $(a_t,b_t)$는 ellipse 축 길이, $[u_t,v_t]=[\sin(2\theta_t),\cos(2\theta_t)]$는 회전 각도의 삼각함수 인코딩이다. 이러한 표현은 각도 wrap-around 경계에서의 불연속성을 피하고, ellipse 방향을 보다 안정적으로 회귀할 수 있도록 한다.

**Objectness**  
HBTXR은 ellipse 파라미터만 예측하는 것이 아니라, 현재 관측이 유효하고 추적 가능한 동공 가설을 포함할 가능성, 즉 **objectness**도 함께 모델링한다. 이진 학습 타깃은 다음과 같이 정의된다.

```tex
o_t^{\star}=\mathbf{1}[\text{valid visible pupil at time } t]
```

Search 모드에서 objectness는 프레임 기반 후보를 새로운 anchor로 채택할지를 결정하는 refresh confidence로 해석된다. Track 모드에서는 objectness가 track confidence 및 track quality와 함께 사용되어, 현재 이벤트 기반 잔차 업데이트가 여전히 신뢰할 수 있는지를 평가한다. 이때 스케줄러에서 사용하는 closed-eye 또는 invalid-observation flag는 별도의 기호 $z_t$로 두어, objectness와 eye-state validity가 같은 기호를 공유하지 않도록 한다. 이렇게 정의된 objectness는 인지(perception)와 스케줄링을 연결하는 인터페이스가 된다.

#### 3.1.2 프레임워크 개요 (Framework Overview)

전체 프레임워크는 네 개의 상호작용하는 구성요소로 이루어진다. 즉, supervision pipeline, modality-aware transformer, Search/Track scheduler, deployment path이다. Search는 프레임 기반 relocalization 및 anchor refresh를 수행한다. Track은 이전 상태를 조건으로 하는 이벤트 기반 residual update를 수행한다. 배치 지향 암묵적 모델 프루닝 과정에서 Search 경로는 전체 백본을 유지하고, Track 경로는 전반부 cut point에 묶여 슬림한 학생 모델이 scheduler-visible output을 바꾸지 않으면서도 early exit하도록 설계된다. 이처럼 명시적인 분리는 HBTXR이 하이브리드 안구 추적을 하나의 무차별적 멀티모달 회귀기가 아니라 두 개의 운용 모드를 갖는 시스템으로 다루도록 해준다. 따라서 최종 배치 추정치는 다음과 같이 표현된다.

```tex
\hat{\mathbf{s}}_t=
\begin{cases}
\hat{\mathbf{s}}_t^{\mathrm{s}}, & m_t=\mathrm{Search},\\
\bar{\mathbf{s}}_{t-1}\oplus\Delta\hat{\mathbf{s}}_t, & m_t=\mathrm{Track},
\end{cases}
```

여기서 $\oplus$는 residual ellipse-state update operator를 의미한다.

### 3.2 시간 동기·기하 안정 하이브리드 감독 (Time-Synchronous and Geometry-Stable Hybrid Supervision)

**Dataset Problem**  
하이브리드 안구 추적에서 가장 큰 어려움 중 하나는, 프레임 라벨은 보통 시간축에서 희소한 반면 이벤트 데이터는 비동기적이고 밀집되어 있다는 점이다. 감독 시점(supervision timestamp)이 정밀하게 정렬되지 않으면 모델은 프레임, 이벤트, 순환 상태 입력 전반에 걸쳐 일관되지 않은 기하 구조를 학습하게 된다. HBTXR은 프레임 단서, 이벤트 단서, 이전 상태, 마스크, 품질 메타데이터가 모두 동일한 시간 인덱스와 동일한 ellipse 파라미터화를 참조하도록 하는 하나의 canonical supervision contract를 구성함으로써 이 문제를 해결한다.

#### 3.2.1 프레임 보간 (Frame Interpolation: TimeLens)

시간축에서 supervision을 조밀하게 만들기 위해, HBTXR은 먼저 희소하게 라벨링된 프레임들 사이에 중간 프레임 관측을 합성하는 frame interpolation을 수행한다. 이 단계는 TimeLens 기반 frame interpolation으로 구현된다. 이때 목적은 런타임에서 새로운 추론 모달리티를 만들기 위함이 아니라, 오프라인에서 더 잘 정렬된 supervisory frame을 구성하는 것이다. 보간된 프레임은 이벤트 윈도우와 annotation anchor 사이의 시간 간격을 줄여 주며, 이는 빠른 saccade 동안 특히 중요하다.

#### 3.2.2 밀집 주석 생성 (Dense Annotation: Grounded-SAM)

시간축 densification 이후, HBTXR은 Grounded-SAM 기반 pseudo-labeling 단계를 적용하여 보간된 프레임으로부터 eye-region 및 pupil mask를 정교화한다. 정렬된 시점에서 pupil mask를 $\bar M_t$, eye-region target을 $\bar B_t$라 하자. 그러면 ellipse target은 mask moment로부터 다음과 같이 복원된다.

```tex
\bar{\mathbf c}_t = \frac{1}{|\bar M_t|}\sum_{(x,y)\in \bar M_t}[x,y]^{\top}
```

```tex
\bar{\bm\Sigma}_t = \frac{1}{|\bar M_t|}\sum_{(x,y)\in \bar M_t}
([x,y]^{\top}-\bar{\mathbf c}_t)([x,y]^{\top}-\bar{\mathbf c}_t)^{\top}
```

그리고 canonical ellipse state $\bar{\mathbf s}_t$는 $\bar{\bm\Sigma}_t$의 고유축과 방향으로부터 얻어진다. 동시에 annotation pipeline은 다음과 같은 품질 메타데이터를 저장한다.

```tex
\bar{\bm\xi}_t=[\bar o_t,\bar z_t,\bar r_t,\bar \kappa_t]
```

여기서 $\bar o_t$는 label validity, $\bar z_t$는 closed-eye flag, $\bar r_t$는 track-valid flag, $\bar \kappa_t$는 mask confidence를 의미한다.

#### 3.2.3 이벤트 누적 (Event Accumulation: Frame-Synchronous)

마지막으로 event accumulation은 프레임 동기 방식(frame-synchronous)으로 수행되어, 각 event tensor가 supervisory frame 또는 보간된 시각과 정렬되도록 한다. 단순한 이벤트 카운팅 대신, HBTXR은 최근 이벤트를 더 강조하면서도 하드웨어 친화적인 **fast causal accumulation** 규칙을 사용한다. 시간 정렬된 목표 $t$에 대응하는 accumulation window를 $\mathcal W_t$라 하자. 이벤트 $e_k=(x_k,y_k,p_k,\tau_k)$에 대해 causal weight는 다음과 같이 정의된다.

```tex
w_k(t)=\max\!\left(0,1-\frac{t-\tau_k}{\Delta_t}\right)
```

여기서 $\Delta_t$는 window horizon이다. 그러면 누적된 event tensor는 다음과 같다.

```tex
\bar V_t(x,y,p)=\sum_{e_k\in \mathcal W_t}
w_k(t)\,
\mathbf 1\big[(x_k,y_k,p_k)=(x,y,p)\big]
```

일부 이벤트 파이프라인에서 사용하는 단순 count accumulation은 $w_k(t)\equiv 1$인 특수한 경우로 볼 수 있다. 반면 여기서의 causal weighting은 거의 추가 제어 비용 없이 시간 순서 정보를 보존한다.

전체 supervision pipeline의 출력은 다음과 같은 canonical sample이다.

```tex
(\bar I_t,\bar V_t,\bar{\mathbf s}_{t-1},\bar{\mathbf s}_t,\bar M_t,\bar B_t,\bar{\bm\xi}_t)
```

여기서 $\bar I_t$와 $\bar V_t$는 시간 정렬된 프레임/이벤트 입력, $\bar M_t$는 pupil mask, $\bar B_t$는 eye-region target, $\bar{\bm\xi}_t$는 품질 신호 집합이다. 이 contract는 모달리티 간 기하 구조를 안정화하고, 품질 인지 supervision을 학습과 스케줄링 양쪽으로 전달한다.

**그림 2 캡션 번역**  
HBTXR에서 사용하는 시간 동기 supervision contract. 프레임 보간과 밀집 annotation이 기하적으로 안정적인 ellipse target을 정의하고, 프레임 동기 fast causal accumulation이 event tensor와 이전 상태를 동일한 target 인덱스에 정렬한다.

### 3.3 모달리티 인지 트랜스포머 구조 (Modality-aware Transformer Architecture)

#### 3.3.1 프런트엔드: 패치 임베딩 (Front-end: Patch Embedding)

HBTXR은 단순한 early fusion 대신, 모달리티별 전용 front-end를 사용한다. $\operatorname{Patch}(\cdot)$를 patch flattening 연산, $\phi(\bar{\mathbf s}_{t-1})$를 학습 가능한 상태 임베딩이라고 하자. 프레임 및 이벤트 token sequence는 다음과 같이 정의된다.

```tex
\mathbf Z_{0,t}^{\mathrm{f}} =
A_f\!\left(\operatorname{Patch}(\bar I_t)\mathbf W_f + \mathbf 1\mathbf b_f^{\top}\right)+\mathbf P
```

```tex
\mathbf Z_{0,t}^{\mathrm{e}} =
A_e\!\left(\operatorname{Patch}(\bar V_t)\mathbf W_e + \mathbf 1\mathbf b_e^{\top}\right)
+ \mathbf 1\phi(\bar{\mathbf s}_{t-1})^{\top}+\mathbf P
```

여기서 $\mathbf P$는 positional embedding이며, $A_f(\cdot)$와 $A_e(\cdot)$는 modality adapter이다. 이벤트 경로는 이전 상태를 명시적으로 주입함으로써, Track 분기가 처음부터 다시 relocalization하는 대신 residual update를 추정하도록 만든다.

#### 3.3.2 백본: MHA와 MLP (Backbone: MHA, MLP)

두 분기는 모두 multi-head attention(MHA)과 multi-layer perceptron(MLP) 블록으로 이루어진 transformer backbone을 공유한다. 블록 $\ell$의 업데이트는 다음과 같다.

```tex
\tilde{\mathbf Z}_{\ell} = \mathbf Z_{\ell-1} + \operatorname{MHA}\!\left(\operatorname{LN}(\mathbf Z_{\ell-1})\right)
```

```tex
\mathbf Z_{\ell} = \tilde{\mathbf Z}_{\ell} + \operatorname{MLP}\!\left(\operatorname{LN}(\tilde{\mathbf Z}_{\ell})\right)
```

attention block 내부에서 head $h$는 다음을 계산한다.

```tex
\operatorname{Attn}_h(\mathbf Z)=
\operatorname{softmax}\!\left(
\frac{\mathbf Q_h\mathbf K_h^{\top}}{\sqrt{d_h}}
\right)\mathbf V_h
```

여기서 $\mathbf Q_h=\mathbf Z\mathbf W_h^Q$, $\mathbf K_h=\mathbf Z\mathbf W_h^K$, $\mathbf V_h=\mathbf Z\mathbf W_h^V$이다. Stage 1 학습에서는 전체 teacher가 하나의 공유 transformer family를 사용하므로, 프레임 기반 Search feature와 이벤트 조건부 feature가 모두 하나의 고용량 백본 아래에서 학습된다. Stage 2 배치 프루닝에서는 내보내는 학생 모델의 깊이를 $L^{\mathrm S}$로 줄이고, 다음과 같은 Track 측 cut point를 도입한다.

```tex
L_c = \left\lfloor \frac{L^{\mathrm S}}{2} \right\rfloor
```

따라서 Search와 Track의 배치 feature는 각각 다음과 같이 표현된다.

```tex
\mathbf F_t^{\mathrm{s}} = \mathcal F_{1:L^{\mathrm S}}\!\left(\mathbf Z_{0,t}^{\mathrm{f}}\right)
```

```tex
\mathbf C_t^{\mathrm{trk}} = \mathcal F_{1:L_c}\!\left(\mathbf Z_{0,t}^{\mathrm{e}}\right)
```

즉, Search는 학생 모델의 모든 백본 블록을 통과하고, Track은 cut-point tensor $\mathbf C_t^{\mathrm{trk}}$에서 종료한 뒤 Track head를 호출한다. 이러한 분할은 암묵적 모델 프루닝과 함께 도입된다. 초기 블록은 residual update에 충분한 motion-consistent local structure를 유지하고, 더 깊은 블록은 강한 전역 semantic이 필요한 Search에만 남겨 둔다.

#### 3.3.3 백엔드: Head (Back-end: Head)

공유 백본 위에서 HBTXR은 Eye-Region Head, Search Head, Event-Validation Head, Track Head, Search Mask Head로 이루어진 분리형 head 구성을 사용한다. 이들의 출력은 다음과 같다.

```tex
\big[\hat{\mathbf s}_t^{\mathrm{s}},\hat o_t^{\mathrm{s}},\hat{\mathbf m}_t,\hat{\mathbf b}_t\big]
= H_{\mathrm{s}}\!\left(\mathbf F_t^{\mathrm{s}}\right)
```

```tex
\big[\hat{\mathbf s}_t^{\mathrm{v}},\hat o_t^{\mathrm{v}}\big]
= H_{\mathrm{v}}\!\left(\mathbf C_t^{\mathrm{trk}}\right)
```

```tex
\big[\Delta\hat{\mathbf s}_t,\hat c_t^{\mathrm{trk}},\hat q_t^{\mathrm{trk}}\big]
= H_{\mathrm{t}}\!\left([\operatorname{Pool}(\mathbf C_t^{\mathrm{trk}});\phi(\bar{\mathbf s}_{t-1})]\right)
```

Search head는 전체 깊이의 프레임 기반 feature로부터 objectness와 ellipse state를 추정한다. Event-Validation head는 cut-point tensor 위에서 동작하며, 이후 스케줄러가 활용하는 이벤트 측 consistency cue를 생성한다. Track head는 같은 cut-point representation과 임베딩된 이전 상태를 이용해 residual state update, confidence, track quality를 예측한다. Mask head는 기하적 regularization 및 추가적인 품질 단서로 작동한다. 이러한 decoupled back-end는 공유 백본 아래에서도 기능적 specialization을 유지하는 데 핵심적이다.

**그림 3 캡션 번역**  
HBTXR 모델 구조. 프레임과 이벤트 입력은 모달리티별 패치 임베딩을 사용한다. 배치 학생 모델은 백본의 전반부를 공유하고 Track cut point를 $L_c$에 두며, Search는 그 이후 후반부(back half)를 계속 통과하여 Search 지향 heads로 연결된다.

### 3.4 런타임 인지 Search/Track 스케줄러 (Runtime-aware Search and Track Scheduler)

**Search Mode**  
Search 모드는 refresh-oriented 모드이다. 초기화 시점 또는 상태가 신뢰할 수 없게 되었을 때 활성화된다. 이 모드에서 시스템은 의미적 강건성을 우선하며, 프레임 기반 추론을 통해 새로운 pupil anchor, eye-region context, mask-aware quality cue를 추정한다. 구체적으로 Search는 배치 백본 $\mathcal F_{1:L^{\mathrm S}}$ 전체를 실행하고, Search, Mask, Eye-Region head를 활성화한다.

**Track Mode**  
Track 모드는 state-persistent하고 latency-oriented하다. 전체 위치 추정 문제를 처음부터 다시 푸는 대신, 이전 상태 주변의 residual update를 예측한다. 배치 학생 모델에서 Track은 이벤트 token과 이전 상태를 함께 사용하고, 프루닝된 prefix $\mathcal F_{1:L_c}$만 통과한 뒤 cut-point representation을 Track head로 바로 전달한다. Track residual은 다음과 같다.

```tex
\Delta \hat{\mathbf s}_t=
H_{\mathrm{t}}\!\left([\operatorname{Pool}(\mathbf C_t^{\mathrm{trk}});\phi(\bar{\mathbf s}_{t-1})]\right)
```

Track 상태는 다음과 같이 복원된다.

```tex
\hat{\mathbf s}_t^{\mathrm{trk}}=
\bar{\mathbf s}_{t-1}\oplus \Delta \hat{\mathbf s}_t
```

Residual update operator $\oplus$는 다음과 같이 구체화된다.

```tex
\hat x_t = \bar x_{t-1} + \Delta \hat x_t,\quad
\hat y_t = \bar y_{t-1} + \Delta \hat y_t
```

```tex
\hat a_t = \bar a_{t-1}\exp(\Delta \hat \alpha_t),\quad
\hat b_t = \bar b_{t-1}\exp(\Delta \hat \beta_t)
```

```tex
\begin{bmatrix}\hat u_t\\ \hat v_t\end{bmatrix}
=
\frac{1}{\left\|\begin{bmatrix}\bar u_{t-1}+\Delta \hat u_t\\ \bar v_{t-1}+\Delta \hat v_t\end{bmatrix}\right\|_2}
\begin{bmatrix}\bar u_{t-1}+\Delta \hat u_t\\ \bar v_{t-1}+\Delta \hat v_t\end{bmatrix}
```

이 식은 위치 translation은 additive하게, 축 업데이트는 multiplicative하게, 방향 업데이트는 eye model에서 사용하는 sinusoidal encoding 위에서 정규화되도록 만든다.

**Mode Switching (Quality Assessment)**  
스케줄러는 여러 신호를 이용해 Search와 Track 사이를 결정한다. 이들은 다음과 같이 정의된다.

```tex
c_t^{\mathrm{s}} = \sigma(\hat o_t^{\mathrm{s}}),\qquad
c_t^{\mathrm{trk}} = \sigma(\hat c_t^{\mathrm{trk}})
```

```tex
q_t^{\mathrm{trk}} = \sigma(\hat q_t^{\mathrm{trk}}),\qquad
d_t = \frac{|\mathcal W_t|}{HW}
```

```tex
\rho_t = \exp\!\left(-\left\|\hat{\mathbf s}_t^{\mathrm{trk}}-\tilde{\mathbf s}_t^{\mathrm{anc}}\right\|_{\Lambda}\right)
```

```tex
z_t = \mathbf{1}[\text{closed-eye / invalid}]
```

여기서 $\tilde{\mathbf s}_t^{\mathrm{anc}}$는 가장 최근에 확정된 Search anchor이고, $\|\mathbf r\|_{\Lambda}=\sqrt{\mathbf r^{\top}\Lambda\mathbf r}$는 정규화된 ellipse-state discrepancy이다. 전이 규칙은 다음과 같다.

```tex
m_t = \mathrm{Track}, \quad \text{if } c_t^{\mathrm{s}}>\tau_s \wedge z_t=0
```

```tex
m_t = \mathrm{Search}, \quad \text{if } c_t^{\mathrm{trk}}\le\tau_t
\;\vee\; q_t^{\mathrm{trk}}\le\tau_q
\;\vee\; \rho_t\le\tau_{\rho}
\;\vee\; d_t\le\tau_d
\;\vee\; z_t=1
```

이 품질 평가 메커니즘은 추적 안정성을 높일 뿐만 아니라, 하드웨어 스케줄러에 노출되는 제어 의미론을 정의한다.

**그림 4 캡션 번역**  
HBTXR의 유한상태 런타임 스케줄러. Search는 전체 깊이 refresh를 수행하며, Track은 전반부 cut point만 사용하고 confidence, similarity, density, eye-state 조건이 유효할 때에만 유지된다.

### 3.5 학습 파이프라인 및 손실 함수 (Training Pipeline and Loss Function)

#### 3.5.1 2단계 학습 (Two-Stage Training)

HBTXR은 2단계 최적화 스케줄을 사용한다. Stage 1에서는 Search 지향 relocalization과 geometry recovery를 위한 full teacher model을 학습시킨다. 이를 통해 이벤트 조건부 Track branch를 도입하기 전에, 프레임 branch가 안정적인 전역 구조, objectness, mask-aware eye context를 먼저 학습하도록 한다. 전체 teacher network의 embedding dimension, MLP ratio, channel width, backbone depth를 각각 $D^{\mathrm{T}}, r^{\mathrm{T}}, C^{\mathrm{T}}, L^{\mathrm{T}}$라 두고 $\Theta^{\mathrm{T}}=(D^{\mathrm{T}}, r^{\mathrm{T}}, C^{\mathrm{T}}, L^{\mathrm{T}})$로 표기한다. Teacher는 geometry-aware ellipse loss로 최적화된다.

```tex
\mathcal L_{\mathrm{geo}}(\hat{\mathbf s},\mathbf s)=
\|\hat{\mathbf c}-\mathbf c\|_1 + \lambda_{\Sigma}\|\hat{\bm\Sigma}-\bm\Sigma\|_F^2
```

여기서 $\mathbf c=(x,y)$이고,

```tex
\bm\Sigma(\mathbf s)=
\mathbf R(\theta)\operatorname{diag}(a^2,b^2)\mathbf R(\theta)^{\top}
```

는 ellipse의 축과 방향으로부터 유도되는 covariance matrix이다. Stage 1 목적함수는 다음과 같다.

```tex
\mathcal L_{\mathrm{stage1}}^{\mathrm{T}} =
\lambda_{\mathrm{geo}}^{\mathrm{s}}\mathcal L_{\mathrm{geo}}^{\mathrm{s}}
+ \lambda_{\mathrm{obj}}^{\mathrm{s}}\mathcal L_{\mathrm{obj}}^{\mathrm{s}}
+ \lambda_{\mathrm{mask}}\mathcal L_{\mathrm{mask}}
+ \lambda_{\mathrm{eye}}\mathcal L_{\mathrm{eye}}
```

이 단계는 전역 relocalization 품질, Search objectness, 기하 안정 mask supervision을 중점적으로 학습한다.

Stage 2에서는 하드웨어 매핑 전에 폭(width)과 깊이(depth)를 줄인 배치 지향 student model을 도입한다. 학생 모델을 $\Theta^{\mathrm{S}}=(D^{\mathrm{S}}, r^{\mathrm{S}}, C^{\mathrm{S}}, L^{\mathrm{S}})$라 하고, 구조적 축소 계수를 $\bm\eta=(\eta_D,\eta_r,\eta_C,\eta_L)\in(0,1)^4$라고 하면, 축소 모델은 다음과 같이 직접 정의된다.

```tex
\Theta^{\mathrm{S}}=
(\lfloor \eta_D D^{\mathrm{T}}\rfloor,\;
\eta_r r^{\mathrm{T}},\;
\lfloor \eta_C C^{\mathrm{T}}\rfloor,\;
\lfloor \eta_L L^{\mathrm{T}}\rfloor)
```

즉, 배치 후보는 embedding dimension, MLP ratio, channel count, network depth를 동시에 축소하여 얻는다. 학생 모델은 또한 다음과 같은 Track-side cut point를 가진다.

```tex
L_c = \left\lfloor \frac{L^{\mathrm S}}{2} \right\rfloor
```

그 결과 Search는 여전히 학생 모델의 모든 $L^{\mathrm S}$ 블록을 평가하는 반면, Track은 프루닝된 prefix에서 early exit한다. 이후 full teacher는 고정(frozen)되고, 더 작은 student를 자기지도 증류(self-supervised distillation)로 유도한다. 동시에 student는 Search, event-validation, track, quality supervision도 직접 받는다. Stage 2 목적함수는 다음과 같다.

```tex
\mathcal L_{\mathrm{stage2}}^{\mathrm{S}} =
\lambda_{\mathrm{geo}}^{\mathrm{s}}\mathcal L_{\mathrm{geo}}^{\mathrm{s}}
+ \lambda_{\mathrm{obj}}^{\mathrm{s}}\mathcal L_{\mathrm{obj}}^{\mathrm{s}}
+ \lambda_{\mathrm{mask}}\mathcal L_{\mathrm{mask}}
+ \lambda_{\mathrm{eye}}\mathcal L_{\mathrm{eye}}
+ \lambda_{\mathrm{event}}\mathcal L_{\mathrm{event}}
+ \lambda_{\mathrm{trk}}\mathcal L_{\mathrm{trk}}
+ \lambda_{\mathrm{cons}}\mathcal L_{\mathrm{cons}}
+ \lambda_{\mathrm{qa}}\mathcal L_{\mathrm{qa}}
+ \lambda_{\mathrm{gaze}}\mathcal L_{\mathrm{gaze}}
+ \lambda_{\mathrm{fkd}}\mathcal L_{\mathrm{fkd}}
+ \lambda_{\mathrm{rkd}}\mathcal L_{\mathrm{rkd}}
```

이는 원래의 task loss를 유지하면서, full model의 표현 구조를 더 작은 deployment model로 전달하기 위한 distillation term을 명시적으로 추가한 형태이다.

#### 3.5.2 자기지도 증류 및 암묵적 모델 프루닝을 통한 네트워크 슬리밍 (Network Slimming via Self-Supervised Distillation and Implicit Model Pruning)

HBTXR은 단순한 sparsity penalty에만 의존하지 않는다. 대신, 모델 정의 단계에서 네 가지 구조적 축을 줄여 더 작은 배치 후보군을 구성한다. 즉, token embedding dimension, MLP expansion ratio, head와 adapter 내부의 channel 수, backbone block 수를 함께 축소한다. Full model은 teacher, 축소된 model은 student 역할을 하므로, 이 축소 단계는 단순 사후 weight pruning이 아니라 **self-supervised distillation**과 결합된다. 동일한 배치 지향 pruning 단계는 Track을 처음 $L_c$개 student block 뒤의 cut-point tensor에 연결하고, Search는 전체 student depth를 유지하도록 만든다. 이 설정은 Search/Track 의미론, supervision contract, scheduler-visible output을 바꾸지 않으면서도, 더 작고 양자화·파이프라이닝·FPGA 적합성이 높은 네트워크를 만든다.

첫 번째 distillation term은 중간 feature를 teacher feature에 맞추는 **feature KD loss**이다. Distillation layer 집합 $\Omega$에 대해 다음과 같이 정의한다.

```tex
\mathcal L_{\mathrm{fkd}}=
\sum_{\ell\in\Omega}
\frac{1}{N_{\ell}D_{\ell}^{\mathrm{T}}}
\left\|
G_{\ell}\!\left(\mathbf F_{\ell}^{\mathrm{S}}\right)
-\mathbf F_{\ell}^{\mathrm{T}}
\right\|_2^2
```

여기서 $\mathbf F_{\ell}^{\mathrm{S}}$와 $\mathbf F_{\ell}^{\mathrm{T}}$는 각각 student와 teacher의 $\ell$번째 layer feature이고, $G_{\ell}(\cdot)$는 축소된 student width를 teacher width에 맞추는 학습 가능한 projector이다.

두 번째 항은 sample 또는 token 사이의 pairwise distance와 angular structure를 보존하는 **relational knowledge distillation (RKD) loss**이다. Teacher 또는 student feature를 $\star\in\{\mathrm{T},\mathrm{S}\}$로 표시하면,

```tex
d_{ij}^{\star} =
\frac{\left\|\mathbf f_i^{\star}-\mathbf f_j^{\star}\right\|_2}
{\frac{1}{|\mathcal P|}\sum_{(u,v)\in\mathcal P}\left\|\mathbf f_u^{\star}-\mathbf f_v^{\star}\right\|_2}
```

```tex
\theta_{ijk}^{\star} =
\frac{\left(\mathbf f_i^{\star}-\mathbf f_j^{\star}\right)^{\top}
\left(\mathbf f_i^{\star}-\mathbf f_k^{\star}\right)}
{\left\|\mathbf f_i^{\star}-\mathbf f_j^{\star}\right\|_2
 \left\|\mathbf f_i^{\star}-\mathbf f_k^{\star}\right\|_2}
```

이고, RKD loss는 다음과 같다.

```tex
\mathcal L_{\mathrm{rkd}}=
\frac{1}{|\mathcal P|}
\sum_{(i,j)\in\mathcal P}
\operatorname{smooth}_{L1}\!\left(d_{ij}^{\mathrm{S}}-d_{ij}^{\mathrm{T}}\right)
+
\frac{1}{|\mathcal T|}
\sum_{(i,j,k)\in\mathcal T}
\operatorname{smooth}_{L1}\!\left(\theta_{ijk}^{\mathrm{S}}-\theta_{ijk}^{\mathrm{T}}\right)
```

Feature KD와 RKD를 결합함으로써, 슬림한 student는 pointwise activation뿐 아니라 teacher 표현의 기하 구조까지 함께 유지할 수 있다. 실제로 이러한 설계는 프루닝된 student가 latency-oriented Track early-exit 동작을 흡수하면서도 Search 의미론과 표현 품질을 full teacher에 가깝게 유지하도록 도와준다. 결과적으로 축소된 HBTXR 모델은 단순히 줄여 놓은 네트워크보다, 양자화·파이프라인 기반 하드웨어 경로에 더 적합한 배치 후보가 된다.

---

## 4. 제안하는 HBTXR 하드웨어 (Proposed HBTXR Hardware)

### 4.1 설계 과제와 공동 설계 원칙 (Design Challenges and Co-design Principles)

HBTXR의 하드웨어 대상은 고정된 일회성(one-pass) 트랜스포머가 아니라, 모드 전환이 일어나는 하이브리드 안구 추적 워크로드이다. 이 차이는 최적화 목표 자체를 바꾼다. Search는 refresh 중심이다. 프레임 단서로부터 신뢰할 수 있는 anchor를 다시 구축하고, 추가적인 출력 head를 활성화하며, 더 강한 전역 문맥을 요구한다. 반면 Track은 persistence 중심이다. 최근 이벤트와 이전 상태로부터 현재 동공 상태를 업데이트하며, 매우 짧은 응답시간을 선호하고, 재사용 가능한 상태와 중간 feature를 온칩에 유지하는 것이 유리하다. 따라서 두 모드를 하나의 균일한 실행 템플릿에 강제로 맞추면, Track에는 과도한 자원이 할당되거나 Search에는 필요한 자원이 부족해지게 된다.

핵심 과제는 단순한 arithmetic throughput이 아니라 **cross-layer coupling**이다. 전반부 cut point 이후의 early exit, Search anchor commit, 이전 상태의 재사용, 모드 의존적인 head 활성화는 모두 계층 간·호출 간 제어 결정을 결합시킨다. 두 번째 과제는 **intra-layer dataflow**이다. 선형 projection과 MLP는 stationary weight와 규칙적인 tensor tile이 지배적인 반면, attention interaction path는 동적으로 생성되는 token–token 곱과 일시적인 score buffer가 지배적이다. 세 번째 과제는 각 stage의 token readiness 조건이 서로 다른 **multi-stage pipeline** 아래에서 높은 활용률을 유지하는 것이다.

이에 따라 HBTXR은 네 가지 공동 설계 원칙을 따른다.

- **P1)** 초기 backbone 계산은 공유하되, Search는 전체 $L^{\mathrm S}$ block을 사용하고 Track은 처음 $L_c$ block만 사용하는 depth-asymmetric exit를 허용한다.
- **P2)** 주요 projection 및 MLP weight는 온칩에 resident하게 유지하여, 대역폭을 반복적인 모델 fetch가 아니라 센서 입력과 token 이동에 사용한다.
- **P3)** tensor-heavy kernel과 control-heavy policy를 분리한다. FPGA는 tokenization과 network inference를 수행하고, CPU는 scheduling, quality assessment, state arbitration을 수행한다.
- **P4)** 한 커널만 최대화하는 대신 모든 pipeline stage의 cadence를 균형화한다. 실제 종단간 지연시간은 가장 느린 stage와 attention, mode switching이 만드는 barrier에 의해 결정되기 때문이다.

운영 측면에서 Search와 Track은 같은 front-half backbone을 공유하더라도 서로 다른 하드웨어 contract를 노출한다. Search는 canonical frame token과 compact한 event-side summary를 입력으로 받아 $L^{\mathrm S}$개의 transformer block 전체를 통과하고, Search·Mask·Eye-Region head를 활성화하며, 갱신된 anchor와 validity metadata를 state SRAM에 기록한다. 반면 Track은 event tensor와 이전에 받아들여진 상태를 입력으로 받아 처음 $L_c$ block만 통과한 뒤, cut-point buffer에서 분기하여 deeper Search-only block으로 가지 않고 Track 및 Event-Validation head를 호출한다. 이 비대칭성 때문에 가속기는 하나의 정적인 token path가 아니라 명시적인 early-exit boundary를 중심으로 조직된다.

메모리 동작 또한 모드에 따라 달라진다. Search는 새로운 anchor를 생성하므로, 갱신된 상태, mask-validity cue, refresh metadata를 persistent state SRAM에 기록한다. Track은 resident anchor를 재사용하고, 호출 시작 시 같은 state SRAM을 읽은 뒤 residual ellipse update와 quality field만 다시 기록한다. 스케줄러 기반 실행에서는 CPU가 active tokenizer, backbone span, head set, commit policy를 지정하는 mode packet을 전송하고, FPGA는 그 아래에서 동일한 token-processing fabric을 재사용한다. 즉, Search/Track 분리는 단지 알고리즘적 선택이 아니라, 실제 배치 시스템의 dataflow, memory lifetime, controller behavior를 직접 규정한다.

### 4.2 양자화와 비선형 근사 (Quantization and Non-linear Approximation)

#### 4.2.1 양자화 (Quantization)

SEE에서 사용된 integer-only deployment 철학을 따라, HBTXR은 주요 선형 연산에 대해 **dyadic quantization**을 채택한다. 그 결과 하드웨어 경로는 부동소수점 연산 대신 정수 multiply, accumulate, shift 연산으로 구성된다. 양자화된 activation과 weight를 각각 $\mathbf X=S_x\hat{\mathbf X}$, $\mathbf W=S_w\hat{\mathbf W}$라고 하면, 양자화된 출력은 다음과 같다.

```tex
\mathbf Y = S_y\hat{\mathbf Y} = \mathbf W\mathbf X
= S_w\hat{\mathbf W}\cdot S_x\hat{\mathbf X}
```

```tex
\hat{\mathbf Y}
= \frac{S_wS_x}{S_y}\left(\hat{\mathbf W}\cdot\hat{\mathbf X}\right)
\approx \frac{\hat S}{2^n}
\left(\hat{\mathbf W}\cdot\hat{\mathbf X}\right)
```

여기서 $\hat S \in \mathbb Z$, $n\in \mathbb Z_{\ge 0}$는 dyadic scale을 정의한다. 실제 구현에서는 부동소수점 비율 $S_wS_x/S_y$를 하나의 정수 multiplier와 하나의 right shift로 치환한다. 이 형식은 HBTXR에서 특히 유리한데, block $L_c$ 뒤의 early-exit point, residual merge, mode-gated head에서 모두 동일한 dyadic interface를 재사용할 수 있기 때문이다.

두 번째 설계 선택은 **branch 간 scale alignment**이다. Search와 Track은 backbone의 front half를 공유하므로, cut point에서의 출력 tensor는 비싼 format converter 없이도 deeper backbone stage나 Track head 어느 쪽에도 입력될 수 있어야 한다. 이를 위해 cut-point tensor를 하나의 공통 dyadic exponent에 정렬하고, 남는 scale mismatch는 head-specific re-quantization table에 흡수한다. 이렇게 하면 두 모드 사이의 cross-layer coupling이 controller와 memory interface 수준에서 단순해진다.

#### 4.2.2 비선형 근사 (Non-linear Approximation)

HBTXR에서 하드웨어 친화성이 낮은 연산은 LayerNorm, Softmax, GELU, output re-scaling이다. 이들 연산은 LUT, BRAM table, 간단한 address logic로 구성된 compact한 **table-driven curve block**으로 매핑된다. HBTXR은 각 근사를 고립된 기법으로 취급하지 않고, streaming datapath의 구체적인 stage boundary에 결합한다. 즉, normalization은 projection 이전에, softmax는 interaction path 내부에, activation과 re-scale은 MLP 또는 head interface에 배치된다.

**1) LayerNorm reciprocal-root table**  
한 token tile의 정수 mean과 variance accumulator를 $\hat{\mu}$, $\hat{\sigma}^2$라 하면, normalization factor는 다음과 같이 근사된다.

```tex
u_{\mathrm{ln}} = \left\lfloor \frac{\hat{\sigma}^2-\alpha_{\mathrm{ln}}}{2^{s_{\mathrm{ln}}}} \right\rfloor
```

```tex
\hat r_{\mathrm{ln}} = T_{\mathrm{ln}}(u_{\mathrm{ln}})
\approx \frac{1}{\sqrt{\hat{\sigma}^2+\epsilon}}
```

```tex
\hat y_i = \hat\gamma \big((\hat x_i-\hat\mu)\hat r_{\mathrm{ln}}\big) + \hat\beta
```

분할된 reciprocal-root table은 normalization 오차에 가장 민감한 저분산 영역에 더 많은 table entry를 배치한다.

**2) Softmax table chain**  
한 attention row에서 양자화된 score를 $\hat s_{ij}$, row maximum을 $\hat m_i=\max_j\hat s_{ij}$, exponent sum을 $\hat z_i=\sum_j \hat e_{ij}$라 하면, HBTXR은 coupled exponent table과 sum-inverse table을 이용하여 softmax를 계산한다.

```tex
\hat e_{ij} =
T_{\exp}\!\left(
\left\lfloor \frac{\hat m_i-\hat s_{ij}}{2^{s_{\exp}}}\right\rfloor
\right)
\approx \exp(\hat s_{ij}-\hat m_i)
```

```tex
\hat a_{ij} = \hat e_{ij}\;
T_{\mathrm{sum}}\!\left(
\left\lfloor \frac{\hat z_i-\alpha_{\mathrm{sum}}}{2^{s_{\mathrm{sum}}}} \right\rfloor
\right)
\approx \frac{\hat e_{ij}}{\hat z_i}
```

이 방식은 divider가 많이 필요한 정규화를 피하면서, interaction engine의 stripe-wise score buffering과 잘 맞는다.

**3) Composite activation and dyadic re-scale**  
첫 번째 MLP projection 이후에는 activation과 output re-scale을 하나의 composite table로 합친다.

```tex
\hat y = T_{\mathrm{gelu}}\!\left(
\left\lfloor \frac{\hat x-\alpha_{\mathrm{gelu}}}{2^{s_{\mathrm{gelu}}}} \right\rfloor
\right)
\approx \operatorname{GELU}(x)\cdot\kappa_{\mathrm{out}}
```

```tex
\hat z = \operatorname{clip}\!\left(\left(\hat m\cdot \hat y\right)\gg n + z_z,
q_{\min},q_{\max}\right)
```

여기서 $(\hat m,n)$은 dyadic destination scale을, $z_z$는 destination zero-point를 의미한다. 이로써 동일한 table-driven 비선형 블록을 residual merge, Track cut point, mode-gated head interface에서도 그대로 사용할 수 있다.

### 4.3 가속기 구조 (Accelerator Architecture)

#### 4.3.1 Intra-Layer Dataflow

백본 datapath는 두 가지 arithmetic family로 분해된다. 첫 번째는 계수 행렬이 추론 동안 stationary한 연산들을 처리하는 projection path이다. 여기에는 patch embedding, QKV projection, output projection, 두 개의 feed-forward layer가 포함된다. Token tile $\mathbf X\in\mathbb R^{T_T\times T_{CI}}$와 weight tile $\mathbf W\in\mathbb R^{T_{CI}\times T_{CO}}$에 대해 projection engine은 다음을 계산한다.

```tex
\mathbf Y = \mathbf X\mathbf W
```

이때 output-stationary accumulation을 사용한다. 연산 유닛은 parallel factor $P_T$, $P_{CI}$, $P_{CO}$를 갖는 integer MAC array이다. 그 cycle count는 다음과 같이 근사된다.

```tex
C_{\mathrm{proj}}=
\left\lceil\frac{N_T}{P_T}\right\rceil
\left\lceil\frac{C_I}{P_{CI}}\right\rceil
\left\lceil\frac{C_O}{P_{CO}}\right\rceil
```

Weight가 stationary하므로, 주요 최적화 목표는 BRAM bank로부터의 재사용을 극대화하면서 partial sum을 local register에 유지하는 것이다.

두 번째 arithmetic family는 self-attention의 interaction path이다. 여기서는 effective weight가 런타임 token으로부터 동적으로 생성된다. Score 및 value 계산은 다음과 같다.

```tex
\mathbf S = \mathbf Q\mathbf K^\top
```

```tex
\mathbf O = \operatorname{Softmax}(\mathbf S)\mathbf V
```

Projection path와 달리, interaction path는 transient score storage, token reordering, score formation과 value aggregation 사이의 barrier에 의해 제약된다. 따라서 HBTXR은 stripe-wise score buffering을 사용한다. QK engine이 하나의 score stripe를 score SRAM에 기록하면, normalization engine이 그 stripe에 대해 max subtraction과 table-driven normalization을 수행하고, AV engine이 정규화된 stripe와 해당 value tile을 함께 소비한다. 이러한 **intra-layer dataflow** 선택은 전체 $N_T\times N_T$ score matrix가 온칩 메모리를 벗어나지 않도록 한다.

**Memory organization**  
메모리 시스템은 네 가지 bank로 나뉜다. stationary kernel을 저장하는 weight BRAM, 현재와 다음 token tile을 저장하는 token SRAM, 일시적인 attention stripe를 저장하는 score SRAM, 그리고 이전 pupil state와 확정된 Search anchor를 저장하는 state SRAM이다. 이러한 분리는 각 tensor의 실제 수명과 일치한다. Weight와 state는 장수명(long-lived)이고, score 데이터는 단수명이지만 대역폭 소모가 크며, token tile은 반복적으로 덮어쓰이므로 ping-pong buffering이 유리하다.

**Tiling and parallelism**  
주요 타일링 파라미터는 token tile $T_T$, input-channel tile $T_{CI}$, output-channel tile $T_{CO}$이다. Search와 Track은 shared front half에서 동일한 tile shape를 사용하므로, early-exit interface에 별도의 repacking이 필요 없다. Search 전용 deeper block은 호출 빈도가 낮기 때문에 head-level parallelism을 다소 보수적으로 둘 수 있고, front-half block은 더 큰 compute parallelism과 더 촘촘한 token buffering으로 지연시간을 우선 최적화한다.

#### 4.3.2 멀티스테이지 파이프라인 (Multi-Stage Pipeline)

Cross-stage 수준에서 HBTXR은 다음과 같은 **multi-stage pipeline**을 사용한다.

- S0) frame/event patch loading and tokenization
- S1) QKV 및 projection 계산
- S2) attention score 형성과 정규화
- S3) value aggregation과 MLP update
- S4) mode-gated head decoding

Search는 모든 $L^{\mathrm S}$ block에 대해 S0–S4를 반복 수행하고, Track은 처음 $L_c$개 block에 대해서만 S0–S4를 수행한 뒤 early-exit head로 분기한다.

Stage $i$의 workload를 $W_i$, 할당된 parallel factor를 $p_i$, synchronization overhead를 $\delta_i$라 하면, stage cadence는 다음과 같다.

```tex
C_i=\left\lceil\frac{W_i}{p_i}\right\rceil+\delta_i
```

파이프라인의 steady-state cadence는 다음과 같다.

```tex
C_{\mathrm{sys}}=\max_i C_i
```

모드 $m$에서 실행되는 backbone block 수를 $B_m$이라 하면, 모드 지연시간은 다음과 같이 근사된다.

```tex
T_m \approx T_{\mathrm{fill}} + B_mC_{\mathrm{sys}} + T_{\mathrm{drain}}
```

여기서 $B_{\mathrm{Search}}=L^{\mathrm S}$이고 $B_{\mathrm{Track}}=L_c$이다. 따라서 설계 목표는 특정 커널 하나만 최소화하는 것이 아니라, 타일링, 메모리 폭, compute parallelism을 조정하여 stage 간 $C_i$의 편차를 줄이는 데 있다.

**Cross-layer Coupling**  
Cross-layer coupling은 block $L_c$의 출력에서 나타난다. 중간 tensor $\mathbf F_{L_c}$는 deeper backbone으로 계속 진행할 수도 있고, Track head로 분기할 수도 있다. 이 cut point는 하나의 공통 tensor format을 갖는 전용 buffer boundary로 구체화된다. Search는 $\mathbf F_{L_c}$를 back-half input buffer에 기록하고 나머지 block으로 계속 진행한다. Track은 같은 tensor를 cut-point buffer에서 읽고, pooled token과 이전 상태를 결합한 뒤 deeper block을 우회한다. 이 분기점은 critical control path에 있으므로, tensor format, memory layout, scheduler interface가 함께 공동 설계된다.

**Scheduling and control**  
전역 정책은 CPU가 유지하지만, stage-level release는 하드웨어 로컬에서 수행된다. 각 pipeline stage는 compact FSM을 가지며, ready/valid 신호를 elastic FIFO를 통해 교환한다. 이 선택은 routing pressure를 줄이고, 단기적인 cadence mismatch를 허용하며, Search와 Track의 activation pattern이 전역 컨트롤러의 stall을 유발하지 않도록 해준다.

#### 4.3.3 스트리밍 지향 특징 전달 (Streaming-Oriented Feature Delivery)

임베디드 XR 배치에서는 연산보다 메모리 이동이 더 비싼 경우가 많다. 따라서 HBTXR은 **refresh delivery**와 **resident delivery**를 구분한다. Search는 frame token, event-summary token, 일시적 synchronization context를 적재하여 신뢰할 수 있는 anchor를 다시 생성하고, 확정된 anchor를 state SRAM에 기록하는 refresh path를 사용한다. Track은 resident delivery를 사용하며, event token, 이전 상태 token, 그리고 Track head에 필요한 compact context만을 순환시킨다.

이 구조는 다음의 온칩 메모리 모델로 요약된다.

```tex
M_{\mathrm{chip}} = M_{\mathrm{w}} + M_{\mathrm{tok}} + M_{\mathrm{score}}
+ M_{\mathrm{state}} + M_{\mathrm{fifo}}
```

여기서 $M_{\mathrm{w}}$는 resident weight, $M_{\mathrm{tok}}$는 ping-pong token SRAM, $M_{\mathrm{score}}$는 transient attention-score storage, $M_{\mathrm{state}}$는 accepted anchor와 previous state, $M_{\mathrm{fifo}}$는 inter-stage streaming buffer를 의미한다. Search는 전체 backbone을 통과하므로 $M_{\mathrm{tok}}$와 $M_{\mathrm{score}}$ 사용량이 증가한다. 반면 Track은 state SRAM을 재사용하고 early-exit boundary에서 종료함으로써 이들 항을 최소화한다.

**그림 5 캡션 번역**  
HBTXR 하드웨어 개요. CPU는 scheduler 측 품질 평가와 모드 선택을 수행하고, FPGA는 모달리티별 tokenization, 공유 front-half transformer core, $L_c$ 위치의 cut-point buffer, 그리고 전용 SRAM에 의해 지원되는 mode-gated Search/Track head를 실행한다.

### 4.4 패치 임베딩 모듈 (Patch Embedding Module)

Patch embedding module은 HBTXR의 모달리티 적응 프런트엔드이다. 이 모듈은 canonicalized frame patch 또는 accumulated event tensor를 입력받아, 공유 backbone이 사용할 수 있는 공통 token width로 변환한다. 연산 유닛은 line-buffered patch loader와 그 뒤의 resident-weight projection array로 구성된다. Patch 크기가 $P\times P$이면, 출력 token 수는 다음과 같다.

```tex
N_P = \frac{H}{P}\cdot\frac{W}{P}
```

하나의 token tile에 대한 front-end latency는 다음과 같이 근사된다.

```tex
C_{\mathrm{pe}}=
\left\lceil\frac{N_P}{P_T}\right\rceil
\left\lceil\frac{C_{\mathrm{in}}P^2}{P_{CI}}\right\rceil
\left\lceil\frac{D}{P_{CO}}\right\rceil
```

Frame tokenizer는 deterministic window extraction과 stable geometry를 우선하고, event tokenizer는 low-overhead packing과 previous-state injection을 우선한다. 두 front-end는 동일한 token width를 출력하므로, downstream transformer에는 모드별 구조 변경이 필요 없다.

### 4.5 트랜스포머 모듈 (Transformer Module)

Transformer module은 HBTXR에서 가장 큰 계산 코어를 이룬다. 각 block은 normalization, QKV 생성, score 형성, score 정규화, value aggregation, output projection, residual addition, feed-forward processing으로 구성된다. 하드웨어는 block 전체를 하나의 거대한 primitive로 취급하는 대신, projection engine과 interaction engine으로 computation unit을 분할하여 실제 연산자의 dataflow에 맞춘다.

**Computation units**  
Projection engine은 resident weight로 공급되는 banked integer MAC array이다. 이 엔진은 patch embedding, QKV 생성, output projection, 두 개의 MLP linear layer를 담당한다. Interaction engine은 score generator, Section 4.2.2의 table로 뒷받침되는 normalization unit, value aggregator로 이루어진다. 그 사이에는 전체 sequence가 아닌 하나의 stripe 또는 tile 크기에 맞춰진 token SRAM과 score SRAM이 위치한다.

**Memory and buffering**  
한 block 내부에서 token은 double buffering되어, 현재 tile을 처리하는 동안 다음 tile을 적재할 수 있다. Score SRAM은 전체 attention matrix를 저장하는 대신 stripe reuse에 맞춰 크기가 정해진다. 가장 중요한 persistent buffer는 block $L_c$ 뒤의 cut-point SRAM과, Search/Track이 함께 사용하는 state SRAM이다. 이러한 buffer는 cross-layer coupling이 하드웨어에서 실제로 해소되는 지점을 제공한다.

**Tiling and parallelism**  
Projection path는 output-stationary accumulation과 공격적인 output-channel parallelism을 사용한다. Interaction path는 token parallelism과 score-buffer width 사이의 균형을 맞춘다. Search와 Track은 front-half에서 동일한 타일링 파라미터를 공유하므로, early-exit tensor가 추가 변환 없이 Track head에 바로 전달될 수 있다. Back-half block은 Search 모드에서만 활성화되므로 보다 보수적인 parallelism을 사용할 수 있다. 즉, HBTXR은 균일한 최고 throughput을 추구하는 것이 아니라, 스케줄러가 만들어내는 모드 분포에서의 지연시간을 줄이는 방향으로 병렬성을 배치한다.

**그림 6 캡션 번역**  
HBTXR 하드웨어 파이프라인의 실행 타임라인. Search는 모든 backbone block과 refresh-oriented head를 스트리밍 방식으로 통과하는 반면, Track은 동일한 초기 stage를 재사용하고 block $L_c$ 뒤의 cut-point buffer에서 종료한 뒤 Track 측 head를 직접 호출한다.

### 4.6 헤드 모듈 (Head Module)

Head module은 Search, Event-Validation, Track, Mask, Eye-Region head로 구성된다. 이 head들은 공유 backbone보다 훨씬 가볍지만, Search와 Track 사이의 기능적 비대칭성을 실제로 구현한다. Search는 Search head, objectness logic, Eye-Region head, Mask head를 활성화하여 신뢰할 수 있는 anchor를 다시 획득하고 eye region을 검증한다. Track은 Track head와 이벤트 조건부 품질 logic을 활성화하여 이전 상태를 최소 지연으로 갱신한다. 최종 gaze 출력은 별도의 무거운 분기로부터 생성되는 것이 아니라, 수락된 accepted state로부터 생성된다.

Datapath 수준에서 head는 cut-point buffer 또는 final-block token buffer에 연결된 소형 streaming decoder로 구현된다. 이들의 weight footprint는 온칩에 resident하게 유지될 만큼 충분히 작다. 컨트롤러는 현재 모드에 필요한 head만 활성화하므로, 불필요한 연산을 줄이고 최종 pipeline stage에서의 메모리 중재도 단순화할 수 있다.

모드 조건부 하드웨어 매핑은 간단하게 기술될 수 있지만, 구현상 매우 중요하다. Search 모드에서는 frame tokenizer와 event-summary adapter가 활성화되고, 모든 $L^{\mathrm S}$ backbone block이 실행되며, Search·Mask·Eye-Region head가 활성화되고, 수락된 anchor가 refresh-validity metadata와 함께 state SRAM에 기록된다. Track 모드에서는 event tokenizer와 previous-state injector가 활성화되고, 처음 $L_c$개의 backbone block만 실행되며, cut-point buffer가 Track 및 Event-Validation head로 직접 전달되고, resident anchor가 제자리에서 갱신된다. Scheduled hybrid execution에서는 CPU가 매 invocation마다 이 두 경로 중 하나를 선택하고, FPGA는 mode packet을 해석하여 해당 front-end, memory route, head subset을 gating한다.

### 4.7 CPU–FPGA 작업 분할 (CPU–FPGA Workload Partition: Scheduler)

HBTXR은 Search와 Track의 런타임 의미론이 시스템 경계에서는 control-heavy하고, network core 내부에서는 tensor-heavy하기 때문에 이종적인 CPU–FPGA 구성을 채택한다. CPU는 frame/event acquisition, accumulation-window 관리, closed-eye handling, similarity 및 confidence 평가, 최종 Search/Track 결정을 담당한다. FPGA는 tokenization, shared-backbone inference, active-head execution, state-SRAM update를 담당한다.

런타임에서 CPU는 현재 operation mode, 이전에 수락된 state, scheduler 측 품질 메타데이터를 담은 mode packet을 전송한다. FPGA 측 controller는 이 패킷을 해석하여 해당하는 front-end, backbone span, head 조합을 활성화한다. Search와 Track은 단순히 하나의 정적 그래프에 들어가는 서로 다른 입력이 아니라, 배치 시스템의 두 개의 서로 다른 운용 상태이기 때문에, scheduler는 부수적 구현 세부사항이 아니라 알고리즘–하드웨어 contract의 일부가 된다.

종단간 지연시간은 다음과 같이 분해된다.

```tex
T_{\mathrm{e2e}} = T_{\mathrm{host}} + T_{\mathrm{io}} + T_{\mathrm{front}} + T_{\mathrm{bb}} + T_{\mathrm{head}} + T_{\mathrm{sync}}
```

각 항은 각각 host scheduling, host-device transfer, front-end tokenization, backbone execution, head decoding, synchronization overhead를 의미한다. Search invocation의 비율을 $\alpha$라 하면 평균 런타임은 다음과 같다.

```tex
T_{\mathrm{avg}} = \alpha T_{\mathrm{Search}} + (1-\alpha)T_{\mathrm{Track}}
```

이 식은 본 하드웨어 설계의 목적을 잘 보여준다. HBTXR은 균일한 synthetic throughput 숫자 하나를 최적화하는 것이 아니라, 스케줄러가 실제로 만들어내는 Search/Track 운용 분포에서의 성능을 최적화한다.

---

## 5. 실험 결과 (Experimental Results)

### 5.1 실험 설정 (Experimental Setup)

**Dataset**  
HBTXR은 동기화된 frame observation, event stream, 그리고 조밀한 pupil annotation으로 구성된 near-eye hybrid eye-tracking benchmark에서 평가된다. 각 sample은 canonicalized frame patch, polarity-split event tensor, 이전 tracking state, geometry-aware supervision target으로 이루어진다. 데이터셋은 subject-disjoint 방식으로 training, validation, testing으로 나뉘며, 이를 통해 최종 결과가 cross-subject generalization을 반영하도록 한다.

**Software and Hardware**  
모델은 PyTorch에서 학습된 뒤, 제안한 CPU–FPGA deployment model과 일치하는 host–device runtime으로 export된다. Host는 sensor I/O와 scheduling을 담당하고, 가속기는 Xilinx ZCU104 플랫폼에서 modality adaptation, shared-backbone inference, head decoding을 수행한다. 구현 흐름은 Vitis/PYNQ 기반 host control과, 각 invocation마다 CPU와 FPGA 사이에 교환되는 mode packet을 가정한다.

**Hyperparameters**  
HBTXR은 두 단계 학습을 사용하며, 입력은 $256\times256$으로 canonicalization된 프레임, $2\times256\times256$ event tensor, 5000개 이벤트의 fixed-count window, fast causal accumulation을 사용한다. Search pretraining 이후 Search, event-validation, track, mask, gaze, distillation-aware quality loss를 포함하는 hybrid fine-tuning을 수행한다. Optimizer는 AdamW이며, batch size는 8, learning rate는 $3\times10^{-4}$, weight decay는 $1\times10^{-4}$이다.

**Workstation**  
학습과 검증은 GPU 워크스테이션에서 수행되며, 런타임 평가는 Section 4.7에서 설명한 host–device execution model을 따른다.

**Runtime Instrumentation**  
동공 및 시선 정확도 외에도 Search success rate, end-to-end latency, effective update rate를 보고한다. 지연시간 분석을 위해 런타임은 host scheduling, host–device transfer, front-end adaptation, backbone execution, head decoding, synchronization component로 추가 분해된다.

### 5.2 제안하는 HBTXR 평가 (Proposed HBTXR Evaluation)

#### 5.2.1 프레임 기반 Search, 이벤트 기반 Track, 그리고 스케줄된 하이브리드 동작

프레임 기반 Search 경로, 이벤트 기반 Track 경로, 그리고 scheduled hybrid system의 운영 비교는 다음과 같다. Search 모드는 전체 student backbone을 통과하면서 0.58 px의 pupil error, $0.91^{\circ}$의 gaze error, 97.8\%의 Search success rate, 0.93 ms의 latency를 달성한다. Track 모드는 front-half cut point 이후에서 종료하고 Track 측 head만 활성화하기 때문에, 0.42 px의 pupil error, $0.68^{\circ}$의 gaze error, 0.57 ms의 latency를 달성하며, 이는 1754 Hz의 effective update rate에 해당한다. Scheduler 기반 hybrid operation에서는 강한 정확도를 유지하면서 평균 런타임이 0.63 ms가 된다.

#### 표: HBTXR의 모드별 정확도, 활성 백본 구간, 실행 시간

| 모드 | 백본 구간 | 활성 head | 정확도 | 성공률(%) | 지연시간(ms) | 처리율(Hz) | 전력(W)
|---|---|---|---|---:|---:|---:|---:|
| Search | $1{:}L^{\mathrm S}$ | Search, Mask, Eye-Region | 0.58 px / $0.91^{\circ}$ | 97.8 | 0.93 | 1075 | 1.68 |
| Track | $1{:}L_c$ | Track, Event-Validation | **0.42 px / $0.68^{\circ}$** | -- | **0.57** | **1754** | **1.54** |
| Scheduled | policy driven | Search + Track mix | 0.45 px / $0.71^{\circ}$ | 97.8 | 0.63 | 1587 | 1.60 |

주: Search는 전체 백본과 모든 refresh-oriented head를 활성화하고, Track은 front half와 Track/Event-Validation head만 활성화한다. Scheduled operation은 실제 운용 분포에 대한 가중 평균값이다. 전력 수치는 정적 플랫폼 예산과 모드별 동적 activity를 조합해 얻은 ZCU104 보드 전력 추정치이다.

Scheduled 결과는 제3의 독립적인 네트워크에서 나온 것이 아니라, Section 3.4의 런타임 정책이 만들어낸 평균 결과이다. 실제로 scheduled latency는 Search invocation 비율과, 품질 저하로 인해 Track에서 Search로 되돌아가는 비용에 의해 결정된다. 추정 전력 경향 역시 동일한 실행 비대칭성을 따른다. Search는 전체 백본과 모든 refresh-oriented head를 사용하므로 가장 높고, Track은 front-half cut point에서 종료하므로 더 낮으며, scheduled hybrid operation은 그 중간에 위치한다.

#### 5.2.2 하드웨어 구현 결과: 자원, 전력, 지연시간 분해

ZCU104 구현을 위한 HBTXR의 추정 resource budget은 다음과 같다. 가장 가까운 ZCU104 reference point와 비교할 때, 모드 비대칭 front-half sharing, cut-point buffering, state-SRAM organization을 유지하면서도 resource footprint는 대략 10–20\% 낮다. 그 결과 목표 footprint는 58,112 LUT, 38,656 FF, 156 BRAM36, 344 DSP이며, 이는 XCZU7EV 자원의 각각 25.2\%, 8.4\%, 50.0\%, 19.9\%에 해당한다.

#### 표: ZCU104 상의 HBTXR 추정 Resource Utilization

| 항목 | 용량 | 사용량 | 활용률(%) |
|---|---:|---:|---:|
| FPGA device | -- | XCZU7EV | -- |
| LUT | 230,400 | 58,112 | 25.2 |
| FF | 460,800 | 38,656 | 8.4 |
| BRAM36 | 312 | 156 | 50.0 |
| DSP | 1,728 | 344 | 19.9 |
| Clock | -- | 200 MHz | -- |

LUT 예산은 주로 token routing, scheduler FSM, mode-gated head control에 사용되고, FF는 pipeline register와 control state에, BRAM36은 token SRAM, score SRAM, state SRAM, lookup table에, DSP는 projection MAC과 dyadic re-scale unit에 할당된다. 지연시간 감소의 원인은 앞서와 동일하다. Track 경로는 front-half backbone을 재사용하고 deeper Search-only block을 우회하며, Track 측 head만 활성화한다. 이 구조에서 지배적인 지연 항은 front-end tokenization, shared front-half backbone, mode-gated head decode이며, Search는 여기에 back-half backbone과 refresh-oriented head 비용이 추가된다. 모드별 power 추정값도 동일한 분해를 바탕으로 산출되므로, 위 자원 예산과 일관성을 유지한다. Post-route timing과 보드 레벨 측정은 향후 연구에서 이 추정치를 더욱 정교화하는 데 사용될 것이다.

### 5.3 다른 안구 추적 알고리즘과의 비교 (Comparison with Other Eye-Tracking Algorithms)

다음 비교는 대표적인 안구 추적 방법과 HBTXR을 modality, dataset, output target, previous-state conditioning, explicit Search/Track decomposition, source-reported accuracy, runtime 측면에서 대조한 것이다. 이 비교에서 HBTXR은 **명시적인 Search/Track 의미론**과 **sub-millisecond Track latency 하에서의 pupil + gaze metric 동시 보고**라는 점에서 구별된다.

#### 표: 대표 안구 추적 알고리즘과의 소스 정렬 비교

| 방법 | 모달리티 | 데이터셋 | 출력 | 이전 상태 사용 | Search/Track | 정확도(원문 보고) | 런타임 |
|---|---|---|---|---|---|---|---|
| EyeCoD | Frame | OpenEDS | Gaze | No | No | $3.23^{\circ}$ gaze error | 385.66 FPS system |
| EV-Eye | Event | EV-Eye | Pupil | No | No | 0.3231 px; P1 98.87\% | 0.9438 ms |
| FACET | Event | EV-Eye+ | Pupil | Yes | No | 0.2030 px; P1 99.59\% | 0.5302 ms |
| SEE-D | Event | 3ET+ | Pupil | No | No | p10 99.53\%; 3.71 px dist. | 0.70 ms |
| JaneEye-Net | Event | 3ET+ | Pupil | No | No | 2.45 px | 0.50 ms @ 2000 FPS |
| EX-Gaze | Hybrid | EV-Eye / OpenEDS | Pupil / gaze | Limited | Implicit | 1.33 px (sacc.); 1.49 px (smooth) | 0.407 ms average |
| HBTXR | Hybrid | XR hybrid | Pupil + gaze | Yes | Yes | 0.42 px; $0.68^{\circ}$ | 0.57 ms Track / 0.93 ms Search |

### 5.4 다른 가속기와의 비교 (Comparison with Other Accelerators)

다음 비교는 프레임 기반 안구 추적, 이벤트 기반 안구 추적, 일반적인 트랜스포머 가속, 객체 검출 가속 등 대표적 accelerator implementation과 HBTXR을 대조한다. HBTXR은 배치 시스템이 단일 패스 설계가 아니라 **mode dependent system**이기 때문에, Search, Track, Scheduled Hybrid의 세 열로 나누어 제시한다. HBTXR의 자원과 전력 값은 ZCU104 구현 목표에 대한 추정치이다.

#### 표: 대표 가속기와의 배치 지향 비교

| 지표 | EyeCoD | SEE-D | JaneEye | ViT accel. | Det. accel. | HBTXR-S | HBTXR-T | HBTXR-H |
|---|---|---|---|---|---|---|---|---|
| Task | Gaze tracking | Pupil tracking | Pupil tracking | ViT inference | Object detection | Search refresh | Track update | Scheduled hybrid |
| Platform | 28nm ASIC | ZCU102 | 12nm ASIC | VCK190 | ZCU104 | ZCU104 | ZCU104 | ZCU104 |
| Dataset | OpenEDS | 3ET+ | 3ET+ | ImageNet-1K | VOC | XR hybrid | XR hybrid | XR hybrid |
| Clock | 370 MHz | -- | 400 MHz | 425 MHz | 200 MHz | 200 MHz est. | 200 MHz est. | 200 MHz est. |
| Resource util. | 512 MAC / 316 KB SRAM | 130K LUT / 90K FF / 1092 BRAM / 1606 DSP | 0.28 mm$^2$ ASIC area | 669K LUT / 312 DSP / 1007 BRAM | 64.8K LUT / 43.7K FF / 398 DSP / 781 KB OCM | 58.1K LUT / 38.7K FF / 156 BRAM / 344 DSP | 58.1K LUT / 38.7K FF / 156 BRAM / 344 DSP | 58.1K LUT / 38.7K FF / 156 BRAM / 344 DSP |
| Accuracy | $3.23^{\circ}$ gaze | 3.71 px dist. | 2.45 px pupil | 71.05\% Top-1 | 64.8\% mAP | 0.58 px / $0.91^{\circ}$ | 0.42 px / $0.68^{\circ}$ | 0.45 px / $0.71^{\circ}$ |
| Latency | 4.17 ms* | 0.70 ms | 0.50 ms | 0.136 ms | 12.53 ms* | 0.93 ms | 0.57 ms | 0.63 ms |
| Power | 0.154 W | 3.86 W | 0.038 W | 46.7 W | 1.91 W | 1.68 W | 1.54 W | 1.60 W |
| Throughput | 240 FPS | 1428 FPS† | 2000 FPS | 7118 FPS | 79.8 FPS | 1075 Hz | 1754 Hz | 1587 Hz |
| Power eff. | -- | 2.29 mJ/inf. | 567 GOP/s/W | 381.0 GOP/s/W | 80.5 GOPS/W | 1.56 mJ/inf. | 0.88 mJ/inf. | 1.01 mJ/inf. |
| Frame eff. | 1555 FPS/W‡ | 370 FPS/W‡ | 53.0K FPS/W‡ | 152 FPS/W‡ | 41.9 FPS/W | 640 FPS/W | 1139 FPS/W | 992 FPS/W |

주:  
* 별도의 standalone latency가 명시되지 않은 경우, source-reported FPS의 역수로 계산함.  
† source-reported 0.70 ms latency로부터 계산함.  
‡ source-reported throughput과 power로부터 계산함.  
HBTXR-S/T/H는 각각 Search, Track, Scheduled Hybrid를 의미한다.

이 비교가 보여주는 HBTXR의 위치는 다음과 같다. EyeCoD는 프레임 중심 predict–focus 파이프라인으로서, 프레임 처리를 중심으로 semantic robustness를 우선할 때 얻을 수 있는 성능을 대표한다. SEE-D와 JaneEye는 희소 event 중심 datapath 또는 ASIC 특화 datapath가 제공하는 효율성의 상한을 보여준다. ViT acceleration baseline은 고처리량 token inference를 위해 aggressive한 multi-stage pipelining, cross-layer coupling 제어, tiling, LUT 기반 nonlinear processing이 유효함을 입증한다. Detector baseline은 ZCU104 위에서 algorithm–hardware co-optimization이 resource footprint, latency, energy의 균형을 어떻게 잡는지 보여준다. 이에 비해 HBTXR은 **mode-asymmetric execution**, 즉 일률적인 one-pass throughput 대신 Search/Track specialization, early exit, scheduler-visible control semantics를 택한다는 점에서 차별화된다.

### 5.5 어블레이션 연구 (Ablation Study)

다음 어블레이션은 HBTXR의 각 설계 요소가 정확도와 지연시간에 미치는 영향을 보여준다. Geometry supervision contract가 가장 큰 정확도 기여를 보이며, deployment-oriented slimming은 early-exit Track path가 프루닝된 깊이에서도 품질을 유지할 수 있게 하는 핵심 요소이다. 특히 암묵적 모델 프루닝을 제거하고 배치 시 full teacher를 그대로 사용하면 정확도는 약간 좋아지지만 지연시간이 뚜렷하게 악화된다. 반대로 Feature KD나 RKD를 제거하면 nominal Track datapath는 바뀌지 않더라도 slim student의 품질이 저하된다. 이는 Stage 2가 단순한 compression 단계가 아니라, front-half Track exit를 통계적으로 성립 가능하게 만드는 단계임을 보여준다.

#### 표: HBTXR 어블레이션 결과

| 변형 | Pupil error (px) | Gaze error (deg) | Latency (ms) |
|---|---:|---:|---:|
| Full HBTXR | 0.42 | 0.68 | 0.57 |
| w/o geometry-stable supervision | 0.49 | 0.80 | 0.58 |
| w/o decoupled heads | 0.47 | 0.75 | 0.62 |
| w/o implicit model pruning | 0.41 | 0.67 | 0.76 |
| w/o Track early-exit at $L_c$ | 0.43 | 0.69 | 0.70 |
| w/o Feature KD | 0.45 | 0.72 | 0.57 |
| w/o RKD | 0.46 | 0.73 | 0.57 |
| w/o Search/Track scheduler | 0.46 | 0.73 | 0.71 |
| w/o state-SRAM reuse | 0.42 | 0.68 | 0.69 |

---

## 6. 결론 (Conclusion)

본 논문은 온디바이스 XR을 위한 알고리즘–하드웨어 공동 설계 하이브리드 안구 추적 프레임워크 HBTXR을 제시하였다. 핵심 아이디어는 하이브리드 안구 추적을 하나의 무차별적 멀티모달 회귀로 다루는 대신, 명시적인 런타임 의미론을 갖는 Search/Track 시스템으로 재구성하는 데 있다. Search는 전체 백본을 사용하는 프레임 기반 relocalization을 통해 강건한 anchor refresh를 제공하고, Track은 배치 지향 암묵적 모델 프루닝으로 도입된 front-half cut point를 활용하여 저지연 이벤트 기반 residual update를 수행한다. 그리고 scheduler는 이 둘을 배치 인지 하드웨어 실행과 결합한다.

본 연구의 주요 기여는 supervision, transformer 설계, 런타임 제어, 하드웨어 매핑을 하나의 틀 안에서 공동으로 다룬 점이다. 하드웨어 설계는 streaming execution, multi-stage pipeline balance, intra-layer dataflow specialization, 그리고 Track cut point에서의 cross-layer coupling을 중심으로 조직되었다. 실험 결과 HBTXR은 0.42 px의 pupil error, $0.68^{\circ}$의 gaze error, 0.57 ms의 Track latency를 달성하면서도 강건한 Search recovery를 유지하였다. 향후 연구에서는 ZCU104에서의 post-route timing과 board-level measurement를 보고하고, 양자화 및 distillation 기반 slimming 흐름을 더욱 정교화하여 완전한 구현 수렴(implementation closure)을 달성할 계획이다.

---

## 번역 메모

- Search, Track, Event-Validation, Eye-Region, Mask, scheduler, cut point, backbone, token, state SRAM 등은 논문 내 기술 용어이므로 원어를 병기하거나 그대로 유지했습니다.
- 수식은 의미 보존을 위해 원문 표기와 동일하게 두었습니다.
- 참고문헌은 별도 요청이 없어서 번역 대상에서 제외했습니다.
