아래는 바로 붙여 넣을 수 있는 형태의 연결 문단이다. 기준 주제는 `XR용 event/hybrid eye tracking`이다.

**1. 국문: 서론 말미 -> Related Work 연결 문단**

기존 시선 추적 연구는 사용되는 센서 모달리티와 시스템 설계 목표에 따라 서로 다른 발전 경로를 보여 왔다. 전통적인 frame-based 방법은 풍부한 공간 정보를 바탕으로 높은 해석 가능성과 안정적인 gaze 추정을 제공해 왔으나, 최근 XR 환경에서 요구되는 고주파, 저지연, 저전력 조건을 만족시키는 데에는 한계를 보인다. 이에 따라 event camera를 활용한 event-based 방법과, frame과 event를 결합한 hybrid 방법이 대안으로 부상하고 있으며, 동시에 실제 디바이스 탑재를 위한 accelerator 및 deployment-oriented 연구도 활발히 진행되고 있다. 따라서 본 절에서는 관련 연구를 입력 모달리티와 시스템 지향성에 따라 frame-based eye tracking, event-based eye tracking, hybrid event-frame eye tracking, 그리고 eye tracking accelerator 및 deployment-oriented methods로 구분하여 정리한다.

**2. 국문: Related Work 말미 -> 제안 방법 연결 문단**

이상의 관련 연구를 종합하면, frame-based 방법은 성숙한 appearance modeling에 기반한 안정적인 추정 성능을 제공하지만 latency와 전력 소모 측면에서 제약이 있으며, event-based 방법은 고속 안구 운동에 대한 반응성과 전력 효율이 우수한 반면 정보 희소성과 학습 난도가 높다. 또한 hybrid 방법은 정확도와 반응성의 균형이 우수하지만 시스템 복잡도가 증가하고, accelerator 중심 연구는 실시간성과 탑재 가능성을 확보하는 데 유리하나 범용성이 제한될 수 있다. 이러한 한계를 고려할 때, XR 환경에서 요구되는 accuracy-latency-power trade-off를 동시에 만족시키기 위해서는 모달리티 특성과 시스템 제약을 함께 반영한 새로운 접근이 필요하다. 이에 본 연구에서는 [제안 방법의 핵심 아이디어: 예, event-frame 상호보완 정보 활용 / 저전력 on-device 구조 / latency-aware inference]에 기반한 새로운 eye tracking 방법을 제안한다.

**3. 영문: End of Introduction -> Related Work transition**

Prior studies on eye tracking have evolved along different directions depending on the sensing modality and system-level objective. Conventional frame-based methods have provided reliable gaze estimation with rich spatial information and strong interpretability, but they remain limited in meeting the stringent requirements of XR systems, particularly in terms of high-frequency sensing, low latency, and low power consumption. In response, event-based approaches and hybrid event-frame approaches have recently emerged as promising alternatives, while accelerator- and deployment-oriented studies have focused on practical on-device feasibility. Accordingly, this section reviews prior work by categorizing it into four groups: frame-based eye tracking, event-based eye tracking, hybrid event-frame eye tracking, and eye tracking accelerator/deployment-oriented methods.

**4. 영문: End of Related Work -> Proposed Method transition**

The above review indicates that each line of research addresses only part of the design space required by practical XR eye tracking. Frame-based methods benefit from mature appearance modeling but suffer from substantial latency and power overhead. Event-based methods offer clear advantages in temporal resolution and energy efficiency, yet they face challenges associated with sparse information and training difficulty. Hybrid approaches provide a better balance between accuracy and responsiveness, although at the cost of increased system complexity, while accelerator-oriented methods improve deployability but often sacrifice generality. Therefore, there remains a need for a new approach that jointly considers modality characteristics and system constraints to better satisfy the accuracy-latency-power trade-off demanded by XR applications. Motivated by this gap, this work proposes [core idea of the proposed method], which is designed to achieve [target objective: e.g., high-frequency, low-latency, power-efficient, on-device eye tracking].

**5. 바로 쓸 수 있는 문장 치환 예시**

- `[제안 방법의 핵심 아이디어]`
  `event stream의 고시간 해상도와 frame의 안정적 appearance 정보를 결합하는 latency-aware hybrid inference 구조`
- `[target objective]`
  `high-frequency and low-latency gaze estimation under on-device power constraints`

**6. 논문체를 더 맞추고 싶으면 이렇게 바꾸면 된다**

제안 방법이 `event-only`이면 마지막 문장을 이렇게 바꾸면 된다.

- 국문:
  `이에 본 연구에서는 event stream의 비동기성과 희소성을 직접 활용하여, 고속 안구 운동에 대해 낮은 지연시간과 높은 전력 효율을 동시에 달성할 수 있는 새로운 event-based eye tracking 방법을 제안한다.`
- 영문:
  `Accordingly, this work presents a novel event-based eye-tracking method that directly exploits the asynchronous and sparse nature of event streams to achieve both low latency and high energy efficiency for rapid eye movements.`

제안 방법이 `hybrid`이면 다음이 더 자연스럽다.

- 국문:
  `이에 본 연구에서는 frame 기반 appearance 정보와 event 기반 motion 정보를 상호보완적으로 결합함으로써, 정확도와 반응성을 동시에 확보할 수 있는 hybrid eye tracking 방법을 제안한다.`
- 영문:
  `Accordingly, this work proposes a hybrid eye-tracking method that combines appearance cues from frames and motion cues from events in a complementary manner, thereby improving both accuracy and responsiveness.`

원하면 다음 답변에서 이 문단들을 포함한 `Introduction + Related Work + Our Motivation` 전체 흐름까지 한 번에 맞춰서 작성하겠다.

======================================================================================================================================================================================================================================
아래는 `Introduction 후반부 -> Related Work -> Research Gap / Motivation -> Contributions`까지 한 번에 이어지는 초안이다.  
논문 주제가 아직 고정되지 않았으므로, 방법 이름과 핵심 아이디어는 `[ ]`로 남겨 두었다. 국문과 영문을 각각 제공한다.

---

## 국문 초안

### 1. Introduction 후반부

XR(Extended Reality) 환경에서의 시선 추적은 gaze-based interaction, foveated rendering, user attention modeling, biometric authentication 등 다양한 응용의 핵심 구성 요소로 자리잡고 있다. 특히 wearable XR 디바이스에서는 빠른 안구 운동을 정확히 포착하면서도 낮은 지연시간과 전력 소모를 동시에 만족해야 하므로, 기존의 범용 시선 추적 기법을 그대로 적용하기 어렵다. 그러나 실제 XR 시스템은 제한된 연산 자원, 배터리 용량, 센서 대역폭, 그리고 엄격한 실시간성 요구조건을 동시에 갖기 때문에, 단순히 추정 정확도만을 높이는 접근만으로는 실용적인 성능을 확보하기 어렵다.

기존 연구는 주로 frame-based eye tracking을 중심으로 발전해 왔으며, 이들 방법은 풍부한 spatial appearance 정보를 활용할 수 있다는 장점으로 인해 오랫동안 표준적인 접근으로 간주되어 왔다. 그러나 프레임 기반 센싱은 높은 데이터 중복도와 motion blur 문제를 수반하며, 빠른 saccade를 추적해야 하는 XR 환경에서는 latency 및 power efficiency 측면의 제약이 분명하다. 이러한 한계를 완화하기 위해 최근에는 event camera를 활용한 event-based eye tracking과, frame과 event를 결합한 hybrid event-frame eye tracking이 활발히 연구되고 있다. 또한 실제 기기 탑재를 위한 accelerator 및 deployment-oriented 연구 역시 중요한 흐름을 형성하고 있다.

그럼에도 불구하고, 기존 방법들은 여전히 몇 가지 한계를 가진다. Frame-based 방법은 높은 accuracy와 안정적인 appearance modeling에 강점을 가지지만 실시간성과 전력 효율 측면에서 불리하다. Event-based 방법은 높은 temporal resolution과 낮은 latency를 제공하지만, 정보 희소성과 representation 설계의 난도로 인해 안정적인 추정이 쉽지 않다. Hybrid 방법은 이러한 두 모달리티의 상호보완성을 활용할 수 있으나, 센서 동기화와 fusion 구조 설계로 인해 시스템 복잡성이 증가한다. 또한 deployment-oriented 방법은 실질적인 on-device feasibility를 보여주지만, 특정 하드웨어 플랫폼에 대한 강한 의존성으로 인해 범용성이 제한될 수 있다. 따라서 XR 환경에서 요구되는 accuracy-latency-power trade-off를 균형 있게 만족시키기 위한 새로운 접근이 필요하다.

이에 본 연구에서는 `[제안 방법의 핵심 아이디어]`에 기반한 `[방법 이름]`을 제안한다. 제안 방법은 `[핵심 목표 1]`, `[핵심 목표 2]`, 그리고 `[핵심 목표 3]`를 동시에 달성하는 것을 목표로 하며, XR 환경에서의 실시간 시선 추적에 보다 적합한 설계를 제공한다.

---

### 2. Related Work

#### 2.1 Frame-based Eye Tracking

전통적인 frame-based eye tracking 방법은 RGB 또는 IR 프레임 영상으로부터 pupil, iris, eyelid 등의 시각적 특징을 직접 추출하고 이를 기반으로 gaze를 추정한다. 대표적으로 *Etracker*는 CNN을 이용하여 blink frame을 제거하고 coarse gaze를 예측한 뒤, geometric model을 통해 최종 gaze를 정밀 보정하는 구조를 제안하였다. *FlatTrack*은 lensless FlatCam과 lightweight gaze regressor를 결합하여 wearable device에서의 form factor를 줄이고자 하였으며, *HE-Tracker*는 HMD 환경에서 eye image와 head motion을 함께 사용함으로써 off-axis 카메라 조건에서의 gaze estimation 성능을 향상시켰다. 이러한 방법들은 풍부한 spatial information을 활용할 수 있으므로 pupil/iris의 구조적 정보를 직접 다루기 용이하며, 기존 computer vision 및 deep learning pipeline과의 호환성이 높다. 그러나 frame acquisition과 processing 과정에서 높은 bandwidth와 computation cost가 요구되므로, motion blur와 temporal redundancy 문제가 크고, XR 환경에서 요구되는 high-frequency, low-latency operation에는 구조적 한계를 가진다.

#### 2.2 Event-based Eye Tracking

Event camera를 활용한 방법은 brightness change를 비동기적으로 감지하는 센싱 메커니즘을 이용하여 초저지연 및 저전력 eye tracking을 구현하고자 한다. *E-Gaze*와 *E-Track*은 event stream으로부터 gaze 또는 pupil location을 직접 추정하는 대표적인 초기 연구이며, 특히 *E-Track*은 fixed-number event representation과 ROI 기반 추론을 통해 계산량을 효과적으로 절감하였다. 또한 *FACET*은 ellipse modeling을 활용하여 빠르고 정확한 pupil localization을 수행하였고, *Swift-Eye*는 blinking 및 partial occlusion 상황에서의 강건성을 향상시키기 위해 anti-blink 설계를 도입하였다. 한편 *AISE*, *SEEN*, *Event-based low-power spiking gaze estimation*과 같은 연구들은 adaptive event sampling, spiking neural network, neuromorphic computation 등을 이용하여 energy efficiency를 더욱 강화하였다. 이러한 방법들은 높은 temporal resolution, sparse computation, low latency, low power consumption 측면에서 뚜렷한 장점을 보인다. 그러나 fixation과 같이 motion이 적은 구간에서는 정보가 희소하며, event representation의 비직관성으로 인해 학습 및 디버깅이 어렵고, 데이터셋 및 benchmark의 성숙도 역시 frame-based 연구에 비해 제한적인 편이다.

#### 2.3 Hybrid Event-frame Eye Tracking

Frame과 event는 서로 상호보완적인 정보를 제공하므로, 이를 함께 활용하는 hybrid event-frame 접근 또한 중요한 연구 흐름으로 자리잡고 있다. *EX-Gaze*는 hybrid event-frame camera를 활용하여 on-device XR 환경에서 high-frequency, low-latency gaze tracking을 구현하고자 하였다. *An Efficient Eye Tracking System Based on Event Camera and SCNN-AKF Fusion*은 event stream과 grayscale information을 sparse CNN 및 adaptive Kalman filtering과 결합하여 반응성과 안정성을 동시에 확보하였다. 또한 *Modeling State Shifting via Local-Global Distillation for Event-Frame Gaze Tracking*은 frame의 안정적인 appearance cue와 event의 dynamic motion cue를 state shifting 관점에서 통합하였으며, *Real-Time Gaze Tracking with Event-Driven Eye Segmentation*은 frame-based segmentation stage를 event-driven 방식으로 보완하여 end-to-end latency를 줄였다. Hybrid 방법은 일반적으로 frame-only 방법보다 빠르고 event-only 방법보다 shape 및 appearance 복원이 용이하므로 accuracy와 responsiveness의 균형이 우수하다. 반면 센서 동기화, modality fusion, hardware integration 측면의 복잡도가 증가하며, 이에 따른 구현 비용 역시 커지는 단점이 있다.

#### 2.4 Eye Tracking Accelerator and Deployment-oriented Methods

실제 XR 기기 탑재를 위해서는 알고리즘 성능뿐 아니라 latency, power, area, form factor를 함께 고려한 accelerator 및 deployment-oriented 연구가 필수적이다. *EyeCoD*는 FlatCam 기반 eye tracking 알고리즘과 accelerator를 공동 설계하여 communication overhead와 system form factor를 동시에 줄였으며, optics-algorithm-hardware co-design의 가능성을 제시하였다. *JaneEye*는 event-based eye tracking을 위한 12 nm ASIC accelerator를 제안하여 2K-FPS 수준의 처리율과 낮은 energy per frame을 목표로 하였고, *SEE*는 sparse CNN과 FPGA SoC를 함께 최적화하여 sub-millisecond latency를 달성하였다. 또한 *Sub-Millisecond Event-Based Eye Tracking on a Resource-Constrained Microcontroller*는 MCU급 자원에서도 실시간 추론이 가능함을 보였으며, *Retina*는 event camera와 spiking hardware를 결합한 neuromorphic deployment 방향을 제시하였다. 이 계열의 연구는 실제 제품화 가능성을 직접적으로 평가한다는 점에서 중요하다. 그러나 대부분 특정 hardware platform과 dataflow에 강하게 최적화되어 있으므로 portability와 generality 측면에서는 제한이 있다.

---

### 3. Research Gap and Motivation

이상의 관련 연구를 종합하면, 기존 방법들은 각각 특정 측면에서는 의미 있는 진전을 보였으나, practical XR eye tracking이 요구하는 정확도, 지연시간, 전력 효율, 구현 복잡도를 동시에 만족시키는 데에는 여전히 한계가 존재한다. Frame-based 방법은 appearance modeling 측면에서 안정적이나 고속 추적과 저전력 구현에 불리하며, event-based 방법은 반응성과 효율 측면에서 우수하지만 정보 희소성과 representation 난이도로 인해 일관된 성능 확보가 쉽지 않다. Hybrid 방법은 이러한 상호보완성을 활용할 수 있으나, multimodal fusion의 복잡성과 시스템 통합 비용이 증가한다. Deployment-oriented 방법은 실제 탑재 가능성을 보여주지만, 종종 특정 플랫폼에 한정된 최적화에 머무르는 경우가 많다.

따라서 본 연구는 이러한 공백을 해결하기 위하여 `[문제 정의]`를 대상으로 `[핵심 기술 요소 1]`, `[핵심 기술 요소 2]`, `[핵심 기술 요소 3]`를 통합한 새로운 접근을 제안한다. 본 연구의 기본 가정은 XR 환경에서의 eye tracking 성능은 단일 모달리티의 우수성만으로 결정되지 않으며, sensing modality, inference structure, 그리고 system-level constraint를 함께 고려할 때 비로소 실용적인 수준의 성능을 달성할 수 있다는 점이다.

---

### 4. Contributions

본 연구의 주요 기여는 다음과 같이 정리할 수 있다.

1. `[제안 문제]`를 해결하기 위한 `[방법 이름]`을 제안한다.  
2. `[핵심 모듈 또는 알고리즘 요소]`를 통해 `[기존 한계]`를 완화한다.  
3. `[실험 환경 또는 디바이스]` 상에서 제안 방법의 효과를 검증하고, 정확도와 latency/power 측면에서 기존 방법 대비 우수성을 보인다.  
4. XR 환경에서 요구되는 accuracy-latency-power trade-off 관점에서 제안 방법의 실용적 의미를 분석한다.

---

## English Draft

### 1. End of Introduction

Eye tracking has become a key enabling component in XR systems for a wide range of applications, including gaze-based interaction, foveated rendering, user attention modeling, and biometric authentication. In wearable XR devices, however, eye-tracking systems are required to capture rapid eye movements accurately while simultaneously satisfying strict constraints on latency and power consumption. Because practical XR systems operate under limited compute resources, battery capacity, sensor bandwidth, and real-time deadlines, improving estimation accuracy alone is insufficient for achieving deployable performance.

Conventional eye-tracking research has primarily evolved around frame-based methods, which have long been regarded as the standard approach due to their ability to exploit rich spatial appearance information. However, frame-based sensing suffers from high temporal redundancy and motion blur, and it becomes increasingly inefficient when high-frequency eye motion must be tracked under strict latency and energy constraints. To address these issues, recent studies have actively explored event-based eye tracking and hybrid event-frame eye tracking, while another line of work has focused on accelerator design and deployment-oriented optimization for practical devices.

Despite these efforts, several important limitations remain. Frame-based methods provide accurate and stable appearance modeling, but they are disadvantaged in latency and power efficiency. Event-based methods offer high temporal resolution and low latency, yet they often suffer from sparse information and representation difficulty. Hybrid methods can exploit the complementarity of frame and event modalities, but they introduce additional complexity in synchronization and multimodal fusion. Accelerator-oriented studies improve deployability, but their benefits are often tied to specific hardware platforms, limiting generality. Therefore, a new approach is still needed to better satisfy the accuracy-latency-power trade-off required by practical XR eye-tracking systems.

To this end, this work proposes `[method name]`, a new eye-tracking approach based on `[core idea of the proposed method]`. The proposed method is designed to simultaneously achieve `[objective 1]`, `[objective 2]`, and `[objective 3]`, thereby providing a more practical solution for real-time XR eye tracking.

---

### 2. Related Work

#### 2.1 Frame-based Eye Tracking

Traditional frame-based eye-tracking methods estimate gaze from RGB or IR image frames by directly extracting visual cues such as the pupil, iris, and eyelids. For example, *Etracker* combines a CNN-based coarse gaze predictor and blink-frame filtering module with a geometric refinement stage for accurate gaze estimation. *FlatTrack* reduces the form factor of wearable eye trackers by using a lensless FlatCam together with a lightweight gaze regressor. *HE-Tracker* further improves gaze estimation in HMD settings by jointly modeling eye images and head motion, which is particularly useful under off-axis camera configurations. These approaches benefit from rich spatial information, making pupil/iris structure modeling and calibration relatively straightforward, and they are highly compatible with mature computer vision and deep learning pipelines. However, frame acquisition and processing incur high bandwidth and computational overhead, which leads to substantial temporal redundancy and motion blur. As a consequence, frame-based methods are fundamentally limited when applied to high-frequency, low-latency XR eye tracking.

#### 2.2 Event-based Eye Tracking

Event-based eye-tracking methods exploit the asynchronous sensing mechanism of event cameras to achieve ultra-low-latency and low-power operation. Representative early approaches such as *E-Gaze* and *E-Track* directly estimate gaze or pupil location from event streams, while *E-Track* further reduces computation through a fixed-number event representation and an ROI-based inference scheme. *FACET* formulates pupil localization through ellipse modeling and achieves fast and accurate tracking using event data alone. *Swift-Eye* improves robustness under blinking and partial occlusion through an anti-blink design tailored to high-frequency near-eye motion analysis. In addition, studies such as *AISE*, *SEEN*, and *Event-based low-power spiking gaze estimation* emphasize energy efficiency by incorporating adaptive event sampling, spiking neural networks, and neuromorphic computation. The main advantages of event-based methods are high temporal resolution, sparse computation, low latency, and low power consumption. Nevertheless, they often suffer from insufficient information during fixation or low-motion periods, and the event representation itself is less intuitive than frame data, which makes training and debugging more difficult. Moreover, the benchmark ecosystem for event-based eye tracking is still less mature than that of frame-based approaches.

#### 2.3 Hybrid Event-frame Eye Tracking

Because frame and event data provide complementary information, hybrid event-frame approaches have emerged as an important research direction. *EX-Gaze* utilizes a hybrid event-frame camera to enable high-frequency and low-latency gaze tracking in on-device XR environments. *An Efficient Eye Tracking System Based on Event Camera and SCNN-AKF Fusion* combines event streams and grayscale information through sparse CNN processing and adaptive Kalman filtering to improve both responsiveness and temporal stability. *Modeling State Shifting via Local-Global Distillation for Event-Frame Gaze Tracking* integrates stable appearance cues from frames and dynamic motion cues from events through a state-shifting formulation. Similarly, *Real-Time Gaze Tracking with Event-Driven Eye Segmentation* reduces end-to-end latency by using event-driven processing to complement or accelerate the otherwise expensive frame-based segmentation stage. In general, hybrid methods achieve a favorable balance between accuracy and responsiveness: they are typically faster than frame-only methods and more informative than event-only methods in recovering appearance and shape details. However, these benefits come at the cost of increased system complexity, including sensor synchronization, multimodal fusion design, and hardware integration.

#### 2.4 Eye Tracking Accelerator and Deployment-oriented Methods

Beyond algorithmic design, practical XR deployment also requires explicit consideration of latency, energy efficiency, area, and form factor. *EyeCoD* jointly designs a FlatCam-based eye-tracking algorithm and a dedicated accelerator to reduce both communication overhead and system form factor, illustrating the value of optics-algorithm-hardware co-design. *JaneEye* presents a 12-nm ASIC accelerator for event-based eye tracking, targeting 2K-FPS throughput with very low energy per frame. *SEE* co-optimizes a sparse CNN and an FPGA SoC platform to achieve sub-millisecond latency, while *Sub-Millisecond Event-Based Eye Tracking on a Resource-Constrained Microcontroller* demonstrates that low-latency inference is feasible even on MCU-class hardware. In addition, *Retina* explores a neuromorphic deployment direction by combining event cameras with spiking hardware. These studies are important because they directly evaluate practical feasibility rather than reporting algorithmic accuracy alone. However, they are often tightly coupled to specific sensors, dataflows, and hardware platforms, which limits portability and generality.

---

### 3. Research Gap and Motivation

The above review shows that existing methods have made meaningful progress along individual dimensions, but they still do not fully satisfy the combined requirements of practical XR eye tracking in terms of accuracy, latency, power efficiency, and implementation complexity. Frame-based methods provide stable appearance modeling but remain inefficient for high-speed and low-power operation. Event-based methods are attractive for responsiveness and efficiency, but they often struggle with sparse information and representation difficulty. Hybrid methods leverage the complementarity of modalities, yet they increase fusion complexity and system integration cost. Deployment-oriented studies improve practical feasibility but are frequently limited to platform-specific optimization.

Motivated by this gap, this work addresses `[problem definition]` by introducing a new approach that integrates `[technical component 1]`, `[technical component 2]`, and `[technical component 3]`. The central premise of this work is that practical XR eye-tracking performance cannot be achieved by optimizing a single modality or metric in isolation; instead, sensing modality, inference structure, and system-level constraints must be considered jointly.

---

### 4. Contributions

The main contributions of this work are summarized as follows.

1. We propose `[method name]` for `[target problem]`.  
2. We introduce `[core module or algorithmic component]` to alleviate `[specific limitation of prior work]`.  
3. We validate the proposed method on `[datasets / devices / platforms]` and demonstrate improvements over prior approaches in terms of accuracy and latency/power efficiency.  
4. We analyze the practical implications of the proposed method from the perspective of the accuracy-latency-power trade-off required in XR scenarios.

---

## 바로 치환할 수 있는 표현

논문이 `event-only`면 아래 표현이 자연스럽다.

- 국문:
  `[제안 방법의 핵심 아이디어] = event stream의 비동기성과 희소성을 직접 활용하는 저지연 추론 구조`
- 영문:
  `[core idea of the proposed method] = a low-latency inference framework that directly exploits the asynchronous and sparse nature of event streams`

논문이 `hybrid event-frame`이면 아래 표현이 자연스럽다.

- 국문:
  `[제안 방법의 핵심 아이디어] = frame 기반 appearance cue와 event 기반 motion cue를 상호보완적으로 결합하는 hybrid inference 구조`
- 영문:
  `[core idea of the proposed method] = a hybrid inference framework that combines appearance cues from frames and motion cues from events in a complementary manner`

다음 단계로 진행하려면 둘 중 하나를 선택하면 된다.

1. 당신의 제안 방법에 맞게 위 초안을 완전히 실문장으로 채워서 완성본으로 만들기  
2. 이 초안을 `IEEE 스타일`에 맞게 더 압축해서 1~1.5페이지 분량으로 정리하기


======================================================================================================================================================================================================================================
아래는 `IEEE 스타일`에 맞춰 `Related Work + Research Gap` 중심으로 압축한 버전이다. 분량은 본문 기준으로 약 `1~1.5페이지` 수준을 염두에 두고 정리했다.

**국문 압축본**

**Related Work**

XR 환경에서의 eye tracking은 입력 모달리티와 시스템 지향성에 따라 크게 frame-based 방법, event-based 방법, hybrid event-frame 방법, 그리고 deployment-oriented 방법으로 구분할 수 있다. 각 계열은 정확도, 지연시간, 전력 소모, 그리고 시스템 복잡도 측면에서 서로 다른 trade-off를 가진다.

Frame-based eye tracking은 RGB 또는 IR 영상으로부터 pupil, iris, eyelid 등의 시각적 특징을 직접 추출하여 gaze를 추정하는 전통적인 접근이다. 예를 들어, *Etracker*는 CNN 기반 coarse estimation과 geometric refinement를 결합하여 gaze를 추정하였고, *FlatTrack*은 lensless FlatCam을 활용하여 wearable form factor를 줄이고자 하였다. 또한 *HE-Tracker*는 head motion과 eye image를 함께 사용하여 HMD 환경에서의 gaze estimation 성능을 향상시켰다. 이러한 방법은 풍부한 spatial appearance 정보를 활용할 수 있으므로 구조적 해석성과 안정성이 높고, 기존 computer vision pipeline과의 호환성도 우수하다. 그러나 높은 frame rate가 요구되는 XR 환경에서는 motion blur, temporal redundancy, 높은 bandwidth 및 computation cost로 인해 latency와 power efficiency 측면의 한계가 명확하다.

이러한 한계를 보완하기 위해 event camera를 활용한 event-based eye tracking이 활발히 연구되고 있다. *E-Gaze*와 *E-Track*은 event stream으로부터 gaze 또는 pupil location을 직접 추정하는 대표적인 초기 연구이며, 특히 *E-Track*은 fixed-number event representation과 ROI 기반 추론을 통해 계산량을 줄였다. *FACET*은 ellipse modeling을 이용하여 빠르고 정확한 pupil localization을 수행하였고, *Swift-Eye*는 blink 및 partial occlusion 환경에서의 강건성을 높이기 위한 anti-blink 설계를 제안하였다. 또한 *AISE*, *SEEN*, *Event-based low-power spiking gaze estimation*과 같은 연구들은 adaptive event sampling, spiking neural networks, neuromorphic processing을 통해 energy efficiency를 더욱 강화하였다. Event-based 방법은 높은 temporal resolution, sparse computation, low latency, low power consumption이라는 장점을 갖지만, motion이 적은 구간에서 정보가 희소하고 representation 설계와 학습 난도가 높으며, benchmark 생태계가 아직 충분히 성숙하지 않았다는 한계를 가진다.

최근에는 frame과 event의 상호보완성을 활용하는 hybrid event-frame 방법도 주목받고 있다. *EX-Gaze*는 hybrid event-frame camera를 이용하여 on-device XR 환경에서 high-frequency, low-latency gaze tracking을 달성하고자 하였다. *An Efficient Eye Tracking System Based on Event Camera and SCNN-AKF Fusion*은 event와 grayscale 정보를 sparse CNN 및 adaptive Kalman filtering과 결합하여 안정성과 반응성을 동시에 확보하였으며, *Modeling State Shifting via Local-Global Distillation for Event-Frame Gaze Tracking*은 frame의 appearance cue와 event의 motion cue를 state shifting 관점에서 통합하였다. 또한 *Real-Time Gaze Tracking with Event-Driven Eye Segmentation*은 event-driven processing을 이용하여 frame-based segmentation stage의 부담을 줄였다. 이 계열은 일반적으로 frame-only보다 빠르고 event-only보다 shape 및 appearance 복원이 용이하여 accuracy와 responsiveness의 균형이 우수하다. 반면 센서 동기화, fusion 구조, hardware integration 측면에서 시스템 복잡도가 증가한다.

한편 practical XR deployment를 위해서는 알고리즘 자체뿐 아니라 latency, energy, area, form factor를 함께 고려하는 연구도 필요하다. *EyeCoD*는 FlatCam 기반 알고리즘과 accelerator를 공동 설계하여 communication overhead와 form factor를 동시에 줄였고, *JaneEye*는 event-based eye tracking을 위한 12 nm ASIC accelerator를 제안하였다. *SEE*는 sparse CNN과 FPGA SoC를 함께 최적화하여 sub-millisecond latency를 달성하였으며, *Sub-Millisecond Event-Based Eye Tracking on a Resource-Constrained Microcontroller*는 MCU급 자원에서도 실시간 추론이 가능함을 보였다. 이러한 연구는 실제 탑재 가능성을 직접 평가한다는 점에서 중요하지만, 대부분 특정 sensor-platform 조합에 강하게 최적화되어 generality와 portability가 제한된다.

이상의 연구를 종합하면, frame-based 방법은 안정적인 appearance modeling에 강점을 가지지만 실시간성과 전력 효율이 제한적이고, event-based 방법은 반응성과 효율이 우수하지만 정보 희소성과 representation difficulty가 문제로 남아 있다. Hybrid 방법은 성능 균형이 우수하나 시스템 복잡도가 높으며, deployment-oriented 방법은 practical feasibility를 확보하는 대신 범용성이 낮다. 따라서 XR 환경에서 요구되는 accuracy-latency-power trade-off를 보다 균형 있게 만족시키기 위해서는 모달리티 특성과 시스템 제약을 함께 고려하는 새로운 접근이 필요하다. 이에 본 연구에서는 `[제안 방법의 핵심 아이디어]`에 기반한 `[방법 이름]`을 제안한다.

---

**English Compressed Version**

**Related Work**

Eye-tracking studies for XR systems can be broadly categorized into frame-based methods, event-based methods, hybrid event-frame methods, and deployment-oriented approaches. These categories differ substantially in terms of accuracy, latency, power consumption, and system complexity, and practical XR eye tracking is largely shaped by the trade-off among these factors.

Frame-based eye tracking is the conventional paradigm in which gaze is estimated from RGB or IR images by explicitly modeling visual cues such as the pupil, iris, and eyelids. For example, *Etracker* combines CNN-based coarse estimation with geometric refinement, while *FlatTrack* employs a lensless FlatCam to reduce the form factor of wearable eye trackers. *HE-Tracker* further improves HMD gaze estimation by jointly modeling eye images and head motion. These methods benefit from rich spatial appearance information and are compatible with mature computer vision pipelines, making them relatively stable and interpretable. However, their reliance on dense image acquisition leads to high bandwidth and computation cost, as well as temporal redundancy and motion blur, which limits their suitability for high-frequency, low-latency XR applications.

To overcome these limitations, event-based eye tracking has attracted increasing attention. Early representative methods such as *E-Gaze* and *E-Track* directly estimate gaze or pupil location from event streams, with *E-Track* further reducing computation through fixed-number event representation and ROI-based inference. *FACET* performs fast and accurate pupil localization through ellipse modeling, whereas *Swift-Eye* improves robustness under blinking and partial occlusion through an anti-blink design. In addition, studies such as *AISE*, *SEEN*, and *Event-based low-power spiking gaze estimation* improve efficiency by incorporating adaptive event sampling, spiking neural networks, and neuromorphic processing. Event-based methods offer clear advantages in temporal resolution, low latency, sparse computation, and low power operation. Nevertheless, they often suffer from insufficient information during low-motion periods, and their event representation is less intuitive, making training and debugging more difficult. Their benchmark ecosystem is also less mature than that of frame-based approaches.

Hybrid event-frame eye tracking has recently emerged as a promising alternative because frame and event data provide complementary information. *EX-Gaze* leverages a hybrid event-frame camera to achieve high-frequency and low-latency gaze tracking in on-device XR settings. *An Efficient Eye Tracking System Based on Event Camera and SCNN-AKF Fusion* combines event streams and grayscale information using sparse CNN processing and adaptive Kalman filtering, while *Modeling State Shifting via Local-Global Distillation for Event-Frame Gaze Tracking* integrates frame appearance cues and event motion cues through a state-shifting formulation. *Real-Time Gaze Tracking with Event-Driven Eye Segmentation* further reduces end-to-end latency by using event-driven processing to complement the expensive frame-based segmentation stage. In general, hybrid methods offer a favorable balance between accuracy and responsiveness. However, these gains come at the cost of increased complexity in sensor synchronization, multimodal fusion, and hardware integration.

In parallel, deployment-oriented studies focus on making eye tracking practical for real XR hardware. *EyeCoD* jointly designs a FlatCam-based tracking algorithm and a dedicated accelerator to reduce both communication overhead and system form factor. *JaneEye* presents a 12-nm ASIC accelerator for event-based eye tracking, and *SEE* co-optimizes sparse CNN inference on an FPGA SoC for sub-millisecond latency. *Sub-Millisecond Event-Based Eye Tracking on a Resource-Constrained Microcontroller* demonstrates real-time feasibility on MCU-class hardware. While these approaches directly address practical latency and energy constraints, they are often tightly coupled to specific platforms, limiting portability and generality.

Overall, prior work has made substantial progress, but no single category fully satisfies the accuracy-latency-power trade-off required by practical XR eye tracking. Frame-based methods remain computationally heavy, event-based methods face sparse-information and representation challenges, hybrid methods increase system complexity, and deployment-oriented methods are often hardware-specific. Therefore, a new method that jointly considers modality characteristics and system constraints is still needed. Motivated by this gap, this work proposes `[method name]`, which is based on `[core idea of the proposed method]`.

---

바로 다음 단계로 할 수 있는 작업은 두 가지다.

1. 위 압축본을 당신의 제안 방법에 맞게 `[방법 이름]`, `[핵심 아이디어]`까지 채워 완성본으로 바꾸기  
2. 이 텍스트를 `IEEE 논문 형식의 LaTeX 문단`으로 바로 변환하기