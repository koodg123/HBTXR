======================================================================================================================================================================================================================================
아래는 업로드한 구조를 기준으로 다듬은 `IEEE 스타일` 초안이다.  
기본은 `영문 본문`으로 작성했고, 요청하신 한국어 초록도 함께 정리했다.  
참고문헌 번호는 아직 확정되지 않았으므로 `[ ]` 형태로 두었다.

---

# 0. Title

**HBTXR: An Algorithm-Hardware Co-Designed Hybrid Eye Tracking System for On-Device XR Applications**

---

# 1. Abstract

## English Abstract

Eye tracking is a key enabling technology for extended reality (XR), supporting gaze-based interaction, foveated rendering, and privacy-preserving on-device perception. However, existing approaches still suffer from a structural trade-off between semantic robustness and temporal responsiveness. Frame-based methods provide stable visual cues but incur high bandwidth cost and relatively large latency, whereas event-based methods offer high temporal resolution but remain limited in relocalization stability and representation consistency.

To address these limitations, this paper presents **HBTXR**, an algorithm-hardware co-designed hybrid eye tracking system that tightly integrates **Frame Search** and **Event Track** for on-device XR applications. HBTXR is built upon four key components. First, we introduce a dense-annotation-aware canonical training contract that stabilizes geometric supervision across heterogeneous sensor inputs. Second, we design a hybrid frame-event eye tracking transformer with decoupled search, track, and mask heads for robust pupil localization and relocalization. Third, we develop a two-stage host-side Search/Track scheduler that supports low-latency runtime adaptation through CPU-FPGA workload partitioning. Fourth, we propose an FPGA accelerator architecture featuring a transformer engine, streaming-based feature reuse, and FSM-based runtime control.

Experimental results show that HBTXR achieves a pupil-center error of **0.42 pixels**, a gaze error of **0.68°**, and an end-to-end latency of **0.57 ms** in tracking mode. These results demonstrate that HBTXR provides a practical solution for real-time on-device eye tracking in next-generation XR systems.

## 국문 초록

시선 추적은 확장현실(XR)에서 시선 기반 인터랙션, 중심와 렌더링, 그리고 프라이버시 보존형 온디바이스 인식을 지원하는 핵심 기술이다. 그러나 기존 방법들은 의미적 강건성과 시간적 응답성 사이의 구조적 trade-off를 완전히 해소하지 못하였다. 프레임 기반 방법은 안정적인 시각 단서를 제공하지만 높은 대역폭 요구와 상대적으로 큰 지연을 수반하며, 이벤트 기반 방법은 높은 시간 해상도를 제공하지만 재탐색 안정성과 표현 일관성 측면에서 한계를 보인다.

본 논문에서는 이러한 한계를 극복하기 위해 프레임 검색(Frame Search)과 이벤트 추적(Event Track)을 긴밀하게 통합한 알고리즘-하드웨어 공동 설계 기반 하이브리드 시선 추적 시스템 **HBTXR**을 제안한다. 제안하는 HBTXR은 네 가지 핵심 요소로 구성된다. 첫째, 이기종 센서 입력 전반에서 기하학적 감독 신호를 안정화하는 조밀 주석 기반 정규화 학습 체계를 제안한다. 둘째, 견고한 동공 위치 추정과 재탐색을 위한 하이브리드 프레임-이벤트 시선 추적 트랜스포머를 설계한다. 셋째, 저지연 런타임 적응을 위해 CPU-FPGA 워크로드 분할을 지원하는 2단계 호스트 측 Search/Track 스케줄러를 구성한다. 넷째, 트랜스포머 엔진, 스트리밍 기반 특징 재사용, FSM 기반 런타임 제어를 포함하는 FPGA 기반 하드웨어 가속기 구조를 제안한다.

실험 결과, 제안하는 HBTXR은 추적 모드에서 **0.42 픽셀의 동공 중심 오차**, **0.68°의 시선 오차**, 그리고 **0.57 ms의 지연 시간**을 달성하였다. 이러한 결과는 HBTXR이 차세대 XR 기기를 위한 온디바이스 실시간 시선 추적의 실용적인 해법이 될 수 있음을 보여준다.

---

# 2. Introduction

Eye tracking has become a fundamental component of modern extended reality (XR) systems, enabling gaze-based interaction, foveated rendering, attention-aware interfaces, and privacy-preserving on-device perception [ ]. In particular, near-eye gaze estimation is essential for improving user experience and system efficiency in head-mounted XR devices, where gaze information must be acquired and processed in real time under stringent energy and latency constraints. Since rapid eye motions such as saccades evolve within sub-millisecond to millisecond time scales, practical XR eye trackers must provide not only high estimation accuracy but also sufficiently high temporal responsiveness.

Conventional eye-tracking systems have been largely developed around frame-based imaging pipelines [ ]. These methods benefit from semantically rich visual information and often provide stable pupil and gaze estimation under controlled settings. However, dense frame acquisition introduces high bandwidth demand, substantial temporal redundancy, and increased power consumption, making it difficult to satisfy the real-time requirements of wearable XR devices. Their performance also degrades under high-speed eye motion due to motion blur and limited effective temporal resolution.

Event-based eye tracking has therefore emerged as an attractive alternative [ ]. Event cameras asynchronously capture brightness changes with high temporal resolution and sparse output, making them well suited for low-latency and low-power sensing. Prior studies have shown that event-based methods can significantly improve responsiveness in fast eye-motion scenarios. Nevertheless, purely event-based approaches often face difficulties in semantic stabilization, relocalization, and representation consistency, especially when event density becomes low or when robust re-initialization is required. In practice, this leads to a trade-off: frame-based methods are semantically stable but temporally expensive, while event-based methods are temporally efficient but less robust in global recovery.

Hybrid event-frame eye tracking has recently been explored to bridge this gap [ ]. Representative studies such as EX-Gaze demonstrate that combining frame-based relocalization with event-based tracking can improve both robustness and responsiveness. Similarly, event-only methods such as FACET show that carefully designed event representations and geometry-aware modeling can achieve efficient pupil localization. However, existing hybrid systems remain fragmented across algorithm design, runtime scheduling, and hardware realization. In particular, prior works have not sufficiently integrated (1) a geometry-stable end-to-end training contract across heterogeneous inputs, (2) a unified hybrid transformer architecture for search and track, (3) a runtime-aware scheduling policy for low-latency switching, and (4) an accelerator-friendly co-design for practical on-device XR deployment.

To make hybrid eye tracking truly practical for XR systems, these components must be designed jointly. Specifically, a deployable system requires a canonical input contract for heterogeneous sensing, geometry-stable supervision targets, a real-time scheduler for mode switching, and a backbone structure that is amenable to efficient acceleration. Motivated by these requirements, this paper presents **HBTXR**, an algorithm-hardware co-designed hybrid eye tracking system for on-device XR applications. As illustrated in Fig. 1, HBTXR combines dense annotation-based canonicalization, dual frame/event sensing, a shared Partial DeiT-Tiny backbone, decoupled search/track/mask heads, a host-side finite-state runtime controller, and CPU-FPGA workload partitioning into a unified framework.

The main contributions of this work are summarized as follows.

1. We propose a **dense-annotation-aware canonical training contract** that stabilizes geometric supervision across heterogeneous frame and event inputs.
2. We design an **end-to-end hybrid event-frame eye tracking transformer** with decoupled search, track, and mask heads for robust localization and relocalization.
3. We develop a **two-stage Search/Track scheduler** with host-side runtime control for low-latency adaptive execution under CPU-FPGA partitioning.
4. We present an **FPGA-oriented algorithm-accelerator co-design**, including a transformer engine, streaming-based feature reuse, and FSM-based runtime scheduling, for practical on-device XR deployment.

The remainder of this paper is organized as follows. Section II reviews prior work on eye tracking and hardware-oriented deployment. Section III presents the proposed HBTXR algorithm. Section IV describes the accelerator architecture and runtime scheduling framework. Section V reports experimental results and comparisons with prior methods and accelerators. Finally, Section VI concludes the paper.

---

# 3. Related Work

## A. Frame-Based Eye Tracking

Frame-based eye tracking has long been the dominant paradigm for gaze estimation. These methods typically rely on RGB or infrared images to extract semantically meaningful visual cues such as the pupil, iris, eyelids, and corneal reflections [ ]. Representative approaches include pupil segmentation- and detection-based pipelines, geometric gaze estimation, and CNN-based appearance modeling. For example, Etracker combines CNN-based coarse gaze prediction with geometric refinement, while HE-Tracker incorporates head motion cues to improve gaze estimation in head-mounted displays. Frame-based systems are attractive because they provide stable semantic information and are compatible with mature computer vision pipelines.

Despite these advantages, frame-based methods are fundamentally constrained by sensing and processing overhead. Dense image acquisition results in substantial bandwidth and memory traffic, and high frame rates are required to capture rapid eye motion without severe temporal aliasing. This makes frame-only methods increasingly inefficient for XR devices, where latency, power, and form factor are tightly constrained. As a result, although frame-based methods remain effective for semantically rich localization and recovery, they are less suitable as standalone solutions for sub-millisecond on-device XR eye tracking.

## B. Event-Based Eye Tracking

Event-based eye tracking leverages the asynchronous sensing property of event cameras to capture fine-grained eye motion with low latency and sparse output [ ]. Early works such as E-Gaze and E-Track directly estimate gaze or pupil location from event streams, demonstrating the potential of event sensing for high-speed eye tracking. FACET further shows that pure event-based ellipse modeling can localize the pupil efficiently, while SEE and JaneEye highlight the feasibility of low-latency event-based eye tracking under hardware-aware system design. In addition, recent methods have explored adaptive event slicing, sparse neural representations, and spiking computation to further improve energy efficiency.

The primary advantage of event-based methods lies in their temporal responsiveness. Because only brightness changes are transmitted, event streams naturally reduce redundant sensing and can support much faster reaction to eye motion than frame-based cameras. However, event-only pipelines often suffer from weak semantic stability, especially during low-motion periods or large appearance changes. They also tend to be more vulnerable to relocalization failure because the global scene context available in frames is absent or limited. Thus, although event-based eye tracking is highly attractive for fast tracking, it often requires additional mechanisms to recover robustness and re-initialization capability.

## C. Hybrid Event-Frame Eye Tracking

Hybrid eye tracking aims to combine the semantic richness of frames with the temporal responsiveness of events. Among prior works, EX-Gaze is a representative example showing that frame-based relocalization and event-based tracking can be integrated to improve robustness and latency simultaneously. Other hybrid approaches combine grayscale information and event streams through fusion modules, temporal filtering, or multi-branch architectures [ ]. These methods suggest that search and track should not be treated as isolated functions but rather as complementary stages within a unified runtime pipeline.

Nevertheless, prior hybrid systems remain limited in several aspects. First, many approaches rely on loosely coupled fusion rather than end-to-end geometry-aware supervision across heterogeneous inputs. Second, search and track are often integrated only algorithmically, without explicit runtime scheduling support. Third, existing hybrid pipelines are rarely designed together with accelerator constraints, making practical on-device XR deployment difficult. These limitations indicate that hybrid eye tracking should be revisited from a system-level perspective in which sensing, model architecture, scheduling, and acceleration are co-designed.

## D. Eye Tracking Accelerator and Deployment-Oriented Methods

A parallel line of research has focused on making eye tracking practical for embedded and XR devices through hardware acceleration and deployment-aware optimization. EyeCoD demonstrates optics-algorithm co-design by combining a FlatCam-based eye tracking pipeline with a dedicated accelerator. SEE co-optimizes sparse event-driven inference on an FPGA SoC platform, and JaneEye proposes an ASIC architecture for high-throughput event-based eye tracking. These studies show that low-latency eye tracking is achievable when the algorithm and hardware stack are jointly optimized.

However, most prior accelerator-oriented designs are tailored either to conventional CNN-style pipelines or to sparse event processing engines. The gap between recent transformer-based hybrid tracking algorithms and accelerator-friendly system design remains largely open. In particular, there is still limited work on unifying hybrid frame/event search-track logic with transformer inference, host-side runtime scheduling, and FPGA-oriented dataflow optimization in a single XR-oriented system. This gap motivates the proposed HBTXR framework.

## E. Discussion and Position of This Work

The above review indicates that no existing category fully satisfies the combined requirements of semantic robustness, temporal responsiveness, and deployability for practical XR eye tracking. Frame-based methods are semantically reliable but expensive in latency and bandwidth. Event-based methods are highly responsive but weaker in relocalization and stability. Hybrid methods improve the balance between the two, yet they often lack runtime-aware and hardware-aware integration. Accelerator-oriented methods improve feasibility but are typically specialized to limited model classes or sensing assumptions.

In contrast, HBTXR is designed to explicitly bridge these gaps. It jointly incorporates dense-annotation-aware canonical supervision, an end-to-end hybrid transformer for search and track, a host-controlled Search/Track scheduler, and an FPGA-oriented accelerator architecture. Therefore, the proposed system is positioned not merely as another hybrid eye tracker, but as an integrated algorithm-hardware co-designed solution for practical on-device XR applications.

---

필요하면 다음 단계로 바로 이어서 정리할 수 있다.

1. 이 초안을 `IEEE LaTeX 형식`으로 변환  
2. `Section IV: Proposed HBTXR Algorithm`까지 같은 톤으로 계속 작성  
3. `Table I comparison 문장`과 `Fig. 1 caption`까지 같이 작성


======================================================================================================================================================================================================================================