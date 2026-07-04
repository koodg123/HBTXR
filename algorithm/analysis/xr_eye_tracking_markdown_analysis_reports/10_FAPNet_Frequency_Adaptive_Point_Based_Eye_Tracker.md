# Paper Summary: FAPNet: An Effective Frequency Adaptive Point-based Eye Tracker

## Analysis Coverage

| Item | Description |
|---|---|
| Source PDF | `fapnet.pdf` |
| Coverage | PDF 본문, 표, 그림 설명, 수식, 실험 결과를 텍스트 추출 기준으로 분석 |
| Not Executed | 코드 실행, dataset download, training reproduction, hardware synthesis는 수행하지 않음 |
| Limitation | PDF에 없는 supplementary/code/reproduction 세부사항은 `논문에 명시되지 않음`으로 표시 |

## Paper Information

| Item | Description |
|---|---|
| **Title** | FAPNet: An Effective Frequency Adaptive Point-based Eye Tracker |
| **Authors** | Xiaopeng Lin, Hongwei Ren, Bojun Cheng |
| **Venue** | arXiv / workshop-style paper |
| **Year** | 2024 |
| **Research Field** | Point-based event eye tracking |
| **Paper Link / DOI** | 논문 PDF 기준 분석; DOI는 파일별로 상이하거나 논문에 명시되지 않음 |
| **Code / Project Link** | 논문에 명시되지 않음 |
| **Analysis Coverage** | Main paper / Appendix included only if present in PDF |

---

# 1. Overview

## 1.1 Problem Statement

- **Target Problem:** frame/voxel conversion은 event sparsity와 fine-grained temporal information을 손실하고, 입력 해상도에 따라 compute가 증가하는 문제.
- **Research Objective:** raw event point cloud를 직접 처리하는 frequency-adaptive point-based tracker를 제안.
- **Why This Problem Matters:** XR, AR/VR, wearable healthcare, foveated rendering, gaze interaction은 low latency, high frequency, low power eye tracking을 요구한다. Event camera는 sparse asynchronous events를 제공하지만, 이를 정확하고 효율적으로 pupil/gaze output으로 변환하는 알고리즘과 system design이 핵심 병목이다.

## 1.2 Core Idea

- **Core Idea:** event point cloud를 sampling/grouping하고, intra-group local aggregation과 inter-group temporal aggregation을 통해 pupil 위치를 회귀한다.
- **Proposed Approach:** event point cloud를 일정 개수로 sampling하고 standardization한다. point set은 KNN/FPS류 grouping을 거쳐 local extractor와 attention으로 local feature를 만들며, inter-sample temporal model이 eye movement context를 반영한다.
- **Key Differentiator:** 기존 frame-based 또는 dense event-frame processing 대비 event sparsity, temporal dynamics, hardware/system constraints 중 하나 이상을 직접 활용한다.

## 1.3 Main Contributions

1. raw event cloud 기반 point network FAPNet 제안
2. frequency adaptive window expanding으로 event sparsity에 대응
3. Inter-Sample LSTM으로 sample 간 temporal dependency 모델링
4. PEPNet 대비 적은 compute로 유사/우수 정확도 달성

## 1.4 Representative Results

| Dataset | Task | Metric | Result | Comparison Condition | Source |
| --- | --- | --- | --- | --- | --- |
| SEET synthetic / evaluation | Pupil tracking | p3 / p5 / p10 / MSE | 0.920 / 0.991 / 0.996 / 1.56 | FAPNet 180×240 point; PEPNet p10 0.998, MSE 1.57 | Experimental tables |
| SEET | Efficiency | Params / FLOPs | 0.29M / 58.7M | PEPNet 0.64M / 443M | Experimental tables |
| AIS 2024 challenge context | Point-based method | p10 | 97.95% | EFFICIENT team result | Challenge survey/Table 1 context |

## 1.5 Brief Overview

> 이 논문은 Point-based event eye tracking 분야에서 frame/voxel conversion은 event sparsity와 fine-grained temporal information을 손실하고, 입력 해상도에 따라 compute가 증가하는 문제.를 해결하기 위해 event point cloud를 sampling/grouping하고, intra-group local aggregation과 inter-group temporal aggregation을 통해 pupil 위치를 회귀한다.라는 설계 방향을 제안한다. 핵심 방법은 event point cloud를 일정 개수로 sampling하고 standardization한다. point set은 KNN/FPS류 grouping을 거쳐 local extractor와 attention으로 local feature를 만들며, inter-sample temporal model이 eye movement context를 반영한다.이다. 실험적으로는 위 표의 대표 수치처럼 accuracy, latency, model complexity, robustness 중 하나 이상의 축에서 기존 방법 대비 개선을 보고한다. 다만 dataset, resolution, online/offline setting, hardware platform 차이가 있어 다른 논문과의 직접 비교는 주의가 필요하다.

---

# 2. Research Background

## 2.1 Research Context

Event-based eye tracking은 Dynamic Vision Sensor (DVS)의 asynchronous event stream을 사용한다. 각 event는 일반적으로 `(x, y, t, p)` 형태로 표현되며, brightness change가 threshold를 넘을 때 발생한다. 따라서 fixation 구간에서는 event가 적고, saccade나 blink처럼 빠른 움직임이 발생하면 event density가 증가한다.

## 2.2 Conventional Approaches

기존 eye tracking은 frame-based camera, infrared illumination, pupil/iris segmentation, ellipse fitting, gaze regression을 사용하는 경우가 많다. DNN 기반 approach는 정확도는 높지만 dense frame processing 때문에 bandwidth, latency, power 측면에서 XR wearable에 부담이 된다.

## 2.3 Limitations of Existing Approaches

- frame-based sensing은 fixed frame rate와 motion blur, high bandwidth 문제를 가진다.
- event-based sensing은 static information이 부족하고, sparse/noisy/irregular event distribution을 처리해야 한다.
- recurrent/attention model은 정확도는 높지만 compute와 memory cost가 커질 수 있다.
- hardware-friendly model과 high-accuracy temporal modeling 사이에 trade-off가 존재한다.

## 2.4 Research Motivation

저자들은 event camera의 high temporal resolution과 sparse output을 활용하면 XR/AR/VR wearable에서 high-frequency eye tracking을 더 효율적으로 구현할 수 있다고 본다. 본 논문은 그중에서도 raw event point cloud를 직접 처리하는 frequency-adaptive point-based tracker를 제안.에 초점을 맞춘다.

## 2.5 Significance of the Problem

이 문제는 foveated rendering의 gaze-to-photon latency, gaze interaction, cognitive/medical analysis, authentication 등에서 직접적인 실용적 의미가 있다. 특히 saccade는 빠르게 발생하므로 sub-ms 또는 kHz-level tracking이 요구될 수 있다.

---

# 3. Related Works

## 3.1 Related Work Categories

| Category | Representative Work | Core Idea | Limitation | Difference from This Paper |
| --- | --- | --- | --- | --- |
| Frame-based eye/gaze tracking | U-Net, RITnet, DeepVOG, EllSeg, appearance/model-based trackers | near-eye frame에서 pupil/iris/eye region을 segment/detect하고 gaze를 회귀 | fixed frame rate, high bandwidth/power, motion blur, kHz tracking 한계 | event stream 또는 event-frame hybrid를 사용해 temporal resolution과 latency/efficiency를 개선 |
| Event-based eye tracking | EBVEYE, EV-Eye, 3ET/3ET+, E-Track, Retina, SEE, challenge methods | DVS event의 sparsity와 high temporal resolution을 활용 | static information loss, sparse labels, noisy events, event representation 선택 문제 | 각 논문은 특정 representation/model/tracker/hardware co-design으로 trade-off를 개선 |
| Efficient edge/hardware-aware models | MobileNet, sparse CNN, SNN, FPGA/ASIC accelerator, quantization | low latency, low power, small model을 위한 구조 최적화 | accuracy degradation, hardware dependency, reproducibility 조건 차이 | 해당 논문별로 model/hardware/system-level 최적화 범위가 다름 |

## 3.2 Position of This Paper

분석 관점에서 이 논문은 위 related work 중 다음 위치에 해당한다.

- **Primary axis:** Point-based event eye tracking
- **Input modality:** event-only 또는 event-frame hybrid 여부는 논문별로 다르며, 본 논문은 `event point cloud를 sampling/grouping하고, intra-group local aggregation과 inter-group temporal aggregation을 통해 pupil 위치를 회귀한다.`에 해당하는 접근을 취한다.
- **Efficiency axis:** model size, FLOPs, latency, hardware power 중 일부를 명시적으로 다룬다. 제공되지 않은 축은 `논문에 명시되지 않음`으로 처리해야 한다.

---

# 4. Key Concepts

## 4.1 Concept Map

| Concept | Definition | Role in This Paper | Source |
| --- | --- | --- | --- |
| Frequency Adaptive Window Expanding | 충분한 event가 없으면 temporal window를 확장 | sparse fixation 대응 | Method |
| Point-based Event Cloud | raw `(x,y,t,p)` points를 직접 입력으로 사용 | resolution-independent complexity | Method |
| Intra/Inter aggregation | local spatio-temporal and sequence-level temporal aggregation | point representation feature extraction | Method |

## 4.2 Core Data Structures / Representations

- **Event stream:** `(x, y, t, p)` tuple sequence. 대부분의 event-based method의 raw input이다.
- **Frame / voxel / event volume / point cloud / graph / sparse patch:** 논문별로 event를 network나 tracker가 처리 가능한 representation으로 바꾼다.
- **Pupil state:** center `(x, y)`, ellipse `(x, y, a, b, θ)`, gaze point/direction 등 task output 형태가 다르다.

## 4.3 Key Equations / Algorithms

논문별 핵심 수식은 위 concept 및 methodology에서 설명한 component에 해당한다. PDF에 명시된 주요 수식이 있는 경우, 본문에서 representation equation, loss equation, state update equation, matching objective 등으로 사용된다. 수식 번호가 명확히 추출되지 않은 경우에는 `논문에 명시되지 않음`으로 둔다.

## 4.4 Concept Relationships

```text
Event Stream
  ↓
Event Representation / Preprocessing
  ↓
Spatial Feature Extraction or Geometric Tracking
  ↓
Temporal Modeling / State Update / Refinement
  ↓
Pupil Center or Ellipse Parameters
  ↓
Optional Gaze Regression / System Output
```

---

# 5. Methodology

## 5.1 Overall Architecture

event point cloud를 일정 개수로 sampling하고 standardization한다. point set은 KNN/FPS류 grouping을 거쳐 local extractor와 attention으로 local feature를 만들며, inter-sample temporal model이 eye movement context를 반영한다.

## 5.2 Input and Preprocessing

- **Input:** event stream, event-frame hybrid input, or frame-only eye image depending on the paper.
- **Preprocessing:** event binning, event volume, binary representation, sparse patch extraction, point sampling, graph construction, segmentation, or synthetic augmentation.
- **Output target:** pupil center, pupil ellipse, semantic eye mask, gaze point/direction, or refined prediction.

## 5.3 Core Modules

- raw event cloud 수집
- frequency adaptive windowing
- sampling/grouping of event points
- intra-group aggregation
- inter-sample LSTM / temporal aggregation
- regression head로 pupil location 출력

## 5.4 Forward Process

```text
Input events / frames
  ↓
Representation or initialization
  ↓
Core model / tracker / accelerator
  ↓
Temporal update or post-processing
  ↓
Pupil / ellipse / gaze output
```

## 5.5 Loss and Optimization

- 논문에 supervised model이 있는 경우 일반적으로 RMSE, SmoothL1, focal loss, IoU/Gaussian IoU, trigonometric loss, reconstruction/modularity loss, distillation loss 중 하나를 사용한다.
- 본 논문에서 구체적 loss가 명시된 경우에는 method 설명에 반영했다.
- Optimizer, learning rate, epoch, batch size는 PDF에 명시된 경우 Section 6에 기록한다. 명시되지 않은 항목은 `논문에 명시되지 않음`.

## 5.6 Training Procedure

논문별로 training procedure가 상이하다. challenge 논문은 PyTorch 기반 training, data augmentation, pretrained/synthetic data 사용 여부를 보고하는 경우가 많다. hardware/system 논문은 training보다 deployment/inference path를 강조한다.

## 5.7 Inference Procedure

본 논문의 inference는 다음 중 하나에 해당한다.

- event bin/window마다 DNN inference
- event-triggered continuous tracking update
- frame-based initialization 후 event-based update
- post-processing refinement
- FPGA/ASIC/SNN/embedded GPU deployment

## 5.8 Computational Characteristics

- 명시 수치가 있는 경우 result table에 parameter, FLOPs/MACs, latency, power, energy를 기록했다.
- 명시되지 않은 경우 임의로 추정하지 않았다.
- 서로 다른 resolution/dataset/hardware에서 측정된 latency와 pixel error는 직접 비교하지 않는다.

---

# 6. Experiments Results

## 6.1 Experimental Setup

| Item | Description |
|---|---|
| Dataset | 논문별로 EV-Eye, 3ET/3ET+, Ini-30, OpenEDS, internal/crowdsourced data, synthetic data 등 사용 |
| Input Resolution | 논문별 상이. 명시되지 않으면 `논문에 명시되지 않음` |
| Metrics | p-accuracy, pixel error, IoU, F1, angular error, latency, energy, params, FLOPs/MACs 등 |
| Hardware | GPU, CPU, Jetson Orin Nano, FPGA ZCU102, ASIC 12nm, Speck neuromorphic processor 등 논문별 상이 |
| Software | PyTorch, TensorRT, Vitis/Vivado, neuromorphic toolchain 등 논문별 상이 |

## 6.2 Quantitative Results

| Dataset | Task | Metric | Result | Comparison Condition | Source |
| --- | --- | --- | --- | --- | --- |
| SEET synthetic / evaluation | Pupil tracking | p3 / p5 / p10 / MSE | 0.920 / 0.991 / 0.996 / 1.56 | FAPNet 180×240 point; PEPNet p10 0.998, MSE 1.57 | Experimental tables |
| SEET | Efficiency | Params / FLOPs | 0.29M / 58.7M | PEPNet 0.64M / 443M | Experimental tables |
| AIS 2024 challenge context | Point-based method | p10 | 97.95% | EFFICIENT team result | Challenge survey/Table 1 context |

## 6.3 Qualitative Results

PDF에 visualization figure가 있는 경우, 주로 다음을 보여준다.

- pupil center/ellipse prediction overlay
- event stream 또는 event frame representation
- tracking trajectory under saccade/blink/fixation
- hardware architecture diagram 또는 dataflow diagram

정량 수치 없이 figure만 있는 claim은 `정성적 근거`로만 취급해야 한다.

## 6.4 Ablation Study

논문에 ablation이 명시된 경우 result table 또는 methodology에 반영했다. ablation이 없거나 제한적인 경우, 필요한 추가 ablation은 다음과 같다.

- event representation별 비교
- temporal module 제거 비교
- augmentation/post-processing 제거 비교
- online/causal vs offline/non-causal 비교
- resolution, event window, event count sensitivity
- hardware latency/power 측정 반복성

## 6.5 Efficiency Analysis

효율 분석은 논문별로 범위가 다르다. hardware 논문은 latency/energy/power/resource를 명시하는 반면, algorithm challenge 논문은 parameter/MACs 중심이다. `FPS`, `ms`, `µJ/frame`, `mW`, `LUT/DSP/BRAM`은 동일 hardware/platform 조건에서만 비교해야 한다.

## 6.6 Result Interpretation

- **강하게 검증된 주장:** result table에 수치와 비교 조건이 명확한 항목.
- **부분적으로 검증된 주장:** dataset 또는 hardware가 제한된 항목.
- **검증이 부족한 주장:** real-world HMD integration, long-term calibration drift, sensor variation, power under full system workload 등.

---

# 7. Summary

## 7.1 What This Paper Does

이 논문은 frame/voxel conversion은 event sparsity와 fine-grained temporal information을 손실하고, 입력 해상도에 따라 compute가 증가하는 문제.를 대상으로 event point cloud를 sampling/grouping하고, intra-group local aggregation과 inter-group temporal aggregation을 통해 pupil 위치를 회귀한다.를 제안한다.

## 7.2 Core Architecture

event point cloud를 일정 개수로 sampling하고 standardization한다. point set은 KNN/FPS류 grouping을 거쳐 local extractor와 attention으로 local feature를 만들며, inter-sample temporal model이 eye movement context를 반영한다.

## 7.3 Main Execution Flow

```text
  ↓ raw event cloud 수집
  ↓ frequency adaptive windowing
  ↓ sampling/grouping of event points
  ↓ intra-group aggregation
  ↓ inter-sample LSTM / temporal aggregation
  ↓ regression head로 pupil location 출력
```

## 7.4 Strongly Supported Parts

- raw event point를 사용해 event sparsity와 temporal resolution을 잘 보존
- spatial resolution에 대한 compute scaling이 frame-based보다 유리
- point cloud 기반 구조는 hardware-friendly sampling/aggregation 연구로 확장 가능

## 7.5 Weak or Unclear Parts

- point sampling/grouping의 latency와 memory access pattern은 accelerator 관점에서 추가 분석 필요
- SNN/FPGA 대비 실제 on-device power 결과는 제한적
- window expansion은 latency 상한을 늘릴 수 있음

## 7.6 Practical Usability

- **Research baseline:** 활용 가능. 단, dataset/protocol 차이를 명확히 해야 한다.
- **On-device deployment:** latency/power/hardware 결과가 있는 논문은 직접 참고 가능. 없는 경우 별도 profiling 필요.
- **Hardware accelerator design:** event representation, sparsity, temporal module, memory access pattern을 accelerator mapping 관점에서 재검토해야 한다.

## 7.7 One-Sentence Summary

> FAPNet: An Effective Frequency Adaptive Point-based Eye Tracker는 Point-based event eye tracking 관점에서 event point cloud를 sampling/grouping하고, intra-group local aggregation과 inter-group temporal aggregation을 통해 pupil 위치를 회귀한다.를 통해 event-based eye tracking의 accuracy-efficiency trade-off를 개선하려는 연구다.

## 7.8 Key Takeaways

1. event representation 선택이 accuracy와 latency를 동시에 좌우한다.
2. temporal modeling은 정확도에 중요하지만 online causality와 compute cost를 함께 봐야 한다.
3. XR deployment에서는 pixel error뿐 아니라 latency, power, update frequency, relocalization rate가 함께 필요하다.
4. 동일 dataset이라도 label frequency, resolution, p-accuracy vs pixel error 차이로 직접 비교가 어렵다.
5. 하드웨어 친화적 설계는 event sparsity, memory access, quantization, module partition을 함께 고려해야 한다.

---

# 8. Pros.

## P1. Problem-method alignment

- **Content:** 논문은 event-based eye tracking의 핵심 병목과 방법론을 비교적 직접적으로 연결한다.
- **Evidence:** event point cloud를 sampling/grouping하고, intra-group local aggregation과 inter-group temporal aggregation을 통해 pupil 위치를 회귀한다.
- **Importance:** XR/wearable system에서 accuracy만이 아니라 latency/power/frequency가 중요하기 때문이다.
- **Related Section/Table/Figure:** Overview, Methodology, representative results table.

## P2. Quantitative support

- **Content:** 대표 결과 수치가 제공된 경우, method의 핵심 주장과 연결된다.
- **Evidence:** Section 6.2 quantitative table.
- **Importance:** 논문 claim과 실제 측정 결과를 구분할 수 있다.
- **Related Section/Table/Figure:** 위 result table의 Source 항목 참조.

## P3. Relevance to future XR eye tracking systems

- **Content:** event camera, temporal modeling, sparse representation, hardware-aware design 중 하나 이상을 다루므로 XR on-device 연구에 직접 관련된다.
- **Evidence:** Research Background 및 Methodology.
- **Importance:** 후속 연구에서 algorithm-hardware co-design, quantization, event-driven accelerator로 확장 가능하다.
- **Related Section/Table/Figure:** Key Concepts, Methodology.

---

# 9. Cons.

## C1. Dataset and protocol dependency

- **Content:** 결과는 특정 dataset, split, resolution, label frequency, metric에 의존한다.
- **Evidence:** Section 6.1 setup 및 result source.
- **Impact:** 다른 논문과 직접 수치 비교 시 잘못된 결론을 낼 수 있다.
- **Suggested Improvement:** 동일 3ET+/EV-Eye split, 동일 resolution, 동일 metric으로 재평가한다.
- **Related Section/Table/Figure:** Experiments Results.

## C2. Deployment completeness gap

- **Content:** 일부 논문은 algorithm accuracy 중심이며 full on-device latency/power, sensor I/O, calibration, thermal constraint를 검증하지 않는다.
- **Evidence:** hardware metric이 없는 항목은 result table에서 `논문에 명시되지 않음`에 해당.
- **Impact:** XR headset integration feasibility 판단이 제한된다.
- **Suggested Improvement:** end-to-end latency boundary, power measurement, sensor-to-processor bandwidth, online causality를 함께 측정한다.
- **Related Section/Table/Figure:** Efficiency Analysis.

## C3. Robustness and generalization gap

- **Content:** blink, lighting, subject diversity, head movement, sensor noise, event threshold variation에 대한 검증 범위가 논문별로 다르다.
- **Evidence:** Cons list: point sampling/grouping의 latency와 memory access pattern은 accelerator 관점에서 추가 분석 필요, SNN/FPGA 대비 실제 on-device power 결과는 제한적, window expansion은 latency 상한을 늘릴 수 있음
- **Impact:** controlled dataset 성능이 실제 XR 사용 환경으로 그대로 이전되지 않을 수 있다.
- **Suggested Improvement:** cross-subject, cross-device, cross-lighting, real-HMD protocol, long-term drift evaluation을 추가한다.
- **Related Section/Table/Figure:** Research Background, Experiments Results.
