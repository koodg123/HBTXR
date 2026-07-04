# XR-Eye-Tracking Paper Analysis Index

## 0. 생성 범위

- Source archive: `XR-Eye-Tracking.zip`
- PDF count: 21
- Output format: Markdown reports
- Paper prompt used: `Paper-Review-Summary-Prompt.md`
- Template handling: `Paper-Review-Summary-Template.md`가 Section 4.4까지만 제공되어, Section 5-9는 prompt 요구사항에 맞추어 확장함.
- Execution/reproduction: 논문 분석만 수행. 코드 실행, dataset download, training, hardware synthesis는 수행하지 않음.

## 1. Report List

| No | Paper | Year | Venue | Report |
| --- | --- | --- | --- | --- |
| 01 | EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras | 2023 | NeurIPS 2023 Datasets and Benchmarks | 01_EV-Eye_Rethinking_High_frequency_Eye_Tracking.md |
| 02 | 3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network | 2023 | IEEE BioCAS | 02_3ET_Change_Based_ConvLSTM.md |
| 03 | Dual-Path Enhancements in Event-Based Eye Tracking: Augmented Robustness and Adaptive Temporal Modeling | 2025 | CVPR Event-based Vision Workshop challenge report style | 03_Dual_Path_Enhancements_Event_Based_Eye_Tracking.md |
| 04 | E-Track: Eye Tracking with Event Camera for Extended Reality (XR) Applications | 2023 | IEEE AICAS | 04_E_Track_Event_Camera_XR.md |
| 05 | Enhancing Eye Feature Estimation from Event Data Streams through Adaptive Inference State Space Modeling | 2026 | arXiv preprint | 05_AISSM_Adaptive_Inference_State_Space_Modeling.md |
| 06 | Event-Based Near-Eye Gaze Tracking Beyond 10,000 Hz | 2022 | IEEE TVCG | 06_Event_Based_Near_Eye_Gaze_Tracking_Beyond_10000Hz.md |
| 07 | EX-Gaze: High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality | 2025 | IEEE TVCG | 07_EX_Gaze_Hybrid_Event_Frame_On_Device_XR.md |
| 08 | EyeGraph: Modularity-aware Spatio Temporal Graph Clustering for Continuous Event-based Eye Tracking | 2024 | NeurIPS Datasets and Benchmarks | 08_EyeGraph_Modularity_Aware_ST_Graph_Clustering.md |
| 09 | FACET: Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality | 2024 | arXiv / IEEE-style preprint | 09_FACET_Fast_Accurate_Ellipse_Modeling.md |
| 10 | FAPNet: An Effective Frequency Adaptive Point-based Eye Tracker | 2024 | arXiv / workshop-style paper | 10_FAPNet_Frequency_Adaptive_Point_Based_Eye_Tracker.md |
| 11 | Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing | 2025 | CVPR Event-based Vision Workshop challenge report style | 11_Inference_Time_Gaze_Refinement_Motion_Aware_Postprocessing.md |
| 12 | MambaPupil: Bidirectional Selective Recurrent Model for Event-based Eye Tracking | 2024 | AIS 2024 Challenge / workshop report | 12_MambaPupil_Bidirectional_Selective_Recurrent_Model.md |
| 13 | Overcoming Data Scarcity for Event-Based Pupil Tracking with Synthetic and Unlabeled Data | 2026 | PACM HCI | 13_Overcoming_Data_Scarcity_Synthetic_Unlabeled.md |
| 14 | A Lightweight Spatiotemporal Network for Online Eye Tracking with Event Camera | 2024 | CVPRW | 14_Pei_Lightweight_Spatiotemporal_Network.md |
| 15 | Rapidly Deploying On-device Eye Tracking by Distilling Visual Foundation Models | 2026 | preprint / Meta Reality Labs | 15_DistillGaze_VFM_On_Device_Eye_Tracking.md |
| 16 | Retina: Low-Power Eye Tracking with Event Camera and Spiking Hardware | 2024 | event-based / neuromorphic systems paper | 16_Retina_Low_Power_SNN_Speck.md |
| 17 | RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking | 2019 | IEEE ICCV Workshop / eye segmentation paper | 17_RITnet_Real_Time_Eye_Semantic_Segmentation.md |
| 18 | Swift-Eye: Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras | 2024 | IEEE TVCG | 18_Swift_Eye_Anti_Blink_High_Frequency.md |
| 19 | Exploring Temporal Dynamics in Event-based Eye Tracker / TDTracker | 2025 | CVPRW 2025 challenge-related paper | 19_TDTracker_Temporal_Dynamics_Event_Eye_Tracker.md |
| 20 | BRAT: Bidirectional Relative Positional Attention Transformer for Event-based Eye Tracking | 2025 | CVPRW | 20_BRAT_Bidirectional_Relative_Positional_Attention.md |
| 21 | Co-designing a Sub-millisecond Latency Event-based Eye Tracking System with Submanifold Sparse CNN | 2024 | CVPRW | 21_SEE_Submillisecond_SCNN_FPGA_Codesign.md |

## 2. 빠른 분류

| Category | Papers |
|---|---|
| Dataset / Benchmark | EV-Eye, EyeGraph, Overcoming Data Scarcity, RITnet |
| Event-only neural tracking | 3ET, E-Track, FACET, FAPNet, MambaPupil, BRAT, TDTracker, AISSM, Pei Spatiotemporal Network |
| Hybrid event-frame tracking | Event-Based Near-Eye Gaze Tracking Beyond 10,000 Hz, EX-Gaze, Swift-Eye, EV-Eye |
| Post-processing / robustness | Inference-Time Gaze Refinement, Dual-Path Enhancements |
| Neuromorphic / hardware | Retina, SEE |
| Foundation model distillation | DistillGaze |

## 3. 검증 기준

- 모든 보고서는 9개 본문 섹션을 포함한다: Overview, Research Background, Related Works, Key Concepts, Methodology, Experiments Results, Summary, Pros., Cons.
- 결과 수치는 가능한 경우 표/Figure/Section 근거를 함께 기록했다.
- 불확실하거나 PDF에서 확인되지 않은 항목은 `논문에 명시되지 않음` 또는 `PDF 기준 확인 제한`으로 표시했다.
- 서로 다른 dataset/resolution/hardware/metric의 직접 비교는 주의하도록 명시했다.
