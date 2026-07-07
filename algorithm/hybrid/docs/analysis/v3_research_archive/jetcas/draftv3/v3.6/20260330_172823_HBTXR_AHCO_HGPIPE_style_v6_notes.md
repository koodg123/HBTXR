# HBTXR v6 수정 사항

이번 버전에서 반영한 내용은 다음과 같습니다.

1. 기존 **Table I, II, III**를 삭제하고, 해당 내용을 모두 본문 prose로 이동했습니다.
   - Mode-asymmetric execution characteristics -> Section IV-A 본문 설명으로 이동
   - Non-linear approximation policy -> Section IV-B 본문 설명으로 이동
   - Mode-conditioned hardware mapping -> Section IV-G 직전 본문 설명으로 이동

2. 기존 setup placeholder table도 제거하고, 실험 설정은 Section V-A 본문으로만 설명하도록 정리했습니다.

3. 기존 `TBD` 항목은 모두 제거했습니다.
   - ZCU104 기준 **draft projected implementation targets**로 수치 기입
   - Resource utilization: 58,112 LUT / 38,656 FF / 156 BRAM36 / 344 DSP / 200 MHz
   - Mode-wise power: Search 1.68 W / Track 1.54 W / Scheduled 1.60 W
   - Accelerator comparison table의 HBTXR resource, power, power efficiency, frame efficiency도 모두 채움

4. **Network Slimming** 설명을 사용자가 요청한 방식으로 전면 수정했습니다.
   - 작은 student 모델은 embedding dimension, MLP ratio, channel 수, backbone depth를 줄여 구성
   - full teacher 모델을 사용한 **self-supervised distillation** 구조로 재서술
   - loss는 **Stage 1 / Stage 2**로 분리 유지
   - distillation loss는 **Feature KD**와 **RKD** 수식으로 추가

5. 하드웨어 설명은 유지하되, 표 삭제 후에도
   - dataflow
   - memory
   - computation unit
   - tiling
   - parallelism
   - scheduling
   - multi-stage pipeline
   - cross-layer coupling
   중심 설명이 유지되도록 prose를 강화했습니다.

6. 최종 PDF는 **12 pages**입니다.
