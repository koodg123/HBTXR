---
source_type: framework
source_name: cgra4ml
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-hw-framework
---

# cgra4ml Analysis

- corpus: `References/HW_Framework`
- local_path: `/home/kjm26/project/PRJXR/References/HW_Framework/cgra4ml`
- category: `SoC/CGRA flow`
- HGTXR relevance: `Too broad for immediate HGTXR; useful for SoC boundary ideas`
- recommended_priority: `P3`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/HW_Framework` 하위의 `cgra4ml` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `1168` |
| 주요 확장자 | `.sv:417, .core:198, .py:82, .waiver:62, .hjson:55, .md:40, .tpl:40, .tcl:38` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `deepsocflow/rtl/*`
- `ibex-soc/*`
- `run/*.py`

## 4. README / 문서 근거 요약

> <!-- https://github.com/abarajithan11/deepsocflow/assets/26372005/113bfd40-cb4a-4940-83f4-d2ef91b47c91 -->
> # CGRA4ML: A Framework to Implement Modern Neural Networks for Scientific Edge Computing ![status](https://github.com/abarajithan11/dnn-engine/actions/workflows/verify.yml/badge.svg)
> cgra4ml is a Python library that helps researchers build, train, and implement their own deep ML models, such as ResNet CNNs, Autoencoders, and Transformers on FPGAs and custom ASIC.
> It takes a lot of effort and expertise to implement highly optimized neural networks on edge platforms. The challenging aspects include:
> - Designing an optimal dataflow architecture
> - Building & verifying an accelerator, optimizing for high-frequency
> - Building the System-on-Chip, verifying and optimizing data bottlenecks
> - Writing C firmware to control the accelerator and verify its correctness
> Often, after all that work, the models do not meet their expected performance due to memory bottlenecks and sub-optimal hardware implementation.
> We present a highly flexible, high-performance accelerator system that can be adjusted to your needs through a simple Python API. The framework is maintained as open source, allowing a user to modify the processing element to their desired data type using customized architecture, easily expand the architecture to meet the desired performance, and implement new neural network models.
> <p align="center"> <img src="docs/overview.png" width="800"> </p>
> ## Execution API

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `SoC/CGRA flow` |
| HGTXR 적용성 | `Too broad for immediate HGTXR; useful for SoC boundary ideas` |
| 우선순위 | `P3` |
| 직접 HW 구현성 | 중간/간접 |

## 6. HGTXR 실험 변환

| 단계 | 작업 | 산출물 | 승격 조건 |
|---|---|---|---|
| SW/분석 | 원본 코드/문서에서 파라미터, 연산자, 데이터플로우를 추출 | source notes, mapping table | HGTXR baseline과 비교 가능한 metric 정의 |
| HLS/RTL 후보화 | HGTXR C3b successor에 맞는 bounded variant로 축소 | compile-time define 또는 별도 experiment variant | csim/csynth 통과 |
| 리소스 정책 | DSP-bound MAC, URAM large buffer, LUTRAM small table 원칙 적용 | block-level resource report | DSP/LUT/URAM 목표 방향 개선 |
| 검증 | accuracy/error/latency/resource를 기존 C3b evidence와 분리 보고 | validation report | static claim이 아닌 측정 artifact 확보 |

## 7. 예상 효과와 리스크

| 항목 | 내용 |
|---|---|
| 예상 효과 | `Too broad for immediate HGTXR; useful for SoC boundary ideas` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`cgra4ml`는 `P3` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
