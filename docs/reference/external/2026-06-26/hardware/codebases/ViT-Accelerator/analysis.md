---
source_type: codebase
source_name: ViT-Accelerator
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-hardware-codebase
---

# ViT-Accelerator Analysis

- corpus: `References/Hardware`
- local_path: `/home/kjm26/project/PRJXR/References/Hardware/ViT-Accelerator`
- category: `HLS ViT`
- HGTXR relevance: `Direct HLS kernel baseline`
- recommended_priority: `P0`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/Hardware` 하위의 `ViT-Accelerator` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `1015` |
| 주요 확장자 | `.png:760, .cpp:49, [no_ext]:31, .sample:28, .txt:28, .py:24, .c:21, .h:20` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `hls_source/kernel.cpp`
- `scripts/run_hls.tcl`

## 4. README / 문서 근거 요약

> # ViT-Accelerator
> This repository demonstrates how to **accelerate a Vision Transformer (ViT)** on an FPGA using **High-Level Synthesis (HLS)** and **Xilinx’s XRT** runtime. It showcases:
> 1. A **systolic-array–style** matrix multiplication (in HLS) for computing parts of the attention mechanism.
> 2. A **softmax** approximation kernel to complete the attention scoring.
> 3. A **host (CPU) code** that manages data transfers and kernel invocation via XRT.
> Below is an overview of each major component, a brief introduction to the ViT concept, and a final study journal describing the learning process from a student’s perspective.
> ## Requirements
> - python=3.11
>   - Library dependencies are listed in host/requirements.txt
> - Xilinx Vitis Toolkit=2022.1
> ## Usage
> ```bash

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `HLS ViT` |
| HGTXR 적용성 | `Direct HLS kernel baseline` |
| 우선순위 | `P0` |
| 직접 HW 구현성 | 높음 |

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
| 예상 효과 | `Direct HLS kernel baseline` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`ViT-Accelerator`는 `P0` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
