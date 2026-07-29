---
source_type: codebase
source_name: HG-PIPE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-hardware-codebase
---

# HG-PIPE Analysis

- corpus: `References/Hardware`
- local_path: `/home/kjm26/project/PRJXR/References/Hardware/HG-PIPE`
- category: `ViT FPGA pipeline`
- HGTXR relevance: `Direct attention/MLP pipeline reference`
- recommended_priority: `P0`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/Hardware` 하위의 `HG-PIPE` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `1240` |
| 주요 확장자 | `.txt:1037, .cpp:36, .xml:31, .v:30, [no_ext]:16, .h:15, .scala:15, .sample:14` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `generated HLS/Verilog`
- `case modules`

## 4. README / 문서 근거 요약

> # HG-PIPE
> <!-- ![Build Status](https://img.shields.io/badge/build-passing-brightgreen) -->
> [English](README.md) | [中文](README.zh-CN.md)
> ![License](https://img.shields.io/badge/license-MIT-blue)
> ![Platform](https://img.shields.io/badge/platform-FPGA-orange)
> **HG-PIPE** is the official open-source implementation of the paper "Vision Transformer Acceleration with Hybrid-Grained Pipeline." It is an FPGA-based accelerator for Vision Transformer (ViT) models. This project aims to accelerate the inference process of Vision Transformer models using hybrid-grained pipeline techniques, achieving outstanding inference performance and energy efficiency. The project provides the implementation of the accelerator as well as corresponding validation methods and on-board testing scripts.
> ---
> ## Accelerator Features
> <!-- Add a table -->
> | LUTs | DSPs | BRAMs | Frequency | FPS (ImageNet@224x224) | TOPs | GOPs/W | Accuracy |
> |:----:|:----:|:-----:|:---------:|:----------------------:|:----:|:------:|:--------:|
> |  669k|  312 | 1006.5| 425MHz    |7118                   | 17.8 |  381.0 |  71.05%  |

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `ViT FPGA pipeline` |
| HGTXR 적용성 | `Direct attention/MLP pipeline reference` |
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
| 예상 효과 | `Direct attention/MLP pipeline reference` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`HG-PIPE`는 `P0` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
