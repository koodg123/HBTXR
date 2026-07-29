---
source_type: codebase
source_name: TATAA
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-hardware-codebase
---

# TATAA Analysis

- corpus: `References/Hardware`
- local_path: `/home/kjm26/project/PRJXR/References/Hardware/TATAA`
- category: `Attention RTL/dataflow`
- HGTXR relevance: `Attention block comparison`
- recommended_priority: `P1`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/Hardware` 하위의 `TATAA` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `393` |
| 주요 확장자 | `.py:75, .txt:55, .sv:51, .v:45, .do:42, .sh:24, .sample:14, .udo:12` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `hardware/rtl/pe_stg_3.sv`
- `data_loader.sv`
- `bsr*.sv`

## 4. README / 문서 근거 요약

> # TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture
> The open-source implementation of the paper "TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture" in ACM Transactions on Reconfigurable Technology and Systems.
> ## Compilation
> The compiler of TATAA to parse Transformer models, generate dataflow and instructions for TATAA processor.
> Please refer to the `./compilation` directory for more details.
> ## Hardware
> The hardware implementation of TATAA processor.
> Please refer to the `./hardware` directory for more details.
> ## Quantization
> The quantization tool to quantize Transformer models in TATAA (int8 + bfloat16).
> Please refer to the `./quantization` directory for more details.
> Also, we listed the required python environment in `./quantization`.

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `Attention RTL/dataflow` |
| HGTXR 적용성 | `Attention block comparison` |
| 우선순위 | `P1` |
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
| 예상 효과 | `Attention block comparison` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`TATAA`는 `P1` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
