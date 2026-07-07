---
source_type: codebase
source_name: Transformer-Accelerator-Based-on-FPGA
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: references-hardware-codebase
---

# Transformer-Accelerator-Based-on-FPGA Analysis

- corpus: `References/Hardware`
- local_path: `/home/kjm26/project/PRJXR/References/Hardware/Transformer-Accelerator-Based-on-FPGA`
- category: `RTL attention blocks`
- HGTXR relevance: `Softmax/MHA reference`
- recommended_priority: `P1`
- evidence_type: static local inventory and integrated reference analysis

## 1. 분석 범위와 판정

이 문서는 `/home/kjm26/project/PRJXR/References/Hardware` 하위의 `Transformer-Accelerator-Based-on-FPGA` 항목을 HGTXR 하드웨어 설계 관점에서 분리 저장한 개별 분석 문서다. 현재 단계에서는 정적 파일 인벤토리와 상위 통합 분석을 근거로 하며, 학습/합성/보드 실행 결과를 새로 생성하지 않았다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 로컬 파일 수 추정 | `70` |
| 주요 확장자 | `.v:23, .sample:14, [no_ext]:11, .h:4, .sv:3, .c:2, .cpp:2, .txt:2` |
| README 감지 | `README.md` |

## 3. 핵심 파일 / 구성요소

- `In Progress/src/Softmax_top.v`

## 4. README / 문서 근거 요약

> # Transformer Accelerator Based on FPGA
> You can run it on pynq z1 (or any other Zynq device, since the systolic array is parameterized). The repository contains the relevant Verilog code, Vivado configuration and C/Python code for sdk/PYNQ testing. The size of the systolic array can be changed, now it is 16X16.
> In the future, I might add some nonlinear hardware acceleration operators (for accelerating ViT, it's a kind of neural network based on Transformer), such as those that compute Softmax, Gelu and LayerNorm functions. I am still working on to improve the accuracy and performance of this part.
> How to reproduce this project:
> 1. In vivado2019.1, create a new project (note that the boardfile is pynq z1, you can download the corresponding boardfile here: https://pynq.readthedocs.io/en/v3.0.0/overlay_design_methodology/board_settings.html ).
> 2. Add all the code to the project
> 3. Run prj.tcl
> 4. Create a wrapper for the block design and set it as the top module.
> 5. Run the generated synthesis and implementation strategies and generate the bitstream.

## 5. 알고리즘 및 하드웨어 관점

| 관점 | 판단 |
|---|---|
| 알고리즘 역할 | `RTL attention blocks` |
| HGTXR 적용성 | `Softmax/MHA reference` |
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
| 예상 효과 | `Softmax/MHA reference` 기반의 실험 후보 확보 |
| 주요 리스크 | 원본 target/device/dataset이 HGTXR과 다를 수 있음 |
| 비적용 조건 | LUT arithmetic 증가, exact baseline 부재, accuracy degradation 미보고 |
| 필요한 후속 증거 | csynth report, routed timing/power, board-smoke JSON 또는 SW accuracy report |

## 8. 현재 결론

`Transformer-Accelerator-Based-on-FPGA`는 `P1` 우선순위의 참고 항목이다. 현재 HGTXR 목표인 DSP 활용률 확대, LUT 압박 완화, URAM/LUTRAM 역할 분리와 충돌하지 않는 범위에서만 실험 후보로 승격한다.
